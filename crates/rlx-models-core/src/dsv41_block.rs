// RLX — versatile ML compiler + runtime. GPLv3.
//! The shared spine of every **DeepSeek-V4.1** builder: the build context, the
//! debug taps, the rotary tables, and the attention prologue/epilogue.
//!
//! Prefill ([`crate::dsv41_graph`]), decode ([`crate::dsv41_decode`]) and the
//! DSpark draft head ([`crate::dsv41_dspark`]) are three ways of walking the same
//! block. What they share lives here and in the two sibling modules:
//!
//! * [`Ctx`] — the graph under construction plus everything needed to put a
//!   weight into it, so a helper takes one argument rather than four;
//! * [`build_qkv`] and [`build_attn_out`] — the projections that bracket
//!   whatever KV a given path assembles in between;
//! * [`crate::dsv41_hc`] — Hyper-Connections, whose pre-mix is *lagged*;
//! * [`crate::dsv41_csa`] — the compressor, the Indexer, and their masks.
//!
//! What the three do *not* share is the middle: prefill assembles a whole
//! sequence's KV, decode extends a cache by one, and DSpark attends over the main
//! model's window plus its own non-causal draft block. That difference is the
//! reason they are separate builders at all.

use crate::dsv41::{DeepseekV41Spec, yarn_inv_freq};
use crate::standard_decoder::{
    Proj, build_v4_o_lora, build_v4_sink_attention, emit_proj, load_norm, load_p, load_proj,
    load_transposed_param, load_v4_wo_a, rope_tail, synth_const, synth_zero,
};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Everything a builder threads through every helper: the graph under
/// construction, its parameters, the packed-weight side table, the checkpoint,
/// the spec, and how many rows this graph is being built for.
///
/// Bundling these is what lets the helpers below take a handful of meaningful
/// arguments instead of a dozen positional ones.
pub(crate) struct Ctx<'a> {
    pub g: Graph,
    pub params: HashMap<String, Vec<f32>>,
    pub packed: &'a mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    pub weights: &'a mut dyn WeightLoader,
    pub spec: &'a DeepseekV41Spec,
    /// Query rows: the sequence length at prefill, `1` at decode,
    /// `dspark_block_size` for a draft block.
    pub rows: usize,
}

impl<'a> Ctx<'a> {
    pub fn new(
        name: &str,
        spec: &'a DeepseekV41Spec,
        weights: &'a mut dyn WeightLoader,
        packed: &'a mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        rows: usize,
    ) -> Self {
        Ctx {
            g: Graph::new(name),
            params: HashMap::new(),
            packed,
            weights,
            spec,
            rows,
        }
    }

    /// `(graph, params)`, for a builder that is done.
    pub fn finish(self, outputs: Vec<NodeId>) -> (Graph, HashMap<String, Vec<f32>>) {
        let mut g = self.g;
        g.set_outputs(outputs);
        (g, self.params)
    }

    pub fn eps(&self) -> f32 {
        self.spec.rms_norm_eps
    }

    /// `rope_head_dim` rounded down to even — the rotation works on pairs.
    pub fn rd(&self) -> usize {
        self.spec.rope_head_dim & !1
    }

    /// A zero bias of width `n`, for the RMSNorm op signature these models lack.
    pub fn zero_bias(&mut self, tag: &str, n: usize) -> NodeId {
        synth_zero(&mut self.g, &mut self.params, tag, n)
    }

    pub fn norm(&mut self, key: &str) -> Result<NodeId> {
        load_norm(&mut self.g, &mut self.params, self.weights, key, 0.0)
    }

    pub fn param(&mut self, key: &str, transpose: bool) -> Result<NodeId> {
        load_p(&mut self.g, &mut self.params, self.weights, key, transpose)
    }

    pub fn transposed(&mut self, key: &str) -> Result<NodeId> {
        load_transposed_param(&mut self.g, &mut self.params, self.weights, key)
    }

    /// Load a projection weight, which may be packed.
    pub fn proj(&mut self, key: &str) -> Result<Proj> {
        load_proj(
            &mut self.g,
            &mut self.params,
            self.packed,
            self.weights,
            key,
        )
    }

    /// Apply an already-loaded projection.
    pub fn apply(&mut self, p: &Proj, x: NodeId, rows: usize, out: usize) -> NodeId {
        emit_proj(&mut self.g, x, p, Shape::new(&[rows, out], DType::F32))
    }

    /// Load a projection and apply it — the common case, where it is used once.
    pub fn project(&mut self, key: &str, x: NodeId, rows: usize, out: usize) -> Result<NodeId> {
        let p = self.proj(key)?;
        Ok(self.apply(&p, x, rows, out))
    }

    pub fn konst(&mut self, tag: &str, data: Vec<f32>, shape: &[usize]) -> NodeId {
        synth_const(&mut self.g, &mut self.params, tag, data, shape)
    }
}

/// Large-negative additive mask entry. Finite (not `-inf`) so a fully-masked row
/// still softmaxes to zeros against the attention sink instead of producing NaN —
/// which is exactly the convention the reference `sparse_attn` kernel adopts for
/// an all-`-1` index row.
pub(crate) const NEG: f32 = -1e30;

/// Bisection taps: `RLX_DSV41_DBG=<stage>` with `RLX_DSV41_DBGLAYER=<n>` cuts the
/// graph short and emits that stage's tensor instead of logits. Stages:
/// `engram`, `xa`, `comp`, `compkv`, `topk`, `sa`, `oinv`, `attn`, `ffn`,
/// `block`.
///
/// This is how the port was walked against the reference, and against CPU when a
/// backend diverged. Two things it insists on:
///
/// * the environment is read **once**, at construction, so a concurrently
///   running test cannot change the tap mid-build;
/// * a tap that never fires is an **error**, not a silent full-graph run —
///   otherwise asking for a stage a layer does not have quietly compares logits
///   instead and reads as a failure of that stage.
pub(crate) struct Tap {
    want: Option<(String, usize)>,
    fired: bool,
}

impl Tap {
    pub fn from_env() -> Self {
        let want = rlx_ir::env::var_result("RLX_DSV41_DBG").ok().map(|stage| {
            let layer = rlx_ir::env::var_result("RLX_DSV41_DBGLAYER")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            (stage, layer)
        });
        Tap { want, fired: false }
    }

    /// True when `stage` at `layer` is the requested tap.
    pub fn hit(&mut self, stage: &str, layer: usize) -> bool {
        let is = matches!(&self.want, Some((s, l)) if s == stage && *l == layer);
        self.fired |= is;
        is
    }

    pub fn check_fired(&self, context: &str) -> Result<()> {
        match (&self.want, self.fired) {
            (Some((stage, layer)), false) => Err(anyhow!(
                "RLX_DSV41_DBG={stage} RLX_DSV41_DBGLAYER={layer} never fired — \
                 there is no `{stage}` stage at layer {layer} in {context}"
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RopeTables {
    pub cos: NodeId,
    pub sin: NodeId,
    pub sin_inv: NodeId,
    pub cos_c: NodeId,
    pub sin_c: NodeId,
}

/// Build the `[n, half]` cos/sin/-sin tables for `positions`.
pub(crate) fn rope_table(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    positions: &[usize],
    rd: usize,
    theta: f64,
    yarn: Option<(usize, f64, f64, f64)>,
    tag: &str,
) -> (NodeId, NodeId, NodeId) {
    let half = (rd / 2).max(1);
    let n = positions.len();
    let (mut cosd, mut sind) = (vec![0f32; n * half], vec![0f32; n * half]);
    for (row, &p) in positions.iter().enumerate() {
        for i in 0..half {
            let fr = match yarn {
                Some((osl, factor, bf, bs)) => yarn_inv_freq(i, rd, theta, osl, factor, bf, bs),
                None => 1.0 / theta.powf(2.0 * i as f64 / rd as f64),
            };
            let (s, c) = (p as f64 * fr).sin_cos();
            cosd[row * half + i] = c as f32;
            sind[row * half + i] = s as f32;
        }
    }
    let neg: Vec<f32> = sind.iter().map(|v| -v).collect();
    let cos = synth_const(g, params, &format!("v41.rope.cos.{tag}"), cosd, &[n, half]);
    let sin = synth_const(g, params, &format!("v41.rope.sin.{tag}"), sind, &[n, half]);
    let sin_inv = synth_const(
        g,
        params,
        &format!("v41.rope.sininv.{tag}"),
        neg,
        &[n, half],
    );
    (cos, sin, sin_inv)
}

/// Turn a score matrix into an additive mask that keeps **exactly `k`** entries
/// per row.
///
/// `score` must already carry `causal_add` (unreachable entries at [`NEG`]) so
/// the selection prefers reachable positions; `causal_add` is re-applied at the
/// end, which is what drops the filler picks a row with fewer than `k` reachable
/// entries necessarily makes.
///
/// **Ties.** The reference selects with `torch.topk`, whose order among equal
/// scores is unspecified — and equal scores are common here, because the Indexer
/// rectifies its head scores and any position every head dislikes scores exactly
/// zero. This keeps the lowest index, which is what `Op::TopK` does (repeated
/// argmax with a strict `>`); a thresholding gate instead keeps *every* tied
/// entry and so silently overruns the `index_topk` budget.
/// Additive `[seq, seq]` sliding-window causal mask: query `qi` sees key `ki`
/// iff `ki <= qi` and `ki` is within the last `window` positions.
///
/// `window` is saturated at 1, so a config that omits `sliding_window` still
/// produces a well-formed (if degenerate) mask rather than an all-masked row.
pub(crate) fn sliding_window_mask(
    ctx: &mut Ctx<'_>,
    seq: usize,
    window: usize,
    tag: &str,
) -> NodeId {
    ctx.konst(tag, sliding_window(seq, window), &[seq, seq])
}

/// The mask values themselves, so the boundary is testable without a graph.
fn sliding_window(seq: usize, window: usize) -> Vec<f32> {
    let window = window.max(1);
    let mut m = vec![0f32; seq * seq];
    for qi in 0..seq {
        for ki in 0..seq {
            if ki > qi || qi - ki >= window {
                m[qi * seq + ki] = NEG;
            }
        }
    }
    m
}

/// The low-rank query, its rank-space form, and the shared KV latent.
pub(crate) struct Qkv {
    /// `q_norm(wq_a(x))` `[rows, q_lora_rank]`; the Indexer scores from this.
    pub qr: NodeId,
    /// RoPE'd query `[rows, n_heads · head_dim]`.
    pub q: NodeId,
    /// RoPE'd KV latent `[rows, head_dim]`, shared by every head (MQA).
    pub kv: NodeId,
}

/// One layer's attention projections, loaded once and applied as often as
/// needed.
///
/// Loading is separated from application because `WeightLoader::take` is
/// destructive and DSpark needs **two** KV latents from the same `wkv` — one
/// from the main model's stream, one from its own draft block. Calling a
/// load-and-apply helper twice would ask for a tensor that is already gone.
pub(crate) struct AttnProj {
    wq_a: Proj,
    q_gain: NodeId,
    zb_ql: NodeId,
    wq_b: Proj,
    wkv: Proj,
    kv_gain: NodeId,
    zb_hd: NodeId,
}

impl AttnProj {
    pub fn load(ctx: &mut Ctx<'_>, lp: &str) -> Result<Self> {
        let (ql, hd) = (ctx.spec.q_lora_rank, ctx.spec.head_dim);
        Ok(AttnProj {
            wq_a: ctx.proj(&format!("{lp}.attn.wq_a.weight"))?,
            q_gain: ctx.norm(&format!("{lp}.attn.q_norm.weight"))?,
            zb_ql: ctx.zero_bias(&format!("{lp}.zb.ql"), ql),
            wq_b: ctx.proj(&format!("{lp}.attn.wq_b.weight"))?,
            wkv: ctx.proj(&format!("{lp}.attn.wkv.weight"))?,
            kv_gain: ctx.norm(&format!("{lp}.attn.kv_norm.weight"))?,
            zb_hd: ctx.zero_bias(&format!("{lp}.zb.hd"), hd),
        })
    }

    /// `wq_a → q_norm → wq_b → partial RoPE`. Returns `(qr, q)`.
    pub fn query(
        &self,
        ctx: &mut Ctx<'_>,
        x: NodeId,
        cos: NodeId,
        sin: NodeId,
        rows: usize,
    ) -> (NodeId, NodeId) {
        let (nh, hd, ql) = (ctx.spec.n_heads, ctx.spec.head_dim, ctx.spec.q_lora_rank);
        let (rd, eps) = (ctx.rd(), ctx.eps());
        let qr = ctx.apply(&self.wq_a, x, rows, ql);
        let qr = ctx.g.rms_norm(qr, self.q_gain, self.zb_ql, eps);
        let q = ctx.apply(&self.wq_b, qr, rows, nh * hd);
        (qr, rope_tail(&mut ctx.g, q, cos, sin, rows, nh, hd, rd))
    }

    /// `wkv → kv_norm → partial RoPE`, for one stream of inputs.
    pub fn kv(
        &self,
        ctx: &mut Ctx<'_>,
        x: NodeId,
        cos: NodeId,
        sin: NodeId,
        rows: usize,
    ) -> NodeId {
        let (hd, rd, eps) = (ctx.spec.head_dim, ctx.rd(), ctx.eps());
        let kv = ctx.apply(&self.wkv, x, rows, hd);
        let kv = ctx.g.rms_norm(kv, self.kv_gain, self.zb_hd, eps);
        rope_tail(&mut ctx.g, kv, cos, sin, rows, 1, hd, rd)
    }
}

/// Query and KV from the same input — what prefill and decode want.
pub(crate) fn build_qkv(
    ctx: &mut Ctx<'_>,
    lp: &str,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    rows: usize,
) -> Result<Qkv> {
    let w = AttnProj::load(ctx, lp)?;
    let (qr, q) = w.query(ctx, x, cos, sin, rows);
    let kv = w.kv(ctx, x, cos, sin, rows);
    Ok(Qkv { qr, q, kv })
}

/// Sink attention over whatever keys the caller assembled. Returns
/// `[rows, n_heads · head_dim]`, still in the cache's rotated frame.
pub(crate) fn build_sink_attn(
    ctx: &mut Ctx<'_>,
    lp: &str,
    q: NodeId,
    kv_all: NodeId,
    mask: NodeId,
    n_keys: usize,
    rows: usize,
) -> Result<NodeId> {
    let (nh, hd) = (ctx.spec.n_heads, ctx.spec.head_dim);
    let sink = ctx.param(&format!("{lp}.attn.attn_sink"), false)?;
    let q3 = ctx.g.reshape_(q, vec![rows as i64, nh as i64, hd as i64]);
    let o = build_v4_sink_attention(
        &mut ctx.g,
        &mut ctx.params,
        q3,
        kv_all,
        mask,
        sink,
        (hd as f32).powf(-0.5),
        rows,
        nh,
        hd,
        n_keys,
        lp,
    );
    Ok(ctx.g.reshape_(o, vec![rows as i64, (nh * hd) as i64]))
}

/// Take the attention output back out of the query's rotation.
///
/// Not decoration: the cache stays in one shared rotated form, so the output has
/// to be un-rotated. It is also the step with no cancelling partner — a RoPE bug
/// that `q·kᵀ` hides shows up here at full size, which is how the Metal rank-2
/// bug was found.
pub(crate) fn inverse_rope(
    ctx: &mut Ctx<'_>,
    o: NodeId,
    cos: NodeId,
    sin_inv: NodeId,
    rows: usize,
) -> NodeId {
    let (nh, hd, rd) = (ctx.spec.n_heads, ctx.spec.head_dim, ctx.rd());
    rope_tail(&mut ctx.g, o, cos, sin_inv, rows, nh, hd, rd)
}

/// The grouped o-LoRA output projection: `wo_a` is block-diagonal over
/// `o_groups`, so each group sees only its own heads.
pub(crate) fn build_o_lora(ctx: &mut Ctx<'_>, lp: &str, o: NodeId, rows: usize) -> Result<NodeId> {
    let s = ctx.spec;
    let (d, n_groups, o_lora) = (s.dim, s.n_groups, s.o_lora_rank);
    let dpg = s.dim_per_group();
    let wo_a = load_v4_wo_a(
        &mut ctx.g,
        &mut ctx.params,
        ctx.weights,
        &format!("{lp}.attn.wo_a.weight"),
        n_groups,
        o_lora,
        dpg,
    )?;
    let wo_b = ctx.transposed(&format!("{lp}.attn.wo_b.weight"))?;
    Ok(build_v4_o_lora(
        &mut ctx.g, o, wo_a, wo_b, rows, n_groups, o_lora, dpg, d,
    ))
}

/// What [`build_attn_out`] attends over. Named fields, because the three
/// `NodeId`s and the two rotary tables are otherwise five interchangeable
/// positional arguments that would swap silently.
pub(crate) struct AttnOut {
    /// `[rows, n_heads · head_dim]`, already rotated.
    pub q: NodeId,
    /// `[n_keys, head_dim]` — one MQA latent serving as both key and value.
    pub kv_all: NodeId,
    /// `[rows, n_keys]` additive mask.
    pub mask: NodeId,
    pub n_keys: usize,
    pub rows: usize,
    /// Rotary tables for the *inverse* rotation applied to the attention
    /// output; `sin_inv` is already negated.
    pub cos: NodeId,
    pub sin_inv: NodeId,
}

/// Sink attention, un-rotate, project — the whole epilogue, for callers that do
/// not need to tap between the steps.
pub(crate) fn build_attn_out(ctx: &mut Ctx<'_>, lp: &str, a: &AttnOut) -> Result<NodeId> {
    let o = build_sink_attn(ctx, lp, a.q, a.kv_all, a.mask, a.n_keys, a.rows)?;
    let o = inverse_rope(ctx, o, a.cos, a.sin_inv, a.rows);
    build_o_lora(ctx, lp, o, a.rows)
}

/// An all-visible additive mask of `n` keys — what decode and the DSpark draft
/// need, where every key the host supplied is real by construction.
pub(crate) fn open_mask(ctx: &mut Ctx<'_>, tag: &str, rows: usize, n: usize) -> NodeId {
    ctx.konst(tag, vec![0f32; rows * n], &[rows, n])
}

#[cfg(test)]
mod tests {
    use super::{NEG, sliding_window};

    /// The window is inclusive of the query and `window - 1` keys before it, so
    /// a query at `qi` sees exactly `min(qi + 1, window)` keys. An off-by-one
    /// here changes nothing structural — the model still runs, it just attends
    /// over the wrong span — so it is worth pinning directly.
    #[test]
    fn window_spans_the_query_and_the_previous_window_minus_one_keys() {
        let (seq, window) = (6usize, 3usize);
        let m = sliding_window(seq, window);
        let visible = |qi: usize| (0..seq).filter(|&ki| m[qi * seq + ki] == 0.0).count();
        assert_eq!(visible(0), 1, "the first query sees only itself");
        assert_eq!(visible(1), 2);
        assert_eq!(visible(2), 3, "full window reached");
        assert_eq!(visible(5), 3, "and it stays at the window width");

        // query 4 sees keys 2, 3, 4 — not 1, and nothing in the future
        assert_eq!(m[4 * seq + 1], NEG);
        assert_eq!(m[4 * seq + 2], 0.0);
        assert_eq!(m[4 * seq + 4], 0.0);
        assert_eq!(m[4 * seq + 5], NEG);
    }

    /// A config with no `sliding_window` leaves `window_size` at `usize::MAX/4`,
    /// which must degrade to a plain causal mask rather than overflowing.
    #[test]
    fn an_unbounded_window_is_a_plain_causal_mask() {
        let seq = 4usize;
        let m = sliding_window(seq, usize::MAX / 4);
        for qi in 0..seq {
            for ki in 0..seq {
                let want = if ki <= qi { 0.0 } else { NEG };
                assert_eq!(m[qi * seq + ki], want, "at ({qi}, {ki})");
            }
        }
    }

    /// `window = 0` would otherwise mask every row completely, including the
    /// diagonal — a model that attends to nothing at all.
    #[test]
    fn a_zero_window_is_saturated_to_one() {
        let seq = 3usize;
        let m = sliding_window(seq, 0);
        for qi in 0..seq {
            assert_eq!(m[qi * seq + qi], 0.0, "query {qi} must see itself");
        }
        assert_eq!(m[2 * seq + 1], NEG, "and nothing else");
    }
}
