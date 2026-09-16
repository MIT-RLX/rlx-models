// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1-Flash** prefill / pipeline-stage graph builder.
//!
//! One block is
//!
//! ```text
//! [engram?] → hc_mixes → hc_reduce(pre_mix_in) → attn_norm → o-LoRA MLA → hc_post
//!           → hc_mixes → hc_reduce(attn_pre)   → ffn_norm  → MoE        → hc_post
//! ```
//!
//! and the Hyper-Connection coefficients are **lagged**: the mix a sublayer
//! computes is consumed by the *next* one, so attention uses what the previous
//! layer's FFN produced and the FFN uses what this attention produced. The stack
//! ends by reducing with the last FFN's mix — V4.1 has no separate `hc_head`.
//!
//! Attention attends over two key sets concatenated into one softmax: a sliding
//! window of raw KV, plus (on `compress_ratio > 0` layers) the `index_topk` best
//! compressed positions. Which layer *produces* those compressed positions and
//! which merely reads them is the CSA2 split — see [`crate::dsv41`].
//!
//! Precision-simulation is deliberately omitted: the reference round-trips
//! activations through FP8/FP4 in place (`act_quant(..., inplace=True)`,
//! `fp4_act_quant(..., inplace=True)`). Those calls change no semantics, only
//! precision, so this builder computes the F32-exact value. Weight-side quant is
//! handled at load time by [`crate::dsv41_quant`].
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`.

use crate::dsv41::{DeepseekV41Spec, ScoreFunc, yarn_inv_freq};
use crate::dsv41_engram::load_and_build_v41_engram;
use crate::standard_decoder::{
    build_hc_sinkhorn, build_v4_o_lora, build_v4_sink_attention,
    const1, emit_proj, load_dense_dequant, load_norm, load_p, load_proj, load_transposed_param,
    load_v4_wo_a, rope_tail, softplus_stable, synth_const, synth_zero,
};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Large-negative additive mask entry. Finite (not `-inf`) so a fully-masked row
/// still softmaxes to zeros against the attention sink instead of producing NaN —
/// which is exactly the convention the reference `sparse_attn` kernel adopts for
/// an all-`-1` index row.
pub(crate) const NEG: f32 = -1e30;

/// What attention layers hand down the stack instead of recomputing, mirroring
/// the reference `SharedAttentionRuntime`. Every field is written by its source
/// layer before any consumer reads it, so one slot each is enough.
#[derive(Default, Clone, Copy)]
pub(crate) struct SharedAttn {
    /// RoPE'd compressed KV `[ncomp, head_dim]` from the last `kv_source_layer`.
    compress_kv: Option<NodeId>,
    /// RoPE'd index keys `[ncomp, index_head_dim]` from the same layer.
    index_k: Option<NodeId>,
    /// Additive `[seq, ncomp]` mask from the last `index_source_layer`.
    topk_mask: Option<NodeId>,
    /// Additive `[seq, ncomp]` candidate-block mask from `candidate_source_layer`.
    candidates: Option<NodeId>,
    /// Compressed-position count the above were built for.
    ncomp: usize,
    /// Pre-RoPE compressor latent, kept only for the `RLX_DSV41_DBG=comp` tap.
    dbg_latent: Option<NodeId>,
}

/// Bisection taps. `RLX_DSV41_DBG=<stage>` with `RLX_DSV41_DBGLAYER=<n>` cuts the
/// graph short and emits that stage's tensor instead of logits, which is how the
/// port was walked against the reference layer by layer. Stages: `engram`,
/// `comp`, `attn`, `ffn`, `block`.
pub(crate) fn dbg_tap() -> Option<(String, usize)> {
    let stage = std::env::var("RLX_DSV41_DBG").ok()?;
    let layer = std::env::var("RLX_DSV41_DBGLAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Some((stage, layer))
}

/// Per-layer RoPE tables. `cos`/`sin` are `[seq, rope_head_dim/2]` at token
/// positions; `sin_inv` is `-sin` (the inverse rotation the attention output
/// gets); `cos_c`/`sin_c` are the same table sampled at `j · ratio`, the position
/// a compressed latent stands for.
#[derive(Clone, Copy)]
pub(crate) struct RopeTables {
    cos: NodeId,
    sin: NodeId,
    sin_inv: NodeId,
    cos_c: NodeId,
    sin_c: NodeId,
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
    let sin_inv = synth_const(g, params, &format!("v41.rope.sininv.{tag}"), neg, &[n, half]);
    (cos, sin, sin_inv)
}

/// Hyper-Connection mixes: RMS-normalize the flattened `hc·dim` stream, project
/// to the `(2+hc)·hc` mixing vector, then Sinkhorn-split it into
/// `(pre, post, comb)`. `hc_fn_t` is the transposed `[hc·dim, (2+hc)·hc]` weight.
///
/// Unlike V4's `build_hc_pre` this does **not** immediately reduce with its own
/// `pre` — V4.1 hands `pre` to the next sublayer, so the reduce is separate
/// ([`hc_reduce`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_mixes(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    hc_fn_t: NodeId,
    scale: NodeId,
    base: NodeId,
    rows: usize,
    hc: usize,
    d: usize,
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
    tag: &str,
) -> (NodeId, NodeId, NodeId) {
    let x_flat = g.reshape_(x, vec![rows as i64, (hc * d) as i64]);
    let sq = g.mul(x_flat, x_flat);
    let ms = g.mean(sq, vec![1], true);
    // `norm_eps` here, `hc_eps` inside the Sinkhorn — the reference uses two
    // different epsilons and they differ by 14 orders of magnitude (1e-20 vs
    // 1e-6), so using one for both perturbs every mixing coefficient.
    let eps_c = const1(g, params, &format!("{tag}.hcm.eps"), norm_eps);
    let ms = g.add(ms, eps_c);
    let rsq = g.rsqrt(ms);
    let mixes = g.mm(x_flat, hc_fn_t);
    let mixes = g.mul(mixes, rsq);
    build_hc_sinkhorn(g, params, mixes, scale, base, rows, hc, hc_eps, iters, tag)
}

/// HC post-expand, `1 → hc` streams:
/// `y[j] = post[j]·x_out + Σ_i comb[i, j]·residual[i]`.
///
/// Note the contraction: the reference sums over the **first** axis of `comb`
/// (`(comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)` aligns `comb`'s
/// leading `hc` with `residual`'s, then reduces it), i.e. `combᵀ · residual`.
/// Contracting the other way is a plausible-looking transpose that survives every
/// shape check and quietly permutes the stream mixing.
pub(crate) fn hc_post(
    g: &mut Graph,
    x_out: NodeId,
    residual: NodeId,
    post: NodeId,
    comb: NodeId,
    rows: usize,
    hc: usize,
    d: usize,
) -> NodeId {
    let (r, h, dd) = (rows as i64, hc as i64, d as i64);
    let post3 = g.reshape_(post, vec![r, h, 1]);
    let xo3 = g.reshape_(x_out, vec![r, 1, dd]);
    let term1 = g.mul(post3, xo3); // [rows, hc, d]
    let comb4 = g.reshape_(comb, vec![r, h, h, 1]); // [rows, i, j, 1]
    let res4 = g.reshape_(residual, vec![r, h, 1, dd]); // [rows, i, 1, d]
    let prod = g.mul(comb4, res4); // [rows, i, j, d]
    let term2 = g.sum(prod, vec![1], false); // Σ_i → [rows, j, d]
    g.add(term1, term2)
}

/// Collapse the `hc` copies into one sublayer input: `Σ_hc pre·x`.
/// `x` is `[rows, hc, d]`, `pre` is `[rows, hc]`.
pub(crate) fn hc_reduce(g: &mut Graph, x: NodeId, pre: NodeId, rows: usize, hc: usize) -> NodeId {
    let pre3 = g.reshape_(pre, vec![rows as i64, hc as i64, 1]);
    let yh = g.mul(pre3, x);
    g.sum(yh, vec![1], false)
}

/// The one-hot initial pre-mix (`make_identity_pre_mix`): stream 0 only.
pub(crate) fn identity_pre_mix(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    rows: usize,
    hc: usize,
) -> NodeId {
    let mut d = vec![0f32; rows * hc];
    for r in 0..rows {
        d[r * hc] = 1.0;
    }
    synth_const(g, params, "v41.hc.identity_pre", d, &[rows, hc])
}

/// **KV Compressor** — pool `ratio` consecutive tokens into one latent with a
/// learned softmax gate, then RMSNorm. `ratio == 1` is a plain projection with no
/// gate at all (and the checkpoint carries no `wgate` for such a layer).
///
/// `x` is the post-`attn_norm` hidden `[seq, dim]`; only the first
/// `seq - seq % ratio` tokens participate (the trailing partial group is what the
/// reference holds back in `kv_state`). Returns the **pre-RoPE** latent
/// `[seq/ratio, head_dim]` — the Indexer needs it unrotated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_v41_compressor(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    seq: usize,
    ratio: usize,
    hd: usize,
    eps: f32,
) -> Result<NodeId> {
    let ncomp = seq / ratio;
    let wkv = load_p(
        g,
        params,
        weights,
        &format!("{lp}.attn.compressor.wkv.weight"),
        true,
    )?;
    let norm_w = load_norm(
        g,
        params,
        weights,
        &format!("{lp}.attn.compressor.norm.weight"),
        0.0,
    )?;
    let zb = synth_zero(g, params, &format!("{lp}.comp.zb"), hd);
    let kv = g.mm(x, wkv); // [seq, hd]
    let pooled = if ratio == 1 {
        kv
    } else {
        let wgate = load_p(
            g,
            params,
            weights,
            &format!("{lp}.attn.compressor.wgate.weight"),
            true,
        )?;
        let score = g.mm(x, wgate);
        let used = ncomp * ratio;
        let kv = g.narrow_(kv, 0, 0, used);
        let score = g.narrow_(score, 0, 0, used);
        let (nw, r, d) = (ncomp as i64, ratio as i64, hd as i64);
        let kv3 = g.reshape_(kv, vec![nw, r, d]);
        let sc3 = g.reshape_(score, vec![nw, r, d]);
        // softmax over the `ratio` axis (per feature): move it last, softmax, back
        let sct = g.transpose_(sc3, vec![0, 2, 1]); // [nwin, hd, ratio]
        let w = g.sm(sct, -1);
        let w = g.transpose_(w, vec![0, 2, 1]); // [nwin, ratio, hd]
        let prod = g.mul(kv3, w);
        g.sum(prod, vec![1], false) // [nwin, hd]
    };
    let pooled = g.reshape_(pooled, vec![ncomp as i64, hd as i64]);
    Ok(g.rms_norm(pooled, norm_w, zb, eps))
}

/// **Indexer scoring** — `Σ_h relu(⟨q[s,h], k[t]⟩) · weights[s,h]`, the learned
/// relevance of compressed position `t` to query `s`.
///
/// `weights = weights_proj(x) · (index_head_dim^-0.5 · index_n_heads^-0.5)`.
/// The reference's `fp4_act_quant` on `q`/`k` is precision-only and omitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_v41_index_score(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    qr: NodeId,           // [seq, q_lora_rank]
    x: NodeId,            // [seq, dim]
    index_k: NodeId,      // [ncomp, index_head_dim]
    wq_b_t: NodeId,       // [q_lora_rank, n_heads·index_head_dim]
    weights_proj_t: NodeId, // [dim, n_heads]
    cos: NodeId,
    sin: NodeId,
    seq: usize,
    nh: usize,
    ihd: usize,
    rd: usize,
    ncomp: usize,
    tag: &str,
) -> NodeId {
    let (sq, n, d, nc) = (seq as i64, nh as i64, ihd as i64, ncomp as i64);
    let q = g.mm(qr, wq_b_t); // [seq, nh·ihd]
    let q = rope_tail(g, q, cos, sin, seq, nh, ihd, rd);
    let q2 = g.reshape_(q, vec![(seq * nh) as i64, d]);
    let kt = g.transpose_(index_k, vec![1, 0]); // [ihd, ncomp]
    let sc = g.mm(q2, kt); // [seq·nh, ncomp]
    let sc = g.relu(sc);
    let sc = g.reshape_(sc, vec![sq, n, nc]);
    let w = g.mm(x, weights_proj_t); // [seq, nh]
    let scale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let sc_c = synth_const(g, params, &format!("{tag}.idx.scale"), vec![scale], &[1, 1]);
    let w = g.mul(w, sc_c);
    let w = g.reshape_(w, vec![sq, n, 1]);
    let prod = g.mul(sc, w);
    g.sum(prod, vec![1], false) // [seq, ncomp]
}

/// Level one of the hierarchical Indexer: keep the `topk_blocks` highest-scoring
/// blocks of `block_size` compressed positions per query, as an additive
/// `[seq, ncomp]` mask.
///
/// A block scores by its best position. Two constants do the bookkeeping the
/// reference does with `±inf`: `reach` marks blocks no position of which the
/// query can see yet (they must never be kept, even when fewer than `topk_blocks`
/// blocks are reachable), and `pin` force-selects the block holding the query's
/// newest position — it is only partly filled and would otherwise lose to an
/// older, full block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_v41_candidate_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    score: NodeId,      // [seq, ncomp], already causally masked
    causal_add: NodeId, // [seq, ncomp]
    seq: usize,
    ncomp: usize,
    // how many compressed positions each query can see (`compress_lens`)
    compress_lens: &[usize],
    block: usize,
    topk_blocks: usize,
    tag: &str,
) -> NodeId {
    let nblocks = ncomp.div_ceil(block);
    let padded = nblocks * block;
    let f = DType::F32;
    // pad the last block out to `block` with NEG so `amax` ignores it
    let score_p = if padded > ncomp {
        let pad = synth_const(
            g,
            params,
            &format!("{tag}.cand.pad"),
            vec![NEG; seq * (padded - ncomp)],
            &[seq, padded - ncomp],
        );
        g.concat_(vec![score, pad], 1)
    } else {
        score
    };
    let blk = g.reshape_(score_p, vec![seq as i64, nblocks as i64, block as i64]);
    let blk_score = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Max,
            axes: vec![2],
            keep_dim: false,
        },
        vec![blk],
        Shape::new(&[seq, nblocks], f),
    ); // [seq, nblocks]

    // Build-time constants: how many compressed positions query `qi` can see, and
    // therefore which blocks are reachable and which one holds its newest.
    let mut reach = vec![0f32; seq * nblocks];
    let mut pin = vec![0f32; seq * nblocks];
    for qi in 0..seq {
        let len = compress_lens[qi];
        for b in 0..nblocks {
            if b * block >= len {
                reach[qi * nblocks + b] = NEG;
            }
        }
        if len > 0 {
            let last = (len - 1) / block;
            pin[qi * nblocks + last] = -NEG;
        }
    }
    let reach_c = synth_const(g, params, &format!("{tag}.cand.reach"), reach, &[seq, nblocks]);
    let pin_c = synth_const(g, params, &format!("{tag}.cand.pin"), pin, &[seq, nblocks]);
    let blk_score = g.add(blk_score, reach_c);
    let blk_score = g.add(blk_score, pin_c);

    // Keep exactly `topk_blocks` blocks (the pinned one always among them), then
    // drop any that were only picked because fewer blocks were reachable.
    let keep = exact_topk_mask(
        g,
        params,
        blk_score,
        reach_c,
        seq,
        nblocks,
        topk_blocks,
        &format!("{tag}.cand"),
    );

    // expand each block's verdict over its `block` positions, then trim the pad
    let keep3 = g.reshape_(keep, vec![seq as i64, nblocks as i64, 1]);
    let ones = synth_const(
        g,
        params,
        &format!("{tag}.cand.ones"),
        vec![1f32; block],
        &[1, 1, block],
    );
    let wide = g.mul(keep3, ones);
    let wide = g.reshape_(wide, vec![seq as i64, padded as i64]);
    let wide = g.narrow_(wide, 1, 0, ncomp);
    g.add(wide, causal_add)
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
#[allow(clippy::too_many_arguments)]
/// Additive `[rows, ncomp]` mask that hides every compressed position a query
/// cannot see yet. `compress_lens[i]` is how many latents query `i` has passed.
pub(crate) fn compressed_causal_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    compress_lens: &[usize],
    ncomp: usize,
    tag: &str,
) -> NodeId {
    let rows = compress_lens.len();
    let mut m = vec![0f32; rows * ncomp];
    for (qi, &len) in compress_lens.iter().enumerate() {
        for c in 0..ncomp {
            if c >= len {
                m[qi * ncomp + c] = NEG;
            }
        }
    }
    synth_const(g, params, tag, m, &[rows, ncomp])
}

pub(crate) fn exact_topk_mask(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    score: NodeId,      // [rows, width], already causally masked
    causal_add: NodeId, // [rows, width]
    rows: usize,
    width: usize,
    k: usize,
    tag: &str,
) -> NodeId {
    let f = DType::F32;
    let k = k.min(width);
    let idx = g.add_node(Op::TopK { k }, vec![score], Shape::new(&[rows, k], f));
    let base = synth_const(
        g,
        params,
        &format!("{tag}.tk.base"),
        vec![NEG; rows * width],
        &[rows, width],
    );
    let updates = synth_const(
        g,
        params,
        &format!("{tag}.tk.keep"),
        vec![0f32; rows * k],
        &[rows, k],
    );
    let kept = g.add_node(
        Op::ScatterElements {
            axis: 1,
            reduction: rlx_ir::op::ScatterNdReduction::None,
        },
        vec![base, idx, updates],
        Shape::new(&[rows, width], f),
    );
    g.add(kept, causal_add)
}

/// Emit one attention block. Returns the `[rows, dim]` sublayer output; any
/// compressed KV / index keys / top-k mask this layer *produces* are written back
/// into `shared`.
#[allow(clippy::too_many_arguments)]
fn build_v41_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    spec: &DeepseekV41Spec,
    il: usize,
    x: NodeId,
    seq: usize,
    rope: &RopeTables,
    win_mask: NodeId,
    shared: &mut SharedAttn,
) -> Result<NodeId> {
    let lp = spec.layer_prefix(il);
    let f = DType::F32;
    let (d, nh, hd, ql) = (spec.dim, spec.n_heads, spec.head_dim, spec.q_lora_rank);
    let rd = spec.rope_head_dim & !1;
    let eps = spec.rms_norm_eps;
    let rows = seq;
    let zb_ql = synth_zero(g, params, &format!("{lp}.zb.ql"), ql);
    let zb_hd = synth_zero(g, params, &format!("{lp}.zb.hd"), hd);

    // q: low-rank, normalized at the rank, then partial RoPE per head
    let wq_a = load_proj(g, params, packed, weights, &format!("{lp}.attn.wq_a.weight"))?;
    let qr = emit_proj(g, x, &wq_a, Shape::new(&[rows, ql], f));
    let q_norm = load_norm(g, params, weights, &format!("{lp}.attn.q_norm.weight"), 0.0)?;
    let qr = g.rms_norm(qr, q_norm, zb_ql, eps);
    let wq_b = load_proj(g, params, packed, weights, &format!("{lp}.attn.wq_b.weight"))?;
    let q = emit_proj(g, qr, &wq_b, Shape::new(&[rows, nh * hd], f));
    let q = rope_tail(g, q, rope.cos, rope.sin, rows, nh, hd, rd);

    // the sliding-window KV: one latent shared by every head (MQA)
    let wkv = load_proj(g, params, packed, weights, &format!("{lp}.attn.wkv.weight"))?;
    let kv = emit_proj(g, x, &wkv, Shape::new(&[rows, hd], f));
    let kv_norm = load_norm(g, params, weights, &format!("{lp}.attn.kv_norm.weight"), 0.0)?;
    let kv = g.rms_norm(kv, kv_norm, zb_hd, eps);
    let kv = rope_tail(g, kv, rope.cos, rope.sin, rows, 1, hd, rd);

    let ratio = spec.ratio(il);
    let ncomp = seq.checked_div(ratio).unwrap_or(0);

    // ── CSA2: produce or reuse the compressed KV and its index keys ──
    if spec.is_kv_source(il) && ncomp > 0 {
        let latent = build_v41_compressor(g, params, weights, &lp, x, seq, ratio, hd, eps)?;
        // The Indexer needs the latent before RoPE, so derive the index keys first.
        if spec.index_head_dim > 0 {
            let ihd = spec.index_head_dim;
            let wk = load_p(
                g,
                params,
                weights,
                &format!("{lp}.attn.indexer.wk.weight"),
                true,
            )?;
            let k_norm = load_norm(
                g,
                params,
                weights,
                &format!("{lp}.attn.indexer.k_norm.weight"),
                0.0,
            )?;
            let zb_i = synth_zero(g, params, &format!("{lp}.zb.ihd"), ihd);
            let k = g.mm(latent, wk);
            let k = g.rms_norm(k, k_norm, zb_i, eps);
            let k = rope_tail(g, k, rope.cos_c, rope.sin_c, ncomp, 1, ihd, rd);
            shared.index_k = Some(k);
        }
        // a latent stands for the first token of its group → position j·ratio
        let comp = rope_tail(g, latent, rope.cos_c, rope.sin_c, ncomp, 1, hd, rd);
        shared.compress_kv = Some(comp);
        shared.ncomp = ncomp;
        shared.dbg_latent = Some(latent);
    }

    let (kv_all, mask, n_keys) = if ncomp == 0 {
        (kv, win_mask, seq)
    } else {
        let comp = shared.compress_kv.ok_or_else(|| {
            anyhow!("deepseek_v41: layer {il} reads compressed KV but no source produced it")
        })?;
        if shared.ncomp != ncomp {
            return Err(anyhow!(
                "deepseek_v41: layer {il} expects {ncomp} compressed positions, source produced {}",
                shared.ncomp
            ));
        }
        // causal visibility of compressed positions: latent c is visible to query
        // qi once qi has passed its last token, i.e. c < (qi+1)/ratio
        let compress_lens: Vec<usize> = (0..seq).map(|qi| (qi + 1) / ratio).collect();
        let causal_c = compressed_causal_mask(
            g,
            params,
            &compress_lens,
            ncomp,
            &format!("{lp}.v41.maskc"),
        );

        // ── the Indexer, when this layer is a source ──
        if spec.is_index_source(il) && spec.index_head_dim > 0 {
            let index_k = shared.index_k.ok_or_else(|| {
                anyhow!("deepseek_v41: layer {il} indexes but no kv source produced index keys")
            })?;
            let wq_b_i = load_proj(
                g,
                params,
                packed,
                weights,
                &format!("{lp}.attn.indexer.wq_b.weight"),
            )?;
            let wq_b_i = match wq_b_i.scheme {
                None => wq_b_i.w,
                Some(_) => {
                    return Err(anyhow!(
                        "deepseek_v41: packed indexer.wq_b is not supported; dequantize at load"
                    ));
                }
            };
            let wpj = load_p(
                g,
                params,
                weights,
                &format!("{lp}.attn.indexer.weights_proj.weight"),
                true,
            )?;
            let mut score = build_v41_index_score(
                g,
                params,
                qr,
                x,
                index_k,
                wq_b_i,
                wpj,
                rope.cos,
                rope.sin,
                seq,
                spec.index_n_heads,
                spec.index_head_dim,
                rd,
                ncomp,
                &lp,
            );
            score = g.add(score, causal_c);
            if spec.is_candidate_source(il) && spec.candidate_block_size > 0 {
                shared.candidates = Some(build_v41_candidate_mask(
                    g,
                    params,
                    score,
                    causal_c,
                    seq,
                    ncomp,
                    &compress_lens,
                    spec.candidate_block_size,
                    spec.candidate_topk_blocks,
                    &lp,
                ));
            } else if spec.uses_candidates(il)
                && let Some(cand) = shared.candidates
            {
                score = g.add(score, cand);
            }
            // Selecting `index_topk` of `ncomp` is a no-op when everything
            // causally valid already fits, which is the whole short-context
            // regime — keep the deterministic mask there rather than paying for
            // a TopK that cannot change the answer.
            // The re-mask must carry the candidate filter too, otherwise a
            // position outside every candidate block could come back through a
            // short row's filler picks.
            let base = match shared.candidates {
                Some(c) if spec.uses_candidates(il) => c,
                _ => causal_c,
            };
            // Selecting `index_topk` of `ncomp` cannot change anything when every
            // reachable position already fits, which is the whole short-context
            // regime — skip the TopK there rather than pay for a no-op.
            let gate = if ncomp > spec.index_topk && spec.index_topk > 0 {
                exact_topk_mask(g, params, score, base, seq, ncomp, spec.index_topk, &lp)
            } else {
                base
            };
            shared.topk_mask = Some(gate);
        }
        let comp_mask = shared.topk_mask.unwrap_or(causal_c);
        let kv_all = g.concat_(vec![kv, comp], 0);
        let full_mask = g.concat_(vec![win_mask, comp_mask], 1);
        (kv_all, full_mask, seq + ncomp)
    };

    let sink = load_p(g, params, weights, &format!("{lp}.attn.attn_sink"), false)?;
    let q3 = g.reshape_(q, vec![rows as i64, nh as i64, hd as i64]);
    let o = build_v4_sink_attention(
        g,
        params,
        q3,
        kv_all,
        mask,
        sink,
        (hd as f32).powf(-0.5),
        rows,
        nh,
        hd,
        n_keys,
        &lp,
    );
    // the query's rotation is removed again so the cache can stay in one shared
    // rotated form
    let o_flat = g.reshape_(o, vec![rows as i64, (nh * hd) as i64]);
    let o_inv = rope_tail(g, o_flat, rope.cos, rope.sin_inv, rows, nh, hd, rd);

    let dpg = spec.dim_per_group();
    let wo_a = load_v4_wo_a(
        g,
        params,
        weights,
        &format!("{lp}.attn.wo_a.weight"),
        spec.n_groups,
        spec.o_lora_rank,
        dpg,
    )?;
    let wo_b = load_transposed_param(g, params, weights, &format!("{lp}.attn.wo_b.weight"))?;
    Ok(build_v4_o_lora(
        g,
        o_inv,
        wo_a,
        wo_b,
        rows,
        spec.n_groups,
        spec.o_lora_rank,
        dpg,
        d,
    ))
}

/// Load `n_experts` per-expert `[out, in]` weights and stack them into the
/// `[E, in, out]` bank [`Op::GroupedMatMul`] expects.
///
/// The bank axis order matters: a `[E, out, in]` bank is silently mis-read, so
/// each expert is transposed on the way in.
pub(crate) fn load_expert_bank(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    proj: &str,
    n_experts: usize,
    k: usize,
    n: usize,
) -> Result<NodeId> {
    let mut data = Vec::with_capacity(n_experts * k * n);
    for e in 0..n_experts {
        let key = format!("{lp}.ffn.experts.{e}.{proj}.weight");
        let (w, shape) = weights.take_transposed(&key)?;
        if shape != vec![k, n] {
            return Err(anyhow!(
                "deepseek_v41: {key} is {shape:?} transposed, expected [{k}, {n}]"
            ));
        }
        data.extend_from_slice(&w);
    }
    let key = format!("{lp}.ffn.experts.{proj}.bank");
    let node = g.param(&key, Shape::new(&[n_experts, k, n], DType::F32));
    params.insert(key, data);
    Ok(node)
}

/// Clamped SwiGLU: `up ∈ [-L, L]`, `gate ≤ L`, then `silu(gate)·up`. The clamps
/// come from training, where they keep FP8/FP4 activations in range.
pub(crate) fn clamped_swiglu(g: &mut Graph, gate: NodeId, up: NodeId, limit: f32) -> NodeId {
    let (gate, up) = if limit > 0.0 {
        (g.clamp_(gate, f32::MIN, limit), g.clamp_(up, -limit, limit))
    } else {
        (gate, up)
    };
    let a = g.silu(gate);
    g.mul(a, up)
}

/// Emit one MoE FFN: top-k routed experts over their own bank plus one shared
/// expert every token goes through.
///
/// The correction bias steers *selection* only — the routing weights come from
/// the unbiased scores. `image_mask` (1.0 inside an image span) swaps in the
/// separate VL bias the model was trained with (`noaux_tc_for_vl`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_v41_moe(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    spec: &DeepseekV41Spec,
    il: usize,
    x: NodeId,
    rows: usize,
    image_mask: Option<NodeId>,
) -> Result<NodeId> {
    let lp = spec.layer_prefix(il);
    let f = DType::F32;
    let d = spec.dim;
    let inter = spec.moe_intermediate_size;
    let (n_experts, top_k) = spec.moe_dims(il);

    let gate_w = load_transposed_param(g, params, weights, &format!("{lp}.ffn.gate.weight"))?;
    let mut logits = g.mm(x, gate_w); // [rows, n_experts]
    if (spec.gate_temp - 1.0).abs() > f32::EPSILON {
        let t = const1(g, params, &format!("{lp}.moe.temp"), 1.0 / spec.gate_temp);
        logits = g.mul(logits, t);
    }
    let scores = match spec.score_func {
        ScoreFunc::Softmax => g.sm(logits, -1),
        ScoreFunc::Sigmoid => g.sigmoid(logits),
        ScoreFunc::SqrtSoftplus => {
            let sp = softplus_stable(g, params, logits, &format!("{lp}.moe.sp"));
            g.sqrt(sp)
        }
    };

    let bias = load_p(g, params, weights, &format!("{lp}.ffn.gate.bias"), false)?;
    // `bias_vl` only exists on vision-enabled checkpoints, and only matters when
    // the prefill actually contains an image span.
    let bias = match (image_mask, spec.vision.is_some()) {
        (Some(m), true) => {
            let bias_vl = load_p(g, params, weights, &format!("{lp}.ffn.gate.bias_vl"), false)?;
            let delta = g.sub(bias_vl, bias);
            let m2 = g.reshape_(m, vec![rows as i64, 1]);
            let scaled = g.mul(m2, delta);
            g.add(scaled, bias)
        }
        _ => bias,
    };
    let route = g.add(scores, bias);

    let top_idx = g.add_node(
        Op::TopK { k: top_k },
        vec![route],
        Shape::new(&[rows, top_k], f),
    );
    let mut top_w = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![scores, top_idx],
        Shape::new(&[rows, top_k], f),
    );
    if spec.norm_topk_prob && top_k > 1 {
        // `+ 1e-20`, not `rms_norm_eps` — the reference pins this constant to
        // what training used regardless of `norm_eps`.
        let denom = g.sum(top_w, vec![1], true);
        let tiny = const1(g, params, &format!("{lp}.moe.tiny"), 1e-20);
        let denom = g.add(denom, tiny);
        top_w = g.div(top_w, denom);
    }
    if (spec.route_scale - 1.0).abs() > f32::EPSILON {
        let sc = const1(g, params, &format!("{lp}.moe.rscale"), spec.route_scale);
        top_w = g.mul(top_w, sc);
    }

    let w1 = load_expert_bank(g, params, weights, &lp, "w1", n_experts, d, inter)?;
    let w3 = load_expert_bank(g, params, weights, &lp, "w3", n_experts, d, inter)?;
    let w2 = load_expert_bank(g, params, weights, &lp, "w2", n_experts, inter, d)?;

    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let e_col = g.narrow_(top_idx, 1, ki, 1);
        let e_idx = g.reshape_(e_col, vec![rows as i64]);
        let w_col = g.narrow_(top_w, 1, ki, 1);
        let gate = g.add_node(
            Op::GroupedMatMul,
            vec![x, w1, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let up = g.add_node(
            Op::GroupedMatMul,
            vec![x, w3, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let glu = clamped_swiglu(g, gate, up, spec.swiglu_limit);
        let down = g.add_node(
            Op::GroupedMatMul,
            vec![glu, w2, e_idx],
            Shape::new(&[rows, d], f),
        );
        let weighted = g.mul(down, w_col);
        acc = Some(match acc {
            None => weighted,
            Some(a) => g.add(a, weighted),
        });
    }
    let routed = acc.ok_or_else(|| anyhow!("deepseek_v41: layer {il} has top_k = 0"))?;

    let se_inter = spec.n_shared_experts.max(1) * inter;
    let s1 = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.ffn.shared_experts.w1.weight"),
    )?;
    let s3 = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.ffn.shared_experts.w3.weight"),
    )?;
    let s2 = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.ffn.shared_experts.w2.weight"),
    )?;
    let sg = emit_proj(g, x, &s1, Shape::new(&[rows, se_inter], f));
    let su = emit_proj(g, x, &s3, Shape::new(&[rows, se_inter], f));
    let sglu = clamped_swiglu(g, sg, su, spec.swiglu_limit);
    let sdown = emit_proj(g, sglu, &s2, Shape::new(&[rows, d], f));
    Ok(g.add(routed, sdown))
}

/// Everything the host must supply alongside `input_ids` for one prefill.
#[derive(Default, Clone)]
pub struct V41Inputs {
    /// Engram row indices, `[seq · n_engram_layers · n_hash_cols]` in that order
    /// — the output of [`crate::dsv41_engram::EngramHashPlan::hash_ids`]. Empty
    /// when the checkpoint has no Engram.
    pub engram_rows: Vec<i64>,
    /// `true` at positions inside an image span. Those positions take no part in
    /// an n-gram and route through the VL gate bias. Empty means text-only.
    pub image_positions: Vec<bool>,
    /// Also emit `main_hidden [rows, dim · n_targets]` — the mean over the
    /// Hyper-Connection copies of the stream entering each
    /// `dspark_target_layer_ids` layer, which is what the DSpark draft head
    /// conditions on ([`crate::dsv41_dspark`]). Off by default so the graph has a
    /// single output.
    pub emit_main_hidden: bool,
}

/// Mean over the `hc` copies: what a DSpark target layer contributes to
/// `main_hidden`. Taken from the stream *entering* the layer (after any Engram
/// write), not from its output.
pub(crate) fn hc_mean(g: &mut Graph, x: NodeId, rows: usize, d: usize) -> NodeId {
    let m = g.mean(x, vec![1], false); // [rows, d]
    g.reshape_(m, vec![rows as i64, d as i64])
}

/// Build a **DeepSeek-V4.1** prefill graph for `seq` tokens. Returns the graph
/// and its parameter map; the single graph input is `input_ids [1, seq]`, the
/// output `logits [seq, vocab_size]`.
pub fn build_deepseek_v41_prefill(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_deepseek_v41_stage(spec, weights, seq, 0..spec.n_layers, true, true, inputs, packed)
}

/// One **pipeline stage**: transformer layers `layers` only, loading only those
/// layers' weights (plus embeddings when `first` and the norm/head when `last`).
///
/// The graph input is `input_ids [1, seq]` when `first`, else the boundary pair
/// `hidden_in [seq, hc_mult, dim]` and `pre_mix_in [seq, hc_mult]` — V4.1 threads
/// the Hyper-Connection pre-mix across blocks, so a stage boundary has to carry
/// it too or the next stage silently restarts from the one-hot mix.
///
/// The output is `logits [seq, vocab]` when `last`, else `hidden_out` plus
/// `pre_mix_out`.
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v41_stage(
    spec: &DeepseekV41Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    layers: std::ops::Range<usize>,
    first: bool,
    last: bool,
    inputs: &V41Inputs,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    spec.validate()?;
    let mut g = Graph::new("deepseek_v41_stage");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hc) = (spec.dim, spec.hc_mult);
    let rows = seq;
    let eps = spec.rms_norm_eps;
    let rd = spec.rope_head_dim & !1;
    let zb_d = synth_zero(&mut g, &mut params, "v41.zb.d", d);

    // Two RoPE tables: plain `rope_theta` for pure sliding-window layers (YaRN
    // off), and `compress_rope_theta` + YaRN for KV-compressed ones. Compressed
    // latents sample the same table at `j · ratio`.
    let positions: Vec<usize> = (0..seq).collect();
    let yarn = (spec.original_seq_len > 0 && spec.rope_factor > 1.0).then_some((
        spec.original_seq_len,
        spec.rope_factor,
        spec.beta_fast,
        spec.beta_slow,
    ));
    let (cos_w, sin_w, sininv_w) = rope_table(
        &mut g,
        &mut params,
        &positions,
        rd,
        spec.rope_theta,
        None,
        "win",
    );
    let (cos_k, sin_k, sininv_k) = rope_table(
        &mut g,
        &mut params,
        &positions,
        rd,
        spec.compress_rope_theta,
        yarn,
        "comp",
    );
    // one compressed-position table per distinct ratio in this stage
    let mut comp_tables: HashMap<usize, (NodeId, NodeId)> = HashMap::new();
    for il in layers.clone() {
        let ratio = spec.ratio(il);
        if ratio == 0 || comp_tables.contains_key(&ratio) {
            continue;
        }
        let ncomp = seq / ratio;
        if ncomp == 0 {
            continue;
        }
        let pos: Vec<usize> = (0..ncomp).map(|j| j * ratio).collect();
        let (c, s, _) = rope_table(
            &mut g,
            &mut params,
            &pos,
            rd,
            spec.compress_rope_theta,
            yarn,
            &format!("c{ratio}"),
        );
        comp_tables.insert(ratio, (c, s));
    }

    // sliding-window causal mask: query qi sees key ki iff ki <= qi and within
    // the last `window_size` positions
    let window = spec.window_size.max(1);
    let mut maskd = vec![0f32; seq * seq];
    for qi in 0..seq {
        for ki in 0..seq {
            if ki > qi || qi - ki >= window {
                maskd[qi * seq + ki] = NEG;
            }
        }
    }
    let win_mask = synth_const(&mut g, &mut params, "v41.mask.win", maskd, &[seq, seq]);

    // image-span mask, when the caller marked any
    let image_mask = (!inputs.image_positions.is_empty()
        && inputs.image_positions.iter().any(|&b| b))
    .then(|| {
        let m: Vec<f32> = (0..seq)
            .map(|i| {
                if inputs.image_positions.get(i).copied().unwrap_or(false) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        synth_const(&mut g, &mut params, "v41.mask.image", m, &[seq, 1])
    });
    // image tokens take no part in an n-gram and get no engram contribution
    let engram_alive = image_mask.map(|_| {
        let m: Vec<f32> = (0..seq)
            .map(|i| {
                if inputs.image_positions.get(i).copied().unwrap_or(false) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        synth_const(&mut g, &mut params, "v41.mask.engram_alive", m, &[seq, 1])
    });

    let (mut h, mut pre_mix) = if first {
        let input_ids = g.input("input_ids", Shape::new(&[1, seq], DType::I32));
        let (embed_w, _, _) = load_dense_dequant(&mut g, &mut params, weights, "embed.weight")?;
        let h0 = g.gather_(embed_w, input_ids, 0); // [1, seq, d]
        let h0 = g.reshape_(h0, vec![rows as i64, 1, d as i64]);
        // expand to hc_mult copies for Hyper-Connections
        let ones_hc = synth_const(&mut g, &mut params, "v41.hc.ones", vec![1f32; hc], &[1, hc, 1]);
        let h = g.mul(h0, ones_hc);
        (h, identity_pre_mix(&mut g, &mut params, rows, hc))
    } else {
        let h = g.input("hidden_in", Shape::new(&[rows, hc, d], f));
        let p = g.input("pre_mix_in", Shape::new(&[rows, hc], f));
        (h, p)
    };

    let mut shared = SharedAttn::default();
    let mut main_hiddens: Vec<NodeId> = Vec::new();
    for il in layers.clone() {
        let lp = spec.layer_prefix(il);

        // ── Engram, before the block reads the stream ──
        if let Some(e) = &spec.engram
            && let Some(hash_idx) = e.layer_hash_index(il)
        {
            {
                let cols = e.n_hash_cols();
                let n_eng = e.layer_ids.len();
                if inputs.engram_rows.len() != seq * n_eng * cols {
                    return Err(anyhow!(
                        "deepseek_v41: engram_rows has {} entries, expected {} (seq {seq} × {n_eng} layers × {cols} cols)",
                        inputs.engram_rows.len(),
                        seq * n_eng * cols
                    ));
                }
                let slice: Vec<f32> = (0..seq)
                    .flat_map(|i| {
                        let base = (i * n_eng + hash_idx) * cols;
                        inputs.engram_rows[base..base + cols]
                            .iter()
                            .map(|&v| v as f32)
                    })
                    .collect();
                let row_ids =
                    synth_const(&mut g, &mut params, &format!("{lp}.engram.rows"), slice, &[seq, cols]);
                h = load_and_build_v41_engram(
                    &mut g,
                    &mut params,
                    weights,
                    &lp,
                    h,
                    row_ids,
                    engram_alive,
                    rows,
                    hc,
                    d,
                    e,
                    eps,
                )?;
                if dbg_tap() == Some(("engram".into(), il)) {
                    g.set_outputs(vec![h]);
                    return Ok((g, params));
                }
            }
        }

        if inputs.emit_main_hidden && spec.dspark_target_layer_ids.contains(&il) {
            main_hiddens.push(hc_mean(&mut g, h, rows, d));
        }

        let rope = RopeTables {
            cos: if spec.ratio(il) > 0 { cos_k } else { cos_w },
            sin: if spec.ratio(il) > 0 { sin_k } else { sin_w },
            sin_inv: if spec.ratio(il) > 0 { sininv_k } else { sininv_w },
            cos_c: comp_tables.get(&spec.ratio(il)).map(|t| t.0).unwrap_or(cos_k),
            sin_c: comp_tables.get(&spec.ratio(il)).map(|t| t.1).unwrap_or(sin_k),
        };

        // ── attention sub-block ──
        let residual = h;
        let fn_a = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.hc_attn_fn"))?;
        let sc_a = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_attn_scale"), false)?;
        let bs_a = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_attn_base"), false)?;
        let (attn_pre, attn_post, attn_comb) = hc_mixes(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            rows,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.a"),
        );
        let xa = hc_reduce(&mut g, h, pre_mix, rows, hc);
        let an = load_norm(&mut g, &mut params, weights, &format!("{lp}.attn_norm.weight"), 0.0)?;
        let xa = g.rms_norm(xa, an, zb_d, eps);
        let attn_out = build_v41_attention(
            &mut g, &mut params, packed, weights, spec, il, xa, seq, &rope, win_mask, &mut shared,
        )?;
        if dbg_tap() == Some(("compkv".into(), il)) {
            let c = shared.compress_kv.ok_or_else(|| anyhow!("layer {il} has no compressed KV"))?;
            g.set_outputs(vec![c]);
            return Ok((g, params));
        }
        if dbg_tap() == Some(("topk".into(), il)) {
            let m = shared
                .topk_mask
                .ok_or_else(|| anyhow!("layer {il} has no compressed-position mask"))?;
            g.set_outputs(vec![m]);
            return Ok((g, params));
        }
        if dbg_tap() == Some(("comp".into(), il)) {
            let latent = shared
                .dbg_latent
                .ok_or_else(|| anyhow!("layer {il} produced no compressor latent to tap"))?;
            g.set_outputs(vec![latent]);
            return Ok((g, params));
        }
        if dbg_tap() == Some(("attn".into(), il)) {
            g.set_outputs(vec![attn_out]);
            return Ok((g, params));
        }
        h = hc_post(&mut g, attn_out, residual, attn_post, attn_comb, rows, hc, d);

        // ── FFN sub-block ──
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_fn"))?;
        let sc_f = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_scale"), false)?;
        let bs_f = load_p(&mut g, &mut params, weights, &format!("{lp}.hc_ffn_base"), false)?;
        let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            rows,
            hc,
            d,
            eps,
            spec.hc_eps,
            spec.hc_mult_sinkhorn_iters,
            &format!("{lp}.f"),
        );
        let xf = hc_reduce(&mut g, h, attn_pre, rows, hc);
        let fnorm = load_norm(&mut g, &mut params, weights, &format!("{lp}.ffn_norm.weight"), 0.0)?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = build_v41_moe(
            &mut g, &mut params, packed, weights, spec, il, xf, rows, image_mask,
        )?;
        if dbg_tap() == Some(("ffn".into(), il)) {
            g.set_outputs(vec![ffn_out]);
            return Ok((g, params));
        }
        h = hc_post(&mut g, ffn_out, residual, ffn_post, ffn_comb, rows, hc, d);
        pre_mix = ffn_pre;
        if dbg_tap() == Some(("block".into(), il)) {
            g.set_outputs(vec![h]);
            return Ok((g, params));
        }
    }

    if last {
        let x = hc_reduce(&mut g, h, pre_mix, rows, hc);
        let fnorm = load_norm(&mut g, &mut params, weights, "norm.weight", 0.0)?;
        let x = g.rms_norm(x, fnorm, zb_d, eps);
        let head = load_proj(&mut g, &mut params, packed, weights, "head.weight")?;
        let logits = emit_proj(&mut g, x, &head, Shape::new(&[rows, spec.vocab_size], f));
        let logits = g.reshape_(logits, vec![rows as i64, spec.vocab_size as i64]);
        let mut outs = vec![logits];
        if inputs.emit_main_hidden {
            outs.push(main_hidden_node(&mut g, &main_hiddens, spec, rows)?);
        }
        g.set_outputs(outs);
    } else {
        let mut outs = vec![h, pre_mix];
        if inputs.emit_main_hidden {
            outs.push(main_hidden_node(&mut g, &main_hiddens, spec, rows)?);
        }
        g.set_outputs(outs);
    }
    Ok((g, params))
}

/// Concatenate the collected target-layer means into `main_hidden`.
///
/// A stage that holds none of the target layers is an error rather than an empty
/// tensor: DSpark would otherwise be conditioned on nothing at all.
pub(crate) fn main_hidden_node(
    g: &mut Graph,
    parts: &[NodeId],
    spec: &DeepseekV41Spec,
    rows: usize,
) -> Result<NodeId> {
    let want = spec.dspark_target_layer_ids.len();
    if parts.len() != want {
        return Err(anyhow!(
            "deepseek_v41: main_hidden needs all {want} target layers {:?}, this stage held {}",
            spec.dspark_target_layer_ids,
            parts.len()
        ));
    }
    let cat = g.concat_(parts.to_vec(), 1);
    Ok(g.reshape_(cat, vec![rows as i64, (spec.dim * want) as i64]))
}
