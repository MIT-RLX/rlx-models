//! The GPT graph — a nanoGPT / GPT-2-style decoder-only transformer, built
//! with the RLX `rlx!` DSL (via `rlx_expr!`, so ordinary Rust loops drive the
//! per-layer structure while the DSL expresses the math). Highlights the
//! training-DX additions: the loss is one DSL line,
//! `mean(cross_entropy(logits, targets))`.
//!
//! Embeddings are done as one-hot `@ table` (byte vocab is only 256, so this is
//! cheap) which keeps every input f32 and every op trivially differentiable —
//! no gather/index dtype handling. The LM head is **tied** to the token
//! embedding (`logits = h · wteᵀ`).

use rlx_tensor::{DType, Func, GraphScope, MaskKind, Tensor, rlx_expr, shape};

use crate::config::{GptConfig, LN_EPS};
use crate::rng::Rng;

/// z-loss weight: `β·mean(logsumexp(logits)²)` pins the softmax log-partition so
/// the logits can't drift to the f32 overflow that NaNs BPE-vocab training
/// (standard PaLM/Gemma guard). Training-only; pruned with the loss for inference.
const Z_LOSS_BETA: f64 = 1e-4;

/// Straight-through round to `dt`'s precision: the forward returns `x` rounded
/// to `dt` (then back to f32), while the backward is the identity in f32. Built
/// as `x + (round(x) − x).detach()` so the rounding residual carries no
/// gradient. Two overflow guards, one per pass:
/// - **forward**: clamp to ±`dtype_max` before the cast, so an activation past
///   the format's range saturates instead of becoming ±inf.
/// - **backward**: the cast (and its range) live entirely inside `.detach()`,
///   so gradients — which reach ~3e4 here and would overflow f16's 65504 — flow
///   in pure f32.
///
/// Together these let a matmul emulate any low-precision forward with an
/// f32-stable accumulate and backward.
fn ste_round(x: &Tensor, dt: DType) -> Tensor {
    let max = crate::precision::max_finite(dt);
    let rounded = x.clamp(-max, max).cast(dt).cast(DType::F32);
    x + &(&rounded - x).stop_gradient()
}

/// Build the GPT as a traced `Func` of named inputs/params. With `with_loss`
/// the single output is the scalar next-token cross-entropy loss (for
/// training); otherwise it is the logits `[batch*T, vocab]` (for generation).
///
/// Inputs (all one-hot f32, filled by the data loader):
/// - `tok`  `[batch*T, vocab]` — input token ids
/// - `pos`  `[batch*T, T]`     — absolute positions
/// - `tgt`  `[batch*T, vocab]` — next-token targets (only when `with_loss`)
///
/// `cdt` is the **compute dtype**: with `DType::BF16`/`F16` the matmul-heavy
/// forward/backward runs in half precision (≈2× on Metal, half the bandwidth)
/// while the parameters, loss, and optimizer stay f32 — standard mixed-precision
/// training. Pass `DType::F32` for full precision (e.g. generation).
pub fn build(cfg: &GptConfig, batch: usize, with_loss: bool, cdt: DType) -> Func {
    let cfg = *cfg;
    Func::new("gpt-tinystories", move |s| {
        forward(s, &cfg, batch, with_loss, cdt)
    })
}

fn forward(
    s: &mut GraphScope,
    cfg: &GptConfig,
    batch: usize,
    with_loss: bool,
    cdt: DType,
) -> Tensor {
    let (v, d, t, nh, dh, ff, b) = (
        cfg.vocab,
        cfg.n_embd,
        cfg.block_size,
        cfg.n_head,
        cfg.head_dim(),
        cfg.ffn(),
        batch,
    );
    let bt = b * t;
    // Reshape a [B*T, D] tensor into per-head [B, T, NH, DH] for `attention`.
    let heads = |x: &Tensor| x.reshape(vec![b as i64, t as i64, nh as i64, dh as i64]);
    let flat = |x: &Tensor| x.reshape(vec![bt as i64, d as i64]);
    // Mixed precision done right: the residual stream, LayerNorm, biases, GELU
    // and attention all stay **f32**. Storing the residual in bf16/f16 collapses
    // a row's variance (7 mantissa bits) → LayerNorm's `1/std` backward blows up
    // to ±3e38 → NaN. Only the big **matmuls** run in `cdt` (bf16/f16), with f32
    // in/out — that's where the FLOPs (and the low-precision speedup) live.
    let mm = |a: &Tensor, w: &Tensor| -> Tensor {
        if cdt == DType::F32 {
            a.matmul(w)
        } else {
            // Any sub-f32 format: straight-through round the matmul inputs to
            // `cdt` on the forward, but keep the accumulate AND the whole
            // backward in f32. A *native* low-precision matmul rounds the
            // gradients too — fine for bf16's 8-bit exponent, fatal for f16
            // (5-bit, max 65504: the grad matmuls reach ~3e4 and overflow to
            // NaN), and numerically fragile for bf16's 7-bit mantissa. STE
            // gives every float size the low-precision *forward* it emulates
            // with an f32-stable backward — see [`ste_round`].
            ste_round(a, cdt).matmul(&ste_round(w, cdt))
        }
    };

    // ── Token + positional embedding (f32 residual stream) ──────────────────
    let wte = s.param("wte", shape![v, d]); // reused (tied) for the LM head
    let wpe = s.param("wpe", shape![t, d]);
    // Token embedding by **gather**, not one-hot @ table: the data loader ships
    // `[B*T]` integer ids (fed as f32, cast to i64 in-graph) instead of a
    // `[B*T, V]` one-hot — ~V× less host→device traffic per step and no fake
    // embedding matmul (this workload is I/O/dispatch-bound, not FLOP-bound).
    let tok_ids = s.input("tok_ids", shape![bt]).cast(DType::I64);
    let tok_emb = wte.gather(&tok_ids, 0); // [B*T, D]
    // Positions are 0..T-1 in order within every sequence, so `wpe` [T,D] just
    // broadcasts over the batch — no positional input or gather needed.
    let tok_bt = tok_emb.reshape(vec![b as i64, t as i64, d as i64]);
    let mut h = (&tok_bt + &wpe).reshape(vec![bt as i64, d as i64]);

    // ── Transformer blocks (matmuls in cdt; everything else f32) ────────────
    for i in 0..cfg.n_layer {
        let ln1g = s.param(format!("ln1_g{i}"), shape![d]);
        let ln1b = s.param(format!("ln1_b{i}"), shape![d]);
        let hn = h.layer_norm(&ln1g, &ln1b, LN_EPS);

        let wq = s.param(format!("wq{i}"), shape![d, d]);
        let wk = s.param(format!("wk{i}"), shape![d, d]);
        let wv = s.param(format!("wv{i}"), shape![d, d]);
        let wo = s.param(format!("wo{i}"), shape![d, d]);
        let q = heads(&mm(&hn, &wq));
        let k = heads(&mm(&hn, &wk));
        let vv = heads(&mm(&hn, &wv));
        let attn = flat(&q.attention(&k, &vv, nh, dh, MaskKind::Causal));
        h = &h + &mm(&attn, &wo); // residual + output projection

        let ln2g = s.param(format!("ln2_g{i}"), shape![d]);
        let ln2b = s.param(format!("ln2_b{i}"), shape![d]);
        let hn2 = h.layer_norm(&ln2g, &ln2b, LN_EPS);

        let w1 = s.param(format!("w1{i}"), shape![d, ff]);
        let b1 = s.param(format!("b1{i}"), shape![ff]);
        let w2 = s.param(format!("w2{i}"), shape![ff, d]);
        let b2 = s.param(format!("b2{i}"), shape![d]);
        let act = (&mm(&hn2, &w1) + &b1).gelu();
        let mlp = &mm(&act, &w2) + &b2;
        h = &h + &mlp; // residual
    }

    // ── Final norm + tied LM head (all f32) ─────────────────────────────────
    let lnfg = s.param("lnf_g", shape![d]);
    let lnfb = s.param("lnf_b", shape![d]);
    let h = h.layer_norm(&lnfg, &lnfb, LN_EPS);
    let logits = h.matmul_t(&wte); // tied head in f32 → [B*T, V]

    if with_loss {
        // Targets ship as `[B*T]` integer ids too; the (label-smoothed) one-hot
        // is rebuilt **on-device** by gathering rows of a constant `[V,V]` table
        // — so only the ids cross the bus, not a `[B*T, V]` one-hot. Row `c` of
        // the table is class `c`'s smoothed distribution: `1-ls+ls/V` on the
        // diagonal, `ls/V` off it (ls = 0 ⇒ a plain one-hot).
        let ls = f64::from(cfg.label_smoothing);
        let vf = v as f64;
        let mut table = vec![ls / vf; v * v];
        for (c, row) in table.chunks_exact_mut(v).enumerate() {
            row[c] = 1.0 - ls + ls / vf;
        }
        let label_table = s.constant_nd(table, vec![v, v], DType::F32);
        let tgt_ids = s.input("tgt_ids", shape![bt]).cast(DType::I64);
        let tgt = label_table.gather(&tgt_ids, 0); // [B*T, V], built on device
        // Mean cross-entropy + z-loss. `logsumexp` is the same log-partition CE
        // forms; penalizing its square keeps the BPE-vocab logits from overflowing.
        let lse = logits.logsumexp(1, false);
        let ce = rlx_expr!(mean(cross_entropy(logits, tgt)));
        let z = rlx_expr!(mean(lse * lse)) * Z_LOSS_BETA;
        &ce + &z
    } else {
        logits
    }
}

/// Seed a freshly-built model with GPT-2-style initialization: `N(0, 0.02)` for
/// linear/embedding weights (residual projections `wo`/`w2` scaled by
/// `1/√(2·n_layer)`), ones for LayerNorm gains, zeros for all biases. Uses the
/// framework's [`Func::init_params`](rlx_tensor::Func::init_params) so one call
/// covers every parameter the graph declares.
pub fn init(model: Func, cfg: &GptConfig, seed: u64) -> Func {
    let n_layer = cfg.n_layer;
    let mut rng = Rng::new(seed);
    model.init_params(move |name, dims| {
        let n: usize = dims.iter().product();
        // Per-layer norms are named `ln1_g{i}` / `ln1_b{i}`, so a plain
        // `ends_with("_g")` misses them (they end in the layer index) and the
        // gains were silently initialized to N(0,0.02) instead of 1.0 — only the
        // unindexed `lnf_g` got the intended 1.0. Strip the trailing index first.
        let base = name.trim_end_matches(|c: char| c.is_ascii_digit());
        if base.ends_with("_g") {
            vec![1.0; n] // LayerNorm gain (ln1_g{i}, ln2_g{i}, lnf_g)
        } else if base.ends_with("_b") || name.starts_with("b1") || name.starts_with("b2") {
            vec![0.0; n] // LayerNorm bias (ln*_b{i}, lnf_b) or MLP bias (b1{i}, b2{i})
        } else {
            let std = if is_residual_proj(name) {
                0.02 / (2.0 * n_layer as f32).sqrt()
            } else {
                0.02
            };
            (0..n).map(|_| rng.normal() * std).collect()
        }
    })
}

/// Attention output projection (`wo`) and MLP down-projection (`w2`) — the
/// residual-path projections that GPT-2 down-scales at init for depth stability.
fn is_residual_proj(name: &str) -> bool {
    name.starts_with("wo") || name.starts_with("w2")
}
