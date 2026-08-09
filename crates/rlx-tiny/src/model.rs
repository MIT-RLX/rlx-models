//! The GPT graph — a decoder-only transformer whose **every weight matrix is
//! synthesized from a tiny codebook** (`Op::SynthMatMul`) instead of stored
//! dense, written as a single **`rlx! { … }` block**: the whole forward (token
//! embedding → the `repeat i in 0..n_layer` transformer stack → final norm →
//! tied LM head → the next-token loss) is one declarative graph.
//!
//! **Codebook weight-synthesis ("functions not data").** A projection `x·W`
//! (`W [k,n]`) is *not* a dense matmul. Each row of `W` is a sequence of `k/ED`
//! entries picked (by a fixed u8 index table) from a small trained codebook
//! `[NE, ED]`, and the weight is reconstructed *inside* the matmul kernel — so a
//! `k·n` weight becomes just `NE·ED` trainable numbers with ~no DRAM weight
//! bytes. Two capacity levers close the gap to a dense model while staying
//! IO/DRAM-minimal: **residual multi-stage VQ** (`W = Σ_s codebook_s[idx_s]`,
//! [`GptConfig::synth_stages`]) and an optional **low-rank correction**
//! (`W += A·Bᵀ`, [`GptConfig::lora_rank`]). The FFN's activation is a
//! per-channel **learnable KAN spline** (`Op::SplineActivation`) rather than a
//! fixed GELU.
//!
//! **Graph shape.** The **per-layer weight tensors** — codebook params, their
//! fixed u8 index constants, and the LoRA factors — are built in plain Rust
//! *before* the block, grouped one `LayerParams` per layer into a single
//! `Vec<LayerParams>`, and adopted by one **`bind layers[];`**, so
//! `layers[i].cb_wq[s]` reads layer `i`, stage `s`'s codebook inside the runtime
//! `repeat i` (layers) × `repeat s` (residual-VQ stages).
//! Token embedding is a **gather** of `wte` by the `[B*T]` integer
//! ids (fed as f32, cast to i64 in-graph) — ~V× less host→device traffic than a
//! one-hot `@ table`; the LM head is **tied** to it (`logits = h · wteᵀ`); and
//! the loss is the DSL sugar `mean(softmax_cross_entropy(logits, tgt_ids))`
//! straight off the integer targets.
//!
//! **Init.** Random ([`init`]) or, via product-quantizing a trained dense
//! `rlx-tinystories` checkpoint into the codebooks/indices ([`SynthInit`] +
//! [`build_dense_init`] / [`init_dense`]), a data-optimal start; a dense teacher
//! can also **distill** into the run ([`build_distill`]).

use std::collections::{HashMap, HashSet};

use rlx_tensor::{DType, Func, GraphScope, Tensor, rlx, shape};

use crate::config::{GptConfig, LN_EPS};
use crate::quantize::pq_multistage;
use crate::rng::Rng;

// ── SynthMatMul + KAN configuration ─────────────────────────────────────────
// Every weight matrix `W [k,n]` is replaced by codebook weight-synthesis: fixed
// u8 indices `[n, k/ED]` (a deterministic quantization structure) select entries
// from a small TRAINED codebook `[NE, ED]`, and `y = x·Wᵀ` reconstructs the weight
// inside the matmul. So each `k·n` weight becomes just `NE·ED` trainable numbers —
// "functions not data". The FFN's GELU becomes a per-channel learnable KAN spline.
const SYNTH_ED: usize = 4; // entry_dim (must divide every input dim: d and ff)
const SYNTH_NE: usize = 256; // codebook entries (u8 index range)
// Residual multi-stage VQ depth is now fully GRAPH-LIVE: the `rlx!` block's
// per-projection weight is `Σ_{s<synth_stages} codebook_s[idx_s]`, summed by a
// runtime `repeat s in 1..synth_stages` stage loop (seeded at stage 0), so
// `GptConfig::synth_stages` drives the graph directly — `--synth-stages N`
// builds N codebook stages per projection (it also still drives the PQ init in
// `SynthInit` and the init scaling). No compile-time stage constant remains.
const KAN_BASIS: u32 = 8; // spline basis functions per channel
const KAN_MIN: f32 = -4.0;
const KAN_MAX: f32 = 4.0;
/// **z-loss** weight: an auxiliary `β·mean(logsumexp(logits)²)` term that pins the
/// softmax log-partition near 0, stopping the logits from drifting to the f32
/// overflow that produced NaNs mid-training (the standard PaLM/Gemma cure). It is
/// a training-only regularizer — the loss cone (and hence this term) is pruned for
/// inference — and adds **no** parameters or inference IO. `1e-4` is the published
/// default and is small enough not to distort the language-model objective.
const Z_LOSS_BETA: f32 = 1e-4;
// Capacity levers to close the quality gap to a dense model — both keep the
// IO/DRAM-minimal, compute-in-kernel profile, and are now config-driven
// (`GptConfig::synth_stages` / `GptConfig::lora_rank`) so quality-vs-compression
// can be swept from the CLI:
//   • residual multi-stage VQ: W = Σ_s codebook_s[idx_s]. Each stage is another
//     tiny u8 index table + L1-resident codebook — MORE in-kernel compute, still
//     ~no weight bytes in DRAM. Multiplies codebook DOF by `synth_stages`.
//   • low-rank correction: W += A·Bᵀ (rank r) — a small dense delta that recovers
//     the degrees of freedom a fixed-assignment codebook structurally can't reach.

/// Data-optimal initialization derived from a **trained dense** checkpoint by
/// product quantization (see [`crate::quantize`]). Carries, per synth layer, the
/// baked u8 index tables (one per residual-VQ stage — consumed at graph-build
/// time by `synth_linear`) and, per parameter name, the initial value the
/// codebooks / LoRA factors / copied embeddings-norms-biases should take
/// (consumed at [`init_dense`] time). The runtime model is byte-for-byte the same
/// shape as a random-init one — only the *values and index assignment* change.
pub struct SynthInit {
    /// `cb_name` → per-stage baked indices `[n·(k/ED)]` (u8, in `[n, k/ED]` order).
    indices: HashMap<String, Vec<Vec<u8>>>,
    /// Parameter name → its PQ-derived (codebook / LoRA) or straight-copied value.
    values: HashMap<String, Vec<f32>>,
}

impl SynthInit {
    /// Build the PQ init from a **dense** `rlx-tinystories` parameter set (same
    /// `wq/wk/wv/wo/w1/w2` weight names, and identical `wte`/`wpe`/norm/bias
    /// names). Each dense weight is product-quantized into the matching synth
    /// layer's per-stage indices + codebooks (residual k-means across stages) and
    /// a truncated-SVD LoRA fit of the leftover; embeddings, norms and biases —
    /// same shape in both models — are copied straight over. Anything the dense
    /// set is missing simply falls back to the random init.
    pub fn from_dense(cfg: &GptConfig, dense: &[(String, Vec<f32>)]) -> Self {
        let d = cfg.n_embd;
        let ff = cfg.ffn();
        let lookup = |name: &str| dense.iter().find(|(n, _)| n == name).map(|(_, v)| v);
        let mut indices: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut consumed: HashSet<String> = HashSet::new();

        for i in 0..cfg.n_layer {
            // (synth layer name, k, n) — the dense source name is the cb_ suffix.
            let specs: [(String, usize, usize); 6] = [
                (format!("cb_wq{i}"), d, d),
                (format!("cb_wk{i}"), d, d),
                (format!("cb_wv{i}"), d, d),
                (format!("cb_wo{i}"), d, d),
                (format!("cb_w1{i}"), d, ff),
                (format!("cb_w2{i}"), ff, d),
            ];
            for (cb_name, k, n) in specs {
                let dense_name = &cb_name[3..]; // strip "cb_"
                let w = match lookup(dense_name) {
                    Some(w) if w.len() == k * n => w,
                    _ => continue, // missing/mismatched dense weight → random fallback
                };
                consumed.insert(dense_name.to_string());
                let (stage_idx, stage_cb, lora_a, lora_b) =
                    pq_multistage(w, k, n, SYNTH_ED, SYNTH_NE, cfg.synth_stages, cfg.lora_rank);
                for (st, cb) in stage_cb.into_iter().enumerate() {
                    values.insert(format!("{cb_name}_s{st}"), cb);
                }
                indices.insert(cb_name.clone(), stage_idx);
                if cfg.lora_rank > 0 {
                    values.insert(format!("{cb_name}_lora_a"), lora_a);
                    values.insert(format!("{cb_name}_lora_b"), lora_b);
                }
            }
        }

        // Embeddings / norms / biases: identical names & shapes → copy straight.
        for (name, data) in dense {
            if consumed.contains(name) {
                continue;
            }
            values.insert(name.clone(), data.clone());
        }
        SynthInit { indices, values }
    }

    /// Per-synth-layer reconstruction fidelity of the stored PQ init against the
    /// dense weights, for the `--init-from` report. Returns `(name, stage-1 rel
    /// error, full multi-stage+LoRA rel error)` — how faithfully the codebooks
    /// encode each real weight (recomputed from the *stored* indices/codebooks,
    /// no k-means rerun). Lower = better; the full error is what the init starts at.
    pub fn reconstruction_report(
        &self,
        cfg: &GptConfig,
        dense: &[(String, Vec<f32>)],
    ) -> Vec<(String, f32, f32)> {
        use crate::quantize::reconstruct;
        let d = cfg.n_embd;
        let ff = cfg.ffn();
        let lookup = |name: &str| dense.iter().find(|(n, _)| n == name).map(|(_, v)| v);
        let mut report = Vec::new();
        for i in 0..cfg.n_layer {
            let specs: [(String, usize, usize); 6] = [
                (format!("cb_wq{i}"), d, d),
                (format!("cb_wk{i}"), d, d),
                (format!("cb_wv{i}"), d, d),
                (format!("cb_wo{i}"), d, d),
                (format!("cb_w1{i}"), d, ff),
                (format!("cb_w2{i}"), ff, d),
            ];
            for (cb_name, k, n) in specs {
                let w = match lookup(&cb_name[3..]) {
                    Some(w) if w.len() == k * n => w,
                    _ => continue,
                };
                let stages = match self.indices.get(&cb_name) {
                    Some(s) => s,
                    None => continue,
                };
                // Stage-1 approx.
                let cb0 = &self.values[&format!("{cb_name}_s0")];
                let approx1 = reconstruct(&stages[0], cb0, k, n, SYNTH_ED);
                let err1 = rel_frob(&approx1, w);
                // Full multi-stage sum + LoRA A·Bᵀ.
                let mut approx = vec![0f32; k * n];
                for (st, idx) in stages.iter().enumerate() {
                    let cb = &self.values[&format!("{cb_name}_s{st}")];
                    let r = reconstruct(idx, cb, k, n, SYNTH_ED);
                    for (x, y) in approx.iter_mut().zip(&r) {
                        *x += *y;
                    }
                }
                if cfg.lora_rank > 0 {
                    if let (Some(a), Some(b)) = (
                        self.values.get(&format!("{cb_name}_lora_a")),
                        self.values.get(&format!("{cb_name}_lora_b")),
                    ) {
                        let r = cfg.lora_rank;
                        for p in 0..k {
                            for j in 0..n {
                                let mut acc = 0f32;
                                for c in 0..r {
                                    acc += a[p * r + c] * b[j * r + c];
                                }
                                approx[p * n + j] += acc;
                            }
                        }
                    }
                }
                let err_full = rel_frob(&approx, w);
                report.push((cb_name, err1, err_full));
            }
        }
        report
    }
}

/// Relative Frobenius error `‖approx − w‖ / ‖w‖`.
fn rel_frob(approx: &[f32], w: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in approx.iter().zip(w) {
        let dd = (*a - *b) as f64;
        num += dd * dd;
        den += (*b as f64) * (*b as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt() as f32
    }
}

/// SplitMix64 — a stateless deterministic hash, so the fixed index structure is
/// identical every time the graph is traced (no RNG state to drift).
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One baked u8 index table `[n, k/ED]` for stage `st` of a synth projection:
/// PQ-derived from a trained dense weight when dense-init is active (data-optimal
/// assignment), else the random-frozen SplitMix structure — the *same* constant
/// either way. Mirrors what the old `synth_linear` baked inline, so a random-init
/// graph keeps its exact index structure. `cb_name` is the `cb_{role}{i}` key.
fn bake_idx(
    ext: &mut GraphScope,
    cb_name: &str,
    n: usize,
    nb: usize,
    seed: u64,
    st: usize,
    init: Option<&SynthInit>,
) -> Tensor {
    let idx: Vec<f64> = match init
        .and_then(|di| di.indices.get(cb_name))
        .and_then(|per_stage| per_stage.get(st))
    {
        Some(bytes) if bytes.len() == n * nb => bytes.iter().map(|&b| b as f64).collect(),
        _ => {
            let sd = seed
                .wrapping_mul(0x1_0000_0001)
                .wrapping_add(st as u64 * 0x9E37_79B9);
            (0..n * nb)
                .map(|p| (splitmix(sd.wrapping_add(p as u64)) % SYNTH_NE as u64) as f64)
                .collect()
        }
    };
    ext.constant_nd(idx, vec![n, nb], DType::U8)
}

/// The per-layer weight tensors for ONE projection role (`wq`/`w1`/…), built
/// role-major so the param-node order — and thus the RNG-seeded init — is
/// stable, then regrouped per layer by [`group_layers`] into [`LayerParams`].
/// A projection `x·W` (`W [k,n]`) becomes `Σ_st x.synth_matmul(idx_s{st},
/// cb_s{st}, ED, NE)` in the block — plus, when `lora_rank>0`, `(x·A)·Bᵀ`. The
/// codebooks keep their `cb_{role}{i}_s{st}` (and `_lora_a`/`_lora_b`) names, so
/// [`init`]/[`init_dense`] and checkpoints bind them unchanged.
struct Proj {
    /// `synth_stages` trained codebook params, per layer: `cb[st][layer]`.
    cb: Vec<Vec<Tensor>>,
    /// Matching baked u8 index constants: `idx[st][layer]`.
    idx: Vec<Vec<Tensor>>,
    /// LoRA factors per layer (empty when `lora_rank == 0`).
    lora_a: Vec<Tensor>,
    lora_b: Vec<Tensor>,
}

/// Build a projection role's [`Proj`] collections in `ext`. `k`/`n` are the
/// dense-equivalent `W [k,n]` dims; `seed_off` is the role's index-seed offset
/// (`wq`=0…`w2`=5, matching the old `synth_linear` call sites).
#[allow(clippy::too_many_arguments)]
fn build_proj(
    ext: &mut GraphScope,
    role: &str,
    k: usize,
    n: usize,
    seed_off: u64,
    n_layer: usize,
    synth_stages: usize,
    lora_rank: usize,
    init: Option<&SynthInit>,
) -> Proj {
    let nb = k / SYNTH_ED;
    let mut cb = vec![Vec::with_capacity(n_layer); synth_stages];
    let mut idx = vec![Vec::with_capacity(n_layer); synth_stages];
    let mut lora_a = Vec::new();
    let mut lora_b = Vec::new();
    for i in 0..n_layer {
        let cb_name = format!("cb_{role}{i}");
        let seed = i as u64 * 6 + seed_off;
        for st in 0..synth_stages {
            cb[st].push(ext.param(format!("{cb_name}_s{st}"), shape![SYNTH_NE, SYNTH_ED]));
            idx[st].push(bake_idx(ext, &cb_name, n, nb, seed, st, init));
        }
        if lora_rank > 0 {
            lora_a.push(ext.param(format!("{cb_name}_lora_a"), shape![k, lora_rank]));
            lora_b.push(ext.param(format!("{cb_name}_lora_b"), shape![n, lora_rank]));
        }
    }
    Proj {
        cb,
        idx,
        lora_a,
        lora_b,
    }
}

/// Every per-layer tensor for ONE transformer block, so the whole model's
/// per-layer weights are a single `Vec<LayerParams>` adopted by one
/// **`bind layers[];`** — `layers[i].cb_wq[s]` reads layer `i`, stage `s`'s
/// codebook inside the block's `repeat i in 0..n_layer` + `repeat s in
/// 0..synth_stages`. Six synth projections (`wq wk wv wo w1 w2`), each a
/// residual multi-stage codebook (`cb_*` params + `idx_*` u8 constants, one
/// `Vec` entry per stage, named `cb_*_s{st}`/`idx_*_s{st}`) plus optional LoRA
/// factors (`lora_*_a/b`), then the two LayerNorm gain/bias pairs, the MLP
/// biases, and the KAN spline coefficients. Field values are cheap tensor
/// handles cloned from the role-major [`Proj`]s, so grouping them changes no
/// graph node or its order — only how the block reads them. When `lora_rank ==
/// 0` the LoRA factors are inert placeholders (the block's `repeat lora_n` is a
/// 0-trip loop that never references them).
#[derive(Clone)]
struct LayerParams {
    cb_wq: Vec<Tensor>,
    idx_wq: Vec<Tensor>,
    lora_wq_a: Tensor,
    lora_wq_b: Tensor,
    cb_wk: Vec<Tensor>,
    idx_wk: Vec<Tensor>,
    lora_wk_a: Tensor,
    lora_wk_b: Tensor,
    cb_wv: Vec<Tensor>,
    idx_wv: Vec<Tensor>,
    lora_wv_a: Tensor,
    lora_wv_b: Tensor,
    cb_wo: Vec<Tensor>,
    idx_wo: Vec<Tensor>,
    lora_wo_a: Tensor,
    lora_wo_b: Tensor,
    cb_w1: Vec<Tensor>,
    idx_w1: Vec<Tensor>,
    lora_w1_a: Tensor,
    lora_w1_b: Tensor,
    cb_w2: Vec<Tensor>,
    idx_w2: Vec<Tensor>,
    lora_w2_a: Tensor,
    lora_w2_b: Tensor,
    ln1_g: Tensor,
    ln1_b: Tensor,
    ln2_g: Tensor,
    ln2_b: Tensor,
    b1: Tensor,
    b2: Tensor,
    coeff: Tensor,
}

/// Group the role-major [`Proj`]s + norm/bias/coeff collections into one
/// `Vec<LayerParams>`. Pure handle-shuffling — creates no graph nodes, so the
/// param-node order (and thus the RNG-seeded init) is unchanged.
#[allow(clippy::too_many_arguments)]
fn group_layers(
    n_layer: usize,
    synth_stages: usize,
    lora_rank: usize,
    wq: &Proj,
    wk: &Proj,
    wv: &Proj,
    wo: &Proj,
    w1: &Proj,
    w2: &Proj,
    ln1_g: &[Tensor],
    ln1_b: &[Tensor],
    ln2_g: &[Tensor],
    ln2_b: &[Tensor],
    b1: &[Tensor],
    b2: &[Tensor],
    coeff: &[Tensor],
) -> Vec<LayerParams> {
    // Per-projection, per-layer: gather the `synth_stages` codebook/index handles
    // into a `Vec<Tensor>` (stage-major), so the block reads `layers[i].cb_wq[s]`.
    let cbs = |p: &Proj, i: usize| -> Vec<Tensor> {
        (0..synth_stages).map(|st| p.cb[st][i].clone()).collect()
    };
    let idxs = |p: &Proj, i: usize| -> Vec<Tensor> {
        (0..synth_stages).map(|st| p.idx[st][i].clone()).collect()
    };
    // LoRA off → reuse an existing handle as an inert placeholder (never adopted).
    let lora = |p: &Proj, i: usize| {
        if lora_rank > 0 {
            (p.lora_a[i].clone(), p.lora_b[i].clone())
        } else {
            (p.cb[0][i].clone(), p.cb[0][i].clone())
        }
    };
    (0..n_layer)
        .map(|i| {
            let (lora_wq_a, lora_wq_b) = lora(wq, i);
            let (lora_wk_a, lora_wk_b) = lora(wk, i);
            let (lora_wv_a, lora_wv_b) = lora(wv, i);
            let (lora_wo_a, lora_wo_b) = lora(wo, i);
            let (lora_w1_a, lora_w1_b) = lora(w1, i);
            let (lora_w2_a, lora_w2_b) = lora(w2, i);
            LayerParams {
                cb_wq: cbs(wq, i),
                idx_wq: idxs(wq, i),
                lora_wq_a,
                lora_wq_b,
                cb_wk: cbs(wk, i),
                idx_wk: idxs(wk, i),
                lora_wk_a,
                lora_wk_b,
                cb_wv: cbs(wv, i),
                idx_wv: idxs(wv, i),
                lora_wv_a,
                lora_wv_b,
                cb_wo: cbs(wo, i),
                idx_wo: idxs(wo, i),
                lora_wo_a,
                lora_wo_b,
                cb_w1: cbs(w1, i),
                idx_w1: idxs(w1, i),
                lora_w1_a,
                lora_w1_b,
                cb_w2: cbs(w2, i),
                idx_w2: idxs(w2, i),
                lora_w2_a,
                lora_w2_b,
                ln1_g: ln1_g[i].clone(),
                ln1_b: ln1_b[i].clone(),
                ln2_g: ln2_g[i].clone(),
                ln2_b: ln2_b[i].clone(),
                b1: b1[i].clone(),
                b2: b2[i].clone(),
                coeff: coeff[i].clone(),
            }
        })
        .collect()
}

/// Knobs for the single graph builder [`build_graph`] behind the public
/// [`build`] / [`build_dense_init`] / [`build_distill`] entry points. The graph
/// is byte-for-byte the same *shape* whatever these are — only which output is
/// kept, the index assignment, and whether a distillation term is added change.
struct BuildOpts<'a> {
    /// Output the scalar next-token loss (`true`, for training) or the
    /// `[B*T, V]` logits (`false`, for generation).
    with_loss: bool,
    /// PQ dense-init: bake data-optimal index tables + seed values from a trained
    /// dense model. `None` = the random SplitMix index structure.
    init: Option<&'a SynthInit>,
    /// Distillation soft-CE weight against a teacher's per-token distribution
    /// (`0.0` = no distillation term; only applied when `with_loss`).
    distill_alpha: f32,
}

/// Build the GPT as a traced `Func`. With `with_loss` the single output is the
/// scalar next-token cross-entropy loss (for training); otherwise it is the
/// logits `[batch*T, vocab]` (for generation).
///
/// Inputs (filled by the data loader — see [`crate::data::Batcher`]):
/// - `tok_ids` `[batch*T]` — input token ids (fed f32, gathered as i64 in-graph)
/// - `tgt_ids` `[batch*T]` — next-token targets (fed f32; only read when `with_loss`)
///
/// Positions need no input: `wpe [T,D]` broadcasts over the batch inside the block.
pub fn build(cfg: &GptConfig, batch: usize, with_loss: bool) -> Func {
    build_graph(
        cfg,
        batch,
        BuildOpts {
            with_loss,
            init: None,
            distill_alpha: 0.0,
        },
    )
}

/// Like [`build`], but bakes **PQ-derived** index tables from a trained dense
/// model (see [`SynthInit::from_dense`]) instead of the random SplitMix
/// assignment. Pair with [`init_dense`] to also seed the codebooks/LoRA from the
/// dense weights. The graph is byte-for-byte the same shape as [`build`]'s.
pub fn build_dense_init(cfg: &GptConfig, batch: usize, with_loss: bool, init: &SynthInit) -> Func {
    build_graph(
        cfg,
        batch,
        BuildOpts {
            with_loss,
            init: Some(init),
            distill_alpha: 0.0,
        },
    )
}

/// A **distillation** training graph: the loss is the usual next-token CE *plus*
/// `alpha ×` a soft cross-entropy against a dense teacher's per-token
/// distribution, fed as a training-only `teacher_logits [B*T, V]` input (softmaxed
/// on-device). No new *model* parameters — the runtime model is unchanged; this
/// only adds an auxiliary loss (and a training-only input) while `alpha > 0`.
/// `init` optionally also PQ-seeds the codebooks (combine `--init-from` + `--distill`).
pub fn build_distill(cfg: &GptConfig, batch: usize, init: Option<&SynthInit>, alpha: f32) -> Func {
    build_graph(
        cfg,
        batch,
        BuildOpts {
            with_loss: true,
            init,
            distill_alpha: alpha,
        },
    )
}

/// Build the GPT forward as a single `rlx! { … }` graph and wrap it in a `Func`.
/// All params/inputs/constants are declared inside the block *except* the
/// per-layer weight tensors, which are built in an outer `GraphScope`, grouped
/// into a `Vec<LayerParams>`, and adopted by one `bind layers[];` (the DSL has
/// no runtime-length parameter family, so the collection carries per-layer
/// distinctness). With `opts.with_loss`
/// the single output is the scalar next-token CE (`+ distill` when
/// `opts.distill_alpha > 0`); otherwise it is the `[B*T, V]` logits.
fn build_graph(cfg: &GptConfig, batch: usize, opts: BuildOpts) -> Func {
    let BuildOpts {
        with_loss,
        init,
        distill_alpha,
    } = opts;
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
    let n_layer = cfg.n_layer;
    let lora_rank = cfg.lora_rank;
    // Residual-VQ stage count — now GRAPH-LIVE (the block's `repeat s in
    // 1..synth_stages` stage loop sums exactly this many codebook stages).
    let synth_stages = cfg.synth_stages;
    // Scalars passed by value (`~x`) into the block's ops.
    let (eps, ed, _ne) = (LN_EPS, SYNTH_ED as u32, SYNTH_NE as u32);
    let (kb, kmin, kmax) = (KAN_BASIS, KAN_MIN, KAN_MAX);
    // Gate the optional LoRA correction / distillation term as a 0-or-1-trip
    // runtime `repeat` inside the block (the DSL can't branch, but a zero-length
    // loop drops the ops — and its unfed inputs stay harmlessly dead).
    let lora_n = usize::from(lora_rank > 0);
    let alpha = distill_alpha;
    let distill_n = usize::from(with_loss && alpha > 0.0);

    // ── Per-layer weight collections (built OUTSIDE the block, adopted by `bind`).
    let mut ext = GraphScope::new("gpt-tiny");
    let wq = build_proj(
        &mut ext,
        "wq",
        d,
        d,
        0,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    let wk = build_proj(
        &mut ext,
        "wk",
        d,
        d,
        1,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    let wv = build_proj(
        &mut ext,
        "wv",
        d,
        d,
        2,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    let wo = build_proj(
        &mut ext,
        "wo",
        d,
        d,
        3,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    let w1 = build_proj(
        &mut ext,
        "w1",
        d,
        ff,
        4,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    let w2 = build_proj(
        &mut ext,
        "w2",
        ff,
        d,
        5,
        n_layer,
        synth_stages,
        lora_rank,
        init,
    );
    // Per-layer norm gains/biases, MLP biases, and KAN spline coeffs (created in
    // this order to keep the param-node order — and thus the RNG-seeded init —
    // stable; see [`group_layers`]).
    let kb_n = KAN_BASIS as usize;
    let ln1_g: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("ln1_g{i}"), shape![d]))
        .collect();
    let ln1_b: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("ln1_b{i}"), shape![d]))
        .collect();
    let ln2_g: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("ln2_g{i}"), shape![d]))
        .collect();
    let ln2_b: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("ln2_b{i}"), shape![d]))
        .collect();
    let b1: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("b1{i}"), shape![ff]))
        .collect();
    let b2: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("b2{i}"), shape![d]))
        .collect();
    let coeff: Vec<Tensor> = (0..n_layer)
        .map(|i| ext.param(format!("coeff{i}"), shape![ff, kb_n]))
        .collect();
    // Group every per-layer tensor into ONE collection, adopted by `bind layers[]`.
    let layers = group_layers(
        n_layer,
        synth_stages,
        lora_rank,
        &wq,
        &wk,
        &wv,
        &wo,
        &w1,
        &w2,
        &ln1_g,
        &ln1_b,
        &ln2_g,
        &ln2_b,
        &b1,
        &b2,
        &coeff,
    );
    // Distillation weight (a rank-0 scalar constant, dead unless distilling).
    let talpha = ext.constant(f64::from(alpha), DType::F32);
    // z-loss weight (rank-0 scalar); pins the logit log-partition. Only ever
    // multiplies the loss branch, so it is pruned along with the loss for inference.
    let zbeta = ext.constant(f64::from(Z_LOSS_BETA), DType::F32);

    // ── ONE graph, both modes. The transformer body (embedding → the
    // `repeat i in 0..n_layer` stack → final norm → tied LM head → logits) is
    // written exactly once; the block always declares the training-only inputs
    // (`tgt_ids`/`teacher_logits`) and computes both `logits` and the scalar
    // `loss`, ending `out logits, loss;` (⇒ `graph.outputs == [logits, loss]`).
    // The two run modes are then selected purely by TRIMMING `graph.outputs`
    // below — no re-declared body. `logits` never depends on
    // `tgt_ids`/`teacher_logits`, so once inference keeps only the `logits`
    // output, DCE prunes the loss cone and those inputs go dead (never fed).
    let mut graph = rlx! {
        graph "gpt-tiny";
        input tok_ids: [bt];
        input tgt_ids: [bt];
        input teacher_logits: [bt, v];
        param wte: [v, d];            // token embedding — reused (tied) as the LM head
        param wpe: [t, d];            // learned positional embedding
        param lnf_g: [d];  param lnf_b: [d];
        bind talpha;
        bind zbeta;
        bind layers[];               // the whole per-layer weight stack, one bind

        // Token embedding by gather (ids fed f32 → i64) + broadcast positions.
        let ids = tok_ids.cast(DType::I64);
        let tok = wte.gather(ids, 0);
        let tok3 = tok.reshape((vec![b as i64, t as i64, d as i64]));
        let emb = tok3 + wpe;
        let h = emb.reshape((vec![bt as i64, d as i64]));

        repeat i in 0..n_layer {
            // ── Attention block: pre-norm → Q/K/V synth-proj → causal attn → O.
            let hn = h.layer_norm(layers[i].ln1_g, layers[i].ln1_b, ~eps);
            // Residual multi-stage VQ: seed at stage 0, then a runtime `repeat s`
            // sums `synth_stages` codebook stages (`W = Σ_s codebook_s[idx_s]`).
            // W_eff FOLD: reconstruct the combined weight (all stages + LoRA) once,
            // then ONE matmul per projection → dense-count GEMMs (1 fwd, 2 bwd);
            // the multi-stage/LoRA gradients collapse to codebook scatters + rank-r
            // products (no batch dim). Bit-equivalent to the per-stage form.
            // W_eff FOLD with a fused native reconstruct (`Op::SynthReconstruct`
            // writes W[k,n] in one dispatch — no separate cast/gather/reshape/
            // transpose), summed with LoRA, then one plain matmul per projection.
            let wq = layers[i].idx_wq[0].synth_reconstruct(layers[i].cb_wq[0], ~ed);
            repeat s in 1..synth_stages { let wq = wq + layers[i].idx_wq[s].synth_reconstruct(layers[i].cb_wq[s], ~ed); }
            repeat lora_n { let wq = wq + layers[i].lora_wq_a.matmul_t(layers[i].lora_wq_b); }
            let q = (hn @ wq).reshape((vec![b as i64, t as i64, nh as i64, dh as i64]));
            let wk = layers[i].idx_wk[0].synth_reconstruct(layers[i].cb_wk[0], ~ed);
            repeat s in 1..synth_stages { let wk = wk + layers[i].idx_wk[s].synth_reconstruct(layers[i].cb_wk[s], ~ed); }
            repeat lora_n { let wk = wk + layers[i].lora_wk_a.matmul_t(layers[i].lora_wk_b); }
            let k = (hn @ wk).reshape((vec![b as i64, t as i64, nh as i64, dh as i64]));
            let wv = layers[i].idx_wv[0].synth_reconstruct(layers[i].cb_wv[0], ~ed);
            repeat s in 1..synth_stages { let wv = wv + layers[i].idx_wv[s].synth_reconstruct(layers[i].cb_wv[s], ~ed); }
            repeat lora_n { let wv = wv + layers[i].lora_wv_a.matmul_t(layers[i].lora_wv_b); }
            let vv = (hn @ wv).reshape((vec![b as i64, t as i64, nh as i64, dh as i64]));
            let attn = q.attention(k, vv, ~nh, ~dh, MaskKind::Causal).reshape((vec![bt as i64, d as i64]));
            let wo = layers[i].idx_wo[0].synth_reconstruct(layers[i].cb_wo[0], ~ed);
            repeat s in 1..synth_stages { let wo = wo + layers[i].idx_wo[s].synth_reconstruct(layers[i].cb_wo[s], ~ed); }
            repeat lora_n { let wo = wo + layers[i].lora_wo_a.matmul_t(layers[i].lora_wo_b); }
            let h = h + (attn @ wo);

            // ── FFN block: pre-norm → synth up-proj + bias → KAN spline → synth down-proj + bias.
            let hn2 = h.layer_norm(layers[i].ln2_g, layers[i].ln2_b, ~eps);
            let w1 = layers[i].idx_w1[0].synth_reconstruct(layers[i].cb_w1[0], ~ed);
            repeat s in 1..synth_stages { let w1 = w1 + layers[i].idx_w1[s].synth_reconstruct(layers[i].cb_w1[s], ~ed); }
            repeat lora_n { let w1 = w1 + layers[i].lora_w1_a.matmul_t(layers[i].lora_w1_b); }
            let up = (hn2 @ w1) + layers[i].b1;
            let act = up.spline_activation(layers[i].coeff, ~kb, ~kmin, ~kmax);
            let w2 = layers[i].idx_w2[0].synth_reconstruct(layers[i].cb_w2[0], ~ed);
            repeat s in 1..synth_stages { let w2 = w2 + layers[i].idx_w2[s].synth_reconstruct(layers[i].cb_w2[s], ~ed); }
            repeat lora_n { let w2 = w2 + layers[i].lora_w2_a.matmul_t(layers[i].lora_w2_b); }
            let h = h + ((act @ w2) + layers[i].b2);
        }

        // Final norm + tied LM head → logits (the inference output).
        let hf = h.layer_norm(lnf_g, lnf_b, ~eps);
        let logits = hf.matmul_t(wte);
        // Next-token loss (the training output). Depends on `tgt_ids` — which is
        // why it's split off and pruned for inference. `softmax_cross_entropy`
        // takes **f32-encoded** class ids (see
        // `Tensor::softmax_cross_entropy_with_logits`), so feed the f32 `tgt_ids`
        // straight through — an I64 cast makes the kernel reinterpret the label
        // bytes as f32, collapsing every index to 0 (invisible under uniform
        // logits, silently wrong once the logits become peaked).
        // Cross-entropy + z-loss: `β·mean(logsumexp(logits)²)` keeps the softmax
        // normalizer near 1, so the logits can't drift to the f32 overflow that
        // NaN'd training (`logsumexp` is the same log-partition CE already forms).
        let zlse = logits.logsumexp(1, false);
        let loss = mean(softmax_cross_entropy(logits, tgt_ids)) + mean(zlse * zlse) * zbeta;
        // Optional distillation: `+ alpha * mean(CE(logits, softmax(teacher)))`.
        repeat distill_n {
            let tp = teacher_logits.softmax(1);
            let kd = mean(cross_entropy(logits, tp));
            let loss = loss + kd * talpha;
        }
        out logits, loss;
    };

    // Select the run mode by trimming to the single output the runtime keys on:
    // generation reads `logits` (`run_on(..).remove(0)`), training reads the
    // scalar `loss` (`value_and_grad` seeds `outputs[0]`). `out logits, loss;`
    // put them at indices 0 and 1 respectively.
    let logits_out = graph.outputs[0];
    let loss_out = graph.outputs[1];
    graph.set_outputs(if with_loss {
        vec![loss_out]
    } else {
        vec![logits_out]
    });
    Func::from_graph(graph)
}

/// Seed a freshly-built model with GPT-2-style initialization: `N(0, 0.02)` for
/// linear/embedding weights (residual projections `wo`/`w2` scaled by
/// `1/√(2·n_layer)`), ones for LayerNorm gains, zeros for all biases. Uses the
/// framework's [`Func::init_params`](rlx_tensor::Func::init_params) so one call
/// covers every parameter the graph declares.
pub fn init(model: Func, cfg: &GptConfig, seed: u64) -> Func {
    let n_layer = cfg.n_layer;
    let synth_stages = cfg.synth_stages;
    let mut rng = Rng::new(seed);
    model.init_params(move |name, dims| {
        let n: usize = dims.iter().product();
        default_init(name, n, n_layer, synth_stages, &mut rng)
    })
}

/// Seed a model from a **dense** PQ init ([`SynthInit::from_dense`], baked into
/// the graph via [`build_dense_init`]). Codebooks, LoRA factors, and the copied
/// embeddings/norms/biases take their [`SynthInit`] values; the KAN spline
/// coeffs stay ≈GELU; anything the dense init doesn't cover falls back to the
/// standard random scheme (with `seed` as its RNG). No new parameters — the same
/// tensors as [`init`], only better-valued.
pub fn init_dense(model: Func, cfg: &GptConfig, init: &SynthInit, seed: u64) -> Func {
    let n_layer = cfg.n_layer;
    let synth_stages = cfg.synth_stages;
    let mut rng = Rng::new(seed);
    model.init_params(move |name, dims| {
        let n: usize = dims.iter().product();
        // PQ-derived / copied value, when present and the right length.
        if let Some(v) = init.values.get(name) {
            if v.len() == n {
                return v.clone();
            }
        }
        // KAN spline coeffs are not PQ-derived — keep the ≈GELU init.
        if name.starts_with("coeff") {
            return gelu_coeff(n);
        }
        default_init(name, n, n_layer, synth_stages, &mut rng)
    })
}

/// The standard GPT-2-style random init for a single parameter: `1.0` for norm
/// gains, `0.0` for biases, ≈GELU for KAN coeffs, and `N(0, 0.02)` (residual
/// projections and residual-VQ stages down-scaled) for codebooks / LoRA-A /
/// embeddings. Shared by [`init`] and [`init_dense`]'s fallback.
fn default_init(
    name: &str,
    n: usize,
    n_layer: usize,
    synth_stages: usize,
    rng: &mut Rng,
) -> Vec<f32> {
    if name.ends_with("_g") {
        vec![1.0; n] // LayerNorm gain
    } else if name.ends_with("_b") || name.starts_with("b1") || name.starts_with("b2") {
        vec![0.0; n] // LayerNorm / MLP bias
    } else if name.starts_with("coeff") {
        gelu_coeff(n)
    } else {
        // Codebook stages (cb_*_s*), LoRA A (cb_*_lora_a) and embeddings
        // (wte/wpe): N(0, 0.02); the synthesized weight W[j,·]=codebook[idx] is
        // then also ~N(0, 0.02). (LoRA B is caught above → zeros, so the dense
        // correction starts off.)
        let mut std = if is_residual_proj(name) {
            0.02 / (2.0 * n_layer as f32).sqrt()
        } else {
            0.02
        };
        // Residual-VQ stages SUM, so each stage codebook starts 1/√stages smaller.
        if name.starts_with("cb_") && name.contains("_s") {
            std /= (synth_stages as f32).sqrt();
        }
        (0..n).map(|_| rng.normal() * std).collect()
    }
}

/// KAN spline coeffs initialized so each per-channel function ≈ GELU. The naive
/// `coeff[c,g] = gelu(center_g)` is a crude *bump-sum*: the Gaussian RBFs overlap
/// (`Σ_g rbf_g(x) ≈ 1.8` near the center), so weighting each by `gelu(center_g)`
/// over-shoots GELU by that factor — the PQ-reconstructed dense (GELU) teacher's
/// FFN then diverges even with faithful weights. Instead, **least-squares fit**
/// the basis to GELU: `coeff = argmin ‖Φ·coeff − gelu(x)‖²` over a dense grid,
/// via the ridge-normal equations `coeff = (ΦᵀΦ + λI)⁻¹ Φᵀ gelu`. Every channel
/// gets the same GELU fit (`n = channels·KAN_BASIS`; `coeff[c,g] = fit[g]`), so
/// the spline reproduces GELU faithfully at zero extra parameters.
fn gelu_coeff(n: usize) -> Vec<f32> {
    let nb = KAN_BASIS as usize;
    let fit = gelu_spline_fit();
    (0..n).map(|p| fit[p % nb] as f32).collect()
}

/// Least-squares fit of the Gaussian-RBF KAN basis (the exact basis
/// `spline_activation_f32` evaluates: `center_g = KAN_MIN + g·step`,
/// `inv_h = 1/step`) to the exact erf-GELU over `[KAN_MIN, KAN_MAX]`. Returns the
/// `KAN_BASIS` per-channel coefficients (shared across channels). See
/// [`gelu_coeff`].
fn gelu_spline_fit() -> Vec<f64> {
    let nb = KAN_BASIS as usize;
    let (lo, hi) = (KAN_MIN as f64, KAN_MAX as f64);
    let step = (hi - lo) / (nb as f64 - 1.0);
    let inv_h = 1.0 / step;
    let center = |g: usize| lo + g as f64 * step;
    // Dense grid so the fit constrains the whole active range (including tails).
    let samples = 512usize;
    let mut ata = vec![0f64; nb * nb]; // ΦᵀΦ (+ ridge)
    let mut aty = vec![0f64; nb]; // Φᵀ·gelu
    let mut phi = vec![0f64; nb];
    for j in 0..samples {
        let x = lo + (hi - lo) * (j as f64 / (samples as f64 - 1.0));
        let y = gelu_exact(x);
        for (g, p) in phi.iter_mut().enumerate() {
            let z = (x - center(g)) * inv_h;
            *p = (-(z * z)).exp();
        }
        for a in 0..nb {
            aty[a] += phi[a] * y;
            for b in 0..nb {
                ata[a * nb + b] += phi[a] * phi[b];
            }
        }
    }
    let lambda = 1e-6; // tiny ridge for numerical stability
    for g in 0..nb {
        ata[g * nb + g] += lambda;
    }
    solve_linear(&mut ata, &mut aty, nb); // aty ← coeff (solved in place)
    aty
}

/// Exact erf-based GELU `0.5·x·(1+erf(x/√2))` (matches the dense teacher's
/// `.gelu()` = `Activation::Gelu`), in f64.
fn gelu_exact(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// erf via Abramowitz & Stegun 7.1.26 (the CPU backend's GELU erf), in f64.
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// Solve `A·x = b` (`A` `n×n` row-major, symmetric PD) in place by Gauss–Jordan
/// elimination with partial pivoting; the solution overwrites `b`. `n` is tiny
/// (`KAN_BASIS`), so an O(n³) direct solve is trivial.
fn solve_linear(a: &mut [f64], b: &mut [f64], n: usize) {
    for col in 0..n {
        // Partial pivot: largest-magnitude entry in this column.
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        if d.abs() < 1e-12 {
            continue; // (ridge keeps A well-conditioned; guard just in case)
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    for i in 0..n {
        let d = a[i * n + i];
        if d.abs() > 1e-12 {
            b[i] /= d;
        }
    }
}

/// Residual-path projections GPT-2 down-scales at init for depth stability — the
/// attention output (`cb_wo`) and MLP down (`cb_w2`) codebooks here.
fn is_residual_proj(name: &str) -> bool {
    name.starts_with("cb_wo") || name.starts_with("cb_w2")
}
