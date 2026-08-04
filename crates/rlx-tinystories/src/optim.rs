//! The canonical **Muon** training recipe: Newton–Schulz-orthogonalized
//! momentum ([`rlx_optim::Muon`]) on the 2-D hidden weight matrices, with
//! **AdamW** on everything Muon isn't meant for — the token/positional
//! embeddings and the 1-D biases / LayerNorm gains. (Muon's own non-2-D
//! fallback is plain SGD-momentum, so routing those to AdamW matches the
//! reference recipe.)
//!
//! A single learning-rate schedule drives both: `set_lr(lr)` sets AdamW to `lr`
//! and Muon to `lr · ratio`, where `ratio = muon_lr / adamw_lr`, so both track
//! the warmup-cosine curve proportionally.

use rlx_tensor::{AdamW, Muon, OptItem, Optimizer};

/// Muon (2-D matrices) + AdamW (embeddings, biases, norms).
///
/// The Muon weight matrices are the optimizer's whole cost — each runs an
/// independent Newton–Schulz orthogonalization (a handful of small Accelerate
/// matmuls). Because a matrix's update depends only on its own momentum buffer
/// (keyed by name) and the shared scalar config, the ~36 matrices are fully
/// independent work. We shard them across `muon.len()` persistent [`Muon`]
/// instances — each holds the momentum for a fixed subset of matrices and runs
/// on its own thread — so the step parallelizes across matrices while staying
/// **bit-identical** to the single-instance serial loop (same per-matrix state,
/// same math; only *which* instance owns a matrix's momentum changes, and that
/// mapping is stable across iterations). AdamW (cheap, elementwise) runs
/// concurrently on the caller thread.
pub struct HybridOptimizer {
    /// `K` persistent Muon shards. Matrix `j` (in stable `param_names` order,
    /// among the Muon-routed params) belongs to shard `j / ceil(n/K)`, so its
    /// momentum lives in exactly one shard for the whole run.
    muon: Vec<Muon>,
    adamw: AdamW,
    /// `muon_lr / adamw_lr`, so one scheduled LR scales both.
    muon_ratio: f32,
}

impl HybridOptimizer {
    /// `adamw_lr` typically ~3e-4; `muon_lr` typically ~2e-2. `weight_decay`
    /// (decoupled) is applied by AdamW; Muon uses a light decay.
    pub fn new(adamw_lr: f32, muon_lr: f32, weight_decay: f32) -> Self {
        // Newton–Schulz iteration count (default 5, the published value). Fewer
        // iterations = fewer small matmuls per weight = cheaper optimizer step,
        // at the cost of a less-orthogonal update. `RLX_TS_NS_STEPS` overrides.
        let ns_steps: u32 = std::env::var("RLX_TS_NS_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        // Number of Muon shards = parallel workers for the Newton–Schulz matmuls.
        // Default = CPU parallelism (capped), since sharding is bit-exact and
        // strictly cheaper; `RLX_TS_MUON_SHARDS=1` forces the serial baseline.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let shards: usize = std::env::var("RLX_TS_MUON_SHARDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(cores.min(8))
            .max(1);
        let base = Muon::new(muon_lr)
            .with_weight_decay(weight_decay * 0.1)
            .with_ns_steps(ns_steps);
        Self {
            muon: (0..shards).map(|_| base.clone()).collect(),
            adamw: AdamW::new(adamw_lr).with_weight_decay(weight_decay),
            muon_ratio: if adamw_lr > 0.0 {
                muon_lr / adamw_lr
            } else {
                0.0
            },
        }
    }

    /// Route a parameter to Muon iff it is a 2-D weight matrix that is **not**
    /// an embedding table (`wte`/`wpe`, which — with the tied head — behave like
    /// embeddings and want AdamW).
    fn use_muon(name: &str, shape: &[usize]) -> bool {
        // Muon's √max(m,n)-scaled orthogonalized step is too aggressive for this
        // small from-scratch model on the larger BPE vocab (NaNs even with z-loss +
        // trust-cap). `RLX_TS_NO_MUON=1` routes everything to AdamW — matching
        // rlx-tiny's all-AdamW recipe for a like-for-like optimizer comparison.
        if std::env::var("RLX_TS_NO_MUON").as_deref() == Ok("1") {
            return false;
        }
        shape.len() == 2 && !name.starts_with("wte") && !name.starts_with("wpe")
    }
}

/// Take one sub-optimizer's step on `it`, then apply the trust-region cap
/// `‖Δθ‖ ≤ ρ·‖θ‖`: snapshot θ, step, and shrink the delta if it exceeds ρ
/// relative to the param norm. Generic over the concrete optimizer so the
/// Muon and AdamW groups can call it on their own `&mut` without dyn dispatch —
/// which is what lets [`HybridOptimizer::step_batch`] run the two groups on
/// separate threads. `RLX_TS_TRUST_RHO=0` disables the cap; default 0.02.
fn stepped<O: Optimizer>(opt: &mut O, it: &mut OptItem<'_>, rho: f32) {
    if rho <= 0.0 {
        opt.step(it.name, it.shape, it.param, it.grad);
        return;
    }
    let p_norm = l2(it.param);
    let snap: Vec<f32> = it.param.to_vec();
    opt.step(it.name, it.shape, it.param, it.grad);
    let mut d2 = 0.0f64;
    for i in 0..it.param.len() {
        let d = (it.param[i] - snap[i]) as f64;
        d2 += d * d;
    }
    let d_norm = d2.sqrt() as f32;
    let cap = rho * p_norm.max(1e-6);
    if d_norm > cap && d_norm > 0.0 {
        let s = cap / d_norm;
        for i in 0..it.param.len() {
            it.param[i] = snap[i] + (it.param[i] - snap[i]) * s;
        }
    }
}

impl Optimizer for HybridOptimizer {
    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        // Single-parameter path (not the training hot path — see `step_batch`).
        // Route Muon params to shard 0 so the rare `step`-only caller stays
        // self-consistent.
        let rho = trust_rho();
        let mut it = OptItem {
            name,
            shape,
            param,
            grad,
        };
        if Self::use_muon(name, shape) {
            stepped(&mut self.muon[0], &mut it, rho);
        } else {
            stepped(&mut self.adamw, &mut it, rho);
        }
    }

    /// Parallel optimizer step. Partition the batch into the Muon 2-D weights
    /// (the entire cost — an independent Newton–Schulz per matrix) and the AdamW
    /// tail (cheap, elementwise), then run each Muon shard on its own thread with
    /// AdamW on the caller thread. Bit-identical to the serial loop: every item's
    /// update is a pure function of its own name-keyed state + its param/grad, the
    /// matrix→shard mapping is stable, and shards/AdamW touch disjoint state.
    fn step_batch(&mut self, items: &mut [OptItem<'_>]) {
        let rho = trust_rho();
        // Partition Muon items to the front, preserving their relative order (so
        // the matrix→shard chunk mapping is stable across iterations); AdamW tail
        // after.
        let mut lo = 0usize;
        for i in 0..items.len() {
            if Self::use_muon(items[i].name, items[i].shape) {
                // Rotate items[lo..=i] right by one → moves items[i] to `lo` while
                // keeping the front block in encounter order (stable partition).
                items[lo..=i].rotate_right(1);
                lo += 1;
            }
        }
        let (muon_items, adamw_items) = items.split_at_mut(lo);
        let n = muon_items.len();
        // One less shard than requested if we have fewer matrices; never 0.
        let k = self.muon.len().min(n.max(1));
        let force_serial = k <= 1
            && !muon_items.is_empty()
            && std::env::var("RLX_TS_OPT_OVERLAP").as_deref() != Ok("1");
        if force_serial {
            // K=1, overlap off → original serial behavior, no threads.
            for it in muon_items.iter_mut() {
                stepped(&mut self.muon[0], it, rho);
            }
            for it in adamw_items.iter_mut() {
                stepped(&mut self.adamw, it, rho);
            }
            return;
        }
        let chunk = n.div_ceil(k);
        let adamw = &mut self.adamw;
        std::thread::scope(|s| {
            let mut rest: &mut [OptItem] = muon_items;
            for shard in self.muon.iter_mut() {
                if rest.is_empty() {
                    break;
                }
                let take = chunk.min(rest.len());
                let (head, tail) = rest.split_at_mut(take);
                rest = tail;
                s.spawn(move || {
                    for it in head.iter_mut() {
                        stepped(shard, it, rho);
                    }
                });
            }
            // AdamW tail on the caller thread, concurrent with the shard workers.
            for it in adamw_items.iter_mut() {
                stepped(adamw, it, rho);
            }
        });
    }

    fn end_iteration(&mut self) {
        for shard in &mut self.muon {
            shard.end_iteration();
        }
        self.adamw.end_iteration();
    }

    fn set_lr(&mut self, lr: f32) {
        self.adamw.set_lr(lr);
        for shard in &mut self.muon {
            shard.set_lr(lr * self.muon_ratio);
        }
    }
}

/// L2 norm (f64 accumulation, stable for the large embedding/weight tensors).
fn l2(x: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for &v in x {
        s += (v as f64) * (v as f64);
    }
    s.sqrt() as f32
}

/// Trust-region radius ρ from `RLX_TS_TRUST_RHO` (default 0.02). 0.02 keeps Muon
/// stable from scratch on the larger BPE vocab — 0.05 let the weights grow ~5%/step
/// coherently until the attention softmax overflowed to NaN (~step 400); 0.02/0.01
/// train cleanly at the same quality. So Muon works by default (no `RLX_TS_NO_MUON`).
fn trust_rho() -> f32 {
    std::env::var("RLX_TS_TRUST_RHO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02)
}
