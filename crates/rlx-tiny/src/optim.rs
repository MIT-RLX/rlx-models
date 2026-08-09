//! The canonical **Muon** training recipe: Newton–Schulz-orthogonalized
//! momentum (`rlx_optim::Muon`) on the 2-D hidden weight matrices, with
//! **AdamW** on everything Muon isn't meant for — the token/positional
//! embeddings and the 1-D biases / LayerNorm gains. (Muon's own non-2-D
//! fallback is plain SGD-momentum, so routing those to AdamW matches the
//! reference recipe.)
//!
//! A single learning-rate schedule drives both: `set_lr(lr)` sets AdamW to `lr`
//! and Muon to `lr · ratio`, where `ratio = muon_lr / adamw_lr`, so both track
//! the warmup-cosine curve proportionally.

use rlx_tensor::{AdamW, Muon, Optimizer};

/// Muon (2-D matrices) + AdamW (embeddings, biases, norms), with a per-group LR
/// boost and per-tensor trust-region cap layered on the AdamW path (see `step`).
pub struct HybridOptimizer {
    muon: Muon,
    adamw: AdamW,
    /// `muon_lr / adamw_lr`, so one scheduled LR scales both.
    muon_ratio: f32,
    /// Effective-LR multiplier for the **synth** degrees of freedom (codebooks,
    /// their LoRA factors, KAN coeffs). AdamW's per-coordinate normalization
    /// otherwise crawls on the ~0.05-scale codebooks and the loss plateaus; a
    /// modest boost restores their step without the √256 Muon blow-up.
    /// Overridable via `RLX_TINY_SYNTH_LR_MULT` (A/B tuning). Default `3.0`.
    synth_lr_mult: f32,
    /// Trust-region cap ρ: after the (boosted) AdamW step, no tensor may move
    /// more than `ρ·‖θ‖` in a single update — a scale-invariant hard guard that
    /// bounds any group's per-step motion (LARS-style). `<= 0` disables.
    /// Overridable via `RLX_TINY_TRUST_RHO`. Default `0.05`.
    trust_rho: f32,
}

impl HybridOptimizer {
    /// `adamw_lr` typically ~3e-4; `muon_lr` typically ~2e-2. `weight_decay`
    /// (decoupled) is applied by AdamW; Muon uses a light decay.
    pub fn new(adamw_lr: f32, muon_lr: f32, weight_decay: f32) -> Self {
        Self {
            muon: Muon::new(muon_lr).with_weight_decay(weight_decay * 0.1),
            adamw: AdamW::new(adamw_lr).with_weight_decay(weight_decay),
            muon_ratio: if adamw_lr > 0.0 {
                muon_lr / adamw_lr
            } else {
                0.0
            },
            synth_lr_mult: env_f32("RLX_TINY_SYNTH_LR_MULT", 3.0),
            trust_rho: env_f32("RLX_TINY_TRUST_RHO", 0.05),
        }
    }

    /// Effective-LR multiplier for `name`: boost the synth DOF, leave everything
    /// else (embeddings / norms / biases) at 1×.
    fn group_mult(&self, name: &str) -> f32 {
        if name.starts_with("cb_") || name.starts_with("coeff") {
            self.synth_lr_mult
        } else {
            1.0
        }
    }

    /// Route a parameter to Muon iff it is a genuine 2-D **dense weight matrix**.
    ///
    /// Muon replaces the momentum with its closest semi-orthogonal matrix and
    /// scales the update by `√max(m,n)` — a recipe that only makes sense for a
    /// linear-map weight matrix. In this "functions not data" model there are
    /// none: every 2-D tensor is a **codebook** (`cb_*`, including its `_lora_*`
    /// factors) or a **KAN spline coefficient table** (`coeff*`). Newton–Schulz
    /// on a `[256,4]` codebook is meaningless, and its `√256 = 16×` scaling makes
    /// the effective step ≈ `lr·16` (~2e-2/elem) — ~60× too large for the ~0.05
    /// codebook values, which drives a chaotic blow-up to NaN after a few hundred
    /// steps. Route all of those (and the `wte`/`wpe` embeddings, which behave
    /// like embeddings under the tied head) to AdamW; keep Muon available for real
    /// dense weight matrices in other models that reuse this optimizer.
    fn use_muon(name: &str, shape: &[usize]) -> bool {
        shape.len() == 2
            && !name.starts_with("wte")
            && !name.starts_with("wpe")
            && !name.starts_with("cb_")
            && !name.starts_with("coeff")
    }
}

impl Optimizer for HybridOptimizer {
    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        if Self::use_muon(name, shape) {
            self.muon.step(name, shape, param, grad);
            return;
        }
        let mult = self.group_mult(name);
        let rho = self.trust_rho;
        // Plain AdamW when there's nothing to adjust (keeps the common path exact).
        if (mult - 1.0).abs() < 1e-9 && rho <= 0.0 {
            self.adamw.step(name, shape, param, grad);
            return;
        }
        // Snapshot θ, take AdamW's natural step, then reshape the delta:
        //   1. per-group LR boost  (Δ ← mult·Δ),
        //   2. trust-region cap    (‖Δ‖ ≤ ρ·‖θ‖).
        let p_norm = l2(param);
        let snap: Vec<f32> = param.to_vec();
        self.adamw.step(name, shape, param, grad);
        let mut d2 = 0.0f64;
        for i in 0..param.len() {
            let d = (param[i] - snap[i]) * mult;
            param[i] = snap[i] + d;
            d2 += (d as f64) * (d as f64);
        }
        if rho > 0.0 {
            let d_norm = d2.sqrt() as f32;
            let cap = rho * p_norm.max(1e-6);
            if d_norm > cap {
                let s = cap / d_norm;
                for i in 0..param.len() {
                    param[i] = snap[i] + (param[i] - snap[i]) * s;
                }
            }
        }
    }

    fn end_iteration(&mut self) {
        self.muon.end_iteration();
        self.adamw.end_iteration();
    }

    fn set_lr(&mut self, lr: f32) {
        self.adamw.set_lr(lr);
        self.muon.set_lr(lr * self.muon_ratio);
    }
}

/// L2 norm in `f64` accumulation (stable for the large embedding/codebook tensors).
fn l2(x: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for &v in x {
        s += (v as f64) * (v as f64);
    }
    s.sqrt() as f32
}

/// Parse an `f32` tuning override from the environment, falling back to `default`.
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
