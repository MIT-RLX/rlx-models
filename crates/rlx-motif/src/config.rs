// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `MotifConfig` (`model_type = "Motif"`) — the Motif-3 family.
//!
//! Reference: `configuration_motif.py` / `modeling_motif.py` on
//! [Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3).
//!
//! Keys present in the published `config.json` that are **dead** in the
//! reference modeling code, and so deliberately not modelled here:
//!
//! * `max_window_layers` — `MotifGDLAttention` picks SWA layers from
//!   `sliding_window_pattern`/`sliding_window_period` only; the "bottom N layers
//!   are local" rule Qwen uses this key for never runs.
//! * `k_ratio` — read into the config object, never used by any layer.
//! * `headwise_attn_output_gate` — read by `MotifGDLAttention.__init__`, but the
//!   forward only ever consults `elementwise_attn_output_gate`.
//! * `attention_dropout` — inference is always eval mode.
//! * `_debug_force_load_balance`, `load_balance_coeff`, `router_aux_loss_coef`,
//!   `output_router_logits` — training-only.
//! * `num_nextn_predict_layers` — the checkpoint carries one `model.mtp_layers.0`
//!   block, but `modeling_motif.py` never instantiates it. Inference ignores it
//!   exactly as the reference does; see [`MotifConfig::validate`].
//! * `mscale` in `rope_scaling` — the config's *top-level* `mscale` is the one
//!   `MotifGDLAttention` folds into the softmax scale.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// Which mask a GDLA layer runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttn {
    /// Full causal attention, YaRN-interpolated RoPE, YaRN `mscale` on the score
    /// scale.
    Global,
    /// Sliding-window causal attention over `w + 1` keys (`[q-w, q]`), plain
    /// `swa_rope_theta` RoPE, no `mscale`.
    Sliding(usize),
}

/// The `rope_scaling` sub-object. Motif-3 ships `rope_type = "yarn"`; the
/// interpolation is applied to `inv_freq` only, never to the cos/sin table.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RopeScaling {
    /// Only `"yarn"` changes anything here; anything else leaves `inv_freq` alone.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// Context-extension factor the low frequencies are divided by.
    #[serde(default)]
    pub factor: Option<f32>,
    /// Context length the model was pretrained at (falls back to
    /// [`MotifConfig::original_seq_len`]).
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Rotations-per-context above which frequencies are left un-interpolated.
    #[serde(default)]
    pub beta_fast: Option<f32>,
    /// Rotations-per-context below which frequencies are fully interpolated.
    #[serde(default)]
    pub beta_slow: Option<f32>,
    /// RoPE base for the global layers (overrides [`MotifConfig::rope_theta`]).
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

/// A parsed Motif `config.json`.
///
/// Field names mirror the HF config verbatim, so a stanza can be read straight
/// off the checkpoint. Derived quantities that the reference computes on the
/// fly — the head split, the per-layer mask, the softmax scale — are methods,
/// not fields; see [`MotifConfig::layer_attn`] and
/// [`MotifConfig::attn_score_scale`].
#[derive(Debug, Clone, Deserialize)]
pub struct MotifConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    /// Width of the **dense** FFN (layers before `n_dense_first_layers`); the
    /// routed experts use [`MotifConfig::moe_intermediate_size`].
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    /// GQA KV heads. Must equal `num_noise_heads` — the two partitions of the
    /// head axis have to coincide; see [`MotifConfig::validate`].
    #[serde(default = "d_heads")]
    pub num_key_value_heads: usize,
    /// Only `"poly_norm"` is built (see [`crate::polynorm`]).
    #[serde(default = "d_act")]
    pub hidden_act: String,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base. The MHC norms use their own hardcoded [`MHC_NORM_EPS`].
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    // ── sliding window ──
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Keys of history a windowed layer sees, *excluding* the query's own
    /// position — pass this straight to `MaskKind::SlidingWindow`.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// `"interleave"` (every `sliding_window_period`-th layer is global) or
    /// `"all"`.
    #[serde(default = "d_swa_pattern")]
    pub sliding_window_pattern: String,
    #[serde(default = "d_swa_period")]
    pub sliding_window_period: usize,
    /// RoPE base for windowed layers — they get a plain, un-interpolated table.
    #[serde(default)]
    pub swa_rope_theta: Option<f32>,

    // ── differential attention ──
    /// Q/K width per head (`qk_nope_head_dim + qk_rope_head_dim`).
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Heads whose output is *subtracted* — one per group of
    /// `grouped_ratio + 1`.
    #[serde(default)]
    pub num_noise_heads: usize,
    /// GDLA V2. V1 raises `NotImplementedError` upstream, so it is rejected.
    #[serde(default)]
    pub diff_v2: bool,
    /// Only `"gdla"` is registered upstream.
    #[serde(default = "d_attn_cls")]
    pub attention_cls: String,
    /// Read by the reference's `__init__` and then never used — see the module
    /// docs.
    #[serde(default)]
    pub headwise_attn_output_gate: bool,
    /// The gate GDLA actually applies: `σ(wq_b_gate(q_latent))` per output
    /// element.
    #[serde(default)]
    pub elementwise_attn_output_gate: bool,

    // ── GDLA low-rank projections ──
    /// Rank of the Q bottleneck (`wq_a` → RMSNorm → `wq_b`).
    #[serde(default)]
    pub q_lora_rank: usize,
    /// Rank of the KV bottleneck; `wkv_a` emits this *plus* `qk_rope_head_dim`
    /// for the one shared RoPE head.
    #[serde(default)]
    pub kv_lora_rank: usize,
    /// Width of the RoPE-carrying slice of Q/K.
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    /// V width per head — narrower than `head_dim` on Motif-3.
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    /// Pretraining context. YaRN only engages when
    /// `max_position_embeddings` exceeds it.
    #[serde(default = "d_orig_seq")]
    pub original_seq_len: usize,
    /// YaRN context-extension factor (top-level twin of
    /// [`RopeScaling::factor`]).
    #[serde(default = "d_one")]
    pub rope_factor: f32,
    /// YaRN attention scale. Folded into the softmax scale of *global* layers as
    /// `mscale²`, never into cos/sin.
    #[serde(default = "d_one")]
    pub mscale: f32,

    // ── MoE ──
    /// Routed experts per MoE layer (0 ⇒ the model is dense throughout).
    #[serde(default)]
    pub num_experts: usize,
    /// Experts each token is routed to.
    #[serde(default = "d_top_k")]
    pub experts_top_k: usize,
    /// Always-on expert alongside the routed ones (0 or 1).
    #[serde(default)]
    pub num_shared_experts: usize,
    /// MoE every Nth layer; 0 disables MoE entirely.
    #[serde(default)]
    pub interleave_moe_layer_step: usize,
    /// Width of a routed expert's FFN (defaults to `intermediate_size`).
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// Only `"sigmoid"` is built.
    #[serde(default = "d_score_func")]
    pub score_func: String,
    /// Normalize the top-k weights to sum to 1 before `route_scale`.
    #[serde(default)]
    pub route_norm: bool,
    /// Multiplier on the (normalized) routing weights.
    #[serde(default = "d_one")]
    pub route_scale: f32,
    /// Pre-weight expert *inputs* instead of outputs — not built.
    #[serde(default)]
    pub score_before_experts: bool,
    /// Leading layers that keep a dense FFN regardless of the MoE schedule.
    #[serde(default)]
    pub n_dense_first_layers: usize,
    /// Training-only knob whose presence implies a `moe.expert_bias` tensor —
    /// the router's selection bias.
    #[serde(default)]
    pub load_balance_coeff: Option<f32>,

    // ── MHC (manifold-constrained hyper-connections) ──
    /// Replace the single residual stream with `mhc_expansion_rate` of them.
    #[serde(default)]
    pub mhc_enabled: bool,
    /// Number of parallel residual streams (`E`).
    #[serde(default = "d_mhc_expansion")]
    pub mhc_expansion_rate: usize,
    /// Alternating row/column normalizations used to make `h_res` doubly
    /// stochastic.
    #[serde(default = "d_sinkhorn_iters")]
    pub mhc_sinkhorn_iters: usize,
    /// `h_post` is scaled by `1 + this`; Motif-3 leaves it at 0.
    #[serde(default)]
    pub mhc_h_post_alpha_end: f32,

    // ── PolyNorm ──
    /// Pass the polynomial weights through a sigmoid (upstream default).
    #[serde(default = "d_true")]
    pub polynorm_sigmoid_weight: bool,
    /// Final multiplier on every PolyNorm FFN output.
    #[serde(default = "d_one")]
    pub polynorm_output_scale: f32,
    /// Clamp on the per-expert PolyNorm bias — routed experts only.
    #[serde(default)]
    pub polynorm_bias_clamp: Option<f32>,
    /// Clamp on the FFN's `gate`/`up` (and, for routed experts, their product).
    #[serde(default)]
    pub hidden_clamp: Option<f32>,

    // ── misc ──
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// MTP heads in the checkpoint. Inference ignores them, as the reference
    /// does; see [`crate::drop_mtp_layers`].
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub eos_token_id: u32,
}

fn d_vocab() -> usize {
    151_936
}
fn d_hidden() -> usize {
    4096
}
fn d_inter() -> usize {
    22_016
}
fn d_layers() -> usize {
    32
}
fn d_heads() -> usize {
    32
}
fn d_act() -> String {
    "silu".into()
}
fn d_max_pos() -> usize {
    32_768
}
fn d_rms_eps() -> f32 {
    1e-6
}
fn d_rope_theta() -> f32 {
    1_000_000.0
}
fn d_swa_pattern() -> String {
    "interleave".into()
}
fn d_swa_period() -> usize {
    2
}
fn d_attn_cls() -> String {
    "basic".into()
}
fn d_orig_seq() -> usize {
    32_768
}
fn d_one() -> f32 {
    1.0
}
fn d_top_k() -> usize {
    2
}
fn d_score_func() -> String {
    "softmax".into()
}
fn d_mhc_expansion() -> usize {
    4
}
fn d_sinkhorn_iters() -> usize {
    20
}
fn d_true() -> bool {
    true
}

/// The RMS-norm epsilon `MHCLayer` hardcodes (`RMSNorm(E*D, eps=1e-6)`) — it is
/// *not* `config.rms_norm_eps`.
pub const MHC_NORM_EPS: f32 = 1e-6;
/// `PolyNormTorch` / `GroupedPolyNorm` epsilon, likewise hardcoded upstream.
pub const POLYNORM_EPS: f32 = 1e-6;
/// `MHCLayer.forward` clamps the pre/post gate logits to ±10 before the sigmoid.
pub const MHC_GATE_CLAMP: f32 = 10.0;
/// …and the Sinkhorn input to ±20 before `exp`.
pub const MHC_SINKHORN_CLAMP: f32 = 20.0;
/// Row/column sums are floored here before the division.
pub const MHC_SINKHORN_FLOOR: f32 = 1e-8;

impl MotifConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    /// Per-head Q/K width (`head_dim`, falling back to `hidden / heads`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    /// Width of the RoPE-carrying slice of Q/K.
    pub fn qk_rope_head_dim(&self) -> usize {
        self.qk_rope_head_dim.unwrap_or(self.head_dim() / 2)
    }

    /// Width of the position-free slice of Q/K.
    pub fn qk_nope_head_dim(&self) -> usize {
        self.head_dim() - self.qk_rope_head_dim()
    }

    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim.unwrap_or(self.head_dim())
    }

    /// Signal heads per noise head: `(H - noise) / noise`.
    pub fn grouped_ratio(&self) -> usize {
        (self.num_attention_heads - self.num_noise_heads) / self.num_noise_heads.max(1)
    }

    /// Heads whose output survives the differential subtraction.
    pub fn n_signal_heads(&self) -> usize {
        self.grouped_ratio() * self.num_noise_heads
    }

    /// `moe_intermediate_size`, defaulting to the dense `intermediate_size`.
    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// `MotifDecoderLayer.moe_enabled`.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        if self.interleave_moe_layer_step == 0 || self.num_experts == 0 {
            return false;
        }
        layer >= self.n_dense_first_layers
            && (layer + 1).is_multiple_of(self.interleave_moe_layer_step)
    }

    /// Which mask/RoPE flavour layer `i` runs under.
    ///
    /// The reference passes `sliding_window + 1` to the attention interface, and
    /// transformers turns a flash `sliding_window = W` into
    /// `window_size = (W - 1, 0)` — so a layer sees `sliding_window` keys of
    /// history plus itself. [`rlx_ir::op::MaskKind::SlidingWindow`] keeps
    /// `q_pos - k_pos <= w`, i.e. the same span for `w = sliding_window`.
    pub fn layer_attn(&self, layer: usize) -> LayerAttn {
        let Some(w) = self.sliding_window.filter(|_| self.use_sliding_window) else {
            return LayerAttn::Global;
        };
        match self.sliding_window_pattern.as_str() {
            "all" => LayerAttn::Sliding(w),
            "interleave" if !layer.is_multiple_of(self.sliding_window_period.max(1)) => {
                LayerAttn::Sliding(w)
            }
            _ => LayerAttn::Global,
        }
    }

    /// True when any layer is a sliding-window layer (⇒ the graph needs the
    /// second, un-interpolated RoPE table).
    pub fn has_sliding_layers(&self) -> bool {
        (0..self.num_hidden_layers).any(|i| matches!(self.layer_attn(i), LayerAttn::Sliding(_)))
    }

    /// `MotifGDLAttention.scaling`: `head_dim^-0.5`, times `mscale²` on global
    /// layers when the context was YaRN-extended.
    pub fn attn_score_scale(&self, layer: usize) -> f32 {
        let base = (self.head_dim() as f32).powf(-0.5);
        let global = matches!(self.layer_attn(layer), LayerAttn::Global);
        if global && self.max_position_embeddings > self.original_seq_len {
            let m = 0.1 * self.mscale * (self.rope_factor.ln()) + 1.0;
            base * m * m
        } else {
            base
        }
    }

    /// `(cos, sin)` of shape `[seq, qk_rope_head_dim / 2]` for the **global**
    /// layers: YaRN-interpolated frequencies, no `attention_scaling` (Motif puts
    /// the YaRN `mscale` on the softmax scale instead of the cos/sin table).
    pub fn rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        tables(seq, &self.global_inv_freq())
    }

    /// `(cos, sin)` for the **sliding-window** layers: plain `swa_rope_theta`
    /// RoPE with no YaRN interpolation (`swa_config.rope_scaling = None`).
    pub fn swa_rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let dim = self.qk_rope_head_dim();
        let theta = self.swa_rope_theta.unwrap_or(self.rope_theta) as f64;
        let inv: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64))
            .collect();
        tables(seq, &inv)
    }

    /// The global-layer `inv_freq` table (`[qk_rope_head_dim / 2]`).
    fn global_inv_freq(&self) -> Vec<f64> {
        let dim = self.qk_rope_head_dim();
        let rs = self.rope_scaling.clone().unwrap_or_default();
        let yarn = rs.rope_type.as_deref() == Some("yarn");
        let theta = if yarn {
            rs.rope_theta.unwrap_or(self.rope_theta) as f64
        } else {
            self.rope_theta as f64
        };
        let base: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64))
            .collect();
        if !yarn {
            return base;
        }
        let factor = rs.factor.unwrap_or(self.rope_factor) as f64;
        let orig = rs
            .original_max_position_embeddings
            .unwrap_or(self.original_seq_len);
        if self.max_position_embeddings <= orig {
            return base;
        }
        let beta_fast = rs.beta_fast.unwrap_or(32.0) as f64;
        let beta_slow = rs.beta_slow.unwrap_or(1.0) as f64;
        yarn_interpolate(base, dim, theta, orig, factor, beta_fast, beta_slow)
    }

    /// Reject configurations this builder cannot honour, with an actionable
    /// message. Deliberately silent about `num_nextn_predict_layers`: the
    /// checkpoint ships an MTP block, `modeling_motif.py` never builds it, and
    /// neither do we.
    pub fn validate(&self) -> Result<()> {
        if self.attention_cls != "gdla" {
            anyhow::bail!(
                "attention_cls={:?} — modeling_motif.py registers only 'gdla' \
                 (MotifDecoderLayer._ATTN_CLS); refusing to guess an older Motif layer",
                self.attention_cls
            );
        }
        if !self.diff_v2 {
            anyhow::bail!(
                "diff_v2=false — GDLA V1 raises NotImplementedError upstream, so there \
                 is no reference to port"
            );
        }
        if self.hidden_act != "poly_norm" {
            anyhow::bail!(
                "hidden_act={:?} — only Motif's trainable PolyNorm activation is built \
                 (the checkpoint carries act_fn.weight/act_fn.bias per FFN)",
                self.hidden_act
            );
        }
        if self.num_noise_heads == 0 {
            anyhow::bail!("num_noise_heads=0 — GDLA needs at least one noise head per group");
        }
        if !(self.num_attention_heads - self.num_noise_heads).is_multiple_of(self.num_noise_heads) {
            anyhow::bail!(
                "num_attention_heads ({}) - num_noise_heads ({}) must be divisible by \
                 num_noise_heads: the head split is (grouped_ratio + 1) heads per group",
                self.num_attention_heads,
                self.num_noise_heads
            );
        }
        if self.num_key_value_heads != self.num_noise_heads {
            anyhow::bail!(
                "num_key_value_heads ({}) != num_noise_heads ({}) — the differential \
                 regroup assumes one noise head per GQA group; a checkpoint where these \
                 differ would pair signal heads with the wrong noise head",
                self.num_key_value_heads,
                self.num_noise_heads
            );
        }
        if self.num_attention_heads != self.num_noise_heads * (self.grouped_ratio() + 1) {
            anyhow::bail!(
                "num_attention_heads ({}) != num_noise_heads * (grouped_ratio + 1)",
                self.num_attention_heads
            );
        }
        if self.q_lora_rank == 0 || self.kv_lora_rank == 0 {
            anyhow::bail!(
                "GDLA is low-rank on both sides: q_lora_rank ({}) and kv_lora_rank ({}) \
                 must both be > 0",
                self.q_lora_rank,
                self.kv_lora_rank
            );
        }
        if !self.elementwise_attn_output_gate {
            anyhow::bail!(
                "elementwise_attn_output_gate=false — the GDLA forward only ever applies \
                 the element-wise gate (headwise_attn_output_gate is dead upstream), and \
                 without it there is no wq_b_gate in the checkpoint to read"
            );
        }
        if self.num_experts > 0 {
            if self.score_func != "sigmoid" {
                anyhow::bail!(
                    "score_func={:?} — the shared group-limited gate kernel scores with \
                     sigmoid; a softmax router needs a different op",
                    self.score_func
                );
            }
            if !self.route_norm {
                anyhow::bail!(
                    "route_norm=false — the shared gate kernel always normalises the \
                     top-k weights before route_scale; refusing to silently normalise"
                );
            }
            if self.route_norm && self.experts_top_k == 1 {
                anyhow::bail!(
                    "experts_top_k=1 with route_norm=true — the shared gate kernel keeps \
                     the raw score at k=1 (DeepSeek's convention: normalising a single \
                     weight is a no-op there), but Motif divides by it, giving exactly \
                     1.0·route_scale. The two disagree, so this config needs its own gate"
                );
            }
            if self.score_before_experts {
                anyhow::bail!(
                    "score_before_experts=true — this builder weights expert *outputs* \
                     (MoE.forward's default path), not inputs"
                );
            }
            if self.num_shared_experts > 1 {
                anyhow::bail!(
                    "num_shared_experts={} — MoE.shared_experts is a single MotifMLP \
                     upstream regardless of the count; >1 has no checkpoint layout",
                    self.num_shared_experts
                );
            }
        }
        if self.mhc_enabled && self.mhc_expansion_rate < 1 {
            anyhow::bail!("mhc_expansion_rate must be >= 1 when mhc_enabled");
        }
        if self.tie_word_embeddings {
            anyhow::bail!(
                "tie_word_embeddings=true — Motif-3 ships an untied lm_head; the tied \
                 path is untested against this checkpoint"
            );
        }
        Ok(())
    }
}

/// YaRN frequency interpolation, transcribed from `_compute_yarn_inv_freq`.
fn yarn_interpolate(
    freqs: Vec<f64>,
    dim: usize,
    theta: f64,
    original_seq_len: usize,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let correction_dim = |rot: f64| {
        dim as f64 * ((original_seq_len as f64) / (rot * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * theta.ln())
    };
    let lo = correction_dim(beta_fast).floor().max(0.0);
    let mut hi = correction_dim(beta_slow).ceil().min(dim as f64 - 1.0);
    if (hi - lo).abs() < f64::EPSILON {
        hi += 0.001;
    }
    freqs
        .into_iter()
        .enumerate()
        .map(|(j, f)| {
            let ramp = (((j as f64) - lo) / (hi - lo)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            f / rope_factor * (1.0 - smooth) + f * smooth
        })
        .collect()
}

/// `[seq, dim/2]` cos/sin tables for [`rlx_ir::RopeStyle::NeoX`] — Motif's
/// `rotate_half` is the half-split rotation, and `emb = cat(freqs, freqs)`
/// means the second half repeats the first.
fn tables(seq: usize, inv_freq: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(seq * inv_freq.len());
    let mut sin = Vec::with_capacity(seq * inv_freq.len());
    for pos in 0..seq {
        for &f in inv_freq {
            let ang = pos as f64 * f;
            cos.push(ang.cos() as f32);
            sin.push(ang.sin() as f32);
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published Motif-3 `config.json`, verbatim.
    const MOTIF3: &str = include_str!("../fixtures/motif-3-config.json");

    #[test]
    fn parses_published_config() {
        let cfg = MotifConfig::from_json_str(MOTIF3).expect("parse");
        cfg.validate().expect("valid");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 53);
        assert_eq!(cfg.num_attention_heads, 80);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert_eq!(cfg.num_noise_heads, 16);
        assert_eq!(cfg.grouped_ratio(), 4);
        assert_eq!(cfg.n_signal_heads(), 64);
        assert_eq!(cfg.head_dim(), 192);
        assert_eq!(cfg.qk_nope_head_dim(), 128);
        assert_eq!(cfg.qk_rope_head_dim(), 64);
        assert_eq!(cfg.v_head_dim(), 128);
        assert_eq!(cfg.q_lora_rank, 1024);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.num_experts, 384);
        assert_eq!(cfg.experts_top_k, 8);
        assert_eq!(cfg.moe_intermediate_size(), 1280);
        assert!(cfg.mhc_enabled && cfg.mhc_expansion_rate == 4);
        assert_eq!(cfg.mhc_sinkhorn_iters, 20);
        assert_eq!(cfg.polynorm_output_scale, 0.5);
        assert_eq!(cfg.polynorm_bias_clamp, Some(0.5));
        assert_eq!(cfg.hidden_clamp, Some(1e6));
        assert_eq!(cfg.num_nextn_predict_layers, 1, "MTP block is present…");
    }

    /// `n_dense_first_layers = 2`, `interleave_moe_layer_step = 1` ⇒ dense MLP on
    /// layers 0 and 1 only. That is exactly what the checkpoint index carries:
    /// 2 × `mlp.*`, 51 × `moe.*`.
    #[test]
    fn moe_schedule_matches_checkpoint() {
        let cfg = MotifConfig::from_json_str(MOTIF3).unwrap();
        let dense: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| !cfg.is_moe_layer(i))
            .collect();
        assert_eq!(dense, vec![0, 1]);
        assert_eq!(
            (0..cfg.num_hidden_layers)
                .filter(|&i| cfg.is_moe_layer(i))
                .count(),
            51
        );
    }

    /// `sliding_window_period = 4` ⇒ global attention every 4th layer starting
    /// at 0; everything else is a 128-key sliding window.
    #[test]
    fn sliding_window_interleave() {
        let cfg = MotifConfig::from_json_str(MOTIF3).unwrap();
        assert_eq!(cfg.layer_attn(0), LayerAttn::Global);
        assert_eq!(cfg.layer_attn(1), LayerAttn::Sliding(128));
        assert_eq!(cfg.layer_attn(3), LayerAttn::Sliding(128));
        assert_eq!(cfg.layer_attn(4), LayerAttn::Global);
        assert_eq!(
            (0..cfg.num_hidden_layers)
                .filter(|&i| cfg.layer_attn(i) == LayerAttn::Global)
                .count(),
            14
        );
    }

    /// Global layers carry the YaRN `mscale²` on the softmax scale; SWA layers
    /// get the plain `head_dim^-0.5`.
    #[test]
    fn mscale_only_on_global_layers() {
        let cfg = MotifConfig::from_json_str(MOTIF3).unwrap();
        let plain = (192f32).powf(-0.5);
        let m = 0.1 * 1.0 * 64f32.ln() + 1.0;
        assert!((cfg.attn_score_scale(0) - plain * m * m).abs() < 1e-7);
        assert!((cfg.attn_score_scale(1) - plain).abs() < 1e-7);
        assert!(m > 1.4 && m < 1.42, "mscale sanity: {m}");
    }

    /// YaRN stretches the low-frequency end (large `j`) by `1/factor` and leaves
    /// the high-frequency end alone; the SWA table is the un-stretched original.
    #[test]
    fn yarn_only_touches_low_frequencies() {
        let cfg = MotifConfig::from_json_str(MOTIF3).unwrap();
        let yarn = cfg.global_inv_freq();
        let half = cfg.qk_rope_head_dim() / 2;
        assert_eq!(yarn.len(), half);
        let plain: Vec<f64> = (0..cfg.qk_rope_head_dim())
            .step_by(2)
            .map(|i| 1.0 / 10000f64.powf(i as f64 / cfg.qk_rope_head_dim() as f64))
            .collect();
        // Correction range for (dim=64, theta=1e4, orig=4096) is [10, 23].
        assert!((yarn[0] - plain[0]).abs() < 1e-12, "j=0 untouched");
        assert!(
            (yarn[half - 1] - plain[half - 1] / 64.0).abs() < 1e-15,
            "j=31 fully interpolated"
        );
        assert!(yarn[15] < plain[15] && yarn[15] > plain[15] / 64.0, "ramp");

        let (cos, sin) = cfg.swa_rope_tables(3);
        assert_eq!(cos.len(), 3 * half);
        assert!(cos[..half].iter().all(|c| (c - 1.0).abs() < 1e-6));
        assert!(sin[..half].iter().all(|s| s.abs() < 1e-6));
        assert!((cos[half] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn rejects_configs_with_no_reference() {
        let base = MotifConfig::from_json_str(MOTIF3).unwrap();
        let mut c = base.clone();
        c.attention_cls = "basic".into();
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("gdla"));
        let mut c = base.clone();
        c.route_norm = false;
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("route_norm"));
        let mut c = base.clone();
        c.num_key_value_heads = 8;
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("num_noise_heads"));
        let mut c = base;
        c.hidden_act = "silu".into();
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("PolyNorm"));
    }
}
