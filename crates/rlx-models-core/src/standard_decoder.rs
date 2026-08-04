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

//! Config-driven **standard causal decoder** builder — the shared graph
//! topology behind the Llama / Qwen2 / Qwen3 / Mistral / SmolLM / Yi /
//! InternLM2 family, so a single builder + a [`DecoderSpec`] covers most
//! dense text LLMs on huggingface.co/mlx-community without a crate per
//! model.
//!
//! The block is: `embed → N×[RMSNorm → q/k/v proj (+opt bias) → opt
//! QK-norm → RoPE → GQA repeat → attention → o_proj → residual →
//! RMSNorm → SwiGLU] → final RMSNorm → (tied|untied) lm_head`.
//!
//! Weights stay packed in the arena: MLX-affine linears lower to a
//! 4-input `Op::DequantMatMul { MlxAffine }`, GGUF K-quant to a 2-input
//! `Op::DequantMatMul`, everything else to dense F32 `Op::MatMul`. This
//! is a verbatim generalization of `rlx-qwen3`'s
//! `build_qwen3_graph_sized_packed`, which now delegates here.
//!
//! Out of scope for this builder (route to a dedicated crate): MoE
//! routing, non-SwiGLU MLPs (Gemma GeGLU), sandwich / `(1+γ)` norms
//! (Gemma), partial-rotary + fused-QKV (Phi), state-space / linear
//! attention (Mamba, Qwen3.5 DeltaNet). [`DecoderSpec::guard_supported`]
//! rejects these with an actionable error instead of silently
//! mis-building.

use anyhow::{Result, anyhow};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::quant::QuantScheme;
use rlx_ir::shape;
use rlx_ir::*;
use std::collections::HashMap;
use std::path::Path;

use crate::weight_loader::WeightLoader;

/// RoPE frequency scaling. `None` = vanilla `1/θ^(2i/d)` (Qwen/Mistral/
/// SmolLM). `Llama3` = the piecewise low/high-frequency rescale (Llama 3.x).
/// `Yarn` = NTK-by-parts interpolation + attention `mscale` (Qwen2.5 long-ctx,
/// DeepSeek). `Linear` = position scaling. Unknown/unhandled `rope_type`s are
/// rejected by [`classify_config`] so they never silently mis-scale.
#[derive(Debug, Clone)]
pub enum RopeScaling {
    None,
    /// Linear position scaling (`rope_type == "linear"`): every inv_freq / factor.
    Linear {
        factor: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: f64,
    },
    /// YaRN (`rope_type == "yarn"`): NTK-by-parts blend of extrapolated and
    /// `/factor`-interpolated frequencies over `[low, high]` dims, plus a scalar
    /// attention factor applied to cos/sin ([`Self::mscale`]).
    Yarn {
        factor: f64,
        original_max_position_embeddings: f64,
        beta_fast: f64,
        beta_slow: f64,
        /// Explicit `attention_factor` from config, else derived `0.1·ln(factor)+1`.
        attention_factor: Option<f64>,
    },
    /// Phi-3/3.5 **LongRoPE**: per-dimension frequency rescale by `short_factor`
    /// (positions ≤ `original_max_position_embeddings`) or `long_factor`, plus a
    /// global attention `mscale`. `inv_freq[i] = base(i) / short_factor[i]` for
    /// prefill (seq ≤ orig covers the validation + typical prompt lengths).
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position_embeddings: f64,
        mscale: f64,
    },
}

impl RopeScaling {
    /// Inverse frequency for rotary dimension `i` (`0..head_dim/2`) given the
    /// base `theta` and `head_dim` — mirrors HF `_compute_*_parameters`.
    fn inv_freq(&self, i: usize, head_dim: usize, theta: f64) -> f64 {
        let d = head_dim as f64;
        let base = theta.powf(-(2.0 * i as f64) / d); // 1/θ^(2i/d)
        match self {
            RopeScaling::None => base,
            RopeScaling::Linear { factor } => base / factor,
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let low_wavelen = original_max_position_embeddings / low_freq_factor;
                let high_wavelen = original_max_position_embeddings / high_freq_factor;
                let wavelen = 2.0 * std::f64::consts::PI / base;
                if wavelen < high_wavelen {
                    base
                } else if wavelen > low_wavelen {
                    base / factor
                } else {
                    let smooth = (original_max_position_embeddings / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * base / factor + smooth * base
                }
            }
            RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                ..
            } => {
                let extrap = base; // original
                let interp = base / factor; // interpolated
                let find_dim = |rot: f64| -> f64 {
                    (d * (original_max_position_embeddings / (rot * 2.0 * std::f64::consts::PI))
                        .ln())
                        / (2.0 * theta.ln())
                };
                let low = find_dim(*beta_fast).floor().max(0.0);
                let high = find_dim(*beta_slow).ceil().min(d - 1.0);
                let denom = if (high - low).abs() < 1e-9 {
                    0.001
                } else {
                    high - low
                };
                let ramp = ((i as f64 - low) / denom).clamp(0.0, 1.0);
                // extrapolation mask = 1 - ramp; blend interp (low dims) ↔ extrap (high dims).
                let extrap_factor = 1.0 - ramp;
                interp * (1.0 - extrap_factor) + extrap * extrap_factor
            }
            // Phi-3/3.5 LongRoPE: per-dim rescale by short_factor (prefill seq ≤ orig).
            RopeScaling::LongRope { short_factor, .. } => {
                base / short_factor.get(i).copied().unwrap_or(1.0)
            }
        }
    }

    /// Scalar multiplied into the cos/sin tables (YaRN's `attention_factor`);
    /// 1.0 for every other scheme.
    fn mscale(&self) -> f64 {
        match self {
            RopeScaling::Yarn {
                factor,
                attention_factor,
                ..
            } => attention_factor.unwrap_or(0.1 * factor.ln() + 1.0),
            RopeScaling::LongRope { mscale, .. } => *mscale,
            _ => 1.0,
        }
    }
}

/// Normalized config for a standard causal decoder — the variation axes
/// the packed builder actually reads. Produced from a HuggingFace
/// `config.json` ([`DecoderSpec::from_config_json`]) or from a model
/// crate's own config (e.g. `Qwen3Config`).
#[derive(Debug, Clone)]
pub struct DecoderSpec {
    /// `model_type` / `architectures[0]` — used to name the graph and in
    /// diagnostics only; topology is driven by the fields below.
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Fraction of `head_dim` that receives RoPE (`partial_rotary_factor`; 1.0 =
    /// full). Phi-4-mini uses 0.75 (rotate first 96 of 128, pass the rest). The
    /// rotated width is `n_rot = (partial_rotary_factor * head_dim)` (even).
    pub partial_rotary_factor: f64,
    /// RoPE frequency scaling (Llama 3.x piecewise, linear, or none).
    pub rope_scaling: RopeScaling,
    /// Activation on the SwiGLU gate. Only `silu`/`swish` are handled by
    /// this builder; anything else is rejected by [`Self::guard_supported`].
    pub hidden_act: String,
    /// Explicit Q/K/V projection bias (Qwen2/2.5). Qwen3/Llama/Mistral: none.
    pub attention_bias: bool,
    /// Phi-3 packs Q/K/V into one `self_attn.qkv_proj` tensor (split by output
    /// rows into `[q_dim, kv_dim, kv_dim]`); most models use separate projs.
    pub fused_qkv: bool,
    /// Phi-3 packs gate+up into one `mlp.gate_up_proj` (split `[inter, inter]`).
    pub fused_gate_up: bool,
    /// Per-head RMSNorm on Q/K before RoPE (Qwen3). Llama/Mistral/Qwen2: off.
    pub qk_norm: bool,
    /// QK-norm gain spans the whole `[heads*head_dim]` projection (OLMoE)
    /// rather than a per-head `[head_dim]` slice (Qwen3). Only read when
    /// `qk_norm` is set.
    pub qk_norm_full: bool,
    pub tie_word_embeddings: bool,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub max_window_layers: usize,
    /// Routed-expert count (>0 → MoE FFN via grouped MLX dequant).
    pub num_experts: usize,
    /// Experts activated per token (top-k); 0 for dense.
    pub num_experts_used: usize,
    /// Renormalize the top-k router weights to sum to 1 (`norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// Per-expert FFN width (`moe_intermediate_size`, Qwen3-MoE/Qwen2-MoE).
    /// `0` → fall back to `intermediate_size` (OLMoE, where they're equal).
    pub moe_intermediate_size: usize,
    /// Always-on shared expert added to the routed sum, gated by
    /// `sigmoid(shared_expert_gate(x))` (Qwen2-MoE). `0` = none.
    pub shared_expert_intermediate_size: usize,

    // ── Gemma-family axes (defaults below = Llama / no-op) ──
    /// Added to every RMSNorm gain: `0.0` (Llama) or `1.0` (Gemma `(1+γ)`).
    pub norm_gain_offset: f32,
    /// Gemma "sandwich" norms: `post_attention` + `pre/post_feedforward`
    /// (4 norms/layer) vs Llama's 2. Changes the residual/norm placement.
    pub sandwich_norms: bool,
    /// Multiplier applied to the embedding output (Gemma uses `√hidden`);
    /// `1.0` = none.
    pub embed_scale: f32,
    /// SwiGLU vs GeGLU gate activation (`"silu"`/`"swish"` → SiLU, `"gelu*"`
    /// → tanh-approx GeLU). Mirrors `hidden_act`; kept explicit for the builder.
    pub gelu_gate: bool,
    /// Gemma3 dual-θ RoPE: `Some((local_theta, pattern))` uses the local θ on
    /// layers where `(i+1) % pattern != 0` and the global `rope_theta` on the
    /// rest; `None` = single-θ RoPE for every layer.
    pub rope_dual: Option<(f64, usize)>,
    /// Explicit attention score scale (`query_pre_attn_scalar**-0.5`, Gemma);
    /// `None` = default `1/√head_dim`.
    pub attn_score_scale: Option<f32>,
    /// Softcap on attention logits (`attn_logit_softcapping`, Gemma2):
    /// `s·tanh(scores/s)`; `None` = off.
    pub attn_logit_softcap: Option<f32>,
    /// Softcap on the final logits (`final_logit_softcapping`, Gemma2):
    /// `s·tanh(logits/s)`; `None` = off.
    pub final_logit_softcap: Option<f32>,
    /// Granite `residual_multiplier`: each residual add is `x + m·sublayer(x)`;
    /// `1.0` = plain residual.
    pub residual_multiplier: f32,
    /// Granite `logits_scaling`: final logits are divided by this; `1.0` = none.
    /// (Argmax/cosine-invariant, but affects softmax temperature.)
    pub logits_scaling: f32,
}

impl DecoderSpec {
    /// Repetition factor for GQA: how many Q heads share each KV head.
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads.max(1)
    }
    /// Q projection output width (`num_attention_heads * head_dim`).
    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    /// K/V projection output width (`num_key_value_heads * head_dim`).
    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Attention mask from the sliding-window config: a window when the
    /// model enables one (Mistral-style), else full causal.
    pub fn attn_mask_kind(&self) -> MaskKind {
        match self.sliding_window {
            Some(w) if self.use_sliding_window && w > 0 => MaskKind::SlidingWindow(w),
            _ => MaskKind::Causal,
        }
    }

    /// Reject topologies this dense-decoder builder does not model, with
    /// an actionable pointer, instead of silently mis-building.
    pub fn guard_supported(&self) -> Result<()> {
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads.max(1))
        {
            return Err(anyhow!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads
            ));
        }
        if self.num_experts > 0 && self.num_experts_used == 0 {
            return Err(anyhow!(
                "standard_decoder: MoE with {} experts but num_experts_used=0 \
                 (missing num_experts_per_tok)",
                self.num_experts
            ));
        }
        let act = self.hidden_act.to_ascii_lowercase();
        // SwiGLU (silu/swish) or GeGLU (gelu*, Gemma) are both handled.
        if !matches!(act.as_str(), "silu" | "swish") && !act.starts_with("gelu") {
            return Err(anyhow!(
                "standard_decoder: activation `{}` not supported (SwiGLU or GeGLU only).",
                self.hidden_act
            ));
        }
        Ok(())
    }
}

/// Reinterpret a little-endian f32 byte buffer as `Vec<f32>` (MLX affine
/// scales / biases arrive from rlx-mlx-io as raw bytes).
fn f32_from_le_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load a weight by key and register it as an F32 `Param` node. When
/// `transpose` is set, the safetensors `[out, in]` layout is swapped to
/// rlx's `[in, out]` matmul convention.
fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    transpose: bool,
) -> Result<NodeId> {
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = g.param(key, ir_shape);
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Register a zero-valued param (RMSNorm `beta`, which these models lack
/// but the IR op signature requires).
fn synth_zero(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    let id = g.param(name, Shape::new(&[len], DType::F32));
    params.insert(name.to_string(), vec![0f32; len]);
    id
}

/// Register a constant param of the given shape/data.
fn synth_const(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> NodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data);
    id
}

/// Load an RMSNorm gain, adding `offset` (Gemma bakes `(1+γ)`; `offset=1.0`).
fn load_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    offset: f32,
) -> Result<NodeId> {
    let (mut data, shape) = weights.take(key)?;
    if offset != 0.0 {
        for v in data.iter_mut() {
            *v += offset;
        }
    }
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Per-head RMSNorm via reshape to `[B*S*heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
fn per_head_rms(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let flat = (batch * seq * heads) as i64;
    let dh = head_dim as i64;
    let r = g.reshape_(x, vec![flat, dh]);
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}

/// Apply RoPE to the **last `rd` dims** of each head (GPT-J interleaved), leaving
/// the leading `hd-rd` "nope" dims untouched. `x` is `[rows, n_heads*hd]`. Mirrors
/// the reference `apply_rotary_emb(x[..., -rd:])`; `sin` chooses forward/inverse.
fn rope_tail(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    rows: usize,
    nh: usize,
    hd: usize,
    rd: usize,
) -> NodeId {
    if rd == 0 {
        return x;
    }
    if rd >= hd {
        return g.rope_n_styled(x, cos, sin, hd, hd, RopeStyle::GptJ);
    }
    // #4: fused partial GptJ tail-rope (one Op::Custom instead of narrow×2 + reshape×3
    // + concat around the rope op).
    if crate::dsv4_opt::opt_rope() {
        return crate::dsv4_opt::emit_rope_tail(g, x, cos, sin, rows, nh, hd, rd);
    }
    let x3 = g.reshape_(x, vec![rows as i64, nh as i64, hd as i64]);
    let nope = g.narrow_(x3, 2, 0, hd - rd); // [rows, nh, hd-rd]
    let tail = g.narrow_(x3, 2, hd - rd, rd); // [rows, nh, rd]
    let tail_flat = g.reshape_(tail, vec![rows as i64, (nh * rd) as i64]);
    let roped = g.rope_n_styled(tail_flat, cos, sin, rd, rd, RopeStyle::GptJ);
    let roped3 = g.reshape_(roped, vec![rows as i64, nh as i64, rd as i64]);
    let cat = g.concat_(vec![nope, roped3], 2); // [rows, nh, hd]
    g.reshape_(cat, vec![rows as i64, (nh * hd) as i64])
}

/// GQA repeat: widen `[B, S, num_kv_heads*head_dim]` by emitting each KV
/// head `group` times (narrow + concat).
fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn gather_last_token(
    g: &mut Graph,
    hidden: NodeId,
    batch: usize,
    last_token_idx: NodeId,
) -> NodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(hidden, idx_2d, 1)
}

/// Result of loading a projection weight. `scale`/`bias` are `Some` only
/// for MLX affine packs (separate per-group tensors); GGUF K-quant
/// carries its scales inside the single packed blob.
struct Proj {
    w: NodeId,
    scheme: Option<QuantScheme>,
    scale: Option<NodeId>,
    bias: Option<NodeId>,
}

fn load_proj(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<Proj> {
    // Structure-only: size the U8 codes param from metadata and DEFER the large
    // `w_q` codes (empty in `packed`; a worker re-loads its shard). Small
    // scales/biases are kept. Only a structure-only loader returns `Some` here;
    // normal loaders return `None` and fall through to the full load below.
    if let Some(meta) = weights.packed_mlx_meta(key)? {
        let scheme = meta.scheme;
        let n = meta.out_shape.first().copied().unwrap_or(0);
        let n_groups = meta.n_groups.max(1);
        let w = g.param(key, Shape::new(&[meta.w_q_len], DType::U8));
        packed.insert(key.to_string(), (Vec::new(), scheme, meta.out_shape));
        // MXFP4 (2 codes/byte) and MXFP8 (1 code/byte) share the packed path:
        // raw codes + per-group E8M0 u8 scales, no zero-point; the op dequants.
        if matches!(
            scheme,
            QuantScheme::MlxMxfp4 { .. } | QuantScheme::MlxMxfp8 { .. }
        ) {
            let s_name = format!("{key}.scales");
            let scale = g.param(&s_name, Shape::new(&[n, n_groups], DType::U8));
            packed.insert(s_name, (meta.scales, scheme, vec![n, n_groups]));
            let b_name = format!("{key}.biases");
            let bias = g.param(&b_name, Shape::new(&[n, n_groups], DType::U8));
            let bb = if meta.biases.is_empty() {
                vec![0u8; n * n_groups]
            } else {
                meta.biases
            };
            packed.insert(b_name, (bb, scheme, vec![n, n_groups]));
            return Ok(Proj {
                w,
                scheme: Some(scheme),
                scale: Some(scale),
                bias: Some(bias),
            });
        }
        if !matches!(scheme, QuantScheme::MlxAffine { .. }) {
            return Err(anyhow!(
                "structure-only: only MLX affine/mxfp4 wired for {key}; got {scheme}"
            ));
        }
        let scales_f32 = f32_from_le_bytes(&meta.scales);
        let biases_f32 = if meta.biases.is_empty() {
            vec![0f32; n * n_groups]
        } else {
            f32_from_le_bytes(&meta.biases)
        };
        let s_name = format!("{key}.scales");
        let scale = g.param(&s_name, Shape::new(&[n, n_groups], DType::F32));
        params.insert(s_name, scales_f32);
        let b_name = format!("{key}.biases");
        let bias = g.param(&b_name, Shape::new(&[n, n_groups], DType::F32));
        params.insert(b_name, biases_f32);
        return Ok(Proj {
            w,
            scheme: Some(scheme),
            scale: Some(scale),
            bias: Some(bias),
        });
    }
    // MLX affine: separate packed codes + per-group scales + biases.
    if let Some(p) = weights.take_packed_mlx(key)? {
        let scheme = p.scheme;
        // MXFP4 dense linear (e.g. gpt-oss router): the CPU `DequantMatMul`
        // mxfp4 path reads the E8M0 group scales as RAW u8 (no zero-point), so
        // bind codes + u8 scales as packed params + a dummy u8 zp to keep the
        // 4-input op shape. (Affine falls through below with f32 scales/biases.)
        if matches!(
            scheme,
            QuantScheme::MlxMxfp4 { .. } | QuantScheme::MlxMxfp8 { .. }
        ) {
            let n = p.out_shape.first().copied().unwrap_or(0);
            let n_groups = p.n_groups().max(1);
            let w = g.param(key, Shape::new(&[p.w_q.len()], DType::U8));
            packed.insert(key.to_string(), (p.w_q, scheme, p.out_shape));
            let s_name = format!("{key}.scales");
            let scale = g.param(&s_name, Shape::new(&[n, n_groups], DType::U8));
            packed.insert(s_name.clone(), (p.scales, scheme, vec![n, n_groups]));
            let b_name = format!("{key}.biases");
            let bias = g.param(&b_name, Shape::new(&[n, n_groups], DType::U8));
            packed.insert(b_name, (vec![0u8; n * n_groups], scheme, vec![n, n_groups]));
            return Ok(Proj {
                w,
                scheme: Some(scheme),
                scale: Some(scale),
                bias: Some(bias),
            });
        }
        if !matches!(scheme, QuantScheme::MlxAffine { .. }) {
            return Err(anyhow!(
                "standard_decoder: only MLX affine/mxfp4 is wired for {key}; got {scheme}"
            ));
        }
        let n = p.out_shape.first().copied().unwrap_or(0);
        let n_groups = p.n_groups().max(1);
        let scales_f32 = f32_from_le_bytes(&p.scales);
        let biases_f32 = if p.biases.is_empty() {
            vec![0f32; n * n_groups]
        } else {
            f32_from_le_bytes(&p.biases)
        };
        let w = g.param(key, Shape::new(&[p.w_q.len()], DType::U8));
        packed.insert(key.to_string(), (p.w_q, scheme, p.out_shape));
        let s_name = format!("{key}.scales");
        let scale = g.param(&s_name, Shape::new(&[n, n_groups], DType::F32));
        params.insert(s_name, scales_f32);
        let b_name = format!("{key}.biases");
        let bias = g.param(&b_name, Shape::new(&[n, n_groups], DType::F32));
        params.insert(b_name, biases_f32);
        return Ok(Proj {
            w,
            scheme: Some(scheme),
            scale: Some(scale),
            bias: Some(bias),
        });
    }
    // GGUF K-quant: single packed blob (scales embedded). Zero-copy — reserve
    // the U8 slot by byte length and store an EMPTY bytes marker; the caller
    // uploads the bytes straight from the loader's mmap at attach (see the
    // `bytes.is_empty()` borrow in the qwen3 high-level runner). Loaders without
    // a zero-copy path (`packed_meta` = None, e.g. MLX) fall back to the owned
    // `take_packed` below.
    if let Some((scheme, shape)) = weights.packed_meta(key) {
        let nbytes = weights
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("packed weight {key}: metadata present but bytes unavailable"))?
            .len();
        let w = g.param(key, Shape::new(&[nbytes], DType::U8));
        packed.insert(key.to_string(), (Vec::new(), scheme, shape));
        return Ok(Proj {
            w,
            scheme: Some(scheme),
            scale: None,
            bias: None,
        });
    }
    if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
        let w = g.param(key, Shape::new(&[bytes.len()], DType::U8));
        packed.insert(key.to_string(), (bytes, scheme, shape));
        return Ok(Proj {
            w,
            scheme: Some(scheme),
            scale: None,
            bias: None,
        });
    }
    // Dense F32 fallback (transpose to [in, out]).
    let w = load_p(g, params, weights, key, /*transpose*/ true)?;
    Ok(Proj {
        w,
        scheme: None,
        scale: None,
        bias: None,
    })
}

/// Apply RoPE to a `[B, S, heads*head_dim]` projection. `flat=true` runs
/// the CPU-correct rank-3 rope directly (correct on every backend);
/// `flat=false` uses the legacy MLX/Metal BHSD packing.
#[allow(clippy::too_many_arguments)]
fn rope_heads(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    dh: usize,
    flat: bool,
) -> NodeId {
    if flat {
        return g.rope(x, cos, sin, dh);
    }
    let x4 = g.reshape_(x, vec![batch as i64, seq as i64, heads as i64, dh as i64]);
    let bhsd = g.transpose_(x4, vec![0, 2, 1, 3]);
    let r = g.rope(bhsd, cos, sin, dh);
    let bshd = g.transpose_(r, vec![0, 2, 1, 3]);
    g.reshape_(bshd, vec![batch as i64, seq as i64, (heads * dh) as i64])
}

/// Emit DequantMatMul (2-input GGUF / 4-input MLX affine) or plain MatMul.
fn emit_proj(g: &mut Graph, input: NodeId, p: &Proj, out_shape: Shape) -> NodeId {
    match p.scheme {
        Some(s) => {
            let mut inputs = vec![input, p.w];
            if let (Some(scale), Some(bias)) = (p.scale, p.bias) {
                inputs.push(scale);
                inputs.push(bias);
            }
            g.add_node(Op::DequantMatMul { scheme: s }, inputs, out_shape)
        }
        None => g.mm(input, p.w),
    }
}

/// Split one MLX-packed fused projection (Phi-3 `qkv_proj` / `gate_up_proj`)
/// into its sub-projections by **output rows**. MLX affine packs each output
/// row independently (group over the input dim), so a row range is a
/// contiguous byte slice of the codes and a contiguous slice of the
/// `[rows, n_groups]` scales/biases — a clean host-side split. `splits` gives
/// `(sub_name, out_rows)` in packed order; returns one [`Proj`] each.
fn load_fused_split_proj(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    key: &str,
    splits: &[(&str, usize)],
) -> Result<Vec<Proj>> {
    let p = weights
        .take_packed_mlx(key)?
        .ok_or_else(|| anyhow!("fused proj not MLX-packed: {key}"))?;
    if !matches!(p.scheme, QuantScheme::MlxAffine { .. }) {
        return Err(anyhow!(
            "fused split only supports MLX affine, got {}",
            p.scheme
        ));
    }
    let scheme = p.scheme;
    let total_rows: usize = splits.iter().map(|&(_, r)| r).sum();
    let in_dim = p.out_shape.get(1).copied().unwrap_or(0);
    let n_groups = p.n_groups().max(1);
    let row_bytes = p.w_q.len() / total_rows.max(1);
    let scales = f32_from_le_bytes(&p.scales);
    let biases = if p.biases.is_empty() {
        vec![0f32; total_rows * n_groups]
    } else {
        f32_from_le_bytes(&p.biases)
    };
    let mut out = Vec::with_capacity(splits.len());
    let mut row0 = 0usize;
    for &(name, rows) in splits {
        let c0 = row0 * row_bytes;
        let c1 = (row0 + rows) * row_bytes;
        let s0 = row0 * n_groups;
        let s1 = (row0 + rows) * n_groups;
        let codes = p.w_q[c0..c1].to_vec();
        let cname = format!("{key}.{name}");
        let w = g.param(&cname, Shape::new(&[codes.len()], DType::U8));
        packed.insert(cname, (codes, scheme, vec![rows, in_dim]));
        let sname = format!("{key}.{name}.scales");
        let scale = g.param(&sname, Shape::new(&[rows, n_groups], DType::F32));
        params.insert(sname, scales[s0..s1].to_vec());
        let bname = format!("{key}.{name}.biases");
        let bias = g.param(&bname, Shape::new(&[rows, n_groups], DType::F32));
        params.insert(bname, biases[s0..s1].to_vec());
        out.push(Proj {
            w,
            scheme: Some(scheme),
            scale: Some(scale),
            bias: Some(bias),
        });
        row0 += rows;
    }
    Ok(out)
}

/// Stacked MLX-affine expert weights for one MoE projection (`gate_proj` /
/// `up_proj` / `down_proj`): concatenate each expert's packed codes / scales /
/// biases along a new leading expert dim. Returns `(codes, scales, biases,
/// scheme, out_features)` for [`Op::DequantGroupedMatMulMlx`].
fn load_stacked_experts_mlx(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    proj: &str,
    n_expert: usize,
) -> Result<(NodeId, NodeId, NodeId, QuantScheme, usize)> {
    let mut codes: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut biases: Vec<f32> = Vec::new();
    let mut out_dim = 0usize;
    let mut n_groups = 1usize;
    let mut scheme = QuantScheme::MlxAffine {
        bits: 4,
        group_size: 64,
    };
    for e in 0..n_expert {
        let key = format!("{lp}.mlp.experts.{e}.{proj}.weight");
        let p = weights
            .take_packed_mlx(&key)?
            .ok_or_else(|| anyhow!("MoE expert weight not MLX-packed: {key}"))?;
        if !matches!(p.scheme, QuantScheme::MlxAffine { .. }) {
            return Err(anyhow!(
                "MoE grouped path only supports MLX affine, got {}",
                p.scheme
            ));
        }
        scheme = p.scheme;
        out_dim = p.out_shape.first().copied().unwrap_or(0);
        n_groups = p.n_groups().max(1);
        codes.extend_from_slice(&p.w_q);
        scales.extend(f32_from_le_bytes(&p.scales));
        if p.biases.is_empty() {
            biases.extend(std::iter::repeat_n(0.0f32, out_dim * n_groups));
        } else {
            biases.extend(f32_from_le_bytes(&p.biases));
        }
    }
    let cname = format!("{lp}.moe.{proj}.codes");
    let cnode = g.param(&cname, Shape::new(&[codes.len()], DType::U8));
    packed.insert(cname, (codes, scheme, vec![n_expert, out_dim, n_groups]));
    let sname = format!("{lp}.moe.{proj}.scales");
    let snode = g.param(
        &sname,
        Shape::new(&[n_expert, out_dim, n_groups], DType::F32),
    );
    params.insert(sname, scales);
    let bname = format!("{lp}.moe.{proj}.biases");
    let bnode = g.param(
        &bname,
        Shape::new(&[n_expert, out_dim, n_groups], DType::F32),
    );
    params.insert(bname, biases);
    Ok((cnode, snode, bnode, scheme, out_dim))
}

/// Same as [`load_stacked_experts_mlx`] but for the **stacked** `switch_mlp`
/// layout (Qwen3-MoE / Qwen2-MoE): all experts live in one already-stacked
/// `[n_expert, out, in]` MLX tensor rather than `n_expert` separate ones. mlx
/// stores it row-major, so its packed bytes are byte-identical to the
/// per-expert concatenation — we just take the single tensor and label the
/// dims from the config (the 2D `MlxPackedLinear.out_shape` is meaningless for
/// a 3D tensor, so `out_dim` is passed in and `n_groups` derived from the
/// scale count).
fn load_switch_experts_mlx(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    proj: &str,
    n_expert: usize,
    out_dim: usize,
) -> Result<(NodeId, NodeId, NodeId, QuantScheme, usize)> {
    load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        lp,
        "mlp.switch_mlp",
        proj,
        n_expert,
        out_dim,
    )
}

/// [`load_switch_experts_mlx`] generalized over the stacked-expert container
/// path (relative to the layer): Qwen/GLM/deepseek use `mlp.switch_mlp`,
/// gpt-oss `mlp.experts`, MiniMax `block_sparse_moe.switch_mlp`. Each is a single
/// already-stacked `[n_expert, out, in]` MLX tensor (affine or mxfp4) → nodes for
/// [`Op::DequantGroupedMatMulMlx`].
#[allow(clippy::too_many_arguments)]
fn load_stacked_group_experts_mlx(
    g: &mut Graph,
    _params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    container: &str,
    proj: &str,
    n_expert: usize,
    out_dim: usize,
) -> Result<(NodeId, NodeId, NodeId, QuantScheme, usize)> {
    let key = format!("{lp}.{container}.{proj}.weight");
    // Structure-only vs materialize: a `StructureLoader` returns metadata with the
    // large `w_q` codes DEFERRED (empty), so a worker streams them from the
    // checkpoint at compile time instead of holding a full RAM copy alongside the
    // arena (the 2× peak that OOMs a node whose stage ≈ its RAM). A normal loader
    // returns `None` from `packed_mlx_meta` and we take the real packed tensor.
    let (scheme, codes, w_q_len, raw_scales, raw_biases) =
        if let Some(meta) = weights.packed_mlx_meta(&key)? {
            (
                meta.scheme,
                Vec::<u8>::new(),
                meta.w_q_len,
                meta.scales,
                meta.biases,
            )
        } else {
            let p = weights
                .take_packed_mlx(&key)?
                .ok_or_else(|| anyhow!("MoE switch_mlp weight not MLX-packed: {key}"))?;
            let wl = p.w_q.len();
            (p.scheme, p.w_q, wl, p.scales, p.biases)
        };
    // MLX-affine: f32 scales+biases. MXFP4 (gpt-oss experts): E8M0 u8 scales
    // (one byte/group) → decode to f32 so the grouped op stays f32-uniform; no
    // zero-point (biases = 0). Both flow through `Op::DequantGroupedMatMulMlx`.
    let (scales, n_groups) = match scheme {
        QuantScheme::MlxAffine { .. } => {
            let s = f32_from_le_bytes(&raw_scales);
            let ng = (s.len() / (n_expert.max(1) * out_dim.max(1))).max(1);
            (s, ng)
        }
        QuantScheme::MlxMxfp4 { .. } => {
            // Raw scales are 1 E8M0 byte per (expert, out, group).
            let ng = (raw_scales.len() / (n_expert.max(1) * out_dim.max(1))).max(1);
            let s: Vec<f32> = raw_scales
                .iter()
                .map(|&b| rlx_mlx_io::mxfp4_scale_e8m0_to_f32(b))
                .collect();
            (s, ng)
        }
        other => {
            return Err(anyhow!(
                "MoE switch_mlp supports MLX affine/mxfp4, got {other}"
            ));
        }
    };
    let biases = if matches!(scheme, QuantScheme::MlxAffine { .. }) && !raw_biases.is_empty() {
        f32_from_le_bytes(&raw_biases)
    } else {
        vec![0f32; n_expert * out_dim * n_groups]
    };
    // Name the codes param by the CHECKPOINT KEY (not a derived `…moe…codes`) so a
    // streaming `ManifestParamSource` re-fetches it via `take_packed_mlx(key)`.
    // In structure mode `codes` is empty (deferred); in normal mode it's the real
    // bytes served in-band by `MapParamSource`.
    let cname = key.clone();
    let cnode = g.param(&cname, Shape::new(&[w_q_len], DType::U8));
    packed.insert(cname, (codes, scheme, vec![n_expert, out_dim, n_groups]));
    // Store per-expert scales/biases as BF16 (2 bytes) instead of expanding the
    // checkpoint's bf16 to f32 — the MoE scale slabs rival the packed codes at
    // 2-bit, so f32 nearly doubled resident memory → swapping. bf16 is the native
    // source dtype so this re-round is exact; the CPU grouped kernel decodes
    // bf16→f32 per expert (`scale_bf16`).
    let to_bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect()
    };
    let sname = format!("{lp}.moe.{proj}.scales");
    let snode = g.param(
        &sname,
        Shape::new(&[n_expert, out_dim, n_groups], DType::BF16),
    );
    packed.insert(
        sname,
        (to_bf16(&scales), scheme, vec![n_expert, out_dim, n_groups]),
    );
    let bname = format!("{lp}.moe.{proj}.biases");
    let bnode = g.param(
        &bname,
        Shape::new(&[n_expert, out_dim, n_groups], DType::BF16),
    );
    packed.insert(
        bname,
        (to_bf16(&biases), scheme, vec![n_expert, out_dim, n_groups]),
    );
    Ok((cnode, snode, bnode, scheme, out_dim))
}

/// Create a `[1]` f32 constant param (for broadcast scalar mul/add).
fn const1(g: &mut Graph, params: &mut HashMap<String, Vec<f32>>, name: &str, v: f32) -> NodeId {
    let node = g.param(name, Shape::new(&[1], DType::F32));
    params.insert(name.to_string(), vec![v]);
    node
}

/// Load an optional per-expert linear bias `{lp}.mlp.experts.{proj}.bias`
/// (`[n_expert, out]`) as a graph param, or `None` if absent.
fn load_expert_bias(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    proj: &str,
    n_expert: usize,
    out: usize,
) -> Option<NodeId> {
    let key = format!("{lp}.mlp.experts.{proj}.bias");
    let (data, _shape) = weights.take(&key).ok()?;
    // Name by the checkpoint key so a streaming source re-fetches it.
    let node = g.param(&key, Shape::new(&[n_expert, out], DType::F32));
    params.insert(key, data);
    Some(node)
}

/// Build the **gpt-oss** MoE FFN block: affine-quant router (+bias) → top-k on
/// logits → softmax over the *selected* logits → per-expert **clamped-SwiGLU**
/// (`gate.clamp(≤limit)`, `up.clamp(±limit)`, `glu=(up+1)·g·σ(α·g)`) over
/// MXFP4-packed stacked experts (`mlp.experts.{gate,up,down}_proj`, optional
/// linear biases) → router-weighted sum. `x_3d` is the pre-MoE normed hidden
/// `[batch, seq, hidden]`; returns `[batch, seq, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn build_gpt_oss_moe_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x_3d: NodeId,
    batch: usize,
    seq: usize,
    hidden: usize,
    n_expert: usize,
    top_k: usize,
    moe_inter: usize,
    swiglu_limit: f32,
    alpha: f32,
) -> Result<NodeId> {
    let f = DType::F32;
    let rows = batch * seq;
    let h_2d = g.reshape_(x_3d, vec![rows as i64, hidden as i64]);

    // Router: affine-quant Linear `[n_expert, hidden]` (+ optional bias).
    let router_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.mlp.router.weight"),
    )?;
    let mut logits = emit_proj(g, h_2d, &router_p, Shape::new(&[rows, n_expert], f));
    let rb_key = format!("{lp}.mlp.router.bias");
    if let Ok((rb, _)) = weights.take(&rb_key) {
        // Name by the checkpoint key so a streaming source re-fetches it.
        let rb_node = g.param(&rb_key, Shape::new(&[n_expert], f));
        params.insert(rb_key, rb);
        logits = g.add(logits, rb_node);
    }
    // gpt-oss routing: top-k on logits, then softmax over the selected logits.
    let top_idx = g.add_node(
        Op::TopK { k: top_k },
        vec![logits],
        Shape::new(&[rows, top_k], DType::F32),
    );
    let top_logits = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![logits, top_idx],
        Shape::new(&[rows, top_k], f),
    );
    let top_w = g.sm(top_logits, -1); // softmax over the top-k axis

    // Stacked MXFP4 experts (`mlp.experts.{proj}`) + optional per-expert bias.
    let (gate_c, gate_s, gate_b, scheme, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        lp,
        "mlp.experts",
        "gate_proj",
        n_expert,
        moe_inter,
    )?;
    let (up_c, up_s, up_b, _, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        lp,
        "mlp.experts",
        "up_proj",
        n_expert,
        moe_inter,
    )?;
    let (down_c, down_s, down_b, _, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        lp,
        "mlp.experts",
        "down_proj",
        n_expert,
        hidden,
    )?;
    let gate_bias = load_expert_bias(g, params, weights, lp, "gate_proj", n_expert, moe_inter);
    let up_bias = load_expert_bias(g, params, weights, lp, "up_proj", n_expert, moe_inter);
    let down_bias = load_expert_bias(g, params, weights, lp, "down_proj", n_expert, hidden);

    let alpha_node = const1(g, params, &format!("{lp}.moe.alpha"), alpha);
    let one_node = const1(g, params, &format!("{lp}.moe.one"), 1.0);

    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let e_col = g.narrow_(top_idx, 1, ki, 1);
        let e_idx = g.reshape_(e_col, vec![rows as i64]);
        let w_col = g.narrow_(top_w, 1, ki, 1); // [rows, 1]

        let mut gate = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![h_2d, gate_c, gate_s, gate_b, e_idx],
            Shape::new(&[rows, moe_inter], f),
        );
        let mut up = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![h_2d, up_c, up_s, up_b, e_idx],
            Shape::new(&[rows, moe_inter], f),
        );
        // Add per-expert linear biases (gather the selected experts' rows).
        if let Some(gb) = gate_bias {
            let g_add = g.add_node(
                Op::Gather { axis: 0 },
                vec![gb, e_idx],
                Shape::new(&[rows, moe_inter], f),
            );
            gate = g.add(gate, g_add);
        }
        if let Some(ub) = up_bias {
            let u_add = g.add_node(
                Op::Gather { axis: 0 },
                vec![ub, e_idx],
                Shape::new(&[rows, moe_inter], f),
            );
            up = g.add(up, u_add);
        }
        // Clamped-SwiGLU: gate.clamp(≤limit); up.clamp(±limit); (up+1)·g·σ(α·g).
        let gate_cl = g.add_node(
            Op::Clamp {
                min: f32::NEG_INFINITY,
                max: swiglu_limit,
            },
            vec![gate],
            Shape::new(&[rows, moe_inter], f),
        );
        let up_cl = g.add_node(
            Op::Clamp {
                min: -swiglu_limit,
                max: swiglu_limit,
            },
            vec![up],
            Shape::new(&[rows, moe_inter], f),
        );
        let ag = g.mul(gate_cl, alpha_node);
        let sig = g.sigmoid(ag);
        let glu = g.mul(gate_cl, sig);
        let up1 = g.add(up_cl, one_node);
        let act = g.mul(up1, glu);

        let mut down = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![act, down_c, down_s, down_b, e_idx],
            Shape::new(&[rows, hidden], f),
        );
        if let Some(db) = down_bias {
            let d_add = g.add_node(
                Op::Gather { axis: 0 },
                vec![db, e_idx],
                Shape::new(&[rows, hidden], f),
            );
            down = g.add(down, d_add);
        }
        let weighted = g.mul(down, w_col);
        acc = Some(match acc {
            None => weighted,
            Some(a) => g.add(a, weighted),
        });
    }
    let out = acc.ok_or_else(|| anyhow!("gpt-oss MoE: top_k=0"))?;
    Ok(g.reshape_(out, vec![batch as i64, seq as i64, hidden as i64]))
}

/// LFM2 / LFM2.5 hybrid ShortConv + attention decoder spec (LiquidAI-native).
#[derive(Debug, Clone)]
pub struct Lfm2Spec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub conv_dim: usize,
    /// `conv_L_cache` — depthwise causal conv1d kernel size.
    pub conv_kernel: usize,
    /// `full_attn_idxs` — layer indices that use GQA attention; the rest ShortConv.
    pub full_attn_layers: Vec<usize>,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
}

/// Register a fixed-shape zero param (causal-conv left padding).
fn register_zeros(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    shape: &[usize],
) -> NodeId {
    let n: usize = shape.iter().product();
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let node = g.param(
        name,
        Shape::new(
            &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            DType::F32,
        ),
    );
    params.insert(name.to_string(), vec![0f32; n]);
    node
}

/// Depthwise **causal** 1-D conv over `[batch, seq, channels]` with a
/// left-pad of `k-1` (BF16 `conv.conv.weight` `[channels, k, 1]` → F32). Uses
/// `Op::Conv` (NCHW `[N,C,L,1]`, kernel `[k,1]`, `groups=channels`).
fn depthwise_causal_conv_lfm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    input_bsc: NodeId,
    batch: usize,
    seq: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    let (wdata, _shape) = weights.take(key)?; // [channels, k, 1] row-major == [channels,1,k,1]
    let pad = register_zeros(
        g,
        params,
        &format!("{key}.causal_pad"),
        &[batch, k - 1, channels],
    );
    let padded = g.concat_(vec![pad, input_bsc], 1); // [batch, (k-1)+seq, channels]
    let width = (k - 1) + seq;
    let bcw = g.transpose_(padded, vec![0, 2, 1]);
    let nchw = g.reshape_(bcw, vec![batch as i64, channels as i64, width as i64, 1]);
    let w = g.param(key, Shape::new(&[channels, 1, k, 1], DType::F32));
    params.insert(key.to_string(), wdata);
    let conv = g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, w],
        Shape::new(&[batch, channels, seq, 1], DType::F32),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, seq as i64]);
    Ok(g.transpose_(bcs, vec![0, 2, 1])) // [batch, seq, channels]
}

/// LFM2 **ShortConv** mixer: `in_proj(x)` → split `B|C|x` → `Bx = B·x` →
/// depthwise causal conv → `y = C·conv(Bx)` → `out_proj(y)`.
#[allow(clippy::too_many_arguments)]
fn build_lfm2_shortconv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    batch: usize,
    seq: usize,
    hidden: usize,
    cdim: usize,
    k: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let in_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.conv.in_proj.weight"),
    )?;
    let bcx = emit_proj(g, x, &in_p, Shape::new(&[batch, seq, 3 * cdim], f));
    let b_part = g.narrow_(bcx, 2, 0, cdim);
    let c_part = g.narrow_(bcx, 2, cdim, cdim);
    let x_part = g.narrow_(bcx, 2, 2 * cdim, cdim);
    let bx = g.mul(b_part, x_part);
    let conv_out = depthwise_causal_conv_lfm(
        g,
        params,
        weights,
        &format!("{lp}.conv.conv.weight"),
        bx,
        batch,
        seq,
        cdim,
        k,
    )?;
    let y = g.mul(c_part, conv_out);
    let out_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.conv.out_proj.weight"),
    )?;
    Ok(emit_proj(
        g,
        y,
        &out_p,
        Shape::new(&[batch, seq, hidden], f),
    ))
}

/// Build an **LFM2 / LFM2.5** prefill graph (ShortConv + GQA-attention hybrid,
/// SwiGLU FFN, tied LM head). Weights load via the packed-affine path
/// (`load_proj` → `Op::DequantMatMul`); norms/conv are dense F32. Returns logits
/// `[seq, vocab]`.
pub fn build_lfm2_prefill(
    spec: &Lfm2Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("lfm2_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = nh / nkv.max(1);
    let eps = spec.rms_norm_eps;
    let inter = spec.intermediate_size;
    let cdim = spec.conv_dim;
    let kc = spec.conv_kernel;

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "lfm.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "lfm.zero_beta.head_dim", dh);

    // RoPE tables (single θ, no scaling).
    let half = dh / 2;
    let mut cos_data = vec![0f32; seq * half];
    let mut sin_data = vec![0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let freq = spec.rope_theta.powf(-(2.0 * i as f64) / dh as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param("lfm.rope.cos", Shape::new(&[seq, half], f));
    params.insert("lfm.rope.cos".into(), cos_data);
    let sin_id = g.param("lfm.rope.sin", Shape::new(&[seq, half], f));
    params.insert("lfm.rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    // Tied embed: [vocab, hidden] dequant'd F32; reused (transposed) as LM head.
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0); // [batch, seq, hidden]

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let op_norm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.operator_norm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, op_norm, zero_beta_hidden, eps);

        let mixed = if spec.full_attn_layers.contains(&il) {
            let kv_dim = nkv * dh;
            let q_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let k_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.k_proj.weight"),
            )?;
            let v_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.v_proj.weight"),
            )?;
            let q = emit_proj(&mut g, normed, &q_p, Shape::new(&[batch, seq, nh * dh], f));
            let k = emit_proj(&mut g, normed, &k_p, Shape::new(&[batch, seq, kv_dim], f));
            let v = emit_proj(&mut g, normed, &v_p, Shape::new(&[batch, seq, kv_dim], f));
            // Per-head QK-norm (`q_layernorm`/`k_layernorm`).
            let qn_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.q_layernorm.weight"),
                0.0,
            )?;
            let kn_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.k_layernorm.weight"),
                0.0,
            )?;
            let qn = per_head_rms(&mut g, q, qn_g, zero_beta_headdim, batch, seq, nh, dh, eps);
            let kn = per_head_rms(&mut g, k, kn_g, zero_beta_headdim, batch, seq, nkv, dh, eps);
            let q_rope = rope_heads(&mut g, qn, cos_id, sin_id, batch, seq, nh, dh, true);
            let k_rope = rope_heads(&mut g, kn, cos_id, sin_id, batch, seq, nkv, dh, true);
            let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
            let v_rep = repeat_kv(&mut g, v, nkv, dh, group);
            let attn_shape = shape::attention_shape(g.shape(q_rope));
            let attn = g.add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: dh,
                    v_head_dim: None,
                    mask_kind: MaskKind::Causal,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q_rope, k_rep, v_rep],
                attn_shape,
            );
            let o_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.out_proj.weight"),
            )?;
            emit_proj(&mut g, attn, &o_p, Shape::new(&[batch, seq, h], f))
        } else {
            build_lfm2_shortconv(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                normed,
                batch,
                seq,
                h,
                cdim,
                kc,
            )?
        };
        let h_after = g.add(h_id, mixed);

        // SwiGLU FFN: w1=gate, w3=up, w2=down.
        let ffn_norm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let ffn_normed = g.rms_norm(h_after, ffn_norm, zero_beta_hidden, eps);
        let w1 = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.feed_forward.w1.weight"),
        )?;
        let w3 = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.feed_forward.w3.weight"),
        )?;
        let w2 = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.feed_forward.w2.weight"),
        )?;
        let gate = emit_proj(&mut g, ffn_normed, &w1, Shape::new(&[batch, seq, inter], f));
        let up = emit_proj(&mut g, ffn_normed, &w3, Shape::new(&[batch, seq, inter], f));
        let gate_act = g.silu(gate);
        let glu = g.mul(gate_act, up);
        let down = emit_proj(&mut g, glu, &w2, Shape::new(&[batch, seq, h], f));
        h_id = g.add(h_after, down);
    }

    let final_norm = load_norm(
        &mut g,
        &mut params,
        weights,
        "model.embedding_norm.weight",
        0.0,
    )?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let hidden_2d = g.reshape_(hidden, vec![(batch * seq) as i64, h as i64]);
    // Tied LM head: logits = hidden @ embedᵀ.
    let embed_t = g.transpose_(embed_w, vec![1, 0]); // [hidden, vocab]
    let logits = g.mm(hidden_2d, embed_t); // [seq, vocab]
    g.set_outputs(vec![logits]);
    Ok((g, params))
}

/// DeepSeek-V2/V3 (+ Moonlight/Kimi) spec: MLA attention + fine-grained MoE.
#[derive(Debug, Clone)]
pub struct DeepseekSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Q low-rank (`q_lora_rank`): 0 = direct `q_proj` (V2-Lite); >0 = q-LoRA
    /// `q_b_proj(q_a_layernorm(q_a_proj(x)))` (DeepSeek-V3 / Kimi-K2).
    pub q_lora_rank: usize,
    /// Absorbed MLA: the checkpoint stores per-head `embed_q`/`unembed_out`
    /// (MultiLinear) instead of a single `kv_b_proj` (GLM-5 / DeepSeek-V3.2).
    /// `false` = standard kv_b_proj (V2-Lite/V3/Kimi-K2).
    pub absorbed_mla: bool,
    /// MLA low-rank / head dims.
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Dense FFN width (the first `first_k_dense_replace` layers).
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    /// `sigmoid` (V3/Moonlight/Kimi) vs `softmax` (V2) router scoring.
    pub sigmoid_gate: bool,
    /// DeepSeek-V4 `score_func="sqrtsoftplus"`: `scores = sqrt(softplus(logits))`
    /// (takes precedence over `sigmoid_gate`). Always top-k-normalized like the
    /// sigmoid path.
    pub sqrtsoftplus_gate: bool,
    /// Clamped-SwiGLU expert limit (DeepSeek-V4 / gpt-oss): before `silu(gate)*up`,
    /// clamp `up ∈ [-L, L]` and `gate ≤ L`. `0.0` disables (plain SwiGLU).
    pub swiglu_limit: f32,
    pub rope_theta: f64,
    /// Decoupled-RoPE frequency scaling (DeepSeek YaRN, or `None`). Applied to
    /// the rope-dim `inv_freq`; deepseek keeps the cos/sin table itself unscaled.
    pub rope_scaling: RopeScaling,
    /// Full attention `softmax_scale` (deepseek folds YaRN `mscale²` into it);
    /// `None` = default `qk_head_dim^-0.5`.
    pub attn_score_scale: Option<f32>,
    /// Decoupled-RoPE pairing: `false` = GptJ (interleaved adjacent pairs),
    /// `true` = NeoX (first-half/second-half).
    pub rope_neox: bool,
    pub rms_norm_eps: f32,
}

/// Load an absorbed-MLA `MultiLinear` (`embed_q`/`unembed_out`) — a per-head 3D
/// weight `[H, a, b]` (dequantized to F32 by the loader) — and fold it into a
/// single 2D matrix `[contract, H*result]` so `kv_latent[.,contract] @ mat`
/// reproduces all heads' projections in ONE matmul (kv-latent is shared across
/// heads). mlx applies `embed_q` with `transpose=False` (slice used as-is,
/// `contract=a`, `result=b`) and `unembed_out` with `transpose=True` (each
/// `[a,b]` slice transposed, `contract=b`, `result=a`). Returns `(node, contract,
/// H*result)`.
fn load_absorbed_multilinear(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    num_heads: usize,
    transpose_slice: bool,
) -> Result<(NodeId, usize, usize)> {
    let (data, shape) = weights.take(key)?; // dequantized F32
    anyhow::ensure!(
        shape.len() == 3 && shape[0] == num_heads,
        "{key}: expected 3D MultiLinear [H,a,b] with H={num_heads}, got {shape:?}"
    );
    let (hh, a, b) = (shape[0], shape[1], shape[2]);
    let (contract, result) = if transpose_slice { (b, a) } else { (a, b) };
    let mut mat = vec![0f32; contract * hh * result];
    for hd in 0..hh {
        for i in 0..contract {
            for j in 0..result {
                // slice[hd] is [a,b] row-major at data[hd*a*b ..]. Non-transposed:
                // element (i over a, j over b) = data[hd,i,j]. Transposed: contract
                // runs over b, result over a → element = data[hd, j, i].
                let v = if transpose_slice {
                    data[(hd * a + j) * b + i]
                } else {
                    data[(hd * a + i) * b + j]
                };
                mat[i * (hh * result) + hd * result + j] = v;
            }
        }
    }
    let node = g.param(key, Shape::new(&[contract, hh * result], DType::F32));
    params.insert(key.to_string(), mat);
    Ok((node, contract, hh * result))
}

/// MLA attention for one DeepSeek layer. Q is direct `q_proj` (`q_lora_rank==0`,
/// V2-Lite) or q-LoRA (`>0`, V3/Kimi); K/V come from a single `kv_b_proj` or the
/// absorbed per-head `embed_q`/`unembed_out` (`absorbed_mla`, GLM-5/V3.2).
///
/// `pub` so an isolation test can validate the absorbed path against the
/// (validated) kv_b_proj path on synthetic weights.
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_mla(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    seq: usize,
    spec: &DeepseekSpec,
) -> Result<NodeId> {
    let f = DType::F32;
    let h = spec.num_attention_heads;
    let nope = spec.qk_nope_head_dim;
    let rope = spec.qk_rope_head_dim;
    let qk = nope + rope;
    let vd = spec.v_head_dim;
    let eps = spec.rms_norm_eps;
    let (b, s) = (batch as i64, seq as i64);

    // Q: direct `q_proj` (V2-Lite) or q-LoRA `q_b_proj(q_a_layernorm(q_a_proj(x)))`
    // (DeepSeek-V3 / Kimi-K2 — those checkpoints have no `q_proj`, only q_a/q_b).
    let q = if spec.q_lora_rank > 0 {
        let qa_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_a_proj.weight"),
        )?;
        let qa = emit_proj(g, x, &qa_p, Shape::new(&[batch, seq, spec.q_lora_rank], f));
        let qa_norm_g = load_norm(
            g,
            params,
            weights,
            &format!("{lp}.self_attn.q_a_layernorm.weight"),
            0.0,
        )?;
        let qa_beta = synth_zero(g, params, &format!("{lp}.qa.beta"), spec.q_lora_rank);
        let qa_n = g.rms_norm(qa, qa_norm_g, qa_beta, 1e-6); // deepseek q_a_layernorm eps = 1e-6
        let qb_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_b_proj.weight"),
        )?;
        emit_proj(g, qa_n, &qb_p, Shape::new(&[batch, seq, h * qk], f))
    } else {
        let q_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        emit_proj(g, x, &q_p, Shape::new(&[batch, seq, h * qk], f))
    };
    // Compressed KV: kv_a_proj_with_mqa → [kv_lora | k_rot].
    let ckv_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
    )?;
    let ckv = emit_proj(
        g,
        x,
        &ckv_p,
        Shape::new(&[batch, seq, spec.kv_lora_rank + rope], f),
    );
    let k_lora = g.narrow_(ckv, 2, 0, spec.kv_lora_rank);
    let k_rot = g.narrow_(ckv, 2, spec.kv_lora_rank, rope); // [b,s,rope]
    let kva_g = load_norm(
        g,
        params,
        weights,
        &format!("{lp}.self_attn.kv_a_layernorm.weight"),
        0.0,
    )?;
    let kva_beta = synth_zero(g, params, &format!("{lp}.kva.beta"), spec.kv_lora_rank);
    let k_lora_n = g.rms_norm(k_lora, kva_g, kva_beta, eps);

    // Per-head Q splits.
    let q4 = g.reshape_(q, vec![b, s, h as i64, qk as i64]);
    let q_pass = g.narrow_(q4, 3, 0, nope);
    let q_rot = g.narrow_(q4, 3, nope, rope);

    // k_nope / value: standard single `kv_b_proj` (V2-Lite/V3/Kimi), OR the
    // ABSORBED form — per-head `embed_q`/`unembed_out` MultiLinear (GLM-5 /
    // DeepSeek-V3.2 store these instead of kv_b_proj). Both produce `k_nope
    // [b,s,h,nope]` and `value [b,s,h,vd]` from the shared kv-latent; the
    // absorbed projections collapse to one matmul each (see
    // `load_absorbed_multilinear`).
    let (k_nope, value) = if spec.absorbed_mla {
        let kv2d = g.reshape_(k_lora_n, vec![b * s, spec.kv_lora_rank as i64]);
        let (eq, _, _) = load_absorbed_multilinear(
            g,
            params,
            weights,
            &format!("{lp}.self_attn.embed_q.weight"),
            h,
            false,
        )?;
        let kn = g.mm(kv2d, eq);
        let k_nope = g.reshape_(kn, vec![b, s, h as i64, nope as i64]);
        let (uo, _, _) = load_absorbed_multilinear(
            g,
            params,
            weights,
            &format!("{lp}.self_attn.unembed_out.weight"),
            h,
            true,
        )?;
        let vf = g.mm(kv2d, uo);
        let value = g.reshape_(vf, vec![b, s, h as i64, vd as i64]);
        (k_nope, value)
    } else {
        let kvb_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{lp}.self_attn.kv_b_proj.weight"),
        )?;
        let kv_up = emit_proj(
            g,
            k_lora_n,
            &kvb_p,
            Shape::new(&[batch, seq, h * (nope + vd)], f),
        );
        let kv4 = g.reshape_(kv_up, vec![b, s, h as i64, (nope + vd) as i64]);
        let k_nope = g.narrow_(kv4, 3, 0, nope);
        let value = g.narrow_(kv4, 3, nope, vd);
        (k_nope, value)
    };

    // Decoupled RoPE. Heads go in the FEATURE dim ([b,s,h*rope]) so the op
    // indexes cos/sin by the true seq position (NOT s*h). Each rope-block of
    // width `rope` is rotated independently.
    let style = if spec.rope_neox {
        RopeStyle::NeoX
    } else {
        RopeStyle::GptJ
    };
    let q_rot_f = g.reshape_(q_rot, vec![b, s, (h * rope) as i64]);
    let q_rot_f = g.rope_styled(q_rot_f, cos, sin, rope, style);
    let q_rot = g.reshape_(q_rot_f, vec![b, s, h as i64, rope as i64]);
    let k_rot_r = g.rope_styled(k_rot, cos, sin, rope, style); // [b,s,rope]
    let k_rot_4 = g.reshape_(k_rot_r, vec![b, s, 1, rope as i64]);
    // Broadcast the single k_rot head to all h heads.
    let ones = g.param(format!("{lp}.mla.kexp"), Shape::new(&[1, 1, h, 1], f));
    params.insert(format!("{lp}.mla.kexp"), vec![1.0; h]);
    let k_rot_h = g.mul(k_rot_4, ones); // [b,s,h,rope]

    let query = g.concat_(vec![q_pass, q_rot], 3); // [b,s,h,qk]
    let key = g.concat_(vec![k_nope, k_rot_h], 3);
    // Pad value to qk_head_dim with zeros (rope-width tail).
    let vzeros = register_zeros(
        g,
        params,
        &format!("{lp}.mla.vzeros"),
        &[batch, seq, h, qk - vd],
    );
    let value_pad = g.concat_(vec![value, vzeros], 3); // [b,s,h,qk]

    let qf = g.reshape_(query, vec![b, s, (h * qk) as i64]);
    let kf = g.reshape_(key, vec![b, s, (h * qk) as i64]);
    let vf = g.reshape_(value_pad, vec![b, s, (h * qk) as i64]);
    let attn_shape = shape::attention_shape(g.shape(qf));
    let out = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: qk,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: spec.attn_score_scale, // deepseek YaRN folds mscale² here
            attn_logit_softcap: None,
        },
        vec![qf, kf, vf],
        attn_shape,
    );
    let out4 = g.reshape_(out, vec![b, s, h as i64, qk as i64]);
    let out_v = g.narrow_(out4, 3, 0, vd);
    let out_flat = g.reshape_(out_v, vec![b, s, (h * vd) as i64]);
    let o_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.o_proj.weight"),
    )?;
    Ok(emit_proj(
        g,
        out_flat,
        &o_p,
        Shape::new(&[batch, seq, spec.hidden_size], f),
    ))
}

/// Fine-grained MoE FFN (DeepSeek-V3 / Moonlight): sigmoid(+correction-bias) or
/// softmax router → top-k → norm+routed_scaling → per-expert SwiGLU (packed
/// `switch_mlp`) + always-on shared experts. `n_group=1` (plain top-k).
#[allow(clippy::too_many_arguments)]
fn build_deepseek_moe(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    batch: usize,
    seq: usize,
    spec: &DeepseekSpec,
) -> Result<NodeId> {
    build_deepseek_moe_c(
        g, params, packed, weights, lp, x, batch, seq, spec, "mlp", None,
    )
}

/// DeepSeek-V4's MoE FFN lives under `{lp}.ffn.*` (vs `mlp.*`); same routing, with
/// optional `hash_ids` (token-id node) for `tid2eid` hash routing on early layers.
#[allow(clippy::too_many_arguments)]
fn build_deepseek_moe_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    batch: usize,
    seq: usize,
    spec: &DeepseekSpec,
    hash_ids: Option<NodeId>,
) -> Result<NodeId> {
    build_deepseek_moe_c(
        g, params, packed, weights, lp, x, batch, seq, spec, "ffn", hash_ids,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_deepseek_moe_c(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    batch: usize,
    seq: usize,
    spec: &DeepseekSpec,
    ct: &str,
    // DeepSeek-V4 hash-routing layers (first `n_hash_layers`): expert indices come
    // from a per-token-id `gate.tid2eid` lookup instead of score top-k. `Some(ids)`
    // is the `[rows]` int token-id node driving that gather; `None` = score top-k.
    hash_ids: Option<NodeId>,
) -> Result<NodeId> {
    let f = DType::F32;
    let rows = batch * seq;
    let h = spec.hidden_size;
    let n = spec.n_routed_experts;
    let top_k = spec.num_experts_per_tok;
    let inter = spec.moe_intermediate_size;
    let h2d = g.reshape_(x, vec![rows as i64, h as i64]);

    let router_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.{ct}.gate.weight"),
    )?;
    let logits = emit_proj(g, h2d, &router_p, Shape::new(&[rows, n], f));
    // Router scores: sqrtsoftplus (V4), sigmoid (V3), or softmax (V2).
    let scores = if spec.sqrtsoftplus_gate {
        let sp = softplus_stable(g, params, logits, &format!("{lp}.moe.sp"));
        g.sqrt(sp)
    } else if spec.sigmoid_gate {
        g.sigmoid(logits)
    } else {
        g.sm(logits, -1)
    };
    // Selection score = scores + e_score_correction_bias (V3 noaux_tc); V2 has no
    // bias. DeepSeek stores it under `mlp.gate.e_score_correction_bias`; Kimi-Linear
    // under `mlp.e_score_correction_bias` — try both.
    // Name the param by whichever checkpoint key holds it (V3 `gate.…` vs Kimi
    // `…`), so a streaming ManifestParamSource re-fetches it (a derived `…moe.ebias`
    // would miss the manifest → zero the routing bias → wrong expert selection).
    let ekey_gate = format!("{lp}.{ct}.gate.e_score_correction_bias");
    let ekey_flat = format!("{lp}.{ct}.e_score_correction_bias");
    let (ebias, ekey) = match weights.take(&ekey_gate) {
        Ok(v) => (Ok(v), ekey_gate),
        Err(_) => (weights.take(&ekey_flat), ekey_flat),
    };
    let route = if let Ok((eb, _)) = ebias {
        let eb_node = g.param(&ekey, Shape::new(&[n], f));
        params.insert(ekey, eb);
        g.add(scores, eb_node)
    } else {
        scores
    };
    // Expert selection: per-token-id `tid2eid` hash lookup on hash-routing layers,
    // else score top-k. `tid2eid` is `[vocab, top_k]` (int expert ids stored f32).
    let top_idx = if let Some(ids) = hash_ids {
        let t2e_key = format!("{lp}.{ct}.gate.tid2eid");
        let (t2e, t2e_shape) = weights.take(&t2e_key)?;
        let vocab = t2e_shape.first().copied().unwrap_or(0);
        // Name the param by the CHECKPOINT KEY (not a derived `…moe.tid2eid`) so a
        // streaming ManifestParamSource re-fetches it via its Raw manifest entry.
        let t2e_node = g.param(&t2e_key, Shape::new(&[vocab, top_k], f));
        params.insert(t2e_key, t2e);
        let sel = g.gather_(t2e_node, ids, 0); // [rows, top_k]
        g.reshape_(sel, vec![rows as i64, top_k as i64])
    } else {
        g.add_node(
            Op::TopK { k: top_k },
            vec![route],
            Shape::new(&[rows, top_k], DType::F32),
        )
    };
    // Weight the selected experts by their ORIGINAL scores (not the biased ones).
    let mut top_w = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![scores, top_idx],
        Shape::new(&[rows, top_k], f),
    );
    if spec.norm_topk_prob {
        let denom = g.sum(top_w, vec![1], true);
        top_w = g.div(top_w, denom);
    }
    if (spec.routed_scaling_factor - 1.0).abs() > f32::EPSILON {
        let sc = const1(
            g,
            params,
            &format!("{lp}.moe.rscale"),
            spec.routed_scaling_factor,
        );
        top_w = g.mul(top_w, sc);
    }

    let swm = format!("{ct}.switch_mlp");
    // Each proj can carry its OWN quant scheme (mixed precision): DeepSeek-V4
    // 2bit-DQ has gate_proj at group_size 32 but up/down at 64. Using one scheme
    // for all three mis-computes n_groups for the others → the grouped kernel
    // reads scales out of bounds → garbage → activation explosion → gibberish.
    let (gate_c, gate_s, gate_b, gate_scheme, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        lp,
        &swm,
        "gate_proj",
        n,
        inter,
    )?;
    let (up_c, up_s, up_b, up_scheme, _) =
        load_stacked_group_experts_mlx(g, params, packed, weights, lp, &swm, "up_proj", n, inter)?;
    let (down_c, down_s, down_b, down_scheme, _) =
        load_stacked_group_experts_mlx(g, params, packed, weights, lp, &swm, "down_proj", n, h)?;

    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let e_col = g.narrow_(top_idx, 1, ki, 1);
        let e_idx = g.reshape_(e_col, vec![rows as i64]);
        let w_col = g.narrow_(top_w, 1, ki, 1);
        let gate = g.add_node(
            Op::DequantGroupedMatMulMlx {
                scheme: gate_scheme,
            },
            vec![h2d, gate_c, gate_s, gate_b, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let up = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme: up_scheme },
            vec![h2d, up_c, up_s, up_b, e_idx],
            Shape::new(&[rows, inter], f),
        );
        // V4 clamped-SwiGLU: up ∈ [-L, L], gate ≤ L, before silu(gate)*up.
        let (gate, up) = if spec.swiglu_limit > 0.0 {
            let l = spec.swiglu_limit;
            (g.clamp_(gate, f32::MIN, l), g.clamp_(up, -l, l))
        } else {
            (gate, up)
        };
        let gate_act = g.silu(gate);
        let glu = g.mul(gate_act, up);
        let down = g.add_node(
            Op::DequantGroupedMatMulMlx {
                scheme: down_scheme,
            },
            vec![glu, down_c, down_s, down_b, e_idx],
            Shape::new(&[rows, h], f),
        );
        let weighted = if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("noweight") {
            down
        } else {
            g.mul(down, w_col)
        };
        acc = Some(match acc {
            None => weighted,
            Some(a) => g.add(a, weighted),
        });
    }
    let routed = acc.ok_or_else(|| anyhow!("deepseek MoE: top_k=0"))?;

    // Shared experts (dense SwiGLU, `n_shared_experts * moe_inter` wide).
    let se_inter = spec.n_shared_experts * inter;
    let sg = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.{ct}.shared_experts.gate_proj.weight"),
    )?;
    let su = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.{ct}.shared_experts.up_proj.weight"),
    )?;
    let sd = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.{ct}.shared_experts.down_proj.weight"),
    )?;
    let sgate = emit_proj(g, h2d, &sg, Shape::new(&[rows, se_inter], f));
    let sup = emit_proj(g, h2d, &su, Shape::new(&[rows, se_inter], f));
    let (sgate, sup) = if spec.swiglu_limit > 0.0 {
        let l = spec.swiglu_limit;
        (g.clamp_(sgate, f32::MIN, l), g.clamp_(sup, -l, l))
    } else {
        (sgate, sup)
    };
    let sgate_act = g.silu(sgate);
    let sglu = g.mul(sgate_act, sup);
    let sdown = emit_proj(g, sglu, &sd, Shape::new(&[rows, h], f));

    let out = match std::env::var("RLX_DSV4_DBG").as_deref() {
        Ok("routed") | Ok("noweight") => routed,
        Ok("shared") => sdown,
        _ => g.add(routed, sdown),
    };
    Ok(g.reshape_(out, vec![batch as i64, seq as i64, h as i64]))
}

// ══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 Hyper-Connections (HC). The hidden state is `hc` parallel streams
// `[rows, hc, d]`. Each block: `hc_pre` mixes streams `hc→1` (Sinkhorn-normalized
// learned weights) → sublayer(1 stream) → `hc_post` expands `1→hc`. Ported from
// deepseek-ai/DeepSeek-V4-Flash `inference/{model.py,kernel.py}`. `pub` for the
// isolation probe (`examples/hc_probe.rs`).
// ══════════════════════════════════════════════════════════════════════════

/// Per-token Sinkhorn split of the `[rows, (2+hc)*hc]` mixing vector into
/// `pre [rows,hc]` (stream-reduce weights), `post [rows,hc]` (expand weights),
/// and `comb [rows,hc,hc]` (Sinkhorn-normalized combination matrix). `scale` is
/// `[3]`, `base` is `[(2+hc)*hc]`. Mirrors `hc_split_sinkhorn_kernel`.
#[allow(clippy::too_many_arguments)]
pub fn build_hc_sinkhorn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    mixes: NodeId,
    scale: NodeId,
    base: NodeId,
    rows: usize,
    hc: usize,
    eps: f32,
    iters: usize,
    tag: &str,
) -> (NodeId, NodeId, NodeId) {
    // #1: fused Sinkhorn HC gate (one Op::Custom instead of ~36 tiny Div/Reduce ops).
    if crate::dsv4_opt::opt_hcgate() {
        return crate::dsv4_opt::emit_hc_gate(g, mixes, scale, base, rows, hc, eps, iters);
    }
    let (r, h) = (rows as i64, hc as i64);
    let eps_c = const1(g, params, &format!("{tag}.hc.eps"), eps);
    let two = const1(g, params, &format!("{tag}.hc.two"), 2.0);
    let s0 = g.narrow_(scale, 0, 0, 1);
    let s1 = g.narrow_(scale, 0, 1, 1);
    let s2 = g.narrow_(scale, 0, 2, 1);
    let b_pre = g.narrow_(base, 0, 0, hc);
    let b_post = g.narrow_(base, 0, hc, hc);
    let b_comb = g.narrow_(base, 0, 2 * hc, hc * hc);
    let m_pre = g.narrow_(mixes, 1, 0, hc);
    let m_post = g.narrow_(mixes, 1, hc, hc);
    let m_comb = g.narrow_(mixes, 1, 2 * hc, hc * hc);

    // pre = sigmoid(m_pre*s0 + b_pre) + eps
    let t = g.mul(m_pre, s0);
    let t = g.add(t, b_pre);
    let t = g.sigmoid(t);
    let pre = g.add(t, eps_c);
    // post = 2*sigmoid(m_post*s1 + b_post)
    let t = g.mul(m_post, s1);
    let t = g.add(t, b_post);
    let t = g.sigmoid(t);
    let post = g.mul(t, two);
    // comb = (m_comb*s2 + b_comb) reshaped [rows,hc,hc]
    let t = g.mul(m_comb, s2);
    let t = g.add(t, b_comb);
    let comb = g.reshape_(t, vec![r, h, h]);
    // comb = softmax(-1) + eps
    let comb = g.sm(comb, -1);
    let comb = g.add(comb, eps_c);
    // comb = comb / (comb.sum(-2) + eps)   [-2 == j == axis 1]
    let d = g.sum(comb, vec![1], true);
    let d = g.add(d, eps_c);
    let mut comb = g.div(comb, d);
    for _ in 0..iters.saturating_sub(1) {
        // / (sum(-1)+eps)  [k == axis 2]
        let d = g.sum(comb, vec![2], true);
        let d = g.add(d, eps_c);
        comb = g.div(comb, d);
        // / (sum(-2)+eps)  [j == axis 1]
        let d = g.sum(comb, vec![1], true);
        let d = g.add(d, eps_c);
        comb = g.div(comb, d);
    }
    (pre, post, comb)
}

/// HC pre-mix: RMS-normalize the flattened streams, project to the mixing vector
/// (`hc_fn_t` is the transposed `[hc*d, (2+hc)*hc]` mix weight), Sinkhorn-split,
/// then reduce `hc` streams → 1 via `Σ_hc pre·x`. Returns `(y [rows,d], post,
/// comb)` — `post`/`comb` feed [`build_hc_post`].
#[allow(clippy::too_many_arguments)]
pub fn build_hc_pre(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    hc_fn_t: NodeId,
    scale: NodeId,
    base: NodeId,
    rows: usize,
    hc: usize,
    d: usize,
    eps: f32,
    iters: usize,
    tag: &str,
) -> (NodeId, NodeId, NodeId) {
    let hcd = hc * d;
    let x_flat = g.reshape_(x, vec![rows as i64, hcd as i64]);
    let sq = g.mul(x_flat, x_flat);
    let ms = g.mean(sq, vec![1], true);
    let eps_c = const1(g, params, &format!("{tag}.hcp.eps"), eps);
    let ms = g.add(ms, eps_c);
    let rsq = g.rsqrt(ms); // [rows,1]
    let mixes = g.mm(x_flat, hc_fn_t); // [rows, mix_hc]
    let mixes = g.mul(mixes, rsq);
    let (pre, post, comb) =
        build_hc_sinkhorn(g, params, mixes, scale, base, rows, hc, eps, iters, tag);
    // y = Σ_hc pre·x
    let pre3 = g.reshape_(pre, vec![rows as i64, hc as i64, 1]);
    let yh = g.mul(pre3, x); // [rows,hc,d]
    let y = g.sum(yh, vec![1], false); // [rows,d]
    (y, post, comb)
}

/// HC head-reduce (final `hc→1`): like `build_hc_pre` but a plain sigmoid gate
/// (no Sinkhorn / post / comb). `y = Σ_hc (sigmoid(mix·scale + base) + eps)·x`,
/// `hc_fn_t` is transposed `[hc*d, hc]`, `scale` is `[1]`, `base` is `[hc]`.
/// Mirrors `ParallelHead.hc_head`.
#[allow(clippy::too_many_arguments)]
pub fn build_hc_head(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    hc_fn_t: NodeId,
    scale: NodeId,
    base: NodeId,
    rows: usize,
    hc: usize,
    d: usize,
    eps: f32,
    tag: &str,
) -> NodeId {
    let hcd = hc * d;
    let x_flat = g.reshape_(x, vec![rows as i64, hcd as i64]);
    let sq = g.mul(x_flat, x_flat);
    let ms = g.mean(sq, vec![1], true);
    let eps_c = const1(g, params, &format!("{tag}.hch.eps"), eps);
    let ms = g.add(ms, eps_c);
    let rsq = g.rsqrt(ms);
    let mixes = g.mm(x_flat, hc_fn_t); // [rows, hc]
    let mixes = g.mul(mixes, rsq);
    let t = g.mul(mixes, scale);
    let t = g.add(t, base);
    let t = g.sigmoid(t);
    let pre = g.add(t, eps_c); // [rows, hc]
    let pre3 = g.reshape_(pre, vec![rows as i64, hc as i64, 1]);
    let yh = g.mul(pre3, x); // [rows,hc,d]
    g.sum(yh, vec![1], false) // [rows,d]
}

/// HC post-expand: `1→hc` streams. `y[j] = post[j]·x_out + Σ_k comb[j,k]·residual[k]`.
/// `x_out [rows,d]` is the sublayer output; `residual [rows,hc,d]` the block input.
pub fn build_hc_post(
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
    let term1 = g.mul(post3, xo3); // [rows,hc,d]
    let comb4 = g.reshape_(comb, vec![r, h, h, 1]);
    let res4 = g.reshape_(residual, vec![r, 1, h, dd]);
    let prod = g.mul(comb4, res4); // [rows,hc,k,d]
    let term2 = g.sum(prod, vec![2], false); // Σ_k → [rows,hc,d]
    g.add(term1, term2)
}

/// DeepSeek-V4 **KV Compressor** core (subsystem #4) — the learned gated-pooling
/// that summarizes each `ratio`-token window into one compressed KV: `score +=
/// APE`, softmax over the window, `Σ (kv · softmax(score))`, then RMSNorm. This
/// is the non-overlap (`ratio != 4`) prefill path; the FP4 / Hadamard quant-sim
/// (`fp4_act_quant`/`rotate_activation`) is precision-only and omitted, and the
/// decoupled RoPE on the compressed tail is applied by the caller. `kv`/`score`
/// are `wkv(x)`/`wgate(x)` `[b*s, hd]` (whole windows, `s % ratio == 0`). Mirrors
/// `Compressor.forward` (deepseek-ai/DeepSeek-V4-Flash). `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_kv_compressor_pool(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    kv: NodeId,
    score: NodeId,
    ape: NodeId,
    norm_w: NodeId,
    b: usize,
    s: usize,
    ratio: usize,
    hd: usize,
    eps: f32,
    tag: &str,
) -> NodeId {
    let nwin = s / ratio;
    let (bb, nw, r, d) = (b as i64, nwin as i64, ratio as i64, hd as i64);
    let kv4 = g.reshape_(kv, vec![bb, nw, r, d]);
    let sc4 = g.reshape_(score, vec![bb, nw, r, d]);
    let ape4 = g.reshape_(ape, vec![1, 1, r, d]);
    let sc4 = g.add(sc4, ape4);
    // softmax over the window axis (2). g.sm normalizes the last axis, so move
    // `ratio` last, softmax, move back.
    let sct = g.transpose_(sc4, vec![0, 1, 3, 2]); // [b,nwin,hd,ratio]
    let w = g.sm(sct, -1);
    let w = g.transpose_(w, vec![0, 1, 3, 2]); // [b,nwin,ratio,hd]
    let pooled = g.mul(kv4, w);
    let pooled = g.sum(pooled, vec![2], false); // [b,nwin,hd]
    let pooled2 = g.reshape_(pooled, vec![(b * nwin) as i64, d]);
    let zb = synth_zero(g, params, &format!("{tag}.comp.zb"), hd);
    g.rms_norm(pooled2, norm_w, zb, eps)
}

/// DeepSeek-V4 **overlapping KV Compressor** (subsystem #4, the `ratio == 4`
/// path) — the `overlap=True` form of `Compressor.forward`
/// (deepseek-ai/DeepSeek-V4-Flash). `wkv`/`wgate` output `coff*hd = 2*hd`; the
/// first `hd` dims are the *overlapping-window* contribution, the last `hd` the
/// *current-window* contribution. `overlap_transform` builds, per compressed
/// window, a `2*ratio` candidate set: positions `[ratio, 2*ratio)` = the current
/// window's `ratio` tokens (their **second** dim-half), positions `[0, ratio)` =
/// the **previous** window's `ratio` tokens (their **first** dim-half, shifted;
/// window 0's previous part is masked `-inf`). Softmax over the `2*ratio` axis
/// (per dim), weighted sum, RMSNorm. Prefill only (`start_pos == 0`,
/// `s % ratio == 0`); FP4/Hadamard omitted (precision-sim / orthogonal-cancels).
/// `kv2`/`score2` are `wkv(x)`/`wgate(x)` `[s, 2*hd]`, `ape` is `[ratio, 2*hd]`.
/// Returns compressed KV `[s/ratio, hd]`. `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_kv_compressor_overlap(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    kv2: NodeId,    // [s, 2*hd]
    score2: NodeId, // [s, 2*hd]
    ape: NodeId,    // [ratio, 2*hd]
    norm_w: NodeId, // [hd]
    s: usize,
    ratio: usize,
    hd: usize,
    eps: f32,
    tag: &str,
) -> NodeId {
    let nwin = s / ratio;
    let (nw, r, d, d2) = (nwin as i64, ratio as i64, hd as i64, (2 * hd) as i64);
    let neg = -1e30f32;
    // [nwin, ratio, 2*hd] + APE (broadcast over windows).
    let kv4 = g.reshape_(kv2, vec![nw, r, d2]);
    let sc4 = g.reshape_(score2, vec![nw, r, d2]);
    let ape3 = g.reshape_(ape, vec![1, r, d2]);
    let sc4 = g.add(sc4, ape3);
    // Split the 2*hd dims: first half = overlap contribution, second = current.
    let kv_first = g.narrow_(kv4, 2, 0, hd); // [nwin, ratio, hd]
    let kv_second = g.narrow_(kv4, 2, hd, hd);
    let sc_first = g.narrow_(sc4, 2, 0, hd);
    let sc_second = g.narrow_(sc4, 2, hd, hd);
    // Previous-window shift: prepend one padded window, drop the last. Window 0's
    // "previous" part is 0 for KV and -inf for score (masked out of the softmax).
    let (kv_prev, sc_prev) = if nwin > 1 {
        let kv_head = g.narrow_(kv_first, 0, 0, nwin - 1); // [nwin-1, ratio, hd]
        let sc_head = g.narrow_(sc_first, 0, 0, nwin - 1);
        let kz = synth_const(
            g,
            params,
            &format!("{tag}.ov.kz"),
            vec![0f32; ratio * hd],
            &[1, ratio, hd],
        );
        let sz = synth_const(
            g,
            params,
            &format!("{tag}.ov.sz"),
            vec![neg; ratio * hd],
            &[1, ratio, hd],
        );
        (
            g.concat_(vec![kz, kv_head], 0),
            g.concat_(vec![sz, sc_head], 0),
        )
    } else {
        let kz = synth_const(
            g,
            params,
            &format!("{tag}.ov.kz1"),
            vec![0f32; ratio * hd],
            &[1, ratio, hd],
        );
        let sz = synth_const(
            g,
            params,
            &format!("{tag}.ov.sz1"),
            vec![neg; ratio * hd],
            &[1, ratio, hd],
        );
        (kz, sz)
    };
    // Stack → [nwin, 2*ratio, hd] : [prev first-half | current second-half].
    let kv_stack = g.concat_(vec![kv_prev, kv_second], 1);
    let sc_stack = g.concat_(vec![sc_prev, sc_second], 1);
    // Softmax over the 2*ratio axis (axis 1), per (window, dim).
    let sct = g.transpose_(sc_stack, vec![0, 2, 1]); // [nwin, hd, 2ratio]
    let w = g.sm(sct, -1);
    let w = g.transpose_(w, vec![0, 2, 1]); // [nwin, 2ratio, hd]
    let pooled = g.mul(kv_stack, w);
    let pooled = g.sum(pooled, vec![1], false); // [nwin, hd]
    let pooled2 = g.reshape_(pooled, vec![nw, d]);
    let zb = synth_zero(g, params, &format!("{tag}.ov.zb"), hd);
    g.rms_norm(pooled2, norm_w, zb, eps)
}

/// DeepSeek-V4 **Indexer scoring core** (subsystem #5) — the learned relevance
/// score `index_score[s,t]` that ranks compressed KV position `t` for query `s`,
/// from `Indexer.forward` (deepseek-ai/DeepSeek-V4-Flash). `q = rope(wq_b(qr))`
/// reshaped to `[s, n_heads, head_dim]`; `index_score = Σ_h relu(⟨q[s,h], kv[t]⟩)
/// · weights[s,h]` where `weights = weights_proj(x) · (hd^-0.5 · n_heads^-0.5)`.
/// The reference's `rotate_activation` (Hadamard) is **orthogonal**, so it cancels
/// in the `q·kv` inner product; `fp4_act_quant` is precision-only — both omitted
/// for the F32-exact score. `qr` is `q_norm(wq_a(x))` `[s, q_lora_rank]`, `wq_b`
/// is `[q_lora_rank, n_heads*head_dim]` and `weights_proj` `[dim, n_heads]` (both
/// mm-ready), `kv_comp` the indexer's compressed KV `[ncomp, head_dim]`. Returns
/// `index_score [s, ncomp]`. `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_v4_indexer_score(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    qr: NodeId,           // [s, q_lora_rank]
    x: NodeId,            // [s, dim]
    kv_comp: NodeId,      // [ncomp, head_dim]
    wq_b: NodeId,         // [q_lora_rank, n_heads*head_dim]
    weights_proj: NodeId, // [dim, n_heads]
    cos_id: NodeId,
    sin_id: NodeId,
    seq: usize,
    nh: usize,
    hd: usize,
    rd: usize,
    ncomp: usize,
    tag: &str,
) -> NodeId {
    let (sq, n, d, nc) = (seq as i64, nh as i64, hd as i64, ncomp as i64);
    let q = g.mm(qr, wq_b); // [s, nh*hd]
    let q = g.rope_n_styled(q, cos_id, sin_id, hd, rd, RopeStyle::NeoX);
    let q3 = g.reshape_(q, vec![(seq * nh) as i64, d]); // [s*nh, hd]
    let kvt = g.transpose_(kv_comp, vec![1, 0]); // [hd, ncomp]
    let sc = g.mm(q3, kvt); // [s*nh, ncomp]
    let sc = g.relu(sc);
    let sc = g.reshape_(sc, vec![sq, n, nc]); // [s, nh, ncomp]
    // weights = weights_proj(x) * (hd^-0.5 * nh^-0.5)  → [s, nh, 1]
    let w = g.mm(x, weights_proj); // [s, nh]
    let scale = (hd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let sc_c = synth_const(g, params, &format!("{tag}.idx.scale"), vec![scale], &[1, 1]);
    let w = g.mul(w, sc_c);
    let w = g.reshape_(w, vec![sq, n, 1]);
    let wsum = g.mul(sc, w); // [s, nh, ncomp]
    g.sum(wsum, vec![1], false) // [s, ncomp]
}

/// DeepSeek-V4 **Indexer top-k gate** (subsystem #5) — turns the dense
/// `index_score [s, ncomp]` + a deterministic `causal_add` mask (`0` where
/// compressed position `t` is causally visible to query `s`, large-negative
/// otherwise) into an additive attention mask that keeps only each query's
/// **top-`k`** compressed positions, matching `index_score.topk(k)` +
/// the causal re-mask in `Indexer.forward`. `k = min(index_topk, ncomp)`. For
/// prefills where every causally-valid count `≤ k` the top-k is a no-op (all
/// valid kept) — the reason the earlier deterministic all-valid mask is exact for
/// sequences up to `index_topk*ratio`; this generalizes to longer context.
/// Threshold = min of the gathered top-`k` scores; the gate is realized
/// arithmetically (`clamp((score-thr)·BIG, .., 0)`) to stay backend-portable and
/// NaN-free (finite `-1e30` mask, not `-inf`). `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_v4_topk_gate(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    idx_score: NodeId,  // [s, ncomp]
    causal_add: NodeId, // [s, ncomp]
    seq: usize,
    ncomp: usize,
    k: usize,
    tag: &str,
) -> NodeId {
    debug_assert!(k <= ncomp, "top-k gate: k={k} must be <= ncomp={ncomp}");
    let f = DType::F32;
    let score_masked = g.add(idx_score, causal_add); // [s, ncomp]
    let top_idx = g.add_node(Op::TopK { k }, vec![score_masked], Shape::new(&[seq, k], f));
    let top_vals = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![score_masked, top_idx],
        Shape::new(&[seq, k], f),
    );
    let thr = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Min,
            axes: vec![1],
            keep_dim: true,
        },
        vec![top_vals],
        Shape::new(&[seq, 1], f),
    );
    let diff = g.sub(score_masked, thr); // broadcast [s,ncomp]-[s,1]
    let big = synth_const(g, params, &format!("{tag}.tk.big"), vec![1e6f32], &[1, 1]);
    let scaled = g.mul(diff, big);
    let gate = g.clamp_(scaled, f32::MIN, 0.0); // 0 where score≥thr (top-k), very-neg else
    // Re-apply causal so masked ties (thr = -1e30 when valid_count ≤ k) stay dropped.
    g.add(gate, causal_add)
}

/// DeepSeek-V4 **sparse-window attention core** (subsystem #3) — the dense-masked
/// correctness form of the reference `sparse_attn`: MQA latent attention where
/// the single `kv [n_keys, head_dim]` serves as both key and value (shared across
/// heads), a per-head learned **`attn_sink`** logit in the softmax denominator,
/// and an additive `mask [rows, n_keys]` realizing the sliding-window +
/// compression selection (`0` allowed, large-negative disallowed). The custom
/// `sparse_attn` kernel only skips the masked positions for efficiency; this
/// computes the identical result densely. Sink is applied by appending it as an
/// extra softmax column then dropping it (steals mass, no value). Returns
/// `o [rows, n_heads, head_dim]`. `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_v4_sink_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    q: NodeId,
    kv: NodeId,
    mask: NodeId,
    sink: NodeId,
    scale: f32,
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    n_keys: usize,
    tag: &str,
) -> NodeId {
    // #2: fused sink-attention (one Op::Custom instead of Softmax + 2 MatMul + views).
    if crate::dsv4_opt::opt_sinkattn() {
        return crate::dsv4_opt::emit_sink_attention(
            g, q, kv, mask, sink, scale, rows, n_heads, head_dim, n_keys,
        );
    }
    let (r, nh, hd, nk) = (rows as i64, n_heads as i64, head_dim as i64, n_keys as i64);
    // scores = (q @ kvᵀ) * scale  → [rows, nh, nk]
    let q2 = g.reshape_(q, vec![(rows * n_heads) as i64, hd]);
    let kv_t = g.transpose_(kv, vec![1, 0]); // [head_dim, n_keys]
    let sc = g.mm(q2, kv_t); // [rows*nh, nk]
    let sc = g.reshape_(sc, vec![r, nh, nk]);
    let sca = const1(g, params, &format!("{tag}.sink.scale"), scale);
    let sc = g.mul(sc, sca);
    // + additive window/compression mask [rows,1,nk]
    let mask3 = g.reshape_(mask, vec![r, 1, nk]);
    let sc = g.add(sc, mask3);
    // append per-head sink as an extra logit column → [rows,nh,nk+1]
    let sink_r = g.reshape_(sink, vec![1, nh, 1]);
    let zc = register_zeros(g, params, &format!("{tag}.sink.zc"), &[rows, n_heads, 1]);
    let sink_col = g.add(zc, sink_r); // [rows,nh,1]
    let sc_ext = g.concat_(vec![sc, sink_col], 2);
    let attn_ext = g.sm(sc_ext, -1);
    let attn = g.narrow_(attn_ext, 2, 0, n_keys); // drop sink column
    // o = attn @ kv  (MQA: shared kv) → [rows,nh,hd]
    let attn2 = g.reshape_(attn, vec![(rows * n_heads) as i64, nk]);
    let o = g.mm(attn2, kv); // [rows*nh, hd]
    g.reshape_(o, vec![r, nh, hd])
}

/// DeepSeek-V4 **grouped o-LoRA output projection** (subsystem #3) — the
/// low-rank output map after sparse-window attention. The per-head attention
/// output `[rows, n_heads*v_head_dim]` is split into `n_groups` groups
/// (`dpg = n_heads*v_head_dim / n_groups` each); each group is projected down by
/// its own `wo_a[g] : [o_lora_rank, dpg]` (the reference `einsum
/// "bsgd,grd->bsgr"`), the groups concatenated to `[rows, n_groups*o_lora_rank]`,
/// then `wo_b` maps up to `dim`. `wo_a` is a `[n_groups, o_lora_rank, dpg]`
/// param; `wo_b_t` is the transposed `[n_groups*o_lora_rank, dim]` weight (for
/// `g.mm`). Mirrors `Attention.forward`'s output block. `pub` for the probe.
#[allow(clippy::too_many_arguments)]
pub fn build_v4_o_lora(
    g: &mut Graph,
    o: NodeId,
    wo_a: NodeId,
    wo_b_t: NodeId,
    rows: usize,
    n_groups: usize,
    o_lora_rank: usize,
    dpg: usize,
    dim: usize,
) -> NodeId {
    let r = rows as i64;
    let mut groups: Vec<NodeId> = Vec::with_capacity(n_groups);
    for grp in 0..n_groups {
        let o_g = g.narrow_(o, 1, grp * dpg, dpg); // [rows, dpg]
        let wa = g.narrow_(wo_a, 0, grp, 1); // [1, o_lora, dpg]
        let wa = g.reshape_(wa, vec![o_lora_rank as i64, dpg as i64]);
        let wa_t = g.transpose_(wa, vec![1, 0]); // [dpg, o_lora]
        groups.push(g.mm(o_g, wa_t)); // [rows, o_lora]
    }
    let cat = g.concat_(groups, 1); // [rows, n_groups*o_lora]
    let cat = g.reshape_(cat, vec![r, (n_groups * o_lora_rank) as i64]);
    let out = g.mm(cat, wo_b_t); // [rows, dim]
    g.reshape_(out, vec![r, dim as i64])
}

// ─────────────────────────────────────────────────────────────────────────────
// DSpark — the GA (`DeepSeek-V4-Flash-0731`) integrated speculative-decoding
// module (stored under the checkpoint `mtp.*` namespace). The main forward is
// correct without it (spec decoding only accelerates the autoregressive loop by
// drafting `dspark_block_size` tokens per step and verifying them against the
// main model). These builders assemble the DSpark stages; each is validated
// cos-exact against the reference `model.py` (`DSparkAttention`/`DSparkMarkovHead`/
// `DSparkConfidenceHead`/`DSparkBlock`). Reused pieces: `build_hc_pre`/`_post`/
// `_head`, `build_v4_sink_attention`, `build_v4_o_lora`, `build_deepseek_moe_ffn`.
// ─────────────────────────────────────────────────────────────────────────────

/// DSpark **Markov head** (`DSparkMarkovHead`) — a bigram-style per-token logit
/// bias. `markov_w1` is a `[vocab, markov_rank]` embedding; `markov_w2` a
/// `[vocab, markov_rank]` head. For token ids `[rows]` it returns
/// `(logits[rows, vocab] = embed @ markov_w2ᵀ, embed[rows, markov_rank])`. The
/// caller adds `logits` into the draft logits and feeds `embed` to the confidence
/// head. `markov_w2_t` is the transposed `[markov_rank, vocab]` weight (for
/// `g.mm`). Mirrors `DSparkMarkovHead.forward`. `pub` for the probe.
pub fn build_dspark_markov_head(
    g: &mut Graph,
    markov_w1: NodeId,
    markov_w2_t: NodeId,
    token_ids: NodeId,
    rows: usize,
    markov_rank: usize,
    vocab: usize,
) -> (NodeId, NodeId) {
    let ids2 = g.reshape_(token_ids, vec![1, rows as i64]);
    let embed = g.gather_(markov_w1, ids2, 0); // [1, rows, markov_rank]
    let embed = g.reshape_(embed, vec![rows as i64, markov_rank as i64]);
    let logits = g.mm(embed, markov_w2_t); // [rows, vocab]
    let logits = g.reshape_(logits, vec![rows as i64, vocab as i64]);
    (logits, embed)
}

/// DSpark **confidence head** (`DSparkConfidenceHead`) — a scalar acceptance score
/// per draft position: `proj(concat(hidden, markov_embed))`. `hidden` is
/// `[rows, dim]`, `markov_embed` `[rows, markov_rank]`, `proj_t` the transposed
/// `[dim+markov_rank, 1]` weight. Returns `[rows]`. Mirrors
/// `DSparkConfidenceHead.forward`. `pub` for the probe.
pub fn build_dspark_confidence_head(
    g: &mut Graph,
    hidden: NodeId,
    markov_embed: NodeId,
    proj_t: NodeId,
    rows: usize,
) -> NodeId {
    let cat = g.concat_(vec![hidden, markov_embed], 1); // [rows, dim+markov_rank]
    let conf = g.mm(cat, proj_t); // [rows, 1]
    g.reshape_(conf, vec![rows as i64])
}

/// Build a **DSpark speculative-decode stage** graph (`Transformer.forward_spec`
/// decode path) for one draft step. Given the main model's target-layer hiddens
/// (`main_hidden [cache_len, dim*n_targets]`) and the `block_size` draft token ids,
/// it: projects `main_x = main_norm(main_proj(main_hidden))` (stage-0), builds the
/// per-stage `DSparkBlock` (HC-wrapped sliding-window `DSparkAttention` where the
/// `block_size` draft queries attend to `[main_kv window ++ block_kv]` with a
/// per-head sink, then HC-wrapped sqrtsoftplus MoE), reduces with `hc_head`,
/// normalizes, and applies the (shared) `lm_head`. Returns a graph with outputs
/// `[block_logits [block, vocab], head_hidden [block, dim]]`. The per-draft Markov
/// logit bias + confidence are applied host-side ([`build_dspark_markov_head`] /
/// [`build_dspark_confidence_head`]) because they are autoregressive over samples.
/// DSpark layers are pure sliding-window (`compress_ratio == 0`), so RoPE uses the
/// raw `rope_theta` base (no YaRN). Weights live under `model.mtp.{stage}.*`
/// (checkpoint `mtp.*` namespace); `lm_head`/`embed_tokens` are shared with the
/// main model. Mirrors `DSparkBlock`/`DSparkAttention` (deepseek-ai GA `model.py`).
pub fn build_dspark_stage(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    block_size: usize,
    cache_len: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("dspark_stage");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let d = spec.dim;
    let hc = spec.hc_mult;
    let nh = spec.n_heads;
    let hd = spec.head_dim;
    let rd = spec.rope_head_dim & !1;
    let ql = spec.q_lora_rank;
    let eps = spec.rms_norm_eps;
    let block = block_size;
    let n_targets = spec.dspark_target_layer_ids.len().max(1);
    let n_stages = spec.n_mtp_layers.max(1);
    let scale = (hd as f32).powf(-0.5);
    let half = (rd / 2).max(1);
    let zb_d = synth_zero(&mut g, &mut params, "ds.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "ds.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "ds.zb.hd", hd);
    let q_ones = synth_const(&mut g, &mut params, "ds.qones", vec![1f32; hd], &[hd]);

    // Sliding RoPE tables (base `rope_theta`, no YaRN — DSpark layers have ratio 0).
    // `cache` positions 0..cache_len; `block` positions cache_len..cache_len+block.
    let rope = |g: &mut Graph,
                params: &mut HashMap<String, Vec<f32>>,
                start: usize,
                n: usize,
                tag: &str| {
        let (mut cosd, mut sind) = (vec![0f32; n * half], vec![0f32; n * half]);
        for p in 0..n {
            for i in 0..half {
                let fr = (spec.rope_theta).powf(-(2.0 * i as f64) / rd as f64);
                let (s, c) = ((start + p) as f64 * fr).sin_cos();
                cosd[p * half + i] = c as f32;
                sind[p * half + i] = s as f32;
            }
        }
        let sinneg: Vec<f32> = sind.iter().map(|v| -v).collect();
        let cos = synth_const(g, params, &format!("ds.cos.{tag}"), cosd, &[n, half]);
        let sin = synth_const(g, params, &format!("ds.sin.{tag}"), sind, &[n, half]);
        let sinv = synth_const(g, params, &format!("ds.sinv.{tag}"), sinneg, &[n, half]);
        (cos, sin, sinv)
    };
    let (cos_c, sin_c, _) = rope(&mut g, &mut params, 0, cache_len, "cache");
    let (cos_b, sin_b, sinv_b) = rope(&mut g, &mut params, cache_len, block, "block");

    // Draft embeds → HC-expand to `hc` streams. `draft_ids` = [input_id, noise…].
    let draft_ids = g.input("draft_ids", Shape::new(&[block], DType::I32));
    let draft_ids2 = g.reshape_(draft_ids, vec![1, block as i64]);
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let xe = g.gather_(embed_w, draft_ids2, 0); // [1, block, d]
    let xe = g.reshape_(xe, vec![block as i64, 1, d as i64]);
    let ones_hc = synth_const(
        &mut g,
        &mut params,
        "ds.hc.ones",
        vec![1f32; hc],
        &[1, hc, 1],
    );
    let mut h = g.mul(xe, ones_hc); // [block, hc, d]

    // main_x = main_norm(main_proj(main_hidden)); main_hidden [cache_len, dim*n_targets].
    let main_hidden = g.input(
        "main_hidden",
        Shape::new(&[cache_len, d * n_targets], DType::F32),
    );
    let mp = load_transposed_param(&mut g, &mut params, weights, "model.mtp.0.main_proj.weight")?;
    let main_x = g.mm(main_hidden, mp); // [cache_len, d]
    let mn = load_norm(
        &mut g,
        &mut params,
        weights,
        "model.mtp.0.main_norm.weight",
        0.0,
    )?;
    let main_x = g.rms_norm(main_x, mn, zb_d, eps);

    for stage in 0..n_stages {
        let lp = format!("model.mtp.{stage}");
        // Window KV cache = rope(kv_norm(wkv(main_x))) at cache positions.
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let ckv = emit_proj(&mut g, main_x, &wkv, Shape::new(&[cache_len, hd], f));
        let kvn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let ckv = g.rms_norm(ckv, kvn, zb_hd, eps);
        let cache_kv = rope_tail(&mut g, ckv, cos_c, sin_c, cache_len, 1, hd, rd);

        // ── HC-wrapped DSpark attention ──
        let residual = h;
        let fn_a =
            load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn_hc.fn"))?;
        let sc_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.scale"),
            false,
        )?;
        let bs_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.base"),
            false,
        )?;
        let (xa, post_a, comb_a) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            block,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.a"),
        );
        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_norm.weight"),
            0.0,
        )?;
        let xa = g.rms_norm(xa, an, zb_d, eps);
        // q-LoRA + per-head norm + block RoPE
        let wqa = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_a.weight"),
        )?;
        let qr = emit_proj(&mut g, xa, &wqa, Shape::new(&[block, ql], f));
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.q_norm.weight"),
            0.0,
        )?;
        let qr = g.rms_norm(qr, qn, zb_ql, eps);
        let wqb = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_b.weight"),
        )?;
        let q = emit_proj(&mut g, qr, &wqb, Shape::new(&[block, nh * hd], f));
        let q = per_head_rms(&mut g, q, q_ones, zb_hd, 1, block, nh, hd, eps);
        let q = rope_tail(&mut g, q, cos_b, sin_b, block, nh, hd, rd);
        // block KV from the draft tokens
        let bkv = emit_proj(&mut g, xa, &wkv, Shape::new(&[block, hd], f));
        let bkv = g.rms_norm(bkv, kvn, zb_hd, eps);
        let bkv = rope_tail(&mut g, bkv, cos_b, sin_b, block, 1, hd, rd);
        // kv_all = [window cache ++ block]; sliding mask (all cache + all block valid)
        let kv_all = g.concat_(vec![cache_kv, bkv], 0); // [cache_len+block, hd]
        let n_keys = cache_len + block;
        let win = spec.window_size.max(1);
        let neg = -1e30f32;
        let mut md = vec![0f32; block * n_keys];
        for qi in 0..block {
            for ki in 0..n_keys {
                // window cache positions within `win` of the query are visible; all
                // block positions are visible (DSpark parallel-draft block attention).
                let vis = if ki < cache_len {
                    cache_len + qi < ki + win // qi absolute = cache_len+qi ; key = ki
                } else {
                    true
                };
                md[qi * n_keys + ki] = if vis { 0.0 } else { neg };
            }
        }
        let mask = synth_const(
            &mut g,
            &mut params,
            &format!("{lp}.ds.mask"),
            md,
            &[block, n_keys],
        );
        let sink = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.attn_sink"),
            false,
        )?;
        let q3 = g.reshape_(q, vec![block as i64, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            scale,
            block,
            nh,
            hd,
            n_keys,
            &format!("{lp}.sa"),
        );
        let o_flat = g.reshape_(o, vec![block as i64, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos_b, sinv_b, block, nh, hd, rd);
        let dpg = nh * hd / spec.n_groups;
        let woa = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wob = load_transposed_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_b.weight"),
        )?;
        let attn_out = build_v4_o_lora(
            &mut g,
            o_inv,
            woa,
            wob,
            block,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        h = build_hc_post(&mut g, attn_out, residual, post_a, comb_a, block, hc, d);

        // ── HC-wrapped sqrtsoftplus MoE FFN ──
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.ffn_hc.fn"))?;
        let sc_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.scale"),
            false,
        )?;
        let bs_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.base"),
            false,
        )?;
        let (xf, post_f, comb_f) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            block,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.f"),
        );
        let fnorm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ds = v4_moe_spec(spec);
        let x3 = g.reshape_(xf, vec![1, block as i64, d as i64]);
        let moe = build_deepseek_moe_ffn(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            x3,
            1,
            block,
            &ds,
            None,
        )?;
        let moe = g.reshape_(moe, vec![block as i64, d as i64]);
        h = build_hc_post(&mut g, moe, residual, post_f, comb_f, block, hc, d);
    }

    // Final stage: hc_head reduce → norm → shared lm_head → block logits.
    let last = n_stages - 1;
    let lp = format!("model.mtp.{last}");
    let hfn = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.hc_head.fn"))?;
    let hsc = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{lp}.hc_head.scale"),
        false,
    )?;
    let hbs = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{lp}.hc_head.base"),
        false,
    )?;
    let x = build_hc_head(
        &mut g,
        &mut params,
        h,
        hfn,
        hsc,
        hbs,
        block,
        hc,
        d,
        spec.hc_eps,
        "ds.head",
    );
    let nrm = load_norm(
        &mut g,
        &mut params,
        weights,
        &format!("{lp}.norm.weight"),
        0.0,
    )?;
    let x = g.rms_norm(x, nrm, zb_d, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[block, spec.vocab_size], f));
    let logits = g.reshape_(logits, vec![block as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits, x]);
    Ok((g, params))
}

/// DSpark **draft head loop** (`DSparkBlock.forward_head`) — the host-side
/// autoregressive step that turns the DSpark stage's block logits + hidden into
/// `block_size` draft tokens plus a per-position confidence. Starting from the
/// verified `input_id`, each position `i` adds the Markov bigram bias
/// ([`build_dspark_markov_head`]) for the current token to `block_logits[i]`,
/// greedily picks the next draft token (`temperature == 0`), and records the
/// Markov embedding; the confidence head ([`build_dspark_confidence_head`]) then
/// scores each `(head_hidden[i], markov_embed[i])`. Returns
/// `(output_ids[block+1], confidence[block])`. Greedy only (the reference's
/// Gumbel `sample` for `temperature > 0` needs an RNG; greedy == argmax matches
/// the verifier's greedy accept below). Pure f32, mirrors `model.py`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_forward_head(
    block_logits: &[f32],    // [block, vocab]
    markov_w1: &[f32],       // [vocab, markov_rank]
    markov_w2: &[f32],       // [vocab, markov_rank]
    head_hidden: &[f32],     // [block, dim]
    confidence_proj: &[f32], // [dim + markov_rank]
    input_id: usize,
    block: usize,
    vocab: usize,
    markov_rank: usize,
    dim: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut output_ids = vec![0usize; block + 1];
    output_ids[0] = input_id;
    let mut confidence = vec![0f32; block];
    for i in 0..block {
        let tok = output_ids[i];
        // Markov bias: logits_bias[v] = Σ_k w1[tok,k]·w2[v,k].
        let embed = &markov_w1[tok * markov_rank..(tok + 1) * markov_rank];
        let mut best = (0usize, f32::MIN);
        for v in 0..vocab {
            let mut bias = 0f32;
            for k in 0..markov_rank {
                bias += embed[k] * markov_w2[v * markov_rank + k];
            }
            let l = block_logits[i * vocab + v] + bias;
            if l > best.1 {
                best = (v, l);
            }
        }
        output_ids[i + 1] = best.0;
        // confidence = proj · concat(head_hidden[i], markov_embed[i]).
        let mut c = 0f32;
        for j in 0..dim {
            c += head_hidden[i * dim + j] * confidence_proj[j];
        }
        for k in 0..markov_rank {
            c += embed[k] * confidence_proj[dim + k];
        }
        confidence[i] = c;
    }
    (output_ids, confidence)
}

/// **DSpark drafter wiring** — turns the main model's accumulated target-layer
/// hiddens into `n_draft` speculative tokens for the verify/accept loop. It builds +
/// runs the DSpark stage ([`build_dspark_stage`] — the `mtp.*` sub-model over
/// `main_hidden [cache_len, dim*n_targets]` and the block draft ids), then applies
/// the host-side autoregressive Markov-bigram + confidence head
/// ([`dspark_forward_head`]) to produce `[input_id, d₁, …]`; the drafted tail
/// (`d₁..`, up to `n_draft`) is returned. This is the `draft_fn` for
/// [`deepseek_v4_generate_speculative`] /
/// [`deepseek_v4_generate_speculative_paged_layerwise`] — and losslessness does NOT
/// depend on draft quality (a wrong guess is rejected by the verifier), so a weak or
/// mis-trained drafter is still correct, just slower. The main model exposes
/// `main_hidden` via [`V4Decoder::main_hidden`] at each accepted position (the driver
/// accumulates the `[cache_len, dim*n_targets]` history). The stage is (re)built per
/// call since `cache_len` grows; amortizing that (fixed max-cache stage + mask, or a
/// per-`cache_len` compile cache) is a perf follow-up, not a correctness one.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_dspark_draft(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    device: rlx_runtime::Device,
    main_hidden: &[f32], // [cache_len, dim*n_targets]
    input_id: u32,
    n_draft: usize,
) -> Result<Vec<u32>> {
    let d = spec.dim;
    let vocab = spec.vocab_size;
    let block = spec.dspark_block_size.max(1);
    let n_targets = spec.dspark_target_layer_ids.len().max(1);
    let markov_rank = spec.dspark_markov_rank.max(1);
    let last = spec.n_mtp_layers.max(1) - 1;
    let cache_len = main_hidden.len() / (d * n_targets);
    // Build + compile the DSpark stage for this cache length.
    let (g, params) = build_dspark_stage(spec, weights, block, cache_len, packed)?;
    let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        device,
    );
    let mut sess = rlx_runtime::Session::new(device).compile_with(g, &opts);
    for (n, dd) in &params {
        sess.set_param(n, dd);
    }
    // Bind the packed (MXFP4) expert codes emitted into `packed` during the build.
    for (n, (bytes, _scheme, _shape)) in packed.iter() {
        sess.set_param_typed(n, bytes, DType::U8);
    }
    // draft_ids = [input_id, noise…]; the head fills the tail autoregressively.
    let mut draft_ids = vec![input_id as f32];
    draft_ids.resize(block, spec.dspark_noise_token_id as f32);
    let out = sess.run(&[
        ("draft_ids", draft_ids.as_slice()),
        ("main_hidden", main_hidden),
    ]);
    // set_outputs order in build_dspark_stage: [logits [block,vocab], head_hidden [block,dim]].
    let block_logits = &out[0];
    let head_hidden = &out[1];
    // Markov + confidence weights live at the FINAL MTP stage.
    let lp = format!("model.mtp.{last}");
    let (mw1, _) = weights.take(&format!("{lp}.markov_head.markov_w1.weight"))?;
    let (mw2, _) = weights.take(&format!("{lp}.markov_head.markov_w2.weight"))?;
    let (cproj, _) = weights.take(&format!("{lp}.confidence_head.proj.weight"))?;
    let (output_ids, _confidence) = dspark_forward_head(
        block_logits,
        &mw1,
        &mw2,
        head_hidden,
        &cproj,
        input_id as usize,
        block,
        vocab,
        markov_rank,
        d,
    );
    // output_ids[0] == input_id; the drafted tail follows.
    Ok(output_ids[1..]
        .iter()
        .take(n_draft)
        .map(|&x| x as u32)
        .collect())
}

/// Greedy speculative **accept length** (`generate` verify step): given the draft
/// tokens proposed for positions `1..=block` and the main model's argmax at each
/// of those positions (`verify_argmax[i]` = argmax of the main-model logits that
/// would produce `draft_ids[i+1]`), return the number of leading draft tokens the
/// main model agrees with — the standard lossless greedy acceptance. `draft_ids`
/// is the `[block+1]` output of [`dspark_forward_head`]; `verify_argmax` is
/// `[block]`. The `block+1`-th token (the first rejected / bonus token) is taken
/// from the main model and is the caller's responsibility.
pub fn dspark_greedy_accept(draft_ids: &[usize], verify_argmax: &[usize]) -> usize {
    let mut n = 0;
    while n < verify_argmax.len() && n + 1 < draft_ids.len() && draft_ids[n + 1] == verify_argmax[n]
    {
        n += 1;
    }
    n
}

/// DeepSeek-V4 (`deepseek_v4`) spec — Hyper-Connections + o-LoRA MLA (+ KV
/// compression) + sqrtsoftplus MoE. Assembles the validated cores
/// ([`build_hc_pre`]/[`build_hc_post`]/[`build_hc_head`],
/// [`build_kv_compressor_pool`], [`build_v4_sink_attention`],
/// [`build_v4_o_lora`], `sqrtsoftplus` [`build_deepseek_moe`]). Ref:
/// deepseek-ai/DeepSeek-V4-Flash `inference/model.py`.
#[derive(Debug, Clone)]
pub struct DeepseekV4Spec {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub hc_mult: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    /// Per-layer KV-compression ratio (0 = none). Prefill-inert when seq < ratio.
    /// `ratio == 4` selects the overlapping compressor + learned Indexer (top-k);
    /// other non-zero ratios use non-overlap pooling + deterministic compressed mask.
    pub compress_ratios: Vec<usize>,
    /// Indexer (ratio-4 layers): head dim / head count / top-k budget. The top-k
    /// gate is a no-op when `seq/ratio <= index_topk`, so it only activates for
    /// long context; `index_head_dim == 0` disables the Indexer entirely.
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub window_size: usize,
    pub first_k_dense_replace: usize,
    /// First `n_hash_layers` MoE layers route via `gate.tid2eid` (per-token-id
    /// expert lookup) instead of score top-k (DeepSeek-V4 = 3). 0 disables.
    pub n_hash_layers: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    pub route_scale: f32,
    pub rope_theta: f64,
    /// RoPE base for KV-compressed layers (`compress_ratio > 0`); DeepSeek-V4 uses
    /// a larger base (160000) there vs `rope_theta` (10000) for pure-sliding layers.
    pub compress_rope_theta: f64,
    /// Clamped-SwiGLU bound (`up ∈ [-L,L]`, `gate ≤ L`); 0 disables. V4 = 10.
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    // ── YaRN (GA `DeepSeek-V4-Flash-0731`; the April preview shipped these too but
    // the builder left them off). YaRN scales the KV-compressed-layer RoPE only:
    // `original_seq_len > 0` enables the NTK-by-parts blend (base `compress_rope_theta`,
    // `rope_factor` interpolation, `beta_fast`/`beta_slow` correction range). Pure
    // sliding-window layers use raw `rope_theta` (YaRN off). No `mscale` (reference
    // `precompute_freqs_cis` applies none; `softmax_scale == head_dim^-0.5`). ──
    /// `original_max_position_embeddings` (65536). 0 disables YaRN entirely.
    pub original_seq_len: usize,
    /// YaRN interpolation `factor` (16 = 65536→1M context).
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    // ── DSpark (GA speculative-decoding module, stored under `mtp.*`). `n_mtp_layers`
    // `DSparkBlock`s reuse the MTP checkpoint slot; the main forward is correct with
    // them loaded-unused. 0 (`dspark_block_size == 0`) disables DSpark. ──
    /// Number of `DSparkBlock` stages (GA = 3). Derived from `compress_ratios`
    /// length beyond `n_layers` when `n_mtp_layers`/`num_nextn_predict_layers` absent.
    pub n_mtp_layers: usize,
    /// Draft tokens generated per DSpark step (GA = 5). 0 disables DSpark.
    pub dspark_block_size: usize,
    /// Padding token id for unfilled draft positions (GA = 128799).
    pub dspark_noise_token_id: usize,
    /// Main-model layer ids whose hidden states feed `main_proj` (GA = [40,41,42]).
    pub dspark_target_layer_ids: Vec<usize>,
    /// Markov-head embedding rank (GA = 256).
    pub dspark_markov_rank: usize,
}

impl DeepseekV4Spec {
    /// Parse a HuggingFace `deepseek_v4` `config.json` (as served by
    /// `mlx-community/DeepSeek-V4-Flash-*`) into a spec. Verified against the real
    /// checkpoint config (43 layers, head_dim 512 / rope 64, q/o-LoRA 1024,
    /// o_groups 8, 256 experts top-6, sqrtsoftplus, hc_mult 4 / 20 sinkhorn iters,
    /// compress_ratios `[0,0,4,128,4,128,…,0]`, index 64×128 top-512, 3 hash
    /// layers, sliding_window 128). `compress_ratios` is truncated to `n_layers`
    /// (the config carries one extra entry for the MTP layer).
    pub fn from_config(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
        };
        let fl = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("deepseek_v4 config missing `{k}`"));
        let n_layers = req("num_hidden_layers")?;
        let ratios_full: Vec<usize> = v
            .get("compress_ratios")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // The config carries `n_mtp_layers` extra `compress_ratios` entries (GA: 46 =
        // 43 main + 3 DSpark). Truncate to the main layers; derive DSpark stage count
        // from the surplus when the field is absent.
        let compress_ratios: Vec<usize> = ratios_full.iter().copied().take(n_layers).collect();
        let derived_mtp = ratios_full.len().saturating_sub(n_layers);
        let moe_inter = req("moe_intermediate_size")?;
        // YaRN params live under `rope_scaling` (HF top config) or as flat
        // ModelArgs-style keys (the reference `inference/config.json`).
        let rs = v.get("rope_scaling");
        let rs_u = |k: &str| {
            rs.and_then(|r| r.get(k))
                .and_then(serde_json::Value::as_u64)
        };
        let rs_f = |k: &str| {
            rs.and_then(|r| r.get(k))
                .and_then(serde_json::Value::as_f64)
        };
        let original_seq_len = rs_u("original_max_position_embeddings")
            .map(|x| x as usize)
            .or_else(|| u("original_seq_len"))
            .unwrap_or(0);
        let rope_factor = rs_f("factor").or_else(|| fl("rope_factor")).unwrap_or(1.0);
        let beta_fast = rs_f("beta_fast")
            .or_else(|| fl("beta_fast"))
            .unwrap_or(32.0);
        let beta_slow = rs_f("beta_slow").or_else(|| fl("beta_slow")).unwrap_or(1.0);
        let dspark_target_layer_ids: Vec<usize> = v
            .get("dspark_target_layer_ids")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let n_mtp_layers = u("n_mtp_layers")
            .or(if derived_mtp > 0 {
                Some(derived_mtp)
            } else {
                None
            })
            .or_else(|| u("num_nextn_predict_layers"))
            .unwrap_or(0);
        Ok(DeepseekV4Spec {
            vocab_size: req("vocab_size")?,
            dim: req("hidden_size")?,
            n_layers,
            hc_mult: u("hc_mult").unwrap_or(4),
            n_heads: req("num_attention_heads")?,
            head_dim: req("head_dim")?,
            rope_head_dim: u("qk_rope_head_dim").unwrap_or(64),
            q_lora_rank: u("q_lora_rank").unwrap_or(0),
            n_groups: u("o_groups").unwrap_or(1),
            o_lora_rank: req("o_lora_rank")?,
            compress_ratios,
            index_head_dim: u("index_head_dim").unwrap_or(0),
            index_n_heads: u("index_n_heads").unwrap_or(0),
            index_topk: u("index_topk").unwrap_or(0),
            // Pure-sliding-window layers use `sliding_window`; absent ⇒ full causal.
            window_size: u("sliding_window").unwrap_or(usize::MAX / 4),
            first_k_dense_replace: u("first_k_dense_replace")
                .or_else(|| u("n_dense_layers"))
                .unwrap_or(0),
            moe_intermediate_size: moe_inter,
            n_routed_experts: req("n_routed_experts")?,
            n_activated_experts: u("num_experts_per_tok").unwrap_or(8),
            n_shared_experts: u("n_shared_experts").unwrap_or(0),
            intermediate_size: u("intermediate_size").unwrap_or(moe_inter),
            route_scale: fl("routed_scaling_factor").unwrap_or(1.0) as f32,
            rope_theta: fl("rope_theta").unwrap_or(10000.0),
            compress_rope_theta: fl("compress_rope_theta").unwrap_or(10000.0),
            swiglu_limit: fl("swiglu_limit").unwrap_or(0.0) as f32,
            rms_norm_eps: fl("rms_norm_eps").unwrap_or(1e-6) as f32,
            hc_sinkhorn_iters: u("hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: fl("hc_eps").unwrap_or(1e-6) as f32,
            n_hash_layers: u("num_hash_layers").unwrap_or(0),
            original_seq_len,
            rope_factor,
            beta_fast,
            beta_slow,
            n_mtp_layers,
            dspark_block_size: u("dspark_block_size").unwrap_or(0),
            dspark_noise_token_id: u("dspark_noise_token_id").unwrap_or(0),
            dspark_target_layer_ids,
            dspark_markov_rank: u("dspark_markov_rank").unwrap_or(256),
        })
    }

    /// Parse a **`deepseek4` GGUF** metadata map into a spec — for the
    /// `bartowski/DeepSeek-V4-Flash-0731-GGUF` (llama.cpp `general.architecture ==
    /// deepseek4`, MXFP4). `meta` is the GGUF key/value store as a JSON object
    /// (`deepseek4.*` + `general.*`); `vocab_size` comes from the `token_embd`
    /// shape / tokenizer (GGUF has no explicit vocab KV). Verified against the real
    /// shard-1 header: block_count 43, embedding_length 4096, head_count 64/kv 1,
    /// key/value_length 512, rope.dimension_count 64, q_lora_rank 1024,
    /// output_lora_rank 1024, output_group_count 8, indexer 64×128 top-512,
    /// compress_ratios[46], compress_rope_freq_base 160000, expert 256×6+1,
    /// expert_weights_scale 1.5, hyper_connection.count 4 / sinkhorn 20,
    /// hash_layer_count 3, swiglu_clamp 10. The GGUF is **base-only** (no MTP/DSpark
    /// tensors), so all `dspark_*`/`n_mtp_layers` are 0. YaRN is read from
    /// `deepseek4.rope.scaling.*` when present (else off).
    pub fn from_gguf_metadata(meta: &serde_json::Value, vocab_size: usize) -> Result<Self> {
        // GGUF metadata keys are `deepseek4.<suffix>`; `general.<suffix>` for the rest.
        let g = |k: &str| meta.get(format!("deepseek4.{k}"));
        let u = |k: &str| g(k).and_then(serde_json::Value::as_u64).map(|x| x as usize);
        let fl = |k: &str| g(k).and_then(serde_json::Value::as_f64);
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("deepseek4 GGUF missing `deepseek4.{k}`"));
        let n_layers = req("block_count")?;
        let ratios_full: Vec<usize> = g("attention.compress_ratios")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let compress_ratios: Vec<usize> = ratios_full.into_iter().take(n_layers).collect();
        let moe_inter = req("expert_feed_forward_length")?;
        // Per-layer swiglu clamp array (all 10.0) — take the first as the scalar bound.
        let swiglu_limit = g("swiglu_clamp_exp")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_f64)
            .or_else(|| fl("swiglu_clamp_exp"))
            .unwrap_or(0.0) as f32;
        Ok(DeepseekV4Spec {
            vocab_size,
            dim: req("embedding_length")?,
            n_layers,
            hc_mult: u("hyper_connection.count").unwrap_or(4),
            n_heads: req("attention.head_count")?,
            head_dim: u("attention.key_length").unwrap_or(512),
            rope_head_dim: u("rope.dimension_count").unwrap_or(64),
            q_lora_rank: u("attention.q_lora_rank").unwrap_or(0),
            n_groups: u("attention.output_group_count").unwrap_or(1),
            o_lora_rank: u("attention.output_lora_rank").unwrap_or(moe_inter),
            compress_ratios,
            index_head_dim: u("attention.indexer.key_length").unwrap_or(0),
            index_n_heads: u("attention.indexer.head_count").unwrap_or(0),
            index_topk: u("attention.indexer.top_k").unwrap_or(0),
            window_size: u("attention.sliding_window").unwrap_or(usize::MAX / 4),
            first_k_dense_replace: 0,
            n_hash_layers: u("hash_layer_count").unwrap_or(0),
            moe_intermediate_size: moe_inter,
            n_routed_experts: req("expert_count")?,
            n_activated_experts: u("expert_used_count").unwrap_or(6),
            n_shared_experts: u("expert_shared_count").unwrap_or(0),
            intermediate_size: moe_inter,
            route_scale: fl("expert_weights_scale").unwrap_or(1.0) as f32,
            rope_theta: fl("rope.freq_base").unwrap_or(10000.0),
            compress_rope_theta: fl("attention.compress_rope_freq_base").unwrap_or(10000.0),
            swiglu_limit,
            rms_norm_eps: fl("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32,
            hc_sinkhorn_iters: u("hyper_connection.sinkhorn_iterations").unwrap_or(20),
            hc_eps: fl("hyper_connection.epsilon").unwrap_or(1e-6) as f32,
            // YaRN: llama.cpp stores it under `rope.scaling.*` when applied (else off).
            original_seq_len: u("rope.scaling.original_context_length").unwrap_or(0),
            rope_factor: fl("rope.scaling.factor").unwrap_or(1.0),
            beta_fast: fl("rope.scaling.beta_fast").unwrap_or(32.0),
            beta_slow: fl("rope.scaling.beta_slow").unwrap_or(1.0),
            // GGUF is base-only — no MTP/DSpark tensors ("MTP: no").
            n_mtp_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: Vec::new(),
            dspark_markov_rank: 0,
        })
    }
}

/// Map a **`deepseek4` GGUF** (llama.cpp `blk.N.*`) tensor name to the HF-style key
/// the V4 builders ([`build_deepseek_v4_stage`]) request, or vice versa — the
/// translation a `WeightLoader` adapter wraps around a `GgufLoader` so the MXFP4
/// K-quant experts flow through the existing GGUF packed path. Covers the names
/// **confirmed** against the real `bartowski/DeepSeek-V4-Flash-0731-GGUF` header
/// (stacked experts `blk.N.ffn_{gate,up,down}_exps.weight`, hash routing
/// `blk.N.ffn_gate_tid2eid.weight`) plus the standard llama.cpp DeepSeek /
/// deepseek4-specific names (`attn_q_a`/`attn_q_b`/`attn_kv_a_mqa`/`attn_o_a`/
/// `attn_o_b`/`attn_compressor_*`/`attn_indexer_*`/`hc_attn_*`/`hc_ffn_*`).
/// Returns the GGUF name for an HF builder key, or `None` if unmapped.
pub fn hf_key_to_deepseek4_gguf(key: &str) -> Option<String> {
    // Top-level (non-layer) tensors.
    match key {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        "model.hc_head.fn" => return Some("output_hc.weight".into()),
        "model.hc_head.scale" => return Some("output_hc_scale.weight".into()),
        "model.hc_head.base" => return Some("output_hc_base.weight".into()),
        _ => {}
    }
    // Per-layer: `model.layers.N.<rest>` → `blk.N.<mapped>`.
    let rest = key.strip_prefix("model.layers.")?;
    let (n, tail) = rest.split_once('.')?;
    let mapped = match tail {
        "attn_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "attn.wq_a.weight" => "attn_q_a.weight",
        "attn.q_norm.weight" => "attn_q_a_norm.weight",
        "attn.wq_b.weight" => "attn_q_b.weight",
        "attn.wkv.weight" => "attn_kv_a_mqa.weight",
        "attn.kv_norm.weight" => "attn_kv_a_norm.weight",
        "attn.wo_a.weight" => "attn_o_a.weight",
        "attn.wo_b.weight" => "attn_o_b.weight",
        "attn.attn_sink" => "attn_sink.weight",
        "attn_hc.fn" => "hc_attn.weight",
        "attn_hc.scale" => "hc_attn_scale.weight",
        "attn_hc.base" => "hc_attn_base.weight",
        "ffn_hc.fn" => "hc_ffn.weight",
        "ffn_hc.scale" => "hc_ffn_scale.weight",
        "ffn_hc.base" => "hc_ffn_base.weight",
        "attn.compressor.wkv.weight" => "attn_compressor_kv.weight",
        "attn.compressor.wgate.weight" => "attn_compressor_gate.weight",
        "attn.compressor.ape" => "attn_compressor_ape.weight",
        "attn.compressor.norm.weight" => "attn_compressor_norm.weight",
        "attn.indexer.wq_b.weight" => "attn_indexer_q_b.weight",
        "attn.indexer.weights_proj.weight" => "attn_indexer_weights.weight",
        "attn.indexer.compressor.wkv.weight" => "attn_indexer_compressor_kv.weight",
        "attn.indexer.compressor.wgate.weight" => "attn_indexer_compressor_gate.weight",
        "attn.indexer.compressor.ape" => "attn_indexer_compressor_ape.weight",
        "attn.indexer.compressor.norm.weight" => "attn_indexer_compressor_norm.weight",
        "ffn.gate.weight" => "ffn_gate_inp.weight",
        "ffn.gate.e_score_correction_bias" => "exp_probs_b.bias",
        "ffn.gate.tid2eid" => "ffn_gate_tid2eid.weight",
        "ffn.switch_mlp.gate_proj.weight" => "ffn_gate_exps.weight",
        "ffn.switch_mlp.up_proj.weight" => "ffn_up_exps.weight",
        "ffn.switch_mlp.down_proj.weight" => "ffn_down_exps.weight",
        "ffn.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
        "ffn.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
        "ffn.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
        _ => return None,
    };
    Some(format!("blk.{n}.{mapped}"))
}

/// Map a V4 builder key (`model.*` HF-style, as the `build_deepseek_v4_*` graphs
/// emit) to the tensor name in a **reference-format** checkpoint — one exported
/// via deepseek-ai `inference/convert.py`, as `Vontra/DeepSeek-V4-Flash-0731-
/// MXFP4-MLX` ships. **Verified against that checkpoint's real 72k-tensor index.**
/// The `attn.*` / `attn.compressor.*` / `attn.indexer.*` / `ffn.gate.{weight,
/// tid2eid}` subnames already match; the deltas: no `model.` prefix; `embed` /
/// `head` / `norm` for embed / lm_head / final norm; `hc_head_*` and per-layer
/// `hc_attn_*` / `hc_ffn_*` for the Hyper-Connection tensors; `shared_experts.
/// w{1,3,2}` for shared gate/up/down. Returns the checkpoint name, or `None`.
/// The **routed** experts are stored PER-EXPERT (`layers.N.ffn.experts.{i}.
/// w{1,3,2}`, not stacked), so a `switch_mlp.*` request has no single name — a
/// loader must gather them via [`dsv4_ref_expert_key`] into the `[n_expert, out,
/// in]` tensor the grouped matmul wants (the remaining structural adapter).
pub fn hf_key_to_dsv4_ref(key: &str) -> Option<String> {
    match key {
        "model.embed_tokens.weight" => return Some("embed.weight".into()),
        "model.norm.weight" => return Some("norm.weight".into()),
        "lm_head.weight" => return Some("head.weight".into()),
        "model.hc_head.fn" => return Some("hc_head_fn".into()),
        "model.hc_head.scale" => return Some("hc_head_scale".into()),
        "model.hc_head.base" => return Some("hc_head_base".into()),
        _ => {}
    }
    let rest = key.strip_prefix("model.layers.")?;
    let (n, tail) = rest.split_once('.')?;
    let mapped = match tail {
        "attn_hc.fn" => "hc_attn_fn".to_string(),
        "attn_hc.scale" => "hc_attn_scale".to_string(),
        "attn_hc.base" => "hc_attn_base".to_string(),
        "ffn_hc.fn" => "hc_ffn_fn".to_string(),
        "ffn_hc.scale" => "hc_ffn_scale".to_string(),
        "ffn_hc.base" => "hc_ffn_base".to_string(),
        "ffn.shared_experts.gate_proj.weight" => "ffn.shared_experts.w1.weight".to_string(),
        "ffn.shared_experts.up_proj.weight" => "ffn.shared_experts.w3.weight".to_string(),
        "ffn.shared_experts.down_proj.weight" => "ffn.shared_experts.w2.weight".to_string(),
        // aux-loss-free routing bias — convert.py renames e_score_correction_bias → bias.
        "ffn.gate.e_score_correction_bias" => "ffn.gate.bias".to_string(),
        // attn.* / attn_norm / ffn_norm / ffn.gate.* / compressor.* / indexer.* keep
        // their subname — only the `model.` prefix is dropped.
        other => other.to_string(),
    };
    Some(format!("layers.{n}.{mapped}"))
}

/// Per-expert routed-MoE tensor name in a reference-format V4 checkpoint
/// (`ffn.experts.{i}.w{1,3,2}`, where w1=gate, w3=up, w2=down). A stacking loader
/// gathers experts `0..n_routed` into the `[n_expert, out, in]` tensor the
/// `DequantGroupedMatMulMlx` op consumes.
pub fn dsv4_ref_expert_key(layer: usize, expert: usize, proj: &str) -> String {
    let w = match proj {
        "gate_proj" => "w1",
        "up_proj" => "w3",
        "down_proj" => "w2",
        _ => "w1",
    };
    format!("layers.{layer}.ffn.experts.{expert}.{w}.weight")
}

/// A [`WeightLoader`] adapter that loads a **reference-format** DeepSeek-V4
/// checkpoint (post `inference/convert.py`, e.g.
/// `Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`) through the builder's HF-style keys.
/// It (1) name-maps every request via [`hf_key_to_dsv4_ref`], and (2) for a
/// stacked `switch_mlp.{gate,up,down}_proj` request, GATHERS the per-expert
/// `ffn.experts.{i}.w{1,3,2}` MXFP4 tensors and concatenates their packed codes +
/// scales into the single `[n_expert, out, in]` tensor the `DequantGroupedMatMulMlx`
/// op consumes. Wrap `MlxLoader::open_lazy(dir)` with this to load the real GA
/// checkpoint on a node (each pipeline stage's layer range fits the node's RAM).
pub struct DsV4RefLoader {
    inner: Box<dyn WeightLoader>,
    n_experts: usize,
}

impl DsV4RefLoader {
    pub fn new(inner: Box<dyn WeightLoader>, n_experts: usize) -> Self {
        Self { inner, n_experts }
    }
    fn map(key: &str) -> String {
        hf_key_to_dsv4_ref(key).unwrap_or_else(|| key.to_string())
    }
    /// `(layer, proj)` if `key` is a stacked routed-expert request, else `None`.
    fn switch_mlp_parts(key: &str) -> Option<(usize, &'static str)> {
        let rest = key.strip_prefix("model.layers.")?;
        let (n, tail) = rest.split_once('.')?;
        let proj = match tail {
            "ffn.switch_mlp.gate_proj.weight" => "gate_proj",
            "ffn.switch_mlp.up_proj.weight" => "up_proj",
            "ffn.switch_mlp.down_proj.weight" => "down_proj",
            _ => return None,
        };
        Some((n.parse().ok()?, proj))
    }
}

impl WeightLoader for DsV4RefLoader {
    fn format_id(&self) -> &'static str {
        "dsv4-ref"
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&Self::map(key))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&Self::map(key))
    }
    fn take_packed_mlx(
        &mut self,
        key: &str,
    ) -> Result<Option<crate::weight_loader::MlxPackedLinear>> {
        if let Some((layer, proj)) = Self::switch_mlp_parts(key) {
            // Prefetch ALL experts' packed tensors in one MADV_WILLNEED batch before
            // the serial gather: the OS issues the reads concurrently (saturating the
            // Thunderbolt/NVMe link) instead of the loop stalling on one latency-bound
            // per-expert mmap fault at a time.
            let ekeys: Vec<String> = (0..self.n_experts)
                .map(|e| dsv4_ref_expert_key(layer, e, proj))
                .collect();
            let erefs: Vec<&str> = ekeys.iter().map(|s| s.as_str()).collect();
            self.inner.prewarm(&erefs);
            let (mut w_q, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
            let mut out_shape: Vec<usize> = Vec::new();
            let mut scheme = None;
            for e in 0..self.n_experts {
                let p = self
                    .inner
                    .take_packed_mlx(&ekeys[e])?
                    .ok_or_else(|| anyhow!("routed expert {e} of {key} is not MLX-packed"))?;
                w_q.extend_from_slice(&p.w_q);
                scales.extend_from_slice(&p.scales);
                biases.extend_from_slice(&p.biases);
                out_shape = p.out_shape;
                scheme = Some(p.scheme);
            }
            return Ok(Some(crate::weight_loader::MlxPackedLinear {
                w_q,
                scales,
                biases,
                scheme: scheme.ok_or_else(|| anyhow!("no experts for {key}"))?,
                out_shape,
            }));
        }
        self.inner.take_packed_mlx(&Self::map(key))
    }
    fn prewarm(&self, keys: &[&str]) {
        // Name-map each key (model.layers.N.* → the ref key) before forwarding, so the
        // inner mmap loader warms the right tensors.
        let mapped: Vec<String> = keys.iter().map(|k| Self::map(k)).collect();
        let refs: Vec<&str> = mapped.iter().map(String::as_str).collect();
        self.inner.prewarm(&refs);
    }
    fn borrow_packed_mlx(&self, key: &str) -> Option<rlx_mlx_io::PackedMlxBorrow<'_>> {
        // Only the per-expert layout can zero-copy borrow; the stacked switch_mlp path
        // must concatenate experts (owned copy).
        if Self::switch_mlp_parts(key).is_some() {
            return None;
        }
        self.inner.borrow_packed_mlx(&Self::map(key))
    }
    fn dontneed_packed_mlx(&self, key: &str) {
        if Self::switch_mlp_parts(key).is_some() {
            return;
        }
        self.inner.dontneed_packed_mlx(&Self::map(key));
    }
}

/// Build a **DeepSeek-V4 (`deepseek_v4`)** prefill graph by assembling the five
/// validated subsystem cores under the Hyper-Connections stream flow:
/// `embed → repeat to hc_mult streams → per block[ hc_pre → o-LoRA MLA (q-LoRA +
/// wkv + optional KV-compressor + sink-attention) → hc_post ; hc_pre → dense/
/// sqrtsoftplus-MoE FFN → hc_post ] → hc_head → norm → lm_head`. Partial RoPE
/// (last `rope_head_dim`) on q/kv with an inverse rope on the attention output.
/// `ratio == 4` layers use the **overlapping** KV compressor
/// ([`build_kv_compressor_overlap`]) plus the learned sparse **Indexer** top-k
/// gate ([`build_v4_indexer_score`] + [`build_v4_topk_gate`]) — the gate is a
/// no-op (all causally-valid compressed positions kept, exactly the deterministic
/// mask) whenever `seq/ratio <= index_topk`, so it only prunes for long context;
/// other non-zero ratios use non-overlap pooling + the deterministic compressed
/// mask. FP4/Hadamard quant-sim omitted (precision-only; the Hadamard rotation is
/// orthogonal and cancels in the Indexer score). Returns logits `[seq, vocab]`.
pub fn build_deepseek_v4_prefill(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_deepseek_v4_stage(spec, weights, seq, 0..spec.n_layers, true, true, packed)
}

/// One **pipeline stage** of DeepSeek-V4: builds only transformer layers
/// `layers`, loading only those layers' weights (+ embeddings if `first`, LM
/// head/final-norm if `last`). The graph input is `input_ids [1,seq]` when
/// `first` else the boundary hidden state `hidden_in [rows, hc_mult, dim]`; the
/// output is `logits [seq, vocab]` when `last` else `hidden_out [rows, hc, dim]`.
/// This is what lets a node build+run its slice of a model larger than its RAM
/// from just its own checkpoint shards (see [`build_deepseek_v4_prefill`] =
/// `0..n_layers, first=true, last=true`).
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v4_stage(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    layers: std::ops::Range<usize>,
    first: bool,
    last: bool,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("deepseek_v4_stage");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let d = spec.dim;
    let hc = spec.hc_mult;
    let nh = spec.n_heads;
    let hd = spec.head_dim;
    let rd = spec.rope_head_dim & !1;
    let ql = spec.q_lora_rank;
    let eps = spec.rms_norm_eps;
    let rows = seq;
    let dbg_layer: usize = std::env::var("RLX_DSV4_DBGLAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(layers.start);
    let scale = (hd as f32).powf(-0.5);
    let zb_d = synth_zero(&mut g, &mut params, "v4.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v4.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v4.zb.hd", hd);

    // RoPE (GPT-J **interleaved**, applied to the LAST `rope_head_dim` dims of each
    // head — see `apply_rotary_emb(q[..., -rd:])` in the reference). Two bases:
    // `rope_theta` for pure-sliding layers (YaRN off), `compress_rope_theta` for
    // KV-compressed layers (ratio>0). GA `DeepSeek-V4-Flash-0731` applies **YaRN**
    // on the compressed-layer table (`precompute_freqs_cis(original_seq_len>0)`):
    // NTK-by-parts blend of interpolated (÷factor) ↔ extrapolated freqs over the
    // `[low,high]` correction range. `RopeScaling::None::inv_freq == θ^(-2i/rd)`,
    // so the sliding table is byte-identical to the pre-YaRN builder; only the
    // compressed table changes. No `mscale` (reference applies none). `half`=rd/2.
    let half = (rd / 2).max(1);
    let sliding_rs = RopeScaling::None;
    let compress_rs = if spec.original_seq_len > 0 && spec.rope_factor > 1.0 {
        RopeScaling::Yarn {
            factor: spec.rope_factor,
            original_max_position_embeddings: spec.original_seq_len as f64,
            beta_fast: spec.beta_fast,
            beta_slow: spec.beta_slow,
            attention_factor: Some(1.0), // no mscale for V4 (unused here regardless)
        }
    } else {
        RopeScaling::None
    };
    let rope_tables = |g: &mut Graph,
                       params: &mut HashMap<String, Vec<f32>>,
                       theta: f64,
                       rs: &RopeScaling,
                       tag: &str| {
        let (mut cosd, mut sind) = (vec![0f32; seq * half], vec![0f32; seq * half]);
        for p in 0..seq {
            for i in 0..half {
                let fr = rs.inv_freq(i, rd, theta);
                let (s, c) = (p as f64 * fr).sin_cos();
                cosd[p * half + i] = c as f32;
                sind[p * half + i] = s as f32;
            }
        }
        let sinneg: Vec<f32> = sind.iter().map(|v| -v).collect();
        let cos = synth_const(g, params, &format!("v4.rope.cos.{tag}"), cosd, &[seq, half]);
        let sin = synth_const(g, params, &format!("v4.rope.sin.{tag}"), sind, &[seq, half]);
        let sin_inv = synth_const(
            g,
            params,
            &format!("v4.rope.sininv.{tag}"),
            sinneg,
            &[seq, half],
        );
        (cos, sin, sin_inv)
    };
    let (cos_m, sin_m, sininv_m) =
        rope_tables(&mut g, &mut params, spec.rope_theta, &sliding_rs, "m");
    let (cos_c, sin_c, sininv_c) = rope_tables(
        &mut g,
        &mut params,
        spec.compress_rope_theta,
        &compress_rs,
        "c",
    );

    // Causal window mask [rows, seq] (window ≥ seq at prefill ⇒ full causal).
    let neg = -1e30f32;
    // Sliding-window causal mask: query qi sees key ki iff ki<=qi AND within the
    // last `window_size` positions (DeepSeek-V4 pure-sliding-window layers).
    let window = spec.window_size.max(1);
    let mut maskd = vec![0f32; seq * seq];
    for qi in 0..seq {
        for ki in 0..seq {
            if ki > qi || qi - ki >= window {
                maskd[qi * seq + ki] = neg;
            }
        }
    }
    let mask_win = synth_const(&mut g, &mut params, "v4.mask.win", maskd, &[seq, seq]);

    let (mut h, input_ids_flat): (NodeId, Option<NodeId>) = if first {
        let input_ids = g.input("input_ids", Shape::new(&[1, seq], DType::I32));
        // Flat token-ids [seq] for `tid2eid` hash-routing gathers on early layers.
        let input_ids_flat = g.reshape_(input_ids, vec![seq as i64]);
        let (embed_w, _, _) =
            load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
        let h0 = g.gather_(embed_w, input_ids, 0); // [1,seq,d]
        let h0 = g.reshape_(h0, vec![rows as i64, 1, d as i64]);
        // HC-expand: repeat the single stream to hc_mult streams → [rows, hc, d].
        let ones_hc = synth_const(
            &mut g,
            &mut params,
            "v4.hc.ones",
            vec![1f32; hc],
            &[1, hc, 1],
        );
        (g.mul(h0, ones_hc), Some(input_ids_flat)) // [rows,hc,d]
    } else {
        // Boundary hidden state fed by the previous stage.
        let hidden_in = g.input("hidden_in", Shape::new(&[rows, hc, d], DType::F32));
        (hidden_in, None)
    };

    for il in layers.clone() {
        let lp = format!("model.layers.{il}");
        let ratio = spec.compress_ratios.get(il).copied().unwrap_or(0);
        // KV-compressed layers rope with `compress_rope_theta`, pure-sliding with base.
        let (cos_id, sin_id, sin_inv) = if ratio > 0 {
            (cos_c, sin_c, sininv_c)
        } else {
            (cos_m, sin_m, sininv_m)
        };

        // ── Attention block (HC-wrapped) ──
        let residual = h;
        let fn_a =
            load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn_hc.fn"))?;
        let sc_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.scale"),
            false,
        )?;
        let bs_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.base"),
            false,
        )?;
        let (xa, post_a, comb_a) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            rows,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.a"),
        );
        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_norm.weight"),
            0.0,
        )?;
        let xa = g.rms_norm(xa, an, zb_d, eps);

        // o-LoRA MLA: q-LoRA + wkv + (compressor) + sink-attention + o-LoRA.
        let wqa = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_a.weight"),
        )?;
        let qr = emit_proj(&mut g, xa, &wqa, Shape::new(&[rows, ql], f));
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.q_norm.weight"),
            0.0,
        )?;
        let qr = g.rms_norm(qr, qn, zb_ql, eps);
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("qa") && il == dbg_layer {
            g.set_outputs(vec![qr]);
            return Ok((g, params));
        }
        let wqb = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_b.weight"),
        )?;
        let q = emit_proj(&mut g, qr, &wqb, Shape::new(&[rows, nh * hd], f));
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("qb") && il == dbg_layer {
            g.set_outputs(vec![q]);
            return Ok((g, params));
        }
        // per-head normalize (rsqrt of mean-sq over head_dim), then partial RoPE.
        let q_ones = synth_const(
            &mut g,
            &mut params,
            &format!("{lp}.v4.qones"),
            vec![1f32; hd],
            &[hd],
        );
        let qn2 = per_head_rms(&mut g, q, q_ones, zb_hd, 1, rows, nh, hd, spec.rms_norm_eps);
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("qhr") && il == dbg_layer {
            g.set_outputs(vec![qn2]);
            return Ok((g, params));
        }
        let q = rope_tail(&mut g, qn2, cos_id, sin_id, rows, nh, hd, rd);
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[rows, hd], f));
        let kvn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let kv = g.rms_norm(kv, kvn, zb_hd, eps);
        let kv = rope_tail(&mut g, kv, cos_id, sin_id, rows, 1, hd, rd);

        // Key set + mask (compression appended when ratio triggers at this seq).
        let (kv_all, mask, n_keys) = if ratio > 0 && seq >= ratio {
            let ncomp = seq / ratio;
            let overlap = ratio == 4; // reference: `overlap = compress_ratio == 4`
            let sfull = seq - seq % ratio;
            // Attended compressed KV — overlapping pooling for ratio-4, else
            // non-overlap. wkv/wgate output `coff*hd` (coff = 1 + overlap).
            let cw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wkv.weight"),
                true,
            )?;
            let ck = g.mm(xa, cw);
            let cgw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wgate.weight"),
                true,
            )?;
            let cg = g.mm(xa, cgw);
            let ape = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.ape"),
                false,
            )?;
            let cnorm = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.norm.weight"),
                0.0,
            )?;
            let comp = if overlap {
                build_kv_compressor_overlap(
                    &mut g,
                    &mut params,
                    ck,
                    cg,
                    ape,
                    cnorm,
                    sfull,
                    ratio,
                    hd,
                    eps,
                    &format!("{lp}.comp"),
                )
            } else {
                build_kv_compressor_pool(
                    &mut g,
                    &mut params,
                    ck,
                    cg,
                    ape,
                    cnorm,
                    1,
                    sfull,
                    ratio,
                    hd,
                    eps,
                    &format!("{lp}.comp"),
                )
            };
            let kv_all = g.concat_(vec![kv, comp], 0); // [window seq | compressed ncomp]

            // Compressed-position causal mask [seq, ncomp]: position c is visible
            // to query qi iff c < (qi+1)/ratio.
            let mut mc = vec![0f32; seq * ncomp];
            for qi in 0..seq {
                for c in 0..ncomp {
                    mc[qi * ncomp + c] = if c < (qi + 1) / ratio { 0.0 } else { neg };
                }
            }
            let causal_c = synth_const(
                &mut g,
                &mut params,
                &format!("{lp}.v4.maskc"),
                mc,
                &[seq, ncomp],
            );

            // Learned Indexer top-k gate (ratio-4 only, and only when there are
            // more compressed positions than the top-k budget — otherwise every
            // causally-valid position is kept and the deterministic mask is exact).
            let indexer_on = overlap && spec.index_head_dim > 0 && ncomp > spec.index_topk;
            let comp_mask = if indexer_on {
                let ihd = spec.index_head_dim;
                let iw = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.compressor.wkv.weight"),
                    true,
                )?;
                let ick = g.mm(xa, iw);
                let igw = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.compressor.wgate.weight"),
                    true,
                )?;
                let icg = g.mm(xa, igw);
                let iape = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.compressor.ape"),
                    false,
                )?;
                let inorm = load_norm(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.compressor.norm.weight"),
                    0.0,
                )?;
                let ikv = build_kv_compressor_overlap(
                    &mut g,
                    &mut params,
                    ick,
                    icg,
                    iape,
                    inorm,
                    sfull,
                    ratio,
                    ihd,
                    eps,
                    &format!("{lp}.icomp"),
                );
                let iwqb = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.wq_b.weight"),
                    true,
                )?;
                let iwpj = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.indexer.weights_proj.weight"),
                    true,
                )?;
                let score = build_v4_indexer_score(
                    &mut g,
                    &mut params,
                    qr,
                    xa,
                    ikv,
                    iwqb,
                    iwpj,
                    cos_id,
                    sin_id,
                    seq,
                    spec.index_n_heads,
                    ihd,
                    rd,
                    ncomp,
                    &format!("{lp}.idx"),
                );
                let k = spec.index_topk.min(ncomp);
                build_v4_topk_gate(
                    &mut g,
                    &mut params,
                    score,
                    causal_c,
                    seq,
                    ncomp,
                    k,
                    &format!("{lp}.idx"),
                )
            } else {
                causal_c
            };
            // Full key mask = [sliding-window causal (seq) | compressed mask (ncomp)].
            let mut mw = vec![0f32; seq * seq];
            for qi in 0..seq {
                for ki in 0..seq {
                    mw[qi * seq + ki] = if ki > qi || qi - ki >= window {
                        neg
                    } else {
                        0.0
                    };
                }
            }
            let win_m = synth_const(
                &mut g,
                &mut params,
                &format!("{lp}.v4.maskw"),
                mw,
                &[seq, seq],
            );
            let full_mask = g.concat_(vec![win_m, comp_mask], 1); // [seq, seq+ncomp]
            (kv_all, full_mask, seq + ncomp)
        } else {
            (kv, mask_win, seq)
        };
        let q3 = g.reshape_(q, vec![rows as i64, nh as i64, hd as i64]);
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("kv") && il == dbg_layer {
            g.set_outputs(vec![kv_all]);
            return Ok((g, params));
        }
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("q") && il == dbg_layer {
            g.set_outputs(vec![q3]);
            return Ok((g, params));
        }
        let sink = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.attn_sink"),
            false,
        )?;
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("xa") && il == dbg_layer {
            g.set_outputs(vec![xa]);
            return Ok((g, params));
        }
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            scale,
            rows,
            nh,
            hd,
            n_keys,
            &format!("{lp}.sa"),
        );
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("o") && il == dbg_layer {
            g.set_outputs(vec![o]);
            return Ok((g, params));
        }
        // inverse RoPE on the output's rope tail (last rd dims).
        let o_flat0 = g.reshape_(o, vec![rows as i64, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat0, cos_id, sin_inv, rows, nh, hd, rd);
        // grouped o-LoRA output.
        let dpg = nh * hd / spec.n_groups;
        let woa = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wob = load_transposed_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_b.weight"),
        )?;
        let attn_out = build_v4_o_lora(
            &mut g,
            o_inv,
            woa,
            wob,
            rows,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("olora") && il == dbg_layer {
            g.set_outputs(vec![attn_out]);
            return Ok((g, params));
        }
        h = build_hc_post(&mut g, attn_out, residual, post_a, comb_a, rows, hc, d);

        // DEBUG: output hidden right after the attention block of the first layer
        // to isolate attention-vs-FFN explosion (RLX_DSV4_DBG=attn).
        if std::env::var("RLX_DSV4_DBG").as_deref() == Ok("attn") && il == dbg_layer {
            g.set_outputs(vec![h]);
            return Ok((g, params));
        }

        // ── FFN block (HC-wrapped): dense SwiGLU or sqrtsoftplus MoE ──
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.ffn_hc.fn"))?;
        let sc_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.scale"),
            false,
        )?;
        let bs_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.base"),
            false,
        )?;
        let (xf, post_f, comb_f) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            rows,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.f"),
        );
        let fnorm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = if il < spec.first_k_dense_replace {
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.up_proj.weight"),
            )?;
            let dn = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.down_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                xf,
                &gp,
                Shape::new(&[rows, spec.intermediate_size], f),
            );
            let upv = emit_proj(
                &mut g,
                xf,
                &up,
                Shape::new(&[rows, spec.intermediate_size], f),
            );
            let sg = g.silu(gate);
            let glu = g.mul(sg, upv);
            emit_proj(&mut g, glu, &dn, Shape::new(&[rows, d], f))
        } else {
            let ds = v4_moe_spec(spec);
            let x3 = g.reshape_(xf, vec![1, rows as i64, d as i64]);
            // First `n_hash_layers` route via `gate.tid2eid` (per-token-id lookup).
            // Hash layers (0..n_hash_layers) live only in the first stage, which
            // is the only stage with `input_ids_flat`.
            let hash_ids = if il < spec.n_hash_layers {
                input_ids_flat
            } else {
                None
            };
            // `build_deepseek_moe_ffn` appends the `ffn` container (ct) itself →
            // pass the LAYER prefix, giving `model.layers.N.ffn.{gate,switch_mlp,…}`.
            let moe = build_deepseek_moe_ffn(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                x3,
                1,
                seq,
                &ds,
                hash_ids,
            )?;
            g.reshape_(moe, vec![rows as i64, d as i64])
        };
        if matches!(
            std::env::var("RLX_DSV4_DBG").as_deref(),
            Ok("moe") | Ok("routed") | Ok("shared") | Ok("noweight")
        ) && il == dbg_layer
        {
            g.set_outputs(vec![ffn_out]);
            return Ok((g, params));
        }
        h = build_hc_post(&mut g, ffn_out, residual, post_f, comb_f, rows, hc, d);
    }

    if last {
        // Head: hc_head reduce → norm → lm_head → logits.
        let hfn = load_transposed_param(&mut g, &mut params, weights, "model.hc_head.fn")?;
        let hsc = load_p(&mut g, &mut params, weights, "model.hc_head.scale", false)?;
        let hbs = load_p(&mut g, &mut params, weights, "model.hc_head.base", false)?;
        let x = build_hc_head(
            &mut g,
            &mut params,
            h,
            hfn,
            hsc,
            hbs,
            rows,
            hc,
            d,
            spec.hc_eps,
            "head",
        );
        let fnorm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
        let x = g.rms_norm(x, fnorm, zb_d, eps);
        let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
        let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[rows, spec.vocab_size], f));
        let logits = g.reshape_(logits, vec![rows as i64, spec.vocab_size as i64]);
        g.set_outputs(vec![logits]);
    } else {
        // Boundary output: the hc-stream hidden state for the next stage.
        g.set_outputs(vec![h]);
    }
    Ok((g, params))
}

/// Single-window **overlapping** compressor pool for the decode path (`ratio == 4`).
/// Given the previous window's `[ratio, 2·hd]` kv/score (with APE, or `-1e30`
/// score for the not-yet-seen window 0) and the current window's `[ratio, 2·hd]`
/// kv/score (with APE), it forms the `2·ratio` overlap candidate set — the
/// previous tokens' **first** `hd` dims ++ the current tokens' **second** `hd`
/// dims — softmaxes over the `2·ratio` candidates per feature, weighted-sums to
/// `[hd]`, and RMSNorms. Matches [`build_kv_compressor_overlap`] for one window
/// (which the prefill uses). No RoPE (the prefill stage doesn't rope `comp`).
#[allow(clippy::too_many_arguments)]
fn build_overlap_pool_single(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prev_kv: NodeId,
    prev_score: NodeId,
    curr_kv: NodeId,
    curr_score: NodeId,
    norm_w: NodeId,
    hd: usize,
    eps: f32,
    tag: &str,
) -> NodeId {
    let d = hd as i64;
    // candidate kv/score [2*ratio, hd]: prev first-half ++ current second-half.
    let pk = g.narrow_(prev_kv, 1, 0, hd);
    let ck = g.narrow_(curr_kv, 1, hd, hd);
    let cand_kv = g.concat_(vec![pk, ck], 0);
    let ps = g.narrow_(prev_score, 1, 0, hd);
    let cs = g.narrow_(curr_score, 1, hd, hd);
    let cand_sc = g.concat_(vec![ps, cs], 0);
    // softmax over the 2*ratio token axis (per hd feature): move it last, sm, back.
    let sct = g.transpose_(cand_sc, vec![1, 0]);
    let w = g.sm(sct, -1);
    let w = g.transpose_(w, vec![1, 0]);
    let pooled = g.mul(cand_kv, w);
    let pooled = g.sum(pooled, vec![0], false); // [hd]
    let pooled2 = g.reshape_(pooled, vec![1, d]);
    let zb = synth_zero(g, params, &format!("{tag}.ovzb"), hd);
    g.rms_norm(pooled2, norm_w, zb, eps)
}

/// Build a **single-token DECODE step** for DeepSeek-V4 with a KV cache — the
/// O(1)-per-token path that replaces re-running the whole prefill each step
/// (which is O(n²): 289 s for a *6-token* prompt). The graph takes the new token
/// id + each layer's cached window KV (`kvcache.{il}` `[cache_len, head_dim]`,
/// the roped MLA latents of the prior tokens) and returns `logits [1, vocab]`
/// plus per-layer state the host threads into the next step. RoPE is applied at
/// absolute position `pos`; the query attends to `[window-cache ++ new (++
/// compressed-cache)]` with the per-head attention sink (all provided positions
/// are valid by construction, so no mask). Correctness is by the standard KV-cache
/// induction: fed the same tokens, the accumulated cache equals what prefill
/// computes internally, so decode logits == prefill logits (validated).
///
/// Returns the output-name list (in graph-output order) so the host can map
/// results and thread the right buffers. Per layer:
/// * `kvnew.{il}` `[1, head_dim]` — the new roped MLA latent → append to the
///   layer's window ring.
/// * **compressed (non-overlap `ratio ∉ {0,4}`) layers** additionally emit either
///   `ck.{il}`/`cg.{il}` `[1, coff·head_dim]` (the raw compressor projections, on a
///   non-firing step → accumulate into the running window buffer) OR `comp.{il}`
///   `[1, head_dim]` (on a firing step, `(pos+1) % ratio == 0` → the pooled
///   RMSNorm'd compressed KV, append to the compressed cache + reset the buffer).
///   The pooled window reuses [`build_kv_compressor_pool`] (no RoPE — the prefill
///   stage doesn't RoPE `comp`, so decode matches). Inputs the host supplies:
///   `kvcache.{il}` `[cache_len, hd]`, `compcache.{il}` `[pos/ratio, hd]`, and on a
///   firing step `partial_ck.{il}`/`partial_cg.{il}` `[ratio-1, coff·hd]`.
///
/// **Overlapping (`ratio == 4`)** layers use [`build_overlap_pool_single`] with a
/// host-shifted previous-window state (`prev_kv`/`prev_score.{il}`); the learned
/// **Indexer** is skipped (a no-op while `ncomp ≤ index_topk`, i.e. context ≤
/// `index_topk·ratio` = 2048 — exactly what the prefill does), and the builder
/// **errors** if that budget is exceeded so it can't silently mis-run long context.
pub fn build_deepseek_v4_decode(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    pos: usize,
    cache_len: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v4_decode_moe(spec, weights, pos, cache_len, packed, false)
}

/// [`build_deepseek_v4_decode`] with an optional **paged MoE split**: when
/// `paged_moe` is set, each routed-MoE layer does NOT emit the in-graph grouped MoE
/// (all experts resident); instead it outputs `moe_in.{il}` (the post-`ffn_norm`
/// hidden) and takes a `moe_out.{il}` input that the host fills with
/// [`PagedGroupedMoe`] (+ shared expert). The driver runs the graph once per MoE
/// layer, feeding back each layer's MoE result — so only the active experts ever
/// touch memory. Dense layers are unaffected.
pub fn build_deepseek_v4_decode_moe(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    pos: usize,
    cache_len: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    paged_moe: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    let mut g = Graph::new("deepseek_v4_decode");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let d = spec.dim;
    let hc = spec.hc_mult;
    let nh = spec.n_heads;
    let hd = spec.head_dim;
    let rd = spec.rope_head_dim & !1;
    let ql = spec.q_lora_rank;
    let eps = spec.rms_norm_eps;
    let scale = (hd as f32).powf(-0.5);
    let half = (rd / 2).max(1);
    let zb_d = synth_zero(&mut g, &mut params, "v4d.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v4d.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v4d.zb.hd", hd);
    let q_ones = synth_const(&mut g, &mut params, "v4d.qones", vec![1f32; hd], &[hd]);

    // RoPE at the single absolute position `pos` (sliding base, no YaRN — decode is
    // only wired for ratio==0 layers). cos/sin/sininv are `[1, half]`.
    let (mut cosd, mut sind) = (vec![0f32; half], vec![0f32; half]);
    for i in 0..half {
        let fr = spec.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
        let (s, c) = (pos as f64 * fr).sin_cos();
        cosd[i] = c as f32;
        sind[i] = s as f32;
    }
    let sinneg: Vec<f32> = sind.iter().map(|v| -v).collect();
    let cos_p = synth_const(&mut g, &mut params, "v4d.cos", cosd, &[1, half]);
    let sin_p = synth_const(&mut g, &mut params, "v4d.sin", sind, &[1, half]);
    let sininv_p = synth_const(&mut g, &mut params, "v4d.sininv", sinneg, &[1, half]);

    // Embed the single new token → HC-expand to `hc` streams.
    let token = g.input("token_id", Shape::new(&[1], DType::I32));
    let token2 = g.reshape_(token, vec![1, 1]);
    let input_ids_flat = g.reshape_(token, vec![1]);
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let h0 = g.gather_(embed_w, token2, 0); // [1,1,d]
    let h0 = g.reshape_(h0, vec![1, 1, d as i64]);
    let ones_hc = synth_const(
        &mut g,
        &mut params,
        "v4d.hc.ones",
        vec![1f32; hc],
        &[1, hc, 1],
    );
    let mut h = g.mul(h0, ones_hc); // [1, hc, d]
    // Extra per-layer outputs (name, node), emitted after `logits`.
    let mut extra_outs: Vec<(String, NodeId)> = Vec::new();

    for il in 0..spec.n_layers {
        let ratio = spec.compress_ratios.get(il).copied().unwrap_or(0);
        let lp = format!("model.layers.{il}");
        // ── HC-wrapped MLA attention (single query, cached KV) ──
        let residual = h;
        let fn_a =
            load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn_hc.fn"))?;
        let sc_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.scale"),
            false,
        )?;
        let bs_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.base"),
            false,
        )?;
        let (xa, post_a, comb_a) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            1,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.da"),
        );
        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_norm.weight"),
            0.0,
        )?;
        let xa = g.rms_norm(xa, an, zb_d, eps);
        let wqa = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_a.weight"),
        )?;
        let qr = emit_proj(&mut g, xa, &wqa, Shape::new(&[1, ql], f));
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.q_norm.weight"),
            0.0,
        )?;
        let qr = g.rms_norm(qr, qn, zb_ql, eps);
        let wqb = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_b.weight"),
        )?;
        let q = emit_proj(&mut g, qr, &wqb, Shape::new(&[1, nh * hd], f));
        let q = per_head_rms(&mut g, q, q_ones, zb_hd, 1, 1, nh, hd, eps);
        let q = rope_tail(&mut g, q, cos_p, sin_p, 1, nh, hd, rd);
        // New token's KV latent (roped) — this is what the host caches.
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[1, hd], f));
        let kvn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let kv = g.rms_norm(kv, kvn, zb_hd, eps);
        let kv_new = rope_tail(&mut g, kv, cos_p, sin_p, 1, 1, hd, rd); // [1, hd]
        extra_outs.push((
            format!("kvnew.{il}"),
            g.reshape_(kv_new, vec![1, hd as i64]),
        ));
        // Window key set = cached window ++ new (all valid — host holds only valid slots).
        let win_kv = if cache_len > 0 {
            let kvc = g.input(format!("kvcache.{il}"), Shape::new(&[cache_len, hd], f));
            g.concat_(vec![kvc, kv_new], 0)
        } else {
            kv_new
        };
        let win_keys = cache_len + 1;
        // Append the compressed KV cache for compress layers (non-overlap).
        let (kv_all, n_keys) = if ratio == 0 {
            (win_kv, win_keys)
        } else {
            let overlap = ratio == 4;
            let coff = if overlap { 2 } else { 1 };
            let firing = (pos + 1).is_multiple_of(ratio);
            let ncomp_before = pos / ratio;
            let ncomp_vis = if firing {
                ncomp_before + 1
            } else {
                ncomp_before
            };
            // The Indexer only prunes when ncomp > index_topk; below that it keeps
            // every causally-valid compressed position (exact, = the deterministic
            // mask, matching prefill's `indexer_on` gate). Refuse the pruning regime.
            if overlap && spec.index_head_dim > 0 && ncomp_vis > spec.index_topk {
                return Err(anyhow!(
                    "build_deepseek_v4_decode: Indexer top-k pruning (ncomp {ncomp_vis} > \
                     index_topk {}) not yet implemented (layer {il}); exact only for context \
                     <= index_topk*ratio",
                    spec.index_topk
                ));
            }
            // Compressor projections for the new token: ck/cg = wkv/wgate(xa).
            let cw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wkv.weight"),
                true,
            )?;
            let ck_t = g.mm(xa, cw); // [1, coff*hd]
            let cgw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wgate.weight"),
                true,
            )?;
            let cg_raw = g.mm(xa, cgw);
            // Overlap adds APE per-token here; non-overlap lets build_kv_compressor_pool
            // add the whole-window APE.
            let cg_t = if overlap {
                let ape = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.ape"),
                    false,
                )?;
                let ape_row = g.narrow_(ape, 0, pos % ratio, 1); // [1, coff*hd]
                g.add(cg_raw, ape_row)
            } else {
                cg_raw
            };
            let compcache = if ncomp_before > 0 {
                Some(g.input(
                    format!("compcache.{il}"),
                    Shape::new(&[ncomp_before, hd], f),
                ))
            } else {
                None
            };
            let comp_all = if firing {
                let cnorm = load_norm(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.norm.weight"),
                    0.0,
                )?;
                let (win_ck, win_cg) = if ratio > 1 {
                    let pck = g.input(
                        format!("partial_ck.{il}"),
                        Shape::new(&[ratio - 1, coff * hd], f),
                    );
                    let pcg = g.input(
                        format!("partial_cg.{il}"),
                        Shape::new(&[ratio - 1, coff * hd], f),
                    );
                    (g.concat_(vec![pck, ck_t], 0), g.concat_(vec![pcg, cg_t], 0))
                } else {
                    (ck_t, cg_t)
                };
                let compressed_t = if overlap {
                    // Overlap pool needs the previous window (host-shifted; window 0 =
                    // 0 kv / -1e30 score → masked).
                    let prev_kv =
                        g.input(format!("prev_kv.{il}"), Shape::new(&[ratio, coff * hd], f));
                    let prev_score = g.input(
                        format!("prev_score.{il}"),
                        Shape::new(&[ratio, coff * hd], f),
                    );
                    build_overlap_pool_single(
                        &mut g,
                        &mut params,
                        prev_kv,
                        prev_score,
                        win_ck,
                        win_cg,
                        cnorm,
                        hd,
                        eps,
                        &format!("{lp}.dov"),
                    )
                } else {
                    let ape = load_p(
                        &mut g,
                        &mut params,
                        weights,
                        &format!("{lp}.attn.compressor.ape"),
                        false,
                    )?;
                    build_kv_compressor_pool(
                        &mut g,
                        &mut params,
                        win_ck,
                        win_cg,
                        ape,
                        cnorm,
                        1,
                        ratio,
                        ratio,
                        hd,
                        eps,
                        &format!("{lp}.dcomp"),
                    )
                };
                extra_outs.push((
                    format!("comp.{il}"),
                    g.reshape_(compressed_t, vec![1, hd as i64]),
                ));
                // Emit the current token's ck/cg too so the host reconstructs the
                // current window (the next `prev` for overlap; ignored non-overlap).
                extra_outs.push((
                    format!("ck.{il}"),
                    g.reshape_(ck_t, vec![1, (coff * hd) as i64]),
                ));
                extra_outs.push((
                    format!("cg.{il}"),
                    g.reshape_(cg_t, vec![1, (coff * hd) as i64]),
                ));
                let all = match compcache {
                    Some(c) => g.concat_(vec![c, compressed_t], 0),
                    None => compressed_t,
                };
                Some(all)
            } else {
                extra_outs.push((
                    format!("ck.{il}"),
                    g.reshape_(ck_t, vec![1, (coff * hd) as i64]),
                ));
                extra_outs.push((
                    format!("cg.{il}"),
                    g.reshape_(cg_t, vec![1, (coff * hd) as i64]),
                ));
                compcache
            };
            match comp_all {
                Some(c) if ncomp_vis > 0 => (g.concat_(vec![win_kv, c], 0), win_keys + ncomp_vis),
                _ => (win_kv, win_keys),
            }
        };
        let mask = synth_const(
            &mut g,
            &mut params,
            &format!("{lp}.v4d.mask"),
            vec![0f32; n_keys],
            &[1, n_keys],
        );
        let sink = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.attn_sink"),
            false,
        )?;
        let q3 = g.reshape_(q, vec![1, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            scale,
            1,
            nh,
            hd,
            n_keys,
            &format!("{lp}.dsa"),
        );
        let o_flat = g.reshape_(o, vec![1, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos_p, sininv_p, 1, nh, hd, rd);
        let dpg = nh * hd / spec.n_groups;
        let woa = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wob = load_transposed_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_b.weight"),
        )?;
        let attn_out = build_v4_o_lora(
            &mut g,
            o_inv,
            woa,
            wob,
            1,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        h = build_hc_post(&mut g, attn_out, residual, post_a, comb_a, 1, hc, d);

        // ── HC-wrapped FFN (dense SwiGLU or sqrtsoftplus MoE) ──
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.ffn_hc.fn"))?;
        let sc_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.scale"),
            false,
        )?;
        let bs_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.base"),
            false,
        )?;
        let (xf, post_f, comb_f) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            1,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.df"),
        );
        let fnorm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = if il < spec.first_k_dense_replace {
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.up_proj.weight"),
            )?;
            let dn = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.down_proj.weight"),
            )?;
            let gate = emit_proj(&mut g, xf, &gp, Shape::new(&[1, spec.intermediate_size], f));
            let upv = emit_proj(&mut g, xf, &up, Shape::new(&[1, spec.intermediate_size], f));
            let sg = g.silu(gate);
            let glu = g.mul(sg, upv);
            emit_proj(&mut g, glu, &dn, Shape::new(&[1, d], f))
        } else if paged_moe {
            // Split at the MoE boundary: emit the post-`ffn_norm` hidden and receive
            // the host-computed (paged) MoE result. `xf` is `[1, d]`.
            extra_outs.push((format!("moe_in.{il}"), xf));
            g.input(format!("moe_out.{il}"), Shape::new(&[1, d], f))
        } else {
            let ds = v4_moe_spec(spec);
            let x3 = g.reshape_(xf, vec![1, 1, d as i64]);
            let hash_ids = if il < spec.n_hash_layers {
                Some(input_ids_flat)
            } else {
                None
            };
            let moe = build_deepseek_moe_ffn(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                x3,
                1,
                1,
                &ds,
                hash_ids,
            )?;
            g.reshape_(moe, vec![1, d as i64])
        };
        h = build_hc_post(&mut g, ffn_out, residual, post_f, comb_f, 1, hc, d);
    }

    // Head: hc_head reduce → norm → lm_head.
    let hfn = load_transposed_param(&mut g, &mut params, weights, "model.hc_head.fn")?;
    let hsc = load_p(&mut g, &mut params, weights, "model.hc_head.scale", false)?;
    let hbs = load_p(&mut g, &mut params, weights, "model.hc_head.base", false)?;
    let x = build_hc_head(
        &mut g,
        &mut params,
        h,
        hfn,
        hsc,
        hbs,
        1,
        hc,
        d,
        spec.hc_eps,
        "v4d.head",
    );
    let fnorm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let x = g.rms_norm(x, fnorm, zb_d, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[1, spec.vocab_size], f));
    let logits = g.reshape_(logits, vec![1, spec.vocab_size as i64]);
    // Outputs: `logits` first, then the per-layer state nodes in emission order.
    let mut outs = vec![logits];
    let mut names = vec!["logits".to_string()];
    for (name, node) in extra_outs {
        names.push(name);
        outs.push(node);
    }
    g.set_outputs(outs);
    Ok((g, params, names))
}

/// **Greedy generation** for DeepSeek-V4 via the KV-cache decode path — the usable
/// end-to-end loop: it decodes the prompt token-by-token (building each layer's
/// window ring + compressor state, so this doubles as the prefill), then emits
/// `n_new` tokens one at a time by feeding back the argmax. This is the O(n)-per-
/// token path (vs re-running the whole forward each step). `make_loader` returns a
/// fresh [`WeightLoader`] per step (the current builder consumes weights via
/// `take`; a lazy loader re-opens cheaply — a fixed-max-cache compile-once runner
/// is the perf follow-up). Correct for context ≤ `index_topk·ratio` (the Indexer
/// is a no-op there; the decode builder errors past it). Returns the `n_new`
/// generated token ids. The sliding-window ring keeps the last `window_size - 1`
/// KV latents so the query sees exactly `window_size` positions (matching prefill's
/// `qi - ki < window` mask).
pub fn deepseek_v4_generate(
    spec: &DeepseekV4Spec,
    mut make_loader: impl FnMut() -> Box<dyn WeightLoader>,
    device: rlx_runtime::Device,
    prompt_ids: &[u32],
    n_new: usize,
) -> Result<Vec<u32>> {
    use rlx_runtime::Session;
    let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        device,
    );
    let nl = spec.n_layers;
    let hd = spec.head_dim;
    let window = spec.window_size.max(1);
    let ratios: Vec<usize> = (0..nl)
        .map(|il| spec.compress_ratios.get(il).copied().unwrap_or(0))
        .collect();
    let coffs: Vec<usize> = ratios.iter().map(|&r| if r == 4 { 2 } else { 1 }).collect();
    let mut win_ring = vec![Vec::<f32>::new(); nl];
    let mut compcache = vec![Vec::<f32>::new(); nl];
    let mut partial_ck = vec![Vec::<f32>::new(); nl];
    let mut partial_cg = vec![Vec::<f32>::new(); nl];
    let mut prev_kv = vec![Vec::<f32>::new(); nl];
    let mut prev_score = vec![Vec::<f32>::new(); nl];
    for il in 0..nl {
        if ratios[il] == 4 {
            prev_kv[il] = vec![0f32; ratios[il] * coffs[il] * hd];
            prev_score[il] = vec![-1e30f32; ratios[il] * coffs[il] * hd];
        }
    }
    // One decode step at absolute position `pos` for `token`; returns its logits and
    // threads the per-layer window ring + compressor state forward.
    let mut step = |pos: usize, token: u32| -> Result<Vec<f32>> {
        let cache_len = if nl > 0 { win_ring[0].len() / hd } else { 0 }; // = min(pos, window-1)
        let mut loader = make_loader();
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (g, params, onames) =
            build_deepseek_v4_decode(spec, &mut *loader, pos, cache_len, &mut packed)?;
        let mut compiled = Session::new(device).compile_with(g, &opts);
        for (n, dd) in &params {
            compiled.set_param(n, dd);
        }
        // Bind packed MoE/quant codes (+ bf16 scale/bias slabs) as raw bytes — a
        // MoE checkpoint's `switch_mlp` experts live here, not in `params`.
        for (n, (b, _s, _)) in &packed {
            compiled.set_param_typed(n, b, DType::U8);
        }
        let mut owned: Vec<(String, Vec<f32>)> = vec![("token_id".into(), vec![token as f32])];
        for il in 0..nl {
            if cache_len > 0 {
                owned.push((format!("kvcache.{il}"), win_ring[il].clone()));
            }
            let ratio = ratios[il];
            if ratio > 0 {
                let firing = (pos + 1).is_multiple_of(ratio);
                if pos / ratio > 0 {
                    owned.push((format!("compcache.{il}"), compcache[il].clone()));
                }
                if firing && ratio > 1 {
                    owned.push((format!("partial_ck.{il}"), partial_ck[il].clone()));
                    owned.push((format!("partial_cg.{il}"), partial_cg[il].clone()));
                }
                if firing && ratio == 4 {
                    owned.push((format!("prev_kv.{il}"), prev_kv[il].clone()));
                    owned.push((format!("prev_score.{il}"), prev_score[il].clone()));
                }
            }
        }
        let inputs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let out = compiled.run(&inputs);
        let get = |name: &str| -> Vec<f32> {
            let i = onames
                .iter()
                .position(|x| x == name)
                .expect("decode output");
            out[i].clone()
        };
        let logits = get("logits");
        let maxlen = (window - 1) * hd;
        for il in 0..nl {
            win_ring[il].extend_from_slice(&get(&format!("kvnew.{il}")));
            if win_ring[il].len() > maxlen {
                let drop = win_ring[il].len() - maxlen;
                win_ring[il].drain(0..drop);
            }
            let ratio = ratios[il];
            if ratio > 0 {
                if (pos + 1).is_multiple_of(ratio) {
                    compcache[il].extend_from_slice(&get(&format!("comp.{il}")));
                    if ratio == 4 {
                        let mut ckw = partial_ck[il].clone();
                        ckw.extend_from_slice(&get(&format!("ck.{il}")));
                        let mut cgw = partial_cg[il].clone();
                        cgw.extend_from_slice(&get(&format!("cg.{il}")));
                        prev_kv[il] = ckw;
                        prev_score[il] = cgw;
                    }
                    partial_ck[il].clear();
                    partial_cg[il].clear();
                } else {
                    partial_ck[il].extend_from_slice(&get(&format!("ck.{il}")));
                    partial_cg[il].extend_from_slice(&get(&format!("cg.{il}")));
                }
            }
        }
        Ok(logits)
    };
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    // Prefill the prompt through the decode path (builds the cache); the last
    // token's logits seed the first generated token.
    let mut logits = vec![0f32; spec.vocab_size];
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        logits = step(pos, tok)?;
    }
    let mut out_ids = Vec::with_capacity(n_new);
    for i in 0..n_new {
        let next = argmax(&logits);
        out_ids.push(next);
        logits = step(prompt_ids.len() + i, next)?;
    }
    Ok(out_ids)
}

/// **Paged-MoE** greedy generation — the end-to-end decode with the MoE **split out
/// of the graph** and run host-side (attention stays in-graph). Mirrors
/// [`deepseek_v4_generate`] but builds the graph in paged mode
/// ([`build_deepseek_v4_decode_moe`] with `paged_moe = true`): each routed-MoE layer
/// emits `moe_in.{il}` (its post-`ffn_norm` hidden) and consumes `moe_out.{il}`.
/// Each decode step runs the graph once per MoE layer, in layer order, filling
/// `moe_out.{il}` from `moe_fn(il, token, xf)` — wire that to [`PagedGroupedMoe`] (+
/// shared expert) so only the active experts ever touch memory. `moe_fn` receives the
/// layer index, the **current decode token id** (for the first `n_hash_layers`, whose
/// experts come from `gate.tid2eid[token]` rather than score top-k — use
/// [`hash_route_experts`]), and the `[dim]` hidden; it returns the `[dim]` MoE result.
/// Dense layers add no passes. Numerically identical to [`deepseek_v4_generate`] when
/// `moe_fn` reproduces the in-graph MoE.
pub fn deepseek_v4_generate_paged(
    spec: &DeepseekV4Spec,
    mut make_loader: impl FnMut() -> Box<dyn WeightLoader>,
    device: rlx_runtime::Device,
    prompt_ids: &[u32],
    n_new: usize,
    mut moe_fn: impl FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
) -> Result<Vec<u32>> {
    use rlx_runtime::Session;
    let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        device,
    );
    let (nl, hd, d) = (spec.n_layers, spec.head_dim, spec.dim);
    let window = spec.window_size.max(1);
    let ratios: Vec<usize> = (0..nl)
        .map(|il| spec.compress_ratios.get(il).copied().unwrap_or(0))
        .collect();
    let coffs: Vec<usize> = ratios.iter().map(|&r| if r == 4 { 2 } else { 1 }).collect();
    let moe_layers: Vec<usize> = (0..nl)
        .filter(|&il| il >= spec.first_k_dense_replace)
        .collect();
    let (mut win_ring, mut compcache) = (vec![Vec::<f32>::new(); nl], vec![Vec::<f32>::new(); nl]);
    let (mut partial_ck, mut partial_cg) =
        (vec![Vec::<f32>::new(); nl], vec![Vec::<f32>::new(); nl]);
    let (mut prev_kv, mut prev_score) = (vec![Vec::<f32>::new(); nl], vec![Vec::<f32>::new(); nl]);
    for il in 0..nl {
        if ratios[il] == 4 {
            prev_kv[il] = vec![0f32; ratios[il] * coffs[il] * hd];
            prev_score[il] = vec![-1e30f32; ratios[il] * coffs[il] * hd];
        }
    }
    let mut step = |pos: usize, token: u32| -> Result<Vec<f32>> {
        let cache_len = if nl > 0 { win_ring[0].len() / hd } else { 0 };
        let mut loader = make_loader();
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (g, params, onames) =
            build_deepseek_v4_decode_moe(spec, &mut *loader, pos, cache_len, &mut packed, true)?;
        let mut compiled = Session::new(device).compile_with(g, &opts);
        for (n, dd) in &params {
            compiled.set_param(n, dd);
        }
        // Bind packed codes for any in-graph quant (attention/dense); routed-MoE
        // experts are paged out of the graph so none appear here.
        for (n, (b, _s, _)) in &packed {
            compiled.set_param_typed(n, b, DType::U8);
        }
        // Base (non-MoE) inputs — identical across every pass this step.
        let mut base: Vec<(String, Vec<f32>)> = vec![("token_id".into(), vec![token as f32])];
        for il in 0..nl {
            if cache_len > 0 {
                base.push((format!("kvcache.{il}"), win_ring[il].clone()));
            }
            let ratio = ratios[il];
            if ratio > 0 {
                let firing = (pos + 1).is_multiple_of(ratio);
                if pos / ratio > 0 {
                    base.push((format!("compcache.{il}"), compcache[il].clone()));
                }
                if firing && ratio > 1 {
                    base.push((format!("partial_ck.{il}"), partial_ck[il].clone()));
                    base.push((format!("partial_cg.{il}"), partial_cg[il].clone()));
                }
                if firing && ratio == 4 {
                    base.push((format!("prev_kv.{il}"), prev_kv[il].clone()));
                    base.push((format!("prev_score.{il}"), prev_score[il].clone()));
                }
            }
        }
        // Assemble inputs = base ++ current moe_out.{il}; run one full pass.
        let mut moe_out: HashMap<usize, Vec<f32>> =
            moe_layers.iter().map(|&il| (il, vec![0f32; d])).collect();
        let run = |compiled: &mut rlx_runtime::CompiledGraph,
                   moe_out: &HashMap<usize, Vec<f32>>| {
            let mut owned = base.clone();
            for &il in &moe_layers {
                owned.push((format!("moe_out.{il}"), moe_out[&il].clone()));
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            compiled.run(&inputs)
        };
        // One pass per MoE layer (causal: `moe_in.{il}` is valid once layers < il are
        // filled), then a final pass for logits + committed KV.
        for &il in &moe_layers {
            let out = run(&mut compiled, &moe_out);
            let xf_i = onames
                .iter()
                .position(|x| *x == format!("moe_in.{il}"))
                .expect("moe_in");
            let xf = out[xf_i].clone();
            moe_out.insert(il, moe_fn(il, token, &xf)?);
        }
        let out = run(&mut compiled, &moe_out);
        let get = |name: &str| -> Vec<f32> {
            out[onames
                .iter()
                .position(|x| x == name)
                .expect("decode output")]
            .clone()
        };
        let logits = get("logits");
        let maxlen = (window - 1) * hd;
        for il in 0..nl {
            win_ring[il].extend_from_slice(&get(&format!("kvnew.{il}")));
            if win_ring[il].len() > maxlen {
                let drop = win_ring[il].len() - maxlen;
                win_ring[il].drain(0..drop);
            }
            let ratio = ratios[il];
            if ratio > 0 {
                if (pos + 1).is_multiple_of(ratio) {
                    compcache[il].extend_from_slice(&get(&format!("comp.{il}")));
                    if ratio == 4 {
                        let mut ckw = partial_ck[il].clone();
                        ckw.extend_from_slice(&get(&format!("ck.{il}")));
                        let mut cgw = partial_cg[il].clone();
                        cgw.extend_from_slice(&get(&format!("cg.{il}")));
                        prev_kv[il] = ckw;
                        prev_score[il] = cgw;
                    }
                    partial_ck[il].clear();
                    partial_cg[il].clear();
                } else {
                    partial_ck[il].extend_from_slice(&get(&format!("ck.{il}")));
                    partial_cg[il].extend_from_slice(&get(&format!("cg.{il}")));
                }
            }
        }
        Ok(logits)
    };
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    let mut logits = vec![0f32; spec.vocab_size];
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        logits = step(pos, tok)?;
    }
    let mut out_ids = Vec::with_capacity(n_new);
    for i in 0..n_new {
        let next = argmax(&logits);
        out_ids.push(next);
        logits = step(prompt_ids.len() + i, next)?;
    }
    Ok(out_ids)
}

/// **Fixed-shape** single-token decode graph for a **compile-once** runner
/// ([`V4Decoder`]). Unlike [`build_deepseek_v4_decode`] — which bakes `pos` into
/// the RoPE table and branches on firing, so it recompiles every token — this has
/// a stable shape: `cos`/`sin`/`sininv` and the per-compress-layer APE row are
/// **inputs**; every layer's window cache is a fixed `[max_win, hd]` and its
/// compressed cache `[max_comp, hd]`, both padded and gated by a per-layer
/// validity `mask.{il}` input; and the compressor **always pools** into one fixed
/// extra key slot that the mask marks valid only on a firing step. So it compiles
/// ONCE and runs per token. Weights load once (into the returned params). Outputs:
/// `logits` + per-layer `kvnew.{il}`, and per-compress-layer `comp.{il}` (the
/// pooled KV — the host appends it on a firing step) + `ck.{il}`/`cg.{il}` (raw
/// projections the host accumulates). Key order per compress layer:
/// `[window(max_win) ++ new(1) ++ compcache(max_comp) ++ compressed(1)]`.
pub fn build_deepseek_v4_decode_fixed(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v4_decode_fixed_stage(
        spec,
        weights,
        0..spec.n_layers,
        true,
        true,
        max_win,
        max_comp,
        packed,
    )
}

/// One **layer-range STAGE** of the fixed-shape decode graph — the distributed
/// decode building block (each cluster node builds only its `layers` and relays
/// the single-token hidden boundary `[1, hc_mult, dim]`, mirroring
/// [`build_deepseek_v4_stage`] for prefill). `first` embeds the token; `last`
/// applies the head → `logits`; otherwise the stage takes a `hidden_in` input and
/// emits a `hidden_out` output. Hash-routing layers must live on the `first`
/// stage (only it has the token id). Each node holds the KV cache for its own
/// `layers`.
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v4_decode_fixed_stage(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    layers: std::ops::Range<usize>,
    first: bool,
    last: bool,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v4_decode_fixed_stage_moe(
        spec, weights, layers, first, last, max_win, max_comp, packed, false, false,
    )
}

/// [`build_deepseek_v4_decode_fixed_stage`] with the **compile-once PAGED MoE split**:
/// `paged_moe` makes each routed-MoE layer emit `moe_in.{il}` + take `moe_out.{il}`
/// (as in [`build_deepseek_v4_decode_moe`]) but on the FIXED-shape graph — so the
/// backbone compiles ONCE (pos/mask/APE are inputs) and the host fills the MoE per
/// token, avoiding the O(L²) per-token recompile of `deepseek_v4_generate_paged`.
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v4_decode_fixed_stage_moe(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    layers: std::ops::Range<usize>,
    first: bool,
    last: bool,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    paged_moe: bool,
    // Attn/post split: build ONLY the attention side of a single MoE layer — emit
    // `moe_in` + the `build_hc_post` inputs (`hcp_residual`/`hcp_postf`/`hcp_combf`)
    // and STOP before the final `hc_post`/head, which move to [`build_v4_post_stage`].
    // Removes the O(2L) two-pass attention recompute. Only for single-layer paged
    // MoE stages (the layerwise driver); a no-op otherwise.
    split_attn: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    let mut g = Graph::new("deepseek_v4_decode_fixed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let d = spec.dim;
    let hc = spec.hc_mult;
    let nh = spec.n_heads;
    let hd = spec.head_dim;
    let rd = spec.rope_head_dim & !1;
    let ql = spec.q_lora_rank;
    let eps = spec.rms_norm_eps;
    let scale = (hd as f32).powf(-0.5);
    let half = (rd / 2).max(1);
    let zb_d = synth_zero(&mut g, &mut params, "v4f.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v4f.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v4f.zb.hd", hd);
    let q_ones = synth_const(&mut g, &mut params, "v4f.qones", vec![1f32; hd], &[hd]);
    // RoPE tables for the current position are INPUTS (host recomputes per step).
    let cos_p = g.input("cos", Shape::new(&[1, half], f));
    let sin_p = g.input("sin", Shape::new(&[1, half], f));
    let sininv_p = g.input("sininv", Shape::new(&[1, half], f));

    // First stage embeds the token + HC-expands; later stages take the relayed
    // hidden boundary. `input_ids_flat` (hash routing) exists only on the first
    // stage — hash layers must live there.
    let (mut h, input_ids_flat): (NodeId, Option<NodeId>) = if first {
        let token = g.input("token_id", Shape::new(&[1], DType::I32));
        let token2 = g.reshape_(token, vec![1, 1]);
        let ids_flat = g.reshape_(token, vec![1]);
        let (embed_w, _, _) =
            load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
        let h0 = g.gather_(embed_w, token2, 0);
        let h0 = g.reshape_(h0, vec![1, 1, d as i64]);
        let ones_hc = synth_const(
            &mut g,
            &mut params,
            "v4f.hc.ones",
            vec![1f32; hc],
            &[1, hc, 1],
        );
        (g.mul(h0, ones_hc), Some(ids_flat))
    } else {
        (g.input("hidden_in", Shape::new(&[1, hc, d], f)), None)
    };
    let mut extra_outs: Vec<(String, NodeId)> = Vec::new();
    // Set true once a split-attn MoE layer emits its `hcp_*` boundary — then the
    // final `hc_post`/head/`main_hidden` are skipped (they live in the post graph).
    let mut attn_only = false;

    for il in layers.clone() {
        let ratio = spec.compress_ratios.get(il).copied().unwrap_or(0);
        let overlap = ratio == 4;
        let coff = if overlap { 2 } else { 1 };
        let lp = format!("model.layers.{il}");
        // HC-wrapped MLA attention.
        let residual = h;
        let fn_a =
            load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn_hc.fn"))?;
        let sc_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.scale"),
            false,
        )?;
        let bs_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.base"),
            false,
        )?;
        let (xa, post_a, comb_a) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            1,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.fa"),
        );
        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_norm.weight"),
            0.0,
        )?;
        let xa = g.rms_norm(xa, an, zb_d, eps);
        let wqa = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_a.weight"),
        )?;
        let qr = emit_proj(&mut g, xa, &wqa, Shape::new(&[1, ql], f));
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.q_norm.weight"),
            0.0,
        )?;
        let qr = g.rms_norm(qr, qn, zb_ql, eps);
        let wqb = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_b.weight"),
        )?;
        let q = emit_proj(&mut g, qr, &wqb, Shape::new(&[1, nh * hd], f));
        let q = per_head_rms(&mut g, q, q_ones, zb_hd, 1, 1, nh, hd, eps);
        let q = rope_tail(&mut g, q, cos_p, sin_p, 1, nh, hd, rd);
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[1, hd], f));
        let kvn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let kv = g.rms_norm(kv, kvn, zb_hd, eps);
        let kv_new = rope_tail(&mut g, kv, cos_p, sin_p, 1, 1, hd, rd);
        extra_outs.push((
            format!("kvnew.{il}"),
            g.reshape_(kv_new, vec![1, hd as i64]),
        ));

        let wcache = g.input(format!("wcache.{il}"), Shape::new(&[max_win, hd], f));
        let (kv_all, n_keys) = if ratio == 0 {
            (g.concat_(vec![wcache, kv_new], 0), max_win + 1)
        } else {
            // Compressor projections + always-pool (result masked unless firing).
            let cw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wkv.weight"),
                true,
            )?;
            let ck_t = g.mm(xa, cw);
            let cgw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wgate.weight"),
                true,
            )?;
            let cg_raw = g.mm(xa, cgw);
            let cnorm = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.norm.weight"),
                0.0,
            )?;
            // Overlap adds the (host-provided) APE row per token; non-overlap lets
            // build_kv_compressor_pool add the whole-window APE.
            let cg_emit = if overlap {
                let ape_row = g.input(format!("ape_row.{il}"), Shape::new(&[1, coff * hd], f));
                g.add(cg_raw, ape_row)
            } else {
                cg_raw
            };
            let pck = g.input(
                format!("partial_ck.{il}"),
                Shape::new(&[ratio - 1, coff * hd], f),
            );
            let pcg = g.input(
                format!("partial_cg.{il}"),
                Shape::new(&[ratio - 1, coff * hd], f),
            );
            let win_ck = g.concat_(vec![pck, ck_t], 0);
            let win_cg = g.concat_(vec![pcg, cg_emit], 0);
            let compressed_t = if overlap {
                let prev_kv = g.input(format!("prev_kv.{il}"), Shape::new(&[ratio, coff * hd], f));
                let prev_score = g.input(
                    format!("prev_score.{il}"),
                    Shape::new(&[ratio, coff * hd], f),
                );
                build_overlap_pool_single(
                    &mut g,
                    &mut params,
                    prev_kv,
                    prev_score,
                    win_ck,
                    win_cg,
                    cnorm,
                    hd,
                    eps,
                    &format!("{lp}.fov"),
                )
            } else {
                let ape = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.ape"),
                    false,
                )?;
                build_kv_compressor_pool(
                    &mut g,
                    &mut params,
                    win_ck,
                    win_cg,
                    ape,
                    cnorm,
                    1,
                    ratio,
                    ratio,
                    hd,
                    eps,
                    &format!("{lp}.fcomp"),
                )
            };
            extra_outs.push((
                format!("comp.{il}"),
                g.reshape_(compressed_t, vec![1, hd as i64]),
            ));
            extra_outs.push((
                format!("ck.{il}"),
                g.reshape_(ck_t, vec![1, (coff * hd) as i64]),
            ));
            extra_outs.push((
                format!("cg.{il}"),
                g.reshape_(cg_emit, vec![1, (coff * hd) as i64]),
            ));
            let compcache = g.input(format!("compcache.{il}"), Shape::new(&[max_comp, hd], f));
            (
                g.concat_(vec![wcache, kv_new, compcache, compressed_t], 0),
                max_win + 1 + max_comp + 1,
            )
        };
        let mask = g.input(format!("mask.{il}"), Shape::new(&[1, n_keys], f));
        let sink = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.attn_sink"),
            false,
        )?;
        let q3 = g.reshape_(q, vec![1, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            scale,
            1,
            nh,
            hd,
            n_keys,
            &format!("{lp}.fsa"),
        );
        let o_flat = g.reshape_(o, vec![1, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos_p, sininv_p, 1, nh, hd, rd);
        let dpg = nh * hd / spec.n_groups;
        let woa = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wob = load_transposed_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_b.weight"),
        )?;
        let attn_out = build_v4_o_lora(
            &mut g,
            o_inv,
            woa,
            wob,
            1,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        h = build_hc_post(&mut g, attn_out, residual, post_a, comb_a, 1, hc, d);

        // HC-wrapped FFN.
        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.ffn_hc.fn"))?;
        let sc_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.scale"),
            false,
        )?;
        let bs_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.base"),
            false,
        )?;
        let (xf, post_f, comb_f) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            1,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.ff"),
        );
        let fnorm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = if il < spec.first_k_dense_replace {
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.up_proj.weight"),
            )?;
            let dn = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.down_proj.weight"),
            )?;
            let gate = emit_proj(&mut g, xf, &gp, Shape::new(&[1, spec.intermediate_size], f));
            let upv = emit_proj(&mut g, xf, &up, Shape::new(&[1, spec.intermediate_size], f));
            let sg = g.silu(gate);
            let glu = g.mul(sg, upv);
            emit_proj(&mut g, glu, &dn, Shape::new(&[1, d], f))
        } else if paged_moe && split_attn {
            // Attn/post split: emit `moe_in` + the `hc_post` inputs and STOP here —
            // `build_v4_post_stage` finishes the layer once the host MoE fills moe_out.
            extra_outs.push((format!("moe_in.{il}"), xf));
            extra_outs.push((
                format!("hcp_residual.{il}"),
                g.reshape_(residual, vec![1, (hc * d) as i64]),
            ));
            extra_outs.push((
                format!("hcp_postf.{il}"),
                g.reshape_(post_f, vec![1, hc as i64]),
            ));
            extra_outs.push((
                format!("hcp_combf.{il}"),
                g.reshape_(comb_f, vec![1, (hc * hc) as i64]),
            ));
            attn_only = true;
            break;
        } else if paged_moe {
            // Compile-once split: emit the post-ffn_norm hidden, take the host MoE.
            extra_outs.push((format!("moe_in.{il}"), xf));
            g.input(format!("moe_out.{il}"), Shape::new(&[1, d], f))
        } else {
            let ds = v4_moe_spec(spec);
            let x3 = g.reshape_(xf, vec![1, 1, d as i64]);
            let hash_ids = if il < spec.n_hash_layers {
                input_ids_flat
            } else {
                None
            };
            let moe = build_deepseek_moe_ffn(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                x3,
                1,
                1,
                &ds,
                hash_ids,
            )?;
            g.reshape_(moe, vec![1, d as i64])
        };
        h = build_hc_post(&mut g, ffn_out, residual, post_f, comb_f, 1, hc, d);
        // DSpark: at each target layer, emit the mean-over-HC-streams hidden. The
        // host concatenates these (in target order) into `main_hidden`, which the
        // DSpark drafter's `main_proj` consumes (Transformer.forward collects
        // `h.mean(dim=2)` at `dspark_target_layer_ids`).
        if spec.dspark_target_layer_ids.contains(&il) {
            let mh = g.mean(h, vec![1], false); // [1, d] over the hc streams
            extra_outs.push((
                format!("main_hidden.{il}"),
                g.reshape_(mh, vec![1, d as i64]),
            ));
        }
    }

    // Last stage: hc_head → norm → lm_head → logits. Others relay the hidden.
    let mut outs;
    let mut names;
    if attn_only {
        // Attn/post split: the graph ends at the `hcp_*`/`moe_in` boundary; the post
        // graph produces `hidden_out`/`logits`. Outputs are the extra_outs only.
        outs = Vec::new();
        names = Vec::new();
    } else if last {
        let hfn = load_transposed_param(&mut g, &mut params, weights, "model.hc_head.fn")?;
        let hsc = load_p(&mut g, &mut params, weights, "model.hc_head.scale", false)?;
        let hbs = load_p(&mut g, &mut params, weights, "model.hc_head.base", false)?;
        let x = build_hc_head(
            &mut g,
            &mut params,
            h,
            hfn,
            hsc,
            hbs,
            1,
            hc,
            d,
            spec.hc_eps,
            "v4f.head",
        );
        let fnorm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
        let x = g.rms_norm(x, fnorm, zb_d, eps);
        let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
        let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[1, spec.vocab_size], f));
        let logits = g.reshape_(logits, vec![1, spec.vocab_size as i64]);
        outs = vec![logits];
        names = vec!["logits".to_string()];
    } else {
        // NB the per-stage hidden boundary (`hidden_out`) is CORRECT on CPU; on the
        // Metal backbone (`--attn-gpu`) a non-`logits` graph output of this op sequence
        // reads back as ZEROS — a Metal backend flush/materialization bug (the routed
        // MoE grouped kernel on Metal is unaffected; CPU backbone + GPU MoE is the
        // validated fast path). Keep CPU-correct.
        outs = vec![g.reshape_(h, vec![1, hc as i64, d as i64])];
        names = vec!["hidden_out".to_string()];
    }
    for (name, node) in extra_outs {
        names.push(name);
        outs.push(node);
    }
    g.set_outputs(outs);
    Ok((g, params, names))
}

/// **Attn/post split — the POST graph.** The weight-free `build_hc_post` (+ the
/// `hc_head → norm → lm_head` on the `last` stage) as its own tiny graph, so the
/// expensive attention + `build_hc_pre` runs ONCE per token (in the attn graph,
/// built with `split_attn=true`) instead of TWICE (the old O(2L) two-pass, which
/// recomputed all of attention just to apply `moe_out`). Inputs (the attn graph's
/// `hcp_*` outputs + the host MoE result): `hcp_residual` `[1,hc*d]`, `hcp_postf`
/// `[1,hc]`, `hcp_combf` `[1,hc*hc]`, `moe_out` `[1,d]`. Output: `hidden_out`
/// `[1,hc,d]` (else `logits` on `last`); `main_hidden` `[1,d]` when `target`.
#[allow(clippy::too_many_arguments)]
pub fn build_v4_post_stage(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    last: bool,
    target: bool,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    let mut g = Graph::new("deepseek_v4_post");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hc) = (spec.dim, spec.hc_mult);
    let eps = spec.rms_norm_eps;
    let zb_d = synth_zero(&mut g, &mut params, "v4post.zb.d", d);
    let residual_in = g.input("hcp_residual", Shape::new(&[1, hc * d], f));
    let residual = g.reshape_(residual_in, vec![1, hc as i64, d as i64]);
    let post_f = g.input("hcp_postf", Shape::new(&[1, hc], f));
    let comb_f = g.input("hcp_combf", Shape::new(&[1, hc * hc], f));
    let moe_out = g.input("moe_out", Shape::new(&[1, d], f));
    let h = build_hc_post(&mut g, moe_out, residual, post_f, comb_f, 1, hc, d);
    let mut outs: Vec<NodeId> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    if last {
        let hfn = load_transposed_param(&mut g, &mut params, weights, "model.hc_head.fn")?;
        let hsc = load_p(&mut g, &mut params, weights, "model.hc_head.scale", false)?;
        let hbs = load_p(&mut g, &mut params, weights, "model.hc_head.base", false)?;
        let x = build_hc_head(
            &mut g,
            &mut params,
            h,
            hfn,
            hsc,
            hbs,
            1,
            hc,
            d,
            spec.hc_eps,
            "v4post.head",
        );
        let fnorm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
        let x = g.rms_norm(x, fnorm, zb_d, eps);
        let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
        let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[1, spec.vocab_size], f));
        outs.push(g.reshape_(logits, vec![1, spec.vocab_size as i64]));
        names.push("logits".to_string());
    } else {
        outs.push(g.reshape_(h, vec![1, hc as i64, d as i64]));
        names.push("hidden_out".to_string());
    }
    if target {
        let mh = g.mean(h, vec![1], false);
        outs.push(g.reshape_(mh, vec![1, d as i64]));
        names.push("main_hidden".to_string());
    }
    g.set_outputs(outs);
    Ok((g, params, names))
}

/// **Compile-once** DeepSeek-V4 greedy decoder. Builds the fixed-shape decode
/// graph ([`build_deepseek_v4_decode_fixed`]) and loads weights **once**, then
/// [`Self::step`]s per token feeding padded caches + validity masks — so the graph
/// compiles a single time (vs [`deepseek_v4_generate`], which recompiles every
/// token). `max_win` should be `window_size - 1` (the sliding-window ring depth);
/// `max_comp` bounds the compressed-KV cache (`max_len / smallest_ratio`).
/// Correct while `ncomp ≤ index_topk` (context ≤ `index_topk·ratio`).
pub struct V4Decoder {
    compiled: rlx_runtime::CompiledGraph,
    out_names: Vec<String>,
    vocab: usize,
    dim: usize,
    hd: usize,
    half: usize,
    rd: usize,
    rope_theta: f64,
    max_win: usize,
    max_comp: usize,
    ratios: Vec<usize>,
    coffs: Vec<usize>,
    ape: Vec<Vec<f32>>, // per overlap layer: [ratio, coff*hd]
    win_ring: Vec<Vec<f32>>,
    compcache: Vec<Vec<f32>>,
    partial_ck: Vec<Vec<f32>>,
    partial_cg: Vec<Vec<f32>>,
    prev_kv: Vec<Vec<f32>>,
    prev_score: Vec<Vec<f32>>,
    layers: std::ops::Range<usize>,
    first: bool,
    last: bool,
    target_layers: Vec<usize>,
    main_hidden: Option<Vec<f32>>,
    pos: usize,
    /// Non-empty when built in **paged-MoE** mode ([`Self::new_stage_paged`]): the
    /// MoE layer indices in this stage whose `moe_out.{il}` the host fills each step
    /// (see [`Self::step_paged`]). Empty ⇒ resident MoE in-graph.
    moe_layers: Vec<usize>,
    /// **Attn/post split** ([`Self::new_stage_paged_split`]): when `Some`, the main
    /// `compiled` graph is attention-only (emits `moe_in` + `hcp_*` and stops), and
    /// this weight-free graph ([`build_v4_post_stage`]) applies `moe_out` to produce
    /// `hidden_out`/`logits` — so attention runs ONCE per token, not twice.
    post: Option<rlx_runtime::CompiledGraph>,
    post_names: Vec<String>,
}

/// A checkpoint of a [`V4Decoder`]'s KV cache + position. Take one before
/// speculatively decoding a draft block, then [`V4Decoder::restore`] to roll back
/// the rejected drafts (the KV-rollback primitive for speculative / beam decoding).
#[derive(Clone)]
pub struct V4DecodeSnapshot {
    win_ring: Vec<Vec<f32>>,
    compcache: Vec<Vec<f32>>,
    partial_ck: Vec<Vec<f32>>,
    partial_cg: Vec<Vec<f32>>,
    prev_kv: Vec<Vec<f32>>,
    prev_score: Vec<Vec<f32>>,
    main_hidden: Option<Vec<f32>>,
    pos: usize,
}

impl V4Decoder {
    /// Snapshot the full KV-cache + compressor state + position.
    pub fn snapshot(&self) -> V4DecodeSnapshot {
        V4DecodeSnapshot {
            win_ring: self.win_ring.clone(),
            compcache: self.compcache.clone(),
            partial_ck: self.partial_ck.clone(),
            partial_cg: self.partial_cg.clone(),
            prev_kv: self.prev_kv.clone(),
            prev_score: self.prev_score.clone(),
            main_hidden: self.main_hidden.clone(),
            pos: self.pos,
        }
    }

    /// Roll the decoder back to a [`Self::snapshot`] — discards every token decoded
    /// since (their window/compressed/partial/prev KV state), so a rejected draft
    /// block leaves no trace.
    pub fn restore(&mut self, s: V4DecodeSnapshot) {
        self.win_ring = s.win_ring;
        self.compcache = s.compcache;
        self.partial_ck = s.partial_ck;
        self.partial_cg = s.partial_cg;
        self.prev_kv = s.prev_kv;
        self.prev_score = s.prev_score;
        self.main_hidden = s.main_hidden;
        self.pos = s.pos;
    }

    /// Assemble the batch-verify graph inputs for `block` tokens at the current pos
    /// from this decoder's cache (window/compressed/partial/prev) — the block
    /// analogue of [`Self::step_io`]'s per-token assembly. `max_comp` = the verify
    /// graph's compressed-cache dim. Handles all layer types incl. the in-block
    /// compressor candidate visibility (fire ⇒ visible to that block query onward).
    fn verify_block_inputs(
        &self,
        block: usize,
        btok: &[u32],
        max_comp: usize,
    ) -> Vec<(String, Vec<f32>)> {
        let (hd, half, rd, mw, pos) = (self.hd, self.half.max(1), self.rd, self.max_win, self.pos);
        let (mut cosb, mut sinb, mut sininvb) = (
            vec![0f32; block * half],
            vec![0f32; block * half],
            vec![0f32; block * half],
        );
        for p in 0..block {
            for i in 0..half {
                let fr = self.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
                let (s, c) = ((pos + p) as f64 * fr).sin_cos();
                cosb[p * half + i] = c as f32;
                sinb[p * half + i] = s as f32;
                sininvb[p * half + i] = -(s as f32);
            }
        }
        let mut owned: Vec<(String, Vec<f32>)> = vec![
            ("token_id".into(), btok.iter().map(|&x| x as f32).collect()),
            ("cos".into(), cosb),
            ("sin".into(), sinb),
            ("sininv".into(), sininvb),
        ];
        let padr = |v: &[f32], rows: usize| -> Vec<f32> {
            let mut o = vec![0f32; rows * hd];
            let n = v.len().min(rows * hd);
            o[..n].copy_from_slice(&v[..n]);
            o
        };
        for il in self.layers.clone() {
            let r = self.ratios[il];
            let coff = self.coffs[il];
            let wlen = self.win_ring[il].len() / hd;
            let ncomp = self.compcache[il].len() / hd;
            owned.push((format!("wcache.{il}"), padr(&self.win_ring[il], mw)));
            let nk = if r == 0 {
                mw + block
            } else {
                mw + block + max_comp + block
            };
            let mut m = vec![-1e30f32; block * nk];
            for row in 0..block {
                for s in 0..wlen.min(mw) {
                    m[row * nk + s] = 0.0; // window valid
                }
                for j in 0..=row {
                    m[row * nk + mw + j] = 0.0; // block-causal
                }
                if r > 0 {
                    for c in 0..ncomp.min(max_comp) {
                        m[row * nk + mw + block + c] = 0.0; // existing compressed (all pre-block)
                    }
                    let cand0 = mw + block + max_comp;
                    for j in 0..block {
                        if (pos + j + 1) % r == 0 && j <= row {
                            m[row * nk + cand0 + j] = 0.0; // in-block compressed, visible
                        }
                    }
                }
            }
            owned.push((format!("mask.{il}"), m));
            if r > 0 {
                owned.push((
                    format!("compcache.{il}"),
                    padr(&self.compcache[il], max_comp),
                ));
                let slots = r - 1;
                let np = self.partial_ck[il].len() / (coff * hd);
                let mut fpk = vec![0f32; (slots - np) * coff * hd];
                fpk.extend_from_slice(&self.partial_ck[il]);
                let mut fpg = vec![0f32; (slots - np) * coff * hd];
                fpg.extend_from_slice(&self.partial_cg[il]);
                owned.push((format!("pck.{il}"), fpk));
                owned.push((format!("pcg.{il}"), fpg));
                if r == 4 {
                    owned.push((
                        format!("ape_idx.{il}"),
                        (0..block).map(|i| ((pos + i) % r) as f32).collect(),
                    ));
                    let pk = if self.prev_kv[il].is_empty() {
                        vec![0f32; r * coff * hd]
                    } else {
                        self.prev_kv[il].clone()
                    };
                    let ps = if self.prev_score[il].is_empty() {
                        vec![-1e30f32; r * coff * hd]
                    } else {
                        self.prev_score[il].clone()
                    };
                    owned.push((format!("prev_kv.{il}"), pk));
                    owned.push((format!("prev_score.{il}"), ps));
                }
            }
        }
        owned
    }

    /// Commit the accepted block prefix (`accept_len+1` tokens) into the cache from
    /// the verify outputs — the block analogue of [`Self::step_io`]'s cache update
    /// (window append + compressor partial/compcache/prev state machine). `get`
    /// fetches a verify output by name. Exact for ≤1 overlap fire per block (block≤5).
    fn commit_block(&mut self, accept_len: usize, get: impl Fn(&str) -> Vec<f32>) {
        let hd = self.hd;
        let maxwin = self.max_win * hd;
        let pos = self.pos;
        let n = accept_len + 1;
        for il in self.layers.clone() {
            let kvn = get(&format!("kvnew.{il}"));
            self.win_ring[il].extend_from_slice(&kvn[..n * hd]);
            if self.win_ring[il].len() > maxwin {
                let d = self.win_ring[il].len() - maxwin;
                self.win_ring[il].drain(0..d);
            }
            let r = self.ratios[il];
            if r > 0 {
                let coff = self.coffs[il];
                let ck = get(&format!("ck.{il}"));
                let cg = get(&format!("cg.{il}"));
                let comp = get(&format!("comp.{il}"));
                let cw = coff * hd;
                for i in 0..n {
                    if (pos + i + 1).is_multiple_of(r) {
                        self.compcache[il].extend_from_slice(&comp[i * hd..(i + 1) * hd]);
                        if r == 4 {
                            let mut ckw = self.partial_ck[il].clone();
                            ckw.extend_from_slice(&ck[i * cw..(i + 1) * cw]);
                            let mut cgw = self.partial_cg[il].clone();
                            cgw.extend_from_slice(&cg[i * cw..(i + 1) * cw]);
                            self.prev_kv[il] = ckw;
                            self.prev_score[il] = cgw;
                        }
                        self.partial_ck[il].clear();
                        self.partial_cg[il].clear();
                    } else {
                        self.partial_ck[il].extend_from_slice(&ck[i * cw..(i + 1) * cw]);
                        self.partial_cg[il].extend_from_slice(&cg[i * cw..(i + 1) * cw]);
                    }
                }
            }
        }
        self.pos = pos + n;
    }

    /// Full-model decoder (all layers, first+last). See [`Self::new_stage`] for a
    /// single pipeline stage.
    pub fn new(
        spec: &DeepseekV4Spec,
        weights: &mut dyn WeightLoader,
        max_win: usize,
        max_comp: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        Self::new_stage(
            spec,
            weights,
            0..spec.n_layers,
            true,
            true,
            max_win,
            max_comp,
            device,
        )
    }

    /// One pipeline **stage** decoder over `layers` (holds only its layers' KV
    /// cache). `first` consumes the token id; `last` produces logits; otherwise it
    /// consumes/produces the hidden boundary. Drive N of these with
    /// [`deepseek_v4_generate_pipelined`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_stage(
        spec: &DeepseekV4Spec,
        weights: &mut dyn WeightLoader,
        layers: std::ops::Range<usize>,
        first: bool,
        last: bool,
        max_win: usize,
        max_comp: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        Self::new_stage_impl(
            spec, weights, layers, first, last, max_win, max_comp, device, false, false,
        )
    }

    /// [`Self::new_stage`] in **compile-once PAGED MoE** mode: the routed-MoE layers
    /// are split out of the graph (`moe_in`/`moe_out`), so the backbone compiles ONCE
    /// and [`Self::step_paged`] fills the MoE host-side per token. This is the fix for
    /// the O(L²) per-token recompile of [`deepseek_v4_generate_paged`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_stage_paged(
        spec: &DeepseekV4Spec,
        weights: &mut dyn WeightLoader,
        layers: std::ops::Range<usize>,
        first: bool,
        last: bool,
        max_win: usize,
        max_comp: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        Self::new_stage_impl(
            spec, weights, layers, first, last, max_win, max_comp, device, true, false,
        )
    }

    /// [`Self::new_stage_paged`] with the **attn/post split** for a single MoE layer:
    /// the main graph is attention-only and a tiny weight-free post graph applies
    /// `moe_out` — halving the per-token attention (no O(2L) two-pass recompute).
    /// Falls back to the plain paged stage for dense / non-single-layer stages.
    #[allow(clippy::too_many_arguments)]
    pub fn new_stage_paged_split(
        spec: &DeepseekV4Spec,
        weights: &mut dyn WeightLoader,
        layers: std::ops::Range<usize>,
        first: bool,
        last: bool,
        max_win: usize,
        max_comp: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        // Only single-layer MoE stages benefit; dense layers have no host-MoE boundary.
        let il = layers.start;
        let is_moe = layers.len() == 1 && il >= spec.first_k_dense_replace;
        Self::new_stage_impl(
            spec, weights, layers, first, last, max_win, max_comp, device, true, is_moe,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_stage_impl(
        spec: &DeepseekV4Spec,
        weights: &mut dyn WeightLoader,
        layers: std::ops::Range<usize>,
        first: bool,
        last: bool,
        max_win: usize,
        max_comp: usize,
        device: rlx_runtime::Device,
        paged_moe: bool,
        split_attn: bool,
    ) -> Result<Self> {
        let nl = spec.n_layers;
        let hd = spec.head_dim;
        let ratios: Vec<usize> = (0..nl)
            .map(|il| spec.compress_ratios.get(il).copied().unwrap_or(0))
            .collect();
        let coffs: Vec<usize> = ratios.iter().map(|&r| if r == 4 { 2 } else { 1 }).collect();
        // Read APE for overlap layers up front — the fixed graph feeds them the
        // `ape_row` input rather than loading APE (so this take() doesn't collide).
        let mut ape = vec![Vec::new(); nl];
        for il in layers.clone() {
            if ratios[il] == 4 {
                let (a, _) = weights.take(&format!("model.layers.{il}.attn.compressor.ape"))?;
                ape[il] = a;
            }
        }
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            device,
        );
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        // `RLX_DSV4_STREAM_COMPILE=1`: STREAM-COMPILE — structure-build the graph
        // (defer the big packed codes: only shapes + small BF16 scales held), compile
        // to allocate the arena, then stream each param into it on demand. The source
        // weights and the compiled arena never coexist at 2×, so a FULL single-node
        // model fits ~1× (its resident size, e.g. ~90 GB for 2-bit) instead of ~2×
        // (~180 GB → OOM on a 64 GB box). Costs a 2nd read of the codes (fast on NVMe);
        // default OFF (eager) since a stage that fits doesn't need it.
        let stream = std::env::var("RLX_DSV4_STREAM_COMPILE").ok().as_deref() == Some("1");
        let (compiled, out_names) = if stream {
            use rlx_distributed::{Param, ParamSource};
            let (g, params, out_names, manifest) = {
                let mut sloader = crate::distributed_bridge::StructureLoader::new(weights);
                let (g, p, o) = build_deepseek_v4_decode_fixed_stage_moe(
                    spec,
                    &mut sloader,
                    layers.clone(),
                    first,
                    last,
                    max_win,
                    max_comp,
                    &mut packed,
                    paged_moe,
                    split_attn,
                )?;
                (g, p, o, sloader.manifest)
            };
            let names: Vec<String> = g
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Param { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let mut compiled = rlx_runtime::Session::new(device).compile_with(g, &opts);
            let synth: HashMap<String, Vec<f32>> =
                params.into_iter().filter(|(_, v)| !v.is_empty()).collect();
            let synth_packed: HashMap<String, Vec<u8>> = packed
                .into_iter()
                .filter_map(|(k, (b, _, _))| (!b.is_empty()).then_some((k, b)))
                .collect();
            let mut source = crate::distributed_bridge::ManifestParamSource {
                loader: weights,
                manifest,
                synth,
                synth_packed,
            };
            for name in &names {
                match source.get(name) {
                    Some(Param::F32(v)) => compiled.set_param(name, &v),
                    Some(Param::Typed(b, dt)) => compiled.set_param_typed(name, &b, dt),
                    None => {}
                }
            }
            compiled.finalize_params();
            (compiled, out_names)
        } else {
            let (g, params, out_names) = build_deepseek_v4_decode_fixed_stage_moe(
                spec,
                weights,
                layers.clone(),
                first,
                last,
                max_win,
                max_comp,
                &mut packed,
                paged_moe,
                split_attn,
            )?;
            let mut compiled = rlx_runtime::Session::new(device).compile_with(g, &opts);
            // Eager: bind while CONSUMING each map so every source buffer drops the
            // moment it's copied into the arena — peak ≈ arena + one param.
            for (n, dd) in params {
                compiled.set_param(&n, &dd);
            }
            for (n, (bytes, _scheme, _shape)) in packed {
                compiled.set_param_typed(&n, &bytes, DType::U8);
            }
            compiled.finalize_params();
            (compiled, out_names)
        };
        let mut prev_kv = vec![Vec::new(); nl];
        let mut prev_score = vec![Vec::new(); nl];
        for il in layers.clone() {
            if ratios[il] == 4 {
                prev_kv[il] = vec![0f32; ratios[il] * coffs[il] * hd];
                prev_score[il] = vec![-1e30f32; ratios[il] * coffs[il] * hd];
            }
        }
        let moe_layers: Vec<usize> = if paged_moe {
            layers
                .clone()
                .filter(|&il| il >= spec.first_k_dense_replace)
                .collect()
        } else {
            Vec::new()
        };
        // Attn/post split: build + compile the tiny weight-free post graph (applies
        // `moe_out` → `hidden_out`/`logits`). Its head/`main_hidden` weights were NOT
        // taken by the attn-only main graph, so they're still available here.
        let (post, post_names) = if split_attn {
            let target = spec.dspark_target_layer_ids.contains(&layers.start);
            let mut ppacked: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (pg, pparams, pnames) =
                build_v4_post_stage(spec, weights, last, target, &mut ppacked)?;
            let mut pc = rlx_runtime::Session::new(device).compile_with(pg, &opts);
            for (n, dd) in pparams {
                pc.set_param(&n, &dd);
            }
            for (n, (bytes, _s, _sh)) in ppacked {
                pc.set_param_typed(&n, &bytes, DType::U8);
            }
            pc.finalize_params();
            (Some(pc), pnames)
        } else {
            (None, Vec::new())
        };
        Ok(Self {
            compiled,
            out_names,
            post,
            post_names,
            vocab: spec.vocab_size,
            dim: spec.dim,
            hd,
            half: (spec.rope_head_dim & !1) / 2,
            rd: spec.rope_head_dim & !1,
            rope_theta: spec.rope_theta,
            max_win,
            max_comp,
            ratios,
            coffs,
            ape,
            win_ring: vec![Vec::new(); nl],
            compcache: vec![Vec::new(); nl],
            partial_ck: vec![Vec::new(); nl],
            partial_cg: vec![Vec::new(); nl],
            prev_kv,
            prev_score,
            layers,
            first,
            last,
            target_layers: spec.dspark_target_layer_ids.clone(),
            main_hidden: None,
            pos: 0,
            moe_layers,
        })
    }

    /// The concatenated mean-over-streams hidden at the DSpark target layers from
    /// the last [`Self::step`] — the `main_hidden` the DSpark drafter consumes.
    /// `Some` only when this stage owns those layers and emitted them.
    pub fn main_hidden(&self) -> Option<&[f32]> {
        self.main_hidden.as_deref()
    }

    /// Full-model convenience: decode one token → `[vocab]` logits (panics if this
    /// isn't a first+last decoder).
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        self.step_io(Some(token), None)
            .0
            .expect("first+last decoder yields logits")
    }

    /// One pipeline-stage step. `first` stages take `token`; others take
    /// `hidden_in` `[1*hc*d]`. Returns `(logits?, hidden_out?)` — `logits` on the
    /// `last` stage, else `hidden_out` to relay to the next stage.
    pub fn step_io(
        &mut self,
        token: Option<u32>,
        hidden_in: Option<&[f32]>,
    ) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
        let pos = self.pos;
        let nl = self.ratios.len();
        let (hd, half, rd) = (self.hd, self.half.max(1), self.rd);
        let (mut cosd, mut sind) = (vec![0f32; half], vec![0f32; half]);
        for i in 0..half {
            let fr = self.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
            let (s, c) = (pos as f64 * fr).sin_cos();
            cosd[i] = c as f32;
            sind[i] = s as f32;
        }
        let sininv: Vec<f32> = sind.iter().map(|v| -v).collect();
        let pad = |v: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0f32; rows * cols];
            let n = v.len().min(rows * cols);
            out[..n].copy_from_slice(&v[..n]);
            out
        };
        let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
        if self.first {
            owned.push((
                "token_id".into(),
                vec![token.expect("first stage needs a token") as f32],
            ));
        } else {
            owned.push((
                "hidden_in".into(),
                hidden_in.expect("stage needs hidden_in").to_vec(),
            ));
        }
        owned.push(("cos".into(), cosd));
        owned.push(("sin".into(), sind));
        owned.push(("sininv".into(), sininv));
        let _ = nl;
        for il in self.layers.clone() {
            let win_len = self.win_ring[il].len() / hd;
            owned.push((
                format!("wcache.{il}"),
                pad(&self.win_ring[il], self.max_win, hd),
            ));
            let ratio = self.ratios[il];
            let coff = self.coffs[il];
            if ratio == 0 {
                let mut m = vec![-1e30f32; self.max_win + 1];
                for e in m.iter_mut().take(win_len) {
                    *e = 0.0;
                }
                m[self.max_win] = 0.0;
                owned.push((format!("mask.{il}"), m));
            } else {
                let ncomp = self.compcache[il].len() / hd;
                let firing = (pos + 1).is_multiple_of(ratio);
                owned.push((
                    format!("compcache.{il}"),
                    pad(&self.compcache[il], self.max_comp, hd),
                ));
                owned.push((
                    format!("partial_ck.{il}"),
                    pad(&self.partial_ck[il], ratio - 1, coff * hd),
                ));
                owned.push((
                    format!("partial_cg.{il}"),
                    pad(&self.partial_cg[il], ratio - 1, coff * hd),
                ));
                if ratio == 4 {
                    let row = pos % ratio;
                    owned.push((
                        format!("ape_row.{il}"),
                        self.ape[il][row * coff * hd..(row + 1) * coff * hd].to_vec(),
                    ));
                    owned.push((format!("prev_kv.{il}"), self.prev_kv[il].clone()));
                    owned.push((format!("prev_score.{il}"), self.prev_score[il].clone()));
                }
                let n_keys = self.max_win + 1 + self.max_comp + 1;
                let mut m = vec![-1e30f32; n_keys];
                for e in m.iter_mut().take(win_len) {
                    *e = 0.0;
                }
                m[self.max_win] = 0.0;
                for c in 0..ncomp {
                    m[self.max_win + 1 + c] = 0.0;
                }
                if firing {
                    m[self.max_win + 1 + self.max_comp] = 0.0;
                }
                owned.push((format!("mask.{il}"), m));
            }
        }
        let inputs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let out = self.compiled.run(&inputs);
        let out_names = &self.out_names;
        let get = |name: &str| -> Vec<f32> {
            let i = out_names
                .iter()
                .position(|x| x == name)
                .expect("decode output");
            out[i].clone()
        };
        let result = if self.last {
            (Some(get("logits")), None)
        } else {
            (None, Some(get("hidden_out")))
        };
        let maxwin_floats = self.max_win * hd;
        for il in self.layers.clone() {
            self.win_ring[il].extend_from_slice(&get(&format!("kvnew.{il}")));
            if self.win_ring[il].len() > maxwin_floats {
                let drop = self.win_ring[il].len() - maxwin_floats;
                self.win_ring[il].drain(0..drop);
            }
            let ratio = self.ratios[il];
            if ratio > 0 {
                if (pos + 1).is_multiple_of(ratio) {
                    self.compcache[il].extend_from_slice(&get(&format!("comp.{il}")));
                    if ratio == 4 {
                        let mut ckw = self.partial_ck[il].clone();
                        ckw.extend_from_slice(&get(&format!("ck.{il}")));
                        let mut cgw = self.partial_cg[il].clone();
                        cgw.extend_from_slice(&get(&format!("cg.{il}")));
                        self.prev_kv[il] = ckw;
                        self.prev_score[il] = cgw;
                    }
                    self.partial_ck[il].clear();
                    self.partial_cg[il].clear();
                } else {
                    self.partial_ck[il].extend_from_slice(&get(&format!("ck.{il}")));
                    self.partial_cg[il].extend_from_slice(&get(&format!("cg.{il}")));
                }
            }
        }
        // Collect the DSpark main_hidden (target layers, in order) if this stage
        // emitted them.
        if !self.target_layers.is_empty() {
            let mut mh = Vec::new();
            let mut have = false;
            for &tl in &self.target_layers {
                let nm = format!("main_hidden.{tl}");
                if let Some(i) = self.out_names.iter().position(|x| *x == nm) {
                    mh.extend_from_slice(&out[i]);
                    have = true;
                }
            }
            self.main_hidden = if have { Some(mh) } else { None };
        }
        self.pos += 1;
        result
    }

    /// Assemble the fixed-shape per-token inputs (token/hidden, RoPE, per-layer
    /// window/compressor cache + mask + APE) for the current position — the input
    /// half of [`Self::step_io`], reused by [`Self::step_paged`].
    fn decode_inputs(
        &self,
        token: Option<u32>,
        hidden_in: Option<&[f32]>,
    ) -> Vec<(String, Vec<f32>)> {
        let pos = self.pos;
        let (hd, half, rd) = (self.hd, self.half.max(1), self.rd);
        let (mut cosd, mut sind) = (vec![0f32; half], vec![0f32; half]);
        for i in 0..half {
            let fr = self.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
            let (s, c) = (pos as f64 * fr).sin_cos();
            cosd[i] = c as f32;
            sind[i] = s as f32;
        }
        let sininv: Vec<f32> = sind.iter().map(|v| -v).collect();
        let pad = |v: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0f32; rows * cols];
            let n = v.len().min(rows * cols);
            out[..n].copy_from_slice(&v[..n]);
            out
        };
        let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
        if self.first {
            owned.push((
                "token_id".into(),
                vec![token.expect("first stage needs a token") as f32],
            ));
        } else {
            owned.push((
                "hidden_in".into(),
                hidden_in.expect("stage needs hidden_in").to_vec(),
            ));
        }
        owned.push(("cos".into(), cosd));
        owned.push(("sin".into(), sind));
        owned.push(("sininv".into(), sininv));
        for il in self.layers.clone() {
            let win_len = self.win_ring[il].len() / hd;
            owned.push((
                format!("wcache.{il}"),
                pad(&self.win_ring[il], self.max_win, hd),
            ));
            let ratio = self.ratios[il];
            let coff = self.coffs[il];
            if ratio == 0 {
                let mut m = vec![-1e30f32; self.max_win + 1];
                for e in m.iter_mut().take(win_len) {
                    *e = 0.0;
                }
                m[self.max_win] = 0.0;
                owned.push((format!("mask.{il}"), m));
            } else {
                let ncomp = self.compcache[il].len() / hd;
                let firing = (pos + 1).is_multiple_of(ratio);
                owned.push((
                    format!("compcache.{il}"),
                    pad(&self.compcache[il], self.max_comp, hd),
                ));
                owned.push((
                    format!("partial_ck.{il}"),
                    pad(&self.partial_ck[il], ratio - 1, coff * hd),
                ));
                owned.push((
                    format!("partial_cg.{il}"),
                    pad(&self.partial_cg[il], ratio - 1, coff * hd),
                ));
                if ratio == 4 {
                    let row = pos % ratio;
                    owned.push((
                        format!("ape_row.{il}"),
                        self.ape[il][row * coff * hd..(row + 1) * coff * hd].to_vec(),
                    ));
                    owned.push((format!("prev_kv.{il}"), self.prev_kv[il].clone()));
                    owned.push((format!("prev_score.{il}"), self.prev_score[il].clone()));
                }
                let n_keys = self.max_win + 1 + self.max_comp + 1;
                let mut m = vec![-1e30f32; n_keys];
                for e in m.iter_mut().take(win_len) {
                    *e = 0.0;
                }
                m[self.max_win] = 0.0;
                for c in 0..ncomp {
                    m[self.max_win + 1 + c] = 0.0;
                }
                if firing {
                    m[self.max_win + 1 + self.max_comp] = 0.0;
                }
                owned.push((format!("mask.{il}"), m));
            }
        }
        owned
    }

    fn out_named(&self, out: &[Vec<f32>], name: &str) -> Vec<f32> {
        out[self
            .out_names
            .iter()
            .position(|x| x == name)
            .expect("decode output")]
        .clone()
    }

    /// Commit KV/compressor state + DSpark `main_hidden` from a run's outputs and
    /// advance the position — the commit half of [`Self::step_io`].
    fn commit_from_outputs(&mut self, out: &[Vec<f32>]) {
        let (hd, pos) = (self.hd, self.pos);
        let out_names = &self.out_names;
        let get = |name: &str| -> Vec<f32> {
            out[out_names
                .iter()
                .position(|x| x == name)
                .expect("decode output")]
            .clone()
        };
        let maxwin_floats = self.max_win * hd;
        for il in self.layers.clone() {
            self.win_ring[il].extend_from_slice(&get(&format!("kvnew.{il}")));
            if self.win_ring[il].len() > maxwin_floats {
                let drop = self.win_ring[il].len() - maxwin_floats;
                self.win_ring[il].drain(0..drop);
            }
            let ratio = self.ratios[il];
            if ratio > 0 {
                if (pos + 1) % ratio == 0 {
                    self.compcache[il].extend_from_slice(&get(&format!("comp.{il}")));
                    if ratio == 4 {
                        let mut ckw = self.partial_ck[il].clone();
                        ckw.extend_from_slice(&get(&format!("ck.{il}")));
                        let mut cgw = self.partial_cg[il].clone();
                        cgw.extend_from_slice(&get(&format!("cg.{il}")));
                        self.prev_kv[il] = ckw;
                        self.prev_score[il] = cgw;
                    }
                    self.partial_ck[il].clear();
                    self.partial_cg[il].clear();
                } else {
                    self.partial_ck[il].extend_from_slice(&get(&format!("ck.{il}")));
                    self.partial_cg[il].extend_from_slice(&get(&format!("cg.{il}")));
                }
            }
        }
        if !self.target_layers.is_empty() {
            let mut mh = Vec::new();
            let mut have = false;
            for &tl in &self.target_layers {
                let nm = format!("main_hidden.{tl}");
                if let Some(i) = self.out_names.iter().position(|x| *x == nm) {
                    mh.extend_from_slice(&out[i]);
                    have = true;
                }
            }
            self.main_hidden = if have { Some(mh) } else { None };
        }
        self.pos += 1;
    }

    /// **Stage-protocol PAGED step** (built via [`Self::new_stage_paged`]). `first`
    /// stages consume `token`, others `hidden_in`; `last` produces `[vocab]` logits,
    /// others the `[1·hc·d]` hidden boundary. `token` is ALWAYS passed to `moe_fn`
    /// (for hash layers). Runs the fixed graph once per MoE layer IN THIS STAGE
    /// (filling `moe_out.{il}` from `moe_fn(il, token, xf)`), then a final pass. Driving
    /// PER-LAYER stages (via [`deepseek_v4_generate_paged_layerwise`]) cuts the
    /// monolithic O(L²) attention recompute + O(L²) compile to O(L).
    pub fn step_io_paged(
        &mut self,
        token: u32,
        hidden_in: Option<&[f32]>,
        mut moe_fn: impl FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
    ) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
        let base = self.decode_inputs(if self.first { Some(token) } else { None }, hidden_in);
        // Attn/post split: run attention ONCE → moe_in + hc_post inputs; host MoE;
        // then the tiny post graph applies moe_out → hidden_out/logits. Halves the
        // per-token attention vs the two-pass below.
        if self.post.is_some() {
            let il = self.layers.start;
            let attn_owned: Vec<(&str, &[f32])> = base
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let attn_out = self.compiled.run(&attn_owned);
            let xf = self.out_named(&attn_out, &format!("moe_in.{il}"));
            let moe = moe_fn(il, token, &xf)?;
            let residual = self.out_named(&attn_out, &format!("hcp_residual.{il}"));
            let postf = self.out_named(&attn_out, &format!("hcp_postf.{il}"));
            let combf = self.out_named(&attn_out, &format!("hcp_combf.{il}"));
            let post_out = {
                let post = self.post.as_mut().expect("post graph");
                post.run(&[
                    ("hcp_residual", residual.as_slice()),
                    ("hcp_postf", postf.as_slice()),
                    ("hcp_combf", combf.as_slice()),
                    ("moe_out", moe.as_slice()),
                ])
            };
            let pget = |name: &str| -> Vec<f32> {
                self.post_names
                    .iter()
                    .position(|x| x == name)
                    .map(|i| post_out[i].clone())
                    .unwrap_or_default()
            };
            let result = if self.last {
                (Some(pget("logits")), None)
            } else {
                (None, Some(pget("hidden_out")))
            };
            // KV cache / compressor commit from the ATTN graph outputs (also bumps
            // `pos`; it sets main_hidden=None since the attn graph has none) — THEN
            // set `main_hidden` from the post graph so it isn't clobbered.
            self.commit_from_outputs(&attn_out);
            if let Some(i) = self.post_names.iter().position(|x| x == "main_hidden") {
                self.main_hidden = Some(post_out[i].clone());
            }
            return Ok(result);
        }
        let d = self.dim;
        let ml = self.moe_layers.clone();
        let mut moe_out: HashMap<usize, Vec<f32>> =
            ml.iter().map(|&il| (il, vec![0f32; d])).collect();
        let run = |compiled: &mut rlx_runtime::CompiledGraph,
                   base: &[(String, Vec<f32>)],
                   mo: &HashMap<usize, Vec<f32>>| {
            let mut owned: Vec<(&str, &[f32])> = base
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let extra: Vec<(String, Vec<f32>)> = ml
                .iter()
                .map(|&j| (format!("moe_out.{j}"), mo[&j].clone()))
                .collect();
            for (n, v) in &extra {
                owned.push((n.as_str(), v.as_slice()));
            }
            compiled.run(&owned)
        };
        // One pass per MoE layer (causal: moe_in.{il} is valid once layers < il filled).
        for &il in &ml {
            let out = run(&mut self.compiled, &base, &moe_out);
            let xf = self.out_named(&out, &format!("moe_in.{il}"));
            moe_out.insert(il, moe_fn(il, token, &xf)?);
        }
        let out = run(&mut self.compiled, &base, &moe_out);
        let result = if self.last {
            (Some(self.out_named(&out, "logits")), None)
        } else {
            (None, Some(self.out_named(&out, "hidden_out")))
        };
        self.commit_from_outputs(&out);
        Ok(result)
    }

    /// [`Self::step_io_paged`] for a whole-model `first && last` decoder → `[vocab]`
    /// logits (compile-once, monolithic — O(L²) attention recompute; prefer the
    /// per-layer [`deepseek_v4_generate_paged_layerwise`] for large models).
    pub fn step_paged(
        &mut self,
        token: u32,
        moe_fn: impl FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        assert!(
            self.last,
            "step_paged needs a `last` stage (produces logits)"
        );
        Ok(self
            .step_io_paged(token, None, moe_fn)?
            .0
            .expect("last stage yields logits"))
    }

    /// Greedy generate `n_new` tokens after decoding `prompt_ids` (prompt included
    /// in the cache). Compile-once: one graph, one weight load, N runs.
    pub fn generate(&mut self, prompt_ids: &[u32], n_new: usize) -> Vec<u32> {
        let mut logits = vec![0f32; self.vocab];
        for &tok in prompt_ids {
            logits = self.step(tok);
        }
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                )
                .0 as u32
        };
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax(&logits);
            out.push(next);
            logits = self.step(next);
        }
        out
    }
}

/// **Pipelined (distributed) greedy generation**: drive an ordered set of stage
/// decoders ([`V4Decoder::new_stage`], contiguous layer ranges, `first` first and
/// `last` last) through the token loop, relaying each stage's hidden boundary to
/// the next. Each stage holds only its layers' KV cache — the in-process reference
/// for a cross-node decode (each `V4Decoder` would live on its own node, the
/// relay over TCP). Returns the `n_new` greedy tokens.
pub fn deepseek_v4_generate_pipelined(
    stages: &mut [V4Decoder],
    prompt_ids: &[u32],
    n_new: usize,
) -> Vec<u32> {
    fn drive(stages: &mut [V4Decoder], token: u32) -> Vec<f32> {
        let mut hidden: Option<Vec<f32>> = None;
        let mut logits: Option<Vec<f32>> = None;
        for (i, stage) in stages.iter_mut().enumerate() {
            let (lo, ho) = if i == 0 {
                stage.step_io(Some(token), None)
            } else {
                stage.step_io(None, hidden.as_deref())
            };
            logits = lo;
            hidden = ho;
        }
        logits.expect("last stage must produce logits")
    }
    let vocab = stages.last().map(|s| s.vocab).unwrap_or(0);
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    let mut logits = vec![0f32; vocab];
    for &tok in prompt_ids {
        logits = drive(stages, tok);
    }
    let mut out = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let next = argmax(&logits);
        out.push(next);
        logits = drive(stages, next);
    }
    out
}

/// **Per-layer PAGED decode** — the O(L)-time compile-once paged decoder. Builds ONE
/// paged stage PER LAYER ([`V4Decoder::new_stage_paged`], `il..il+1`), so the backbone
/// compiles in `nl` small LINEAR steps (not one O(L²) monolith) and each token threads
/// the hidden through the stages, running **each layer's attention exactly once** per
/// pass (the monolithic [`deepseek_v4_generate_paged`] re-ran all layers' attention
/// every MoE pass → O(L²)). The MoE stays paged host-side via `moe_fn(il, token, xf)`
/// (wire to [`PagedGroupedMoe`]/[`paged_moe_forward`] + [`dense_swiglu_ffn`], hash via
/// [`hash_route_experts`]). `make_loader` yields a fresh loader per stage build.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_generate_paged_layerwise(
    spec: &DeepseekV4Spec,
    mut make_loader: impl FnMut() -> Box<dyn WeightLoader>,
    device: rlx_runtime::Device,
    max_win: usize,
    max_comp: usize,
    prompt_ids: &[u32],
    n_new: usize,
    mut moe_fn: impl FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
) -> Result<Vec<u32>> {
    let nl = spec.n_layers;
    let mut stages: Vec<V4Decoder> = Vec::with_capacity(nl);
    for il in 0..nl {
        stages.push(V4Decoder::new_stage_paged(
            spec,
            &mut *make_loader(),
            il..il + 1,
            il == 0,
            il == nl - 1,
            max_win,
            max_comp,
            device,
        )?);
    }
    // One token: thread the hidden boundary through the per-layer stages; each stage
    // fills its own MoE via `moe_fn` inside `step_io_paged`.
    fn run_token(
        stages: &mut [V4Decoder],
        token: u32,
        moe_fn: &mut dyn FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let mut hidden: Option<Vec<f32>> = None;
        let mut logits: Option<Vec<f32>> = None;
        for st in stages.iter_mut() {
            let (lg, hd) = st.step_io_paged(token, hidden.as_deref(), &mut *moe_fn)?;
            logits = lg;
            hidden = hd;
        }
        Ok(logits.expect("last stage yields logits"))
    }
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    let mut logits = vec![0f32; spec.vocab_size];
    for &tok in prompt_ids {
        logits = run_token(&mut stages, tok, &mut moe_fn)?;
    }
    let mut out = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let next = argmax(&logits);
        out.push(next);
        logits = run_token(&mut stages, next, &mut moe_fn)?;
    }
    Ok(out)
}

/// Tune a stage stream for a **high-bandwidth link (Thunderbolt bridge, ~10-40
/// Gbps)**: `TCP_NODELAY` (no Nagle — the per-token boundary is tiny and
/// latency-critical) + large `SO_SNDBUF`/`SO_RCVBUF` to fill the bandwidth-delay
/// product (default 4 MiB, override `RLX_V4_SOCKBUF`). The transport itself is
/// interface-agnostic — route over Thunderbolt by passing the TB-bridge IPs
/// (macOS `bridge0`, Linux `thunderbolt-net`) as the worker/peer addresses.
fn v4_configure_stream(s: &std::net::TcpStream) {
    s.set_nodelay(true).ok();
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let buf: libc::c_int = std::env::var("RLX_V4_SOCKBUF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4 * 1024 * 1024);
        let fd = s.as_raw_fd();
        for opt in [libc::SO_SNDBUF, libc::SO_RCVBUF] {
            unsafe {
                libc::setsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    opt,
                    &buf as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
            }
        }
    }
}

/// Length-prefixed f32 tensor over TCP: `u64` LE count + `count` LE f32s. A
/// zero-length message is the stop sentinel.
fn v4_send_tensor(s: &mut std::net::TcpStream, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    // Coalesce the length prefix + payload into ONE buffer → one write_all → one
    // TCP segment: fewer syscalls and no split-packet on a NODELAY fast link, so
    // the per-hop relay latency is a single round-trip, not two.
    let mut frame = Vec::with_capacity(8 + data.len() * 4);
    frame.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for x in data {
        frame.extend_from_slice(&x.to_le_bytes());
    }
    s.write_all(&frame)?;
    s.flush()
}
fn v4_recv_tensor(s: &mut std::net::TcpStream) -> std::io::Result<Vec<f32>> {
    use std::io::Read;
    let mut lb = [0u8; 8];
    s.read_exact(&mut lb)?;
    let n = u64::from_le_bytes(lb) as usize;
    let mut buf = vec![0u8; n * 4];
    s.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// **Serve one pipeline-stage decoder over TCP** (the cross-node worker). Accepts
/// a connection, then loops `{ recv input → step_io → send output }` until a
/// zero-length stop message, holding this stage's KV cache across the loop. A
/// `first` stage receives the token id as a `[1]` tensor; others receive the
/// hidden boundary. The `last` stage sends logits; others send the next hidden.
pub fn serve_v4_decode_stage(
    dec: &mut V4Decoder,
    listener: std::net::TcpListener,
) -> std::io::Result<()> {
    let (mut stream, _) = listener.accept()?;
    v4_configure_stream(&stream);
    loop {
        let input = v4_recv_tensor(&mut stream)?;
        if input.is_empty() {
            break; // stop
        }
        let (logits, hidden) = if dec.first {
            dec.step_io(Some(input[0] as u32), None)
        } else {
            dec.step_io(None, Some(&input))
        };
        let out = if dec.last {
            logits.expect("last stage logits")
        } else {
            hidden.expect("stage hidden")
        };
        v4_send_tensor(&mut stream, &out)?;
    }
    Ok(())
}

/// **Coordinator**: drive a pipelined greedy decode over TCP across the stage
/// workers at `worker_addrs` (stage order — `[0]` is `first`, the last emits
/// logits). Per token it sends the token to stage 0 and relays each stage's
/// boundary to the next, argmaxing the final logits; sends a stop to every worker
/// at the end. Returns the `n_new` greedy token ids. The distributed counterpart
/// of [`deepseek_v4_generate_pipelined`].
pub fn run_v4_decode_pipelined_tcp(
    worker_addrs: &[String],
    vocab: usize,
    prompt_ids: &[u32],
    n_new: usize,
) -> std::io::Result<Vec<u32>> {
    let mut streams: Vec<std::net::TcpStream> = Vec::new();
    for a in worker_addrs {
        let s = std::net::TcpStream::connect(a)?;
        v4_configure_stream(&s);
        streams.push(s);
    }
    fn drive(streams: &mut [std::net::TcpStream], token: u32) -> std::io::Result<Vec<f32>> {
        v4_send_tensor(&mut streams[0], &[token as f32])?;
        let mut b = v4_recv_tensor(&mut streams[0])?;
        for i in 1..streams.len() {
            v4_send_tensor(&mut streams[i], &b)?;
            b = v4_recv_tensor(&mut streams[i])?;
        }
        Ok(b)
    }
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    let mut logits = vec![0f32; vocab];
    for &tok in prompt_ids {
        logits = drive(&mut streams, tok)?;
    }
    let mut out = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let next = argmax(&logits);
        out.push(next);
        logits = drive(&mut streams, next)?;
    }
    for s in &mut streams {
        v4_send_tensor(s, &[])?; // stop
    }
    Ok(out)
}

/// **Batch (seq=block) VERIFY graph** — the DSpark speculative-decode accelerator.
/// Processes all `block` draft tokens in ONE main-model pass on top of the KV
/// cache (one ~5.4 GB weight-read instead of `block` sequential ones — the whole
/// point), returning `logits [block, vocab]` and each layer's `kvnew.{il}
/// [block, hd]` so the host can commit the accepted prefix. Fixed-shape (compile
/// once): `cos`/`sin`/`sininv` are `[block, half]` inputs; each layer takes a
/// padded window cache `[max_win, hd]` + compressed cache `[max_comp, hd]` and a
/// validity mask `[block, n_keys]` (window valid slots, **causal within the block**,
/// visible compressed positions). It attends to the **existing** compressed cache
/// but does NOT form new compressed KV inside the block — so it is EXACT whenever
/// no compressor fires within the block (`block < ratio`; on GA the ratio-128
/// layers never fire for `block ≤ 5`). Ratio-4 in-block compression is the
/// remaining exactness refinement (the host can re-check accepted tokens with the
/// exact seq=1 decode). Mirrors [`build_deepseek_v4_decode_fixed_stage`] at
/// `rows = block`, minus the compressor pooling.
pub fn build_deepseek_v4_verify_block(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    block: usize,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v4_verify_block_moe(spec, weights, block, max_win, max_comp, packed, false)
}

/// [`build_deepseek_v4_verify_block`] with the **paged-MoE split** — the routed
/// experts leave the graph as a `moe_in.{il} [block,d]` output + `moe_out.{il}
/// [block,d]` input, so the 156 GB experts are host-paged (like the decode stage)
/// and the verify block runs on a resident backbone. This is the enabling piece
/// for speculative decode on the FULL paged model (`deepseek_v4_generate_speculative_paged`).
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v4_verify_block_moe(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    block: usize,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    paged_moe: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    build_deepseek_v4_verify_block_stage(
        spec,
        weights,
        block,
        max_win,
        max_comp,
        packed,
        0..spec.n_layers,
        paged_moe,
    )
}

/// [`build_deepseek_v4_verify_block_moe`] restricted to a **layer range** — the
/// O(L) layerwise verify that replaces the monolithic O(L²) causal fill. When
/// `layers.start == 0` the block tokens are embedded; otherwise `h` comes from a
/// `hidden_in [block,hc,d]` input. When `layers.end == n_layers` the graph applies
/// `hc_head`/`norm`/`lm_head` (output `logits [block,vocab]`); otherwise it emits
/// `hidden_out [block,hc,d]`. A layerwise driver runs one single-layer stage per
/// layer, threading `hidden_out → hidden_in` and reading each layer's expert ONCE
/// for all `block` rows, so a K-token block verify costs O(L) backbone passes
/// (2 per paged MoE layer to fill `moe_out`) instead of the monolithic O(L²).
/// Composing all single-layer stages is bit-identical to the whole-model build
/// (`deepseek_v4_layerwise_verify_matches_monolithic` regression test).
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek_v4_verify_block_stage(
    spec: &DeepseekV4Spec,
    weights: &mut dyn WeightLoader,
    block: usize,
    max_win: usize,
    max_comp: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    layers: std::ops::Range<usize>,
    paged_moe: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<String>)> {
    let mut g = Graph::new("deepseek_v4_verify_block");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let (d, hc, nh, hd) = (spec.dim, spec.hc_mult, spec.n_heads, spec.head_dim);
    let rd = spec.rope_head_dim & !1;
    let ql = spec.q_lora_rank;
    let eps = spec.rms_norm_eps;
    let scale = (hd as f32).powf(-0.5);
    let half = (rd / 2).max(1);
    let b = block;
    let zb_d = synth_zero(&mut g, &mut params, "v4v.zb.d", d);
    let zb_ql = synth_zero(&mut g, &mut params, "v4v.zb.ql", ql);
    let zb_hd = synth_zero(&mut g, &mut params, "v4v.zb.hd", hd);
    let q_ones = synth_const(&mut g, &mut params, "v4v.qones", vec![1f32; hd], &[hd]);
    let cos_p = g.input("cos", Shape::new(&[b, half], f));
    let sin_p = g.input("sin", Shape::new(&[b, half], f));
    let sininv_p = g.input("sininv", Shape::new(&[b, half], f));

    // Token ids embed the block (first stage only) and drive hash-MoE layers.
    let need_token = layers.start == 0 || layers.clone().any(|il| il < spec.n_hash_layers);
    let token_opt = if need_token {
        Some(g.input("token_id", Shape::new(&[b], DType::I32)))
    } else {
        None
    };
    let input_ids_flat = token_opt.map(|t| g.reshape_(t, vec![b as i64]));
    let mut h = if layers.start == 0 {
        let token = token_opt.expect("token_id for embed");
        let token2 = g.reshape_(token, vec![1, b as i64]);
        let (embed_w, _, _) =
            load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
        let h0 = g.gather_(embed_w, token2, 0); // [1, b, d]
        let h0 = g.reshape_(h0, vec![b as i64, 1, d as i64]);
        let ones_hc = synth_const(
            &mut g,
            &mut params,
            "v4v.hc.ones",
            vec![1f32; hc],
            &[1, hc, 1],
        );
        g.mul(h0, ones_hc) // [b, hc, d]
    } else {
        g.input("hidden_in", Shape::new(&[b, hc, d], f))
    };
    let mut extra_outs: Vec<(String, NodeId)> = Vec::new();

    for il in layers.clone() {
        let ratio = spec.compress_ratios.get(il).copied().unwrap_or(0);
        let lp = format!("model.layers.{il}");
        let residual = h;
        let fn_a =
            load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.attn_hc.fn"))?;
        let sc_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.scale"),
            false,
        )?;
        let bs_a = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_hc.base"),
            false,
        )?;
        let (xa, post_a, comb_a) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_a,
            sc_a,
            bs_a,
            b,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.va"),
        );
        let an = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn_norm.weight"),
            0.0,
        )?;
        let xa = g.rms_norm(xa, an, zb_d, eps);
        let wqa = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_a.weight"),
        )?;
        let qr = emit_proj(&mut g, xa, &wqa, Shape::new(&[b, ql], f));
        let qn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.q_norm.weight"),
            0.0,
        )?;
        let qr = g.rms_norm(qr, qn, zb_ql, eps);
        let wqb = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wq_b.weight"),
        )?;
        let q = emit_proj(&mut g, qr, &wqb, Shape::new(&[b, nh * hd], f));
        let q = per_head_rms(&mut g, q, q_ones, zb_hd, 1, b, nh, hd, eps);
        let q = rope_tail(&mut g, q, cos_p, sin_p, b, nh, hd, rd);
        let wkv = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.attn.wkv.weight"),
        )?;
        let kv = emit_proj(&mut g, xa, &wkv, Shape::new(&[b, hd], f));
        let kvn = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.kv_norm.weight"),
            0.0,
        )?;
        let kv = g.rms_norm(kv, kvn, zb_hd, eps);
        let kv_new = rope_tail(&mut g, kv, cos_p, sin_p, b, 1, hd, rd); // [b, hd]
        extra_outs.push((
            format!("kvnew.{il}"),
            g.reshape_(kv_new, vec![b as i64, hd as i64]),
        ));

        // Compressor projections for each block token (ck = wkv(xa), cg = wgate(xa)
        // [+ per-token APE for the overlapping ratio-4 compressor]) — the raw state
        // the host commits into partial_ck/cg, and what the in-block pool consumes.
        // Matches the seq=1 decode exactly; the APE row (pos+i)%ratio is selected by
        // a host-provided `ape_idx.{il}` gather so the graph stays compile-once.
        let comp_proj: Option<(NodeId, NodeId, usize, bool)> = if ratio > 0 {
            let overlap = ratio == 4;
            let coff = if overlap { 2 } else { 1 };
            let cw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wkv.weight"),
                true,
            )?;
            let ck_t = g.mm(xa, cw); // [b, coff*hd]
            let cgw = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.attn.compressor.wgate.weight"),
                true,
            )?;
            let cg_raw = g.mm(xa, cgw);
            let cg_t = if overlap {
                let ape = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.ape"),
                    false,
                )?;
                let ape_idx = g.input(format!("ape_idx.{il}"), Shape::new(&[b], DType::I32));
                let idx2 = g.reshape_(ape_idx, vec![1, b as i64]);
                let sel = g.gather_(ape, idx2, 0); // [1, b, coff*hd]
                let sel = g.reshape_(sel, vec![b as i64, (coff * hd) as i64]);
                g.add(cg_raw, sel)
            } else {
                cg_raw
            };
            extra_outs.push((
                format!("ck.{il}"),
                g.reshape_(ck_t, vec![b as i64, (coff * hd) as i64]),
            ));
            extra_outs.push((
                format!("cg.{il}"),
                g.reshape_(cg_t, vec![b as i64, (coff * hd) as i64]),
            ));
            Some((ck_t, cg_t, coff, overlap))
        } else {
            None
        };

        let wcache = g.input(format!("wcache.{il}"), Shape::new(&[max_win, hd], f));
        let (kv_all, n_keys) = match comp_proj {
            None => (g.concat_(vec![wcache, kv_new], 0), max_win + b),
            // NON-OVERLAP compress: existing compcache + IN-BLOCK candidate pool. For
            // each block position j, pool the ratio-token window ending at j from
            // `[partial_pad(ratio-1) ++ block_ck]` (host front-pads the partial so
            // window[j] = full[j..j+ratio] is exactly the ratio tokens ending at
            // pos+j). The visibility mask exposes cand_comp[j] only when j is a real
            // fire and j ≤ query. `comp.{il}` is emitted for the host commit.
            Some((ck_t, cg_t, coff, false)) => {
                let compcache = g.input(format!("compcache.{il}"), Shape::new(&[max_comp, hd], f));
                let pck = g.input(format!("pck.{il}"), Shape::new(&[ratio - 1, coff * hd], f));
                let pcg = g.input(format!("pcg.{il}"), Shape::new(&[ratio - 1, coff * hd], f));
                let full_ck = g.concat_(vec![pck, ck_t], 0); // [ratio-1+b, coff*hd]
                let full_cg = g.concat_(vec![pcg, cg_t], 0);
                let ape = load_p(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.ape"),
                    false,
                )?;
                let cnorm = load_norm(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.norm.weight"),
                    0.0,
                )?;
                let mut cands: Vec<NodeId> = Vec::with_capacity(b);
                for j in 0..b {
                    let wck = g.narrow_(full_ck, 0, j, ratio);
                    let wcg = g.narrow_(full_cg, 0, j, ratio);
                    let c = build_kv_compressor_pool(
                        &mut g,
                        &mut params,
                        wck,
                        wcg,
                        ape,
                        cnorm,
                        1,
                        ratio,
                        ratio,
                        hd,
                        eps,
                        &format!("{lp}.vp{j}"),
                    );
                    cands.push(g.reshape_(c, vec![1, hd as i64]));
                }
                let cand_comp = g.concat_(cands, 0); // [b, hd]
                extra_outs.push((format!("comp.{il}"), cand_comp));
                (
                    g.concat_(vec![wcache, kv_new, compcache, cand_comp], 0),
                    max_win + b + max_comp + b,
                )
            }
            // OVERLAP (ratio-4): same in-block candidate pool + visibility mask as
            // non-overlap, but each window pools with the PREVIOUS window via
            // build_overlap_pool_single (host prev_kv/prev_score — the ≤1 in-block
            // fire for block≤5 reuses the block-start prev).
            Some((ck_t, cg_t, coff, true)) => {
                let compcache = g.input(format!("compcache.{il}"), Shape::new(&[max_comp, hd], f));
                let pck = g.input(format!("pck.{il}"), Shape::new(&[ratio - 1, coff * hd], f));
                let pcg = g.input(format!("pcg.{il}"), Shape::new(&[ratio - 1, coff * hd], f));
                let prev_kv = g.input(format!("prev_kv.{il}"), Shape::new(&[ratio, coff * hd], f));
                let prev_score = g.input(
                    format!("prev_score.{il}"),
                    Shape::new(&[ratio, coff * hd], f),
                );
                let full_ck = g.concat_(vec![pck, ck_t], 0);
                let full_cg = g.concat_(vec![pcg, cg_t], 0);
                let cnorm = load_norm(
                    &mut g,
                    &mut params,
                    weights,
                    &format!("{lp}.attn.compressor.norm.weight"),
                    0.0,
                )?;
                let mut cands: Vec<NodeId> = Vec::with_capacity(b);
                for j in 0..b {
                    let wck = g.narrow_(full_ck, 0, j, ratio);
                    let wcg = g.narrow_(full_cg, 0, j, ratio);
                    let c = build_overlap_pool_single(
                        &mut g,
                        &mut params,
                        prev_kv,
                        prev_score,
                        wck,
                        wcg,
                        cnorm,
                        hd,
                        eps,
                        &format!("{lp}.vov{j}"),
                    );
                    cands.push(c); // [1, hd]
                }
                let cand_comp = g.concat_(cands, 0); // [b, hd]
                extra_outs.push((format!("comp.{il}"), cand_comp));
                (
                    g.concat_(vec![wcache, kv_new, compcache, cand_comp], 0),
                    max_win + b + max_comp + b,
                )
            }
        };
        let mask = g.input(format!("mask.{il}"), Shape::new(&[b, n_keys], f));
        let sink = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.attn_sink"),
            false,
        )?;
        let q3 = g.reshape_(q, vec![b as i64, nh as i64, hd as i64]);
        let o = build_v4_sink_attention(
            &mut g,
            &mut params,
            q3,
            kv_all,
            mask,
            sink,
            scale,
            b,
            nh,
            hd,
            n_keys,
            &format!("{lp}.vsa"),
        );
        let o_flat = g.reshape_(o, vec![b as i64, (nh * hd) as i64]);
        let o_inv = rope_tail(&mut g, o_flat, cos_p, sininv_p, b, nh, hd, rd);
        let dpg = nh * hd / spec.n_groups;
        let woa = load_v4_wo_a(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_a.weight"),
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
        )?;
        let wob = load_transposed_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.wo_b.weight"),
        )?;
        let attn_out = build_v4_o_lora(
            &mut g,
            o_inv,
            woa,
            wob,
            b,
            spec.n_groups,
            spec.o_lora_rank,
            dpg,
            d,
        );
        h = build_hc_post(&mut g, attn_out, residual, post_a, comb_a, b, hc, d);

        let residual = h;
        let fn_f = load_transposed_param(&mut g, &mut params, weights, &format!("{lp}.ffn_hc.fn"))?;
        let sc_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.scale"),
            false,
        )?;
        let bs_f = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_hc.base"),
            false,
        )?;
        let (xf, post_f, comb_f) = build_hc_pre(
            &mut g,
            &mut params,
            h,
            fn_f,
            sc_f,
            bs_f,
            b,
            hc,
            d,
            spec.hc_eps,
            spec.hc_sinkhorn_iters,
            &format!("{lp}.vf"),
        );
        let fnorm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ffn_norm.weight"),
            0.0,
        )?;
        let xf = g.rms_norm(xf, fnorm, zb_d, eps);
        let ffn_out = if il < spec.first_k_dense_replace {
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.up_proj.weight"),
            )?;
            let dn = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.ffn.down_proj.weight"),
            )?;
            let gate = emit_proj(&mut g, xf, &gp, Shape::new(&[b, spec.intermediate_size], f));
            let upv = emit_proj(&mut g, xf, &up, Shape::new(&[b, spec.intermediate_size], f));
            let sg = g.silu(gate);
            let glu = g.mul(sg, upv);
            emit_proj(&mut g, glu, &dn, Shape::new(&[b, d], f))
        } else if paged_moe {
            // Paged: emit the block's post-ffn_norm hiddens; host fills the MoE for
            // all `block` tokens (one expert read amortized across the block).
            extra_outs.push((format!("moe_in.{il}"), xf));
            g.input(format!("moe_out.{il}"), Shape::new(&[b, d], f))
        } else {
            let ds = v4_moe_spec(spec);
            let x3 = g.reshape_(xf, vec![1, b as i64, d as i64]);
            let hash_ids = if il < spec.n_hash_layers {
                input_ids_flat
            } else {
                None
            };
            let moe = build_deepseek_moe_ffn(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                x3,
                1,
                b,
                &ds,
                hash_ids,
            )?;
            g.reshape_(moe, vec![b as i64, d as i64])
        };
        h = build_hc_post(&mut g, ffn_out, residual, post_f, comb_f, b, hc, d);
        // DSpark: emit the mean-over-HC hidden at each target layer (per block row) so
        // the driver can accumulate `main_hidden` history for the drafter — mirrors the
        // decode stage's `main_hidden.{il}` emission.
        if spec.dspark_target_layer_ids.contains(&il) {
            let mh = g.mean(h, vec![1], false); // [b, d] over the hc streams
            extra_outs.push((
                format!("main_hidden.{il}"),
                g.reshape_(mh, vec![b as i64, d as i64]),
            ));
        }
    }

    // Last stage applies the head (→ `logits`); interior stages thread `hidden_out`.
    let (primary, primary_name) = if layers.end == spec.n_layers {
        let hfn = load_transposed_param(&mut g, &mut params, weights, "model.hc_head.fn")?;
        let hsc = load_p(&mut g, &mut params, weights, "model.hc_head.scale", false)?;
        let hbs = load_p(&mut g, &mut params, weights, "model.hc_head.base", false)?;
        let x = build_hc_head(
            &mut g,
            &mut params,
            h,
            hfn,
            hsc,
            hbs,
            b,
            hc,
            d,
            spec.hc_eps,
            "v4v.head",
        );
        let fnorm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
        let x = g.rms_norm(x, fnorm, zb_d, eps);
        let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
        let logits = emit_proj(&mut g, x, &head_p, Shape::new(&[b, spec.vocab_size], f));
        (
            g.reshape_(logits, vec![b as i64, spec.vocab_size as i64]),
            "logits".to_string(),
        )
    } else {
        (
            g.reshape_(h, vec![b as i64, hc as i64, d as i64]),
            "hidden_out".to_string(),
        )
    };
    let mut outs = vec![primary];
    let mut names = vec![primary_name];
    for (name, node) in extra_outs {
        names.push(name);
        outs.push(node);
    }
    g.set_outputs(outs);
    Ok((g, params, names))
}

/// **End-to-end DSpark speculative decode** (greedy, LOSSLESS ⇒ byte-identical
/// output to [`V4Decoder::generate`], but fewer full-model passes). Each round:
/// take the definite next token `t = argmax(L)`, ask `draft_fn(t)` for up to
/// `block-1` guesses, verify the whole `[t, d₁..dₖ]` block in ONE main pass via
/// `verify` ([`build_deepseek_v4_verify_block`]), accept the longest greedy-matching
/// prefix, and commit the accepted tokens' KV **straight from the verify pass** — so
/// `a+1` tokens cost one draft + one verify weight-read instead of `a+1` of them.
///
/// A bad draft just falls back to one token/round (never wrong — that's the
/// losslessness). Correct for ANY topology: on a config with compress layers it
/// decodes exactly (seq=1) since their per-token compressor state can't be
/// reproduced from the sliding commit — the batch acceleration applies to sliding
/// topologies, and unlocking it for GA needs in-block compressor pooling in the
/// verify graph (a distinct piece). See the `has_compress` guard.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_generate_speculative(
    dec: &mut V4Decoder,
    verify: &mut rlx_runtime::CompiledGraph,
    vnames: &[String],
    spec: &DeepseekV4Spec,
    block: usize,
    max_comp: usize,
    mut draft_fn: impl FnMut(u32) -> Vec<u32>,
    prompt: &[u32],
    n_new: usize,
) -> Vec<u32> {
    let vocab = spec.vocab_size;
    let argmax = |l: &[f32]| -> u32 {
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in l.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        bi as u32
    };
    let mut last = vec![0f32; vocab];
    for &tk in prompt {
        last = dec.step(tk);
    }
    let mut out: Vec<u32> = Vec::new();
    while out.len() < n_new {
        let t = argmax(&last);
        let drafts = draft_fn(t);
        let k = drafts.len().min(block - 1);
        // Fixed-size block [t, d₁..dₖ, pad…]; causal ⇒ the padded tail can't affect
        // the first k+1 rows.
        let mut blk = vec![t];
        blk.extend_from_slice(&drafts[..k]);
        let bl = blk.len(); // k+1
        blk.resize(block, 0);
        // Assemble ALL verify inputs from the cache (window + compressor state) and
        // run the whole block in one pass.
        let owned = dec.verify_block_inputs(block, &blk, max_comp);
        let inputs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let vout = verify.run(&inputs);
        let vpos = |name: &str| {
            vnames
                .iter()
                .position(|x| x == name)
                .expect("verify output")
        };
        let logits = &vout[vpos("logits")]; // [block, vocab]
        let row_argmax = |r: usize| argmax(&logits[r * vocab..(r + 1) * vocab]);
        // Accept the longest run where draft[a+1] == argmax(previous row).
        let mut a = 0usize;
        while a < bl - 1 && blk[a + 1] == row_argmax(a) {
            a += 1;
        }
        for j in 0..=a {
            out.push(blk[j]);
        }
        let last_next = logits[a * vocab..(a + 1) * vocab].to_vec();
        // Commit the accepted prefix's KV + compressor state straight from the pass.
        let get = |nm: &str| vout[vpos(nm)].clone();
        dec.commit_block(a, get);
        last = last_next;
    }
    out.truncate(n_new);
    out
}

/// **Paged** DSpark speculative decode — [`deepseek_v4_generate_speculative`] for the
/// FULL 156 GB model: both the prompt decoder (`dec`, a first+last paged
/// [`V4Decoder`]) and the block verifier (`verify`, built by
/// [`build_deepseek_v4_verify_block_moe`] with `paged_moe=true`) leave the routed
/// experts as a host `moe_in`/`moe_out` boundary. `moe_fn(il, tokens, moe_in[n,d]) →
/// moe_out[n,d]` fills them (one expert read amortized across the block's `n` rows —
/// the whole point on the I/O-bound paged model). Verify fills `moe_out.{il}`
/// causally (a pass per MoE layer, reading `moe_in.{il}` then the host MoE), then
/// reads block logits. **Lossless** (greedy accept). NB the monolithic verify's
/// per-layer fill is O(L²) backbone — fine for the expert-I/O-bound regime (experts
/// read once/block); a layerwise verify would make it O(L) (follow-up).
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_generate_speculative_paged(
    dec: &mut V4Decoder,
    verify: &mut rlx_runtime::CompiledGraph,
    vnames: &[String],
    spec: &DeepseekV4Spec,
    block: usize,
    max_comp: usize,
    mut moe_fn: impl FnMut(usize, &[u32], &[f32]) -> Result<Vec<f32>>,
    mut draft_fn: impl FnMut(u32) -> Vec<u32>,
    prompt: &[u32],
    n_new: usize,
) -> Result<Vec<u32>> {
    let vocab = spec.vocab_size;
    let d = spec.dim;
    let argmax = |l: &[f32]| -> u32 {
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in l.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        bi as u32
    };
    let ml: Vec<usize> = (spec.first_k_dense_replace..spec.n_layers).collect();
    // Prompt prefill via the paged decoder (wrap the batch moe_fn for its 1-token step).
    let mut last = vec![0f32; vocab];
    for &tk in prompt {
        let (l, _) = dec.step_io_paged(tk, None, |il, t, xf| moe_fn(il, &[t], xf))?;
        last = l.expect("first+last paged decoder yields logits");
    }
    let mut out: Vec<u32> = Vec::new();
    while out.len() < n_new {
        let t = argmax(&last);
        let drafts = draft_fn(t);
        let k = drafts.len().min(block - 1);
        let mut blk = vec![t];
        blk.extend_from_slice(&drafts[..k]);
        let bl = blk.len(); // k+1 real rows
        blk.resize(block, 0);
        let base = dec.verify_block_inputs(block, &blk, max_comp);
        // Fill moe_out.{il} causally (one verify pass per MoE layer).
        let mut moe_out: HashMap<usize, Vec<f32>> =
            ml.iter().map(|&il| (il, vec![0f32; block * d])).collect();
        let run = |v: &mut rlx_runtime::CompiledGraph,
                   base: &[(String, Vec<f32>)],
                   mo: &HashMap<usize, Vec<f32>>|
         -> Vec<Vec<f32>> {
            let extra: Vec<(String, Vec<f32>)> = ml
                .iter()
                .map(|&j| (format!("moe_out.{j}"), mo[&j].clone()))
                .collect();
            let mut owned: Vec<(&str, &[f32])> = base
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            for (n, v) in &extra {
                owned.push((n.as_str(), v.as_slice()));
            }
            v.run(&owned)
        };
        let vpos = |name: &str| {
            vnames
                .iter()
                .position(|x| x == name)
                .expect("verify output")
        };
        for &il in &ml {
            let vout = run(verify, &base, &moe_out);
            let xf = vout[vpos(&format!("moe_in.{il}"))].clone(); // [block, d]
            moe_out.insert(il, moe_fn(il, &blk, &xf)?);
        }
        let vout = run(verify, &base, &moe_out);
        let logits = &vout[vpos("logits")]; // [block, vocab]
        let row_argmax = |r: usize| argmax(&logits[r * vocab..(r + 1) * vocab]);
        let mut a = 0usize;
        while a < bl - 1 && blk[a + 1] == row_argmax(a) {
            a += 1;
        }
        for j in 0..=a {
            out.push(blk[j]);
        }
        last = logits[a * vocab..(a + 1) * vocab].to_vec();
        let get = |nm: &str| vout[vpos(nm)].clone();
        dec.commit_block(a, get);
    }
    out.truncate(n_new);
    Ok(out)
}

/// **Layerwise** paged DSpark speculative decode — the O(L) replacement for
/// [`deepseek_v4_generate_speculative_paged`]'s monolithic O(L²) verify. The
/// monolithic verify runs the whole-model graph `L+1` times (each pass recomputing
/// EVERY layer's attention just to advance the causal `moe_out` fill by one layer) ⇒
/// O(L²) backbone. This runs ONE single-layer verify stage per layer
/// ([`build_deepseek_v4_verify_block_stage`], `layers = il..il+1`), threading
/// `hidden_out → hidden_in`: a paged MoE layer costs 2 stage runs (fill `moe_out.{il}`
/// then finish `hidden_out`), a dense layer costs 1 ⇒ O(L) backbone per block, while
/// STILL reading each layer's expert exactly once for all `block` rows. `stages[il]`/
/// `stage_names[il]` are the compiled single-layer verify graphs (built with the SAME
/// `paged_moe` as deployment). Lossless (byte-identical to greedy) — validated against
/// the monolithic verify by `deepseek_v4_layerwise_verify_matches_monolithic` and
/// end-to-end against greedy by `deepseek_v4_speculative_*`.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_generate_speculative_paged_layerwise(
    dec: &mut V4Decoder,
    stages: &mut [rlx_runtime::CompiledGraph],
    stage_names: &[Vec<String>],
    spec: &DeepseekV4Spec,
    block: usize,
    max_comp: usize,
    mut moe_fn: impl FnMut(usize, &[u32], &[f32]) -> Result<Vec<f32>>,
    mut draft_fn: impl FnMut(u32, &[f32]) -> Vec<u32>,
    prompt: &[u32],
    n_new: usize,
) -> Result<Vec<u32>> {
    assert_eq!(stages.len(), spec.n_layers, "one verify stage per layer");
    assert_eq!(stage_names.len(), spec.n_layers, "one name set per stage");
    let vocab = spec.vocab_size;
    let d = spec.dim;
    let argmax = |l: &[f32]| -> u32 {
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in l.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        bi as u32
    };
    // Prompt prefill via the paged decoder (its own 1-token MoE fill). We accumulate
    // `main_hist` — the per-token target-layer hiddens (`[cache_len, dim*n_targets]`)
    // the DSpark drafter consumes via `draft_fn(input_id, &main_hist)`.
    let mut last = vec![0f32; vocab];
    let mut main_hist: Vec<f32> = Vec::new();
    for &tk in prompt {
        let (l, _) = dec.step_io_paged(tk, None, |il, t, xf| moe_fn(il, &[t], xf))?;
        last = l.expect("first+last paged decoder yields logits");
        if let Some(mh) = dec.main_hidden() {
            main_hist.extend_from_slice(mh);
        }
    }
    let mut out: Vec<u32> = Vec::new();
    while out.len() < n_new {
        let t = argmax(&last);
        let drafts = draft_fn(t, &main_hist);
        let k = drafts.len().min(block - 1);
        let mut blk = vec![t];
        blk.extend_from_slice(&drafts[..k]);
        let bl = blk.len(); // k+1 real rows
        blk.resize(block, 0);
        let base = dec.verify_block_inputs(block, &blk, max_comp);
        // Thread the block hidden through one single-layer stage per layer (O(L)).
        let mut collected: HashMap<String, Vec<f32>> = HashMap::new();
        let mut hidden: Vec<f32> = Vec::new();
        let mut logits: Vec<f32> = Vec::new();
        for il in 0..spec.n_layers {
            // Feed this stage exactly its declared inputs: shared rope + token (embed on
            // stage 0 / hash layers) + layer-`il` cache state + hidden_in (interior).
            let suffix = format!(".{il}");
            let mut ins: Vec<(String, Vec<f32>)> = Vec::new();
            for (n, v) in &base {
                if n == "cos" || n == "sin" || n == "sininv" {
                    ins.push((n.clone(), v.clone()));
                } else if n == "token_id" {
                    if il == 0 || il < spec.n_hash_layers {
                        ins.push((n.clone(), v.clone()));
                    }
                } else if n.ends_with(&suffix) {
                    ins.push((n.clone(), v.clone()));
                }
            }
            if il > 0 {
                ins.push(("hidden_in".into(), hidden.clone()));
            }
            let names = &stage_names[il];
            let pos = |nm: &str| names.iter().position(|x| x == nm);
            let is_moe = il >= spec.first_k_dense_replace;
            let vout = if is_moe {
                // Pass 1 (moe_out=0) → moe_in.{il}; host MoE; Pass 2 → hidden/logits.
                let mut i1 = ins.clone();
                i1.push((format!("moe_out.{il}"), vec![0f32; block * d]));
                let r1: Vec<(&str, &[f32])> =
                    i1.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
                let o1 = stages[il].run(&r1);
                let xf = o1[pos(&format!("moe_in.{il}")).expect("moe_in output")].clone();
                let mo = moe_fn(il, &blk, &xf)?;
                let mut i2 = ins;
                i2.push((format!("moe_out.{il}"), mo));
                let r2: Vec<(&str, &[f32])> =
                    i2.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
                stages[il].run(&r2)
            } else {
                let r: Vec<(&str, &[f32])> = ins
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_slice()))
                    .collect();
                stages[il].run(&r)
            };
            for nm in [
                format!("kvnew.{il}"),
                format!("ck.{il}"),
                format!("cg.{il}"),
                format!("comp.{il}"),
                format!("main_hidden.{il}"),
            ] {
                if let Some(p) = pos(&nm) {
                    collected.insert(nm.clone(), vout[p].clone());
                }
            }
            if il + 1 == spec.n_layers {
                logits = vout[pos("logits").expect("logits output")].clone();
            } else {
                hidden = vout[pos("hidden_out").expect("hidden_out output")].clone();
            }
        }
        let row_argmax = |r: usize| argmax(&logits[r * vocab..(r + 1) * vocab]);
        let mut a = 0usize;
        while a < bl - 1 && blk[a + 1] == row_argmax(a) {
            a += 1;
        }
        for j in 0..=a {
            out.push(blk[j]);
        }
        last = logits[a * vocab..(a + 1) * vocab].to_vec();
        dec.commit_block(a, |nm| collected.get(nm).cloned().unwrap_or_default());
        // Extend the drafter's main_hidden history with the accepted positions'
        // target-layer hiddens (in `dspark_target_layer_ids` order → [n, dim*n_targets]).
        for j in 0..=a {
            for &tl in &spec.dspark_target_layer_ids {
                if let Some(mh) = collected.get(&format!("main_hidden.{tl}")) {
                    main_hist.extend_from_slice(&mh[j * d..(j + 1) * d]);
                }
            }
        }
    }
    out.truncate(n_new);
    Ok(out)
}

/// Source of individual MoE experts, read ON DEMAND — the core of **Expert Paging**:
/// only the `top_k` active experts per token ever touch memory, so the ~140 GB of GA
/// experts stay on disk (mmap). `fetch` returns expert `e`'s `proj`
/// (`"gate_proj"`/`"up_proj"`/`"down_proj"`) for layer `il` as row-major `[out, in]`
/// F32 (dequantized). A real impl reads a SINGLE per-expert tensor from the mmap
/// loader (only its pages fault in); cold experts are never materialized — turning a
/// resident 156 GB into ~backbone + 6/256·experts ≈ 20 GB.
pub trait ExpertSource {
    fn fetch(&mut self, il: usize, e: usize, proj: &str) -> Result<Vec<f32>>;

    /// **Parallel prefetch** the given `(layer, expert)` set before the serial
    /// `fetch` calls — issues all disk page-faults at once (uses the SSD's queue
    /// depth) instead of stalling one round-trip per tensor. Default no-op (in-memory
    /// / already-warm sources). Called by [`paged_moe_forward`] for a token's active
    /// experts. Overlaps the expert-paging I/O, which is otherwise latency-bound.
    fn prewarm(&mut self, _experts: &[(usize, usize)]) {}
}

fn softplus_host(x: f32) -> f32 {
    // ln(1+e^x) in the stable max(x,0)+ln(1+e^-|x|) form (matches softplus_stable).
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

/// Dense (shared-expert / dense-FFN) SwiGLU for one token: `down · (silu(clamp(gate·x))
/// · clamp(up·x))`. `gate`/`up` are `[inter, h]` row-major, `down` is `[h, inter]`;
/// `swiglu_limit > 0` applies V4's clamp (gate ≤ L, up ∈ [-L, L]). Matches the
/// in-graph shared expert — add its output to [`PagedGroupedMoe`]'s routed result to
/// form the complete MoE layer host-side.
pub fn dense_swiglu_ffn(
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    swiglu_limit: f32,
) -> Vec<f32> {
    let h = x.len();
    let inter = gate.len() / h.max(1);
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let mut glu = vec![0f32; inter];
    for (r, gr) in glu.iter_mut().enumerate() {
        let (mut a, mut b) = (0f32, 0f32);
        for i in 0..h {
            a += gate[r * h + i] * x[i];
            b += up[r * h + i] * x[i];
        }
        if swiglu_limit > 0.0 {
            a = a.min(swiglu_limit);
            b = b.clamp(-swiglu_limit, swiglu_limit);
        }
        *gr = silu(a) * b;
    }
    let mut out = vec![0f32; h];
    for (r, o) in out.iter_mut().enumerate() {
        let mut s = 0f32;
        for i in 0..inter {
            s += down[r * inter + i] * glu[i];
        }
        *o = s;
    }
    out
}

/// **Hash routing** (DeepSeek-V4's first `n_hash_layers`): the `top_k` experts for a
/// token come from a fixed `gate.tid2eid` table `[vocab, top_k]` (expert ids stored as
/// f32), NOT from score top-k. Returns `tid2eid[token]` as expert indices — feed as
/// `hash_eids` to [`paged_moe_route`] on those layers. Mirrors the in-graph
/// `Gather(tid2eid, token)` in `build_deepseek_moe_c`.
pub fn hash_route_experts(tid2eid: &[f32], top_k: usize, token: u32) -> Vec<usize> {
    let base = token as usize * top_k;
    (0..top_k)
        .map(|j| {
            tid2eid
                .get(base + j)
                .map(|&v| v.max(0.0) as usize)
                .unwrap_or(0)
        })
        .collect()
}

/// Route one token: V4 `sqrtsoftplus` scores → select `top_k` experts (hash `tid2eid`
/// row when given, else score top-k on `scores + ebias`) → per-expert weights (the
/// ORIGINAL scores, optionally `norm_topk_prob`-normalized and `routed_scaling`d).
/// Returns `(expert_ids, weights)`. Matches `build_deepseek_moe_c` exactly; split out
/// so the paging selection is independently checkable against the graph's TopK.
pub fn paged_moe_route(
    spec: &DeepseekSpec,
    h2d: &[f32],
    router_w: &[f32],
    ebias: Option<&[f32]>,
    hash_eids: Option<&[usize]>,
) -> (Vec<usize>, Vec<f32>) {
    let (h, n, top_k) = (
        spec.hidden_size,
        spec.n_routed_experts,
        spec.num_experts_per_tok,
    );
    let mut scores = vec![0f32; n];
    for (e, se) in scores.iter_mut().enumerate() {
        let mut s = 0f32;
        for i in 0..h {
            s += router_w[e * h + i] * h2d[i];
        }
        *se = softplus_host(s).sqrt();
    }
    let top: Vec<usize> = if let Some(eids) = hash_eids {
        eids.to_vec()
    } else {
        let mut route: Vec<(usize, f32)> = (0..n)
            .map(|e| (e, scores[e] + ebias.map_or(0.0, |b| b[e])))
            .collect();
        // Descending by selection score; index tie-break (stable) — order among the
        // chosen set doesn't affect the summed output.
        route.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        route[..top_k].iter().map(|&(e, _)| e).collect()
    };
    let mut w: Vec<f32> = top.iter().map(|&e| scores[e]).collect();
    if spec.norm_topk_prob {
        let s: f32 = w.iter().sum();
        if s != 0.0 {
            w.iter_mut().for_each(|x| *x /= s);
        }
    }
    if (spec.routed_scaling_factor - 1.0).abs() > f32::EPSILON {
        w.iter_mut().for_each(|x| *x *= spec.routed_scaling_factor);
    }
    (top, w)
}

/// **Fused** dequant-matvec of one packed expert projection: `out[out_dim] =
/// x[in_dim] @ dequant(W)ᵀ`, reading the packed codes directly (no `[out,in]` f32
/// materialization). MXFP4/affine use the fused kernels; other schemes fall back to
/// materialize-then-matvec.
fn fused_matvec(
    p: &crate::weight_loader::MlxPackedLinear,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    match p.scheme {
        QuantScheme::MlxMxfp4 { group_size } => {
            let sc = expert_scales_f32(p); // e8m0 → f32
            Ok(rlx_mlx_io::dequant_matvec_mxfp4(
                x, &p.w_q, &sc, group_size, in_dim, out_dim,
            )?)
        }
        QuantScheme::MlxAffine { bits, group_size } => {
            let sc = f32_from_le_bytes(&p.scales);
            let ng = in_dim / (group_size as usize).max(1);
            let bi = if p.biases.is_empty() {
                vec![0f32; out_dim * ng]
            } else {
                f32_from_le_bytes(&p.biases)
            };
            Ok(rlx_mlx_io::dequant_matvec_affine(
                x,
                &p.w_q,
                &sc,
                &bi,
                bits as u32,
                group_size,
                in_dim,
                out_dim,
            )?)
        }
        _ => {
            let w = dequant_packed_linear(p)?; // [out, in] f32 (mxfp8 etc.)
            Ok((0..out_dim)
                .map(|j| (0..in_dim).map(|i| x[i] * w[j * in_dim + i]).sum::<f32>())
                .collect())
        }
    }
}

/// **FUSED host paged MoE** — like [`paged_moe_forward`] but takes packed experts
/// ([`PackedExpertSource`]) and runs FUSED dequant-matvec directly on the codes, so
/// no expert weight is ever materialized to `[out,in]` f32 (the MXFP4 routed experts
/// read `n·k/2` bytes, not `n·k·4`). Bit-exact with `paged_moe_forward`.
#[allow(clippy::too_many_arguments)]
pub fn paged_moe_forward_fused(
    spec: &DeepseekSpec,
    il: usize,
    h2d: &[f32],
    router_w: &[f32],
    ebias: Option<&[f32]>,
    hash_eids: Option<&[usize]>,
    src: &mut dyn PackedExpertSource,
    shared: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Vec<f32>> {
    let (h, inter) = (spec.hidden_size, spec.moe_intermediate_size);
    let (top, w) = paged_moe_route(spec, h2d, router_w, ebias, hash_eids);
    let active: Vec<(usize, usize)> = top.iter().map(|&e| (il, e)).collect();
    src.prewarm(&active);
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let lim = spec.swiglu_limit;
    let mut out = vec![0f32; h];
    for (ki, &e) in top.iter().enumerate() {
        let pg = src.fetch_packed(il, e, "gate_proj")?; // [inter, h]
        let pu = src.fetch_packed(il, e, "up_proj")?;
        let pd = src.fetch_packed(il, e, "down_proj")?; // [h, inter]
        let gate = fused_matvec(&pg, h2d, inter, h)?;
        let up = fused_matvec(&pu, h2d, inter, h)?;
        let mut glu = vec![0f32; inter];
        for (r, gr) in glu.iter_mut().enumerate() {
            let (mut sg, mut su) = (gate[r], up[r]);
            if lim > 0.0 {
                sg = sg.min(lim);
                su = su.clamp(-lim, lim);
            }
            *gr = silu(sg) * su;
        }
        let down = fused_matvec(&pd, &glu, h, inter)?;
        for (r, o) in out.iter_mut().enumerate() {
            *o += w[ki] * down[r];
        }
    }
    if let Some((sg_w, su_w, sd_w)) = shared {
        let sh = dense_swiglu_ffn(h2d, sg_w, su_w, sd_w, lim);
        for (o, s) in out.iter_mut().zip(&sh) {
            *o += s;
        }
    }
    Ok(out)
}

/// Host-side MoE for one token with **expert paging**: [`paged_moe_route`] → load
/// ONLY the selected experts via `src` → clamped-SwiGLU → weighted sum (+ dense
/// shared expert). Numerically matches the in-graph `build_deepseek_moe_ffn` but
/// never materializes the full expert set. `shared` = `(gate[se·h], up[se·h],
/// down[h·se])` row-major, or `None`.
#[allow(clippy::too_many_arguments)]
pub fn paged_moe_forward(
    spec: &DeepseekSpec,
    il: usize,
    h2d: &[f32],
    router_w: &[f32],
    ebias: Option<&[f32]>,
    hash_eids: Option<&[usize]>,
    src: &mut dyn ExpertSource,
    shared: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Vec<f32>> {
    let (h, inter) = (spec.hidden_size, spec.moe_intermediate_size);
    let (top, w) = paged_moe_route(spec, h2d, router_w, ebias, hash_eids);
    // Parallel-prefetch this token's active experts before the serial fetch loop —
    // overlaps the disk page-faults instead of one round-trip per tensor.
    let active: Vec<(usize, usize)> = top.iter().map(|&e| (il, e)).collect();
    src.prewarm(&active);
    use rayon::prelude::*;
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let lim = spec.swiglu_limit;
    let mut out = vec![0f32; h];
    for (ki, &e) in top.iter().enumerate() {
        let w1 = src.fetch(il, e, "gate_proj")?; // [inter, h]
        let w3 = src.fetch(il, e, "up_proj")?; // [inter, h]
        let w2 = src.fetch(il, e, "down_proj")?; // [h, inter]
        // clamped-SwiGLU over the inter rows, parallel.
        let mut glu = vec![0f32; inter];
        glu.par_iter_mut().enumerate().for_each(|(r, gr)| {
            let (mut sg, mut su) = (0f32, 0f32);
            for i in 0..h {
                let x = h2d[i];
                sg += w1[r * h + i] * x;
                su += w3[r * h + i] * x;
            }
            if lim > 0.0 {
                sg = sg.min(lim);
                su = su.clamp(-lim, lim);
            }
            *gr = silu(sg) * su;
        });
        // down-proj over the h rows, parallel; accumulate the weighted contribution.
        let contrib: Vec<f32> = (0..h)
            .into_par_iter()
            .map(|r| {
                let mut s = 0f32;
                for i in 0..inter {
                    s += w2[r * inter + i] * glu[i];
                }
                w[ki] * s
            })
            .collect();
        for (o, c) in out.iter_mut().zip(&contrib) {
            *o += c;
        }
    }
    if let Some((sg_w, su_w, sd_w)) = shared {
        let se = sg_w.len() / h;
        let mut glu = vec![0f32; se];
        for (r, gr) in glu.iter_mut().enumerate() {
            let (mut a, mut b) = (0f32, 0f32);
            for i in 0..h {
                let x = h2d[i];
                a += sg_w[r * h + i] * x;
                b += su_w[r * h + i] * x;
            }
            *gr = silu(a) * b;
        }
        for (r, o) in out.iter_mut().enumerate() {
            let mut s = 0f32;
            for i in 0..se {
                s += sd_w[r * se + i] * glu[i];
            }
            *o += s;
        }
    }
    Ok(out)
}

/// **Batched host-side MoE with expert paging — the WINNING reduce strategy (S3
/// grouped scatter-reduce)** wired into the paging path. Given `batch` tokens
/// `x = [batch, h]` (row-major), route each, **group the tokens by expert**, fetch
/// each active expert's weights **exactly once per batch** (the paging + memory-band
/// win), compute its whole token-group in parallel, and **scatter-add** the weighted
/// SwiGLU output back per token.
///
/// vs the per-token [`paged_moe_forward`] (S1 token-major, which re-fetches +
/// re-reads an expert's weights for every token that picks it) this reads each
/// active expert's weights ONCE for its `m` tokens and parallelizes the group's
/// GEMV rows across cores — the microbench (`moe_reduce_bench`) measured this at
/// 1.6–11× faster (widening with batch and on slower CPUs) with byte-identical
/// output. `hash_eids[b]` = optional per-token hash route; `shared` = dense shared
/// expert applied to every token.
#[allow(clippy::too_many_arguments)]
pub fn paged_moe_forward_batched(
    spec: &DeepseekSpec,
    il: usize,
    x: &[f32],
    batch: usize,
    router_w: &[f32],
    ebias: Option<&[f32]>,
    hash_eids: Option<&[Vec<usize>]>,
    src: &mut dyn ExpertSource,
    shared: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;
    let (h, inter, n) = (
        spec.hidden_size,
        spec.moe_intermediate_size,
        spec.n_routed_experts,
    );
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let lim = spec.swiglu_limit;
    // Route every token, then invert into per-expert token groups: expert →
    // [(token, gate_weight)]. This is the "group tokens by expert" of S3.
    let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for b in 0..batch {
        let he = hash_eids.map(|hs| hs[b].as_slice());
        let (top, w) = paged_moe_route(spec, &x[b * h..b * h + h], router_w, ebias, he);
        for (e, gw) in top.into_iter().zip(w) {
            groups[e].push((b, gw));
        }
    }
    let mut out = vec![0f32; batch * h];
    for (e, toks) in groups.iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        // Fetch this expert ONCE for its whole token group (the paging win).
        let w1 = src.fetch(il, e, "gate_proj")?; // [inter, h]
        let w3 = src.fetch(il, e, "up_proj")?; // [inter, h]
        let w2 = src.fetch(il, e, "down_proj")?; // [h, inter]
        // Compute each token in the group in parallel (rows across cores), producing
        // its weighted [h] contribution; reduce serially into `out` (each token is
        // distinct within one expert's group, so there is no write conflict).
        let contrib: Vec<(usize, Vec<f32>)> = toks
            .par_iter()
            .map(|&(b, gw)| {
                let xb = &x[b * h..b * h + h];
                let mut glu = vec![0f32; inter];
                for (r, gr) in glu.iter_mut().enumerate() {
                    let (mut sg, mut su) = (0f32, 0f32);
                    for i in 0..h {
                        let xi = xb[i];
                        sg += w1[r * h + i] * xi;
                        su += w3[r * h + i] * xi;
                    }
                    if lim > 0.0 {
                        sg = sg.min(lim);
                        su = su.clamp(-lim, lim);
                    }
                    *gr = silu(sg) * su;
                }
                let mut d = vec![0f32; h];
                for (r, dr) in d.iter_mut().enumerate() {
                    let mut s = 0f32;
                    for i in 0..inter {
                        s += w2[r * inter + i] * glu[i];
                    }
                    *dr = gw * s;
                }
                (b, d)
            })
            .collect();
        for (b, d) in contrib {
            let o = &mut out[b * h..b * h + h];
            for (oi, di) in o.iter_mut().zip(&d) {
                *oi += di;
            }
        }
    }
    // Dense shared expert: same weights for every token → one grouped GEMM over all
    // `batch` tokens (parallel rows), scatter-add.
    if let Some((sg_w, su_w, sd_w)) = shared {
        let se = sg_w.len() / h;
        let shared_out: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let xb = &x[b * h..b * h + h];
                let mut glu = vec![0f32; se];
                for (r, gr) in glu.iter_mut().enumerate() {
                    let (mut a, mut bb) = (0f32, 0f32);
                    for i in 0..h {
                        let xi = xb[i];
                        a += sg_w[r * h + i] * xi;
                        bb += su_w[r * h + i] * xi;
                    }
                    *gr = silu(a) * bb;
                }
                let mut d = vec![0f32; h];
                for (r, dr) in d.iter_mut().enumerate() {
                    let mut s = 0f32;
                    for i in 0..se {
                        s += sd_w[r * se + i] * glu[i];
                    }
                    *dr = s;
                }
                d
            })
            .collect();
        for (b, d) in shared_out.into_iter().enumerate() {
            let o = &mut out[b * h..b * h + h];
            for (oi, di) in o.iter_mut().zip(&d) {
                *oi += di;
            }
        }
    }
    Ok(out)
}

/// Dequantize a packed MLX Linear to row-major `[out, in]` F32.
pub fn dequant_packed_linear(p: &crate::weight_loader::MlxPackedLinear) -> Result<Vec<f32>> {
    let out = p.out_shape.first().copied().unwrap_or(0);
    let n_groups = p.n_groups().max(1);
    Ok(match p.scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let scales = f32_from_le_bytes(&p.scales);
            let biases = if p.biases.is_empty() {
                vec![0f32; out * n_groups]
            } else {
                f32_from_le_bytes(&p.biases)
            };
            rlx_mlx_io::dequant_affine_f32(
                &p.w_q,
                &scales,
                &biases,
                bits as u32,
                group_size,
                out,
                n_groups,
            )?
        }
        QuantScheme::MlxMxfp4 { group_size } => {
            rlx_mlx_io::dequant_mxfp4_f32(&p.w_q, &p.scales, group_size, out, n_groups)?
        }
        QuantScheme::MlxMxfp8 { group_size } => {
            // DeepSeek-V4 shared experts + some attention projs are 8-bit MXFP8.
            rlx_mlx_io::dequant_mxfp8_f32(&p.w_q, &p.scales, group_size, out, n_groups)?
        }
        other => return Err(anyhow!("paged expert: unsupported scheme {other}")),
    })
}

/// [`ExpertSource`] over an MLX checkpoint's **per-expert** reference tensors
/// (`layers.{il}.ffn.experts.{e}.w{1,3,2}.weight`) — the REAL Expert Paging: each
/// `fetch` reads + dequantizes a SINGLE expert straight from the mmap loader, so
/// only the ~6/256 active experts per token ever leave disk (backing the 156 GB GA
/// with ≈ backbone + a handful of experts resident). Wrap the RAW mmap loader
/// (e.g. `MlxLoader::open_lazy`), not the stacking [`DsV4RefLoader`].
pub struct PagedMlxExperts<'a> {
    pub loader: &'a mut dyn WeightLoader,
}
impl ExpertSource for PagedMlxExperts<'_> {
    fn fetch(&mut self, il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
        let key = dsv4_ref_expert_key(il, e, proj);
        let p = self
            .loader
            .take_packed_mlx(&key)?
            .ok_or_else(|| anyhow!("paged expert {key}: not MLX-packed"))?;
        dequant_packed_linear(&p)
    }
}

/// PACKED (still-quantized) access to one MoE expert — the paging primitive for the
/// **GPU** grouped path ([`PagedGroupedMoe`]): returns the per-expert MLX-packed
/// Linear (codes + scales, undequantized) so the native `DequantGroupedMatMulMlx`
/// kernel dequants + matmuls it **on-device**, instead of the host dequant that
/// [`ExpertSource`] does. Only the batch's active experts are ever fetched.
pub trait PackedExpertSource {
    fn fetch_packed(
        &mut self,
        il: usize,
        e: usize,
        proj: &str,
    ) -> Result<crate::weight_loader::MlxPackedLinear>;

    /// **Parallel prefetch** the given `(layer, expert)` set before they are fetched
    /// serially — overlaps the disk reads (the expert-paging bottleneck) by issuing
    /// all page-faults at once instead of one round-trip per tensor. Default no-op
    /// (in-memory sources). Called by [`PagedGroupedMoe`] for a batch's NEW experts.
    fn prewarm(&mut self, _experts: &[(usize, usize)]) {}

    /// **Zero-copy borrow-only** hot path: if this source can borrow the projection's
    /// packed bytes straight from its mmap, call `sink(codes, scales)` (which writes
    /// them into the device slot) and return `Some(sink_result)`; otherwise return
    /// `None` and the caller falls back to [`Self::fetch_packed`]. Returning `None`
    /// (rather than doing an owned fetch here) keeps this a pure `&self` borrow, so
    /// the caller can decide between zero-copy and owned without a borrow conflict.
    /// Halves the per-expert copy (codes go mmap→device with no owned `Vec`).
    fn with_packed_borrowed(
        &mut self,
        _il: usize,
        _e: usize,
        _proj: &str,
        _sink: &mut dyn FnMut(&[u8], &[u8]) -> bool,
    ) -> Option<bool> {
        None
    }
}

impl PackedExpertSource for PagedMlxExperts<'_> {
    fn fetch_packed(
        &mut self,
        il: usize,
        e: usize,
        proj: &str,
    ) -> Result<crate::weight_loader::MlxPackedLinear> {
        let key = dsv4_ref_expert_key(il, e, proj);
        self.loader
            .take_packed_mlx(&key)?
            .ok_or_else(|| anyhow!("paged expert {key}: not MLX-packed"))
    }

    fn prewarm(&mut self, experts: &[(usize, usize)]) {
        // Warm every active expert's 3 projections in one parallel page-fault pass.
        let keys: Vec<String> = experts
            .iter()
            .flat_map(|&(il, e)| {
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .map(move |p| dsv4_ref_expert_key(il, e, p))
            })
            .collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.loader.prewarm(&refs);
    }
}

/// Fused, parallel raw-scale-bytes → BF16 little-endian bytes (the device
/// grouped-dequant's scale format). Decodes MXFP4 `e8m0` (or reads f32) and
/// narrows to bf16 in ONE rayon pass — no intermediate f32 `Vec` and no
/// `flat_map` — since this is the paging hot loop's dominant CPU cost. Element
/// order matches [`expert_scales_f32`] output bit-for-bit.
/// Takes raw `&[u8]` so it works on both owned and zero-copy-borrowed mmap scales.
fn scales_bf16_le(scales: &[u8], scheme: QuantScheme) -> Vec<u8> {
    use rayon::prelude::*;
    match scheme {
        QuantScheme::MlxMxfp4 { .. } => {
            let mut out = vec![0u8; scales.len() * 2];
            out.par_chunks_mut(2)
                .zip(scales.par_iter())
                .for_each(|(o, &b)| {
                    let bf = (rlx_mlx_io::mxfp4_scale_e8m0_to_f32(b).to_bits() >> 16) as u16;
                    o.copy_from_slice(&bf.to_le_bytes());
                });
            out
        }
        _ => {
            let n = scales.len() / 4;
            let mut out = vec![0u8; n * 2];
            out.par_chunks_mut(2).enumerate().for_each(|(i, o)| {
                let f = f32::from_le_bytes([
                    scales[i * 4],
                    scales[i * 4 + 1],
                    scales[i * 4 + 2],
                    scales[i * 4 + 3],
                ]);
                let bf = (f.to_bits() >> 16) as u16;
                o.copy_from_slice(&bf.to_le_bytes());
            });
            out
        }
    }
}

/// Owned-`MlxPackedLinear` convenience over [`scales_bf16_le`].
fn expert_scales_bf16_le(p: &crate::weight_loader::MlxPackedLinear) -> Vec<u8> {
    scales_bf16_le(&p.scales, p.scheme)
}

/// Per-expert scales (as stored in [`MlxPackedLinear`]) → f32 `[out * n_groups]`,
/// matching exactly what [`dequant_packed_linear`] decodes (so the on-device grouped
/// dequant and the host reference agree bit-for-bit).
fn expert_scales_f32(p: &crate::weight_loader::MlxPackedLinear) -> Vec<f32> {
    match p.scheme {
        QuantScheme::MlxMxfp4 { .. } => p
            .scales
            .iter()
            .map(|&b| rlx_mlx_io::mxfp4_scale_e8m0_to_f32(b))
            .collect(),
        _ => f32_from_le_bytes(&p.scales),
    }
}

/// **GPU-accelerated paged batched MoE** — the winning S3 grouped scatter-reduce
/// ([`paged_moe_forward_batched`]) wired onto the native `DequantGroupedMatMulMlx`
/// kernel end-to-end. The compiled graph runs gate/up/down as three grouped
/// dequant-matmuls with a clamped-SwiGLU between them, all **on-device** (Metal/MLX
/// GPU, or CPU) — replacing the host dequant-to-f32 + CPU GEMM of the pure-host
/// path. Only the batch's DISTINCT active experts are uploaded into a compact
/// `a_cap`-slot param buffer (**paging preserved**: ≤ a_cap of the model's experts
/// ever resident), and the graph compiles ONCE and is reused across batches — so the
/// per-batch cost is just the active-expert upload + one run.
///
/// Build with [`PagedGroupedMoe::new`], then call [`forward`](Self::forward) per
/// batch. Rows are the flattened `(token, chosen-expert)` pairs (`≤ batch·top_k`),
/// padded to `m_cap`; the output is scatter-reduced back to `[batch, hidden]`.
/// Cumulative µs / counts for [`PagedGroupedMoe::forward`] IO-vs-compute split
/// (`RLX_IO_PROFILE`): `(prewarm, fetch+prep, upload, gpu-run, new-experts, calls)`.
/// Read + reset via [`paged_moe_io_profile_take`].
pub static PGM_PREWARM_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PGM_FETCH_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PGM_UPLOAD_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PGM_COMPUTE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PGM_NEW_EXPERTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PGM_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Read and reset the [`PagedGroupedMoe`] IO profile counters:
/// `(prewarm_ms, fetch_ms, upload_ms, compute_ms, new_experts, calls)`.
pub fn paged_moe_io_profile_take() -> (f64, f64, f64, f64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let ms = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / 1000.0;
    (
        ms(&PGM_PREWARM_US),
        ms(&PGM_FETCH_US),
        ms(&PGM_UPLOAD_US),
        ms(&PGM_COMPUTE_US),
        PGM_NEW_EXPERTS.swap(0, Relaxed),
        PGM_CALLS.swap(0, Relaxed),
    )
}

pub struct PagedGroupedMoe {
    compiled: rlx_runtime::CompiledGraph,
    a_cap: usize,
    m_cap: usize,
    hidden: usize,
    inter: usize,
    ng_h: usize,
    ng_i: usize,
    scheme: QuantScheme,
    bpr_h: usize, // packed bytes per gate/up row (K = hidden)
    bpr_i: usize, // packed bytes per down row  (K = inter)
    // Persistent compact host buffers for the 3 projs (codes / bf16 scales / bf16
    // biases), uploaded to the device only when an expert slot changes.
    gate_c: Vec<u8>,
    gate_s: Vec<u8>,
    gate_b: Vec<u8>,
    up_c: Vec<u8>,
    up_s: Vec<u8>,
    up_b: Vec<u8>,
    down_c: Vec<u8>,
    down_s: Vec<u8>,
    down_b: Vec<u8>,
    // Expert residency (LRU): which (layer, expert) occupies each compact slot.
    // Keying by (layer, expert) — not expert alone — keeps it correct across layers
    // in a real decode, where the same expert id in two layers is a different tensor.
    slot_of: HashMap<(usize, usize), usize>,
    slot_expert: Vec<Option<(usize, usize)>>,
    slot_tick: Vec<u64>,
    tick: u64,
    dirty: bool,
    /// Incremental per-slot upload: write only a changed expert's slot bytes into
    /// the device buffers (via `CompiledGraph::set_param_range`) instead of
    /// re-uploading the whole `a_cap`-slot buffer each step. Set false the first
    /// time the backend reports the partial write unsupported → whole-buffer
    /// fallback. Makes a large `a_cap` (cross-token residency) cheap to maintain.
    incremental: bool,
    /// Zero-copy fetch: borrow packed codes/scales straight from the loader's mmap
    /// into the device slot (no owned `MlxPackedLinear` copy). **Opt-in** via
    /// `RLX_PGM_ZEROCOPY=1`. Default OFF because a controlled A/B showed it is ~neutral
    /// to slightly slower in RELEASE: the copy it removes is cheap once optimized
    /// (~10 GB/s memcpy), while the per-expert `MADV_DONTNEED` it must add (to bound
    /// the page cache) roughly offsets that. (It WAS faster in debug, but that only
    /// reflects debug's slow memcpy.) Kept available for the cold-drive case where the
    /// saved copy may matter, and as the wiring for a future stream-to-device path.
    zerocopy: bool,
    /// Count of expert-projection uploads performed (telemetry: how much the
    /// residency cache saved — steady-state hot sets drive this toward zero).
    pub uploads: usize,
}

impl PagedGroupedMoe {
    /// Compile the grouped-MoE graph for a fixed capacity: up to `m_cap` rows
    /// (`≥ batch·top_k`) routed across up to `a_cap` distinct experts, hidden/inter
    /// dims, `group_size` (K-groups), and `swiglu_limit` (`0.0` = plain SwiGLU).
    /// `scheme` is the expert quant (`MlxMxfp4`/`MlxAffine`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: rlx_runtime::Device,
        a_cap: usize,
        m_cap: usize,
        hidden: usize,
        inter: usize,
        group_size: usize,
        swiglu_limit: f32,
        scheme: QuantScheme,
    ) -> Self {
        use rlx_ir::infer::GraphExt;
        let f = DType::F32;
        let (ng_h, ng_i) = (hidden / group_size, inter / group_size);
        // Bytes/row of packed codes: MXFP4 packs 2 codes/byte, affine 1 byte/code.
        let bpr = |in_dim: usize| -> usize {
            match scheme {
                QuantScheme::MlxMxfp4 { .. } => in_dim / 2,
                _ => in_dim,
            }
        };
        let mut g = Graph::new("paged_grouped_moe");
        let x = g.input("x", Shape::new(&[m_cap, hidden], f));
        let eidx = g.input("eidx", Shape::new(&[m_cap], f));
        // gate/up: [inter, hidden] per expert; down: [hidden, inter] per expert.
        let gate_c = g.param(
            "gate_c",
            Shape::new(&[a_cap * inter * bpr(hidden)], DType::U8),
        );
        let gate_s = g.param("gate_s", Shape::new(&[a_cap, inter, ng_h], DType::BF16));
        let gate_b = g.param("gate_b", Shape::new(&[a_cap, inter, ng_h], DType::BF16));
        let up_c = g.param(
            "up_c",
            Shape::new(&[a_cap * inter * bpr(hidden)], DType::U8),
        );
        let up_s = g.param("up_s", Shape::new(&[a_cap, inter, ng_h], DType::BF16));
        let up_b = g.param("up_b", Shape::new(&[a_cap, inter, ng_h], DType::BF16));
        let down_c = g.param(
            "down_c",
            Shape::new(&[a_cap * hidden * bpr(inter)], DType::U8),
        );
        let down_s = g.param("down_s", Shape::new(&[a_cap, hidden, ng_i], DType::BF16));
        let down_b = g.param("down_b", Shape::new(&[a_cap, hidden, ng_i], DType::BF16));

        let gate = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![x, gate_c, gate_s, gate_b, eidx],
            Shape::new(&[m_cap, inter], f),
        );
        let up = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![x, up_c, up_s, up_b, eidx],
            Shape::new(&[m_cap, inter], f),
        );
        // Clamped SwiGLU: glu = silu(min(gate, L)) * clamp(up, ±L)  (matches the
        // host `paged_moe_forward_batched`; L≤0 ⇒ plain SwiGLU).
        let (gate_a, up_a) = if swiglu_limit > 0.0 {
            let gc = g.add_node(
                Op::Clamp {
                    min: f32::NEG_INFINITY,
                    max: swiglu_limit,
                },
                vec![gate],
                Shape::new(&[m_cap, inter], f),
            );
            let uc = g.add_node(
                Op::Clamp {
                    min: -swiglu_limit,
                    max: swiglu_limit,
                },
                vec![up],
                Shape::new(&[m_cap, inter], f),
            );
            (gc, uc)
        } else {
            (gate, up)
        };
        let act = g.silu(gate_a);
        let glu = g.mul(act, up_a);
        let down = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![glu, down_c, down_s, down_b, eidx],
            Shape::new(&[m_cap, hidden], f),
        );
        g.set_outputs(vec![down]);
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill(device);
        let compiled = rlx_runtime::Session::new(device).compile_with(g, &opts);
        let (bpr_h, bpr_i) = match scheme {
            QuantScheme::MlxMxfp4 { .. } => (hidden / 2, inter / 2),
            _ => (hidden, inter),
        };
        // Host buffers stay EMPTY on the incremental fast path (direct per-slot
        // device writes) — sized lazily only if the backend forces the whole-buffer
        // fallback, so a large `a_cap` residency costs no idle host RAM (paging's
        // whole point). See [`Self::write_slot`].
        Self {
            compiled,
            a_cap,
            m_cap,
            hidden,
            inter,
            ng_h,
            ng_i,
            scheme,
            bpr_h,
            bpr_i,
            gate_c: Vec::new(),
            gate_s: Vec::new(),
            gate_b: Vec::new(),
            up_c: Vec::new(),
            up_s: Vec::new(),
            up_b: Vec::new(),
            down_c: Vec::new(),
            down_s: Vec::new(),
            down_b: Vec::new(),
            incremental: true,
            zerocopy: rlx_ir::env::flag("RLX_PGM_ZEROCOPY"),
            slot_of: HashMap::new(),
            slot_expert: vec![None; a_cap],
            slot_tick: vec![0; a_cap],
            tick: 0,
            dirty: false,
            uploads: 0,
        }
    }

    /// Write one expert's packed gate/up/down into compact `slot`. Fast path
    /// (`incremental`): decode scales to bf16 once and partial-write codes (straight
    /// from the packed source, zero-copy) + scales + zero-bias DIRECTLY into the
    /// device slot — no big host-buffer round-trip, no re-copy. Fallback path
    /// (backend without partial writes): populate the persistent host buffers and
    /// mark them dirty for a whole-buffer upload in [`forward`](Self::forward).
    fn write_slot(
        &mut self,
        slot: usize,
        gate: &crate::weight_loader::MlxPackedLinear,
        up: &crate::weight_loader::MlxPackedLinear,
        down: &crate::weight_loader::MlxPackedLinear,
    ) {
        self.uploads += 1;
        let (cs_g, se_g) = (self.inter * self.bpr_h, self.inter * self.ng_h);
        let (cs_d, se_d) = (self.hidden * self.bpr_i, self.hidden * self.ng_i);
        if self.incremental {
            let zero = vec![0u8; se_g.max(se_d) * 2];
            let c = &mut self.compiled;
            let mut proj = |cn: &str,
                            sn: &str,
                            bn: &str,
                            cslab: usize,
                            se: usize,
                            p: &crate::weight_loader::MlxPackedLinear| {
                debug_assert_eq!(p.w_q.len(), cslab, "packed code slab size mismatch");
                let sb = expert_scales_bf16_le(p);
                let mut o = c.set_param_range(cn, slot * cslab, &p.w_q); // codes: zero-copy
                o &= c.set_param_range(sn, slot * se * 2, &sb);
                o &= c.set_param_range(bn, slot * se * 2, &zero[..se * 2]);
                o
            };
            let ok = proj("gate_c", "gate_s", "gate_b", cs_g, se_g, gate)
                & proj("up_c", "up_s", "up_b", cs_g, se_g, up)
                & proj("down_c", "down_s", "down_b", cs_d, se_d, down);
            if ok {
                return;
            }
            // Partial write unsupported on this backend → fall back for the run.
            self.incremental = false;
        }
        // Whole-buffer host path: size the persistent buffers on first use, then
        // populate + upload when dirty.
        if self.gate_c.is_empty() {
            let (code_g, scale_g) = (self.a_cap * cs_g, self.a_cap * se_g * 2);
            let (code_d, scale_d) = (self.a_cap * cs_d, self.a_cap * se_d * 2);
            self.gate_c = vec![0u8; code_g];
            self.gate_s = vec![0u8; scale_g];
            self.gate_b = vec![0u8; scale_g];
            self.up_c = vec![0u8; code_g];
            self.up_s = vec![0u8; scale_g];
            self.up_b = vec![0u8; scale_g];
            self.down_c = vec![0u8; code_d];
            self.down_s = vec![0u8; scale_d];
            self.down_b = vec![0u8; scale_d];
        }
        // Whole-buffer host path: populate host buffers, uploaded when dirty.
        let put = |codes: &mut [u8],
                   scales: &mut [u8],
                   biases: &mut [u8],
                   code_slab: usize,
                   scale_elems: usize,
                   p: &crate::weight_loader::MlxPackedLinear| {
            let co = slot * code_slab;
            codes[co..co + code_slab].fill(0);
            codes[co..co + p.w_q.len()].copy_from_slice(&p.w_q);
            let sb = expert_scales_bf16_le(p);
            let so = slot * scale_elems * 2;
            scales[so..so + scale_elems * 2].fill(0);
            scales[so..so + sb.len()].copy_from_slice(&sb);
            biases[so..so + scale_elems * 2].fill(0); // zero bias
        };
        let (mut gc, mut gs, mut gb) = (
            std::mem::take(&mut self.gate_c),
            std::mem::take(&mut self.gate_s),
            std::mem::take(&mut self.gate_b),
        );
        let (mut uc, mut us, mut ub) = (
            std::mem::take(&mut self.up_c),
            std::mem::take(&mut self.up_s),
            std::mem::take(&mut self.up_b),
        );
        let (mut dc, mut ds, mut db) = (
            std::mem::take(&mut self.down_c),
            std::mem::take(&mut self.down_s),
            std::mem::take(&mut self.down_b),
        );
        put(&mut gc, &mut gs, &mut gb, cs_g, se_g, gate);
        put(&mut uc, &mut us, &mut ub, cs_g, se_g, up);
        put(&mut dc, &mut ds, &mut db, cs_d, se_d, down);
        (self.gate_c, self.gate_s, self.gate_b) = (gc, gs, gb);
        (self.up_c, self.up_s, self.up_b) = (uc, us, ub);
        (self.down_c, self.down_s, self.down_b) = (dc, ds, db);
        self.dirty = true;
    }

    /// **Zero-copy** slot write: for each proj, ask the source to borrow the packed
    /// codes+scales straight from its mmap ([`PackedExpertSource::with_packed_borrowed`])
    /// and write them into the device slot with NO owned `MlxPackedLinear` copy (codes
    /// go mmap→device directly; scales decode mmap→bf16→device). Returns `Ok(true)`
    /// when all 3 projs were written this way; `Ok(false)` when the source can't borrow
    /// (caller uses the owned [`Self::write_slot`]) or the backend lacks partial writes
    /// (sets `incremental=false`, caller falls back). No owned fetch happens here.
    fn write_expert_zerocopy(
        &mut self,
        slot: usize,
        il: usize,
        e: usize,
        src: &mut dyn PackedExpertSource,
    ) -> Result<bool> {
        let (cs_g, se_g) = (self.inter * self.bpr_h, self.inter * self.ng_h);
        let (cs_d, se_d) = (self.hidden * self.bpr_i, self.hidden * self.ng_i);
        let zero = vec![0u8; se_g.max(se_d) * 2];
        let scheme = self.scheme;
        let projs = [
            ("gate_proj", "gate_c", "gate_s", "gate_b", cs_g, se_g),
            ("up_proj", "up_c", "up_s", "up_b", cs_g, se_g),
            ("down_proj", "down_c", "down_s", "down_b", cs_d, se_d),
        ];
        for (proj, cn, sn, bn, cslab, se) in projs {
            // Scope the `self.compiled` borrow so we can mutate `self.incremental` after.
            let (res, fail) = {
                let compiled = &mut self.compiled;
                let z = &zero;
                let mut fail = false;
                let mut sink = |codes: &[u8], scales: &[u8]| -> bool {
                    debug_assert_eq!(codes.len(), cslab, "borrowed code slab size mismatch");
                    let sb = scales_bf16_le(scales, scheme);
                    let mut o = compiled.set_param_range(cn, slot * cslab, codes);
                    o &= compiled.set_param_range(sn, slot * se * 2, &sb);
                    o &= compiled.set_param_range(bn, slot * se * 2, &z[..se * 2]);
                    if !o {
                        fail = true;
                    }
                    o
                };
                let res = src.with_packed_borrowed(il, e, proj, &mut sink);
                (res, fail)
            };
            if res.is_none() {
                // Source can't borrow this proj → caller uses the owned path.
                return Ok(false);
            }
            if fail {
                // Backend has no partial-write path → fall back to whole-buffer upload.
                self.incremental = false;
                return Ok(false);
            }
        }
        self.uploads += 1;
        Ok(true)
    }

    /// Run one batch. `x` = `[batch, hidden]` token activations; `routes[b]` =
    /// `(expert_id, gate_weight)` pairs for token `b` (from [`paged_moe_route`]).
    ///
    /// Experts already resident from a previous call are **reused in place** — only
    /// experts new to this batch are fetched (paging) + written into a free/LRU slot,
    /// and the device buffers are re-uploaded ONLY if something changed. So a stable
    /// hot set makes steady-state cost ≈ the on-device grouped GEMM + the per-token
    /// `x`/`eidx` upload (see [`uploads`](Self::uploads)).
    pub fn forward(
        &mut self,
        il: usize,
        x: &[f32],
        batch: usize,
        routes: &[Vec<(usize, f32)>],
        src: &mut dyn PackedExpertSource,
    ) -> Result<Vec<f32>> {
        let h = self.hidden;
        self.tick += 1;
        let now = self.tick;
        // Rows = flattened (token, expert, weight); distinct active experts.
        let mut rows: Vec<(usize, usize, f32)> = Vec::new(); // (token, global_e, weight)
        let mut active: Vec<usize> = Vec::new();
        let mut seen: HashMap<usize, ()> = HashMap::new();
        for (b, r) in routes.iter().enumerate().take(batch) {
            for &(e, w) in r {
                if seen.insert(e, ()).is_none() {
                    active.push(e);
                }
                rows.push((b, e, w));
            }
        }
        if active.len() > self.a_cap {
            return Err(anyhow!(
                "PagedGroupedMoe: {} distinct active experts exceeds a_cap={}",
                active.len(),
                self.a_cap
            ));
        }
        if rows.len() > self.m_cap {
            return Err(anyhow!(
                "PagedGroupedMoe: {} rows exceeds m_cap={}",
                rows.len(),
                self.m_cap
            ));
        }
        // Parallel-prefetch the NEW (non-resident) experts before the serial fetch —
        // overlaps the disk page-faults (the paging bottleneck) instead of stalling
        // one round-trip per tensor. No-op for in-memory sources.
        let new_experts: Vec<(usize, usize)> = active
            .iter()
            .filter(|&&e| !self.slot_of.contains_key(&(il, e)))
            .map(|&e| (il, e))
            .collect();
        use std::sync::atomic::Ordering::Relaxed;
        let prof = rlx_ir::env::flag("RLX_IO_PROFILE");
        PGM_CALLS.fetch_add(1, Relaxed);
        PGM_NEW_EXPERTS.fetch_add(new_experts.len() as u64, Relaxed);
        if !new_experts.is_empty() {
            let t = std::time::Instant::now();
            src.prewarm(&new_experts);
            if prof {
                PGM_PREWARM_US.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
            }
        }
        // Assign each active expert a slot: reuse if resident, else a free/LRU slot.
        // Residency is keyed by (layer, expert) so it stays correct across layers.
        for &e in &active {
            let ekey = (il, e);
            if !self.slot_of.contains_key(&ekey) {
                // Pick a free slot, else evict the LRU slot NOT in the active set.
                let slot = if let Some(s) = self.slot_expert.iter().position(|x| x.is_none()) {
                    s
                } else {
                    let active_set: std::collections::HashSet<(usize, usize)> =
                        active.iter().map(|&a| (il, a)).collect();
                    let mut best = None;
                    for s in 0..self.a_cap {
                        let occ = self.slot_expert[s].unwrap();
                        if !active_set.contains(&occ)
                            && best.is_none_or(|(_, t)| self.slot_tick[s] < t)
                        {
                            best = Some((s, self.slot_tick[s]));
                        }
                    }
                    best.map(|(s, _)| s)
                        .ok_or_else(|| anyhow!("PagedGroupedMoe: no evictable slot"))?
                };
                if let Some(old) = self.slot_expert[slot] {
                    self.slot_of.remove(&old);
                }
                let tr = std::time::Instant::now();
                // Zero-copy fast path: borrow codes+scales from the loader's mmap and
                // write straight to the device slot (no owned MlxPackedLinear copy).
                let zc = if self.incremental && self.zerocopy {
                    self.write_expert_zerocopy(slot, il, e, src)?
                } else {
                    false
                };
                if !zc {
                    // Owned path: fetch the 3 projs, then write_slot (owned incremental
                    // or whole-buffer host fallback).
                    let gate = src.fetch_packed(il, e, "gate_proj")?;
                    let up = src.fetch_packed(il, e, "up_proj")?;
                    let down = src.fetch_packed(il, e, "down_proj")?;
                    self.write_slot(slot, &gate, &up, &down);
                }
                if prof {
                    PGM_FETCH_US.fetch_add(tr.elapsed().as_micros() as u64, Relaxed);
                }
                self.slot_expert[slot] = Some(ekey);
                self.slot_of.insert(ekey, slot);
            }
            let s = self.slot_of[&ekey];
            self.slot_tick[s] = now;
        }
        // Upload buffers only if a slot changed since the last run.
        let tu = std::time::Instant::now();
        if self.dirty {
            self.compiled
                .set_param_typed("gate_c", &self.gate_c, DType::U8);
            self.compiled
                .set_param_typed("gate_s", &self.gate_s, DType::BF16);
            self.compiled
                .set_param_typed("gate_b", &self.gate_b, DType::BF16);
            self.compiled.set_param_typed("up_c", &self.up_c, DType::U8);
            self.compiled
                .set_param_typed("up_s", &self.up_s, DType::BF16);
            self.compiled
                .set_param_typed("up_b", &self.up_b, DType::BF16);
            self.compiled
                .set_param_typed("down_c", &self.down_c, DType::U8);
            self.compiled
                .set_param_typed("down_s", &self.down_s, DType::BF16);
            self.compiled
                .set_param_typed("down_b", &self.down_b, DType::BF16);
            self.dirty = false;
        }
        if prof {
            PGM_UPLOAD_US.fetch_add(tu.elapsed().as_micros() as u64, Relaxed);
        }
        // Padded row inputs: x_exp[m_cap, h], eidx[m_cap] (slot index per row).
        let mut x_exp = vec![0f32; self.m_cap * h];
        let mut eidx = vec![0f32; self.m_cap];
        for (r, &(b, e, _)) in rows.iter().enumerate() {
            x_exp[r * h..r * h + h].copy_from_slice(&x[b * h..b * h + h]);
            eidx[r] = self.slot_of[&(il, e)] as f32;
        }
        let tc = std::time::Instant::now();
        let out_rows = self
            .compiled
            .run(&[("x", x_exp.as_slice()), ("eidx", eidx.as_slice())])[0]
            .clone();
        if prof {
            PGM_COMPUTE_US.fetch_add(tc.elapsed().as_micros() as u64, Relaxed);
        }
        // Scatter-reduce: out[token] += weight * down_row.
        let mut out = vec![0f32; batch * h];
        for (r, &(b, _, w)) in rows.iter().enumerate() {
            let dr = &out_rows[r * h..r * h + h];
            let ob = &mut out[b * h..b * h + h];
            for (oi, di) in ob.iter_mut().zip(dr) {
                *oi += w * di;
            }
        }
        Ok(out)
    }
}

/// **GPU dense shared expert** — the always-on shared FFN (`silu(clamp(x·Wg))·clamp(x·Wu)`
/// then `·Wd`) run ON-DEVICE, so the paged decode's LAST host-side MoE piece moves to
/// the GPU alongside [`PagedGroupedMoe`]'s routed experts. Compiles ONE dense-SwiGLU
/// graph (f32 weights) and reuses it across layers; [`forward`](Self::forward) uploads
/// the layer's shared weights (already dequantized to f32) + runs. Bit-exact with the
/// host [`dense_swiglu_ffn`].
pub struct SharedExpertGpu {
    compiled: rlx_runtime::CompiledGraph,
    dim: usize,
    se: usize,
}

impl SharedExpertGpu {
    /// Compile the dense-SwiGLU graph for `dim`-wide tokens and `se_inter` hidden.
    pub fn new(
        device: rlx_runtime::Device,
        dim: usize,
        se_inter: usize,
        swiglu_limit: f32,
    ) -> Self {
        use rlx_ir::infer::GraphExt;
        let f = DType::F32;
        let mut g = Graph::new("shared_expert_gpu");
        let x = g.input("x", Shape::new(&[1, dim], f));
        // Params stored TRANSPOSED for `x[1,dim] @ W` matmuls: gate/up = [dim, se],
        // down = [se, dim] (host transposes on upload).
        let sg = g.param("sg", Shape::new(&[dim, se_inter], f));
        let su = g.param("su", Shape::new(&[dim, se_inter], f));
        let sd = g.param("sd", Shape::new(&[se_inter, dim], f));
        let gate = g.mm(x, sg); // [1, se]
        let up = g.mm(x, su);
        let (gate, up) = if swiglu_limit > 0.0 {
            let l = swiglu_limit;
            (
                g.add_node(
                    Op::Clamp {
                        min: f32::NEG_INFINITY,
                        max: l,
                    },
                    vec![gate],
                    Shape::new(&[1, se_inter], f),
                ),
                g.add_node(
                    Op::Clamp { min: -l, max: l },
                    vec![up],
                    Shape::new(&[1, se_inter], f),
                ),
            )
        } else {
            (gate, up)
        };
        let act = g.silu(gate);
        let glu = g.mul(act, up);
        let down = g.mm(glu, sd); // [1, dim]
        g.set_outputs(vec![down]);
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill(device);
        let compiled = rlx_runtime::Session::new(device).compile_with(g, &opts);
        Self {
            compiled,
            dim,
            se: se_inter,
        }
    }

    /// Run the shared expert on-device. `sg`/`su` are `[se, dim]` row-major (gate/up),
    /// `sd` is `[dim, se]` (down) — the same layout as [`dense_swiglu_ffn`]. Returns
    /// `[dim]`.
    pub fn forward(&mut self, x: &[f32], sg: &[f32], su: &[f32], sd: &[f32]) -> Result<Vec<f32>> {
        let (dim, se) = (self.dim, self.se);
        let tr = |v: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut o = vec![0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    o[c * rows + r] = v[r * cols + c];
                }
            }
            o
        };
        self.compiled.set_param("sg", &tr(sg, se, dim)); // [se,dim]→[dim,se]
        self.compiled.set_param("su", &tr(su, se, dim));
        self.compiled.set_param("sd", &tr(sd, dim, se)); // [dim,se]→[se,dim]
        Ok(self.compiled.run(&[("x", x)])[0].clone())
    }
}

/// LRU expert cache sized by a [`ResourceBudget`](crate::resource::ResourceBudget) —
/// wraps any [`ExpertSource`] so the hottest `cap` `(layer, expert, proj)` weight
/// tensors stay RESIDENT and are reused across tokens, while the rest stream on
/// demand from the inner source. This is what makes `budget.resident_experts(...)`
/// do real work: it bounds the expert-paging working set to the configured
/// RAM/expert budget (LRU-evicting the coldest when full). `hits`/`misses` expose
/// the reuse rate. `cap == 0` disables caching (always stream).
pub struct CachedExpertSource<S: ExpertSource> {
    inner: S,
    cache: HashMap<(usize, usize, u8), (Vec<f32>, u64)>,
    cap: usize,
    tick: u64,
    pub hits: usize,
    pub misses: usize,
}

impl<S: ExpertSource> CachedExpertSource<S> {
    /// Cache up to `cap` `(layer, expert, proj)` tensors (≈ `cap/3` experts).
    pub fn new(inner: S, cap: usize) -> Self {
        Self {
            inner,
            cache: HashMap::new(),
            cap,
            tick: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Size the cache from a resource budget: `resident_experts × 3` projs.
    pub fn from_budget(
        inner: S,
        budget: &crate::resource::ResourceBudget,
        n_experts: usize,
        bytes_per_expert: usize,
        backbone_bytes: usize,
    ) -> Self {
        let k = budget.resident_experts(n_experts, bytes_per_expert, backbone_bytes);
        Self::new(inner, k.saturating_mul(3))
    }

    /// Resident `(layer,expert,proj)` tensor count.
    pub fn resident(&self) -> usize {
        self.cache.len()
    }
}

impl<S: ExpertSource> ExpertSource for CachedExpertSource<S> {
    fn fetch(&mut self, il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
        if self.cap == 0 {
            return self.inner.fetch(il, e, proj);
        }
        let pj = match proj {
            "gate_proj" => 0u8,
            "up_proj" => 1,
            _ => 2,
        };
        let key = (il, e, pj);
        self.tick += 1;
        if let Some((v, t)) = self.cache.get_mut(&key) {
            *t = self.tick;
            self.hits += 1;
            return Ok(v.clone());
        }
        self.misses += 1;
        let v = self.inner.fetch(il, e, proj)?;
        if self.cache.len() >= self.cap {
            // Evict the least-recently-used entry (min tick).
            if let Some((&ek, _)) = self.cache.iter().min_by_key(|(_, (_, t))| *t) {
                self.cache.remove(&ek);
            }
        }
        self.cache.insert(key, (v.clone(), self.tick));
        Ok(v)
    }

    fn prewarm(&mut self, experts: &[(usize, usize)]) {
        // Only prefetch experts NOT already fully cached (all 3 projs resident) —
        // cache hits skip disk entirely, so warming them wastes I/O.
        let need: Vec<(usize, usize)> = experts
            .iter()
            .copied()
            .filter(|&(il, e)| !(0..3).all(|pj| self.cache.contains_key(&(il, e, pj))))
            .collect();
        if !need.is_empty() {
            self.inner.prewarm(&need);
        }
    }
}

/// Load a 2D weight as a `[in, out]` param (transposed from the stored `[out,
/// in]`) for direct `g.mm(x, w)`. Dequantizes if packed.
fn load_transposed_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    let (data, shape) = weights.take_transposed(key)?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Load V4 `wo_a` (`[n_groups*o_lora_rank, dpg]`) dense-F32, reshaped to the 3D
/// `[n_groups, o_lora_rank, dpg]` param [`build_v4_o_lora`] slices.
fn load_v4_wo_a(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    n_groups: usize,
    o_lora_rank: usize,
    dpg: usize,
) -> Result<NodeId> {
    let (data, _shape) = weights.take(key)?;
    let node = g.param(key, Shape::new(&[n_groups, o_lora_rank, dpg], DType::F32));
    params.insert(key.to_string(), data);
    Ok(node)
}

/// [`DeepseekV4Spec`] → the [`DeepseekSpec`] view used by [`build_deepseek_moe_ffn`]
/// (sqrtsoftplus + always-normalized + shared expert).
pub fn v4_moe_spec(spec: &DeepseekV4Spec) -> DeepseekSpec {
    DeepseekSpec {
        vocab_size: spec.vocab_size,
        hidden_size: spec.dim,
        num_hidden_layers: spec.n_layers,
        num_attention_heads: spec.n_heads,
        q_lora_rank: 0,
        absorbed_mla: false,
        kv_lora_rank: 0,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: 0,
        v_head_dim: 0,
        intermediate_size: spec.intermediate_size,
        moe_intermediate_size: spec.moe_intermediate_size,
        n_routed_experts: spec.n_routed_experts,
        num_experts_per_tok: spec.n_activated_experts,
        n_shared_experts: spec.n_shared_experts,
        first_k_dense_replace: spec.first_k_dense_replace,
        routed_scaling_factor: spec.route_scale,
        norm_topk_prob: true,
        sigmoid_gate: false,
        sqrtsoftplus_gate: true,
        swiglu_limit: spec.swiglu_limit,
        rope_theta: spec.rope_theta,
        rope_scaling: RopeScaling::None,
        attn_score_scale: None,
        rope_neox: true,
        rms_norm_eps: spec.rms_norm_eps,
    }
}

/// Build a **DeepSeek-V2/V3 / Moonlight** prefill graph: MLA attention +
/// dense-FFN (first `first_k_dense_replace` layers) + fine-grained MoE. Packed
/// affine/mxfp4 weights stay quantized (grouped-matmul experts). Untied head.
pub fn build_deepseek_prefill(
    spec: &DeepseekSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("deepseek_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let eps = spec.rms_norm_eps;
    let rope = spec.qk_rope_head_dim;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "ds.zero_beta.hidden", h);

    // Decoupled-RoPE cos/sin for the rope-dim (GptJ), θ = rope_theta, no scaling.
    let half = rope / 2;
    let mut cos_data = vec![0f32; seq * half];
    let mut sin_data = vec![0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            // YaRN-scaled inv_freq for the rope dims (head_dim = qk_rope_head_dim);
            // deepseek leaves the cos/sin table itself unscaled (mscale²→attention).
            let freq = spec.rope_scaling.inv_freq(i, rope, spec.rope_theta);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param("ds.rope.cos", Shape::new(&[seq, half], f));
    params.insert("ds.rope.cos".into(), cos_data);
    let sin_id = g.param("ds.rope.sin", Shape::new(&[seq, half], f));
    params.insert("ds.rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);
        let attn = build_deepseek_mla(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            normed,
            cos_id,
            sin_id,
            batch,
            seq,
            spec,
        )?;
        let post_attn = g.add(h_id, attn);
        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let ffn = if il < spec.first_k_dense_replace {
            // Dense SwiGLU FFN.
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let dp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.down_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed2,
                &gp,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let upn = emit_proj(
                &mut g,
                normed2,
                &up,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let gate_act = g.silu(gate);
            let glu = g.mul(gate_act, upn);
            emit_proj(&mut g, glu, &dp, Shape::new(&[batch, seq, h], f))
        } else {
            build_deepseek_moe(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                normed2,
                batch,
                seq,
                spec,
            )?
        };
        h_id = g.add(post_attn, ffn);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// GLM-4.5 (`glm4_moe`) spec: GQA + partial-RoPE attention + deepseek-style MoE.
#[derive(Debug, Clone)]
pub struct Glm4MoeSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Fraction of head_dim that gets RoPE (GLM: 0.5 → rotate first half).
    pub partial_rotary_factor: f64,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub rope_theta: f64,
    /// RoPE pairing within the rotated portion (GptJ interleaved vs NeoX half).
    pub rope_neox: bool,
    pub rms_norm_eps: f32,
}

impl Glm4MoeSpec {
    /// A [`DeepseekSpec`] view carrying just the MoE fields (MLA fields unused by
    /// [`build_deepseek_moe`]) — GLM's fine-grained MoE is identical to deepseek's.
    fn as_moe_spec(&self) -> DeepseekSpec {
        DeepseekSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            q_lora_rank: 0,
            absorbed_mla: false,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            n_routed_experts: self.n_routed_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            n_shared_experts: self.n_shared_experts,
            first_k_dense_replace: self.first_k_dense_replace,
            routed_scaling_factor: self.routed_scaling_factor,
            norm_topk_prob: self.norm_topk_prob,
            sigmoid_gate: true, // GLM-4.5 uses the sigmoid+correction-bias gate
            sqrtsoftplus_gate: false,
            swiglu_limit: 0.0,
            rope_theta: self.rope_theta,
            rope_scaling: RopeScaling::None,
            attn_score_scale: None,
            rope_neox: false,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

/// Build a **GLM-4.5 (`glm4_moe`)** prefill graph: GQA attention (Q/K/V biases +
/// partial RoPE, no qk-norm) + dense-FFN (first layer) + deepseek-style
/// fine-grained MoE. Untied head. (MTP `num_nextn_predict_layers` skipped.)
pub fn build_glm4moe_prefill(
    spec: &Glm4MoeSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("glm4_moe_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = nh / nkv.max(1);
    let eps = spec.rms_norm_eps;
    let n_rot = ((spec.partial_rotary_factor * dh as f64) as usize) & !1; // even
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "glm.zero_beta.hidden", h);
    let moe_spec = spec.as_moe_spec();

    // Partial-RoPE cos/sin. The rope op reads cos/sin with row stride
    // `head_dim/2`, so the table MUST be `[seq, head_dim/2]` even though only the
    // first `n_rot/2` columns are populated (sizing it `[seq, n_rot/2]` makes the
    // op read out-of-bounds → zeros for pos>0). θ, no scaling.
    let tabw = dh / 2;
    let half = n_rot / 2;
    let mut cos_data = vec![0f32; seq * tabw];
    let mut sin_data = vec![0f32; seq * tabw];
    for pos in 0..seq {
        for i in 0..half {
            let freq = spec.rope_theta.powf(-(2.0 * i as f64) / n_rot as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * tabw + i] = c as f32;
            sin_data[pos * tabw + i] = s as f32;
        }
    }
    let cos_id = g.param("glm.rope.cos", Shape::new(&[seq, tabw], f));
    params.insert("glm.rope.cos".into(), cos_data);
    let sin_id = g.param("glm.rope.sin", Shape::new(&[seq, tabw], f));
    params.insert("glm.rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);

        // GQA attention (Q/K/V biases; partial RoPE on the first n_rot dims).
        let kv_dim = nkv * dh;
        let q_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let k_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let v_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let mut q = emit_proj(&mut g, normed, &q_p, Shape::new(&[batch, seq, nh * dh], f));
        let mut k = emit_proj(&mut g, normed, &k_p, Shape::new(&[batch, seq, kv_dim], f));
        let mut v = emit_proj(&mut g, normed, &v_p, Shape::new(&[batch, seq, kv_dim], f));
        let qb = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_proj.bias"),
            false,
        )?;
        let kb = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_proj.bias"),
            false,
        )?;
        let vb = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.v_proj.bias"),
            false,
        )?;
        q = g.add(q, qb);
        k = g.add(k, kb);
        v = g.add(v, vb);
        let rstyle = if spec.rope_neox {
            RopeStyle::NeoX
        } else {
            RopeStyle::GptJ
        };
        let q_rope = g.rope_n_styled(q, cos_id, sin_id, dh, n_rot, rstyle);
        let k_rope = g.rope_n_styled(k, cos_id, sin_id, dh, n_rot, rstyle);
        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);
        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_rope, k_rep, v_rep],
            attn_shape,
        );
        let o_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, &o_p, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let ffn = if il < spec.first_k_dense_replace {
            let gp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let up = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let dp = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.down_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed2,
                &gp,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let upn = emit_proj(
                &mut g,
                normed2,
                &up,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let gate_act = g.silu(gate);
            let glu = g.mul(gate_act, upn);
            emit_proj(&mut g, glu, &dp, Shape::new(&[batch, seq, h], f))
        } else {
            build_deepseek_moe(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                normed2,
                batch,
                seq,
                &moe_spec,
            )?
        };
        h_id = g.add(post_attn, ffn);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// MiniMax-M2 (`minimax`) spec: GQA + FULL qk-norm + partial RoPE + `block_sparse_moe`
/// fine-grained MoE (sigmoid+correction-bias, always-normalized, no shared experts).
#[derive(Debug, Clone)]
pub struct MinimaxSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Rotated width (`rotary_dim`, e.g. 64 of head_dim 128) — partial RoPE.
    pub rotary_dim: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
}

/// Build a **MiniMax-M2 (`minimax`)** prefill graph. Standard transformer (NOT
/// Lightning/linear attention): GQA with a FULL `q_norm`/`k_norm` (RMSNorm over
/// the whole projection) + partial RoPE (NeoX), then a `block_sparse_moe`
/// fine-grained MoE — sigmoid gate + `e_score_correction_bias`, top-k,
/// always-normalized weights, NO shared experts, NO routed-scaling. Untied head.
pub fn build_minimax_prefill(
    spec: &MinimaxSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("minimax_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = nh / nkv.max(1);
    let eps = spec.rms_norm_eps;
    let n_rot = spec.rotary_dim & !1;
    let n = spec.num_local_experts;
    let top_k = spec.num_experts_per_tok;
    let inter = spec.moe_intermediate_size;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "mm.zb.hidden", h);
    let zero_beta_q = synth_zero(&mut g, &mut params, "mm.zb.q", nh * dh);
    let zero_beta_k = synth_zero(&mut g, &mut params, "mm.zb.k", nkv * dh);

    // Partial-RoPE cos/sin (table stride head_dim/2; fill first rotary_dim/2).
    let tabw = dh / 2;
    let half = n_rot / 2;
    let mut cos_data = vec![0f32; seq * tabw];
    let mut sin_data = vec![0f32; seq * tabw];
    for pos in 0..seq {
        for i in 0..half {
            let freq = spec.rope_theta.powf(-(2.0 * i as f64) / n_rot as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * tabw + i] = c as f32;
            sin_data[pos * tabw + i] = s as f32;
        }
    }
    let cos_id = g.param("mm.rope.cos", Shape::new(&[seq, tabw], f));
    params.insert("mm.rope.cos".into(), cos_data);
    let sin_id = g.param("mm.rope.sin", Shape::new(&[seq, tabw], f));
    params.insert("mm.rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);

        // GQA attention — FULL qk-norm over the whole projection, then partial RoPE.
        let kv_dim = nkv * dh;
        let q_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let k_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let v_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let q = emit_proj(&mut g, normed, &q_p, Shape::new(&[batch, seq, nh * dh], f));
        let k = emit_proj(&mut g, normed, &k_p, Shape::new(&[batch, seq, kv_dim], f));
        let v = emit_proj(&mut g, normed, &v_p, Shape::new(&[batch, seq, kv_dim], f));
        let qn_g = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_norm.weight"),
            0.0,
        )?;
        let kn_g = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_norm.weight"),
            0.0,
        )?;
        let q = g.rms_norm(q, qn_g, zero_beta_q, eps);
        let k = g.rms_norm(k, kn_g, zero_beta_k, eps);
        let q_rope = g.rope_n_styled(q, cos_id, sin_id, dh, n_rot, RopeStyle::NeoX);
        let k_rope = g.rope_n_styled(k, cos_id, sin_id, dh, n_rot, RopeStyle::NeoX);
        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);
        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_rope, k_rep, v_rep],
            attn_shape,
        );
        let o_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, &o_p, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        // block_sparse_moe: sigmoid+bias gate → top-k → normalize → SwiGLU experts.
        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let rows = batch * seq;
        let h2d = g.reshape_(normed2, vec![rows as i64, h as i64]);
        let router_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.block_sparse_moe.gate.weight"),
        )?;
        let logits = emit_proj(&mut g, h2d, &router_p, Shape::new(&[rows, n], f));
        let sig = g.sigmoid(logits);
        let (eb, _) = weights.take(&format!("{lp}.block_sparse_moe.e_score_correction_bias"))?;
        let eb_node = g.param(format!("{lp}.mm.ebias"), Shape::new(&[n], f));
        params.insert(format!("{lp}.mm.ebias"), eb);
        let route = g.add(sig, eb_node);
        let top_idx = g.add_node(
            Op::TopK { k: top_k },
            vec![route],
            Shape::new(&[rows, top_k], DType::F32),
        );
        let top_w = g.add_node(
            Op::GatherElements { axis: 1 },
            vec![sig, top_idx],
            Shape::new(&[rows, top_k], f),
        );
        let denom = g.sum(top_w, vec![1], true);
        let top_w = g.div(top_w, denom); // always normalized
        let (gc, gs, gb, scheme, _) = load_stacked_group_experts_mlx(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            "block_sparse_moe.switch_mlp",
            "gate_proj",
            n,
            inter,
        )?;
        let (uc, us, ub, _, _) = load_stacked_group_experts_mlx(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            "block_sparse_moe.switch_mlp",
            "up_proj",
            n,
            inter,
        )?;
        let (dc, ds, db, _, _) = load_stacked_group_experts_mlx(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            "block_sparse_moe.switch_mlp",
            "down_proj",
            n,
            h,
        )?;
        let mut acc: Option<NodeId> = None;
        for ki in 0..top_k {
            let e_col = g.narrow_(top_idx, 1, ki, 1);
            let e_idx = g.reshape_(e_col, vec![rows as i64]);
            let w_col = g.narrow_(top_w, 1, ki, 1);
            let gate = g.add_node(
                Op::DequantGroupedMatMulMlx { scheme },
                vec![h2d, gc, gs, gb, e_idx],
                Shape::new(&[rows, inter], f),
            );
            let up = g.add_node(
                Op::DequantGroupedMatMulMlx { scheme },
                vec![h2d, uc, us, ub, e_idx],
                Shape::new(&[rows, inter], f),
            );
            let ga = g.silu(gate);
            let glu = g.mul(ga, up);
            let down = g.add_node(
                Op::DequantGroupedMatMulMlx { scheme },
                vec![glu, dc, ds, db, e_idx],
                Shape::new(&[rows, h], f),
            );
            let weighted = g.mul(down, w_col);
            acc = Some(match acc {
                None => weighted,
                Some(a) => g.add(a, weighted),
            });
        }
        let moe = g.reshape_(
            acc.ok_or_else(|| anyhow!("minimax: top_k=0"))?,
            vec![batch as i64, seq as i64, h as i64],
        );
        h_id = g.add(post_attn, moe);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// Nemotron-H (`nemotron_h`) spec — NVIDIA's hybrid **Mamba-2 / attention / MoE
/// (+ dense-MLP)** decoder. Per-layer block type comes from
/// `hybrid_override_pattern` (one char per layer: `M`=Mamba-2 mixer, `*`=NoPE
/// GQA attention, `E`=fine-grained MoE, `-`=ReLU² MLP). Untied LM head; attention
/// carries **no positional encoding** (NoPE — see mlx-lm `nemotron_h.py`).
#[derive(Debug, Clone)]
pub struct NemotronHSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// One block-type char per layer (`M` / `*` / `E` / `-`).
    pub hybrid_pattern: Vec<char>,
    // ── attention (`*`) ──
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    // ── Mamba-2 mixer (`M`) ──
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    pub ssm_state_size: usize,
    pub conv_kernel: usize,
    pub n_groups: usize,
    pub use_conv_bias: bool,
    pub time_step_limit: (f32, f32),
    // ── dense MLP (`-`) ──
    pub intermediate_size: usize,
    // ── MoE (`E`) ──
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub rms_norm_eps: f32,
}

/// Load a `[heads]` decay vector transformed **host-side** to `a = -exp(A_log)`
/// (the value `Op::Mamba2` wants: it computes `dA = exp(dt·a)`, matching mlx-lm's
/// `A = -exp(A_log)`).
fn load_neg_exp(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    let (mut data, shape) = weights.take(key)?;
    for v in data.iter_mut() {
        *v = -(v.exp());
    }
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Numerically-stable `softplus(x) = relu(x) + log(1 + exp(-|x|))` composed from
/// primitive ops (no dedicated builder method). `name` seeds the `[1]` constant.
fn softplus_stable(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    name: &str,
) -> NodeId {
    let ax = g.abs(x);
    let nax = g.neg(ax);
    let enax = g.exp(nax);
    let one = const1(g, params, &format!("{name}.one"), 1.0);
    let e1 = g.add(enax, one);
    let le1 = g.log(e1);
    let rx = g.relu(x);
    g.add(rx, le1)
}

/// Nemotron-H **Mamba-2 (SSD) mixer**. `in_proj(x)` → split `gate | conv_input |
/// dt` → depthwise causal conv1d (+bias) + SiLU → split `x_ssm | B | C` →
/// group→head repeat of B/C → `Op::Mamba2` (with `a=-exp(A_log)`,
/// `dt=clamp(softplus(dt+dt_bias))`) → `+ D·x_ssm` skip → gated group-RMSNorm
/// (`silu(gate)·y`, RMS over `hidden/n_groups`, ×weight) → `out_proj`. Mirrors
/// mlx-lm `NemotronHMamba2Mixer`; reuses the validated `Op::Mamba2` scan.
///
/// `pub` so an isolation test can validate the mixer numerically against an
/// inline reference without needing a full (30B+) checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn build_nemotron_h_mamba2(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    mp: &str,
    x: NodeId,
    seq: usize,
    hidden: usize,
    spec: &NemotronHSpec,
) -> Result<NodeId> {
    let f = DType::F32;
    let batch = 1;
    let nh = spec.mamba_num_heads;
    let dh = spec.mamba_head_dim;
    let st = spec.ssm_state_size;
    let ng = spec.n_groups;
    let k = spec.conv_kernel;
    let eps = spec.rms_norm_eps;
    let inter = nh * dh; // mamba intermediate_size
    let conv_dim = inter + 2 * ng * st;
    let in_out = inter + conv_dim + nh; // gate | conv_input | dt
    let hpg = nh / ng.max(1); // heads per group

    let in_p = load_proj(g, params, packed, weights, &format!("{mp}.in_proj.weight"))?;
    let proj = emit_proj(g, x, &in_p, Shape::new(&[batch, seq, in_out], f));
    let gate = g.narrow_(proj, 2, 0, inter);
    let conv_input = g.narrow_(proj, 2, inter, conv_dim);
    let dt_raw = g.narrow_(proj, 2, inter + conv_dim, nh);

    // Depthwise causal conv1d (+ optional bias) then SiLU.
    let mut conv = depthwise_causal_conv_lfm(
        g,
        params,
        weights,
        &format!("{mp}.conv1d.weight"),
        conv_input,
        batch,
        seq,
        conv_dim,
        k,
    )?;
    if spec.use_conv_bias {
        let cb = load_p(g, params, weights, &format!("{mp}.conv1d.bias"), false)?;
        conv = g.add(conv, cb);
    }
    let conv = g.silu(conv);

    let x_ssm = g.narrow_(conv, 2, 0, inter);
    let b_flat = g.narrow_(conv, 2, inter, ng * st);
    let c_flat = g.narrow_(conv, 2, inter + ng * st, ng * st);

    // SSM op layout: x `[b,s,heads,head_dim]`; B/C `[b,s,heads,state]` (repeat
    // each group across its `heads_per_group` consecutive heads).
    let x4 = g.reshape_(x_ssm, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
    let b_rep = repeat_kv(g, b_flat, ng, st, hpg);
    let c_rep = repeat_kv(g, c_flat, ng, st, hpg);
    let b4 = g.reshape_(b_rep, vec![batch as i64, seq as i64, nh as i64, st as i64]);
    let c4 = g.reshape_(c_rep, vec![batch as i64, seq as i64, nh as i64, st as i64]);

    // dt = clamp(softplus(dt_raw + dt_bias), lo, hi); a = -exp(A_log).
    let dt_bias = load_p(g, params, weights, &format!("{mp}.dt_bias"), false)?;
    let dt = g.add(dt_raw, dt_bias);
    let dt = softplus_stable(g, params, dt, &format!("{mp}.sp"));
    let dt = g.clamp_(dt, spec.time_step_limit.0, spec.time_step_limit.1);
    let a = load_neg_exp(g, params, weights, &format!("{mp}.A_log"))?;

    let y = g.mamba2(
        x4,
        dt,
        a,
        b4,
        c4,
        dh,
        st,
        Shape::new(&[batch, seq, nh, dh], f),
    );
    // + D·x_ssm skip (D per head, broadcast over head_dim).
    let d = load_p(g, params, weights, &format!("{mp}.D"), false)?;
    let d4 = g.reshape_(d, vec![1, 1, nh as i64, 1]);
    let dx = g.mul(x4, d4);
    let y = g.add(y, dx);
    let y = g.reshape_(y, vec![batch as i64, seq as i64, inter as i64]);

    // Gated group-RMSNorm: silu(gate)·y → RMS per group (size inter/n_groups,
    // no per-group gain) → ×norm.weight.
    let sg = g.silu(gate);
    let gated = g.mul(sg, y);
    let gsize = inter / ng.max(1);
    let ones = synth_const(
        g,
        params,
        &format!("{mp}.gn.ones"),
        vec![1f32; gsize],
        &[gsize],
    );
    let zeros = synth_zero(g, params, &format!("{mp}.gn.zeros"), gsize);
    let flat = g.reshape_(gated, vec![(batch * seq * ng) as i64, gsize as i64]);
    let normed = g.rms_norm(flat, ones, zeros, eps);
    let nback = g.reshape_(normed, vec![batch as i64, seq as i64, inter as i64]);
    let norm_w = load_norm(g, params, weights, &format!("{mp}.norm.weight"), 0.0)?;
    let y2 = g.mul(nback, norm_w);

    let out_p = load_proj(g, params, packed, weights, &format!("{mp}.out_proj.weight"))?;
    Ok(emit_proj(
        g,
        y2,
        &out_p,
        Shape::new(&[batch, seq, hidden], f),
    ))
}

/// Nemotron-H **NoPE GQA attention** (`*` block). Q/K/V/O projections (optional
/// bias), GQA KV-repeat, causal `Op::Attention` with default `head_dim^-0.5`
/// scale — and NO rotary/positional encoding (mlx-lm applies none).
#[allow(clippy::too_many_arguments)]
fn build_nemotron_h_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    mp: &str,
    x: NodeId,
    seq: usize,
    hidden: usize,
    spec: &NemotronHSpec,
) -> Result<NodeId> {
    let f = DType::F32;
    let batch = 1;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = nh / nkv.max(1);
    let kv_dim = nkv * dh;
    let q_p = load_proj(g, params, packed, weights, &format!("{mp}.q_proj.weight"))?;
    let k_p = load_proj(g, params, packed, weights, &format!("{mp}.k_proj.weight"))?;
    let v_p = load_proj(g, params, packed, weights, &format!("{mp}.v_proj.weight"))?;
    let mut q = emit_proj(g, x, &q_p, Shape::new(&[batch, seq, nh * dh], f));
    let mut k = emit_proj(g, x, &k_p, Shape::new(&[batch, seq, kv_dim], f));
    let mut v = emit_proj(g, x, &v_p, Shape::new(&[batch, seq, kv_dim], f));
    if spec.attention_bias {
        let qb = load_p(g, params, weights, &format!("{mp}.q_proj.bias"), false)?;
        let kb = load_p(g, params, weights, &format!("{mp}.k_proj.bias"), false)?;
        let vb = load_p(g, params, weights, &format!("{mp}.v_proj.bias"), false)?;
        q = g.add(q, qb);
        k = g.add(k, kb);
        v = g.add(v, vb);
    }
    let k_rep = repeat_kv(g, k, nkv, dh, group);
    let v_rep = repeat_kv(g, v, nkv, dh, group);
    let attn_shape = shape::attention_shape(g.shape(q));
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k_rep, v_rep],
        attn_shape,
    );
    let o_p = load_proj(g, params, packed, weights, &format!("{mp}.o_proj.weight"))?;
    Ok(emit_proj(
        g,
        attn,
        &o_p,
        Shape::new(&[batch, seq, hidden], f),
    ))
}

/// ReLU² feed-forward: `down_proj(relu(up_proj(x))²)` — Nemotron-H's `-` block
/// and its shared-expert MLP (mlx-lm `NemotronHMLP`, `nn.relu2`).
#[allow(clippy::too_many_arguments)]
fn build_nemotron_h_mlp(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    mp: &str,
    x: NodeId,
    seq: usize,
    hidden: usize,
    inter: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let up_p = load_proj(g, params, packed, weights, &format!("{mp}.up_proj.weight"))?;
    let u = emit_proj(g, x, &up_p, Shape::new(&[1, seq, inter], f));
    let r = g.relu(u);
    let r2 = g.mul(r, r);
    let down_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{mp}.down_proj.weight"),
    )?;
    Ok(emit_proj(g, r2, &down_p, Shape::new(&[1, seq, hidden], f)))
}

/// Nemotron-H **fine-grained MoE** (`E` block). deepseek-style aux-loss-free
/// routing: `sigmoid(gate) + e_score_correction_bias` selects top-k (bias used
/// for SELECTION only; weights gather the RAW sigmoid), optional top-k
/// normalization, ×`routed_scaling_factor`; experts are **ReLU²** MLPs
/// (`switch_mlp.fc1`/`fc2`), plus a ReLU² shared expert on the block input.
/// (Group-limited routing `n_group>1` is unimplemented — in-scope configs use
/// `n_group == topk_group == 1`; asserted below.)
#[allow(clippy::too_many_arguments)]
fn build_nemotron_h_moe(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    mp: &str,
    x: NodeId,
    seq: usize,
    hidden: usize,
    spec: &NemotronHSpec,
) -> Result<NodeId> {
    let f = DType::F32;
    let batch = 1;
    let rows = batch * seq;
    let n = spec.n_routed_experts;
    let top_k = spec.num_experts_per_tok;
    let inter = spec.moe_intermediate_size;
    anyhow::ensure!(
        spec.n_group <= 1 && spec.topk_group <= 1,
        "nemotron_h MoE group-limited routing (n_group={}, topk_group={}) not wired",
        spec.n_group,
        spec.topk_group
    );

    let h2d = g.reshape_(x, vec![rows as i64, hidden as i64]);
    let gate_p = load_proj(g, params, packed, weights, &format!("{mp}.gate.weight"))?;
    let logits = emit_proj(g, h2d, &gate_p, Shape::new(&[rows, n], f));
    let sig = g.sigmoid(logits);
    let (eb, _) = weights.take(&format!("{mp}.gate.e_score_correction_bias"))?;
    let eb_node = g.param(format!("{mp}.ebias"), Shape::new(&[n], f));
    params.insert(format!("{mp}.ebias"), eb);
    let route = g.add(sig, eb_node);
    let top_idx = g.add_node(
        Op::TopK { k: top_k },
        vec![route],
        Shape::new(&[rows, top_k], DType::F32),
    );
    let mut top_w = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![sig, top_idx],
        Shape::new(&[rows, top_k], f),
    );
    if spec.norm_topk_prob && top_k > 1 {
        let denom = g.sum(top_w, vec![1], true);
        top_w = g.div(top_w, denom);
    }
    let rs = const1(
        g,
        params,
        &format!("{mp}.rscale"),
        spec.routed_scaling_factor,
    );
    top_w = g.mul(top_w, rs);

    // Stacked ReLU² experts: fc1 (up, `[n, inter, hidden]`), fc2 (down, `[n, hidden, inter]`).
    let (f1c, f1s, f1b, scheme, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        mp,
        "switch_mlp",
        "fc1",
        n,
        inter,
    )?;
    let (f2c, f2s, f2b, _, _) = load_stacked_group_experts_mlx(
        g,
        params,
        packed,
        weights,
        mp,
        "switch_mlp",
        "fc2",
        n,
        hidden,
    )?;
    let mut acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let e_col = g.narrow_(top_idx, 1, ki, 1);
        let e_idx = g.reshape_(e_col, vec![rows as i64]);
        let w_col = g.narrow_(top_w, 1, ki, 1);
        let up = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![h2d, f1c, f1s, f1b, e_idx],
            Shape::new(&[rows, inter], f),
        );
        let r = g.relu(up);
        let r2 = g.mul(r, r);
        let down = g.add_node(
            Op::DequantGroupedMatMulMlx { scheme },
            vec![r2, f2c, f2s, f2b, e_idx],
            Shape::new(&[rows, hidden], f),
        );
        let weighted = g.mul(down, w_col);
        acc = Some(match acc {
            None => weighted,
            Some(p) => g.add(p, weighted),
        });
    }
    let mut y = g.reshape_(
        acc.ok_or_else(|| anyhow!("nemotron_h: MoE top_k=0"))?,
        vec![batch as i64, seq as i64, hidden as i64],
    );

    // Shared expert (ReLU² MLP) on the block input, added unscaled.
    if spec.n_shared_experts > 0 {
        let up_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{mp}.shared_experts.up_proj.weight"),
        )?;
        let su = emit_proj(
            g,
            x,
            &up_p,
            Shape::new(&[batch, seq, spec.moe_shared_expert_intermediate_size], f),
        );
        let sr = g.relu(su);
        let sr2 = g.mul(sr, sr);
        let down_p = load_proj(
            g,
            params,
            packed,
            weights,
            &format!("{mp}.shared_experts.down_proj.weight"),
        )?;
        let sd = emit_proj(g, sr2, &down_p, Shape::new(&[batch, seq, hidden], f));
        y = g.add(y, sd);
    }
    Ok(y)
}

/// Build a **Nemotron-H (`nemotron_h`)** prefill graph — NVIDIA's hybrid
/// Mamba-2 / NoPE-attention / ReLU²-MoE (+ dense MLP) decoder. Per-layer block
/// type follows `spec.hybrid_pattern`; block = `x + mixer(RMSNorm(x))`. Reuses
/// the validated `Op::Mamba2` SSD scan + the deepseek-style MoE router. Untied
/// LM head. Returns logits `[seq, vocab]`.
pub fn build_nemotron_h_prefill(
    spec: &NemotronHSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("nemotron_h_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let eps = spec.rms_norm_eps;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "nh.zb.hidden", h);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "backbone.embeddings.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for (il, blk) in spec.hybrid_pattern.iter().enumerate() {
        let lp = format!("backbone.layers.{il}");
        let mp = format!("{lp}.mixer");
        let norm = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, norm, zero_beta_hidden, eps);
        let mixed = match blk {
            'M' => build_nemotron_h_mamba2(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                normed,
                seq,
                h,
                spec,
            )?,
            '*' => build_nemotron_h_attention(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                normed,
                seq,
                h,
                spec,
            )?,
            'E' => build_nemotron_h_moe(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                normed,
                seq,
                h,
                spec,
            )?,
            '-' => build_nemotron_h_mlp(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                normed,
                seq,
                h,
                spec.intermediate_size,
            )?,
            other => return Err(anyhow!("nemotron_h: unknown block type '{other}'")),
        };
        h_id = g.add(h_id, mixed);
    }

    let fnorm = load_norm(&mut g, &mut params, weights, "backbone.norm_f.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, fnorm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// Hunyuan-V3 (`hy_v3`, `HYV3ForCausalLM`) spec — Tencent's GQA + per-head
/// qk-norm + full-RoPE + deepseek-style fine-grained MoE decoder. Standard
/// transformer family (NOT MLA/SSM). Layers `< first_k_dense_replace` use a
/// dense SwiGLU MLP; the rest use the MoE (sigmoid router + `expert_bias`
/// aux-loss-free selection, top-k, `route_norm` normalization, ×router_scaling,
/// SwiGLU experts + one SwiGLU shared expert). Untied head.
#[derive(Debug, Clone)]
pub struct HyV3Spec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Per-head RMSNorm on Q/K (over `head_dim`), applied AFTER RoPE (Hunyuan
    /// family convention — see transformers `hunyuan_v1_moe`).
    pub qk_norm: bool,
    /// Dense-MLP width for the first `first_k_dense_replace` layers.
    pub intermediate_size: usize,
    /// Per-expert SwiGLU width (`moe_intermediate_size` == `expert_hidden_dim`).
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Number of shared experts (shared SwiGLU width = `moe_intermediate_size ×
    /// num_shared_experts`); 0 disables the shared path.
    pub num_shared_experts: usize,
    pub first_k_dense_replace: usize,
    /// Normalize the top-k routing weights (`route_norm`).
    pub route_norm: bool,
    pub router_scaling_factor: f32,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
}

/// Build a **Hunyuan-V3 (`hy_v3`)** prefill graph. GQA (`head_dim` explicit,
/// not `hidden/heads`) with per-head qk-norm applied AFTER full RoPE, then a
/// deepseek-style MoE (or a dense SwiGLU MLP for the first
/// `first_k_dense_replace` layers). Reuses the validated MoE router
/// (sigmoid + `expert_bias` select, raw-sigmoid weights, `route_norm`,
/// ×router_scaling) + SwiGLU shared expert. Untied LM head. Returns logits
/// `[seq, vocab]`.
///
/// hy_v3 is newer than both mlx-lm and transformers 5.3.0 (no public reference);
/// this is correct-by-construction from the config + verified tensor shapes +
/// the Hunyuan-family attention convention. Giants (80L / 192 experts) → not
/// run-validated.
pub fn build_hy_v3_prefill(
    spec: &HyV3Spec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("hy_v3_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = nh / nkv.max(1);
    let eps = spec.rms_norm_eps;
    let n = spec.num_experts;
    let top_k = spec.num_experts_per_tok;
    let inter = spec.moe_intermediate_size;
    let shared_w = spec.moe_intermediate_size * spec.num_shared_experts.max(1);
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "hy.zb.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "hy.zb.head_dim", dh);

    // Full-RoPE tables ([seq, head_dim/2]).
    let half = dh / 2;
    let mut cos_data = vec![0f32; seq * half];
    let mut sin_data = vec![0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let freq = spec.rope_theta.powf(-(2.0 * i as f64) / dh as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param("hy.rope.cos", Shape::new(&[seq, half], f));
    params.insert("hy.rope.cos".into(), cos_data);
    let sin_id = g.param("hy.rope.sin", Shape::new(&[seq, half], f));
    params.insert("hy.rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);

        // ── GQA attention: proj → RoPE → per-head qk-norm → attn → o_proj ──
        let kv_dim = nkv * dh;
        let q_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let k_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let v_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let q = emit_proj(&mut g, normed, &q_p, Shape::new(&[batch, seq, nh * dh], f));
        let k = emit_proj(&mut g, normed, &k_p, Shape::new(&[batch, seq, kv_dim], f));
        let v = emit_proj(&mut g, normed, &v_p, Shape::new(&[batch, seq, kv_dim], f));
        let mut q = rope_heads(&mut g, q, cos_id, sin_id, batch, seq, nh, dh, true);
        let mut k = rope_heads(&mut g, k, cos_id, sin_id, batch, seq, nkv, dh, true);
        if spec.qk_norm {
            let qn_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.q_norm.weight"),
                0.0,
            )?;
            let kn_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.k_norm.weight"),
                0.0,
            )?;
            q = per_head_rms(&mut g, q, qn_g, zero_beta_headdim, batch, seq, nh, dh, eps);
            k = per_head_rms(&mut g, k, kn_g, zero_beta_headdim, batch, seq, nkv, dh, eps);
        }
        let k_rep = repeat_kv(&mut g, k, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);
        let attn_shape = shape::attention_shape(g.shape(q));
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k_rep, v_rep],
            attn_shape,
        );
        let o_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, &o_p, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        // ── MLP: dense SwiGLU for layer < first_k_dense_replace, else MoE ──
        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let mlp_out = if il < spec.first_k_dense_replace {
            let gate_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let up_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed2,
                &gate_p,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let up = emit_proj(
                &mut g,
                normed2,
                &up_p,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let sg = g.silu(gate);
            let glu = g.mul(sg, up);
            let down_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.down_proj.weight"),
            )?;
            emit_proj(&mut g, glu, &down_p, Shape::new(&[batch, seq, h], f))
        } else {
            let mp = format!("{lp}.mlp");
            let rows = batch * seq;
            let h2d = g.reshape_(normed2, vec![rows as i64, h as i64]);
            let router_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{mp}.router.gate.weight"),
            )?;
            let logits = emit_proj(&mut g, h2d, &router_p, Shape::new(&[rows, n], f));
            let sig = g.sigmoid(logits);
            let (eb, _) = weights.take(&format!("{mp}.router.expert_bias"))?;
            let eb_node = g.param(format!("{mp}.ebias"), Shape::new(&[n], f));
            params.insert(format!("{mp}.ebias"), eb);
            let route = g.add(sig, eb_node);
            let top_idx = g.add_node(
                Op::TopK { k: top_k },
                vec![route],
                Shape::new(&[rows, top_k], DType::F32),
            );
            let mut top_w = g.add_node(
                Op::GatherElements { axis: 1 },
                vec![sig, top_idx],
                Shape::new(&[rows, top_k], f),
            );
            if spec.route_norm && top_k > 1 {
                let denom = g.sum(top_w, vec![1], true);
                top_w = g.div(top_w, denom);
            }
            let rs = const1(
                &mut g,
                &mut params,
                &format!("{mp}.rscale"),
                spec.router_scaling_factor,
            );
            top_w = g.mul(top_w, rs);
            let (gc, gs, gb, scheme, _) = load_stacked_group_experts_mlx(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                "switch_mlp",
                "gate_proj",
                n,
                inter,
            )?;
            let (uc, us, ub, _, _) = load_stacked_group_experts_mlx(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                "switch_mlp",
                "up_proj",
                n,
                inter,
            )?;
            let (dc, ds, db, _, _) = load_stacked_group_experts_mlx(
                &mut g,
                &mut params,
                packed,
                weights,
                &mp,
                "switch_mlp",
                "down_proj",
                n,
                h,
            )?;
            let mut acc: Option<NodeId> = None;
            for ki in 0..top_k {
                let e_col = g.narrow_(top_idx, 1, ki, 1);
                let e_idx = g.reshape_(e_col, vec![rows as i64]);
                let w_col = g.narrow_(top_w, 1, ki, 1);
                let gate = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![h2d, gc, gs, gb, e_idx],
                    Shape::new(&[rows, inter], f),
                );
                let up = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![h2d, uc, us, ub, e_idx],
                    Shape::new(&[rows, inter], f),
                );
                let sg = g.silu(gate);
                let glu = g.mul(sg, up);
                let down = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![glu, dc, ds, db, e_idx],
                    Shape::new(&[rows, h], f),
                );
                let weighted = g.mul(down, w_col);
                acc = Some(match acc {
                    None => weighted,
                    Some(p) => g.add(p, weighted),
                });
            }
            let mut y = g.reshape_(
                acc.ok_or_else(|| anyhow!("hy_v3: MoE top_k=0"))?,
                vec![batch as i64, seq as i64, h as i64],
            );
            // SwiGLU shared expert on the block input, added unscaled.
            if spec.num_shared_experts > 0 {
                let sg_p = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{mp}.shared_mlp.gate_proj.weight"),
                )?;
                let su_p = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{mp}.shared_mlp.up_proj.weight"),
                )?;
                let sgate = emit_proj(
                    &mut g,
                    normed2,
                    &sg_p,
                    Shape::new(&[batch, seq, shared_w], f),
                );
                let sup = emit_proj(
                    &mut g,
                    normed2,
                    &su_p,
                    Shape::new(&[batch, seq, shared_w], f),
                );
                let ssg = g.silu(sgate);
                let sglu = g.mul(ssg, sup);
                let sd_p = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{mp}.shared_mlp.down_proj.weight"),
                )?;
                let sdown = emit_proj(&mut g, sglu, &sd_p, Shape::new(&[batch, seq, h], f));
                y = g.add(y, sdown);
            }
            y
        };
        h_id = g.add(post_attn, mlp_out);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// Kimi-Linear (`kimi_linear`, `KimiLinearForCausalLM`) spec — Moonshot's hybrid
/// **KDA (Kimi Delta Attention, fine-grained-gated delta-net linear attn) +
/// NoPE-MLA** decoder with deepseek-style MoE. `kda_layers` (1-indexed) select
/// KDA layers; the rest are MLA (`mla_use_nope`). Layers `< first_k_dense_replace`
/// use a dense SwiGLU MLP, the rest the MoE. Untied head.
#[derive(Debug, Clone)]
pub struct KimiLinearSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// 1-indexed layer numbers that use KDA linear attention (`kda_layers`);
    /// layers not listed use NoPE-MLA.
    pub kda_layers: Vec<usize>,
    // ── KDA (`linear_attn_config`) ──
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub kda_conv_kernel: usize,
    // ── MLA (reuses the validated DeepSeek MLA, NoPE) ──
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    // ── MoE / dense ──
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    pub moe_renormalize: bool,
    pub rms_norm_eps: f32,
}

/// Build a **Kimi Delta Attention (KDA)** block — the fine-grained-gated
/// delta-net linear attention that is Kimi-Linear's novel primitive (not
/// expressible via `Op::GatedDeltaNet`, whose gate is per-head scalar; KDA's is
/// per-key-dim). Mirrors mlx-lm `KimiDeltaAttention` + `gated_delta_update`:
///
/// q/k/v_proj → depthwise-causal short-conv(+SiLU) → per-head RMS-norm & scale
/// (`q·=scale²`, `k·=scale`, eps 1e-6) → gates `a=f_b(f_a(x))`, `b=b_proj(x)`,
/// output-gate `g=g_b(g_a(x))`; `beta=σ(b)`, decay `G=exp(-exp(A_log)·softplus(
/// a+dt_bias))` (per-key-dim) → delta-rule recurrence (state `[Hv,Dv,Dk]`,
/// unrolled over seq since no op emits per-step outputs) → per-head o-norm ×
/// `σ(output-gate)` → o_proj. `pub` for isolation validation.
#[allow(clippy::too_many_arguments)]
pub fn build_kimi_kda(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    sa: &str,
    x: NodeId,
    seq: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    conv_k: usize,
    rms_eps: f32,
) -> Result<NodeId> {
    let f = DType::F32;
    let batch = 1;
    let hv = num_heads;
    let dk = head_dim; // Dk == Dv for KDA
    let p = hv * dk; // projection_dim
    let scale = (dk as f32).powf(-0.5);
    let ones_dk = synth_const(
        g,
        params,
        &format!("{sa}.kda.ones_dk"),
        vec![1f32; dk],
        &[dk],
    );
    let zeros_dk = synth_zero(g, params, &format!("{sa}.kda.zeros_dk"), dk);

    // q/k/v projections → short conv (depthwise causal, no bias) → SiLU.
    let shortconv = |g: &mut Graph,
                     params: &mut HashMap<String, Vec<f32>>,
                     packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
                     weights: &mut dyn WeightLoader,
                     proj: &str,
                     conv: &str|
     -> Result<NodeId> {
        let wp = load_proj(g, params, packed, weights, &format!("{sa}.{proj}.weight"))?;
        let projected = emit_proj(g, x, &wp, Shape::new(&[batch, seq, p], f));
        let c = depthwise_causal_conv_lfm(
            g,
            params,
            weights,
            &format!("{sa}.{conv}.conv.weight"),
            projected,
            batch,
            seq,
            p,
            conv_k,
        )?;
        Ok(g.silu(c))
    };
    let q = shortconv(g, params, packed, weights, "q_proj", "q_conv")?;
    let k = shortconv(g, params, packed, weights, "k_proj", "k_conv")?;
    let v = shortconv(g, params, packed, weights, "v_proj", "v_conv")?;

    // Per-head RMS-norm (eps 1e-6, no gain) then scale: q·=scale², k·=scale.
    let qn = per_head_rms(g, q, ones_dk, zeros_dk, batch, seq, hv, dk, 1e-6);
    let kn = per_head_rms(g, k, ones_dk, zeros_dk, batch, seq, hv, dk, 1e-6);
    let q_s = const1(g, params, &format!("{sa}.kda.qscale"), scale * scale);
    let k_s = const1(g, params, &format!("{sa}.kda.kscale"), scale);
    let q = g.mul(qn, q_s);
    let k = g.mul(kn, k_s);
    let q4 = g.reshape_(q, vec![batch as i64, seq as i64, hv as i64, dk as i64]);
    let k4 = g.reshape_(k, vec![batch as i64, seq as i64, hv as i64, dk as i64]);
    let v4 = g.reshape_(v, vec![batch as i64, seq as i64, hv as i64, dk as i64]);

    // Gates: a = f_b(f_a(x)) [.,Hv,Dk]; b = b_proj(x) [.,Hv]; out-gate = g_b(g_a(x)) [.,P].
    let fa = load_proj(g, params, packed, weights, &format!("{sa}.f_a_proj.weight"))?;
    let a_pre = emit_proj(g, x, &fa, Shape::new(&[batch, seq, dk], f));
    let fb = load_proj(g, params, packed, weights, &format!("{sa}.f_b_proj.weight"))?;
    let a_logits = emit_proj(g, a_pre, &fb, Shape::new(&[batch, seq, p], f));
    let bp = load_proj(g, params, packed, weights, &format!("{sa}.b_proj.weight"))?;
    let b_logits = emit_proj(g, x, &bp, Shape::new(&[batch, seq, hv], f));
    let beta = g.sigmoid(b_logits); // [1,seq,Hv]
    let ga = load_proj(g, params, packed, weights, &format!("{sa}.g_a_proj.weight"))?;
    let g_pre = emit_proj(g, x, &ga, Shape::new(&[batch, seq, dk], f));
    let gb = load_proj(g, params, packed, weights, &format!("{sa}.g_b_proj.weight"))?;
    let out_gate = emit_proj(g, g_pre, &gb, Shape::new(&[batch, seq, p], f));

    // decay G = exp(-exp(A_log) · softplus(a + dt_bias)) — per-key-dim.
    let dt_bias = load_p(g, params, weights, &format!("{sa}.dt_bias"), false)?; // [P]
    let dt_bias4 = g.reshape_(dt_bias, vec![1, 1, hv as i64, dk as i64]);
    let a4 = g.reshape_(
        a_logits,
        vec![batch as i64, seq as i64, hv as i64, dk as i64],
    );
    let a_db = g.add(a4, dt_bias4);
    let sp = softplus_stable(g, params, a_db, &format!("{sa}.kda.sp"));
    let neg_exp_a = load_neg_exp(g, params, weights, &format!("{sa}.A_log"))?; // [Hv] = -exp(A_log)
    let neg_exp_a4 = g.reshape_(neg_exp_a, vec![1, 1, hv as i64, 1]);
    let g_arg = g.mul(sp, neg_exp_a4);
    let g_gate = g.exp(g_arg); // [1,seq,Hv,Dk]

    // Delta-rule recurrence, unrolled over seq. State S [Hv, Dv, Dk] (Dv==Dk).
    let dv = dk;
    let s0 = synth_zero(g, params, &format!("{sa}.kda.S0"), hv * dv * dk);
    let mut state = g.reshape_(s0, vec![hv as i64, dv as i64, dk as i64]);
    let mut ys: Vec<NodeId> = Vec::with_capacity(seq);
    for t in 0..seq {
        let slice3 = |g: &mut Graph, x: NodeId| -> NodeId {
            // [1,seq,Hv,Dk] → token t → [Hv,1,Dk]
            let n = g.narrow_(x, 1, t, 1);
            g.reshape_(n, vec![hv as i64, 1, dk as i64])
        };
        let q_t = slice3(g, q4); // [Hv,1,Dk]
        let k_t = slice3(g, k4);
        let g_t = slice3(g, g_gate);
        let v_n = g.narrow_(v4, 1, t, 1);
        let v_t = g.reshape_(v_n, vec![hv as i64, dv as i64, 1]); // [Hv,Dv,1]
        let beta_n = g.narrow_(beta, 1, t, 1);
        let beta_t = g.reshape_(beta_n, vec![hv as i64, 1, 1]); // [Hv,1,1]

        let s_decay = g.mul(state, g_t); // [Hv,Dv,Dk]·[Hv,1,Dk]
        let skt = g.mul(s_decay, k_t); // [Hv,Dv,Dk]
        let kv_mem = g.sum(skt, vec![2], true); // [Hv,Dv,1]
        let diff = g.sub(v_t, kv_mem); // [Hv,Dv,1]
        let delta = g.mul(diff, beta_t); // [Hv,Dv,1]
        let outer = g.mul(k_t, delta); // [Hv,1,Dk]·[Hv,Dv,1] → [Hv,Dv,Dk]
        state = g.add(s_decay, outer);
        let sqt = g.mul(state, q_t); // [Hv,Dv,Dk]
        let y_t = g.sum(sqt, vec![2], false); // [Hv,Dv]
        let y_t = g.reshape_(y_t, vec![1, 1, hv as i64, dv as i64]);
        ys.push(y_t);
    }
    let y = g.concat_(ys, 1); // [1,seq,Hv,Dv]
    let y_flat = g.reshape_(y, vec![batch as i64, seq as i64, p as i64]);

    // o_norm (per-head RMS over Dv, with gain) × sigmoid(out-gate) → o_proj.
    let o_norm_g = load_norm(g, params, weights, &format!("{sa}.o_norm.weight"), 0.0)?;
    let o_beta = synth_zero(g, params, &format!("{sa}.kda.o_beta"), dv);
    let y_normed = per_head_rms(g, y_flat, o_norm_g, o_beta, batch, seq, hv, dv, rms_eps);
    let gate_act = g.sigmoid(out_gate);
    let gated = g.mul(y_normed, gate_act);
    let o_p = load_proj(g, params, packed, weights, &format!("{sa}.o_proj.weight"))?;
    Ok(emit_proj(
        g,
        gated,
        &o_p,
        Shape::new(&[batch, seq, hidden], f),
    ))
}

/// Build a **Kimi-Linear (`kimi_linear`)** prefill graph. Per-layer attention is
/// KDA (`build_kimi_kda`) or NoPE-MLA (the validated `build_deepseek_mla` with
/// identity cos/sin); the FFN is a dense SwiGLU (first `first_k_dense_replace`
/// layers) or the validated deepseek MoE. Untied head. Returns logits
/// `[seq, vocab]`.
///
/// KDA is the novel piece and is numerically validated in isolation
/// (`examples/kimi_kda_probe.rs`) vs the mlx-lm `gated_delta_ops` reference.
/// 48B-A3B MoE → e2e validation deferred.
pub fn build_kimi_linear_prefill(
    spec: &KimiLinearSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("kimi_linear_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let eps = spec.rms_norm_eps;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "kl.zb.hidden", h);

    // Identity RoPE tables for the NoPE-MLA layers (cos=1, sin=0 → no rotation).
    let rope = spec.qk_rope_head_dim;
    let half = (rope / 2).max(1);
    let cos_id = synth_const(
        &mut g,
        &mut params,
        "kl.rope.cos",
        vec![1f32; seq * half],
        &[seq, half],
    );
    let sin_id = synth_zero(&mut g, &mut params, "kl.rope.sin", seq * half);
    let sin_id = g.reshape_(sin_id, vec![seq as i64, half as i64]);

    // DeepSeek MLA/MoE reuse spec (NoPE via identity rope; default qk^-0.5 scale).
    let ds = DeepseekSpec {
        vocab_size: spec.vocab_size,
        hidden_size: h,
        num_hidden_layers: spec.num_hidden_layers,
        num_attention_heads: spec.num_attention_heads,
        q_lora_rank: 0,      // Kimi-Linear MLA uses direct q_proj (no q-LoRA)
        absorbed_mla: false, // Kimi stores kv_b_proj (not absorbed)
        kv_lora_rank: spec.kv_lora_rank,
        qk_nope_head_dim: spec.qk_nope_head_dim,
        qk_rope_head_dim: spec.qk_rope_head_dim,
        v_head_dim: spec.v_head_dim,
        intermediate_size: spec.intermediate_size,
        moe_intermediate_size: spec.moe_intermediate_size,
        n_routed_experts: spec.num_experts,
        num_experts_per_tok: spec.num_experts_per_tok,
        n_shared_experts: spec.num_shared_experts,
        first_k_dense_replace: spec.first_k_dense_replace,
        routed_scaling_factor: spec.routed_scaling_factor,
        norm_topk_prob: spec.moe_renormalize,
        sigmoid_gate: true,
        sqrtsoftplus_gate: false,
        swiglu_limit: 0.0,
        rope_theta: 10000.0,
        rope_scaling: RopeScaling::None,
        attn_score_scale: None,
        rope_neox: true,
        rms_norm_eps: eps,
    };

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let is_kda = spec.kda_layers.contains(&(il + 1));
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);
        let attn_out = if is_kda {
            build_kimi_kda(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn"),
                normed,
                seq,
                h,
                spec.kda_num_heads,
                spec.kda_head_dim,
                spec.kda_conv_kernel,
                eps,
            )?
        } else {
            build_deepseek_mla(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                normed,
                cos_id,
                sin_id,
                batch,
                seq,
                &ds,
            )?
        };
        let post_attn = g.add(h_id, attn_out);

        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let mlp_out = if il < spec.first_k_dense_replace {
            let gate_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let up_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed2,
                &gate_p,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let up = emit_proj(
                &mut g,
                normed2,
                &up_p,
                Shape::new(&[batch, seq, spec.intermediate_size], f),
            );
            let sg = g.silu(gate);
            let glu = g.mul(sg, up);
            let down_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.down_proj.weight"),
            )?;
            emit_proj(&mut g, glu, &down_p, Shape::new(&[batch, seq, h], f))
        } else {
            build_deepseek_moe(
                &mut g,
                &mut params,
                packed,
                weights,
                &lp,
                normed2,
                batch,
                seq,
                &ds,
            )?
        };
        h_id = g.add(post_attn, mlp_out);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// gpt-oss decoder spec (attention-with-sinks + MXFP4 MoE).
#[derive(Debug, Clone)]
pub struct GptOssSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    /// Expert FFN width (`intermediate_size`).
    pub moe_inter: usize,
    pub swiglu_limit: f32,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub rms_norm_eps: f32,
}

/// Dequantize a packed MLX embed/linear to a dense F32 `[out, in]` param via
/// [`WeightLoader::take_packed_mlx`] (which honors per-module quant configs) —
/// the dense `take` path uses only the GLOBAL group_size, so it mis-widens a
/// per-module-affine embed in a mixed-quant checkpoint (gpt-oss: global mxfp4
/// gs=32 but embed affine gs=64). Registers the F32 table and returns its node.
fn load_dense_dequant(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, usize, usize)> {
    // Dense (already-F32) embed/head: take verbatim (e.g. synthetic tests).
    let Some(p) = weights.take_packed_mlx(key)? else {
        let (data, shape) = weights.take(key)?;
        let out = shape.first().copied().unwrap_or(0);
        let inn = shape.get(1).copied().unwrap_or(0);
        let node = g.param(key, Shape::new(&[out, inn], DType::F32));
        params.insert(key.to_string(), data);
        return Ok((node, out, inn));
    };
    let out = p.out_shape.first().copied().unwrap_or(0);
    let inn = p.out_shape.get(1).copied().unwrap_or(0);
    let n_groups = p.n_groups().max(1);
    let w = match p.scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let scales = f32_from_le_bytes(&p.scales);
            let biases = if p.biases.is_empty() {
                vec![0f32; out * n_groups]
            } else {
                f32_from_le_bytes(&p.biases)
            };
            rlx_mlx_io::dequant_affine_f32(
                &p.w_q,
                &scales,
                &biases,
                bits as u32,
                group_size,
                out,
                n_groups,
            )?
        }
        QuantScheme::MlxMxfp4 { group_size } => {
            rlx_mlx_io::dequant_mxfp4_f32(&p.w_q, &p.scales, group_size, out, n_groups)?
        }
        other => return Err(anyhow!("{key}: unsupported embed scheme {other}")),
    };
    let node = g.param(key, Shape::new(&[out, inn], DType::F32));
    params.insert(key.to_string(), w);
    Ok((node, out, inn))
}

/// gpt-oss attention with per-head **sinks**: a learned per-head logit joins the
/// softmax denominator (implemented as an extra score column, dropped after
/// softmax so it steals probability mass but contributes no value). GQA + RoPE +
/// Q/K/V/O biases; causal mask (sliding-window == full for seq ≤ window).
#[allow(clippy::too_many_arguments)]
fn build_gpt_oss_attention_with_sinks(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    causal_mask: NodeId,
    batch: usize,
    seq: usize,
    hidden: usize,
    nh: usize,
    nkv: usize,
    dh: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let group = nh / nkv.max(1);
    let kv_dim = nkv * dh;
    let q_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.q_proj.weight"),
    )?;
    let k_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.k_proj.weight"),
    )?;
    let v_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.v_proj.weight"),
    )?;
    let mut q = emit_proj(g, x, &q_p, Shape::new(&[batch, seq, nh * dh], f));
    let mut k = emit_proj(g, x, &k_p, Shape::new(&[batch, seq, kv_dim], f));
    let mut v = emit_proj(g, x, &v_p, Shape::new(&[batch, seq, kv_dim], f));
    // gpt-oss has attention_bias=true.
    let qb = load_p(
        g,
        params,
        weights,
        &format!("{lp}.self_attn.q_proj.bias"),
        false,
    )?;
    let kb = load_p(
        g,
        params,
        weights,
        &format!("{lp}.self_attn.k_proj.bias"),
        false,
    )?;
    let vb = load_p(
        g,
        params,
        weights,
        &format!("{lp}.self_attn.v_proj.bias"),
        false,
    )?;
    q = g.add(q, qb);
    k = g.add(k, kb);
    v = g.add(v, vb);
    let q_rope = rope_heads(g, q, cos, sin, batch, seq, nh, dh, true);
    let k_rope = rope_heads(g, k, cos, sin, batch, seq, nkv, dh, true);
    let k_rep = repeat_kv(g, k_rope, nkv, dh, group); // [b, seq, nh*dh]
    let v_rep = repeat_kv(g, v, nkv, dh, group);

    // → [b, nh, seq, dh].
    let to_heads = |g: &mut Graph, t: NodeId| -> NodeId {
        let r = g.reshape_(t, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
        g.transpose_(r, vec![0, 2, 1, 3])
    };
    let q4 = to_heads(g, q_rope);
    let k4 = to_heads(g, k_rep);
    let v4 = to_heads(g, v_rep);
    let kt = g.transpose_(k4, vec![0, 1, 3, 2]); // [b, nh, dh, seq]
    let scores = g.mm(q4, kt); // [b, nh, seq, seq]
    let scale = const1(
        g,
        params,
        &format!("{lp}.attn.scale"),
        1.0 / (dh as f32).sqrt(),
    );
    let scores = g.mul(scores, scale);
    let scores = g.add(scores, causal_mask); // [seq,seq] broadcasts

    // Sink column: sinks[nh] → [1,nh,1,1] broadcast to [b,nh,seq,1].
    let (sinks, _) = weights.take(&format!("{lp}.self_attn.sinks"))?;
    let sink_name = format!("{lp}.attn.sinks");
    let sink_p = g.param(&sink_name, Shape::new(&[1, nh, 1, 1], DType::F32));
    params.insert(sink_name, sinks);
    let sink_zeros = register_zeros(
        g,
        params,
        &format!("{lp}.attn.sink_zeros"),
        &[batch, nh, seq, 1],
    );
    let sink_col = g.add(sink_zeros, sink_p); // [b, nh, seq, 1]
    let scores_aug = g.concat_(vec![scores, sink_col], 3); // [b, nh, seq, seq+1]
    let probs_aug = g.sm(scores_aug, -1);
    let probs = g.narrow_(probs_aug, 3, 0, seq); // drop the sink column
    let out4 = g.mm(probs, v4); // [b, nh, seq, dh]
    let out_bshd = g.transpose_(out4, vec![0, 2, 1, 3]); // [b, seq, nh, dh]
    let out = g.reshape_(out_bshd, vec![batch as i64, seq as i64, (nh * dh) as i64]);

    let o_p = load_proj(
        g,
        params,
        packed,
        weights,
        &format!("{lp}.self_attn.o_proj.weight"),
    )?;
    let mut o = emit_proj(g, out, &o_p, Shape::new(&[batch, seq, hidden], f));
    let ob = load_p(
        g,
        params,
        weights,
        &format!("{lp}.self_attn.o_proj.bias"),
        false,
    )?;
    o = g.add(o, ob);
    Ok(o)
}

/// Build a **gpt-oss** prefill graph: embed → N×[input_layernorm →
/// attention-with-sinks → residual → post_attention_layernorm → MXFP4 MoE →
/// residual] → final norm → untied LM head. Returns logits `[seq, vocab]`.
/// (Sliding/full attention is treated as full causal — exact for seq ≤ window.)
pub fn build_gpt_oss_prefill(
    spec: &GptOssSpec,
    weights: &mut dyn WeightLoader,
    seq: usize,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("gpt_oss_prefill");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let batch = 1;
    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let eps = spec.rms_norm_eps;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "gptoss.zero_beta.hidden", h);

    // RoPE (YaRN) tables.
    let half = dh / 2;
    let mut cos_data = vec![0f32; seq * half];
    let mut sin_data = vec![0f32; seq * half];
    let mscale = spec.rope_scaling.mscale();
    for pos in 0..seq {
        for i in 0..half {
            let freq = spec.rope_scaling.inv_freq(i, dh, spec.rope_theta);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_data[pos * half + i] = (c * mscale) as f32;
            sin_data[pos * half + i] = (s * mscale) as f32;
        }
    }
    let cos_id = g.param("gptoss.rope.cos", Shape::new(&[seq, half], f));
    params.insert("gptoss.rope.cos".into(), cos_data);
    let sin_id = g.param("gptoss.rope.sin", Shape::new(&[seq, half], f));
    params.insert("gptoss.rope.sin".into(), sin_data);

    // Causal mask [seq, seq] (0 on/below diagonal, −inf above).
    let mut mask_data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let causal_mask = g.param("gptoss.causal_mask", Shape::new(&[seq, seq], f));
    params.insert("gptoss.causal_mask".into(), mask_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let (embed_w, _, _) =
        load_dense_dequant(&mut g, &mut params, weights, "model.embed_tokens.weight")?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    for il in 0..spec.num_hidden_layers {
        let lp = format!("model.layers.{il}");
        let in_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            0.0,
        )?;
        let normed = g.rms_norm(h_id, in_ln, zero_beta_hidden, eps);
        let attn = build_gpt_oss_attention_with_sinks(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            normed,
            cos_id,
            sin_id,
            causal_mask,
            batch,
            seq,
            h,
            nh,
            nkv,
            dh,
        )?;
        let post_attn = g.add(h_id, attn);
        let post_ln = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            0.0,
        )?;
        let normed2 = g.rms_norm(post_attn, post_ln, zero_beta_hidden, eps);
        let moe = build_gpt_oss_moe_ffn(
            &mut g,
            &mut params,
            packed,
            weights,
            &lp,
            normed2,
            batch,
            seq,
            h,
            spec.num_experts,
            spec.experts_per_token,
            spec.moe_inter,
            spec.swiglu_limit,
            1.702,
        )?;
        h_id = g.add(post_attn, moe);
    }

    let final_norm = load_norm(&mut g, &mut params, weights, "model.norm.weight", 0.0)?;
    let hidden = g.rms_norm(h_id, final_norm, zero_beta_hidden, eps);
    // Untied LM head (affine-packed).
    let head_p = load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
    let logits = emit_proj(
        &mut g,
        hidden,
        &head_p,
        Shape::new(&[batch, seq, spec.vocab_size], f),
    );
    let logits2d = g.reshape_(logits, vec![(batch * seq) as i64, spec.vocab_size as i64]);
    g.set_outputs(vec![logits2d]);
    Ok((g, params))
}

/// Build a standard causal-decoder IR graph with packed weights.
///
/// Verbatim generalization of `rlx-qwen3`'s `build_qwen3_graph_sized_packed`
/// — pass an empty `packed` map; it is filled with U8 codes for every
/// quantized matmul weight. When `with_lm_head` is false the output is
/// the post-final-norm hidden `[batch, seq, hidden]`; otherwise logits.
/// With `last_token_from_input`, a `last_token_idx` input gathers the
/// final row before the LM head so only one logit row is produced.
#[allow(clippy::too_many_arguments)]
pub fn build_standard_decoder_packed(
    spec: &DecoderSpec,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    // When true, the graph takes a fused `inputs_embeds [batch, seq, hidden]`
    // F32 input directly (skipping the token-id embedding gather) — used by
    // speech/vision prefixes that splice encoder features into the token stream.
    embeds_input: bool,
    packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    spec.guard_supported()?;

    let mut g = Graph::new(format!("{}_packed", spec.arch));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = spec.hidden_size;
    let nh = spec.num_attention_heads;
    let nkv = spec.num_key_value_heads;
    let dh = spec.head_dim;
    let group = spec.kv_group_size();
    let eps = spec.rms_norm_eps;

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "decoder.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "decoder.zero_beta.head_dim", dh);

    // RoPE tables sized to compile `seq`. Partial rotary (`partial_rotary_factor`
    // < 1, e.g. Phi-4-mini 0.75) rotates only the first `n_rot` of `head_dim`:
    // the table stays `[seq, head_dim/2]` (rope op stride) but only the first
    // `rot_half = n_rot/2` columns are populated (rest stay 0), and the inverse
    // frequencies use `n_rot` as the rotary width (so long/short factors index
    // correctly). Full rotary keeps `rot_dim == head_dim`, `rot_half == half` →
    // byte-identical to before.
    let half = dh / 2;
    let n_rot = if spec.partial_rotary_factor < 1.0 {
        ((spec.partial_rotary_factor * dh as f64) as usize) & !1
    } else {
        dh
    };
    let rot_dim = n_rot;
    let rot_half = n_rot / 2;
    let rope_len = seq;
    let build_rope = |g: &mut Graph,
                      params: &mut HashMap<String, Vec<f32>>,
                      theta: f64,
                      sc: &RopeScaling,
                      tag: &str|
     -> (NodeId, NodeId) {
        let mut cos_data = vec![0f32; rope_len * half];
        let mut sin_data = vec![0f32; rope_len * half];
        let mscale = sc.mscale();
        for pos in 0..rope_len {
            for i in 0..rot_half {
                let freq = sc.inv_freq(i, rot_dim, theta);
                let angle = pos as f64 * freq;
                let (s, c) = angle.sin_cos();
                cos_data[pos * half + i] = (c * mscale) as f32;
                sin_data[pos * half + i] = (s * mscale) as f32;
            }
        }
        let cn = format!("rope.cos.{tag}");
        let sn = format!("rope.sin.{tag}");
        let cos = g.param(&cn, Shape::new(&[rope_len, half], DType::F32));
        params.insert(cn, cos_data);
        let sin = g.param(&sn, Shape::new(&[rope_len, half], DType::F32));
        params.insert(sn, sin_data);
        (cos, sin)
    };
    // Global (default) table — `rope_scaling` (yarn/llama3/…) applies here.
    let (cos_id, sin_id) = build_rope(
        &mut g,
        &mut params,
        spec.rope_theta,
        &spec.rope_scaling,
        "global",
    );
    // Gemma3 dual-θ: a second plain-RoPE table at the local base frequency, used
    // on the sliding layers; `(cos_l, sin_l, pattern)`.
    let local_rope = spec.rope_dual.map(|(local_theta, pattern)| {
        let (cl, sl) = build_rope(
            &mut g,
            &mut params,
            local_theta,
            &RopeScaling::None,
            "local",
        );
        (cl, sl, pattern)
    });

    let last_token_idx = if with_lm_head && last_token_from_input {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::I32)))
    } else {
        None
    };

    let mut h_id = if embeds_input {
        // Caller supplies fused inputs_embeds (e.g. audio features spliced into
        // the token-embedding stream) — no gather, no embed table.
        g.input("inputs_embeds", Shape::new(&[batch, seq, h], DType::F32))
    } else {
        // Token IDs MUST be I32. Embedding stays F32 — gather needs dequant'd table.
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
        let embed_w = load_p(
            &mut g,
            &mut params,
            weights,
            "model.embed_tokens.weight",
            false,
        )?;
        g.gather_(embed_w, input_ids, 0)
    };
    // Gemma √hidden / Granite `embedding_multiplier` embed scale; no-op for Llama/Qwen.
    if (spec.embed_scale - 1.0).abs() > f32::EPSILON {
        let sc = synth_const(
            &mut g,
            &mut params,
            "decoder.embed_scale",
            vec![spec.embed_scale],
            &[1],
        );
        h_id = g.mul(h_id, sc);
    }
    // Granite `residual_multiplier`: reused per residual add (`x + m·sublayer(x)`).
    let resid_mul = if (spec.residual_multiplier - 1.0).abs() > f32::EPSILON {
        Some(synth_const(
            &mut g,
            &mut params,
            "decoder.residual_mul",
            vec![spec.residual_multiplier],
            &[1],
        ))
    } else {
        None
    };

    // Flat rank-3 rope is correct on every backend; BHSD is legacy.
    let rope_flat = rlx_ir::env::flag_or("RLX_QWEN3_ROPE_FLAT", true);
    // Diagnostic: capture a layer-0 intermediate as the graph output.
    let diag_stage = rlx_ir::env::var("RLX_DIAG_STAGE");
    let mut diag_node: Option<NodeId> = None;

    // Diagnostic: cap the layer count to bisect a per-layer backend divergence.
    let n_layers = rlx_ir::env::var("RLX_MAX_LAYERS")
        .and_then(|s| s.parse::<usize>().ok())
        .map(|c| c.min(spec.num_hidden_layers))
        .unwrap_or(spec.num_hidden_layers);
    // MoE is decided PER LAYER by probing for expert tensors, so mixed
    // architectures (some dense + some sparse layers — `mlp_only_layers`,
    // `decoder_sparse_step`, DeepSeek `first_k_dense_replace`) route each layer
    // correctly instead of assuming a uniform stack.
    let all_keys: std::collections::HashSet<String> =
        weights.remaining_keys().into_iter().collect();
    for layer_idx in 0..n_layers {
        let lp = format!("model.layers.{layer_idx}");
        let ofs = spec.norm_gain_offset;

        let in_ln_g = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            ofs,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);
        if layer_idx == 0 && diag_stage.as_deref() == Some("normed_in") {
            diag_node = Some(normed_in);
        }

        let q_dim = nh * dh;
        let kv_dim = nkv * dh;
        let (q_p, k_p, v_p) = if spec.fused_qkv {
            // Phi-3: one `qkv_proj` packed `[q_dim + 2*kv_dim, hidden]`.
            let mut parts = load_fused_split_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.qkv_proj.weight"),
                &[("q", q_dim), ("k", kv_dim), ("v", kv_dim)],
            )?;
            let v = parts.pop().unwrap();
            let k = parts.pop().unwrap();
            let q = parts.pop().unwrap();
            (q, k, v)
        } else {
            let q_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let k_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.k_proj.weight"),
            )?;
            let v_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.self_attn.v_proj.weight"),
            )?;
            (q_p, k_p, v_p)
        };
        let mut q = emit_proj(&mut g, normed_in, &q_p, Shape::new(&[batch, seq, q_dim], f));
        let mut k = emit_proj(
            &mut g,
            normed_in,
            &k_p,
            Shape::new(&[batch, seq, kv_dim], f),
        );
        let mut v = emit_proj(
            &mut g,
            normed_in,
            &v_p,
            Shape::new(&[batch, seq, kv_dim], f),
        );

        // Qwen2/2.5 ship explicit Q/K/V bias vectors; Qwen3/Llama/Mistral do not.
        if spec.attention_bias {
            let q_bias = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.q_proj.bias"),
                false,
            )?;
            let k_bias = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.k_proj.bias"),
                false,
            )?;
            let v_bias = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.v_proj.bias"),
                false,
            )?;
            q = g.add(q, q_bias);
            k = g.add(k, k_bias);
            v = g.add(v, v_bias);
        }

        // Qwen3/Gemma3 apply per-head RMS-norm on Q/K before RoPE ("QK-norm").
        let (q_normed, k_normed) = if spec.qk_norm {
            let q_norm_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.q_norm.weight"),
                ofs,
            )?;
            let k_norm_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.k_norm.weight"),
                ofs,
            )?;
            let (qn, kn) = if spec.qk_norm_full {
                // OLMoE: one RMSNorm over the entire `[heads*head_dim]` vector.
                let q_beta = synth_zero(&mut g, &mut params, &format!("{lp}.q_norm.beta"), nh * dh);
                let k_beta =
                    synth_zero(&mut g, &mut params, &format!("{lp}.k_norm.beta"), nkv * dh);
                (
                    g.rms_norm(q, q_norm_g, q_beta, eps),
                    g.rms_norm(k, k_norm_g, k_beta, eps),
                )
            } else {
                (
                    per_head_rms(
                        &mut g,
                        q,
                        q_norm_g,
                        zero_beta_headdim,
                        batch,
                        seq,
                        nh,
                        dh,
                        eps,
                    ),
                    per_head_rms(
                        &mut g,
                        k,
                        k_norm_g,
                        zero_beta_headdim,
                        batch,
                        seq,
                        nkv,
                        dh,
                        eps,
                    ),
                )
            };
            (qn, kn)
        } else {
            (q, k)
        };

        // Gemma3 dual-θ: local (sliding) θ on layers where `(i+1) % pattern != 0`,
        // global θ on the rest. Single-θ everywhere else.
        let (cos_l, sin_l) = match local_rope {
            Some((cl, sl, pat)) if !(layer_idx + 1).is_multiple_of(pat) => (cl, sl),
            _ => (cos_id, sin_id),
        };
        // Partial rotary (n_rot < head_dim): rotate only the first n_rot dims
        // (NeoX rotate-half, flat layout), pass the rest through. Full rotary
        // uses the existing `rope_heads` path unchanged.
        let (q_rope, k_rope) = if n_rot < dh {
            (
                g.rope_n_styled(q_normed, cos_l, sin_l, dh, n_rot, RopeStyle::NeoX),
                g.rope_n_styled(k_normed, cos_l, sin_l, dh, n_rot, RopeStyle::NeoX),
            )
        } else {
            (
                rope_heads(
                    &mut g, q_normed, cos_l, sin_l, batch, seq, nh, dh, rope_flat,
                ),
                rope_heads(
                    &mut g, k_normed, cos_l, sin_l, batch, seq, nkv, dh, rope_flat,
                ),
            )
        };

        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        // Build Op::Attention directly to carry the optional Gemma score scale
        // (`query_pre_attn_scalar**-0.5`) and attention-logit softcap (Gemma2).
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: spec.attn_mask_kind(),
                score_scale: spec.attn_score_scale,
                attn_logit_softcap: spec.attn_logit_softcap,
            },
            vec![q_rope, k_rep, v_rep],
            attn_shape,
        );

        let o_p = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        if layer_idx == 0 && diag_stage.as_deref() == Some("attn") {
            diag_node = Some(attn);
        }
        let attn_out = emit_proj(&mut g, attn, &o_p, Shape::new(&[batch, seq, h], f));
        // Gemma normalizes the attention output BEFORE the residual add
        // (`post_attention_layernorm`); Llama uses that tensor as the pre-FFN
        // norm after the residual instead.
        let attn_contrib = if spec.sandwich_norms {
            let post_attn_ln = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.post_attention_layernorm.weight"),
                ofs,
            )?;
            g.rms_norm(attn_out, post_attn_ln, zero_beta_hidden, eps)
        } else {
            attn_out
        };
        // Granite scales the sublayer output by `residual_multiplier` (else no-op).
        let attn_contrib = match resid_mul {
            Some(m) => g.mul(attn_contrib, m),
            None => attn_contrib,
        };
        let post_attn = g.add(h_id, attn_contrib);
        if layer_idx == 0 && diag_stage.as_deref() == Some("post_attn") {
            diag_node = Some(post_attn);
        }

        // Pre-FFN norm: Gemma `pre_feedforward_layernorm`, else reuse `post_attention_layernorm`.
        let pre_ffn_key = if spec.sandwich_norms {
            "pre_feedforward_layernorm.weight"
        } else {
            "post_attention_layernorm.weight"
        };
        let pre_ffn_g = load_norm(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.{pre_ffn_key}"),
            ofs,
        )?;
        let normed_pre = g.rms_norm(post_attn, pre_ffn_g, zero_beta_hidden, eps);

        let inter = spec.intermediate_size;
        // OLMoE stores per-expert tensors (`mlp.experts.{e}.*`); Qwen3-MoE /
        // Qwen2-MoE stack them into one `mlp.switch_mlp.*` tensor.
        let switch_moe = all_keys.contains(&format!("{lp}.mlp.switch_mlp.gate_proj.weight"));
        let layer_is_moe = spec.num_experts > 0
            && (switch_moe || all_keys.contains(&format!("{lp}.mlp.experts.0.gate_proj.weight")));
        let mut ffn_out = if layer_is_moe {
            // ── MoE FFN: router → softmax → top-k → grouped MLX-affine expert
            //    matmul (only k experts per token) → prob-weighted sum. ──
            let n_expert = spec.num_experts;
            let top_k = spec.num_experts_used.max(1);
            // Qwen3-MoE / Qwen2-MoE route through a narrower expert FFN
            // (`moe_intermediate_size`); OLMoE reuses `intermediate_size`.
            let moe_inter = if spec.moe_intermediate_size > 0 {
                spec.moe_intermediate_size
            } else {
                inter
            };
            let rows = batch * seq;
            let h_2d = g.reshape_(normed_pre, vec![rows as i64, h as i64]);
            // Router: a small quantized Linear `[n_expert, hidden]`.
            let router_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.gate.weight"),
            )?;
            let router_logits =
                emit_proj(&mut g, h_2d, &router_p, Shape::new(&[rows, n_expert], f));
            let probs = g.sm(router_logits, -1);
            let top_idx = g.add_node(
                Op::TopK { k: top_k },
                vec![probs],
                Shape::new(&[rows, top_k], DType::F32),
            );
            // `gather_` is ONNX-Gather (would give `[rows, rows, k]`); the
            // per-row top-k weight lookup is `GatherElements` (torch.gather).
            let mut top_probs = g.add_node(
                Op::GatherElements { axis: 1 },
                vec![probs, top_idx],
                Shape::new(&[rows, top_k], f),
            ); // [rows, top_k]
            if spec.norm_topk_prob {
                // Qwen3-MoE renormalizes the selected weights to sum to 1.
                let denom = g.sum(top_probs, vec![1], true); // [rows, 1]
                top_probs = g.div(top_probs, denom);
            }
            // Stacked `switch_mlp` (Qwen3/Qwen2-MoE) vs per-expert (OLMoE).
            // Expert FFN dims: gate/up map `hidden→moe_inter`, down `moe_inter→hidden`.
            let (gate_c, gate_s, gate_b, scheme, _) = if switch_moe {
                load_switch_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "gate_proj",
                    n_expert,
                    moe_inter,
                )?
            } else {
                load_stacked_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "gate_proj",
                    n_expert,
                )?
            };
            let (up_c, up_s, up_b, _, _) = if switch_moe {
                load_switch_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "up_proj",
                    n_expert,
                    moe_inter,
                )?
            } else {
                load_stacked_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "up_proj",
                    n_expert,
                )?
            };
            let (down_c, down_s, down_b, _, _) = if switch_moe {
                load_switch_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "down_proj",
                    n_expert,
                    h,
                )?
            } else {
                load_stacked_experts_mlx(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &lp,
                    "down_proj",
                    n_expert,
                )?
            };
            let mut acc: Option<NodeId> = None;
            for ki in 0..top_k {
                let e_idx_col = g.narrow_(top_idx, 1, ki, 1);
                let e_idx = g.reshape_(e_idx_col, vec![rows as i64]);
                let p_col = g.narrow_(top_probs, 1, ki, 1); // [rows, 1]
                let gate = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![h_2d, gate_c, gate_s, gate_b, e_idx],
                    Shape::new(&[rows, moe_inter], f),
                );
                let up = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![h_2d, up_c, up_s, up_b, e_idx],
                    Shape::new(&[rows, moe_inter], f),
                );
                let gate_act = g.silu(gate);
                let glu = g.mul(gate_act, up);
                let down = g.add_node(
                    Op::DequantGroupedMatMulMlx { scheme },
                    vec![glu, down_c, down_s, down_b, e_idx],
                    Shape::new(&[rows, h], f),
                );
                let weighted = g.mul(down, p_col);
                acc = Some(match acc {
                    None => weighted,
                    Some(a) => g.add(a, weighted),
                });
            }
            let mut moe_out = acc.unwrap(); // [rows, hidden]
            // Qwen2-MoE shared expert: an always-on SwiGLU FFN whose output is
            // scaled by `sigmoid(shared_expert_gate(x))` and added to the
            // routed sum. Tensor names live under `{lp}.mlp.shared_expert.*`.
            if spec.shared_expert_intermediate_size > 0 {
                let se_inter = spec.shared_expert_intermediate_size;
                let se_gate = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                )?;
                let se_up = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                )?;
                let se_down = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                )?;
                let sg = emit_proj(&mut g, h_2d, &se_gate, Shape::new(&[rows, se_inter], f));
                let su = emit_proj(&mut g, h_2d, &se_up, Shape::new(&[rows, se_inter], f));
                let sg_act = g.silu(sg);
                let sglu = g.mul(sg_act, su);
                let sd = emit_proj(&mut g, sglu, &se_down, Shape::new(&[rows, h], f));
                // Gate: a plain (unquantized) Linear [1, hidden] → sigmoid.
                let gate_w = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.shared_expert_gate.weight"),
                )?;
                let gate_logit = emit_proj(&mut g, h_2d, &gate_w, Shape::new(&[rows, 1], f));
                let gate_sig = g.sigmoid(gate_logit); // [rows, 1]
                let shared = g.mul(sd, gate_sig);
                moe_out = g.add(moe_out, shared);
            }
            g.reshape_(moe_out, vec![batch as i64, seq as i64, h as i64])
        } else {
            let (gate_p, up_p) = if spec.fused_gate_up {
                // Phi-3: one `gate_up_proj` packed `[2*inter, hidden]`.
                let mut parts = load_fused_split_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.gate_up_proj.weight"),
                    &[("gate", inter), ("up", inter)],
                )?;
                let up = parts.pop().unwrap();
                let gate = parts.pop().unwrap();
                (gate, up)
            } else {
                let gate_p = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.gate_proj.weight"),
                )?;
                let up_p = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    &format!("{lp}.mlp.up_proj.weight"),
                )?;
                (gate_p, up_p)
            };
            let down_p = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                &format!("{lp}.mlp.down_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed_pre,
                &gate_p,
                Shape::new(&[batch, seq, inter], f),
            );
            let up = emit_proj(
                &mut g,
                normed_pre,
                &up_p,
                Shape::new(&[batch, seq, inter], f),
            );
            // GeGLU (Gemma, tanh-approx GeLU) vs SwiGLU (SiLU).
            let gate_act = if spec.gelu_gate {
                g.gelu_approx(gate)
            } else {
                g.silu(gate)
            };
            let glu = g.mul(gate_act, up);
            emit_proj(&mut g, glu, &down_p, Shape::new(&[batch, seq, h], f))
        };
        // Gemma normalizes the FFN output before the residual (`post_feedforward_layernorm`).
        if spec.sandwich_norms {
            let post_ffn_g = load_norm(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.post_feedforward_layernorm.weight"),
                ofs,
            )?;
            ffn_out = g.rms_norm(ffn_out, post_ffn_g, zero_beta_hidden, eps);
        }
        let ffn_contrib = match resid_mul {
            Some(m) => g.mul(ffn_out, m),
            None => ffn_out,
        };
        h_id = g.add(post_attn, ffn_contrib);
        if layer_idx == 0 && diag_stage.as_deref() == Some("layer0") {
            diag_node = Some(h_id);
        }
    }

    let final_ln_g = load_norm(
        &mut g,
        &mut params,
        weights,
        "model.norm.weight",
        spec.norm_gain_offset,
    )?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

    let logits_or_hidden = if with_lm_head {
        let head_input = if let Some(idx) = last_token_idx {
            gather_last_token(&mut g, hidden, batch, idx)
        } else {
            hidden
        };
        if diag_stage.as_deref() == Some("head_input") {
            diag_node = Some(head_input);
        }
        let logit_rows = if last_token_from_input { 1 } else { seq };
        // Tied embeddings: transpose the F32 embed once at build time and
        // reuse it as the LM head (packed re-take of embed is unavailable —
        // the F32 dequant copy was already taken for the gather table).
        let lm_head_proj = if spec.tie_word_embeddings {
            let embed = params
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
            let vocab = spec.vocab_size;
            let hidden_size = spec.hidden_size;
            let mut transposed = vec![0f32; embed.len()];
            for vv in 0..vocab {
                for hi in 0..hidden_size {
                    transposed[hi * vocab + vv] = embed[vv * hidden_size + hi];
                }
            }
            let name = "decoder.lm_head.tied_t";
            let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
            params.insert(name.to_string(), transposed);
            Proj {
                w: id,
                scheme: None,
                scale: None,
                bias: None,
            }
        } else {
            load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?
        };
        let mut logits = emit_proj(
            &mut g,
            head_input,
            &lm_head_proj,
            Shape::new(&[batch, logit_rows, spec.vocab_size], f),
        );
        // Granite divides logits by `logits_scaling` (argmax/cos-invariant, but
        // sets the softmax temperature).
        if (spec.logits_scaling - 1.0).abs() > f32::EPSILON {
            let inv = synth_const(
                &mut g,
                &mut params,
                "decoder.logits_scaling.inv",
                vec![1.0 / spec.logits_scaling],
                &[1],
            );
            logits = g.mul(logits, inv);
        }
        // Gemma2 final-logit softcap: `s·tanh(logits/s)`.
        if let Some(cap) = spec.final_logit_softcap {
            let inv = synth_const(
                &mut g,
                &mut params,
                "decoder.logit_softcap.inv",
                vec![1.0 / cap],
                &[1],
            );
            let capc = synth_const(
                &mut g,
                &mut params,
                "decoder.logit_softcap.cap",
                vec![cap],
                &[1],
            );
            let scaled = g.mul(logits, inv);
            let t = g.tanh(scaled);
            g.mul(t, capc)
        } else {
            logits
        }
    } else {
        hidden
    };
    let out = diag_node.unwrap_or(logits_or_hidden);

    g.set_outputs(vec![out]);
    Ok((g, params))
}

// ─── config.json → DecoderSpec (arch-agnostic) ────────────────────────

fn cfg_usize(v: &serde_json::Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
}

/// Parse `config.rope_scaling` → [`RopeScaling`]. Recognizes the HF
/// `rope_type`/`type` == `"llama3"` piecewise rescale and `"linear"`;
/// unknown / `"default"` / `null` → [`RopeScaling::None`]. `text` supplies
/// `max_position_embeddings` for llama3's original-context fallback.
fn parse_rope_scaling(rs: Option<&serde_json::Value>, text: &serde_json::Value) -> RopeScaling {
    let Some(obj) = rs.and_then(|x| x.as_object()) else {
        return RopeScaling::None;
    };
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|x| x.as_str())
        .unwrap_or("default");
    let getf = |k: &str| obj.get(k).and_then(|x| x.as_f64());
    match rope_type {
        "llama3" => RopeScaling::Llama3 {
            factor: getf("factor").unwrap_or(8.0),
            low_freq_factor: getf("low_freq_factor").unwrap_or(1.0),
            high_freq_factor: getf("high_freq_factor").unwrap_or(4.0),
            original_max_position_embeddings: getf("original_max_position_embeddings")
                .or_else(|| text.get("max_position_embeddings").and_then(|x| x.as_f64()))
                .unwrap_or(8192.0),
        },
        "linear" => RopeScaling::Linear {
            factor: getf("factor").unwrap_or(1.0),
        },
        "yarn" => RopeScaling::Yarn {
            factor: getf("factor").unwrap_or(1.0),
            original_max_position_embeddings: getf("original_max_position_embeddings")
                .or_else(|| text.get("max_position_embeddings").and_then(|x| x.as_f64()))
                .unwrap_or(4096.0),
            beta_fast: getf("beta_fast").unwrap_or(32.0),
            beta_slow: getf("beta_slow").unwrap_or(1.0),
            // HF spells it `attention_factor`; some mlx configs use `attn_factor`.
            attention_factor: getf("attention_factor").or_else(|| getf("attn_factor")),
        },
        // Phi-3/3.5 LongRoPE (`longrope` / legacy `su`): per-dim short/long factor
        // arrays + attention mscale = √(1 + ln(max/orig)/ln(orig)).
        "longrope" | "su" | "phi3" => {
            let getarr = |k: &str| {
                obj.get(k)
                    .and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
                    .unwrap_or_default()
            };
            let orig = getf("original_max_position_embeddings")
                .or_else(|| {
                    text.get("original_max_position_embeddings")
                        .and_then(|x| x.as_f64())
                })
                .unwrap_or(4096.0);
            let max_pos = text
                .get("max_position_embeddings")
                .and_then(|x| x.as_f64())
                .unwrap_or(orig);
            let scale = max_pos / orig.max(1.0);
            let mscale = if scale <= 1.0 {
                1.0
            } else {
                (1.0 + scale.ln() / orig.max(2.0).ln()).sqrt()
            };
            RopeScaling::LongRope {
                short_factor: getarr("short_factor"),
                long_factor: getarr("long_factor"),
                original_max_position_embeddings: orig,
                mscale,
            }
        }
        // `dynamic` (NTK) rescales the base only past the original context; for
        // seq ≤ original_max it is the identity, which is correct for prefill.
        _ => RopeScaling::None,
    }
}

/// Recognized `rope_scaling.rope_type`s (the rest → [`classify_config`] reject,
/// so an unhandled scaling can't silently mis-scale a "supported" model).
fn rope_type_supported(rs: Option<&serde_json::Value>) -> bool {
    let Some(obj) = rs.and_then(|x| x.as_object()) else {
        return true; // no rope_scaling block → plain RoPE
    };
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|x| x.as_str())
        .unwrap_or("default");
    matches!(
        rope_type,
        "default" | "linear" | "llama3" | "yarn" | "dynamic" | "longrope" | "su" | "phi3"
    )
}

impl DecoderSpec {
    /// Parse a HuggingFace `config.json` into a [`DecoderSpec`], inferring
    /// the topology flags (`qk_norm`, `attention_bias`, tied lm_head) from
    /// the loader's actual tensor names when they aren't stated in config —
    /// so a dense decoder whose `model_type` this code has never seen still
    /// builds correctly with no code change.
    ///
    /// `loader` is probed read-only via [`WeightLoader::remaining_keys`];
    /// nothing is taken.
    pub fn from_config_json(dir: &Path, loader: &dyn WeightLoader) -> Result<Self> {
        let bytes = std::fs::read(dir.join("config.json"))
            .map_err(|e| anyhow!("read {:?}/config.json: {e}", dir))?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| anyhow!("parse config.json: {e}"))?;

        // Reject topologies the generic builder can't model up front, with the
        // same verdict the catalog reports — fail loud instead of mis-building.
        let sup = classify_config(&v);
        if !sup.supported {
            return Err(anyhow!(
                "standard_decoder: unsupported model (arch={}): {} — needs a dedicated crate",
                sup.arch,
                sup.reason
            ));
        }

        // Some multimodal configs nest the LM under `text_config`.
        let text = v.get("text_config").unwrap_or(&v);

        let arch = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .or_else(|| {
                v.get("architectures")
                    .and_then(|a| a.as_array())
                    .and_then(|a| a.first())
                    .and_then(|x| x.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "decoder".to_string());

        let hidden_size = cfg_usize(text, "hidden_size")
            .ok_or_else(|| anyhow!("config.json missing hidden_size"))?;
        let num_attention_heads = cfg_usize(text, "num_attention_heads")
            .ok_or_else(|| anyhow!("config.json missing num_attention_heads"))?;
        let num_key_value_heads =
            cfg_usize(text, "num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim =
            cfg_usize(text, "head_dim").unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
        let num_hidden_layers = cfg_usize(text, "num_hidden_layers")
            .ok_or_else(|| anyhow!("config.json missing num_hidden_layers"))?;

        let keys = loader.remaining_keys();
        let has = |suffix: &str| keys.iter().any(|k| k.ends_with(suffix));

        // Infer topology flags from tensors first (authoritative), then
        // fall back to any explicit config value.
        let qk_norm = has("self_attn.q_norm.weight");
        // OLMoE norms the full `[heads*head_dim]` Q/K vector; Qwen3-family
        // norm per-head over `[head_dim]`.
        let qk_norm_full = qk_norm && arch == "olmoe";
        let attention_bias = has("self_attn.q_proj.bias")
            || text
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
        // Phi-3 fuses Q/K/V and gate/up into single tensors.
        let fused_qkv = has("self_attn.qkv_proj.weight");
        let fused_gate_up = has("mlp.gate_up_proj.weight");
        // Tied when config says so, OR when no untied lm_head weight exists.
        let tie_word_embeddings = text
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or_else(|| !has("lm_head.weight"));

        let rms_norm_eps = text
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-6) as f32;
        let rope_theta = text
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .unwrap_or(10_000.0);
        let rope_scaling = parse_rope_scaling(text.get("rope_scaling"), text);
        let partial_rotary_factor = text
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .unwrap_or(1.0);
        // Gemma configs name it `hidden_activation` (gelu_pytorch_tanh).
        let hidden_act = text
            .get("hidden_act")
            .or_else(|| text.get("hidden_activation"))
            .and_then(|x| x.as_str())
            .unwrap_or("silu")
            .to_string();
        let sliding_window = text
            .get("sliding_window")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);
        let use_sliding_window = text
            .get("use_sliding_window")
            .and_then(|x| x.as_bool())
            .unwrap_or(sliding_window.is_some());
        let max_window_layers = cfg_usize(text, "max_window_layers").unwrap_or(usize::MAX);
        let num_experts = cfg_usize(text, "num_experts")
            .or_else(|| cfg_usize(text, "n_routed_experts"))
            .or_else(|| cfg_usize(text, "num_local_experts"))
            .unwrap_or(0);
        let num_experts_used = cfg_usize(text, "num_experts_per_tok")
            .or_else(|| cfg_usize(text, "num_experts_used"))
            .unwrap_or(0);
        let norm_topk_prob = text
            .get("norm_topk_prob")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let moe_intermediate_size = cfg_usize(text, "moe_intermediate_size").unwrap_or(0);
        // Qwen2-MoE shared expert. Its FFN width has its own config key; the
        // `shared_expert_gate` tensor confirms the branch is present.
        let shared_expert_intermediate_size = if has("mlp.shared_expert_gate.weight") {
            cfg_usize(text, "shared_expert_intermediate_size").unwrap_or(0)
        } else {
            0
        };

        let vocab_size = cfg_usize(text, "vocab_size")
            .ok_or_else(|| anyhow!("config.json missing vocab_size"))?;
        let intermediate_size = cfg_usize(text, "intermediate_size")
            .ok_or_else(|| anyhow!("config.json missing intermediate_size"))?;

        // Gemma-family axes. Detected by arch (`gemma*`); the loader-probed
        // `pre_feedforward_layernorm` confirms the sandwich structure.
        let is_gemma = arch.to_ascii_lowercase().starts_with("gemma");
        let is_granite = arch.to_ascii_lowercase().starts_with("granite");
        let getf = |k: &str| text.get(k).and_then(|x| x.as_f64());
        let sandwich_norms = is_gemma && has("pre_feedforward_layernorm.weight");
        let norm_gain_offset = if is_gemma { 1.0 } else { 0.0 };
        let gelu_gate = !matches!(hidden_act.to_ascii_lowercase().as_str(), "silu" | "swish");
        // Gemma scales the embedding by √hidden; Granite by `embedding_multiplier`.
        let embed_scale = if is_granite {
            getf("embedding_multiplier").unwrap_or(1.0) as f32
        } else if is_gemma {
            (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        // Granite scales every residual add by `residual_multiplier` and divides
        // final logits by `logits_scaling`.
        let residual_multiplier = getf("residual_multiplier").unwrap_or(1.0) as f32;
        let logits_scaling = getf("logits_scaling").unwrap_or(1.0) as f32;
        // Gemma3 alternates local (sliding, `rope_local_base_freq`) and global
        // (`rope_theta`) layers by `sliding_window_pattern`.
        let rope_dual = if is_gemma {
            match (
                text.get("rope_local_base_freq").and_then(|x| x.as_f64()),
                cfg_usize(text, "sliding_window_pattern"),
            ) {
                (Some(local), Some(pat)) if pat > 0 => Some((local, pat)),
                _ => None,
            }
        } else {
            None
        };
        // Attention score scale override: Gemma `query_pre_attn_scalar**-0.5`,
        // Granite `attention_multiplier` (used directly as the scale).
        let attn_score_scale = if is_granite {
            getf("attention_multiplier").map(|x| x as f32)
        } else {
            getf("query_pre_attn_scalar").map(|q| 1.0 / (q as f32).sqrt())
        };
        let attn_logit_softcap = text
            .get("attn_logit_softcapping")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32);
        let final_logit_softcap = text
            .get("final_logit_softcapping")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32);

        Ok(DecoderSpec {
            arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            partial_rotary_factor,
            rope_scaling,
            hidden_act,
            attention_bias,
            fused_qkv,
            fused_gate_up,
            qk_norm,
            qk_norm_full,
            tie_word_embeddings,
            sliding_window,
            use_sliding_window,
            max_window_layers,
            num_experts,
            num_experts_used,
            norm_topk_prob,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_gain_offset,
            sandwich_norms,
            embed_scale,
            gelu_gate,
            rope_dual,
            attn_score_scale,
            attn_logit_softcap,
            final_logit_softcap,
            residual_multiplier,
            logits_scaling,
        })
    }
}

/// Whether the generic `build_standard_decoder_packed` builder can run a
/// checkpoint, decided from its `config.json` alone (no weights). Authoritative
/// companion to [`DecoderSpec::from_config_json`] and the `mlx_catalog` tool, so
/// "supported" always means exactly what the builder handles.
#[derive(Debug, Clone)]
pub struct ModelSupport {
    pub supported: bool,
    /// The family ("dense SwiGLU+RMSNorm decoder") or the blocker
    /// ("MoE: 8 experts", "non-SwiGLU activation `gelu`", "Phi family", …).
    pub reason: String,
    pub arch: String,
    pub hidden_act: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub tie_word_embeddings: bool,
    pub rope: &'static str,
}

/// Families that pass the feature checks but have topology the generic builder
/// can't model (config `model_type` / `architectures[0]`, lowercased). Gemma
/// (GeGLU) and MoE are caught by the feature checks instead; this catches Phi
/// (fused QKV), state-space, and multimodal/encoder-decoder arches.
fn arch_is_known_unsupported(arch: &str) -> Option<&'static str> {
    let a = arch.to_ascii_lowercase();
    // Plain Gemma3 text (GeGLU + sandwich (1+γ) norms + √h embed + dual-θ rope,
    // no softcap) IS handled now. Everything else Gemma differs — gemma1/2
    // (softcap), gemma3n (AltUp/Laurel/MatFormer), gemma4 (altup) — keep rejected.
    if a.starts_with("gemma") && !matches!(a.as_str(), "gemma3" | "gemma3_text" | "gemma2") {
        return Some("Gemma 1/3n/4 (AltUp / other — gemma2 & gemma3 are generic-supported)");
    }
    // Phi-3 (fused QKV + gate/up, full rotary, RMSNorm) IS generic-supported.
    // Phi-2/Phi-1 (partial rotary + LayerNorm) and Phi-3 long-context (longrope)
    // are caught by the partial-rotary / rope_type / LayerNorm feature checks.
    if a.starts_with("phi") && a != "phi3" {
        return Some("Phi 1/2/MoE (partial rotary + LayerNorm / fused variants)");
    }
    // Qwen3.5/3.6/next = gated-DeltaNet + full-attention hybrid (needs rlx-qwen35).
    if a.contains("qwen3_5") || a.contains("qwen3next") || a.contains("qwen3_next") || a == "qwen35"
    {
        return Some("Qwen3.5 hybrid (gated DeltaNet — needs rlx-qwen35)");
    }
    // GPT-OSS = MoE + attention sinks.
    if a.contains("gpt_oss") || a.contains("gptoss") {
        return Some("GPT-OSS (MoE + attention sinks)");
    }
    if a.contains("mamba")
        || a.starts_with("rwkv")
        || a.contains("jamba")
        || a.contains("recurrent")
        || a.contains("nemotron_h")
    {
        return Some("state-space / Mamba-hybrid arch");
    }
    // Granite MoE variants still need the MoE path; dense Granite (scalar
    // multipliers) is handled by the generic builder now.
    if a.starts_with("granite") && a.contains("moe") {
        return Some("GraniteMoE (routed experts)");
    }
    if a.ends_with("_vl")
        || a.contains("qwen2_vl")
        || a.contains("qwen3_vl")
        || a.contains("llava")
        || a.contains("vision")
        || a.contains("clip")
        || a.contains("siglip")
        || a.contains("whisper")
        || a.starts_with("t5")
        || a.starts_with("bart")
    {
        return Some("multimodal / encoder-decoder / non-decoder arch");
    }
    None
}

/// Classify a parsed `config.json` against the generic decoder builder.
/// MoE architectures whose expert layout the generic builder reproduces
/// bit-for-bit: router `mlp.gate` → softmax → top-k → per-expert SwiGLU with
/// stacked MLX-affine weights, no shared expert. OLMoE is validated against
/// mlx-lm; others use the same `experts.{e}.{gate,up,down}_proj` layout.
fn moe_arch_supported(arch: &str) -> bool {
    matches!(
        arch.to_ascii_lowercase().as_str(),
        // Shared `mlp.experts.{e}.{gate,up,down}_proj` + `mlp.gate` router,
        // `moe_intermediate_size`, optional Qwen2-MoE shared expert. OLMoE is
        // oracle-validated; Qwen3-MoE / Qwen2-MoE reuse the same code path.
        "olmoe" | "qwen3_moe" | "qwen2_moe"
    )
}

pub fn classify_config(v: &serde_json::Value) -> ModelSupport {
    let text = v.get("text_config").unwrap_or(v);
    let arch = v
        .get("model_type")
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.get("architectures")
                .and_then(|a| a.as_array())
                .and_then(|a| a.first())
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "decoder".to_string());
    let hidden_act = text
        .get("hidden_act")
        .or_else(|| text.get("hidden_activation"))
        .and_then(|x| x.as_str())
        .unwrap_or("silu")
        .to_string();
    let num_experts = cfg_usize(text, "num_experts")
        .or_else(|| cfg_usize(text, "n_routed_experts"))
        .or_else(|| cfg_usize(text, "num_local_experts"))
        .unwrap_or(0);
    let vocab_size = cfg_usize(text, "vocab_size").unwrap_or(0);
    let hidden_size = cfg_usize(text, "hidden_size").unwrap_or(0);
    let num_hidden_layers = cfg_usize(text, "num_hidden_layers").unwrap_or(0);
    let tie_word_embeddings = text
        .get("tie_word_embeddings")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let rope = match parse_rope_scaling(text.get("rope_scaling"), text) {
        RopeScaling::None => "none",
        RopeScaling::Linear { .. } => "linear",
        RopeScaling::Llama3 { .. } => "llama3",
        RopeScaling::Yarn { .. } => "yarn",
        RopeScaling::LongRope { .. } => "longrope",
    };
    let has_rms = text.get("rms_norm_eps").is_some();
    let has_ln = text.get("layer_norm_eps").is_some()
        || text.get("layer_norm_epsilon").is_some()
        || text.get("norm_eps").is_some();
    let al = hidden_act.to_ascii_lowercase();
    // SwiGLU (silu/swish) or GeGLU (gelu*, Gemma-style gated MLP).
    let act_ok = matches!(al.as_str(), "silu" | "swish") || al.starts_with("gelu");

    let (supported, reason) = if let Some(r) = arch_is_known_unsupported(&arch) {
        (false, r.to_string())
    } else if num_experts > 0 && !moe_arch_supported(&arch) {
        (
            false,
            format!("MoE ({arch}): {num_experts} experts, unvalidated layout"),
        )
    } else if !act_ok {
        (false, format!("non-SwiGLU activation `{hidden_act}`"))
    } else if !has_rms && has_ln {
        (false, "LayerNorm (no RMSNorm)".to_string())
    } else if !rope_type_supported(text.get("rope_scaling")) {
        let rt = text
            .get("rope_scaling")
            .and_then(|x| x.get("rope_type").or_else(|| x.get("type")))
            .and_then(|x| x.as_str())
            .unwrap_or("?");
        (false, format!("unhandled rope_scaling `{rt}`"))
    } else if hidden_size == 0
        || num_hidden_layers == 0
        || vocab_size == 0
        || cfg_usize(text, "num_attention_heads").is_none()
        || cfg_usize(text, "intermediate_size").is_none()
    {
        (
            false,
            "incomplete / not a causal-decoder config".to_string(),
        )
    } else {
        (true, "dense SwiGLU+RMSNorm decoder".to_string())
    };

    ModelSupport {
        supported,
        reason,
        arch,
        hidden_act,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_experts,
        tie_word_embeddings,
        rope,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_v4_from_config_matches_real_checkpoint() {
        // Exact fields from the GA `deepseek-ai/DeepSeek-V4-Flash-0731/config.json`
        // (verified against the HF config + reference inference/config.json). Asserts
        // from_config maps them — including YaRN (rope_scaling) and DSpark — to the
        // layout confirmed against the checkpoint tensor index.
        let mut cfg = serde_json::json!({
            "model_type": "deepseek_v4",
            "vocab_size": 129280, "hidden_size": 4096, "num_hidden_layers": 43,
            "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
            "index_head_dim": 128, "index_n_heads": 64, "index_topk": 512,
            "sliding_window": 128, "num_hash_layers": 3,
            "n_routed_experts": 256, "num_experts_per_tok": 6, "n_shared_experts": 1,
            "moe_intermediate_size": 2048, "routed_scaling_factor": 1.5,
            "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
            "rms_norm_eps": 1e-6, "hc_mult": 4,
            "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "swiglu_limit": 10.0,
            "num_nextn_predict_layers": 1, "expert_dtype": "fp4",
            "rope_scaling": {
                "type": "yarn", "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32, "beta_slow": 1
            },
            "dspark_block_size": 5, "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42], "dspark_markov_rank": 256,
        });
        // Real GA pattern: [0,0,4,128,…,4,0,0,0] — 46 entries (43 layers + 3 DSpark).
        let ratios: Vec<usize> = (0..46)
            .map(|i| {
                if i < 2 || i >= 43 {
                    0
                } else if i % 2 == 0 {
                    4
                } else {
                    128
                }
            })
            .collect();
        cfg["compress_ratios"] = serde_json::json!(ratios);
        let s = DeepseekV4Spec::from_config(&cfg).unwrap();
        assert_eq!(s.n_layers, 43);
        assert_eq!(s.vocab_size, 129280);
        assert_eq!(s.dim, 4096);
        assert_eq!(s.n_heads, 64);
        assert_eq!(s.head_dim, 512);
        assert_eq!(s.rope_head_dim, 64);
        assert_eq!(s.q_lora_rank, 1024);
        assert_eq!(s.o_lora_rank, 1024);
        assert_eq!(s.n_groups, 8);
        assert_eq!(s.index_head_dim, 128);
        assert_eq!(s.index_n_heads, 64);
        assert_eq!(s.index_topk, 512);
        assert_eq!(s.window_size, 128);
        assert_eq!(s.n_hash_layers, 3);
        assert_eq!(s.n_routed_experts, 256);
        assert_eq!(s.n_activated_experts, 6);
        assert_eq!(s.n_shared_experts, 1);
        assert_eq!(s.hc_mult, 4);
        assert_eq!(s.hc_sinkhorn_iters, 20);
        assert_eq!(s.first_k_dense_replace, 0); // V4 has no dense FFN layers
        assert_eq!(s.compress_rope_theta, 160000.0);
        // YaRN (compressed-layer RoPE): rope_scaling → spec fields.
        assert_eq!(s.original_seq_len, 65536);
        assert_eq!(s.rope_factor, 16.0);
        assert_eq!(s.beta_fast, 32.0);
        assert_eq!(s.beta_slow, 1.0);
        // DSpark (GA speculative decoding): 3 stages (derived from the 46−43 surplus
        // compress_ratios entries, since the HF config only exposes 1 nextn layer).
        assert_eq!(s.n_mtp_layers, 3);
        assert_eq!(s.dspark_block_size, 5);
        assert_eq!(s.dspark_noise_token_id, 128799);
        assert_eq!(s.dspark_target_layer_ids, vec![40, 41, 42]);
        assert_eq!(s.dspark_markov_rank, 256);
        // compress_ratios truncated to n_layers (config carries 3 DSpark entries).
        assert_eq!(s.compress_ratios.len(), 43);
        assert_eq!(s.compress_ratios[0], 0);
        assert_eq!(s.compress_ratios[1], 0);
        assert_eq!(s.compress_ratios[2], 4); // ratio-4 → overlap + Indexer
        assert_eq!(s.compress_ratios[3], 128); // ratio-128 → non-overlap, no Indexer
        assert_eq!(s.compress_ratios[41], 128);
        assert_eq!(s.compress_ratios[42], 4); // last real layer; 43..45 (0) are DSpark
    }

    #[test]
    fn deepseek_v4_yarn_freqs_match_reference() {
        // Validate the YaRN compressed-layer RoPE freqs (`RopeScaling::Yarn::inv_freq`,
        // which the V4 builder feeds into the compress table) against an INDEPENDENT
        // reimplementation of the GA reference `precompute_freqs_cis` (model.py), with
        // the exact GA params: dim=rope_head_dim=64, base=compress_rope_theta=160000,
        // factor=16, original_seq_len=65536, beta_fast=32, beta_slow=1.
        let dim = 64usize;
        let base = 160000.0f64;
        let factor = 16.0f64;
        let max_seq = 65536.0f64;
        let (beta_fast, beta_slow) = (32.0f64, 1.0f64);
        let two_pi = 2.0 * std::f64::consts::PI;
        let find_corr = |rot: f64| dim as f64 * (max_seq / (rot * two_pi)).ln() / (2.0 * base.ln());
        let low = find_corr(beta_fast).floor().max(0.0);
        let high = find_corr(beta_slow).ceil().min(dim as f64 - 1.0);
        let denom = if (high - low).abs() < f64::EPSILON {
            0.001
        } else {
            high - low
        };
        let rs = RopeScaling::Yarn {
            factor,
            original_max_position_embeddings: max_seq,
            beta_fast,
            beta_slow,
            attention_factor: Some(1.0),
        };
        let mut max_err = 0.0f64;
        for i in 0..dim / 2 {
            let raw = 1.0 / base.powf(2.0 * i as f64 / dim as f64);
            let ramp = ((i as f64 - low) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            let reference = raw / factor * (1.0 - smooth) + raw * smooth;
            let got = rs.inv_freq(i, dim, base);
            max_err = max_err.max((reference - got).abs());
        }
        assert!(
            max_err < 1e-12,
            "YaRN inv_freq vs reference max_err={max_err:e}"
        );
        // Sliding-window layers (RopeScaling::None) must reproduce raw θ^(-2i/rd)
        // exactly (byte-identical to the pre-YaRN builder).
        let none = RopeScaling::None;
        for i in 0..dim / 2 {
            let raw = 10000.0f64.powf(-(2.0 * i as f64) / dim as f64);
            assert!((none.inv_freq(i, dim, 10000.0) - raw).abs() < 1e-15);
        }
    }

    #[test]
    fn deepseek_v4_dspark_heads_match_reference() {
        use rlx_runtime::{Device, Session};
        let rnd = |seed: f64, i: usize| -> f32 {
            let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
            (x - x.floor()) as f32 - 0.5
        };
        let (rows, vocab, mr, dim) = (3usize, 7usize, 4usize, 5usize);
        let ids = [2usize, 5, 0];
        let w1: Vec<f32> = (0..vocab * mr).map(|i| rnd(1.0, i)).collect(); // [vocab, mr]
        let w2: Vec<f32> = (0..vocab * mr).map(|i| rnd(2.0, i)).collect(); // [vocab, mr]
        let hidden: Vec<f32> = (0..rows * dim).map(|i| rnd(3.0, i)).collect();
        let proj: Vec<f32> = (0..(dim + mr)).map(|i| rnd(4.0, i)).collect(); // [1, dim+mr]
        // transposed weights for g.mm
        let mut w2_t = vec![0f32; mr * vocab]; // [mr, vocab]
        for v in 0..vocab {
            for k in 0..mr {
                w2_t[k * vocab + v] = w2[v * mr + k];
            }
        }
        let proj_t = proj.clone(); // [dim+mr, 1]

        let mut g = Graph::new("dspark_heads");
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        let tok = g.input("tok", Shape::new(&[rows], DType::I32));
        let hin = g.input("hidden", Shape::new(&[rows, dim], DType::F32));
        let pw1 = g.param("w1", Shape::new(&[vocab, mr], DType::F32));
        params.insert("w1".into(), w1.clone());
        let pw2t = g.param("w2t", Shape::new(&[mr, vocab], DType::F32));
        params.insert("w2t".into(), w2_t);
        let ppt = g.param("pt", Shape::new(&[dim + mr, 1], DType::F32));
        params.insert("pt".into(), proj_t);
        let (logits, embed) = build_dspark_markov_head(&mut g, pw1, pw2t, tok, rows, mr, vocab);
        let conf = build_dspark_confidence_head(&mut g, hin, embed, ppt, rows);
        g.set_outputs(vec![logits, conf]);
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, dd) in &params {
            compiled.set_param(n, dd);
        }
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let out = compiled.run(&[("tok", ids_f32.as_slice()), ("hidden", hidden.as_slice())]);
        let got_logits = &out[0];
        let got_conf = &out[1];

        // reference
        let mut max_le = 0f32;
        let mut max_ce = 0f32;
        for t in 0..rows {
            let tok = ids[t];
            // embed = w1[tok]; logit[v] = Σ_k embed[k]·w2[v,k]
            for v in 0..vocab {
                let mut s = 0f32;
                for k in 0..mr {
                    s += w1[tok * mr + k] * w2[v * mr + k];
                }
                max_le = max_le.max((s - got_logits[t * vocab + v]).abs());
            }
            // conf = Σ_j cat[j]·proj[j], cat = [hidden(dim), embed(mr)]
            let mut c = 0f32;
            for j in 0..dim {
                c += hidden[t * dim + j] * proj[j];
            }
            for k in 0..mr {
                c += w1[tok * mr + k] * proj[dim + k];
            }
            max_ce = max_ce.max((c - got_conf[t]).abs());
        }
        assert!(max_le < 1e-4, "markov logits err {max_le:e}");
        assert!(max_ce < 1e-4, "confidence err {max_ce:e}");
    }

    #[test]
    fn deepseek_v4_dspark_stage_builds_and_compiles() {
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::{Device, Session};
        // A synthetic in-memory loader: dense F32 for everything except the MoE
        // `switch_mlp` experts, which are served as minimal MLX-affine packed blobs
        // (8-bit, 1 group). Validates the full DSpark forward_spec wiring/shapes.
        struct DsMemLoader {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
        }
        impl WeightLoader for DsMemLoader {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
            fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
                // (n_expert, out, in) for the three switch_mlp projections.
                let dims = if key.ends_with("switch_mlp.gate_proj.weight")
                    || key.ends_with("switch_mlp.up_proj.weight")
                    || key.ends_with("switch_mlp.down_proj.weight")
                {
                    Some((4usize, 8usize, 8usize)) // n_expert, out, in
                } else {
                    None
                };
                let Some((ne, out, inn)) = dims else {
                    return Ok(None);
                };
                let f32_le =
                    |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
                Ok(Some(MlxPackedLinear {
                    w_q: vec![0u8; ne * out * inn],           // 8-bit codes (1 byte/elem)
                    scales: f32_le(&vec![0.01f32; ne * out]), // 1 group
                    biases: f32_le(&vec![0.0f32; ne * out]),
                    scheme: QuantScheme::MlxAffine {
                        bits: 8,
                        group_size: inn as u32,
                    },
                    out_shape: vec![out, inn],
                }))
            }
        }

        let (vocab, dim, hc, nh, hd, rd, ql) = (16, 8, 2, 2, 4, 2, 6);
        let (ngrp, olora, inter, ne, top) = (2usize, 3usize, 8usize, 4usize, 2usize);
        let (n_mtp, block, cache_len, n_targets) = (2usize, 3usize, 4usize, 2usize);
        let mix_hc = (2 + hc) * hc;
        let hcd = hc * dim;
        let dpg = nh * hd / ngrp;
        let spec = DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: 43,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            compress_ratios: vec![0; 43],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: 64,
            first_k_dense_replace: 0,
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            n_activated_experts: top,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 10.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: n_mtp,
            dspark_block_size: block,
            dspark_noise_token_id: 5,
            dspark_target_layer_ids: vec![40, 41],
            dspark_markov_rank: 4,
        };

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put =
            |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
                sd += 1.0;
                let n: usize = shape.iter().product();
                let data: Vec<f32> = (0..n)
                    .map(|i| {
                        let x = ((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453;
                        (x - x.floor()) as f32 - 0.5
                    })
                    .collect();
                t.insert(k, (data, shape));
            };
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);
        put(
            &mut t,
            "model.mtp.0.main_proj.weight".into(),
            vec![dim, dim * n_targets],
        );
        put(&mut t, "model.mtp.0.main_norm.weight".into(), vec![dim]);
        for s in 0..n_mtp {
            let p = format!("model.mtp.{s}");
            put(&mut t, format!("{p}.attn.wkv.weight"), vec![hd, dim]);
            put(&mut t, format!("{p}.attn.kv_norm.weight"), vec![hd]);
            put(&mut t, format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.attn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.attn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.attn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
            put(&mut t, format!("{p}.attn.q_norm.weight"), vec![ql]);
            put(&mut t, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
            put(&mut t, format!("{p}.attn.attn_sink"), vec![nh]);
            put(
                &mut t,
                format!("{p}.attn.wo_a.weight"),
                vec![ngrp * olora, dpg],
            );
            put(
                &mut t,
                format!("{p}.attn.wo_b.weight"),
                vec![dim, ngrp * olora],
            );
            put(&mut t, format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.ffn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.ffn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.ffn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.ffn.gate.weight"), vec![ne, dim]);
            put(
                &mut t,
                format!("{p}.ffn.gate.e_score_correction_bias"),
                vec![ne],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.gate_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.up_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.down_proj.weight"),
                vec![dim, inter],
            );
        }
        let last = n_mtp - 1;
        put(
            &mut t,
            format!("model.mtp.{last}.hc_head.fn"),
            vec![hc, hcd],
        );
        put(&mut t, format!("model.mtp.{last}.hc_head.scale"), vec![1]);
        put(&mut t, format!("model.mtp.{last}.hc_head.base"), vec![hc]);
        put(&mut t, format!("model.mtp.{last}.norm.weight"), vec![dim]);

        let mut loader = DsMemLoader { t };
        let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (graph, _params) =
            build_dspark_stage(&spec, &mut loader, block, cache_len, &mut packed)
                .expect("build_dspark_stage");
        assert_eq!(graph.outputs.len(), 2, "logits + head hidden");
        // Compile on CPU: validates every DSpark-specific op composes (main_proj mm,
        // window-cache RoPE, cat(cache,block), sliding sink-attention, o-LoRA, HC,
        // hc_head, lm_head) plus the reused sqrtsoftplus MoE grouped path.
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let _compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    }

    #[test]
    fn deepseek_v4_dspark_forward_head_and_accept() {
        let rnd = |seed: f64, i: usize| -> f32 {
            let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
            (x - x.floor()) as f32 - 0.5
        };
        let (block, vocab, mr, dim) = (3usize, 5usize, 2usize, 4usize);
        let bl: Vec<f32> = (0..block * vocab).map(|i| rnd(1.0, i)).collect();
        let w1: Vec<f32> = (0..vocab * mr).map(|i| rnd(2.0, i)).collect();
        let w2: Vec<f32> = (0..vocab * mr).map(|i| rnd(3.0, i)).collect();
        let hh: Vec<f32> = (0..block * dim).map(|i| rnd(4.0, i)).collect();
        let proj: Vec<f32> = (0..(dim + mr)).map(|i| rnd(5.0, i)).collect();
        let input_id = 1usize;
        let (ids, conf) =
            dspark_forward_head(&bl, &w1, &w2, &hh, &proj, input_id, block, vocab, mr, dim);

        // Independent reference of the autoregressive Markov loop.
        let mut ref_ids = vec![0usize; block + 1];
        ref_ids[0] = input_id;
        let mut ref_conf = vec![0f32; block];
        for i in 0..block {
            let tok = ref_ids[i];
            let mut best = (0usize, f32::MIN);
            for v in 0..vocab {
                let mut bias = 0f32;
                for k in 0..mr {
                    bias += w1[tok * mr + k] * w2[v * mr + k];
                }
                let l = bl[i * vocab + v] + bias;
                if l > best.1 {
                    best = (v, l);
                }
            }
            ref_ids[i + 1] = best.0;
            let mut c = 0f32;
            for j in 0..dim {
                c += hh[i * dim + j] * proj[j];
            }
            for k in 0..mr {
                c += w1[tok * mr + k] * proj[dim + k];
            }
            ref_conf[i] = c;
        }
        assert_eq!(ids, ref_ids);
        for i in 0..block {
            assert!((conf[i] - ref_conf[i]).abs() < 1e-5);
        }

        // Greedy accept: draft [1,7,4,9]; main argmax agrees on 7,4 then diverges.
        assert_eq!(dspark_greedy_accept(&[1, 7, 4, 9], &[7, 4, 2]), 2);
        assert_eq!(dspark_greedy_accept(&[1, 7, 4, 9], &[3, 4, 2]), 0); // first rejected
        assert_eq!(dspark_greedy_accept(&[1, 7, 4, 9], &[7, 4, 9]), 3); // all accepted
    }

    #[test]
    fn deepseek_v4_from_gguf_metadata_matches_real_header() {
        // Exact deepseek4.* metadata read from the real
        // bartowski/DeepSeek-V4-Flash-0731-GGUF shard-1 header.
        let ratios: Vec<usize> = (0..46)
            .map(|i| {
                if i < 2 || i >= 43 {
                    0
                } else if i % 2 == 0 {
                    4
                } else {
                    128
                }
            })
            .collect();
        let clamp: Vec<f64> = vec![10.0; 43];
        let meta = serde_json::json!({
            "general.architecture": "deepseek4",
            "deepseek4.block_count": 43,
            "deepseek4.context_length": 1048576u64,
            "deepseek4.embedding_length": 4096,
            "deepseek4.attention.head_count": 64,
            "deepseek4.attention.head_count_kv": 1,
            "deepseek4.rope.freq_base": 10000.0,
            "deepseek4.rope.dimension_count": 64,
            "deepseek4.attention.layer_norm_rms_epsilon": 1e-6,
            "deepseek4.expert_count": 256,
            "deepseek4.expert_used_count": 6,
            "deepseek4.expert_shared_count": 1,
            "deepseek4.attention.key_length": 512,
            "deepseek4.attention.value_length": 512,
            "deepseek4.attention.q_lora_rank": 1024,
            "deepseek4.attention.output_lora_rank": 1024,
            "deepseek4.attention.output_group_count": 8,
            "deepseek4.attention.sliding_window": 128,
            "deepseek4.expert_feed_forward_length": 2048,
            "deepseek4.expert_weights_scale": 1.5,
            "deepseek4.attention.indexer.head_count": 64,
            "deepseek4.attention.indexer.key_length": 128,
            "deepseek4.attention.indexer.top_k": 512,
            "deepseek4.attention.compress_ratios": ratios,
            "deepseek4.attention.compress_rope_freq_base": 160000.0,
            "deepseek4.hyper_connection.count": 4,
            "deepseek4.hyper_connection.sinkhorn_iterations": 20,
            "deepseek4.hyper_connection.epsilon": 1e-6,
            "deepseek4.hash_layer_count": 3,
            "deepseek4.swiglu_clamp_exp": clamp,
        });
        let s = DeepseekV4Spec::from_gguf_metadata(&meta, 129280).unwrap();
        assert_eq!(s.vocab_size, 129280);
        assert_eq!(s.dim, 4096);
        assert_eq!(s.n_layers, 43);
        assert_eq!(s.n_heads, 64);
        assert_eq!(s.head_dim, 512);
        assert_eq!(s.rope_head_dim, 64);
        assert_eq!(s.q_lora_rank, 1024);
        assert_eq!(s.o_lora_rank, 1024);
        assert_eq!(s.n_groups, 8);
        assert_eq!(s.index_head_dim, 128);
        assert_eq!(s.index_n_heads, 64);
        assert_eq!(s.index_topk, 512);
        assert_eq!(s.window_size, 128);
        assert_eq!(s.n_hash_layers, 3);
        assert_eq!(s.n_routed_experts, 256);
        assert_eq!(s.n_activated_experts, 6);
        assert_eq!(s.n_shared_experts, 1);
        assert_eq!(s.moe_intermediate_size, 2048);
        assert_eq!(s.hc_mult, 4);
        assert_eq!(s.hc_sinkhorn_iters, 20);
        assert_eq!(s.route_scale, 1.5);
        assert_eq!(s.rope_theta, 10000.0);
        assert_eq!(s.compress_rope_theta, 160000.0);
        assert_eq!(s.swiglu_limit, 10.0);
        assert_eq!(s.compress_ratios.len(), 43);
        assert_eq!(s.compress_ratios[42], 4);
        // GGUF is base-only: no DSpark/MTP.
        assert_eq!(s.n_mtp_layers, 0);
        assert_eq!(s.dspark_block_size, 0);

        // Name mapping (confirmed against the real GGUF header).
        assert_eq!(
            hf_key_to_deepseek4_gguf("model.layers.5.attn.wq_a.weight").as_deref(),
            Some("blk.5.attn_q_a.weight")
        );
        assert_eq!(
            hf_key_to_deepseek4_gguf("model.layers.7.ffn.switch_mlp.gate_proj.weight").as_deref(),
            Some("blk.7.ffn_gate_exps.weight")
        );
        assert_eq!(
            hf_key_to_deepseek4_gguf("model.layers.0.ffn.gate.tid2eid").as_deref(),
            Some("blk.0.ffn_gate_tid2eid.weight")
        );
        assert_eq!(
            hf_key_to_deepseek4_gguf("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(hf_key_to_deepseek4_gguf("nonexistent.key"), None);
    }

    #[test]
    fn deepseek_v4_ref_name_map() {
        // Verified against the real Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX index
        // (1328 builder keys → all resolve to real tensors).
        assert_eq!(
            hf_key_to_dsv4_ref("model.embed_tokens.weight").as_deref(),
            Some("embed.weight")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("lm_head.weight").as_deref(),
            Some("head.weight")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.norm.weight").as_deref(),
            Some("norm.weight")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.hc_head.fn").as_deref(),
            Some("hc_head_fn")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.0.attn.wq_a.weight").as_deref(),
            Some("layers.0.attn.wq_a.weight")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.5.attn_hc.fn").as_deref(),
            Some("layers.5.hc_attn_fn")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.5.ffn_hc.scale").as_deref(),
            Some("layers.5.hc_ffn_scale")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.2.attn.compressor.ape").as_deref(),
            Some("layers.2.attn.compressor.ape")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.2.attn.indexer.wq_b.weight").as_deref(),
            Some("layers.2.attn.indexer.wq_b.weight")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.3.ffn.gate.e_score_correction_bias").as_deref(),
            Some("layers.3.ffn.gate.bias")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.0.ffn.gate.tid2eid").as_deref(),
            Some("layers.0.ffn.gate.tid2eid")
        );
        assert_eq!(
            hf_key_to_dsv4_ref("model.layers.7.ffn.shared_experts.down_proj.weight").as_deref(),
            Some("layers.7.ffn.shared_experts.w2.weight")
        );
        assert_eq!(
            dsv4_ref_expert_key(7, 42, "gate_proj"),
            "layers.7.ffn.experts.42.w1.weight"
        );
        assert_eq!(
            dsv4_ref_expert_key(7, 42, "down_proj"),
            "layers.7.ffn.experts.42.w2.weight"
        );
        assert_eq!(
            dsv4_ref_expert_key(7, 42, "up_proj"),
            "layers.7.ffn.experts.42.w3.weight"
        );
        assert_eq!(hf_key_to_dsv4_ref("nonexistent"), None);
    }

    #[test]
    fn deepseek_v4_ref_loader_maps_and_stacks() {
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use std::sync::{Arc, Mutex};
        // Recording mock: logs each key the adapter requests of the inner loader,
        // and returns a fixed-size packed tensor so we can check the stacking.
        struct Mock {
            reqs: Arc<Mutex<Vec<String>>>,
            per_wq: usize,
            per_sc: usize,
        }
        impl WeightLoader for Mock {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.reqs.lock().unwrap().push(k.into());
                Ok((vec![0.0], vec![1]))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.reqs.lock().unwrap().push(k.into());
                Ok((vec![0.0], vec![1, 1]))
            }
            fn len(&self) -> usize {
                0
            }
            fn remaining_keys(&self) -> Vec<String> {
                vec![]
            }
            fn take_packed_mlx(&mut self, k: &str) -> Result<Option<MlxPackedLinear>> {
                self.reqs.lock().unwrap().push(k.into());
                Ok(Some(MlxPackedLinear {
                    w_q: vec![7u8; self.per_wq],
                    scales: vec![3u8; self.per_sc],
                    biases: vec![],
                    scheme: QuantScheme::MlxMxfp4 { group_size: 32 },
                    out_shape: vec![2048, 4096],
                }))
            }
        }
        let reqs = Arc::new(Mutex::new(Vec::new()));
        let mock = Mock {
            reqs: reqs.clone(),
            per_wq: 10,
            per_sc: 4,
        };
        let n_experts = 3;
        let mut ad = DsV4RefLoader::new(Box::new(mock), n_experts);

        // Name mapping on take.
        ad.take("model.layers.5.attn.q_norm.weight").unwrap();
        ad.take("model.embed_tokens.weight").unwrap();
        // Stacked routed experts: gathers per-expert w1 and concatenates.
        let p = ad
            .take_packed_mlx("model.layers.5.ffn.switch_mlp.gate_proj.weight")
            .unwrap()
            .unwrap();
        // A non-expert packed request name-maps straight through.
        ad.take_packed_mlx("model.layers.5.attn.wq_a.weight")
            .unwrap();

        let r = reqs.lock().unwrap();
        assert!(r.contains(&"layers.5.attn.q_norm.weight".to_string()));
        assert!(r.contains(&"embed.weight".to_string()));
        assert!(r.contains(&"layers.5.attn.wq_a.weight".to_string()));
        for e in 0..n_experts {
            assert!(
                r.contains(&format!("layers.5.ffn.experts.{e}.w1.weight")),
                "missing gathered expert {e}"
            );
        }
        // Stacked codes/scales = n_experts × per-expert.
        assert_eq!(p.w_q.len(), n_experts * 10);
        assert_eq!(p.scales.len(), n_experts * 4);
        assert!(matches!(p.scheme, QuantScheme::MlxMxfp4 { .. }));
    }

    #[test]
    fn cached_expert_source_budget() {
        use crate::resource::ResourceBudget;
        struct Counting;
        impl ExpertSource for Counting {
            fn fetch(&mut self, _il: usize, e: usize, _proj: &str) -> Result<Vec<f32>> {
                Ok(vec![e as f32; 4])
            }
        }
        // Budget caps 2 experts resident → cache holds 2×3 = 6 (layer,expert,proj) tensors.
        let budget = ResourceBudget {
            max_ram_bytes: None,
            max_resident_experts: Some(2),
        };
        let mut src = CachedExpertSource::from_budget(Counting, &budget, 256, 1 << 20, 0);
        // Cold-fetch experts 0,1,2 × 3 projs = 9 streams; cache bounds to 6, evicting e0.
        for e in 0..3 {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                src.fetch(0, e, p).unwrap();
            }
        }
        assert!(
            src.resident() <= 6,
            "cache bounded by budget, got {}",
            src.resident()
        );
        assert_eq!(src.misses, 9);
        assert_eq!(src.hits, 0);
        // Expert 2 is the most recent → all cached → hits (reuse across tokens).
        for p in ["gate_proj", "up_proj", "down_proj"] {
            src.fetch(0, 2, p).unwrap();
        }
        assert_eq!(src.hits, 3, "recent experts reused from RAM");
        // Expert 0 was LRU-evicted → re-streams.
        let before = src.misses;
        src.fetch(0, 0, "gate_proj").unwrap();
        assert_eq!(
            src.misses,
            before + 1,
            "evicted expert re-streams on demand"
        );
    }

    #[test]
    fn paged_expert_moe_matches() {
        use rlx_ir::op::Op;
        use rlx_runtime::{Device, Session};
        let f = DType::F32;
        let (h, n, top_k, inter, se_n) = (4usize, 8usize, 3usize, 6usize, 1usize);
        let spec = DeepseekSpec {
            vocab_size: 16,
            hidden_size: h,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            q_lora_rank: 0,
            absorbed_mla: false,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            intermediate_size: inter,
            moe_intermediate_size: inter,
            n_routed_experts: n,
            num_experts_per_tok: top_k,
            n_shared_experts: se_n,
            first_k_dense_replace: 0,
            routed_scaling_factor: 1.5,
            norm_topk_prob: true,
            sigmoid_gate: false,
            sqrtsoftplus_gate: true,
            swiglu_limit: 7.0,
            rope_theta: 10000.0,
            rope_scaling: RopeScaling::None,
            attn_score_scale: None,
            rope_neox: true,
            rms_norm_eps: 1e-6,
        };
        let rnd = |seed: usize| -> f32 {
            (((seed.wrapping_mul(2654435761)) % 1000) as f32) / 500.0 - 1.0
        };
        let h2d: Vec<f32> = (0..h).map(|i| rnd(i + 1)).collect();
        let router_w: Vec<f32> = (0..n * h).map(|i| rnd(i + 100)).collect(); // [n, h]
        let ebias: Vec<f32> = (0..n).map(|e| rnd(e + 50) * 0.1).collect();
        let mut gate: Vec<Vec<f32>> = Vec::new();
        let mut up: Vec<Vec<f32>> = Vec::new();
        let mut down: Vec<Vec<f32>> = Vec::new();
        for e in 0..n {
            gate.push((0..inter * h).map(|i| rnd(e * 1000 + i + 7)).collect());
            up.push((0..inter * h).map(|i| rnd(e * 1000 + i + 5000)).collect());
            down.push((0..h * inter).map(|i| rnd(e * 1000 + i + 9000)).collect());
        }
        let se = inter * se_n;
        let sg: Vec<f32> = (0..se * h).map(|i| rnd(i + 20000)).collect(); // [se, h]
        let su: Vec<f32> = (0..se * h).map(|i| rnd(i + 30000)).collect();
        let sd: Vec<f32> = (0..h * se).map(|i| rnd(i + 40000)).collect(); // [h, se]

        // (1) Selection == graph TopK (same ops as build_deepseek_moe_c → independent).
        let mut g = Graph::new("route_chk");
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        let h2d_in = g.input("h2d", Shape::new(&[1, h], f));
        let rw = g.param("rw", Shape::new(&[h, n], f)); // [h, n] = router_w transposed
        let mut rwt = vec![0f32; h * n];
        for i in 0..h {
            for e in 0..n {
                rwt[i * n + e] = router_w[e * h + i];
            }
        }
        params.insert("rw".into(), rwt);
        let logits = g.mm(h2d_in, rw);
        let sp = softplus_stable(&mut g, &mut params, logits, "chk");
        let scores_n = g.sqrt(sp);
        let eb = g.param("eb", Shape::new(&[n], f));
        params.insert("eb".into(), ebias.clone());
        let route = g.add(scores_n, eb);
        let top_idx = g.add_node(
            Op::TopK { k: top_k },
            vec![route],
            Shape::new(&[1, top_k], f),
        );
        g.set_outputs(vec![top_idx]);
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
        for (nm, dd) in &params {
            c.set_param(nm, dd);
        }
        let gout = c.run(&[("h2d", h2d.as_slice())]);
        let mut graph_top: Vec<usize> = gout[0].iter().map(|&x| x as usize).collect();
        graph_top.sort_unstable();
        let (ptop0, _) = paged_moe_route(&spec, &h2d, &router_w, Some(&ebias), None);
        let mut ptop = ptop0.clone();
        ptop.sort_unstable();
        assert_eq!(
            ptop, graph_top,
            "paged routing must select the same experts as the graph TopK"
        );

        // (2) paged_moe_forward == full reference, loading ONLY the active experts.
        struct HE {
            gate: Vec<Vec<f32>>,
            up: Vec<Vec<f32>>,
            down: Vec<Vec<f32>>,
            loads: usize,
        }
        impl ExpertSource for HE {
            fn fetch(&mut self, _il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
                self.loads += 1;
                Ok(match proj {
                    "gate_proj" => self.gate[e].clone(),
                    "up_proj" => self.up[e].clone(),
                    _ => self.down[e].clone(),
                })
            }
        }
        let mut src = HE {
            gate: gate.clone(),
            up: up.clone(),
            down: down.clone(),
            loads: 0,
        };
        let paged = paged_moe_forward(
            &spec,
            0,
            &h2d,
            &router_w,
            Some(&ebias),
            None,
            &mut src,
            Some((&sg, &su, &sd)),
        )
        .unwrap();
        assert_eq!(
            src.loads,
            top_k * 3,
            "expert paging must touch ONLY the {top_k} active experts (×3 projs)"
        );
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let (top, w) = paged_moe_route(&spec, &h2d, &router_w, Some(&ebias), None);
        let mut refo = vec![0f32; h];
        for (ki, &e) in top.iter().enumerate() {
            let mut glu = vec![0f32; inter];
            for (r, gr) in glu.iter_mut().enumerate() {
                let (mut a, mut b) = (0f32, 0f32);
                for i in 0..h {
                    a += gate[e][r * h + i] * h2d[i];
                    b += up[e][r * h + i] * h2d[i];
                }
                *gr = silu(a.min(7.0)) * b.clamp(-7.0, 7.0);
            }
            for (r, o) in refo.iter_mut().enumerate() {
                let mut s = 0f32;
                for i in 0..inter {
                    s += down[e][r * inter + i] * glu[i];
                }
                *o += w[ki] * s;
            }
        }
        let mut sglu = vec![0f32; se];
        for (r, gr) in sglu.iter_mut().enumerate() {
            let (mut a, mut b) = (0f32, 0f32);
            for i in 0..h {
                a += sg[r * h + i] * h2d[i];
                b += su[r * h + i] * h2d[i];
            }
            *gr = silu(a) * b;
        }
        for (r, o) in refo.iter_mut().enumerate() {
            let mut s = 0f32;
            for i in 0..se {
                s += sd[r * se + i] * sglu[i];
            }
            *o += s;
        }
        let err = paged
            .iter()
            .zip(&refo)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-4, "paged MoE output must match reference: {err:e}");

        // (3) HASH ROUTING (GA's first `n_hash_layers`): experts come from a `tid2eid`
        // row, not score top-k. Paging must select exactly those experts, weight them
        // by their ORIGINAL scores, and still touch only `top_k` of them.
        let hash_eids = vec![6usize, 1, 4];
        let (htop, hw) = paged_moe_route(&spec, &h2d, &router_w, Some(&ebias), Some(&hash_eids));
        assert_eq!(
            htop, hash_eids,
            "hash routing must select the tid2eid experts"
        );
        let mut src2 = HE {
            gate: gate.clone(),
            up: up.clone(),
            down: down.clone(),
            loads: 0,
        };
        let hpaged = paged_moe_forward(
            &spec,
            0,
            &h2d,
            &router_w,
            Some(&ebias),
            Some(&hash_eids),
            &mut src2,
            None,
        )
        .unwrap();
        assert_eq!(
            src2.loads,
            top_k * 3,
            "hash routing must also touch ONLY the {top_k} selected experts"
        );
        let mut href = vec![0f32; h];
        for (ki, &e) in hash_eids.iter().enumerate() {
            let mut glu = vec![0f32; inter];
            for (r, gr) in glu.iter_mut().enumerate() {
                let (mut a, mut b) = (0f32, 0f32);
                for i in 0..h {
                    a += gate[e][r * h + i] * h2d[i];
                    b += up[e][r * h + i] * h2d[i];
                }
                *gr = silu(a.min(7.0)) * b.clamp(-7.0, 7.0);
            }
            for (r, o) in href.iter_mut().enumerate() {
                let mut s = 0f32;
                for i in 0..inter {
                    s += down[e][r * inter + i] * glu[i];
                }
                *o += hw[ki] * s;
            }
        }
        let herr = hpaged
            .iter()
            .zip(&href)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            herr < 1e-4,
            "paged hash-routed MoE must match reference: {herr:e}"
        );

        // (4) BATCHED S3 (grouped scatter-reduce) == per-token S1 loop, and fetches
        // each active expert ONCE per batch (not once per token that picks it).
        let bsz = 5usize;
        let mut xb = vec![0f32; bsz * h];
        for b in 0..bsz {
            for i in 0..h {
                // Distinct token vectors so the routing differs across the batch.
                xb[b * h + i] =
                    h2d[i] * (0.7 + 0.1 * b as f32) + 0.01 * (b as f32 - i as f32).sin();
            }
        }
        // S1 reference: run the per-token paging path for each token, concatenate.
        let mut s1 = vec![0f32; bsz * h];
        let mut distinct: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for b in 0..bsz {
            let tok = &xb[b * h..b * h + h];
            let (tt, _) = paged_moe_route(&spec, tok, &router_w, Some(&ebias), None);
            distinct.extend(tt);
            let mut s = HE {
                gate: gate.clone(),
                up: up.clone(),
                down: down.clone(),
                loads: 0,
            };
            let o = paged_moe_forward(
                &spec,
                0,
                tok,
                &router_w,
                Some(&ebias),
                None,
                &mut s,
                Some((&sg, &su, &sd)),
            )
            .unwrap();
            s1[b * h..b * h + h].copy_from_slice(&o);
        }
        // S3 batched.
        let mut sb = HE {
            gate: gate.clone(),
            up: up.clone(),
            down: down.clone(),
            loads: 0,
        };
        let s3 = paged_moe_forward_batched(
            &spec,
            0,
            &xb,
            bsz,
            &router_w,
            Some(&ebias),
            None,
            &mut sb,
            Some((&sg, &su, &sd)),
        )
        .unwrap();
        let berr = s1
            .iter()
            .zip(&s3)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            berr < 1e-4,
            "batched S3 MoE must match per-token S1: {berr:e}"
        );
        // Each DISTINCT active expert fetched exactly once (×3 projs) — the paging win.
        assert_eq!(
            sb.loads,
            distinct.len() * 3,
            "batched paging must fetch each distinct active expert ONCE per batch, not per token"
        );
    }

    #[test]
    fn paged_grouped_moe_gpu_matches_host() {
        // The GPU grouped-MoE path (PagedGroupedMoe, on-device DequantGroupedMatMulMlx)
        // must equal the host S3 path (paged_moe_forward_batched) that dequants + GEMMs
        // on CPU — same synthetic MXFP4 per-expert weights, same routing. Runs on
        // Device::Cpu (the grouped op has a CPU impl); Metal/MLX is a drop-in device swap.
        use crate::weight_loader::MlxPackedLinear;
        use rlx_ir::quant::QuantScheme;
        let (h, inter, gs) = (64usize, 32usize, 32usize);
        let (n, top_k, batch) = (6usize, 2usize, 4usize);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let spec = DeepseekSpec {
            vocab_size: 16,
            hidden_size: h,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            q_lora_rank: 0,
            absorbed_mla: false,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            intermediate_size: inter,
            moe_intermediate_size: inter,
            n_routed_experts: n,
            num_experts_per_tok: top_k,
            n_shared_experts: 0,
            first_k_dense_replace: 0,
            routed_scaling_factor: 1.5,
            norm_topk_prob: true,
            sigmoid_gate: false,
            sqrtsoftplus_gate: true,
            swiglu_limit: 7.0,
            rope_theta: 10000.0,
            rope_scaling: RopeScaling::None,
            attn_score_scale: None,
            rope_neox: true,
            rms_norm_eps: 1e-6,
        };
        let rnd =
            |seed: usize| -> f32 { ((seed.wrapping_mul(2654435761) % 1000) as f32) / 500.0 - 1.0 };

        // Synthetic per-expert MXFP4 packed weights. gate/up: out=inter,in=h; down: out=h,in=inter.
        let mk = |e: usize, proj: u8, out: usize, inn: usize| -> MlxPackedLinear {
            let ng = inn / gs;
            let w_q: Vec<u8> = (0..out * (inn / 2))
                .map(|i| ((i * 31 + e * 17 + proj as usize * 7 + 3) % 256) as u8)
                .collect();
            // e8m0 scale bytes near 1.0 (0x7f == 2^0) so dequant magnitudes are sane.
            let scales: Vec<u8> = (0..out * ng)
                .map(|i| (0x7c + ((i + e) % 6)) as u8)
                .collect();
            MlxPackedLinear {
                w_q,
                scales,
                biases: Vec::new(),
                scheme,
                out_shape: vec![out, inn],
            }
        };
        // Build the full expert bank (used by both paths).
        let bank: Vec<[MlxPackedLinear; 3]> = (0..n)
            .map(|e| [mk(e, 0, inter, h), mk(e, 1, inter, h), mk(e, 2, h, inter)])
            .collect();

        // Packed source (GPU path) + dequantizing source (host reference) over the bank.
        struct PackedBank<'a>(&'a [[MlxPackedLinear; 3]]);
        impl PackedExpertSource for PackedBank<'_> {
            fn fetch_packed(
                &mut self,
                _il: usize,
                e: usize,
                proj: &str,
            ) -> Result<MlxPackedLinear> {
                let j = match proj {
                    "gate_proj" => 0,
                    "up_proj" => 1,
                    _ => 2,
                };
                Ok(self.0[e][j].clone())
            }
        }
        struct DequantBank<'a>(&'a [[MlxPackedLinear; 3]]);
        impl ExpertSource for DequantBank<'_> {
            fn fetch(&mut self, _il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
                let j = match proj {
                    "gate_proj" => 0,
                    "up_proj" => 1,
                    _ => 2,
                };
                dequant_packed_linear(&self.0[e][j])
            }
        }

        let x: Vec<f32> = (0..batch * h).map(|i| rnd(i + 1) * 0.5).collect();
        let router_w: Vec<f32> = (0..n * h).map(|i| rnd(i + 100)).collect();
        // Route each token identically for both paths (no shared expert here).
        let routes: Vec<Vec<(usize, f32)>> = (0..batch)
            .map(|b| {
                let (top, w) = paged_moe_route(&spec, &x[b * h..b * h + h], &router_w, None, None);
                top.into_iter().zip(w).collect()
            })
            .collect();

        // Host S3 reference.
        let mut href = DequantBank(&bank);
        let host =
            paged_moe_forward_batched(&spec, 0, &x, batch, &router_w, None, None, &mut href, None)
                .unwrap();

        // GPU grouped path (Device::Cpu here).
        let mut moe = PagedGroupedMoe::new(
            rlx_runtime::Device::Cpu,
            n,
            batch * top_k,
            h,
            inter,
            gs,
            spec.swiglu_limit,
            scheme,
        );
        let mut psrc = PackedBank(&bank);
        let gpu = moe.forward(0, &x, batch, &routes, &mut psrc).unwrap();

        let err = host
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = host.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
        assert!(
            err / mag < 1e-3,
            "GPU grouped MoE must match host S3: rel_err {:e} (abs {err:e}, mag {mag:e})",
            err / mag
        );
    }

    #[test]
    fn shared_expert_gpu_matches_host() {
        // The on-device dense shared expert must equal the host dense_swiglu_ffn.
        use rlx_runtime::Device;
        let (dim, se) = (16usize, 8usize);
        let lim = 7.0f32;
        let rnd = |s: usize| ((s.wrapping_mul(2654435761) % 1000) as f32) / 500.0 - 1.0;
        let x: Vec<f32> = (0..dim).map(|i| rnd(i + 1) * 0.5).collect();
        let sg: Vec<f32> = (0..se * dim).map(|i| rnd(i + 5) * 0.2).collect(); // [se, dim]
        let su: Vec<f32> = (0..se * dim).map(|i| rnd(i + 9) * 0.2).collect();
        let sd: Vec<f32> = (0..dim * se).map(|i| rnd(i + 13) * 0.2).collect(); // [dim, se]
        let host = dense_swiglu_ffn(&x, &sg, &su, &sd, lim);
        let mut g = SharedExpertGpu::new(Device::Cpu, dim, se, lim);
        let gpu = g.forward(&x, &sg, &su, &sd).unwrap();
        let err = host
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err < 1e-4,
            "GPU shared expert must match host dense_swiglu_ffn: err {err:e}"
        );
        assert_eq!(gpu.len(), dim);
    }

    #[test]
    fn paged_moe_fused_matches_dequant() {
        // FUSED dequant-matvec (reads packed codes) must equal the dequant-then-matmul
        // path bit-for-bit, on synthetic MXFP4 experts + shared.
        use crate::weight_loader::MlxPackedLinear;
        use rlx_ir::quant::QuantScheme;
        let (h, inter, gs, ne, topk) = (64usize, 32usize, 32usize, 6usize, 3usize);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let rnd = |s: usize| ((s.wrapping_mul(2654435761) % 1000) as f32) / 500.0 - 1.0;
        let mk = |e: usize, pj: u8, out: usize, inn: usize| -> MlxPackedLinear {
            let ng = inn / gs;
            MlxPackedLinear {
                w_q: (0..out * (inn / 2))
                    .map(|i| ((i * 31 + e * 17 + pj as usize * 7 + 3) % 256) as u8)
                    .collect(),
                scales: (0..out * ng)
                    .map(|i| (0x7c + ((i + e) % 6)) as u8)
                    .collect(),
                biases: Vec::new(),
                scheme,
                out_shape: vec![out, inn],
            }
        };
        let bank: Vec<[MlxPackedLinear; 3]> = (0..ne)
            .map(|e| [mk(e, 0, inter, h), mk(e, 1, inter, h), mk(e, 2, h, inter)])
            .collect();
        struct Bank<'a>(&'a [[MlxPackedLinear; 3]]);
        impl ExpertSource for Bank<'_> {
            fn fetch(&mut self, _il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
                let j = match proj {
                    "gate_proj" => 0,
                    "up_proj" => 1,
                    _ => 2,
                };
                dequant_packed_linear(&self.0[e][j])
            }
        }
        impl PackedExpertSource for Bank<'_> {
            fn fetch_packed(
                &mut self,
                _il: usize,
                e: usize,
                proj: &str,
            ) -> Result<MlxPackedLinear> {
                let j = match proj {
                    "gate_proj" => 0,
                    "up_proj" => 1,
                    _ => 2,
                };
                Ok(self.0[e][j].clone())
            }
        }
        let spec = DeepseekSpec {
            vocab_size: 0,
            hidden_size: h,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            q_lora_rank: 0,
            absorbed_mla: false,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            intermediate_size: inter,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            num_experts_per_tok: topk,
            n_shared_experts: 1,
            first_k_dense_replace: 0,
            routed_scaling_factor: 1.5,
            norm_topk_prob: true,
            sigmoid_gate: false,
            sqrtsoftplus_gate: true,
            swiglu_limit: 7.0,
            rope_theta: 10000.0,
            rope_scaling: RopeScaling::None,
            attn_score_scale: None,
            rope_neox: true,
            rms_norm_eps: 1e-6,
        };
        let x: Vec<f32> = (0..h).map(|i| rnd(i + 1) * 0.5).collect();
        let router: Vec<f32> = (0..ne * h).map(|i| rnd(i + 100)).collect();
        let (sg, su, sd): (Vec<f32>, Vec<f32>, Vec<f32>) = (
            (0..inter * h).map(|i| rnd(i + 5) * 0.1).collect(),
            (0..inter * h).map(|i| rnd(i + 9) * 0.1).collect(),
            (0..h * inter).map(|i| rnd(i + 13) * 0.1).collect(),
        );
        let a = paged_moe_forward(
            &spec,
            0,
            &x,
            &router,
            None,
            None,
            &mut Bank(&bank),
            Some((&sg, &su, &sd)),
        )
        .unwrap();
        let b = paged_moe_forward_fused(
            &spec,
            0,
            &x,
            &router,
            None,
            None,
            &mut Bank(&bank),
            Some((&sg, &su, &sd)),
        )
        .unwrap();
        let err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            err < 1e-4,
            "fused matvec MoE must match dequant path: err {err:e}"
        );
    }

    #[test]
    fn deepseek_v4_paged_decode_matches_resident() {
        // END-TO-END paged decode: a tiny 1-layer MoE V4 model decoded with the MoE
        // SPLIT OUT of the graph (attention in-graph, MoE host-side via
        // `paged_moe_forward`) must produce the SAME tokens as the fully-resident
        // `deepseek_v4_generate`. MXFP4 experts (bf16-exact scales) ⇒ the in-graph
        // grouped op and the host dequant agree bit-for-bit.
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::{Device, Session};
        let (vocab, dim, hc, nh, hd, rd, ql) = (16usize, 8, 2, 2, 4, 2, 6);
        let (ngrp, olora, inter, ne, nact, gs) = (2usize, 3, 8, 4, 2, 8);
        let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let spec = DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: 1,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            compress_ratios: vec![0],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: 64,
            first_k_dense_replace: 0,
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            n_activated_experts: nact,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: vec![0],
            dspark_markov_rank: 0,
        };

        // Dense f32 weights + stacked MXFP4 switch_mlp experts.
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                       k: String,
                       shape: Vec<usize>| {
            sd += 1.0;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    (((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453).fract() as f32
                        * 0.4
                        - 0.2
                })
                .collect();
            t.insert(k, (data, shape));
        };
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        let p = "model.layers.0";
        put(&mut t, format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
        put(&mut t, format!("{p}.attn_hc.scale"), vec![3]);
        put(&mut t, format!("{p}.attn_hc.base"), vec![mix_hc]);
        put(&mut t, format!("{p}.attn_norm.weight"), vec![dim]);
        put(&mut t, format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
        put(&mut t, format!("{p}.attn.q_norm.weight"), vec![ql]);
        put(&mut t, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
        put(&mut t, format!("{p}.attn.wkv.weight"), vec![hd, dim]);
        put(&mut t, format!("{p}.attn.kv_norm.weight"), vec![hd]);
        put(&mut t, format!("{p}.attn.attn_sink"), vec![nh]);
        put(
            &mut t,
            format!("{p}.attn.wo_a.weight"),
            vec![ngrp * olora, dpg],
        );
        put(
            &mut t,
            format!("{p}.attn.wo_b.weight"),
            vec![dim, ngrp * olora],
        );
        put(&mut t, format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
        put(&mut t, format!("{p}.ffn_hc.scale"), vec![3]);
        put(&mut t, format!("{p}.ffn_hc.base"), vec![mix_hc]);
        put(&mut t, format!("{p}.ffn_norm.weight"), vec![dim]);
        put(&mut t, format!("{p}.ffn.gate.weight"), vec![ne, dim]); // router [n_expert, dim]
        // Shared expert (n_shared=1, se_inter=inter): dense f32 SwiGLU.
        put(
            &mut t,
            format!("{p}.ffn.shared_experts.gate_proj.weight"),
            vec![inter, dim],
        );
        put(
            &mut t,
            format!("{p}.ffn.shared_experts.up_proj.weight"),
            vec![inter, dim],
        );
        put(
            &mut t,
            format!("{p}.ffn.shared_experts.down_proj.weight"),
            vec![dim, inter],
        );
        put(&mut t, "model.hc_head.fn".into(), vec![hc, hcd]);
        put(&mut t, "model.hc_head.scale".into(), vec![1]);
        put(&mut t, "model.hc_head.base".into(), vec![hc]);
        put(&mut t, "model.norm.weight".into(), vec![dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);

        // Stacked MXFP4 switch_mlp: gate/up [ne,inter,dim], down [ne,dim,inter].
        let mk_stack = |out: usize, inn: usize, salt: usize| -> (Vec<u8>, Vec<u8>) {
            let ng = inn / gs;
            let w_q: Vec<u8> = (0..ne * out * (inn / 2))
                .map(|i| ((i * 37 + salt * 11 + 5) % 256) as u8)
                .collect();
            let scales: Vec<u8> = (0..ne * out * ng)
                .map(|i| (0x7b + ((i + salt) % 4)) as u8)
                .collect(); // ~2^-4..2^-1
            (w_q, scales)
        };
        let mut packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)> = HashMap::new();
        packed.insert(format!("{p}.ffn.switch_mlp.gate_proj.weight"), {
            let (w, s) = mk_stack(inter, dim, 1);
            (w, s, inter, dim)
        });
        packed.insert(format!("{p}.ffn.switch_mlp.up_proj.weight"), {
            let (w, s) = mk_stack(inter, dim, 2);
            (w, s, inter, dim)
        });
        packed.insert(format!("{p}.ffn.switch_mlp.down_proj.weight"), {
            let (w, s) = mk_stack(dim, inter, 3);
            (w, s, dim, inter)
        });

        #[derive(Clone)]
        struct Mem {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
            packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)>,
            ne: usize,
            scheme: QuantScheme,
        }
        impl WeightLoader for Mem {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
            fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
                let Some((w_q, scales, out, inn)) = self.packed.get(key) else {
                    return Ok(None);
                };
                Ok(Some(MlxPackedLinear {
                    w_q: w_q.clone(),
                    scales: scales.clone(),
                    biases: Vec::new(),
                    scheme: self.scheme,
                    out_shape: vec![self.ne * out, *inn],
                }))
            }
        }
        let mem = Mem {
            t,
            packed: packed.clone(),
            ne,
            scheme,
        };
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let ds = v4_moe_spec(&spec);
        let router: Vec<f32> = mem.t[&format!("{p}.ffn.gate.weight")].0.clone(); // [ne, dim]
        let sg = mem.t[&format!("{p}.ffn.shared_experts.gate_proj.weight")]
            .0
            .clone();
        let su = mem.t[&format!("{p}.ffn.shared_experts.up_proj.weight")]
            .0
            .clone();
        let sd = mem.t[&format!("{p}.ffn.shared_experts.down_proj.weight")]
            .0
            .clone();
        struct Sliced<'a> {
            packed: &'a HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)>,
            gs: usize,
            scheme: QuantScheme,
        }
        impl Sliced<'_> {
            fn slice(&self, il: usize, e: usize, proj: &str) -> MlxPackedLinear {
                let key = format!("model.layers.{il}.ffn.switch_mlp.{proj}.weight");
                let (w_q, scales, out, inn) = &self.packed[&key];
                let (slab, sb) = (out * inn / 2, out * (inn / self.gs));
                MlxPackedLinear {
                    w_q: w_q[e * slab..(e + 1) * slab].to_vec(),
                    scales: scales[e * sb..(e + 1) * sb].to_vec(),
                    biases: Vec::new(),
                    scheme: self.scheme,
                    out_shape: vec![*out, *inn],
                }
            }
        }
        impl ExpertSource for Sliced<'_> {
            fn fetch(&mut self, il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
                dequant_packed_linear(&self.slice(il, e, proj))
            }
        }
        impl PackedExpertSource for Sliced<'_> {
            fn fetch_packed(&mut self, il: usize, e: usize, proj: &str) -> Result<MlxPackedLinear> {
                Ok(self.slice(il, e, proj))
            }
        }
        let token = 5u32;

        // ── Resident single decode step (MoE in-graph) → logits_res ──
        let mut pk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (rg, rp, rn) =
            build_deepseek_v4_decode(&spec, &mut mem.clone(), 0, 0, &mut pk).unwrap();
        let mut rc = Session::new(Device::Cpu).compile_with(rg, &opts);
        for (n, dd) in &rp {
            rc.set_param(n, dd);
        }
        for (n, (b, _s, _)) in &pk {
            rc.set_param_typed(n, b, DType::U8);
        }
        let rout = rc.run(&[("token_id", &[token as f32])]);
        let logits_res = rout[rn.iter().position(|x| x == "logits").unwrap()].clone();

        // ── Paged single decode step: attention in-graph, MoE host-side ──
        let mut pk2: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (pg, pp, pn) =
            build_deepseek_v4_decode_moe(&spec, &mut mem.clone(), 0, 0, &mut pk2, true).unwrap();
        let mut pc = Session::new(Device::Cpu).compile_with(pg, &opts);
        for (n, dd) in &pp {
            pc.set_param(n, dd);
        }
        // Pass 1: extract moe_in.0 (= xf) with moe_out.0 = zeros.
        let zeros = vec![0f32; dim];
        let tok = [token as f32];
        let p1 = pc.run(&[("token_id", &tok), ("moe_out.0", zeros.as_slice())]);
        let xf = p1[pn.iter().position(|x| *x == "moe_in.0").unwrap()].clone();

        // Compute the MoE with the graph's OWN kernel: a standalone
        // `build_deepseek_moe_ffn(xf)` graph — bit-exact to the resident inline MoE.
        // (This validates the SPLIT PLUMBING; using PagedGroupedMoe / a host MoE for
        // `moe_out` instead is the deployment path — its numerics are validated
        // separately in `paged_grouped_moe_gpu_matches_host` / `paged_expert_moe_matches`.)
        let mut gm = Graph::new("moe_only");
        let mut mp: HashMap<String, Vec<f32>> = HashMap::new();
        let mut mpk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let xf_in = gm.input("xf", Shape::new(&[1, 1, dim], DType::F32));
        let moe_node = build_deepseek_moe_ffn(
            &mut gm,
            &mut mp,
            &mut mpk,
            &mut mem.clone(),
            p,
            xf_in,
            1,
            1,
            &ds,
            None,
        )
        .unwrap();
        let moe_flat = gm.reshape_(moe_node, vec![1, dim as i64]);
        gm.set_outputs(vec![moe_flat]);
        let mut mc = Session::new(Device::Cpu).compile_with(gm, &opts);
        for (n, dd) in &mp {
            mc.set_param(n, dd);
        }
        for (n, (b, _s, _)) in &mpk {
            mc.set_param_typed(n, b, DType::U8);
        }
        let moe_out = mc.run(&[("xf", xf.as_slice())])[0].clone();

        // The HOST MoE (paged_moe_forward on sliced experts + shared) must reproduce
        // the in-graph MoE now that the expert codes are bound — this is the
        // deployment path (moe_fn = host/PagedGroupedMoe MoE).
        let mut src = Sliced {
            packed: &packed,
            gs,
            scheme,
        };
        let moe_host = paged_moe_forward(
            &ds,
            0,
            &xf,
            &router,
            None,
            None,
            &mut src,
            Some((&sg, &su, &sd)),
        )
        .unwrap();
        let merr = moe_out
            .iter()
            .zip(&moe_host)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mmag = moe_out
            .iter()
            .map(|v| v.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
        assert!(
            merr / mmag < 1e-3,
            "host paged_moe_forward must match in-graph MoE: rel_err {:e} (abs {merr:e})",
            merr / mmag
        );

        // Pass 2: feed the MoE result back → logits. Must EXACTLY equal resident.
        let p2 = pc.run(&[("token_id", &tok), ("moe_out.0", moe_out.as_slice())]);
        let logits_paged = p2[pn.iter().position(|x| x == "logits").unwrap()].clone();

        let err = logits_res
            .iter()
            .zip(&logits_paged)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = logits_res
            .iter()
            .map(|v| v.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
        assert!(
            logits_paged.iter().all(|x| x.is_finite()) && err / mag < 1e-4,
            "paged-split logits must match resident: rel_err {:e} (abs {err:e}, mag {mag:e})",
            err / mag
        );
        assert_eq!(logits_res.len(), vocab);

        // ── FULL end-to-end: paged generate driver == resident generate ──
        // Resident `deepseek_v4_generate` now binds the packed MoE codes (the fix);
        // the paged driver runs L+1 passes/token with the MoE filled by the graph's
        // own kernel ⇒ BYTE-IDENTICAL generated tokens (prompt prefill + greedy).
        let prompt: Vec<u32> = vec![2, 7, 1, 4];
        let resident_tok =
            deepseek_v4_generate(&spec, || Box::new(mem.clone()), Device::Cpu, &prompt, 4).unwrap();
        let paged_tok = deepseek_v4_generate_paged(
            &spec,
            || Box::new(mem.clone()),
            Device::Cpu,
            &prompt,
            4,
            |il, _tok, xfin| {
                // In-graph MoE for layer `il` (paged out of the decode graph, run here).
                let mut gm = Graph::new("moe_step");
                let mut mp: HashMap<String, Vec<f32>> = HashMap::new();
                let mut mpk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
                let xi = gm.input("xf", Shape::new(&[1, 1, dim], DType::F32));
                let lp = format!("model.layers.{il}");
                let mo = build_deepseek_moe_ffn(
                    &mut gm,
                    &mut mp,
                    &mut mpk,
                    &mut mem.clone(),
                    &lp,
                    xi,
                    1,
                    1,
                    &ds,
                    None,
                )?;
                let mof = gm.reshape_(mo, vec![1, dim as i64]);
                gm.set_outputs(vec![mof]);
                let mut c = Session::new(Device::Cpu).compile_with(gm, &opts);
                for (n, dd) in &mp {
                    c.set_param(n, dd);
                }
                for (n, (b, _s, _)) in &mpk {
                    c.set_param_typed(n, b, DType::U8);
                }
                Ok(c.run(&[("xf", xfin)])[0].clone())
            },
        )
        .unwrap();
        assert_eq!(
            resident_tok, paged_tok,
            "paged generate must match resident token-for-token"
        );
        assert_eq!(resident_tok.len(), 4);

        // ── Deployment path: paged generate with the HOST paged MoE (sliced experts
        // + shared) as moe_fn — the real expert-paging decode. Host MoE matches the
        // graph to ~1e-3, so greedy tokens track resident here too.
        let deploy_tok = deepseek_v4_generate_paged(
            &spec,
            || Box::new(mem.clone()),
            Device::Cpu,
            &prompt,
            4,
            |il, _tok, xfin| {
                let mut s = Sliced {
                    packed: &packed,
                    gs,
                    scheme,
                };
                paged_moe_forward(
                    &ds,
                    il,
                    xfin,
                    &router,
                    None,
                    None,
                    &mut s,
                    Some((&sg, &su, &sd)),
                )
            },
        )
        .unwrap();
        assert_eq!(
            deploy_tok, resident_tok,
            "host-paged-MoE decode must match resident tokens"
        );

        // ── The clean deployment API: PagedGroupedMoe (GPU grouped kernel, routed)
        // + dense_swiglu_ffn (shared) as moe_fn. Routed part uses the SAME grouped op
        // as the graph ⇒ token-exact vs resident. ──
        let mut gmoe = PagedGroupedMoe::new(
            Device::Cpu,
            ne,
            nact,
            dim,
            inter,
            gs,
            spec.swiglu_limit,
            scheme,
        );
        let grouped_tok = deepseek_v4_generate_paged(
            &spec,
            || Box::new(mem.clone()),
            Device::Cpu,
            &prompt,
            4,
            |il, _tok, xfin| {
                let (top, w) = paged_moe_route(&ds, xfin, &router, None, None);
                let routes = vec![top.into_iter().zip(w).collect::<Vec<_>>()];
                let mut s = Sliced {
                    packed: &packed,
                    gs,
                    scheme,
                };
                let routed = gmoe.forward(il, xfin, 1, &routes, &mut s)?;
                let shared = dense_swiglu_ffn(xfin, &sg, &su, &sd, spec.swiglu_limit);
                Ok(routed.iter().zip(&shared).map(|(a, b)| a + b).collect())
            },
        )
        .unwrap();
        assert_eq!(
            grouped_tok, resident_tok,
            "GPU-grouped-kernel paged decode must match resident tokens"
        );

        // ── COMPILE-ONCE paged decode: V4Decoder::new_stage_paged compiles the fixed
        // backbone ONCE, step_paged fills the MoE host-side per token (no per-token
        // recompile) ⇒ same tokens as the recompile-per-token generate (and resident). ──
        let (mw, mc) = (spec.window_size, 1usize);
        let mut pdec = V4Decoder::new_stage_paged(
            &spec,
            &mut mem.clone(),
            0..spec.n_layers,
            true,
            true,
            mw,
            mc,
            Device::Cpu,
        )
        .unwrap();
        let mut moe_graph = |il: usize, _t: u32, xf: &[f32]| -> Result<Vec<f32>> {
            let mut gm = Graph::new("moe_co");
            let mut mp: HashMap<String, Vec<f32>> = HashMap::new();
            let mut mpk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let xi = gm.input("xf", Shape::new(&[1, 1, dim], DType::F32));
            let lp = format!("model.layers.{il}");
            let mo = build_deepseek_moe_ffn(
                &mut gm,
                &mut mp,
                &mut mpk,
                &mut mem.clone(),
                &lp,
                xi,
                1,
                1,
                &ds,
                None,
            )?;
            let mf = gm.reshape_(mo, vec![1, dim as i64]);
            gm.set_outputs(vec![mf]);
            let mut c = Session::new(Device::Cpu).compile_with(gm, &opts);
            for (n, dd) in &mp {
                c.set_param(n, dd);
            }
            for (n, (b, _s, _)) in &mpk {
                c.set_param_typed(n, b, DType::U8);
            }
            Ok(c.run(&[("xf", xf)])[0].clone())
        };
        let argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                )
                .0 as u32
        };
        let mut logits = vec![0f32; vocab];
        for &tk in &prompt {
            logits = pdec.step_paged(tk, &mut moe_graph).unwrap();
        }
        let mut co_tok = Vec::new();
        for _ in 0..4 {
            let nx = argmax(&logits);
            co_tok.push(nx);
            logits = pdec.step_paged(nx, &mut moe_graph).unwrap();
        }
        assert_eq!(
            co_tok, resident_tok,
            "compile-once paged decode (V4Decoder::step_paged) must match resident tokens"
        );

        // ── PER-LAYER paged decode (O(L) compile + O(2L) attention, one stage/layer)
        // must also match resident token-for-token. ──
        let lw_tok = deepseek_v4_generate_paged_layerwise(
            &spec,
            || Box::new(mem.clone()),
            Device::Cpu,
            mw,
            mc,
            &prompt,
            4,
            &mut moe_graph,
        )
        .unwrap();
        assert_eq!(
            lw_tok, resident_tok,
            "per-layer paged decode must match resident tokens"
        );

        // ── PAGED VERIFY BLOCK (Gap 1): build_deepseek_v4_verify_block_moe(paged) with
        // moe_out filled host-side must MATCH the resident verify block bit-for-bit —
        // enabling speculative decode on the paged model. ──
        {
            let block = 3usize;
            let half = (rd / 2).max(1);
            let (mut cosb, mut sinb, mut sininvb) = (
                vec![0f32; block * half],
                vec![0f32; block * half],
                vec![0f32; block * half],
            );
            for pp in 0..block {
                for i in 0..half {
                    let fr = (spec.rope_theta as f64).powf(-(2.0 * i as f64) / rd as f64);
                    let (s, c) = (pp as f64 * fr).sin_cos();
                    cosb[pp * half + i] = c as f32;
                    sinb[pp * half + i] = s as f32;
                    sininvb[pp * half + i] = -s as f32;
                }
            }
            let nkeys = mw + block;
            let mut mask = vec![-1e30f32; block * nkeys];
            for r in 0..block {
                for j in 0..=r {
                    mask[r * nkeys + mw + j] = 0.0;
                }
            }
            let vtok: Vec<u32> = (0..block).map(|i| ((i * 3 + 1) % vocab) as u32).collect();
            let toks_f: Vec<f32> = vtok.iter().map(|&x| x as f32).collect();
            let base_inputs = || -> Vec<(String, Vec<f32>)> {
                let mut o = vec![
                    ("token_id".to_string(), toks_f.clone()),
                    ("cos".into(), cosb.clone()),
                    ("sin".into(), sinb.clone()),
                    ("sininv".into(), sininvb.clone()),
                ];
                for il in 0..spec.n_layers {
                    o.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                    o.push((format!("mask.{il}"), mask.clone()));
                }
                o
            };
            // Resident verify block.
            let mut rvk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (rvg, rvp, rvn) =
                build_deepseek_v4_verify_block(&spec, &mut mem.clone(), block, mw, mc, &mut rvk)
                    .unwrap();
            let mut rvc = Session::new(Device::Cpu).compile_with(rvg, &opts);
            for (n, dd) in &rvp {
                rvc.set_param(n, dd);
            }
            for (n, (b, _, _)) in &rvk {
                rvc.set_param_typed(n, b, DType::U8);
            }
            let ri = base_inputs();
            let rin: Vec<(&str, &[f32])> =
                ri.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
            let rvout = rvc.run(&rin);
            let rlog = rvout[rvn.iter().position(|x| x == "logits").unwrap()].clone();
            // Paged verify block: moe_in/moe_out split, host fills the block MoE
            // (build_deepseek_moe_ffn seq=block — the same op the resident graph runs).
            let mut pvk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (pvg, pvp, pvn) = build_deepseek_v4_verify_block_moe(
                &spec,
                &mut mem.clone(),
                block,
                mw,
                mc,
                &mut pvk,
                true,
            )
            .unwrap();
            let mut pvc = Session::new(Device::Cpu).compile_with(pvg, &opts);
            for (n, dd) in &pvp {
                pvc.set_param(n, dd);
            }
            for (n, (b, _, _)) in &pvk {
                pvc.set_param_typed(n, b, DType::U8);
            }
            let ml: Vec<usize> = (spec.first_k_dense_replace..spec.n_layers).collect();
            let mut moe_out: HashMap<usize, Vec<f32>> =
                ml.iter().map(|&il| (il, vec![0f32; block * dim])).collect();
            let block_moe = |il: usize, xf: &[f32]| -> Vec<f32> {
                let mut gm = Graph::new("moe_blk");
                let mut mp: HashMap<String, Vec<f32>> = HashMap::new();
                let mut mpk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
                let xi = gm.input("xf", Shape::new(&[1, block, dim], DType::F32));
                let lp = format!("model.layers.{il}");
                let mo = build_deepseek_moe_ffn(
                    &mut gm,
                    &mut mp,
                    &mut mpk,
                    &mut mem.clone(),
                    &lp,
                    xi,
                    1,
                    block,
                    &ds,
                    None,
                )
                .unwrap();
                let mf = gm.reshape_(mo, vec![block as i64, dim as i64]);
                gm.set_outputs(vec![mf]);
                let mut c = Session::new(Device::Cpu).compile_with(gm, &opts);
                for (n, dd) in &mp {
                    c.set_param(n, dd);
                }
                for (n, (b, _s, _)) in &mpk {
                    c.set_param_typed(n, b, DType::U8);
                }
                c.run(&[("xf", xf)])[0].clone()
            };
            let run_pv = |pvc: &mut rlx_runtime::CompiledGraph,
                          mo: &HashMap<usize, Vec<f32>>|
             -> Vec<Vec<f32>> {
                let mut o = base_inputs();
                for &il in &ml {
                    o.push((format!("moe_out.{il}"), mo[&il].clone()));
                }
                let inp: Vec<(&str, &[f32])> =
                    o.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
                pvc.run(&inp)
            };
            for &il in &ml {
                let pv = run_pv(&mut pvc, &moe_out);
                let xf = pv[pvn
                    .iter()
                    .position(|x| x == &format!("moe_in.{il}"))
                    .unwrap()]
                .clone();
                moe_out.insert(il, block_moe(il, &xf));
            }
            let pv = run_pv(&mut pvc, &moe_out);
            let plog = &pv[pvn.iter().position(|x| x == "logits").unwrap()];
            let err = rlog
                .iter()
                .zip(plog)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                err < 1e-3,
                "paged verify block must match resident verify block: {err:e}"
            );
        }

        // ── HASH ROUTING: first n_hash_layers pick experts from gate.tid2eid[token],
        // not score top-k. Resident graph does Gather(tid2eid, token); the paged
        // moe_fn does hash_route_experts(tid2eid, token) → paged_moe_forward. ──
        let mut spec_h = spec.clone();
        spec_h.n_hash_layers = 1; // layer 0 becomes hash-routed
        let mut tid2eid = vec![0f32; vocab * nact];
        for tk in 0..vocab {
            for j in 0..nact {
                tid2eid[tk * nact + j] = ((tk + j * 2 + 1) % ne) as f32;
            }
        }
        let mut mem_h = mem.clone();
        mem_h.t.insert(
            format!("{p}.ffn.gate.tid2eid"),
            (tid2eid.clone(), vec![vocab, nact]),
        );
        let res_h =
            deepseek_v4_generate(&spec_h, || Box::new(mem_h.clone()), Device::Cpu, &prompt, 4)
                .unwrap();
        let paged_h = deepseek_v4_generate_paged(
            &spec_h,
            || Box::new(mem_h.clone()),
            Device::Cpu,
            &prompt,
            4,
            |il, tok, xfin| {
                let hash =
                    (il < spec_h.n_hash_layers).then(|| hash_route_experts(&tid2eid, nact, tok));
                let mut s = Sliced {
                    packed: &packed,
                    gs,
                    scheme,
                };
                paged_moe_forward(
                    &ds,
                    il,
                    xfin,
                    &router,
                    None,
                    hash.as_deref(),
                    &mut s,
                    Some((&sg, &su, &sd)),
                )
            },
        )
        .unwrap();
        assert_eq!(
            res_h, paged_h,
            "hash-routed paged decode must match resident tokens"
        );
    }

    #[test]
    fn deepseek_v4_layerwise_verify_matches_monolithic() {
        // O(L) LAYERWISE verify: composing single-layer `build_deepseek_v4_verify_block_stage`
        // graphs (threading `hidden_out → hidden_in`) must be BIT-IDENTICAL to the
        // monolithic whole-model verify — proving the O(L²)→O(L) transform is lossless.
        // 2 layers, sliding-only (the paged moe_in/moe_out split is validated separately
        // by the Gap-1 block in `deepseek_v4_paged_decode_matches_resident`).
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::{Device, Session};
        let (vocab, dim, hc, nh, hd, rd, ql) = (16usize, 8, 2, 2, 4, 2, 6);
        let (ngrp, olora, inter, ne, nact, gs) = (2usize, 3, 8, 4, 2, 8);
        let nl = 2usize;
        let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let spec = DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: nl,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            compress_ratios: vec![0; nl],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: 64,
            first_k_dense_replace: 0,
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            n_activated_experts: nact,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: vec![0],
            dspark_markov_rank: 0,
        };
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                       k: String,
                       shape: Vec<usize>| {
            sd += 1.0;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    (((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453).fract() as f32
                        * 0.4
                        - 0.2
                })
                .collect();
            t.insert(k, (data, shape));
        };
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        let mk_stack = |out: usize, inn: usize, salt: usize| -> (Vec<u8>, Vec<u8>) {
            let ng = inn / gs;
            let w_q: Vec<u8> = (0..ne * out * (inn / 2))
                .map(|i| ((i * 37 + salt * 11 + 5) % 256) as u8)
                .collect();
            let scales: Vec<u8> = (0..ne * out * ng)
                .map(|i| (0x7b + ((i + salt) % 4)) as u8)
                .collect();
            (w_q, scales)
        };
        let mut packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)> = HashMap::new();
        for li in 0..nl {
            let p = format!("model.layers.{li}");
            put(&mut t, format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.attn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.attn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.attn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
            put(&mut t, format!("{p}.attn.q_norm.weight"), vec![ql]);
            put(&mut t, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
            put(&mut t, format!("{p}.attn.wkv.weight"), vec![hd, dim]);
            put(&mut t, format!("{p}.attn.kv_norm.weight"), vec![hd]);
            put(&mut t, format!("{p}.attn.attn_sink"), vec![nh]);
            put(
                &mut t,
                format!("{p}.attn.wo_a.weight"),
                vec![ngrp * olora, dpg],
            );
            put(
                &mut t,
                format!("{p}.attn.wo_b.weight"),
                vec![dim, ngrp * olora],
            );
            put(&mut t, format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.ffn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.ffn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.ffn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.ffn.gate.weight"), vec![ne, dim]);
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.gate_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.up_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.down_proj.weight"),
                vec![dim, inter],
            );
            packed.insert(format!("{p}.ffn.switch_mlp.gate_proj.weight"), {
                let (w, s) = mk_stack(inter, dim, 1);
                (w, s, inter, dim)
            });
            packed.insert(format!("{p}.ffn.switch_mlp.up_proj.weight"), {
                let (w, s) = mk_stack(inter, dim, 2);
                (w, s, inter, dim)
            });
            packed.insert(format!("{p}.ffn.switch_mlp.down_proj.weight"), {
                let (w, s) = mk_stack(dim, inter, 3);
                (w, s, dim, inter)
            });
        }
        put(&mut t, "model.hc_head.fn".into(), vec![hc, hcd]);
        put(&mut t, "model.hc_head.scale".into(), vec![1]);
        put(&mut t, "model.hc_head.base".into(), vec![hc]);
        put(&mut t, "model.norm.weight".into(), vec![dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);

        #[derive(Clone)]
        struct Mem {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
            packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)>,
            ne: usize,
            scheme: QuantScheme,
        }
        impl WeightLoader for Mem {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
            fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
                let Some((w_q, scales, out, inn)) = self.packed.get(key) else {
                    return Ok(None);
                };
                Ok(Some(MlxPackedLinear {
                    w_q: w_q.clone(),
                    scales: scales.clone(),
                    biases: Vec::new(),
                    scheme: self.scheme,
                    out_shape: vec![self.ne * out, *inn],
                }))
            }
        }
        let mem = Mem {
            t,
            packed,
            ne,
            scheme,
        };
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let (block, mw, mc_win) = (3usize, 8usize, 0usize);
        let half = (rd / 2).max(1);
        let (mut cosb, mut sinb, mut sininvb) = (
            vec![0f32; block * half],
            vec![0f32; block * half],
            vec![0f32; block * half],
        );
        for pp in 0..block {
            for i in 0..half {
                let fr = (spec.rope_theta as f64).powf(-(2.0 * i as f64) / rd as f64);
                let (s, c) = (pp as f64 * fr).sin_cos();
                cosb[pp * half + i] = c as f32;
                sinb[pp * half + i] = s as f32;
                sininvb[pp * half + i] = -s as f32;
            }
        }
        let nkeys = mw + block;
        let mut mask = vec![-1e30f32; block * nkeys];
        for r in 0..block {
            for j in 0..=r {
                mask[r * nkeys + mw + j] = 0.0;
            }
        }
        let vtok: Vec<u32> = (0..block).map(|i| ((i * 3 + 1) % vocab) as u32).collect();
        let toks_f: Vec<f32> = vtok.iter().map(|&x| x as f32).collect();

        // ── MONOLITHIC verify (all layers, MoE in-graph) ──
        let mut mk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (mg, mpm, mn) =
            build_deepseek_v4_verify_block(&spec, &mut mem.clone(), block, mw, mc_win, &mut mk)
                .unwrap();
        let mut mcs = Session::new(Device::Cpu).compile_with(mg, &opts);
        for (n, dd) in &mpm {
            mcs.set_param(n, dd);
        }
        for (n, (b, _, _)) in &mk {
            mcs.set_param_typed(n, b, DType::U8);
        }
        let mono_inputs = {
            let mut o = vec![
                ("token_id".to_string(), toks_f.clone()),
                ("cos".into(), cosb.clone()),
                ("sin".into(), sinb.clone()),
                ("sininv".into(), sininvb.clone()),
            ];
            for il in 0..nl {
                o.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                o.push((format!("mask.{il}"), mask.clone()));
            }
            o
        };
        let mi: Vec<(&str, &[f32])> = mono_inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let mono_out = mcs.run(&mi);
        let mono_log = mono_out[mn.iter().position(|x| x == "logits").unwrap()].clone();

        // ── LAYERWISE: one single-layer stage per layer, thread hidden_out → hidden_in ──
        let mut hidden: Option<Vec<f32>> = None;
        let mut layer_log: Vec<f32> = Vec::new();
        for il in 0..nl {
            let mut sk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (sg2, sp2, sn2) = build_deepseek_v4_verify_block_stage(
                &spec,
                &mut mem.clone(),
                block,
                mw,
                mc_win,
                &mut sk,
                il..il + 1,
                false,
            )
            .unwrap();
            let mut sc = Session::new(Device::Cpu).compile_with(sg2, &opts);
            for (n, dd) in &sp2 {
                sc.set_param(n, dd);
            }
            for (n, (b, _, _)) in &sk {
                sc.set_param_typed(n, b, DType::U8);
            }
            let mut o: Vec<(String, Vec<f32>)> = vec![
                ("cos".into(), cosb.clone()),
                ("sin".into(), sinb.clone()),
                ("sininv".into(), sininvb.clone()),
                (format!("wcache.{il}"), vec![0f32; mw * hd]),
                (format!("mask.{il}"), mask.clone()),
            ];
            if il == 0 {
                o.push(("token_id".into(), toks_f.clone()));
            } else {
                o.push(("hidden_in".into(), hidden.clone().unwrap()));
            }
            let inp: Vec<(&str, &[f32])> =
                o.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
            let out = sc.run(&inp);
            if il + 1 == nl {
                layer_log = out[sn2.iter().position(|x| x == "logits").unwrap()].clone();
            } else {
                hidden = Some(out[sn2.iter().position(|x| x == "hidden_out").unwrap()].clone());
            }
        }
        let err = mono_log
            .iter()
            .zip(&layer_log)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            layer_log.iter().all(|x| x.is_finite()) && err < 1e-4,
            "layerwise verify must match monolithic verify: max_abs_err {err:e}"
        );
        assert_eq!(layer_log.len(), block * vocab);
    }

    #[test]
    fn deepseek_v4_fusion_ablation() {
        // ABLATION STUDY: toggle each graph-fusion flag (#1 hc-gate, #2 sink-attn,
        // #4 rope), build the DSV4 decode graph, count RUNTIME ops (the identified
        // bottleneck = kernel launches), run on CPU (must stay bit-exact vs baseline),
        // and time it. Prints a ranked table. 2-layer synthetic model, rd<hd so the
        // partial-rope path (#4) is exercised.
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::{Device, Session};
        use std::time::Instant;
        // Representative dims: hc=4 (real HC width), hd=64/nh=8 so attention matmuls
        // are non-trivial (exposes whether a fused kernel actually beats BLAS), rd<hd.
        let (vocab, dim, hc, nh, hd, rd, ql) = (32usize, 128, 4, 8, 64, 32, 64);
        let (ngrp, olora, inter, ne, nact, gs) = (2usize, 16, 64, 4, 2, 8);
        let nl = 4usize;
        let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let spec = DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: nl,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            compress_ratios: vec![0; nl],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: 64,
            first_k_dense_replace: 0,
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            n_activated_experts: nact,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: vec![0],
            dspark_markov_rank: 0,
        };
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                       k: String,
                       shape: Vec<usize>| {
            sd += 1.0;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    (((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453).fract() as f32
                        * 0.4
                        - 0.2
                })
                .collect();
            t.insert(k, (data, shape));
        };
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        let mk_stack = |out: usize, inn: usize, salt: usize| -> (Vec<u8>, Vec<u8>) {
            let ng = inn / gs;
            let w_q: Vec<u8> = (0..ne * out * (inn / 2))
                .map(|i| ((i * 37 + salt * 11 + 5) % 256) as u8)
                .collect();
            let scales: Vec<u8> = (0..ne * out * ng)
                .map(|i| (0x7b + ((i + salt) % 4)) as u8)
                .collect();
            (w_q, scales)
        };
        let mut packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)> = HashMap::new();
        for li in 0..nl {
            let p = format!("model.layers.{li}");
            put(&mut t, format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.attn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.attn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.attn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
            put(&mut t, format!("{p}.attn.q_norm.weight"), vec![ql]);
            put(&mut t, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
            put(&mut t, format!("{p}.attn.wkv.weight"), vec![hd, dim]);
            put(&mut t, format!("{p}.attn.kv_norm.weight"), vec![hd]);
            put(&mut t, format!("{p}.attn.attn_sink"), vec![nh]);
            put(
                &mut t,
                format!("{p}.attn.wo_a.weight"),
                vec![ngrp * olora, dpg],
            );
            put(
                &mut t,
                format!("{p}.attn.wo_b.weight"),
                vec![dim, ngrp * olora],
            );
            put(&mut t, format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.ffn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.ffn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.ffn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.ffn.gate.weight"), vec![ne, dim]);
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.gate_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.up_proj.weight"),
                vec![inter, dim],
            );
            put(
                &mut t,
                format!("{p}.ffn.shared_experts.down_proj.weight"),
                vec![dim, inter],
            );
            packed.insert(format!("{p}.ffn.switch_mlp.gate_proj.weight"), {
                let (w, s) = mk_stack(inter, dim, 1);
                (w, s, inter, dim)
            });
            packed.insert(format!("{p}.ffn.switch_mlp.up_proj.weight"), {
                let (w, s) = mk_stack(inter, dim, 2);
                (w, s, inter, dim)
            });
            packed.insert(format!("{p}.ffn.switch_mlp.down_proj.weight"), {
                let (w, s) = mk_stack(dim, inter, 3);
                (w, s, dim, inter)
            });
        }
        put(&mut t, "model.hc_head.fn".into(), vec![hc, hcd]);
        put(&mut t, "model.hc_head.scale".into(), vec![1]);
        put(&mut t, "model.hc_head.base".into(), vec![hc]);
        put(&mut t, "model.norm.weight".into(), vec![dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);

        #[derive(Clone)]
        struct Mem {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
            packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)>,
            ne: usize,
            scheme: QuantScheme,
        }
        impl WeightLoader for Mem {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
            fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
                let Some((w_q, scales, out, inn)) = self.packed.get(key) else {
                    return Ok(None);
                };
                Ok(Some(MlxPackedLinear {
                    w_q: w_q.clone(),
                    scales: scales.clone(),
                    biases: Vec::new(),
                    scheme: self.scheme,
                    out_shape: vec![self.ne * out, *inn],
                }))
            }
        }
        let mem = Mem {
            t,
            packed,
            ne,
            scheme,
        };
        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );

        // Op classifier: total nodes, runtime ops (exclude Param/Input/Constant),
        // Reduce+Div (the Sinkhorn signature), and fused Custom ops.
        fn classify(g: &Graph) -> (usize, usize, usize, usize) {
            use rlx_ir::Op;
            let (mut total, mut runtime, mut reducediv, mut custom) = (0, 0, 0, 0);
            for n in g.nodes() {
                total += 1;
                match &n.op {
                    Op::Param { .. } | Op::Input { .. } | Op::Constant { .. } => {}
                    Op::Custom { .. } => {
                        runtime += 1;
                        custom += 1;
                    }
                    Op::Reduce { .. } => {
                        runtime += 1;
                        reducediv += 1;
                    }
                    Op::Binary(rlx_ir::op::BinaryOp::Div) => {
                        runtime += 1;
                        reducediv += 1;
                    }
                    _ => runtime += 1,
                }
            }
            (total, runtime, reducediv, custom)
        }

        let configs: &[(&str, &[&str])] = &[
            ("baseline", &[]),
            ("#1 hcgate", &["RLX_OPT_HCGATE"]),
            ("#2 sinkattn", &["RLX_OPT_SINKATTN"]),
            ("#4 rope", &["RLX_OPT_ROPE"]),
            (
                "all(1+2+4)",
                &["RLX_OPT_HCGATE", "RLX_OPT_SINKATTN", "RLX_OPT_ROPE"],
            ),
        ];
        const REPS: usize = 30;
        let token = 5u32;
        let mut base_logits: Option<Vec<f32>> = None;
        let mut rows: Vec<(String, usize, usize, usize, usize, f64, f32)> = Vec::new();
        for (name, flags) in configs {
            rlx_ir::env::clear_overrides();
            for f in *flags {
                rlx_ir::env::set(*f, "1");
            }
            let mut pk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (g, params, names) =
                build_deepseek_v4_decode(&spec, &mut mem.clone(), 0, 0, &mut pk).unwrap();
            let (total, runtime, reducediv, custom) = classify(&g);
            let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
            for (n, dd) in &params {
                c.set_param(n, dd);
            }
            for (n, (b, _, _)) in &pk {
                c.set_param_typed(n, b, DType::U8);
            }
            let li = names.iter().position(|x| x == "logits").unwrap();
            let mut logits = c.run(&[("token_id", &[token as f32])])[li].clone(); // warm
            let t0 = Instant::now();
            for _ in 0..REPS {
                logits = c.run(&[("token_id", &[token as f32])])[li].clone();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
            let err = match &base_logits {
                Some(b) => logits
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
                None => 0.0,
            };
            if base_logits.is_none() {
                base_logits = Some(logits.clone());
            }
            rows.push((name.to_string(), total, runtime, reducediv, custom, ms, err));
        }
        rlx_ir::env::clear_overrides();

        let base_rt = rows[0].2 as f64;
        println!(
            "\n══ DSV4 fusion ablation ({nl}-layer synthetic decode, hc={hc} hd={hd}, per token) ══"
        );
        println!(
            "{:<14} {:>6} {:>8} {:>8} {:>7} {:>9} {:>10} {:>10}",
            "config", "nodes", "runtime", "R+Div", "custom", "time/tok", "vs base", "max|Δlogit|"
        );
        for (name, total, runtime, reducediv, custom, ms, err) in &rows {
            let redux = 100.0 * (1.0 - *runtime as f64 / base_rt);
            println!(
                "{:<14} {:>6} {:>8} {:>8} {:>7} {:>7.3}ms {:>9.1}% {:>10.2e}",
                name, total, runtime, reducediv, custom, ms, redux, err
            );
        }
        println!("(runtime = ops excluding Param/Input/Constant; R+Div = Sinkhorn signature)\n");

        for (name, _, _, _, _, _, err) in &rows {
            assert!(
                *err < 2e-3,
                "config '{name}' must match baseline: max|Δ| {err:e}"
            );
        }
        // The #1 hc-gate fusion must remove the most runtime ops of the single-flag runs.
        let rt = |n: &str| rows.iter().find(|r| r.0 == n).unwrap().2;
        assert!(
            rt("#1 hcgate") < rt("baseline"),
            "hc-gate must cut runtime ops"
        );
        assert!(rt("all(1+2+4)") <= rt("#1 hcgate"), "all-on cuts the most");
    }

    #[test]
    fn deepseek_v4_dspark_draft_produces_valid_block() {
        // DRAFTER WIRING: `deepseek_v4_dspark_draft` builds+runs the DSpark stage
        // (`mtp.0.*` sub-model over main_hidden) and applies the Markov+confidence head
        // to emit a block of drafts. It must produce exactly `n_draft` in-vocab token
        // ids from a synthetic 1-stage drafter — validating the full stage→head wiring
        // (input/output plumbing, packed-code binding, output slicing). Draft *quality*
        // is irrelevant to correctness (the verifier rejects wrong guesses); the
        // lossless verify/accept loop is validated by `deepseek_v4_decode_matches_prefill`.
        use crate::weight_loader::{MlxPackedLinear, WeightLoader};
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::Device;
        let (vocab, dim, hc, nh, hd, rd, ql) = (16usize, 8, 2, 2, 4, 2, 6);
        let (ngrp, olora, inter, ne, nact, gs) = (2usize, 3, 8, 4, 2, 8);
        let (block, mrank) = (4usize, 3usize);
        let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: gs as u32,
        };
        let spec = DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: 1,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            compress_ratios: vec![0],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: 64,
            first_k_dense_replace: 0,
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: ne,
            n_activated_experts: nact,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: 1,
            dspark_block_size: block,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: vec![0],
            dspark_markov_rank: mrank,
        };
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                       k: String,
                       shape: Vec<usize>| {
            sd += 1.0;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    (((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453).fract() as f32
                        * 0.4
                        - 0.2
                })
                .collect();
            t.insert(k, (data, shape));
        };
        // Shared embed + lm_head.
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);
        let mp = "model.mtp.0";
        // main projection (n_targets=1 ⇒ [dim,dim]) + norm.
        put(&mut t, format!("{mp}.main_proj.weight"), vec![dim, dim]);
        put(&mut t, format!("{mp}.main_norm.weight"), vec![dim]);
        // HC-wrapped attention.
        put(&mut t, format!("{mp}.attn_hc.fn"), vec![mix_hc, hcd]);
        put(&mut t, format!("{mp}.attn_hc.scale"), vec![3]);
        put(&mut t, format!("{mp}.attn_hc.base"), vec![mix_hc]);
        put(&mut t, format!("{mp}.attn_norm.weight"), vec![dim]);
        put(&mut t, format!("{mp}.attn.wq_a.weight"), vec![ql, dim]);
        put(&mut t, format!("{mp}.attn.q_norm.weight"), vec![ql]);
        put(&mut t, format!("{mp}.attn.wq_b.weight"), vec![nh * hd, ql]);
        put(&mut t, format!("{mp}.attn.wkv.weight"), vec![hd, dim]);
        put(&mut t, format!("{mp}.attn.kv_norm.weight"), vec![hd]);
        put(&mut t, format!("{mp}.attn.attn_sink"), vec![nh]);
        put(
            &mut t,
            format!("{mp}.attn.wo_a.weight"),
            vec![ngrp * olora, dpg],
        );
        put(
            &mut t,
            format!("{mp}.attn.wo_b.weight"),
            vec![dim, ngrp * olora],
        );
        // HC-wrapped MoE FFN.
        put(&mut t, format!("{mp}.ffn_hc.fn"), vec![mix_hc, hcd]);
        put(&mut t, format!("{mp}.ffn_hc.scale"), vec![3]);
        put(&mut t, format!("{mp}.ffn_hc.base"), vec![mix_hc]);
        put(&mut t, format!("{mp}.ffn_norm.weight"), vec![dim]);
        put(&mut t, format!("{mp}.ffn.gate.weight"), vec![ne, dim]);
        put(
            &mut t,
            format!("{mp}.ffn.shared_experts.gate_proj.weight"),
            vec![inter, dim],
        );
        put(
            &mut t,
            format!("{mp}.ffn.shared_experts.up_proj.weight"),
            vec![inter, dim],
        );
        put(
            &mut t,
            format!("{mp}.ffn.shared_experts.down_proj.weight"),
            vec![dim, inter],
        );
        // Final reduce + heads.
        put(&mut t, format!("{mp}.hc_head.fn"), vec![hc, hcd]);
        put(&mut t, format!("{mp}.hc_head.scale"), vec![1]);
        put(&mut t, format!("{mp}.hc_head.base"), vec![hc]);
        put(&mut t, format!("{mp}.norm.weight"), vec![dim]);
        put(
            &mut t,
            format!("{mp}.markov_head.markov_w1.weight"),
            vec![vocab, mrank],
        );
        put(
            &mut t,
            format!("{mp}.markov_head.markov_w2.weight"),
            vec![vocab, mrank],
        );
        put(
            &mut t,
            format!("{mp}.confidence_head.proj.weight"),
            vec![dim + mrank],
        );

        let mk_stack = |out: usize, inn: usize, salt: usize| -> (Vec<u8>, Vec<u8>) {
            let ng = inn / gs;
            let w_q: Vec<u8> = (0..ne * out * (inn / 2))
                .map(|i| ((i * 37 + salt * 11 + 5) % 256) as u8)
                .collect();
            let scales: Vec<u8> = (0..ne * out * ng)
                .map(|i| (0x7b + ((i + salt) % 4)) as u8)
                .collect();
            (w_q, scales)
        };
        let mut packed_w: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)> = HashMap::new();
        packed_w.insert(format!("{mp}.ffn.switch_mlp.gate_proj.weight"), {
            let (w, s) = mk_stack(inter, dim, 1);
            (w, s, inter, dim)
        });
        packed_w.insert(format!("{mp}.ffn.switch_mlp.up_proj.weight"), {
            let (w, s) = mk_stack(inter, dim, 2);
            (w, s, inter, dim)
        });
        packed_w.insert(format!("{mp}.ffn.switch_mlp.down_proj.weight"), {
            let (w, s) = mk_stack(dim, inter, 3);
            (w, s, dim, inter)
        });

        #[derive(Clone)]
        struct Mem {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
            packed: HashMap<String, (Vec<u8>, Vec<u8>, usize, usize)>,
            ne: usize,
            scheme: QuantScheme,
        }
        impl WeightLoader for Mem {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
            fn take_packed_mlx(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
                let Some((w_q, scales, out, inn)) = self.packed.get(key) else {
                    return Ok(None);
                };
                Ok(Some(MlxPackedLinear {
                    w_q: w_q.clone(),
                    scales: scales.clone(),
                    biases: Vec::new(),
                    scheme: self.scheme,
                    out_shape: vec![self.ne * out, *inn],
                }))
            }
        }
        let mut mem = Mem {
            t,
            packed: packed_w,
            ne,
            scheme,
        };
        // main_hidden [cache_len, dim*n_targets=dim] — some prior context.
        let cache_len = 5usize;
        let main_hidden: Vec<f32> = (0..cache_len * dim)
            .map(|i| ((i as f64 * 0.37).sin() * 0.2) as f32)
            .collect();
        let n_draft = block - 1;
        let drafts = deepseek_v4_dspark_draft(
            &spec,
            &mut mem,
            &mut HashMap::new(),
            Device::Cpu,
            &main_hidden,
            3u32,
            n_draft,
        )
        .unwrap();
        assert_eq!(drafts.len(), n_draft, "drafter must return n_draft tokens");
        assert!(
            drafts.iter().all(|&x| (x as usize) < vocab),
            "all drafted ids must be in-vocab: {drafts:?}"
        );
    }

    #[test]
    fn hash_route_experts_selects_tid2eid_row() {
        // tid2eid[token] row = the token's expert ids (f32-encoded).
        let (vocab, top_k) = (5usize, 3usize);
        let mut t2e = vec![0f32; vocab * top_k];
        for tk in 0..vocab {
            for j in 0..top_k {
                t2e[tk * top_k + j] = ((tk * 7 + j) % 11) as f32;
            }
        }
        for tk in 0..vocab {
            let got = hash_route_experts(&t2e, top_k, tk as u32);
            let want: Vec<usize> = (0..top_k).map(|j| (tk * 7 + j) % 11).collect();
            assert_eq!(got, want, "token {tk}");
        }
        // Out-of-range token → zeros (graceful).
        assert_eq!(hash_route_experts(&t2e, top_k, 99), vec![0, 0, 0]);
    }

    #[test]
    fn deepseek_v4_decode_matches_prefill() {
        use crate::weight_loader::WeightLoader;
        use rlx_ir::quant::QuantScheme;
        use rlx_runtime::{Device, Session};
        // Simple dense-F32 in-memory loader (re-created per build since `take`
        // consumes). All-sliding (ratio 0) + all-dense FFN so no MoE packing.
        #[derive(Clone)]
        struct Mem {
            t: HashMap<String, (Vec<f32>, Vec<usize>)>,
        }
        impl WeightLoader for Mem {
            fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                self.t.get(k).cloned().ok_or_else(|| anyhow!("missing {k}"))
            }
            fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
                let (d, s) = self.take(k)?;
                let (r, c) = (s[0], s[1]);
                let mut o = vec![0f32; d.len()];
                for i in 0..r {
                    for j in 0..c {
                        o[j * r + i] = d[i * c + j];
                    }
                }
                Ok((o, vec![c, r]))
            }
            fn len(&self) -> usize {
                self.t.len()
            }
            fn remaining_keys(&self) -> Vec<String> {
                self.t.keys().cloned().collect()
            }
        }

        let (vocab, dim, hc, nh, hd, rd, ql) = (16, 8, 2, 2, 4, 2, 6);
        let (ngrp, olora, inter, nl) = (2usize, 3usize, 10usize, 3usize);
        let mix_hc = (2 + hc) * hc;
        let hcd = hc * dim;
        let dpg = nh * hd / ngrp;
        let mk_spec = |win: usize| DeepseekV4Spec {
            vocab_size: vocab,
            dim,
            n_layers: nl,
            hc_mult: hc,
            n_heads: nh,
            head_dim: hd,
            rope_head_dim: rd,
            q_lora_rank: ql,
            n_groups: ngrp,
            o_lora_rank: olora,
            // layer0 sliding, layer1 non-overlap compress (ratio 2), layer2 overlap (ratio 4)
            compress_ratios: vec![0, 2, 4],
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
            window_size: win,
            first_k_dense_replace: nl, // all dense FFN
            n_hash_layers: 0,
            moe_intermediate_size: inter,
            n_routed_experts: 4,
            n_activated_experts: 2,
            n_shared_experts: 1,
            intermediate_size: inter,
            route_scale: 1.5,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            original_seq_len: 0,
            rope_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_mtp_layers: 0,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: vec![nl - 1], // last layer → DSpark main_hidden
            dspark_markov_rank: 0,
        };

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut sd = 0.0f64;
        let mut put =
            |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
                sd += 1.0;
                let n: usize = shape.iter().product();
                let data: Vec<f32> = (0..n)
                    .map(|i| {
                        let x = ((i as f64 + 1.0) * (sd + 1.3) * 12.9898).sin() * 43758.5453;
                        ((x - x.floor()) as f32 - 0.5) * 0.4
                    })
                    .collect();
                t.insert(k, (data, shape));
            };
        let spec_ratios = [0usize, 2, 4]; // must match compress_ratios in mk_spec
        put(&mut t, "model.embed_tokens.weight".into(), vec![vocab, dim]);
        for il in 0..nl {
            let p = format!("model.layers.{il}");
            put(&mut t, format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.attn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.attn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.attn_norm.weight"), vec![dim]);
            put(&mut t, format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
            put(&mut t, format!("{p}.attn.q_norm.weight"), vec![ql]);
            put(&mut t, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
            put(&mut t, format!("{p}.attn.wkv.weight"), vec![hd, dim]);
            put(&mut t, format!("{p}.attn.kv_norm.weight"), vec![hd]);
            put(&mut t, format!("{p}.attn.attn_sink"), vec![nh]);
            put(
                &mut t,
                format!("{p}.attn.wo_a.weight"),
                vec![ngrp * olora, dpg],
            );
            put(
                &mut t,
                format!("{p}.attn.wo_b.weight"),
                vec![dim, ngrp * olora],
            );
            // Compressor weights (coff=2 for overlap ratio-4, else 1).
            let lratio = spec_ratios[il];
            if lratio > 0 {
                let coff = if lratio == 4 { 2 } else { 1 };
                put(
                    &mut t,
                    format!("{p}.attn.compressor.wkv.weight"),
                    vec![coff * hd, dim],
                );
                put(
                    &mut t,
                    format!("{p}.attn.compressor.wgate.weight"),
                    vec![coff * hd, dim],
                );
                put(
                    &mut t,
                    format!("{p}.attn.compressor.ape"),
                    vec![lratio, coff * hd],
                );
                put(&mut t, format!("{p}.attn.compressor.norm.weight"), vec![hd]);
            }
            put(&mut t, format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
            put(&mut t, format!("{p}.ffn_hc.scale"), vec![3]);
            put(&mut t, format!("{p}.ffn_hc.base"), vec![mix_hc]);
            put(&mut t, format!("{p}.ffn_norm.weight"), vec![dim]);
            put(
                &mut t,
                format!("{p}.ffn.gate_proj.weight"),
                vec![inter, dim],
            );
            put(&mut t, format!("{p}.ffn.up_proj.weight"), vec![inter, dim]);
            put(
                &mut t,
                format!("{p}.ffn.down_proj.weight"),
                vec![dim, inter],
            );
        }
        put(&mut t, "model.hc_head.fn".into(), vec![hc, hcd]);
        put(&mut t, "model.hc_head.scale".into(), vec![1]);
        put(&mut t, "model.hc_head.base".into(), vec![hc]);
        put(&mut t, "model.norm.weight".into(), vec![dim]);
        put(&mut t, "lm_head.weight".into(), vec![vocab, dim]);

        let opts = crate::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let seq = 8usize; // ≥ 8 so the ratio-4 overlap layer fires twice (windows 0 & 1)
        let ids: Vec<u32> = (0..seq).map(|i| ((i * 5 + 3) % vocab) as u32).collect();

        // ── Prefill: full sequence in one pass ──
        let mut pk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (pg, pp) =
            build_deepseek_v4_prefill(&mk_spec(64), &mut Mem { t: t.clone() }, seq, &mut pk)
                .unwrap();
        let mut pc = Session::new(Device::Cpu).compile_with(pg, &opts);
        for (n, dd) in &pp {
            pc.set_param(n, dd);
        }
        let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
        let prefill = pc
            .run(&[("input_ids", ids_f32.as_slice())])
            .into_iter()
            .next()
            .unwrap();

        // ── Decode: one token at a time, threading per-layer KV + compressor state ──
        let ratios = [0usize, 2, 4];
        let coffs: Vec<usize> = ratios.iter().map(|&r| if r == 4 { 2 } else { 1 }).collect();
        let mut win_ring: Vec<Vec<f32>> = vec![Vec::new(); nl]; // window KV (kvnew history)
        let mut compcache: Vec<Vec<f32>> = vec![Vec::new(); nl]; // compressed KV (compress layers)
        let mut partial_ck: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut partial_cg: Vec<Vec<f32>> = vec![Vec::new(); nl];
        // Overlap (ratio-4) previous-window state: window 0 = 0 kv / -1e30 score.
        let mut prev_kv: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut prev_score: Vec<Vec<f32>> = vec![Vec::new(); nl];
        for il in 0..nl {
            if ratios[il] == 4 {
                prev_kv[il] = vec![0f32; ratios[il] * coffs[il] * hd];
                prev_score[il] = vec![-1e30f32; ratios[il] * coffs[il] * hd];
            }
        }
        for tpos in 0..seq {
            let mut dk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (dg, dp, onames) = build_deepseek_v4_decode(
                &mk_spec(64),
                &mut Mem { t: t.clone() },
                tpos,
                tpos,
                &mut dk,
            )
            .unwrap();
            let mut dc = Session::new(Device::Cpu).compile_with(dg, &opts);
            for (n, dd) in &dp {
                dc.set_param(n, dd);
            }
            // Assemble the inputs the host must supply this step (owned so the
            // &[f32] refs outlive `run`).
            let mut owned: Vec<(String, Vec<f32>)> =
                vec![("token_id".into(), vec![ids[tpos] as f32])];
            for il in 0..nl {
                if tpos > 0 {
                    owned.push((format!("kvcache.{il}"), win_ring[il].clone()));
                }
                let ratio = ratios[il];
                if ratio > 0 {
                    let firing = (tpos + 1) % ratio == 0;
                    if tpos / ratio > 0 {
                        owned.push((format!("compcache.{il}"), compcache[il].clone()));
                    }
                    if firing && ratio > 1 {
                        owned.push((format!("partial_ck.{il}"), partial_ck[il].clone()));
                        owned.push((format!("partial_cg.{il}"), partial_cg[il].clone()));
                    }
                    if firing && ratio == 4 {
                        owned.push((format!("prev_kv.{il}"), prev_kv[il].clone()));
                        owned.push((format!("prev_score.{il}"), prev_score[il].clone()));
                    }
                }
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let out = dc.run(&inputs);
            let get = |name: &str| -> Vec<f32> {
                let i = onames
                    .iter()
                    .position(|x| x == name)
                    .unwrap_or_else(|| panic!("no output {name}"));
                out[i].clone()
            };
            let dlog = get("logits");
            let pslice = &prefill[tpos * vocab..(tpos + 1) * vocab];
            let max_err = dlog
                .iter()
                .zip(pslice)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_err < 2e-3,
                "decode vs prefill logits mismatch at pos {tpos}: max_err {max_err:e}"
            );
            // Thread the state forward.
            for il in 0..nl {
                win_ring[il].extend_from_slice(&get(&format!("kvnew.{il}")));
                let ratio = ratios[il];
                if ratio > 0 {
                    if (tpos + 1) % ratio == 0 {
                        compcache[il].extend_from_slice(&get(&format!("comp.{il}")));
                        // Overlap: the just-completed window becomes the next `prev`.
                        if ratio == 4 {
                            let mut ckw = partial_ck[il].clone();
                            ckw.extend_from_slice(&get(&format!("ck.{il}")));
                            let mut cgw = partial_cg[il].clone();
                            cgw.extend_from_slice(&get(&format!("cg.{il}")));
                            prev_kv[il] = ckw;
                            prev_score[il] = cgw;
                        }
                        partial_ck[il].clear();
                        partial_cg[il].clear();
                    } else {
                        partial_ck[il].extend_from_slice(&get(&format!("ck.{il}")));
                        partial_cg[il].extend_from_slice(&get(&format!("cg.{il}")));
                    }
                }
            }
        }

        // ── Generate loop: KV-cache greedy == naive full-prefill greedy ──
        // window=3 (< sequence) exercises the sliding-window RING; ratios [0,2,4]
        // cover all layer types incl the overlap compressor's prev-window shift.
        let gspec = mk_spec(3);
        let prompt = [ids[0], ids[1]];
        let n_new = 3usize;
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                )
                .0 as u32
        };
        // Naive greedy: re-prefill the full growing sequence each step.
        let mut nids = prompt.to_vec();
        for _ in 0..n_new {
            let mut nk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (ng, np) =
                build_deepseek_v4_prefill(&gspec, &mut Mem { t: t.clone() }, nids.len(), &mut nk)
                    .unwrap();
            let mut nc = Session::new(Device::Cpu).compile_with(ng, &opts);
            for (n, dd) in &np {
                nc.set_param(n, dd);
            }
            let nf: Vec<f32> = nids.iter().map(|&x| x as f32).collect();
            let nl_out = nc
                .run(&[("input_ids", nf.as_slice())])
                .into_iter()
                .next()
                .unwrap();
            let next = argmax(&nl_out[(nids.len() - 1) * vocab..nids.len() * vocab]);
            nids.push(next);
        }
        let naive_gen = nids[prompt.len()..].to_vec();
        // KV-cache generate (fresh loader per step).
        let tmap = t.clone();
        let kv_gen = deepseek_v4_generate(
            &gspec,
            || Box::new(Mem { t: tmap.clone() }) as Box<dyn WeightLoader>,
            Device::Cpu,
            &prompt,
            n_new,
        )
        .unwrap();
        assert_eq!(
            kv_gen, naive_gen,
            "KV-cache generate must match naive full-prefill greedy"
        );

        // ── Compile-once V4Decoder == per-step generate (one graph, one load) ──
        // max_win = window-1 = 2; max_comp bounds the compressed cache.
        let mut dec =
            V4Decoder::new(&mk_spec(3), &mut Mem { t: t.clone() }, 2, 4, Device::Cpu).unwrap();
        let once_gen = dec.generate(&prompt, n_new);
        assert_eq!(
            once_gen, kv_gen,
            "compile-once V4Decoder must match per-step generate"
        );

        // ── DSpark interface: the main model emits `main_hidden` (mean-over-streams
        // hidden at the target layer) — what the speculative drafter consumes. ──
        let mut mhdec =
            V4Decoder::new(&mk_spec(3), &mut Mem { t: t.clone() }, 2, 4, Device::Cpu).unwrap();
        mhdec.step(prompt[0]);
        let mh = mhdec
            .main_hidden()
            .expect("main_hidden emitted at the target layer");
        assert_eq!(mh.len(), dim, "main_hidden is [dim] per target layer");
        assert!(mh.iter().all(|v| v.is_finite()), "main_hidden finite");

        // ── KV rollback: snapshot → decode → restore → re-decode is identical (the
        // speculative-decode reject primitive). ──
        let mut rdec =
            V4Decoder::new(&mk_spec(3), &mut Mem { t: t.clone() }, 2, 4, Device::Cpu).unwrap();
        rdec.step(ids[0]);
        rdec.step(ids[1]);
        let snap = rdec.snapshot();
        let l1 = rdec.step(ids[2]);
        let l1b = rdec.step(ids[3]);
        rdec.restore(snap);
        let l2 = rdec.step(ids[2]);
        let l2b = rdec.step(ids[3]);
        let e1 = l1
            .iter()
            .zip(&l2)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let e2 = l1b
            .iter()
            .zip(&l2b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            e1 < 1e-4 && e2 < 1e-4,
            "restore must reproduce state: {e1:e} {e2:e}"
        );

        // ── seq=block BATCH VERIFY graph == `block` sequential seq=1 decodes (the
        // DSpark accelerator; spec-decode is lossless ⇒ identical logits). Sliding
        // config at pos 0 so no compressor fires inside the block ⇒ EXACT. ──
        {
            let mut ss = mk_spec(3);
            ss.compress_ratios = vec![0, 0, 0]; // all sliding
            let block = 3usize;
            let (mw, mc) = (2usize, 4usize);
            let vtok: Vec<u32> = (0..block).map(|i| ((i * 3 + 1) % vocab) as u32).collect();
            // Reference: the exact seq=1 decoder, one token at a time.
            let mut rdec =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let reflog: Vec<Vec<f32>> = vtok.iter().map(|&tk| rdec.step(tk)).collect();
            // Batch verify graph (whole block in one pass).
            let mut vk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (vg, vp, vnames) = build_deepseek_v4_verify_block(
                &ss,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut vk,
            )
            .unwrap();
            let mut vc = Session::new(Device::Cpu).compile_with(vg, &opts);
            for (n, dd) in &vp {
                vc.set_param(n, dd);
            }
            let half = (rd / 2).max(1);
            let (mut cosb, mut sinb, mut sininvb) = (
                vec![0f32; block * half],
                vec![0f32; block * half],
                vec![0f32; block * half],
            );
            for p in 0..block {
                for i in 0..half {
                    let fr = ss.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
                    let (s, c) = (p as f64 * fr).sin_cos();
                    cosb[p * half + i] = c as f32;
                    sinb[p * half + i] = s as f32;
                    sininvb[p * half + i] = -s as f32;
                }
            }
            let nkeys = mw + block;
            let mut mask = vec![-1e30f32; block * nkeys];
            for r in 0..block {
                for j in 0..=r {
                    mask[r * nkeys + mw + j] = 0.0; // block-causal; window empty at pos 0
                }
            }
            let toks_f: Vec<f32> = vtok.iter().map(|&x| x as f32).collect();
            let mut owned: Vec<(String, Vec<f32>)> = vec![
                ("token_id".into(), toks_f),
                ("cos".into(), cosb),
                ("sin".into(), sinb),
                ("sininv".into(), sininvb),
            ];
            for il in 0..ss.n_layers {
                owned.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                owned.push((format!("mask.{il}"), mask.clone()));
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let vout = vc.run(&inputs);
            let vi = vnames.iter().position(|x| x == "logits").unwrap();
            let vlog = &vout[vi];
            for p in 0..block {
                let slice = &vlog[p * vocab..(p + 1) * vocab];
                let err = reflog[p]
                    .iter()
                    .zip(slice)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    err < 1e-4,
                    "batch verify row {p} must match seq=1 decode: err {err:e}"
                );
            }
        }

        // ── END-TO-END speculative decode == greedy (LOSSLESS), with a perfect
        // drafter (full block acceptance) AND a trivial one (fallback to 1/round). ──
        {
            let mut ss = mk_spec(8); // window 8 ≥ prompt(2)+gen(6) ⇒ no in-block eviction
            ss.compress_ratios = vec![0, 0, 0];
            let (mw, mc, block) = (7usize, 4usize, 3usize);
            let prompt = [1u32, 4];
            let n_new = 6usize;
            let mut gdec =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let gref = gdec.generate(&prompt, n_new); // greedy reference
            let mut vk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (vg, vp, vnames) = build_deepseek_v4_verify_block(
                &ss,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut vk,
            )
            .unwrap();
            let mut vc = Session::new(Device::Cpu).compile_with(vg, &opts);
            for (n, dd) in &vp {
                vc.set_param(n, dd);
            }
            // Perfect drafter → full acceptance (exercises multi-token commit).
            let gref2 = gref.clone();
            let mut cur = 0usize;
            let perfect = move |_t: u32| -> Vec<u32> {
                let (s, e) = ((cur + 1).min(gref2.len()), (cur + block).min(gref2.len()));
                cur += block;
                gref2[s..e].to_vec()
            };
            let mut sdec =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let spec_out = deepseek_v4_generate_speculative(
                &mut sdec, &mut vc, &vnames, &ss, block, mc, perfect, &prompt, n_new,
            );
            assert_eq!(
                spec_out, gref,
                "speculative (perfect drafter) must equal greedy"
            );
            // Trivial drafter (draft 0s) → mostly rejected ⇒ greedy fallback.
            let mut tdec =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let triv = |_t: u32| -> Vec<u32> { vec![0u32; block - 1] };
            let triv_out = deepseek_v4_generate_speculative(
                &mut tdec, &mut vc, &vnames, &ss, block, mc, triv, &prompt, n_new,
            );
            assert_eq!(
                triv_out, gref,
                "speculative (trivial drafter) must equal greedy"
            );

            // ── LAYERWISE paged driver (O(L)): one single-layer verify stage per layer,
            // hidden threaded across them, must ALSO equal greedy — proving the O(L²)→O(L)
            // driver is lossless end-to-end (not just the graph, per
            // `deepseek_v4_layerwise_verify_matches_monolithic`). ──
            let build_stages =
                |sp: &DeepseekV4Spec| -> (Vec<rlx_runtime::CompiledGraph>, Vec<Vec<String>>) {
                    let mut st = Vec::new();
                    let mut nm = Vec::new();
                    for il in 0..sp.n_layers {
                        let mut sk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> =
                            HashMap::new();
                        let (sg, sp2, sn) = build_deepseek_v4_verify_block_stage(
                            sp,
                            &mut Mem { t: t.clone() },
                            block,
                            mw,
                            mc,
                            &mut sk,
                            il..il + 1,
                            false,
                        )
                        .unwrap();
                        let mut sc = Session::new(Device::Cpu).compile_with(sg, &opts);
                        for (n, dd) in &sp2 {
                            sc.set_param(n, dd);
                        }
                        for (n, (b, _, _)) in &sk {
                            sc.set_param_typed(n, b, DType::U8);
                        }
                        st.push(sc);
                        nm.push(sn);
                    }
                    (st, nm)
                };
            // All-dense model ⇒ no paged MoE layer fires; the moe_fn must never be called.
            let no_moe = || {
                move |_il: usize, _tok: &[u32], _xf: &[f32]| -> Result<Vec<f32>> {
                    Err(anyhow!("all-dense model: no paged MoE layers"))
                }
            };
            let (mut lst, lnm) = build_stages(&ss);
            let mut ldec =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let gref4 = gref.clone();
            let mut cur2 = 0usize;
            let perfect2 = move |_t: u32, _mh: &[f32]| -> Vec<u32> {
                let (s, e) = ((cur2 + 1).min(gref4.len()), (cur2 + block).min(gref4.len()));
                cur2 += block;
                gref4[s..e].to_vec()
            };
            let lw_out = deepseek_v4_generate_speculative_paged_layerwise(
                &mut ldec,
                &mut lst,
                &lnm,
                &ss,
                block,
                mc,
                no_moe(),
                perfect2,
                &prompt,
                n_new,
            )
            .unwrap();
            assert_eq!(
                lw_out, gref,
                "layerwise paged speculative (perfect drafter) must equal greedy"
            );
            let mut ldec2 =
                V4Decoder::new(&ss, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let triv3 = |_t: u32, _mh: &[f32]| -> Vec<u32> { vec![0u32; block - 1] };
            let lw_triv = deepseek_v4_generate_speculative_paged_layerwise(
                &mut ldec2,
                &mut lst,
                &lnm,
                &ss,
                block,
                mc,
                no_moe(),
                triv3,
                &prompt,
                n_new,
            )
            .unwrap();
            assert_eq!(
                lw_triv, gref,
                "layerwise paged speculative (trivial drafter) must equal greedy"
            );

            // COMPRESS topology ([0,2,4]) → the guard decodes exactly (seq=1) and
            // still equals greedy (correctness holds; acceleration is gated off).
            let cs = mk_spec(8); // real compress_ratios [0,2,4]
            let mut cgdec =
                V4Decoder::new(&cs, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let cgref = cgdec.generate(&prompt, n_new);
            let mut ck: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (cvg, cvp, cvn) = build_deepseek_v4_verify_block(
                &cs,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut ck,
            )
            .unwrap();
            let mut cvc = Session::new(Device::Cpu).compile_with(cvg, &opts);
            for (n, dd) in &cvp {
                cvc.set_param(n, dd);
            }
            let mut csdec =
                V4Decoder::new(&cs, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let triv2 = |_t: u32| -> Vec<u32> { vec![0u32; block - 1] };
            let c_out = deepseek_v4_generate_speculative(
                &mut csdec, &mut cvc, &cvn, &cs, block, mc, triv2, &prompt, n_new,
            );
            assert_eq!(
                c_out, cgref,
                "speculative on compress topology must equal greedy (guard path)"
            );

            // Layerwise driver on the COMPRESS topology: the per-layer compressor state
            // (compcache/pck/pcg/prev_kv/prev_score/ape_idx.{il}) must thread through the
            // single-layer stages exactly as the monolithic verify does.
            let (mut clst, clnm) = build_stages(&cs);
            let mut cldec =
                V4Decoder::new(&cs, &mut Mem { t: t.clone() }, mw, mc, Device::Cpu).unwrap();
            let triv4 = |_t: u32, _mh: &[f32]| -> Vec<u32> { vec![0u32; block - 1] };
            let clw = deepseek_v4_generate_speculative_paged_layerwise(
                &mut cldec,
                &mut clst,
                &clnm,
                &cs,
                block,
                mc,
                no_moe(),
                triv4,
                &prompt,
                n_new,
            )
            .unwrap();
            assert_eq!(
                clw, cgref,
                "layerwise paged speculative on compress must equal greedy"
            );
        }

        // ── Verify graph emits compressor projections (ck/cg) matching seq=1 EXACTLY
        // — the foundation for in-block pooling + the partial-state commit. Overlap
        // (ratio-4) config at pos 0, block 3: the compressor fires at pos 3 (outside
        // the block) ⇒ no in-block compression ⇒ h (hence ck/cg) is exact, incl. the
        // per-token APE gather. ──
        {
            let mut ss = mk_spec(8);
            ss.compress_ratios = vec![0, 0, 4]; // only layer 2 compresses (overlap)
            let ratios = [0usize, 0, 4];
            let (mw, mc, block) = (7usize, 4usize, 3usize);
            let vtok: Vec<u32> = vec![1, 4, 0];
            // seq=1 reference: capture ck.2/cg.2 at pos 0,1,2 (no fire ⇒ no compcache/partial).
            let mut wr: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let (mut s_ck, mut s_cg): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
            for tpos in 0..block {
                let mut dk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
                let (dg, dp, onames) =
                    build_deepseek_v4_decode(&ss, &mut Mem { t: t.clone() }, tpos, tpos, &mut dk)
                        .unwrap();
                let mut dc = Session::new(Device::Cpu).compile_with(dg, &opts);
                for (n, dd) in &dp {
                    dc.set_param(n, dd);
                }
                let mut owned: Vec<(String, Vec<f32>)> =
                    vec![("token_id".into(), vec![vtok[tpos] as f32])];
                if tpos > 0 {
                    for il in 0..nl {
                        owned.push((format!("kvcache.{il}"), wr[il].clone()));
                    }
                }
                let inputs: Vec<(&str, &[f32])> = owned
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_slice()))
                    .collect();
                let out = dc.run(&inputs);
                let get = |nm: &str| -> Vec<f32> {
                    out[onames.iter().position(|x| x == nm).unwrap()].clone()
                };
                s_ck.push(get("ck.2"));
                s_cg.push(get("cg.2"));
                for il in 0..nl {
                    wr[il].extend_from_slice(&get(&format!("kvnew.{il}")));
                }
            }
            // Verify graph ck.2/cg.2 for the same block at pos 0.
            let mut vk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (vg, vp, vnames) = build_deepseek_v4_verify_block(
                &ss,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut vk,
            )
            .unwrap();
            let mut vc = Session::new(Device::Cpu).compile_with(vg, &opts);
            for (n, dd) in &vp {
                vc.set_param(n, dd);
            }
            let half = (rd / 2).max(1);
            let (mut cosb, mut sinb, mut sininvb) = (
                vec![0f32; block * half],
                vec![0f32; block * half],
                vec![0f32; block * half],
            );
            for p in 0..block {
                for i in 0..half {
                    let fr = ss.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
                    let (s, c) = (p as f64 * fr).sin_cos();
                    cosb[p * half + i] = c as f32;
                    sinb[p * half + i] = s as f32;
                    sininvb[p * half + i] = -s as f32;
                }
            }
            let mut owned: Vec<(String, Vec<f32>)> = vec![
                ("token_id".into(), vtok.iter().map(|&x| x as f32).collect()),
                ("cos".into(), cosb),
                ("sin".into(), sinb),
                ("sininv".into(), sininvb),
            ];
            for il in 0..nl {
                owned.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                let r = ratios[il];
                let coff = if r == 4 { 2 } else { 1 };
                let nk = if r == 0 {
                    mw + block
                } else {
                    mw + block + mc + block
                };
                let mut m = vec![-1e30f32; block * nk];
                for row in 0..block {
                    for j in 0..=row {
                        m[row * nk + mw + j] = 0.0; // block-causal; window/compcache/cand empty at pos 0
                    }
                }
                owned.push((format!("mask.{il}"), m));
                if r > 0 {
                    owned.push((format!("compcache.{il}"), vec![0f32; mc * hd]));
                    owned.push((format!("pck.{il}"), vec![0f32; (r - 1) * coff * hd]));
                    owned.push((format!("pcg.{il}"), vec![0f32; (r - 1) * coff * hd]));
                    if r == 4 {
                        owned.push((
                            format!("ape_idx.{il}"),
                            (0..block).map(|i| (i % r) as f32).collect(),
                        ));
                        owned.push((format!("prev_kv.{il}"), vec![0f32; r * coff * hd]));
                        owned.push((format!("prev_score.{il}"), vec![-1e30f32; r * coff * hd]));
                    }
                }
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let vout = vc.run(&inputs);
            let vget =
                |nm: &str| -> &Vec<f32> { &vout[vnames.iter().position(|x| x == nm).unwrap()] };
            let (vck, vcg) = (vget("ck.2"), vget("cg.2")); // [block, 2·hd]
            let w = 2 * hd; // coff=2 for overlap
            for p in 0..block {
                let eck = s_ck[p]
                    .iter()
                    .zip(&vck[p * w..(p + 1) * w])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let ecg = s_cg[p]
                    .iter()
                    .zip(&vcg[p * w..(p + 1) * w])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    eck < 1e-4 && ecg < 1e-4,
                    "verify ck/cg row {p} must match seq=1: {eck:e} {ecg:e}"
                );
            }
        }

        // ── IN-BLOCK NON-OVERLAP POOL: verify == seq=1 with an in-block fire. [0,2,4]
        // at pos 0, block 3: layer 1 (ratio-2, non-overlap) fires at pos 1 INSIDE the
        // block ⇒ the verify forms cand_comp[1] and masks it visible to queries 1,2;
        // layer 2 (ratio-4) fires at pos 3 (outside). Exact-match logits validate the
        // candidate pool AND the visibility mask. ──
        {
            let ss = mk_spec(8); // [0,2,4]
            let ratios = [0usize, 2, 4];
            let coffs = [1usize, 1, 2];
            let (mw, mc, block) = (7usize, 4usize, 3usize);
            let vtok: Vec<u32> = vec![1, 4, 0];
            // seq=1 reference logits at pos 0,1,2, threading layer-1 compressor state.
            let mut wr: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut cc: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut pck: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut pcg: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut s_log: Vec<Vec<f32>> = Vec::new();
            for tpos in 0..block {
                let mut dk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
                let (dg, dp, on) =
                    build_deepseek_v4_decode(&ss, &mut Mem { t: t.clone() }, tpos, tpos, &mut dk)
                        .unwrap();
                let mut dc = Session::new(Device::Cpu).compile_with(dg, &opts);
                for (n, dd) in &dp {
                    dc.set_param(n, dd);
                }
                let mut owned: Vec<(String, Vec<f32>)> =
                    vec![("token_id".into(), vec![vtok[tpos] as f32])];
                for il in 0..nl {
                    if tpos > 0 {
                        owned.push((format!("kvcache.{il}"), wr[il].clone()));
                    }
                    let r = ratios[il];
                    if r > 0 {
                        if tpos >= r {
                            owned.push((format!("compcache.{il}"), cc[il].clone()));
                        }
                        if (tpos + 1) % r == 0 && r > 1 {
                            owned.push((format!("partial_ck.{il}"), pck[il].clone()));
                            owned.push((format!("partial_cg.{il}"), pcg[il].clone()));
                        }
                    }
                }
                let inputs: Vec<(&str, &[f32])> = owned
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_slice()))
                    .collect();
                let out = dc.run(&inputs);
                let get = |nm: &str| -> Vec<f32> {
                    out[on.iter().position(|x| x == nm).unwrap()].clone()
                };
                s_log.push(get("logits"));
                for il in 0..nl {
                    wr[il].extend_from_slice(&get(&format!("kvnew.{il}")));
                    let r = ratios[il];
                    if r > 0 {
                        if (tpos + 1) % r == 0 {
                            cc[il].extend_from_slice(&get(&format!("comp.{il}")));
                            pck[il].clear();
                            pcg[il].clear();
                        } else {
                            pck[il].extend_from_slice(&get(&format!("ck.{il}")));
                            pcg[il].extend_from_slice(&get(&format!("cg.{il}")));
                        }
                    }
                }
            }
            // Verify the whole block at pos 0.
            let mut vk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (vg, vp, vn) = build_deepseek_v4_verify_block(
                &ss,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut vk,
            )
            .unwrap();
            let mut vc = Session::new(Device::Cpu).compile_with(vg, &opts);
            for (n, dd) in &vp {
                vc.set_param(n, dd);
            }
            let half = (rd / 2).max(1);
            let (mut cosb, mut sinb, mut sininvb) = (
                vec![0f32; block * half],
                vec![0f32; block * half],
                vec![0f32; block * half],
            );
            for p in 0..block {
                for i in 0..half {
                    let fr = ss.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
                    let (s, c) = (p as f64 * fr).sin_cos();
                    cosb[p * half + i] = c as f32;
                    sinb[p * half + i] = s as f32;
                    sininvb[p * half + i] = -s as f32;
                }
            }
            let mut owned: Vec<(String, Vec<f32>)> = vec![
                ("token_id".into(), vtok.iter().map(|&x| x as f32).collect()),
                ("cos".into(), cosb),
                ("sin".into(), sinb),
                ("sininv".into(), sininvb),
            ];
            for il in 0..nl {
                owned.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                let r = ratios[il];
                let coff = coffs[il];
                let nk = if r == 0 {
                    mw + block
                } else {
                    mw + block + mc + block
                };
                let mut m = vec![-1e30f32; block * nk];
                for row in 0..block {
                    for j in 0..=row {
                        m[row * nk + mw + j] = 0.0; // block-causal (window empty)
                    }
                    if r > 0 {
                        let cand0 = mw + block + mc;
                        for j in 0..block {
                            if (j + 1) % r == 0 && j <= row {
                                m[row * nk + cand0 + j] = 0.0; // in-block compressed, visible
                            }
                        }
                    }
                }
                owned.push((format!("mask.{il}"), m));
                if r > 0 {
                    owned.push((format!("compcache.{il}"), vec![0f32; mc * hd]));
                    owned.push((format!("pck.{il}"), vec![0f32; (r - 1) * coff * hd]));
                    owned.push((format!("pcg.{il}"), vec![0f32; (r - 1) * coff * hd]));
                    if r == 4 {
                        owned.push((
                            format!("ape_idx.{il}"),
                            (0..block).map(|i| (i % r) as f32).collect(),
                        ));
                        owned.push((format!("prev_kv.{il}"), vec![0f32; r * coff * hd]));
                        owned.push((format!("prev_score.{il}"), vec![-1e30f32; r * coff * hd]));
                    }
                }
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let vout = vc.run(&inputs);
            let vlog = &vout[vn.iter().position(|x| x == "logits").unwrap()];
            for p in 0..block {
                let err = s_log[p]
                    .iter()
                    .zip(&vlog[p * vocab..(p + 1) * vocab])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    err < 1e-3,
                    "in-block non-overlap pool: verify row {p} must match seq=1: {err:e}"
                );
            }
        }

        // ── IN-BLOCK OVERLAP POOL (ratio-4): verify == seq=1 with an in-block ratio-4
        // fire. [0,0,4], 2 primed tokens (dec at pos 2), block [2,3,4]: the overlap
        // compressor fires at pos 3 (block j=1), pooling with the window-0 prev. ──
        {
            let mut ss = mk_spec(8);
            ss.compress_ratios = vec![0, 0, 4];
            let ratios = [0usize, 0, 4];
            let coffs = [1usize, 1, 2];
            let (mw, mc, block) = (7usize, 4usize, 3usize);
            let start = 2usize; // prime pos 0,1 → block starts at pos 2
            let all: Vec<u32> = vec![1, 4, 0, 2, 5]; // pos 0..5
            // seq=1 reference, snapshotting the compressor state at pos `start`.
            let mut wr: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut cc: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut pck: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut pcg: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut pkv: Vec<Vec<f32>> = vec![Vec::new(); nl];
            let mut psc: Vec<Vec<f32>> = vec![Vec::new(); nl];
            for il in 0..nl {
                if ratios[il] == 4 {
                    pkv[il] = vec![0f32; ratios[il] * coffs[il] * hd];
                    psc[il] = vec![-1e30f32; ratios[il] * coffs[il] * hd];
                }
            }
            let (mut snap_wr, mut snap_pck, mut snap_pcg, mut snap_pkv, mut snap_psc) = (
                wr.clone(),
                pck.clone(),
                pcg.clone(),
                pkv.clone(),
                psc.clone(),
            );
            let mut s_log: Vec<Vec<f32>> = Vec::new();
            for (tpos, &tok) in all.iter().enumerate() {
                if tpos == start {
                    snap_wr = wr.clone();
                    snap_pck = pck.clone();
                    snap_pcg = pcg.clone();
                    snap_pkv = pkv.clone();
                    snap_psc = psc.clone();
                }
                let mut dk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
                let (dg, dp, on) =
                    build_deepseek_v4_decode(&ss, &mut Mem { t: t.clone() }, tpos, tpos, &mut dk)
                        .unwrap();
                let mut dc = Session::new(Device::Cpu).compile_with(dg, &opts);
                for (n, dd) in &dp {
                    dc.set_param(n, dd);
                }
                let mut owned: Vec<(String, Vec<f32>)> =
                    vec![("token_id".into(), vec![tok as f32])];
                for il in 0..nl {
                    if tpos > 0 {
                        owned.push((format!("kvcache.{il}"), wr[il].clone()));
                    }
                    let r = ratios[il];
                    if r > 0 {
                        if tpos >= r {
                            owned.push((format!("compcache.{il}"), cc[il].clone()));
                        }
                        if (tpos + 1) % r == 0 && r > 1 {
                            owned.push((format!("partial_ck.{il}"), pck[il].clone()));
                            owned.push((format!("partial_cg.{il}"), pcg[il].clone()));
                        }
                        if (tpos + 1) % r == 0 && r == 4 {
                            owned.push((format!("prev_kv.{il}"), pkv[il].clone()));
                            owned.push((format!("prev_score.{il}"), psc[il].clone()));
                        }
                    }
                }
                let inputs: Vec<(&str, &[f32])> = owned
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_slice()))
                    .collect();
                let out = dc.run(&inputs);
                let get = |nm: &str| -> Vec<f32> {
                    out[on.iter().position(|x| x == nm).unwrap()].clone()
                };
                if tpos >= start {
                    s_log.push(get("logits"));
                }
                for il in 0..nl {
                    wr[il].extend_from_slice(&get(&format!("kvnew.{il}")));
                    let r = ratios[il];
                    if r > 0 {
                        if (tpos + 1) % r == 0 {
                            cc[il].extend_from_slice(&get(&format!("comp.{il}")));
                            if r == 4 {
                                let mut ckw = pck[il].clone();
                                ckw.extend_from_slice(&get(&format!("ck.{il}")));
                                let mut cgw = pcg[il].clone();
                                cgw.extend_from_slice(&get(&format!("cg.{il}")));
                                pkv[il] = ckw;
                                psc[il] = cgw;
                            }
                            pck[il].clear();
                            pcg[il].clear();
                        } else {
                            pck[il].extend_from_slice(&get(&format!("ck.{il}")));
                            pcg[il].extend_from_slice(&get(&format!("cg.{il}")));
                        }
                    }
                }
            }
            // Verify the block [pos start..start+3] on the snapshot state.
            let mut vk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
            let (vg, vp, vn) = build_deepseek_v4_verify_block(
                &ss,
                &mut Mem { t: t.clone() },
                block,
                mw,
                mc,
                &mut vk,
            )
            .unwrap();
            let mut vc = Session::new(Device::Cpu).compile_with(vg, &opts);
            for (n, dd) in &vp {
                vc.set_param(n, dd);
            }
            let half = (rd / 2).max(1);
            let (mut cosb, mut sinb, mut sininvb) = (
                vec![0f32; block * half],
                vec![0f32; block * half],
                vec![0f32; block * half],
            );
            for p in 0..block {
                for i in 0..half {
                    let fr = ss.rope_theta.powf(-(2.0 * i as f64) / rd as f64);
                    let (s, c) = ((start + p) as f64 * fr).sin_cos();
                    cosb[p * half + i] = c as f32;
                    sinb[p * half + i] = s as f32;
                    sininvb[p * half + i] = -s as f32;
                }
            }
            let btok: Vec<f32> = all[start..start + block]
                .iter()
                .map(|&x| x as f32)
                .collect();
            let mut owned: Vec<(String, Vec<f32>)> = vec![
                ("token_id".into(), btok),
                ("cos".into(), cosb),
                ("sin".into(), sinb),
                ("sininv".into(), sininvb),
            ];
            for il in 0..nl {
                let r = ratios[il];
                let coff = coffs[il];
                let wlen = snap_wr[il].len() / hd;
                let padw = {
                    let mut o = vec![0f32; mw * hd];
                    let nn = snap_wr[il].len().min(mw * hd);
                    o[..nn].copy_from_slice(&snap_wr[il][..nn]);
                    o
                };
                owned.push((format!("wcache.{il}"), padw));
                let nk = if r == 0 {
                    mw + block
                } else {
                    mw + block + mc + block
                };
                let mut m = vec![-1e30f32; block * nk];
                for row in 0..block {
                    for s in 0..wlen.min(mw) {
                        m[row * nk + s] = 0.0; // window valid
                    }
                    for j in 0..=row {
                        m[row * nk + mw + j] = 0.0; // block-causal
                    }
                    if r > 0 {
                        let cand0 = mw + block + mc;
                        for j in 0..block {
                            if (start + j + 1).is_multiple_of(r) && j <= row {
                                m[row * nk + cand0 + j] = 0.0; // in-block compressed, visible
                            }
                        }
                    }
                }
                owned.push((format!("mask.{il}"), m));
                if r > 0 {
                    owned.push((format!("compcache.{il}"), vec![0f32; mc * hd])); // ncomp=0 at pos 2
                    // front-pad the partial to ratio-1 slots
                    let slots = r - 1;
                    let np = snap_pck[il].len() / (coff * hd);
                    let mut fpk = vec![0f32; (slots - np) * coff * hd];
                    fpk.extend_from_slice(&snap_pck[il]);
                    let mut fpg = vec![0f32; (slots - np) * coff * hd];
                    fpg.extend_from_slice(&snap_pcg[il]);
                    owned.push((format!("pck.{il}"), fpk));
                    owned.push((format!("pcg.{il}"), fpg));
                    if r == 4 {
                        owned.push((
                            format!("ape_idx.{il}"),
                            (0..block).map(|i| ((start + i) % r) as f32).collect(),
                        ));
                        owned.push((format!("prev_kv.{il}"), snap_pkv[il].clone()));
                        owned.push((format!("prev_score.{il}"), snap_psc[il].clone()));
                    }
                }
            }
            let inputs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let vout = vc.run(&inputs);
            let vlog = &vout[vn.iter().position(|x| x == "logits").unwrap()];
            for p in 0..block {
                let err = s_log[p]
                    .iter()
                    .zip(&vlog[p * vocab..(p + 1) * vocab])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    err < 1e-3,
                    "in-block OVERLAP pool: verify row {p} must match seq=1: {err:e}"
                );
            }
        }

        // ── Distributed decode: a 2-stage layer split (relaying the hidden
        // boundary) == the single-stage decode. Validates the cross-node seam for
        // decode (the analogue of the prefill stage-split). Single step, pos 0,
        // empty caches — proves the hidden `[1,hc,d]` boundary; per-stage cache
        // threading is the already-validated single-node logic. ──
        let (mw, mc) = (2usize, 4usize);
        let gspec2 = mk_spec(3);
        let mk_inputs = |lrange: std::ops::Range<usize>, is_first: bool, hin: Option<Vec<f32>>| {
            let half = (rd / 2).max(1);
            let mut o: Vec<(String, Vec<f32>)> = Vec::new();
            if is_first {
                o.push(("token_id".into(), vec![ids[0] as f32]));
            } else {
                o.push(("hidden_in".into(), hin.unwrap()));
            }
            o.push(("cos".into(), vec![1f32; half])); // pos 0 → cos 0 =1, sin 0 =0
            o.push(("sin".into(), vec![0f32; half]));
            o.push(("sininv".into(), vec![0f32; half]));
            for il in lrange {
                let ratio = [0usize, 2, 4][il];
                let coff = if ratio == 4 { 2 } else { 1 };
                o.push((format!("wcache.{il}"), vec![0f32; mw * hd]));
                if ratio == 0 {
                    let mut m = vec![-1e30f32; mw + 1];
                    m[mw] = 0.0; // only the new token is valid
                    o.push((format!("mask.{il}"), m));
                } else {
                    o.push((format!("compcache.{il}"), vec![0f32; mc * hd]));
                    o.push((
                        format!("partial_ck.{il}"),
                        vec![0f32; (ratio - 1) * coff * hd],
                    ));
                    o.push((
                        format!("partial_cg.{il}"),
                        vec![0f32; (ratio - 1) * coff * hd],
                    ));
                    if ratio == 4 {
                        let ape = &t[&format!("model.layers.{il}.attn.compressor.ape")].0;
                        o.push((format!("ape_row.{il}"), ape[0..coff * hd].to_vec()));
                        o.push((format!("prev_kv.{il}"), vec![0f32; ratio * coff * hd]));
                        o.push((
                            format!("prev_score.{il}"),
                            vec![-1e30f32; ratio * coff * hd],
                        ));
                    }
                    let mut m = vec![-1e30f32; mw + 1 + mc + 1];
                    m[mw] = 0.0; // new valid; window/comp empty; not firing at pos 0
                    o.push((format!("mask.{il}"), m));
                }
            }
            o
        };
        let run = |g: Graph,
                   p: &HashMap<String, Vec<f32>>,
                   names: &[String],
                   inp: &[(String, Vec<f32>)],
                   want: &str| {
            let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
            for (n, dd) in p {
                c.set_param(n, dd);
            }
            let r: Vec<(&str, &[f32])> = inp
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let out = c.run(&r);
            out[names.iter().position(|x| x == want).unwrap()].clone()
        };
        // 1-stage (all layers).
        let mut fk = HashMap::new();
        let (fg, fp, fnm) =
            build_deepseek_v4_decode_fixed(&gspec2, &mut Mem { t: t.clone() }, mw, mc, &mut fk)
                .unwrap();
        let full_logits = run(fg, &fp, &fnm, &mk_inputs(0..nl, true, None), "logits");
        // 2-stage: layers [0,1] then [2], relaying `hidden`.
        let mut s0k = HashMap::new();
        let (s0g, s0p, s0nm) = build_deepseek_v4_decode_fixed_stage(
            &gspec2,
            &mut Mem { t: t.clone() },
            0..2,
            true,
            false,
            mw,
            mc,
            &mut s0k,
        )
        .unwrap();
        let hidden = run(s0g, &s0p, &s0nm, &mk_inputs(0..2, true, None), "hidden_out");
        let mut s1k = HashMap::new();
        let (s1g, s1p, s1nm) = build_deepseek_v4_decode_fixed_stage(
            &gspec2,
            &mut Mem { t: t.clone() },
            2..3,
            false,
            true,
            mw,
            mc,
            &mut s1k,
        )
        .unwrap();
        let split_logits = run(
            s1g,
            &s1p,
            &s1nm,
            &mk_inputs(2..3, false, Some(hidden)),
            "logits",
        );
        let split_err = full_logits
            .iter()
            .zip(&split_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            split_err < 1e-4,
            "2-stage decode split must match 1-stage: {split_err:e}"
        );

        // ── Pipelined (distributed) generate: 2 stage decoders relaying the hidden
        // per token == the single-node V4Decoder. The in-process reference for a
        // cross-node decode run. ──
        let st0 = V4Decoder::new_stage(
            &mk_spec(3),
            &mut Mem { t: t.clone() },
            0..2,
            true,
            false,
            2,
            4,
            Device::Cpu,
        )
        .unwrap();
        let st1 = V4Decoder::new_stage(
            &mk_spec(3),
            &mut Mem { t: t.clone() },
            2..3,
            false,
            true,
            2,
            4,
            Device::Cpu,
        )
        .unwrap();
        let mut stages = vec![st0, st1];
        let pipe_gen = deepseek_v4_generate_pipelined(&mut stages, &prompt, n_new);
        assert_eq!(
            pipe_gen, once_gen,
            "pipelined 2-stage decode must match single-node V4Decoder"
        );

        // ── Distributed decode over REAL TCP (worker threads on localhost) ==
        // single-node. Each stage builds inside its thread (nothing non-Send
        // crosses) and serves; the coordinator drives + stops them. ──
        let stage_cfg = [(0..2usize, true, false), (2..3usize, false, true)];
        let mut addrs: Vec<String> = Vec::new();
        let mut handles = Vec::new();
        for (lr, fst, lst) in stage_cfg {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            addrs.push(listener.local_addr().unwrap().to_string());
            let tt = t.clone();
            let sp = mk_spec(3);
            handles.push(std::thread::spawn(move || {
                let mut dec =
                    V4Decoder::new_stage(&sp, &mut Mem { t: tt }, lr, fst, lst, 2, 4, Device::Cpu)
                        .unwrap();
                serve_v4_decode_stage(&mut dec, listener).unwrap();
            }));
        }
        let tcp_gen = run_v4_decode_pipelined_tcp(&addrs, vocab, &prompt, n_new).unwrap();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            tcp_gen, once_gen,
            "TCP distributed decode must match single-node V4Decoder"
        );
    }

    #[test]
    fn rope_llama3_boundaries() {
        let s = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        };
        let base = |i: usize| 1.0 / 10000f64.powf(2.0 * i as f64 / 128.0);
        // i=0: high-frequency (short wavelen 2π ≪ 8192/4) → unchanged.
        assert!((s.inv_freq(0, 128, 10000.0) - base(0)).abs() < 1e-12);
        // i=63: low-frequency (long wavelen ≫ 8192/1) → /factor.
        assert!((s.inv_freq(63, 128, 10000.0) - base(63) / 8.0).abs() < 1e-15);
        // None is the vanilla base (identity scaling).
        assert!((RopeScaling::None.inv_freq(5, 128, 10000.0) - base(5)).abs() < 1e-15);
    }

    fn base_spec() -> DecoderSpec {
        DecoderSpec {
            arch: "test".into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            rope_scaling: RopeScaling::None,
            hidden_act: "silu".into(),
            attention_bias: false,
            fused_qkv: false,
            fused_gate_up: false,
            qk_norm: false,
            qk_norm_full: false,
            tie_word_embeddings: true,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: usize::MAX,
            num_experts: 0,
            num_experts_used: 0,
            norm_topk_prob: false,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_gain_offset: 0.0,
            sandwich_norms: false,
            embed_scale: 1.0,
            gelu_gate: false,
            rope_dual: None,
            attn_score_scale: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
        }
    }

    #[test]
    fn guard_rejects_moe_and_geglu() {
        base_spec().guard_supported().unwrap();
        // MoE with a missing `num_experts_per_tok` is rejected (can't route).
        let mut moe = base_spec();
        moe.num_experts = 8;
        assert!(
            moe.guard_supported()
                .unwrap_err()
                .to_string()
                .contains("MoE")
        );
        // A well-formed MoE spec (top-k set) is accepted by the guard.
        moe.num_experts_used = 2;
        assert!(moe.guard_supported().is_ok());
        // GeGLU (gelu) is now supported (Gemma path).
        let mut gelu = base_spec();
        gelu.hidden_act = "gelu_pytorch_tanh".into();
        assert!(gelu.guard_supported().is_ok());
        // A genuinely unhandled activation is still rejected.
        let mut relu = base_spec();
        relu.hidden_act = "relu".into();
        assert!(relu.guard_supported().is_err());
    }

    /// Minimal loader that only answers `remaining_keys` (all `from_config_json`
    /// needs for topology inference).
    struct KeyProbe(Vec<String>);
    impl WeightLoader for KeyProbe {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn take(&mut self, _k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            unreachable!()
        }
        fn take_transposed(&mut self, _k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            unreachable!()
        }
        fn remaining_keys(&self) -> Vec<String> {
            self.0.clone()
        }
    }

    fn write_cfg(json: &str) -> std::path::PathBuf {
        // Unique-enough temp dir without Date/rand: hash the json.
        let mut h: u64 = 1469598103934665603;
        for b in json.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let dir = std::env::temp_dir().join(format!("rlx_decspec_{h:016x}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), json).unwrap();
        dir
    }

    #[test]
    fn infer_qwen3_style_qk_norm_from_tensors() {
        let dir = write_cfg(
            r#"{"model_type":"qwen3","vocab_size":151936,"hidden_size":1024,
                "intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,
                "num_key_value_heads":8,"head_dim":128,"rope_theta":1000000,
                "tie_word_embeddings":true,"hidden_act":"silu"}"#,
        );
        let keys = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.q_norm.weight".to_string(),
        ];
        let spec = DecoderSpec::from_config_json(&dir, &KeyProbe(keys)).unwrap();
        assert_eq!(spec.arch, "qwen3");
        assert!(spec.qk_norm, "q_norm tensor → qk_norm inferred true");
        assert!(!spec.attention_bias);
        assert!(spec.tie_word_embeddings);
        assert!(matches!(spec.rope_scaling, RopeScaling::None));
    }

    #[test]
    fn infer_llama3_rope_and_no_qk_norm() {
        let dir = write_cfg(
            r#"{"model_type":"llama","vocab_size":128256,"hidden_size":2048,
                "intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":32,
                "num_key_value_heads":8,"head_dim":64,"rope_theta":500000.0,
                "tie_word_embeddings":true,"hidden_act":"silu",
                "rope_scaling":{"rope_type":"llama3","factor":32.0,"low_freq_factor":1.0,
                "high_freq_factor":4.0,"original_max_position_embeddings":8192}}"#,
        );
        // No q_norm tensor → qk_norm must infer false.
        let keys = vec!["model.layers.0.self_attn.q_proj.weight".to_string()];
        let spec = DecoderSpec::from_config_json(&dir, &KeyProbe(keys)).unwrap();
        assert!(!spec.qk_norm);
        assert_eq!(spec.head_dim, 64);
        assert!(matches!(spec.rope_scaling, RopeScaling::Llama3 { factor, .. } if factor == 32.0));
    }

    #[test]
    fn infer_untied_head_when_lm_head_present_and_config_silent() {
        let dir = write_cfg(
            r#"{"model_type":"mistral","vocab_size":32000,"hidden_size":64,
                "intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":8,
                "num_key_value_heads":8,"rope_theta":10000.0,"hidden_act":"silu"}"#,
        );
        // lm_head.weight present + config silent on tie → infer untied.
        let keys = vec!["lm_head.weight".to_string()];
        let spec = DecoderSpec::from_config_json(&dir, &KeyProbe(keys)).unwrap();
        assert!(!spec.tie_word_embeddings);
        // head_dim derived from hidden/heads when absent.
        assert_eq!(spec.head_dim, 8);
    }

    #[test]
    fn classify_supported_and_rejected() {
        let c = |j: &str| classify_config(&serde_json::from_str::<serde_json::Value>(j).unwrap());
        let base = |extra: &str| {
            format!(
                r#"{{"hidden_size":2048,"num_hidden_layers":16,"num_attention_heads":32,
                    "vocab_size":128256,"intermediate_size":8192,"rms_norm_eps":1e-5,{extra}}}"#
            )
        };
        // Dense Llama/Qwen → supported.
        assert!(c(&base(r#""model_type":"llama","hidden_act":"silu""#)).supported);
        assert!(c(&base(r#""model_type":"qwen2","hidden_act":"silu""#)).supported);
        // Phi-3.5 LongRoPE → supported (generic builder handles longrope now).
        assert!(
            c(&base(
                r#""model_type":"phi3","hidden_act":"silu","rope_scaling":{"type":"longrope","short_factor":[1.0],"long_factor":[1.0]}"#
            ))
            .supported
        );
        // gemma2 & gemma3 are generic-supported (GeGLU/sandwich/softcap handled);
        // gemma3n (AltUp) is not.
        assert!(
            c(&base(
                r#""model_type":"gemma3_text","hidden_activation":"gelu_pytorch_tanh""#
            ))
            .supported
        );
        assert!(
            c(&base(
                r#""model_type":"gemma2","hidden_activation":"gelu_pytorch_tanh""#
            ))
            .supported
        );
        let g3n = c(&base(
            r#""model_type":"gemma3n","hidden_activation":"gelu_pytorch_tanh""#,
        ));
        assert!(!g3n.supported && g3n.reason.contains("Gemma"));
        // MoE via num_local_experts (gpt_oss) → rejected.
        let m = c(&base(
            r#""model_type":"gpt_oss","hidden_act":"silu","num_local_experts":32"#,
        ));
        assert!(!m.supported);
        // Phi-3 (fused QKV/gate_up, full rotary) → supported; Phi-2 (partial
        // rotary) still rejected.
        assert!(c(&base(r#""model_type":"phi3","hidden_act":"silu""#)).supported);
        assert!(
            !c(&base(
                r#""model_type":"phi","hidden_act":"silu","partial_rotary_factor":0.4"#
            ))
            .supported
        );
        // Qwen3.5 hybrid → rejected by arch.
        assert!(!c(&base(r#""model_type":"qwen3_5","hidden_act":"silu""#)).supported);
        // A genuinely unhandled activation (not silu/swish/gelu) → rejected.
        let ge = c(&base(r#""model_type":"foo","hidden_act":"relu""#));
        assert!(!ge.supported);
        // MoE via n experts on an unvalidated arch → rejected.
        assert!(
            !c(&base(
                r#""model_type":"qwen2","hidden_act":"silu","num_experts":8"#
            ))
            .supported
        );
        // OLMoE (validated MoE layout) → supported.
        let olmoe = c(&base(
            r#""model_type":"olmoe","hidden_act":"silu","num_experts":64,"num_experts_per_tok":8"#,
        ));
        assert!(olmoe.supported, "olmoe rejected: {}", olmoe.reason);
        assert_eq!(olmoe.num_experts, 64);
        // Qwen3-MoE / Qwen2-MoE (switch_mlp stacked experts) → supported.
        let q3moe = c(&base(
            r#""model_type":"qwen3_moe","hidden_act":"silu","num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":768,"norm_topk_prob":true"#,
        ));
        assert!(q3moe.supported, "qwen3_moe rejected: {}", q3moe.reason);
        let q2moe = c(&base(
            r#""model_type":"qwen2_moe","hidden_act":"silu","num_experts":60,"num_experts_per_tok":4,"moe_intermediate_size":1408,"shared_expert_intermediate_size":5632"#,
        ));
        assert!(q2moe.supported, "qwen2_moe rejected: {}", q2moe.reason);
        // A still-unvalidated MoE arch (Mixtral, different tensor names) → rejected.
        assert!(
            !c(&base(
                r#""model_type":"mixtral","hidden_act":"silu","num_experts":8,"num_experts_per_tok":2"#
            ))
            .supported
        );
        // Partial rotary (Phi-4-mini `partial_rotary_factor` 0.75) is now handled
        // by the generic builder (rotate first n_rot dims, pass the rest) — run-
        // validated on Phi-4-mini-instruct-4bit (exact prefill argmax 12650 vs
        // mlx-lm). Accepted as a dense SwiGLU+RMSNorm decoder.
        assert!(
            c(&base(
                r#""model_type":"phi3","hidden_act":"silu","partial_rotary_factor":0.75"#
            ))
            .supported
        );
        // yarn rope IS handled → supported; an unknown rope_type is rejected.
        assert!(
            c(&base(
                r#""model_type":"qwen3","hidden_act":"silu","rope_scaling":{"rope_type":"yarn","factor":4.0}"#
            ))
            .supported
        );
        let u = c(&base(
            r#""model_type":"qwen3","hidden_act":"silu","rope_scaling":{"rope_type":"mrope_xyz","factor":4.0}"#,
        ));
        assert!(!u.supported && u.reason.contains("rope_scaling"));
    }

    #[test]
    fn yarn_inv_freq_matches_transformers() {
        // DeepSeek-R1-Qwen3 config: head_dim=128, θ=1e6, factor=4, orig=32768,
        // beta_fast=32, beta_slow=1, explicit attn_factor=0.8782488562869419.
        let y = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32768.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: Some(0.878_248_856_286_941_9),
        };
        let refs = [
            (0usize, 1.0f64),
            (1, 0.805_842_188),
            (16, 0.031_622_777),
            (32, 0.000_602_941),
            (48, 7.906e-6),
            (63, 3.1e-7),
        ];
        for (i, want) in refs {
            let got = y.inv_freq(i, 128, 1e6);
            assert!(
                (got - want).abs() <= want.abs() * 1e-4 + 1e-9,
                "yarn inv_freq[{i}] = {got}, want {want}"
            );
        }
        assert!((y.mscale() - 0.878_248_856_286_941_9).abs() < 1e-12);
        // Derived mscale when config omits attention_factor: 0.1·ln(4)+1.
        let yd = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32768.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        };
        assert!((yd.mscale() - (0.1 * 4.0_f64.ln() + 1.0)).abs() < 1e-12);
        // None path still equals vanilla 1/θ^(2i/d) (no regression).
        let n = RopeScaling::None;
        assert!((n.inv_freq(3, 128, 1e6) - 1.0 / 1e6_f64.powf(6.0 / 128.0)).abs() < 1e-12);
    }
}
