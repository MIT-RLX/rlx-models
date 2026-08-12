// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 configuration — HF `config.json` and GGUF `llama.*` metadata.

use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Llama32RopeType {
    #[default]
    Default,
    #[serde(rename = "llama3")]
    Llama3,
}

/// Dense-arch structural family for the packed (`builder.rs`) block deltas.
///
/// Every variant except [`DenseArch::Llama`] carries a small per-arch topology
/// difference (norm placement / kind, FFN shape, residual wiring) that the
/// packed builder applies. `Llama` reproduces the stock Llama/Granite/Phi block
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseArch {
    /// Stock Llama block (also granite / exaone / mistral / phi paths).
    Llama,
    /// OLMo-2: no pre-attn/pre-ffn norm; full-projection Q/K RMSNorm; a
    /// post-attention and post-feedforward RMSNorm applied to each sub-layer
    /// output *before* the residual add.
    Olmo2,
    /// Nemotron (dense): LayerNorm (+bias) for input/pre-ffn/final norms; a
    /// gate-less squared-ReLU FFN (`down(relu(up(x))²)`); partial RoPE.
    Nemotron,
    /// Cohere / Command-R / Cohere2: a single LayerNorm (no bias) feeds BOTH
    /// attention and MLP whose outputs are summed into one residual
    /// (`h = x + attn(ln(x)) + mlp(ln(x))`); `logit_scale` MULTIPLIES logits.
    Cohere,
    /// GLM-4 (0414): four RMSNorms per layer — pre-attn, post-attn (pre
    /// residual), pre-ffn, post-ffn (pre residual); fused gate∥up MLP.
    Glm4,
    /// ChatGLM / GLM-Edge: standard pre-norm (input + pre-ffn RMSNorm); fused
    /// gate∥up MLP; partial-or-full RoPE per `rope.dimension_count`.
    ChatGlm,
    /// Muse Glimmer (Meta) — four RMSNorms per layer like [`DenseArch::Glm4`]
    /// (pre-attn, post-attn, pre-ffn, post-ffn), but with four extra deltas:
    ///   - per-head-dim Q/K RMSNorm (Qwen3-style, `[head_dim]` gains),
    ///   - a sigmoid **attention output gate** applied between SDPA and
    ///     `o_proj` (`attn_out * sigmoid(W_gate @ pre_attn_normed)`),
    ///   - interleaved local/global attention: 3 sliding-window layers (RoPE)
    ///     then 1 full-attention layer (**NoPE**), repeating,
    ///   - an unweighted RMSNorm on the token embeddings, `logit_scale` as a
    ///     logit MULTIPLIER, and a final `tanh` logit softcap.
    ///
    /// The two post-norms use a distinct, much smaller epsilon (1e-8) than the
    /// pre-norms (`attention.layer_norm_rms_epsilon`, 1e-5). Matches llama.cpp
    /// `src/models/muse-glimmer.cpp`.
    MuseGlimmer,
}

/// Which normalization the arch uses for its per-layer norms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// RMSNorm (Llama / OLMo-2 / GLM).
    Rms,
    /// Mean-subtracting LayerNorm (Nemotron with bias, Cohere without).
    LayerNorm,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32RopeScaling {
    pub factor: f32,
    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f32,
    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: Llama32RopeType,
}

fn default_low_freq_factor() -> f32 {
    1.0
}
fn default_high_freq_factor() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    /// Explicit head dim (Llama 3.x); when absent, derived from hidden/heads.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<Llama32RopeScaling>,
    /// Granite embedding multiplier — input token embeddings are multiplied by
    /// this scalar (`GraniteForCausalLM.embedding_multiplier`, GGUF
    /// `granite.embedding_scale`). `None` = no scaling (plain Llama).
    #[serde(default, alias = "embedding_multiplier")]
    pub embedding_scale: Option<f32>,
    /// Granite residual multiplier — every attention / MLP sub-layer output is
    /// multiplied by this before the residual add (`residual_multiplier`, GGUF
    /// `granite.residual_scale`).
    #[serde(default, alias = "residual_multiplier")]
    pub residual_scale: Option<f32>,
    /// Granite attention score scale — replaces the default `1/sqrt(head_dim)`
    /// softmax scale (`attention_multiplier`, GGUF `granite.attention.scale`).
    #[serde(default, alias = "attention_multiplier")]
    pub attention_scale: Option<f32>,
    /// Granite logit divisor — final logits are divided by this scalar
    /// (`logits_scaling`, GGUF `granite.logit_scale`).
    #[serde(default, alias = "logits_scaling")]
    pub logit_scale: Option<f32>,
    /// Looped Transformer depth multiplier (Nanbeige). Layers are applied
    /// `num_loops` times with **shared** weights and **separate** KV slots
    /// per loop iteration. Default `1` = standard single pass.
    #[serde(default = "default_num_loops")]
    pub num_loops: usize,
    /// When `false` (Nanbeige default), apply `model.norm` after every loop
    /// pass. When `true`, only the final norm after the last loop runs
    /// (standard Llama).
    #[serde(default)]
    pub skip_loop_final_norm: bool,
    /// RoPE pairing flavor. GGUF Llama weights are permuted by the HF→GGUF
    /// converter for llama.cpp's interleaved (`NORM`) RoPE, so GGUF-backed
    /// inference must rotate with [`rlx_ir::RopeStyle::GptJ`]; HF-safetensors
    /// checkpoints use [`rlx_ir::RopeStyle::NeoX`] (default). Not present in
    /// HF `config.json`, so skipped during deserialization.
    #[serde(skip)]
    pub rope_style: rlx_ir::RopeStyle,
    /// GGUF `general.architecture` tag when loaded from GGUF (`llama`, `phi3`, …).
    #[serde(skip)]
    pub gguf_arch: Option<String>,
    /// Rotary dimension when it differs from `head_dim` (Phi-3 partial RoPE).
    #[serde(skip)]
    pub rope_dim: Option<usize>,
    /// Sliding-window width for local attention layers
    /// (`{arch}.attention.sliding_window`, llama.cpp `n_swa`). A local layer
    /// attends to keys within `w` positions **inclusive** of the query
    /// (`q_pos - k_pos <= w`), matching llama.cpp `LLAMA_SWA_TYPE_STANDARD`
    /// and [`rlx_ir::op::MaskKind::SlidingWindow`]. `None` = all layers full.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Local/global interleave period
    /// (`{arch}.attention.sliding_window_pattern`). Mirrors llama.cpp
    /// `llama_hparams::set_swa_pattern(p)` with `dense_first = false`: layer
    /// `i` is sliding-window when `i % p < p - 1`, so `p = 4` yields three
    /// local layers followed by one global (full-attention) layer.
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// Final logit `tanh` softcap (`{arch}.final_logit_softcapping`, Gemma-2
    /// style): `logits = cap * tanh(logits / cap)`. Applied AFTER
    /// [`Self::final_logit_multiplier`], matching llama.cpp's ordering.
    #[serde(default, alias = "final_logit_softcapping")]
    pub final_logit_softcap: Option<f32>,
}

fn default_num_loops() -> usize {
    1
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}

impl Llama32Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn from_gguf(raw: &GgufFile) -> anyhow::Result<Self> {
        llama32_cfg_from_gguf(raw)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// Leading per-head dims that receive RoPE (equals `head_dim` for Llama;
    /// may be smaller for Phi-3 partial RoPE).
    pub fn n_rot(&self) -> usize {
        self.rope_dim
            .filter(|&r| r > 0 && r <= self.head_dim())
            .unwrap_or_else(|| self.head_dim())
    }

    pub fn uses_partial_rope(&self) -> bool {
        self.n_rot() < self.head_dim()
    }

    pub fn is_phi_arch(&self) -> bool {
        matches!(self.gguf_arch.as_deref(), Some("phi3") | Some("phi4"))
    }

    /// Attention softmax scale: Granite's `attention_multiplier` when present,
    /// else the default `1/sqrt(head_dim)` (returned as `None` so the op picks
    /// its own default). Granite-2b uses `0.015625` (= 1/head_dim, not
    /// 1/sqrt(head_dim)), so this must override the op default.
    pub fn attn_score_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    /// Whether any Granite-style scalar multiplier is active. When false the
    /// builder emits the stock Llama residual/embed/logit math unchanged.
    pub fn has_granite_scalars(&self) -> bool {
        self.embedding_scale.is_some()
            || self.residual_scale.is_some()
            || self.attention_scale.is_some()
            || self.logit_scale.is_some()
    }

    /// Structural arch family for the packed builder's per-arch block deltas.
    pub fn dense_arch(&self) -> DenseArch {
        match self.gguf_arch.as_deref() {
            Some("olmo2") | Some("olmo") => DenseArch::Olmo2,
            Some("nemotron") => DenseArch::Nemotron,
            Some("cohere") | Some("command-r") | Some("cohere2") => DenseArch::Cohere,
            Some("glm4") => DenseArch::Glm4,
            Some("chatglm") => DenseArch::ChatGlm,
            Some("muse-glimmer") => DenseArch::MuseGlimmer,
            _ => DenseArch::Llama,
        }
    }

    /// Local/global interleave period, or `None` when every layer is
    /// full-attention. Only Muse Glimmer drives this from GGUF metadata today;
    /// Cohere2 keeps its hardcoded pattern in [`Self::cohere2_nope_pattern`].
    pub fn swa_pattern(&self) -> Option<usize> {
        match self.dense_arch() {
            DenseArch::MuseGlimmer => self.sliding_window_pattern.filter(|&p| p > 1),
            _ => None,
        }
    }

    /// Whether layer `idx` is a **sliding-window (local)** layer. llama.cpp
    /// `set_swa_pattern(p)`: `is_swa(i) = i % p < p - 1`. With no pattern every
    /// layer is global.
    pub fn is_swa_layer(&self, idx: usize) -> bool {
        self.swa_pattern().is_some_and(|p| idx % p < p - 1)
    }

    /// Sliding-window width to use for layer `idx`, or `None` for a full
    /// (global) attention layer.
    pub fn attn_window_for_layer(&self, idx: usize) -> Option<usize> {
        self.sliding_window
            .filter(|&w| w > 0 && self.is_swa_layer(idx))
    }

    /// Whether layer `idx` skips RoPE entirely (NoPE).
    ///
    /// Two arches land here with the *same* predicate but different framings:
    /// Cohere2 states it as "global layers are NoPE" (`(i+1) % p == 0`), and
    /// Muse Glimmer as "RoPE runs on the SWA layers" (`!is_swa(i)`, i.e.
    /// `i % p == p - 1`). Those are the same set of layers.
    pub fn is_nope_layer(&self, idx: usize) -> bool {
        if let Some(p) = self.cohere2_nope_pattern() {
            return (idx + 1).is_multiple_of(p);
        }
        self.swa_pattern().is_some() && !self.is_swa_layer(idx)
    }

    /// Epsilon for the two **post**-norms (post-attention / post-FFN). Muse
    /// Glimmer hardcodes 1e-8 here — deliberately different from the pre-norms'
    /// `attention.layer_norm_rms_epsilon` (1e-5); see llama.cpp
    /// `muse-glimmer.cpp` (`post_norm_eps`). Every other arch reuses
    /// `rms_norm_eps`.
    pub fn post_norm_eps(&self) -> f32 {
        match self.dense_arch() {
            DenseArch::MuseGlimmer => 1e-8,
            _ => self.rms_norm_eps as f32,
        }
    }

    /// Whether the arch gates the attention output with
    /// `sigmoid(W_gate @ pre_attn_normed)` before `o_proj`.
    pub fn uses_attn_out_gate(&self) -> bool {
        self.dense_arch() == DenseArch::MuseGlimmer
    }

    /// Whether an **unweighted** RMSNorm is applied to the token embeddings
    /// before the first block (Muse Glimmer's `build_norm(inpL, nullptr, …)`).
    /// This is not the same as Gemma's `sqrt(d_model)` embedding multiplier.
    pub fn normalizes_input_embeddings(&self) -> bool {
        self.dense_arch() == DenseArch::MuseGlimmer
    }

    /// Cohere2 (Command-R7B) interleaves sliding-window and full-attention
    /// layers and applies **NoPE (no RoPE) on the global/full-attention
    /// layers** (mlx-lm `cohere2.py`: RoPE only when `use_sliding_window`;
    /// `use_sliding_window = (i+1) % pattern != 0`). Returns the pattern (so a
    /// layer is global-NoPE when `(layer_idx+1) % pattern == 0`), or `None` for
    /// plain Cohere / Command-R (all layers use RoPE). llama.cpp hardcodes the
    /// cohere2 pattern to 4.
    pub fn cohere2_nope_pattern(&self) -> Option<usize> {
        matches!(self.gguf_arch.as_deref(), Some("cohere2")).then_some(4)
    }

    /// Normalization flavor for this arch's per-layer norms.
    pub fn norm_kind(&self) -> NormKind {
        match self.dense_arch() {
            DenseArch::Nemotron | DenseArch::Cohere => NormKind::LayerNorm,
            _ => NormKind::Rms,
        }
    }

    /// Whether the arch needs the packed (`builder.rs`) block deltas, so the
    /// CPU / MLX F32-`rlx-flow` path must be bypassed (as for Phi / Granite).
    /// A stock-Llama block still runs on the F32 flow.
    pub fn needs_arch_packed_builder(&self) -> bool {
        self.dense_arch() != DenseArch::Llama
    }

    /// Cohere applies `logit_scale` as a MULTIPLIER on the final logits
    /// (`logits *= logit_scale`), unlike Granite which DIVIDES by it
    /// (`logits /= logits_scaling`). Returns the effective multiplier for the
    /// active arch, or `None` when no logit scaling applies.
    pub fn final_logit_multiplier(&self) -> Option<f32> {
        match self.logit_scale {
            Some(ls) if ls != 0.0 => Some(
                if matches!(
                    self.dense_arch(),
                    DenseArch::Cohere | DenseArch::MuseGlimmer
                ) {
                    ls
                } else {
                    1.0 / ls
                },
            ),
            _ => None,
        }
    }

    /// Physical decoder blocks stored in the checkpoint.
    pub fn physical_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// KV-cache / execution depth after unrolling `num_loops`.
    pub fn kv_layers(&self) -> usize {
        self.num_hidden_layers.saturating_mul(self.num_loops.max(1))
    }

    /// Map an execution-slot index to the shared weight layer index.
    pub fn weight_layer_index(&self, exec_idx: usize) -> usize {
        let n = self.num_hidden_layers.max(1);
        exec_idx % n
    }

    #[cfg(test)]
    pub(crate) fn tiny_test() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
            embedding_scale: None,
            residual_scale: None,
            attention_scale: None,
            logit_scale: None,
            num_loops: 1,
            skip_loop_final_norm: false,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
            sliding_window: None,
            sliding_window_pattern: None,
            final_logit_softcap: None,
        }
    }
}

pub fn llama32_cfg_from_gguf(raw: &GgufFile) -> anyhow::Result<Llama32Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("llama");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("llama.")?;
            if arch_prefix == "llama" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> anyhow::Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("llama.embedding_length")? as usize;
    let num_attention_heads = get_u32("llama.attention.head_count")? as usize;
    let head_dim_key = get_u32("llama.attention.key_length")
        .ok()
        .map(|v| v as usize);
    let rope_dim = get_u32("llama.rope.dimension_count")
        .ok()
        .map(|v| v as usize);
    // Nemotron (and GLM-4) store `rope.dimension_count` as the PARTIAL rotary
    // width (e.g. 64) while the true head_dim is `hidden/heads` (e.g. 128) and
    // no `key_length` is present. The generic `key_length.or(rope_dim)` heuristic
    // would mistake that partial width for the head dim — so when the arch has
    // no explicit key_length, pin head_dim from `hidden/heads`.
    let derived_head_dim = if head_dim_key.is_none() && num_attention_heads > 0 {
        Some(hidden_size / num_attention_heads)
    } else {
        None
    };
    let head_dim = head_dim_key.or(derived_head_dim).or(rope_dim);

    let rope_scaling = match get_meta("llama.rope.scaling.type").and_then(MetaValue::as_str) {
        Some("none") | None => {
            // Llama 3.x often bakes scaling into rope_freqs.weight; HF fields may be absent.
            None
        }
        Some("linear") | Some("yarn") | Some("longrope") => {
            let factor = get_f32("llama.rope.scaling.factor")
                .or_else(|| get_f32("llama.rope.scale_linear"))
                .unwrap_or(1.0);
            let original = get_u32("llama.rope.scaling.original_context_length")
                .map(|v| v as usize)
                .unwrap_or(8192);
            Some(Llama32RopeScaling {
                factor,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original,
                rope_type: Llama32RopeType::Llama3,
            })
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported llama.rope.scaling.type: {other:?}"
            ));
        }
    };

    // Granite (IBM) scalar multipliers — read from `granite.*` metadata. These
    // are `None` for every other Llama-shaped arch, leaving the stock math
    // intact. Keys mirror llama.cpp `LLM_KV_{EMBEDDING,RESIDUAL,LOGIT}_SCALE`
    // and `LLM_KV_ATTENTION_SCALE`.
    let embedding_scale = get_f32("llama.embedding_scale");
    let residual_scale = get_f32("llama.residual_scale");
    let attention_scale = get_f32("llama.attention.scale");
    let logit_scale = get_f32("llama.logit_scale");

    // Interleaved local/global attention (Muse Glimmer). `sliding_window` is
    // llama.cpp's `n_swa`; `sliding_window_pattern` its `set_swa_pattern`
    // period. Absent for every plain-Llama GGUF, leaving all layers global.
    let sliding_window = get_u32("llama.attention.sliding_window")
        .ok()
        .map(|v| v as usize);
    let sliding_window_pattern = get_u32("llama.attention.sliding_window_pattern")
        .ok()
        .map(|v| v as usize);
    let final_logit_softcap = get_f32("llama.final_logit_softcapping");

    // RoPE flavor: the HF→GGUF converter permutes Q/K for llama.cpp's
    // interleaved (NORM / GPT-J) rope only for NORM-type arches (llama,
    // granite). NEOX-type arches (exaone, olmo2, cohere/command-r) are stored
    // un-permuted and must rotate with NeoX rotate-half.
    let rope_style = match arch_prefix {
        // NeoX rotate-half (no converter permutation)
        "phi3" | "phi4" | "exaone" | "olmo" | "olmo2" | "cohere" | "command-r" | "cohere2"
        | "glm4" | "chatglm" | "nemotron" => rlx_ir::RopeStyle::NeoX,
        // GPT-J interleaved (converter-permuted): llama, granite, muse-glimmer
        // (llama.cpp lists it under `LLAMA_ROPE_TYPE_NORM`), everything else.
        _ => rlx_ir::RopeStyle::GptJ,
    };

    Ok(Llama32Config {
        vocab_size: infer_vocab_size_from_gguf(raw),
        hidden_size,
        intermediate_size: get_u32("llama.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("llama.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("llama.attention.head_count_kv")? as usize,
        max_position_embeddings: get_u32("llama.context_length").unwrap_or(8192) as usize,
        rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
        rope_theta: get_f32("llama.rope.freq_base").unwrap_or(500_000.0) as f64,
        hidden_act: "silu".into(),
        tie_word_embeddings: get_bool("llama.tie_word_embeddings").unwrap_or_else(|| {
            // Llama-2 / TinyLlama GGUF often omits the flag; untied checkpoints
            // carry a separate `output.weight` tensor.
            !raw.tensors.contains_key("output.weight")
        }),
        attention_bias: false,
        head_dim,
        rope_scaling,
        embedding_scale,
        residual_scale,
        attention_scale,
        logit_scale,
        num_loops: get_u32("llama.num_loops").map(|v| v as usize).unwrap_or(1),
        skip_loop_final_norm: get_bool("llama.skip_loop_final_norm").unwrap_or(false),
        rope_style,
        gguf_arch: Some(arch_prefix.to_string()),
        // Partial-RoPE marker. Phi gates on `key_length` (unchanged). Nemotron /
        // GLM-4 carry a partial `rope.dimension_count` without `key_length`, so
        // also treat any rope_dim strictly smaller than the resolved head_dim as
        // the partial rotary width. Full-rope models (rope_dim == head_dim) keep
        // `None` → `n_rot()` falls back to the full head_dim.
        rope_dim: {
            let hd = head_dim.unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
            rope_dim.filter(|r| {
                (head_dim_key.is_some() && *r <= head_dim_key.unwrap()) || (*r > 0 && *r < hd)
            })
        },
        sliding_window,
        sliding_window_pattern,
        final_logit_softcap,
    })
}

/// Resolve vocab size from GGUF metadata / tensors. Llama-3 GGUF carries
/// `llama.vocab_size`; older llama-tagged files (TinyLlama, SmolLM2, …) often
/// only expose `tokenizer.ggml.tokens` or an embed row count.
fn infer_vocab_size_from_gguf(raw: &GgufFile) -> usize {
    if let Some(v) = raw
        .metadata
        .get("llama.vocab_size")
        .and_then(MetaValue::as_u32)
    {
        return v as usize;
    }
    if let Some(MetaValue::Array(tokens)) = raw.metadata.get("tokenizer.ggml.tokens") {
        if !tokens.is_empty() {
            return tokens.len();
        }
    }
    for name in ["token_embd.weight", "model.embed_tokens.weight"] {
        if let Some(t) = raw.tensors.get(name) {
            if !t.shape.is_empty() {
                return t.shape[0];
            }
        }
    }
    128_256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama32_1b_like() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let cfg: Llama32Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_group_size(), 4);
        assert!(cfg.rope_scaling.is_some());
    }

    #[test]
    fn gguf_vocab_inferred_from_tokenizer_tokens() {
        use rlx_gguf::GgmlType;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_llama32_vocab_{}_{}_{}.gguf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&9u64.to_le_bytes()); // metadata keys

        let write_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        let write_string_array = |buf: &mut Vec<u8>, k: &str, items: &[String]| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
        };

        write_str(&mut buf, "general.architecture", "llama");
        write_u32(&mut buf, "llama.embedding_length", 2048);
        write_u32(&mut buf, "llama.feed_forward_length", 5632);
        write_u32(&mut buf, "llama.block_count", 22);
        write_u32(&mut buf, "llama.attention.head_count", 32);
        write_u32(&mut buf, "llama.attention.head_count_kv", 4);
        write_u32(&mut buf, "llama.context_length", 2048);
        write_u32(&mut buf, "llama.rope.freq_base", 10_000);
        let vocab = 128u32;
        let tokens: Vec<String> = (0..vocab).map(|i| format!("t{i}")).collect();
        write_string_array(&mut buf, "tokenizer.ggml.tokens", &tokens);

        let embed_bytes = vocab as u64 * 2048 * 4;
        for (name, rows, cols, offset) in [
            ("token_embd.weight", vocab as u64, 2048u64, 0u64),
            ("output.weight", 2048u64, vocab as u64, embed_bytes),
        ] {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&rows.to_le_bytes());
            buf.extend_from_slice(&cols.to_le_bytes());
            buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        let n_floats = (vocab as usize * 2048) * 2;
        for _ in 0..n_floats {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let raw = rlx_gguf::GgufFile::from_path(&path).expect("parse tinyllama-like gguf");
        let cfg = llama32_cfg_from_gguf(&raw).expect("llama32 config");
        assert_eq!(cfg.vocab_size, vocab as usize);
        assert!(!cfg.tie_word_embeddings);
        std::fs::remove_file(path).ok();
    }

    /// Granite `granite.*` scalar multipliers + NORM/GPT-J rope parse from GGUF.
    #[test]
    fn gguf_granite_scalars_parse() {
        use rlx_gguf::GgmlType;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_llama32_granite_{}_{}.gguf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&12u64.to_le_bytes()); // metadata keys

        let write_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        let write_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        write_str(&mut buf, "general.architecture", "granite");
        write_u32(&mut buf, "granite.embedding_length", 2048);
        write_u32(&mut buf, "granite.feed_forward_length", 8192);
        write_u32(&mut buf, "granite.block_count", 40);
        write_u32(&mut buf, "granite.attention.head_count", 32);
        write_u32(&mut buf, "granite.attention.head_count_kv", 8);
        write_u32(&mut buf, "granite.vocab_size", 49159);
        write_f32(&mut buf, "granite.embedding_scale", 12.0);
        write_f32(&mut buf, "granite.residual_scale", 0.22);
        write_f32(&mut buf, "granite.attention.scale", 0.015625);
        write_f32(&mut buf, "granite.logit_scale", 8.0);
        write_f32(&mut buf, "granite.rope.freq_base", 10_000_000.0);

        // one dummy f32 tensor
        let name = "token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let raw = rlx_gguf::GgufFile::from_path(&path).expect("parse granite gguf");
        let cfg = llama32_cfg_from_gguf(&raw).expect("granite config");
        assert_eq!(cfg.gguf_arch.as_deref(), Some("granite"));
        assert_eq!(cfg.embedding_scale, Some(12.0));
        assert_eq!(cfg.residual_scale, Some(0.22));
        assert_eq!(cfg.attention_scale, Some(0.015625));
        assert_eq!(cfg.logit_scale, Some(8.0));
        assert!(cfg.has_granite_scalars());
        assert_eq!(cfg.attn_score_scale(), Some(0.015625));
        // Granite is NORM/permuted → GPT-J rotate flavor (same as llama).
        assert_eq!(cfg.rope_style, rlx_ir::RopeStyle::GptJ);
        assert!((cfg.rope_theta - 10_000_000.0).abs() < 1.0);
        std::fs::remove_file(path).ok();
    }

    /// Muse Glimmer 30B hparams parse from a `muse-glimmer.*` GGUF header, with
    /// the exact key/value set `unsloth/Muse-Glimmer-30B-GGUF` ships.
    #[test]
    fn gguf_muse_glimmer_hparams_parse() {
        use rlx_gguf::GgmlType;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_llama32_muse_{}_{}.gguf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&14u64.to_le_bytes()); // metadata keys

        let write_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        let write_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        write_str(&mut buf, "general.architecture", "muse-glimmer");
        write_u32(&mut buf, "muse-glimmer.block_count", 52);
        write_u32(&mut buf, "muse-glimmer.context_length", 131_072);
        write_u32(&mut buf, "muse-glimmer.embedding_length", 6656);
        write_u32(&mut buf, "muse-glimmer.feed_forward_length", 19_968);
        write_u32(&mut buf, "muse-glimmer.attention.head_count", 32);
        write_u32(&mut buf, "muse-glimmer.attention.head_count_kv", 2);
        write_u32(&mut buf, "muse-glimmer.attention.key_length", 128);
        write_u32(&mut buf, "muse-glimmer.attention.value_length", 128);
        write_u32(&mut buf, "muse-glimmer.attention.sliding_window", 2048);
        write_u32(&mut buf, "muse-glimmer.attention.sliding_window_pattern", 4);
        write_f32(&mut buf, "muse-glimmer.rope.freq_base", 500_000.0);
        write_f32(&mut buf, "muse-glimmer.final_logit_softcapping", 20.0);
        write_f32(&mut buf, "muse-glimmer.logit_scale", 0.196_116_13);

        // token_embd + a separate output.weight → untied LM head.
        for (name, offset) in [("token_embd.weight", 0u64), ("output.weight", 16u64)] {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&4u64.to_le_bytes());
            buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..8 {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let raw = rlx_gguf::GgufFile::from_path(&path).expect("parse muse-glimmer gguf");
        let cfg = llama32_cfg_from_gguf(&raw).expect("muse-glimmer config");
        std::fs::remove_file(path).ok();

        assert_eq!(cfg.gguf_arch.as_deref(), Some("muse-glimmer"));
        assert_eq!(cfg.dense_arch(), DenseArch::MuseGlimmer);
        assert_eq!(cfg.hidden_size, 6656);
        assert_eq!(cfg.num_hidden_layers, 52);
        assert_eq!(cfg.intermediate_size, 19_968);
        // head_dim comes from key_length (128), NOT hidden/heads (208).
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.q_proj_dim(), 4096);
        assert_eq!(cfg.kv_proj_dim(), 256);
        assert_eq!(cfg.kv_group_size(), 16);
        assert_eq!(cfg.sliding_window, Some(2048));
        assert_eq!(cfg.sliding_window_pattern, Some(4));
        assert_eq!(cfg.final_logit_softcap, Some(20.0));
        assert!(!cfg.tie_word_embeddings);
        // llama.cpp lists MUSE_GLIMMER under LLAMA_ROPE_TYPE_NORM → the
        // converter permutes Q/K, so rotate with the interleaved GPT-J flavor.
        assert_eq!(cfg.rope_style, rlx_ir::RopeStyle::GptJ);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1.0);
        // `logit_scale` MULTIPLIES here (Cohere semantics), it does not divide.
        let m = cfg.final_logit_multiplier().expect("logit multiplier");
        assert!((m - 0.196_116_13).abs() < 1e-6, "got {m}");
        // Post-norms use 1e-8, pre-norms the GGUF rms eps.
        assert_eq!(cfg.post_norm_eps(), 1e-8);
        assert!(cfg.uses_attn_out_gate());
        assert!(cfg.normalizes_input_embeddings());
        assert!(cfg.needs_arch_packed_builder());
    }

    /// llama.cpp `set_swa_pattern(4)` ⇒ layers 0,1,2 sliding-window (with RoPE)
    /// and layer 3 full-attention (NoPE), repeating. Layer 51 (the last of 52)
    /// lands on a global layer.
    #[test]
    fn muse_glimmer_local_global_layer_pattern() {
        let mut cfg = Llama32Config::tiny_test();
        cfg.gguf_arch = Some("muse-glimmer".into());
        cfg.sliding_window = Some(2048);
        cfg.sliding_window_pattern = Some(4);
        cfg.num_hidden_layers = 52;

        for (idx, want_local) in [
            (0, true),
            (1, true),
            (2, true),
            (3, false),
            (4, true),
            (7, false),
            (50, true),
            (51, false),
        ] {
            assert_eq!(cfg.is_swa_layer(idx), want_local, "layer {idx} locality");
            // RoPE runs on the local layers; the global ones are NoPE.
            assert_eq!(cfg.is_nope_layer(idx), !want_local, "layer {idx} nope");
            assert_eq!(
                cfg.attn_window_for_layer(idx),
                want_local.then_some(2048),
                "layer {idx} window"
            );
        }
    }

    /// A plain Llama GGUF must keep every layer global/RoPE'd — the new
    /// interleave helpers are inert without the `muse-glimmer` arch tag.
    #[test]
    fn llama_arch_has_no_sliding_window_layers() {
        let cfg = Llama32Config::tiny_test();
        assert_eq!(cfg.dense_arch(), DenseArch::Llama);
        assert!(cfg.swa_pattern().is_none());
        for idx in 0..8 {
            assert!(!cfg.is_swa_layer(idx));
            assert!(!cfg.is_nope_layer(idx));
            assert_eq!(cfg.attn_window_for_layer(idx), None);
        }
        assert_eq!(cfg.post_norm_eps(), cfg.rms_norm_eps as f32);
        assert!(!cfg.uses_attn_out_gate());
        assert!(!cfg.normalizes_input_embeddings());
    }

    /// Cohere2's NoPE predicate is unchanged by the shared `is_nope_layer`
    /// helper: global layers are `(i+1) % 4 == 0`, and it has no SWA window.
    #[test]
    fn cohere2_nope_pattern_preserved() {
        let mut cfg = Llama32Config::tiny_test();
        cfg.gguf_arch = Some("cohere2".into());
        assert_eq!(cfg.cohere2_nope_pattern(), Some(4));
        for (idx, want_nope) in [(0, false), (1, false), (2, false), (3, true), (7, true)] {
            assert_eq!(cfg.is_nope_layer(idx), want_nope, "layer {idx}");
        }
        // Cohere2 doesn't drive the windowed-mask path.
        assert!(cfg.swa_pattern().is_none());
        assert_eq!(cfg.attn_window_for_layer(3), None);
    }
}
