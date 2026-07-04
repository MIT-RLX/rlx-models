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

//! Gemma family configuration — HF `config.json` and GGUF metadata.

use rlx_flow::blocks::{GemmaLayerStyle, gemma_strided_layer_mask, gemma2_layer_mask};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_ir::op::MaskKind;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GemmaArch {
    #[default]
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl GemmaArch {
    pub fn sliding_window_stride(self) -> usize {
        match self {
            GemmaArch::Gemma3 | GemmaArch::Gemma4 => 6,
            _ => 0,
        }
    }

    fn from_gguf_tag(tag: &str) -> Self {
        match tag {
            "gemma2" => GemmaArch::Gemma2,
            "gemma3" | "gemma3n" => GemmaArch::Gemma3,
            "gemma4" | "gemma4moe" | "gemma4_unified" | "gemma4_unified_text" => GemmaArch::Gemma4,
            _ => GemmaArch::Gemma,
        }
    }
}

/// One entry in the Gemma 4 `text_config.layer_types` array. The
/// repeating "5 sliding + 1 full" Gemma 3 pattern is just a special
/// case of this richer per-layer schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaLayerType {
    SlidingAttention,
    FullAttention,
}

/// Nested rope_parameters block. Gemma 4 12B carries per-attention-kind
/// rope parameters: sliding layers use `theta=1e4` with full rotation,
/// full-attention layers use `theta=1e6` with `partial_rotary_factor`
/// (p-RoPE rotating only the leading slice).
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct GemmaRopeParameters {
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<GemmaRopeKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GemmaRopeKind {
    #[default]
    Default,
    Proportional,
    Linear,
    Dynamic,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GemmaRopeMap {
    #[serde(default)]
    pub sliding_attention: Option<GemmaRopeParameters>,
    #[serde(default)]
    pub full_attention: Option<GemmaRopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmaConfig {
    #[serde(default)]
    pub arch: GemmaArch,
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
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    pub effective_num_layers: Option<usize>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_used: usize,
    #[serde(default)]
    pub expert_ffn_size: usize,
    #[serde(default = "default_expert_weights_scale")]
    pub expert_weights_scale: f32,

    // ── Gemma 4 unified additions ──────────────────────────────────
    /// Per-layer attention kind. Empty for Gemma <=3 — fall back to
    /// the strided pattern derived from `arch.sliding_window_stride`.
    #[serde(default)]
    pub layer_types: Vec<GemmaLayerType>,
    /// Per-attention-kind rope settings. Empty for Gemma <=3.
    #[serde(default)]
    pub rope_parameters: GemmaRopeMap,
    /// Head dim for full-attention (global) layers. `None` ⇒ reuse
    /// the base `head_dim`. Gemma 4 12B sets this to 512 while the
    /// sliding `head_dim` stays at 256.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    /// Num KV heads for full-attention layers. `None` ⇒ reuse the
    /// base `num_key_value_heads`. Gemma 4 12B sets this to 1.
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    /// When true (Gemma 4 12B), the K projection is reused as V at
    /// load time — weights only ship `.k_proj` and `.v_proj` becomes
    /// an alias.
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// When `"vision"`, media placeholder spans use bidirectional
    /// attention on sliding layers (Gemma 4 unified).
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,

    // ── Gemma 4 E2B (mobile / edge) additions ──────────────────────
    /// Per-Layer Embedding width per layer (Gemma 4 E2B: 256). `0` ⇒
    /// the model has no Per-Layer Embeddings (flagship / GGUF path).
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    /// Vocabulary size of the per-layer embedding table. `0` ⇒ reuse
    /// `vocab_size`. Gemma 4 E2B: 262144.
    #[serde(default)]
    pub vocab_size_per_layer_input: usize,
    /// Number of trailing decoder layers that *reuse* (rather than
    /// recompute) KV from an earlier same-type layer. `0` ⇒ every
    /// layer computes its own KV (flagship). Gemma 4 E2B: 20.
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    /// When true, KV-shared layers double their MLP intermediate size
    /// (Gemma 4 E2B: 6144 → 12288 on layers ≥ `first_kv_shared_layer`).
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    /// When true the (flagship / A4B) MoE block is active. Gemma 4 E2B
    /// is dense (`false`).
    #[serde(default)]
    pub enable_moe_block: bool,
    /// End-of-generation token ids (from GGUF + llama.cpp EOG set). When
    /// non-empty, [`Self::is_eog_token`] matches greedy stop semantics.
    #[serde(default)]
    pub eog_token_ids: Vec<u32>,
}

impl GemmaConfig {
    /// Whether `tok` ends generation (llama.cpp `is_eog_token` parity).
    pub fn is_eog_token(&self, tok: u32) -> bool {
        self.eog_token_ids.contains(&tok)
    }
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_expert_weights_scale() -> f32 {
    1.0
}

impl GemmaConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        // Gemma 4 unified (e.g. `google/gemma-4-12B`) nests the LM
        // hyperparameters under `text_config` because the same file
        // also carries vision + audio configs. Pick that subtree if
        // it looks like the unified shape, otherwise stay flat.
        let value: serde_json::Value = serde_json::from_str(&data)?;
        let lm_value = match value.get("text_config") {
            Some(tc) if tc.is_object() => tc.clone(),
            _ => value.clone(),
        };
        let lm_value = normalize_hf_null_usize_fields(lm_value);
        let mut cfg: Self = serde_json::from_value(lm_value)?;
        if cfg.arch == GemmaArch::Gemma {
            cfg.arch = infer_arch_from_json(&data);
        }
        Ok(cfg)
    }

    pub fn from_gguf(raw: &GgufFile) -> anyhow::Result<Self> {
        gemma_cfg_from_gguf(raw)
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

    pub fn layer_style(&self) -> GemmaLayerStyle {
        match self.arch {
            GemmaArch::Gemma => GemmaLayerStyle::Gemma,
            GemmaArch::Gemma2 => GemmaLayerStyle::Gemma2,
            GemmaArch::Gemma3 => GemmaLayerStyle::Gemma3,
            GemmaArch::Gemma4 => GemmaLayerStyle::Gemma4,
        }
    }

    pub fn active_num_layers(&self) -> usize {
        self.effective_num_layers.unwrap_or(self.num_hidden_layers)
    }

    pub fn is_moe(&self) -> bool {
        self.arch == GemmaArch::Gemma4 && self.num_experts > 0
    }

    // ── Gemma 4 E2B: Per-Layer Embeddings + KV sharing ─────────────

    /// Whether this checkpoint carries Per-Layer Embeddings (Gemma 4
    /// E2B/E4B mobile). Drives the extra `embed_tokens_per_layer`,
    /// `per_layer_*` projection/gate weights in the builder.
    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    /// Width of one per-layer embedding slice (`0` when absent).
    pub fn ple_width(&self) -> usize {
        self.hidden_size_per_layer_input
    }

    /// Vocab of the per-layer embedding table (defaults to `vocab_size`).
    pub fn ple_vocab_size(&self) -> usize {
        if self.vocab_size_per_layer_input > 0 {
            self.vocab_size_per_layer_input
        } else {
            self.vocab_size
        }
    }

    /// Index of the first decoder layer that *reuses* (shares) KV from
    /// an earlier layer. Layers `< first_kv_shared_layer` compute fresh
    /// KV; layers `>=` it reuse. Returns `num_hidden_layers` (i.e. no
    /// sharing) when `num_kv_shared_layers == 0`.
    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    /// Whether layer `i` reuses KV from an earlier same-type layer.
    pub fn is_kv_shared_layer(&self, layer: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer >= self.first_kv_shared_layer()
    }

    /// The source layer whose KV a shared layer reuses: the last
    /// *fresh* layer (`< first_kv_shared_layer`) of the **same**
    /// attention kind (sliding vs full). Returns `layer` itself when
    /// the layer is not shared (it computes its own KV).
    pub fn kv_source_layer(&self, layer: usize) -> usize {
        if !self.is_kv_shared_layer(layer) {
            return layer;
        }
        let want_full = self.is_full_attention_layer(layer);
        let boundary = self.first_kv_shared_layer();
        (0..boundary)
            .rev()
            .find(|&src| self.is_full_attention_layer(src) == want_full)
            .unwrap_or(layer)
    }

    /// MLP intermediate size for layer `i`. Gemma 4 E2B doubles the
    /// intermediate width on KV-shared layers when `use_double_wide_mlp`
    /// is set; all other layers use the base `intermediate_size`.
    pub fn layer_intermediate_size(&self, layer: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared_layer(layer) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    /// Gemma 4 unified: bidirectional attention inside vision/audio spans.
    pub fn use_bidirectional_vision(&self) -> bool {
        self.use_bidirectional_attention.as_deref() == Some("vision")
    }

    pub fn expert_ffn_dim(&self) -> usize {
        if self.expert_ffn_size > 0 {
            self.expert_ffn_size
        } else {
            self.intermediate_size
        }
    }

    pub fn attn_score_scale(&self) -> Option<f32> {
        match self.arch {
            GemmaArch::Gemma => None,
            // llama.cpp gemma4.cpp:11 "Gemma4 uses self.scaling = 1.0
            // (no pre-attn scaling)". Q is RMS-normed per-head before
            // attention so Q·K is already bounded — applying the
            // standard 1/sqrt(head_dim) on top *crushes* the scores
            // (12B head_dim=256 → 16× too small). Use unit scale.
            GemmaArch::Gemma4 => Some(1.0),
            GemmaArch::Gemma2 | GemmaArch::Gemma3 => {
                if let Some(s) = self.query_pre_attn_scalar {
                    // HF / llama.cpp / mlx: `query_pre_attn_scalar**-0.5` (scale Q, not 1/s).
                    Some(1.0 / s.sqrt())
                } else {
                    Some(1.0 / (self.head_dim() as f32).sqrt())
                }
            }
        }
    }

    /// Per-layer attention options driving the prefill self-attn block:
    /// `(mask kind, softmax score scale, attention logit soft-cap)`.
    /// The mask varies across Gemma variants:
    ///
    /// - Gemma 1 / no sliding window → all-causal.
    /// - Gemma 2 → alternating sliding-window via [`gemma2_layer_mask`].
    /// - Gemma 3 / 4 → strided pattern via
    ///   [`gemma_strided_layer_mask`] (stride-6: every 6th layer is
    ///   full causal, others are sliding-window).
    /// Sliding-window size for layer `i`'s KV ring buffer during decode
    /// (Gemma 3/4 ISWA). `None` for full-attention / non-sliding layers.
    pub fn layer_sliding_kv_window(&self, layer: usize) -> Option<usize> {
        match (self.arch, self.sliding_window) {
            (GemmaArch::Gemma3 | GemmaArch::Gemma4, Some(w)) if !self.is_full_attention_layer(layer) => {
                Some(w)
            }
            (GemmaArch::Gemma2, Some(w)) if layer % 2 == 0 => Some(w),
            _ => None,
        }
    }

    /// Per-layer `(kv_dim, window)` trim spec for [`LayerKvCache::trim_sliding_window_per_layer`].
    pub fn sliding_kv_trim_spec(&self, kv_dims: &[usize]) -> Vec<Option<(usize, usize)>> {
        (0..self.num_hidden_layers)
            .map(|layer| {
                let kd = kv_dims.get(layer).copied().unwrap_or_else(|| {
                    self.layer_num_kv_heads(layer) * self.layer_head_dim(layer)
                });
                self.layer_sliding_kv_window(layer).map(|w| (kd, w))
            })
            .collect()
    }

    pub fn layer_attn_options(&self, layer: usize) -> (MaskKind, Option<f32>, Option<f32>) {
        let scale = self.attn_score_scale();
        let softcap = self.attn_logit_softcapping;
        let mask = match (self.arch, self.sliding_window) {
            (_, None) => MaskKind::Causal,
            (GemmaArch::Gemma2, Some(w)) => gemma2_layer_mask(layer, w),
            (GemmaArch::Gemma3 | GemmaArch::Gemma4, Some(w)) => {
                gemma_strided_layer_mask(layer, w, self.arch.sliding_window_stride())
            }
            _ => MaskKind::Causal,
        };
        (mask, scale, softcap)
    }

    #[cfg(test)]
    pub(crate) fn tiny_test() -> Self {
        Self {
            arch: GemmaArch::Gemma,
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            head_dim: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            sliding_window: None,
            query_pre_attn_scalar: None,
            effective_num_layers: None,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            expert_weights_scale: 1.0,
            layer_types: Vec::new(),
            rope_parameters: GemmaRopeMap::default(),
            global_head_dim: None,
            num_global_key_value_heads: None,
            attention_k_eq_v: false,
            use_bidirectional_attention: None,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            use_double_wide_mlp: false,
            enable_moe_block: false,
            eog_token_ids: Vec::new(),
        }
    }

    // ── Per-layer dispatch (Gemma 4 unified). ──────────────────────
    //
    // For Gemma 1/2/3 the `layer_types` array is empty and these
    // helpers reduce to the existing strided pattern; for Gemma 4
    // they read the explicit array so each layer can ship its own
    // (head_dim, num_kv_heads, n_rot, rope_theta).

    /// Whether layer `i` is a full-attention (global) layer rather
    /// than a sliding-window one. Falls back to the strided pattern
    /// (every `stride`-th layer is global) when `layer_types` is
    /// unset.
    pub fn is_full_attention_layer(&self, layer: usize) -> bool {
        if !self.layer_types.is_empty() {
            return matches!(
                self.layer_types.get(layer),
                Some(GemmaLayerType::FullAttention),
            );
        }
        let stride = self.arch.sliding_window_stride();
        stride > 1 && (layer + 1).is_multiple_of(stride)
    }

    /// Per-layer head_dim. Sliding layers always use the base
    /// `head_dim`; full-attention layers use `global_head_dim` when
    /// set (Gemma 4 12B: 512 vs base 256).
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_full_attention_layer(layer) {
            self.global_head_dim.unwrap_or_else(|| self.head_dim())
        } else {
            self.head_dim()
        }
    }

    /// Per-layer V-aliased-to-K flag. For Gemma 4 specifically:
    /// SWA layers ship an independent v_proj weight; full-attention
    /// layers (every 6th) omit v_proj and alias V to K. Other arches
    /// fall back to the uniform `attention_k_eq_v`.
    pub fn layer_k_eq_v(&self, layer: usize) -> bool {
        if matches!(self.arch, GemmaArch::Gemma4) {
            // HF `use_alternative_attention = attention_k_eq_v && !is_sliding`.
            // The flagship (12B) sets `attention_k_eq_v=true` so full-attention
            // layers alias V→K; E2B sets it false and ships a real v_proj on
            // every layer, so it must NOT alias.
            return self.attention_k_eq_v && self.is_full_attention_layer(layer);
        }
        self.attention_k_eq_v
    }

    /// Per-layer KV head count. Sliding layers use
    /// `num_key_value_heads`; full-attention layers use
    /// `num_global_key_value_heads` when set (Gemma 4 12B: 1 vs 8).
    pub fn layer_num_kv_heads(&self, layer: usize) -> usize {
        if self.is_full_attention_layer(layer) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    /// Number of leading per-head dimensions that get RoPE-rotated
    /// in layer `i`. Returns `layer_head_dim` for "default" RoPE,
    /// or `floor(partial_rotary_factor * head_dim)` for p-RoPE.
    pub fn layer_n_rot(&self, layer: usize) -> usize {
        let dh = self.layer_head_dim(layer);
        let params = self.layer_rope_parameters(layer);
        let kind = params
            .and_then(|p| p.rope_type)
            .unwrap_or(GemmaRopeKind::Default);
        let factor = params.and_then(|p| p.partial_rotary_factor);
        match (kind, factor) {
            (GemmaRopeKind::Proportional, Some(f)) if f > 0.0 && f < 1.0 => {
                ((dh as f32) * f).floor() as usize
            }
            _ => dh,
        }
    }

    /// RoPE base frequency for layer `i`. Falls back to the
    /// top-level `rope_theta` when the unified map omits the entry.
    pub fn layer_rope_theta(&self, layer: usize) -> f64 {
        self.layer_rope_parameters(layer)
            .and_then(|p| p.rope_theta)
            .map(|t| t as f64)
            .unwrap_or(self.rope_theta)
    }

    fn layer_rope_parameters(&self, layer: usize) -> Option<&GemmaRopeParameters> {
        if self.is_full_attention_layer(layer) {
            self.rope_parameters.full_attention.as_ref()
        } else {
            self.rope_parameters.sliding_attention.as_ref()
        }
    }
}

/// HF dense Gemma 4 checkpoints use JSON `null` for unused MoE keys.
fn normalize_hf_null_usize_fields(mut value: serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    for key in [
        "num_experts",
        "num_experts_used",
        "top_k_experts",
        "expert_ffn_size",
        "moe_intermediate_size",
        "hidden_size_per_layer_input",
    ] {
        if obj.get(key).is_some_and(|v| v.is_null()) {
            obj.insert(key.to_string(), serde_json::Value::from(0usize));
        }
    }
    value
}

fn infer_arch_from_json(raw: &str) -> GemmaArch {
    // Detect Gemma 4 first — its unified config also contains a
    // nested `gemma4_unified_text` model_type that we want to catch
    // even when the outer `model_type` is `gemma4_unified` or the
    // architecture is `Gemma4UnifiedForConditionalGeneration`.
    if raw.contains("\"gemma4_unified\"")
        || raw.contains("\"gemma4_unified_text\"")
        || raw.contains("\"gemma4\"")
        || raw.contains("\"gemma4moe\"")
        || raw.contains("Gemma4UnifiedForConditionalGeneration")
        || raw.contains("Gemma4ForCausalLM")
    {
        return GemmaArch::Gemma4;
    }
    if raw.contains("\"model_type\"") {
        if raw.contains("\"gemma2\"") {
            return GemmaArch::Gemma2;
        }
        if raw.contains("\"gemma3\"") {
            return GemmaArch::Gemma3;
        }
    }
    GemmaArch::Gemma
}

/// llama.cpp encodes Gemma 4 proportional (p-)RoPE in `rope_freqs.weight`:
/// rotated dim pairs are `1.0`, unrotated pairs are ~`1e30` so RoPE skips
/// them (see `conversion/gemma.py` `generate_extra_tensors`).
fn infer_gemma4_full_partial_rotary(raw: &GgufFile, global_head_dim: usize) -> Option<f32> {
    if global_head_dim == 0 {
        return None;
    }
    let (factors, _) = raw.dequant_f32("rope_freqs.weight").ok()?;
    if factors.is_empty() {
        return None;
    }
    const SUPPRESS_THRESH: f32 = 1e20;
    let rotated_pairs = factors.iter().filter(|&&f| f < SUPPRESS_THRESH).count();
    if rotated_pairs == 0 || rotated_pairs >= factors.len() {
        return None;
    }
    let n_rot = rotated_pairs * 2;
    Some(n_rot as f32 / global_head_dim as f32)
}

/// Build [`GemmaRopeMap`] for Gemma 3 GGUF checkpoints.
///
/// HF / llama.cpp use `rope_theta=1e4` on sliding-window layers and
/// `rope_theta=1e6` on strided full-attention layers. GGUF often ships only
/// `gemma3.rope.freq_base` (the global value); sliding layers default to 10k.
fn gemma3_rope_map_from_gguf(get_f32: &impl Fn(&str) -> Option<f32>) -> GemmaRopeMap {
    let full_theta = get_f32("gemma.rope.freq_base");
    let swa_theta = get_f32("gemma.rope.freq_base_swa").or(Some(10_000.0));
    let mk = |theta: f32| GemmaRopeParameters {
        rope_theta: Some(theta),
        rope_type: Some(GemmaRopeKind::Default),
        partial_rotary_factor: None,
    };
    GemmaRopeMap {
        sliding_attention: swa_theta.map(mk),
        full_attention: full_theta.map(mk),
    }
}

/// Build [`GemmaRopeMap`] for Gemma 4 GGUF checkpoints.
fn gemma4_rope_map_from_gguf(
    raw: &GgufFile,
    get_f32: &impl Fn(&str) -> Option<f32>,
    get_u32_opt: &impl Fn(&str) -> Option<u32>,
    swa_head_dim: Option<usize>,
    global_head_dim: Option<usize>,
) -> GemmaRopeMap {
    let full_theta = get_f32("gemma.rope.freq_base");
    let swa_theta = get_f32("gemma.rope.freq_base_swa");

    let full_partial = global_head_dim.and_then(|ghd| {
        infer_gemma4_full_partial_rotary(raw, ghd).or_else(|| {
            // Flagship Gemma 4 12B/31B with distinct global head dim.
            swa_head_dim.filter(|&swa| ghd > swa).map(|_| 0.25)
        })
    });

    let swa_partial = match (swa_head_dim, get_u32_opt("gemma.rope.dimension_count_swa")) {
        (Some(hd), Some(n_rot)) if (n_rot as usize) < hd => Some(n_rot as f32 / hd as f32),
        _ => None,
    };

    GemmaRopeMap {
        sliding_attention: swa_theta.map(|t| GemmaRopeParameters {
            rope_theta: Some(t),
            rope_type: Some(if swa_partial.is_some() {
                GemmaRopeKind::Proportional
            } else {
                GemmaRopeKind::Default
            }),
            partial_rotary_factor: swa_partial,
        }),
        full_attention: full_theta.map(|t| GemmaRopeParameters {
            rope_theta: Some(t),
            rope_type: full_partial.map(|_| GemmaRopeKind::Proportional),
            partial_rotary_factor: full_partial,
        }),
    }
}

pub fn gemma_cfg_from_gguf(raw: &GgufFile) -> anyhow::Result<GemmaConfig> {
    let arch_tag = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("gemma");
    let arch_prefix = arch_tag;
    let arch = GemmaArch::from_gguf_tag(arch_tag);

    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("gemma.")?;
            if arch_prefix == "gemma" {
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
    // Some Gemma 4 GGUF writers encode per-layer attention as an
    // Array instead of a scalar — e.g. `gemma4.attention.head_count_kv`
    // can be `[60 × i32]` (16 for sliding layers, 4 for global). For
    // the uniform-attention `GemmaConfig` we take the first array
    // element (typically the dominant sliding-layer value) and let
    // the per-layer overrides on `GemmaConfig` (e.g.
    // `num_global_key_value_heads`) capture the global-layer
    // exception. Falls back to scalar `as_u32` for older writers.
    let get_first_u32 = |k: &str| -> anyhow::Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_first_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_u32_opt = |k: &str| -> Option<u32> { get_meta(k).and_then(MetaValue::as_u32) };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("gemma.embedding_length")? as usize;
    let num_attention_heads = get_first_u32("gemma.attention.head_count")? as usize;
    // Newer GGUF writers (Gemma 4) don't include `gemma.vocab_size`;
    // infer it from the tokenizer.ggml.tokens array length when the
    // scalar isn't present. Falls back to 256_000 only if neither
    // path resolves.
    let resolved_vocab_size: usize = get_u32("gemma.vocab_size")
        .ok()
        .map(|v| v as usize)
        .or_else(|| {
            raw.metadata
                .get("tokenizer.ggml.tokens")
                .and_then(MetaValue::as_array)
                .map(|a| a.len())
        })
        .unwrap_or(256_000);
    // Gemma 4 has TWO layer types with DIFFERENT head_dims:
    //   - Sliding-window layers (majority): key_length_swa, e.g. 256
    //   - Full-attention layers (every 6th): key_length, e.g. 512
    // For non-Gemma-4 archs key_length is per-head directly.
    let head_dim = if matches!(arch, GemmaArch::Gemma4) {
        // SWA dim is the default; full layers get global_head_dim below.
        get_first_u32("gemma.attention.key_length_swa")
            .ok()
            .or_else(|| get_first_u32("gemma.attention.key_length").ok())
            .map(|v| v as usize)
    } else {
        get_first_u32("gemma.attention.key_length")
            .ok()
            .or_else(|| get_first_u32("gemma.rope.dimension_count").ok())
            .map(|v| v as usize)
    };

    // Gemma 4: gather full-attention layer dims when distinct from SWA.
    // The metadata stores head_count_kv as a 48-element array — find any
    // value that differs from the first element; that's the global head
    // count. Same for key_length (full head_dim).
    let global_head_dim = if matches!(arch, GemmaArch::Gemma4) {
        let swa = head_dim.unwrap_or(0);
        let full = get_first_u32("gemma.attention.key_length")
            .ok()
            .map(|v| v as usize);
        match full {
            Some(f) if f != swa => Some(f),
            _ => None,
        }
    } else {
        None
    };
    let num_global_key_value_heads = if matches!(arch, GemmaArch::Gemma4) {
        let sliding_kv = get_first_u32("gemma.attention.head_count_kv")
            .ok()
            .map(|v| v as usize);
        let kv_array = raw
            .metadata
            .get(&format!("{arch_prefix}.attention.head_count_kv"))
            .or_else(|| raw.metadata.get("gemma.attention.head_count_kv"))
            .and_then(MetaValue::as_array);
        if let (Some(swa_kv), Some(arr)) = (sliding_kv, kv_array) {
            arr.iter().find_map(|v| match v {
                MetaValue::I32(n) if (*n as usize) != swa_kv => Some(*n as usize),
                MetaValue::U32(n) if (*n as usize) != swa_kv => Some(*n as usize),
                _ => None,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(GemmaConfig {
        arch,
        vocab_size: resolved_vocab_size,
        hidden_size,
        intermediate_size: get_u32("gemma.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("gemma.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_first_u32("gemma.attention.head_count_kv")? as usize,
        max_position_embeddings: get_u32("gemma.context_length").unwrap_or(8192) as usize,
        rms_norm_eps: get_f32("gemma.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        rope_theta: get_f32("gemma.rope.freq_base").unwrap_or(10_000.0) as f64,
        tie_word_embeddings: get_bool("gemma.tie_word_embeddings").unwrap_or(true),
        attention_bias: get_bool("gemma.attention.bias").unwrap_or(false),
        head_dim,
        attn_logit_softcapping: if std::env::var("RLX_GEMMA_NO_ATTN_SOFTCAP").as_deref() == Ok("1") {
            None
        } else if let Ok(v) = std::env::var("RLX_GEMMA_ATTN_SOFTCAP_FORCE") {
            v.parse::<f32>().ok()
        } else {
            get_f32("gemma.attn_logit_softcapping")
        },
        final_logit_softcapping: get_f32("gemma.final_logit_softcapping"),
        sliding_window: get_u32("gemma.attention.sliding_window")
            .ok()
            .map(|v| v as usize),
        query_pre_attn_scalar: get_f32("gemma.attention.query_pre_attn_scalar"),
        effective_num_layers: get_u32("gemma.block_count_effective")
            .ok()
            .map(|v| v as usize),
        num_experts: get_u32("gemma.expert_count").unwrap_or(0) as usize,
        num_experts_used: get_u32("gemma.expert_used_count").unwrap_or(0) as usize,
        expert_ffn_size: get_u32("gemma.expert_feed_forward_length").unwrap_or(0) as usize,
        expert_weights_scale: get_f32("gemma.expert_weights_scale").unwrap_or(1.0),
        // GGUF doesn't carry the Gemma 4 unified per-layer schema
        // yet; the dense path falls back to the strided pattern and
        // uniform head dims that match every Gemma 4 GGUF currently
        // emitted by llama.cpp.
        layer_types: Vec::new(),
        // Gemma 4 ships per-attention-kind rope params in the GGUF:
        //   gemma4.rope.freq_base       = 1e6 (full-attention layers)
        //   gemma4.rope.freq_base_swa   = 1e4 (sliding-window layers)
        // Without populating this, layer_rope_theta returns the global
        // freq_base for ALL layers — SWA layers RoPE with the wrong
        // base → wrong K rotation → bad attention scores. Build the
        // GemmaRopeMap from the metadata so layer_rope_theta(swa) and
        // layer_rope_theta(full) split correctly.
        rope_parameters: if matches!(arch, GemmaArch::Gemma4) {
            gemma4_rope_map_from_gguf(raw, &get_f32, &get_u32_opt, head_dim, global_head_dim)
        } else if matches!(arch, GemmaArch::Gemma3) {
            gemma3_rope_map_from_gguf(&get_f32)
        } else {
            GemmaRopeMap::default()
        },
        global_head_dim,
        num_global_key_value_heads,
        // For Gemma 4, V-aliased-to-K is PER-LAYER: only full-attention
        // layers omit v_proj. The base scalar stays true (matches the
        // common case + existing tests + Gemma 4 12B/31B unified
        // checkpoints where the dominant pattern is V-as-K). Callers in
        // the graph builder should consult `cfg.layer_k_eq_v(i)` to
        // pick up the SWA-layer V-independent case.
        attention_k_eq_v: matches!(arch, GemmaArch::Gemma4),
        use_bidirectional_attention: None,
        // Per-Layer Embeddings + KV sharing are Gemma 4 E2B (mobile)
        // features that only ship in the HF safetensors checkpoint, not
        // in any GGUF emitted by llama.cpp today. Default them off so the
        // GGUF path keeps the flagship dense behavior.
        hidden_size_per_layer_input: get_u32("gemma.embedding_length_per_layer_input").unwrap_or(0)
            as usize,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: get_u32("gemma.attention.shared_kv_layers").unwrap_or(0) as usize,
        use_double_wide_mlp: get_bool("gemma.use_double_wide_mlp").unwrap_or(false),
        enable_moe_block: get_u32("gemma.expert_count").unwrap_or(0) > 0,
        eog_token_ids: gemma_eog_tokens_from_gguf(raw, arch),
    })
}

/// llama.cpp EOG tokens for Gemma chat models (see load log `EOG token`).
fn gemma_eog_tokens_from_gguf(raw: &GgufFile, arch: GemmaArch) -> Vec<u32> {
    let mut ids = match arch {
        GemmaArch::Gemma2 | GemmaArch::Gemma3 | GemmaArch::Gemma4 => vec![1, 106, 212],
        _ => Vec::new(),
    };
    if let Some(eos) = raw
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(MetaValue::as_u32)
    {
        let eos = eos as u32;
        if !ids.contains(&eos) {
            ids.push(eos);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed copy of `google/gemma-4-12B`'s `config.json` — only the
    /// fields the loader actually consumes plus the surrounding shape
    /// (top-level `model_type`, nested `text_config`) that proves we
    /// unwrap the unified layout correctly.
    const GEMMA_4_12B_CONFIG: &str = r#"{
      "architectures": ["Gemma4UnifiedForConditionalGeneration"],
      "model_type": "gemma4_unified",
      "tie_word_embeddings": true,
      "text_config": {
        "model_type": "gemma4_unified_text",
        "vocab_size": 262144,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "attention_k_eq_v": true,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "final_logit_softcapping": 30.0,
        "sliding_window": 1024,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
        ],
        "rope_parameters": {
          "full_attention":    { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
          "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
        }
      }
    }"#;

    #[test]
    fn gemma_4_12b_unified_config_parses_text_subtree() {
        let dir = std::env::temp_dir();
        let path = dir.join("rlx_gemma_gemma4_12b_test_config.json");
        std::fs::write(&path, GEMMA_4_12B_CONFIG).unwrap();
        let cfg = GemmaConfig::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.arch, GemmaArch::Gemma4);
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.hidden_size, 3840);
        assert_eq!(cfg.intermediate_size, 15_360);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 256);
        assert_eq!(cfg.global_head_dim, Some(512));
        assert_eq!(cfg.num_global_key_value_heads, Some(1));
        assert!(cfg.attention_k_eq_v);
        assert_eq!(cfg.sliding_window, Some(1024));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types.len(), 48);
        // Stride-6 sliding-window pattern carried over from Gemma 3.
        assert_eq!(cfg.arch.sliding_window_stride(), 6);
    }

    /// Regression: Gemma 4 31B Q4_K_M GGUF (unsloth) encodes
    /// `gemma4.attention.head_count_kv` as an `Array[60 × i32]`
    /// (per-layer KV head count) and omits `gemma.vocab_size`
    /// entirely — vocab_size comes from `tokenizer.ggml.tokens`
    /// array length. `gemma_cfg_from_gguf` must:
    ///   1. Read scalar/array uniformly via `MetaValue::as_first_u32`.
    ///   2. Fall back to tokens-array length when the scalar is missing.
    ///   3. Set `attention_k_eq_v = true` automatically on Gemma 4.
    #[test]
    fn gemma4_gguf_per_layer_array_and_tokens_vocab() {
        use rlx_gguf::{GgufFile, GgufWriter, MetaValue};
        let mut w = GgufWriter::new();
        // Smallest field set the loader needs to succeed:
        w.set_meta("general.architecture", MetaValue::String("gemma4".into()));
        w.set_meta("gemma4.embedding_length", MetaValue::U32(5376));
        w.set_meta("gemma4.feed_forward_length", MetaValue::U32(21_504));
        w.set_meta("gemma4.block_count", MetaValue::U32(60));
        // head_count: U32 scalar (matches unsloth's layout = global KV heads = 4).
        w.set_meta("gemma4.attention.head_count", MetaValue::U32(4));
        // head_count_kv: Array of per-layer i32 — first element is
        // the sliding-layer KV count (16).
        let layer_kv: Vec<MetaValue> = (0..60)
            .map(|i| MetaValue::I32(if i == 5 { 4 } else { 16 }))
            .collect();
        w.set_meta("gemma4.attention.head_count_kv", MetaValue::Array(layer_kv));
        // Tokens array — implies vocab_size = 262_144 without the
        // scalar `gemma.vocab_size`.
        let tokens: Vec<MetaValue> = (0..262_144)
            .map(|_| MetaValue::String(String::new()))
            .collect();
        w.set_meta("tokenizer.ggml.tokens", MetaValue::Array(tokens));

        let path = std::env::temp_dir().join("rlx_gemma_gemma4_array_kv_test.gguf");
        w.write_to_path(&path).unwrap();
        let raw = GgufFile::from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let cfg = gemma_cfg_from_gguf(&raw).unwrap();
        assert_eq!(cfg.arch, GemmaArch::Gemma4);
        assert_eq!(cfg.vocab_size, 262_144, "vocab from tokens-array length");
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.intermediate_size, 21_504);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_attention_heads, 4);
        // First array element = 16 (sliding-layer KV heads).
        assert_eq!(
            cfg.num_key_value_heads, 16,
            "as_first_u32 should pick array[0], not panic on Array variant"
        );
        // Gemma 4 implies attention_k_eq_v.
        assert!(cfg.attention_k_eq_v, "Gemma 4 should default k_eq_v=true");
    }

    #[test]
    fn hf_null_moe_fields_default_to_zero() {
        let json = r#"{"num_experts": null, "top_k_experts": null}"#;
        let v = normalize_hf_null_usize_fields(serde_json::from_str(json).unwrap());
        let obj = v.as_object().unwrap();
        assert_eq!(obj["num_experts"], 0);
        assert_eq!(obj["top_k_experts"], 0);
    }

    #[test]
    fn infer_gemma4_partial_rotary_from_rope_freqs_pattern() {
        use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};
        let mut w = GgufWriter::new();
        w.set_meta("general.architecture", MetaValue::String("gemma4".into()));
        w.set_meta("gemma4.embedding_length", MetaValue::U32(3840));
        w.set_meta("gemma4.feed_forward_length", MetaValue::U32(15_360));
        w.set_meta("gemma4.block_count", MetaValue::U32(48));
        w.set_meta("gemma4.attention.head_count", MetaValue::U32(16));
        w.set_meta("gemma4.attention.head_count_kv", MetaValue::U32(8));
        w.set_meta("gemma4.context_length", MetaValue::U32(8192));
        w.set_meta("gemma4.rope.freq_base", MetaValue::F32(1_000_000.0));
        w.set_meta("gemma4.rope.freq_base_swa", MetaValue::F32(10_000.0));
        w.set_meta("gemma4.rope.dimension_count", MetaValue::U32(512));
        w.set_meta("gemma4.rope.dimension_count_swa", MetaValue::U32(256));
        w.set_meta("gemma4.attention.key_length", MetaValue::U32(512));
        w.set_meta("gemma4.attention.key_length_swa", MetaValue::U32(256));
        // Mimic llama.cpp Gemma 4 p-RoPE: 64 rotated pairs + 192 suppressed.
        let mut factors = vec![1.0f32; 64];
        factors.extend(std::iter::repeat_n(1e30f32, 192));
        let factor_bytes: Vec<u8> = factors.iter().flat_map(|f| f.to_le_bytes()).collect();
        w.add_tensor_bytes("rope_freqs.weight", vec![256], GgmlType::F32, factor_bytes)
            .unwrap();
        let path = std::env::temp_dir().join("rlx_gemma_gemma4_rope_freqs_test.gguf");
        w.write_to_path(&path).unwrap();
        let raw = GgufFile::from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let cfg = gemma_cfg_from_gguf(&raw).unwrap();
        assert_eq!(cfg.arch, GemmaArch::Gemma4);
        assert_eq!(cfg.layer_n_rot(5), 128, "full layer p-RoPE from rope_freqs");
        assert_eq!(cfg.layer_n_rot(0), 256, "swa layer full rotation");
        let full = cfg.rope_parameters.full_attention.as_ref().unwrap();
        assert_eq!(full.partial_rotary_factor, Some(0.25));
        assert_eq!(full.rope_type, Some(GemmaRopeKind::Proportional));
    }

    #[test]
    fn gemma_4_12b_per_layer_dispatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("rlx_gemma_gemma4_12b_dispatch_config.json");
        std::fs::write(&path, GEMMA_4_12B_CONFIG).unwrap();
        let cfg = GemmaConfig::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Sliding layer 0 — base shapes + full rotary on theta=1e4.
        assert!(!cfg.is_full_attention_layer(0));
        assert_eq!(cfg.layer_head_dim(0), 256);
        assert_eq!(cfg.layer_num_kv_heads(0), 8);
        assert_eq!(cfg.layer_n_rot(0), 256);
        assert!((cfg.layer_rope_theta(0) - 10_000.0).abs() < 1e-3);

        // Full-attention layer 5 (1-indexed: 6th layer) — global
        // shapes, p-RoPE (0.25 of head_dim_full=512 → 128), theta=1e6.
        assert!(cfg.is_full_attention_layer(5));
        assert_eq!(cfg.layer_head_dim(5), 512);
        assert_eq!(cfg.layer_num_kv_heads(5), 1);
        assert_eq!(cfg.layer_n_rot(5), 128);
        assert!((cfg.layer_rope_theta(5) - 1_000_000.0).abs() < 1e-3);

        // Last layer (index 47, 1-indexed 48) is also full-attention.
        assert!(cfg.is_full_attention_layer(47));
    }

    #[test]
    fn gemma3_gguf_rope_map_splits_swa_and_full_theta() {
        use rlx_gguf::{GgufFile, GgufWriter, MetaValue};
        let mut w = GgufWriter::new();
        w.set_meta("general.architecture", MetaValue::String("gemma3".into()));
        w.set_meta("gemma3.block_count", MetaValue::U32(18));
        w.set_meta("gemma3.embedding_length", MetaValue::U32(640));
        w.set_meta("gemma3.feed_forward_length", MetaValue::U32(2048));
        w.set_meta("gemma3.attention.head_count", MetaValue::U32(4));
        w.set_meta("gemma3.attention.head_count_kv", MetaValue::U32(1));
        w.set_meta("gemma3.attention.key_length", MetaValue::U32(256));
        w.set_meta("gemma3.attention.sliding_window", MetaValue::U32(512));
        w.set_meta("gemma3.rope.freq_base", MetaValue::F32(1_000_000.0));
        w.set_meta(
            "gemma3.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-6),
        );
        w.set_meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(vec![MetaValue::String("a".into())]),
        );
        let path = std::env::temp_dir().join("rlx_gemma3_rope_map_test.gguf");
        w.write_to_path(&path).unwrap();
        let raw = GgufFile::from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let cfg = gemma_cfg_from_gguf(&raw).unwrap();
        assert_eq!(cfg.arch, GemmaArch::Gemma3);
        assert!((cfg.layer_rope_theta(0) - 10_000.0).abs() < 1e-3);
        assert!((cfg.layer_rope_theta(5) - 1_000_000.0).abs() < 1e-3);
        assert_eq!(cfg.attn_score_scale(), Some(1.0 / 16.0));
    }

    #[test]
    fn gemma3_sliding_kv_trim_spec_marks_stride_layers_only() {
        use rlx_gguf::{GgufFile, GgufWriter, MetaValue};
        let mut w = GgufWriter::new();
        w.set_meta("general.architecture", MetaValue::String("gemma3".into()));
        w.set_meta("gemma3.block_count", MetaValue::U32(18));
        w.set_meta("gemma3.embedding_length", MetaValue::U32(640));
        w.set_meta("gemma3.feed_forward_length", MetaValue::U32(2048));
        w.set_meta("gemma3.attention.head_count", MetaValue::U32(4));
        w.set_meta("gemma3.attention.head_count_kv", MetaValue::U32(1));
        w.set_meta("gemma3.attention.key_length", MetaValue::U32(256));
        w.set_meta("gemma3.attention.sliding_window", MetaValue::U32(512));
        w.set_meta("gemma3.rope.freq_base", MetaValue::F32(1_000_000.0));
        w.set_meta(
            "gemma3.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-6),
        );
        w.set_meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(vec![MetaValue::String("a".into())]),
        );
        let path = std::env::temp_dir().join("rlx_gemma3_iswa_trim_test.gguf");
        w.write_to_path(&path).unwrap();
        let raw = GgufFile::from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let cfg = gemma_cfg_from_gguf(&raw).unwrap();
        let kv_dims: Vec<usize> = (0..cfg.num_hidden_layers)
            .map(|l| cfg.layer_num_kv_heads(l) * cfg.layer_head_dim(l))
            .collect();
        let spec = cfg.sliding_kv_trim_spec(&kv_dims);
        assert_eq!(spec.len(), 18);
        assert_eq!(spec[0], Some((256, 512)));
        assert_eq!(spec[5], None);
        assert_eq!(spec[17], None);
    }

    #[test]
    fn pre_gemma4_archs_keep_uniform_layer_shape() {
        // Without `layer_types` / `rope_parameters` the per-layer
        // accessors collapse to the base values so Gemma 3 / 2 / 1
        // continue to round-trip the existing flow.
        let mut cfg = GemmaConfig::tiny_test();
        cfg.arch = GemmaArch::Gemma3;
        cfg.head_dim = Some(64);
        cfg.num_key_value_heads = 2;
        cfg.rope_theta = 1_000.0;
        for i in 0..cfg.num_hidden_layers {
            assert_eq!(cfg.layer_head_dim(i), 64);
            assert_eq!(cfg.layer_num_kv_heads(i), 2);
            assert_eq!(cfg.layer_n_rot(i), 64);
            assert!((cfg.layer_rope_theta(i) - 1_000.0).abs() < 1e-3);
        }
    }

    #[test]
    fn infer_arch_picks_up_gemma4_markers() {
        assert_eq!(
            infer_arch_from_json(r#"{"model_type":"gemma4_unified"}"#),
            GemmaArch::Gemma4,
        );
        assert_eq!(
            infer_arch_from_json(r#"{"architectures":["Gemma4UnifiedForConditionalGeneration"]}"#),
            GemmaArch::Gemma4,
        );
        assert_eq!(
            infer_arch_from_json(r#"{"model_type":"gemma3"}"#),
            GemmaArch::Gemma3,
        );
    }

    /// Trimmed `google/gemma-4-E2B-it` text_config — exercises the
    /// Per-Layer-Embedding + KV-sharing + double-wide-MLP fields and the
    /// helpers that drive the builder.
    const GEMMA_4_E2B_CONFIG: &str = r#"{
      "model_type": "gemma4",
      "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_kv_shared_layers": 20,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": true,
        "enable_moe_block": false,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "sliding_window": 512,
        "tie_word_embeddings": false,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
        ],
        "rope_parameters": {
          "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        }
      }
    }"#;

    #[test]
    fn gemma_4_e2b_config_parses_ple_and_kv_sharing() {
        let cfg: GemmaConfig = {
            let value: serde_json::Value = serde_json::from_str(GEMMA_4_E2B_CONFIG).unwrap();
            let lm = value.get("text_config").cloned().unwrap();
            let mut c: GemmaConfig =
                serde_json::from_value(normalize_hf_null_usize_fields(lm)).unwrap();
            c.arch = infer_arch_from_json(GEMMA_4_E2B_CONFIG);
            c
        };
        assert_eq!(cfg.arch, GemmaArch::Gemma4);
        // PLE
        assert!(cfg.has_ple());
        assert_eq!(cfg.ple_width(), 256);
        assert_eq!(cfg.ple_vocab_size(), 262144);
        // KV sharing: 35 - 20 = 15
        assert_eq!(cfg.first_kv_shared_layer(), 15);
        for i in 0..15 {
            assert!(!cfg.is_kv_shared_layer(i), "layer {i} should be fresh");
            assert_eq!(cfg.kv_source_layer(i), i);
        }
        for i in 15..35 {
            assert!(cfg.is_kv_shared_layer(i), "layer {i} should be shared");
        }
        // Full-attention layers sit at 4,9,14,...; last fresh full = 14,
        // last fresh sliding = 13.
        assert!(cfg.is_full_attention_layer(19));
        assert_eq!(
            cfg.kv_source_layer(19),
            14,
            "shared full reuses last fresh full"
        );
        assert_eq!(cfg.kv_source_layer(34), 14);
        assert!(!cfg.is_full_attention_layer(15));
        assert_eq!(
            cfg.kv_source_layer(15),
            13,
            "shared sliding reuses last fresh sliding"
        );
        assert_eq!(cfg.kv_source_layer(20), 13);
        // Double-wide MLP only on shared layers.
        assert_eq!(cfg.layer_intermediate_size(0), 6144);
        assert_eq!(cfg.layer_intermediate_size(14), 6144);
        assert_eq!(cfg.layer_intermediate_size(15), 12288);
        assert_eq!(cfg.layer_intermediate_size(34), 12288);
    }

    #[test]
    fn non_e2b_config_has_no_ple_or_sharing() {
        let cfg = GemmaConfig::tiny_test();
        assert!(!cfg.has_ple());
        assert_eq!(cfg.first_kv_shared_layer(), cfg.num_hidden_layers);
        assert!(!cfg.is_kv_shared_layer(0));
        assert_eq!(cfg.kv_source_layer(0), 0);
        assert_eq!(cfg.layer_intermediate_size(0), cfg.intermediate_size);
    }
}
