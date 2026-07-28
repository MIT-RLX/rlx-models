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

//! Qwen3 graph builder — packed weights and legacy probes.
//!
//! Prefill and decode graphs are assembled via [`crate::flow::Qwen3Flow`]
//! (native [`ModelFlow`]). This module retains the packed-weight prefill
//! entry point, which now delegates to the family-generic
//! [`rlx_core::build_standard_decoder_packed`] (Qwen3 is one member of the
//! standard causal-decoder family it covers).

use crate::config::Qwen3Config;
use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::{DecoderSpec, RopeScaling};
use rlx_ir::op::MaskKind;
use rlx_ir::*;
use std::collections::HashMap;

/// Build a Qwen3 causal-LM IR graph.
///
/// When `with_lm_head` is `false`, the output is the post-norm hidden
/// state `[batch, seq, hidden_size]` (useful for embedding-style
/// pooling or for inserting a custom head). When `true`, the output is
/// logits `[batch, seq, vocab_size]`.
///
/// When `with_kv_outputs` is `true`, each layer's post-RoPE K and post-projection
/// V tensors (both shape `[batch, seq, kv_proj_dim]`, pre-GQA-repeat) are appended
/// to the graph outputs in order `[main, k_0, v_0, k_1, v_1, ..., k_{N-1}, v_{N-1}]`.
/// Used to seed the KV cache for decode mode.
///
/// Tied embeddings (`cfg.tie_word_embeddings = true`) are handled by
/// reusing the `model.embed_tokens.weight` parameter node via a
/// graph-level transpose — no data duplication, one extra `Transpose`
/// op per model.
pub fn build_qwen3_graph_sized(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::Qwen3PrefillOpts {
        batch,
        seq,
        with_lm_head,
        with_kv_outputs,
        with_qk_outputs: false,
        last_logits_only: false,
        profile: None,
        rope_cos: None,
        rope_sin: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_qwen3_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_qwen3_graph_sized_last_logits(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::Qwen3PrefillOpts {
        batch,
        seq,
        with_lm_head: true,
        with_kv_outputs,
        with_qk_outputs: false,
        last_logits_only: true,
        profile: None,
        rope_cos: None,
        rope_sin: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_qwen3_prefill_built(
        cfg, weights, &opts,
    )?)
}

/// Attention mask from config: a sliding window when the model enables one
/// (Gemma / Mistral-style), else full causal. `Op::Attention` lowers
/// `SlidingWindow` on CPU/Metal/MLX; a `window >= seq` is equivalent to
/// causal. (CUDA/ROCm/wgpu decompose attention and currently honor only the
/// causal mask — sliding window there is future work.)
pub(crate) fn attn_mask_kind(cfg: &Qwen3Config) -> MaskKind {
    match cfg.sliding_window {
        Some(w) if cfg.use_sliding_window && w > 0 => MaskKind::SlidingWindow(w),
        _ => MaskKind::Causal,
    }
}

/// Build a Qwen3 decode-mode IR graph for a single new token given
/// a cached past of `past_seq` tokens. Inputs are:
///
///   - `input_ids` shape `[batch, 1]`
///   - `rope_cos` shape `[1, head_dim/2]` — host pre-narrows the full
///     cos table at the new token's absolute position (= `past_seq`).
///   - `rope_sin` shape `[1, head_dim/2]` — likewise.
///   - For each layer `i` in `0..num_hidden_layers`:
///     - `past_k_{i}` shape `[batch, past_seq, kv_proj_dim]`
///     - `past_v_{i}` shape `[batch, past_seq, kv_proj_dim]`
///
/// Outputs in order:
///
///   - `logits` shape `[batch, 1, vocab_size]`
///   - For each layer `i`: `new_k_{i}`, `new_v_{i}` — both
///     `[batch, past_seq + 1, kv_proj_dim]` — the cache to feed back
///     in on the next decode step.
///
/// The IR's `Op::Attention` with `MaskKind::Causal` correctly handles
/// `Lq=1` vs `Lk=past_seq+1` after the kernel fix in
/// `rlx-cpu/src/executor.rs` (Q's absolute position = `past_seq`, so
/// all `Lk` positions ≤ past_seq are attended; the upper-triangular
/// fill becomes a no-op).
pub fn build_qwen3_decode_graph_sized(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_qwen3_decode_graph_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_qwen3_decode_hir_sized(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_qwen3_decode_hir_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_qwen3_decode_hir_sized_ext(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    use crate::flow::{Qwen3DecodeOpts, build_qwen3_decode_flow};

    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        ragged_rope: false,
        export_qk: false,
        profile: None,
    };
    build_qwen3_decode_flow(cfg, weights, &opts)
}

/// Extended decode-mode builder.
///
/// `use_custom_mask`:
///   - `false` (default): use `MaskKind::Causal`, no mask input. Behavior
///     identical to [`build_qwen3_decode_graph_sized`]. Graph shape is
///     specialized to the exact `past_seq`.
///   - `true`: take a `mask` input of shape `[batch, past_seq + 1]` and
///     apply it via `MaskKind::Custom`. Lets a bucketed compile cache
///     pad `past_k`/`past_v` up to the bucket's upper bound and mask
///     the padded positions so they don't contribute to attention. The
///     graph is then reusable for any actual past length ≤ `past_seq`.
pub fn build_qwen3_decode_graph_sized_ext(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use crate::flow::{Qwen3DecodeOpts, build_qwen3_decode_graph};

    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        ragged_rope: false,
        export_qk: false,
        profile: None,
    };
    build_qwen3_decode_graph(cfg, weights, &opts)
}

/// Decode builder for **ragged** batched decode: a per-sequence RoPE table
/// (`rope_cos`/`rope_sin` shaped `[batch, head_dim/2]`) plus the custom mask, so
/// each sequence in the batch may sit at a different absolute position / cache
/// length. The graph is otherwise identical to
/// [`build_qwen3_decode_graph_sized_ext`] with `use_custom_mask = true`.
pub fn build_qwen3_decode_graph_sized_ragged(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use crate::flow::{Qwen3DecodeOpts, build_qwen3_decode_graph};

    let opts = Qwen3DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask: true,
        ragged_rope: true,
        export_qk: false,
        profile: None,
    };
    build_qwen3_decode_graph(cfg, weights, &opts)
}

// ────────────────────────────────────────────────────────────────
// Packed-weights mode — Op::DequantMatMul for the big projections.
//
// The default builder dequants every K-quant tensor to F32 at load
// (~7-9× memory expansion). For models that won't fit in unified
// memory after that — Qwen3-14B+, Qwen3.6-27B-MTP, etc. — this
// alternate builder keeps the per-layer + LM-head matmul weights
// as packed bytes in the arena and emits `Op::DequantMatMul` so
// the kernel dequants per matmul invocation.
//
// The block topology (RMSNorm → q/k/v (+bias) → opt QK-norm → RoPE →
// GQA → attention → o_proj → residual → RMSNorm → SwiGLU) is shared
// with the whole Llama / Qwen2 / Mistral / SmolLM family, so the body
// lives in `rlx_core::build_standard_decoder_packed`; here we only map
// `Qwen3Config` → `DecoderSpec` and forward.
// ────────────────────────────────────────────────────────────────

/// Normalized [`DecoderSpec`] for a [`Qwen3Config`]. Qwen3 sets
/// `rope_scaling: null`, so this is [`RopeScaling::None`] and the RoPE
/// table matches the historical qwen3 packed builder bit-for-bit.
pub fn qwen3_decoder_spec(cfg: &Qwen3Config) -> DecoderSpec {
    DecoderSpec {
        arch: "qwen3".to_string(),
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rms_norm_eps: cfg.rms_norm_eps as f32,
        rope_theta: cfg.rope_theta,
        rope_scaling: RopeScaling::None,
        hidden_act: cfg.hidden_act.clone(),
        attention_bias: cfg.attention_bias,
        fused_qkv: false,
        fused_gate_up: false,
        qk_norm: cfg.qk_norm,
        qk_norm_full: false,
        tie_word_embeddings: cfg.tie_word_embeddings,
        sliding_window: cfg.sliding_window,
        use_sliding_window: cfg.use_sliding_window,
        max_window_layers: cfg.max_window_layers,
        num_experts: cfg.num_experts,
        num_experts_used: 0,
        norm_topk_prob: false,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        // Qwen3 is not a Gemma-family model: all Gemma axes are no-ops.
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
        // Qwen3 uses full RoPE over the entire head_dim.
        partial_rotary_factor: 1.0,
    }
}

/// Companion to [`build_qwen3_graph_sized`] that keeps quantized weights
/// packed in the arena (MLX affine → 4-input `Op::DequantMatMul`, GGUF
/// K-quant → 2-input). Pass an empty `packed` map; it is filled with the
/// U8 codes for every per-layer + LM-head matmul weight. Cuts load-time
/// memory ~7-9× — the loadable-but-slow path for large checkpoints.
///
/// Thin wrapper over [`rlx_core::build_standard_decoder_packed`] via
/// [`qwen3_decoder_spec`]; see that function for the block topology.
pub fn build_qwen3_graph_sized_packed(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let spec = qwen3_decoder_spec(cfg);
    rlx_core::build_standard_decoder_packed(
        &spec,
        weights,
        batch,
        seq,
        with_lm_head,
        last_token_from_input,
        /*embeds_input*/ false,
        packed,
    )
}
