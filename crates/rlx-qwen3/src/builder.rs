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
//! (native [`ModelFlow`]). This module retains packed-weight builders.
//!
//! Qwen3 specifics handled here:
//!   - **GQA** via graph-level KV head repetition (narrow + concat).
//!     The existing `Op::Attention` only carries `num_heads`, so K/V
//!     are widened from `num_kv_heads * head_dim` to
//!     `num_heads * head_dim` before the attention call. A real GQA
//!     op replaces this in Phase 2 alongside the KV-cache kernels.
//!   - **QK-norm**: per-head RMSNorm on Q and K (before RoPE), via a
//!     reshape to `[B*S*heads, head_dim]` so the existing RMSNorm
//!     kernel sees its canonical 2D layout.
//!   - **Causal attention** via `MaskKind::Causal` — no mask tensor.
//!   - **Sliding window** is parsed from config but Phase 1 treats
//!     every layer as full causal; the per-layer SWA wiring lands
//!     when backend kernels confirm `MaskKind::SlidingWindow` support.

use crate::config::Qwen3Config;
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
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

/// Per-head RMSNorm via reshape to `[B*S*heads, head_dim]`, norm with
/// `gamma=[head_dim]`, then reshape back to `[B, S, heads*head_dim]`.
/// Keeps the kernel on its canonical 2D contract.
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

/// GQA repeat: widen `[B, S, num_kv_heads * head_dim]` to
/// `[B, S, num_kv_heads * group * head_dim]` by emitting each KV head
/// `group` times in sequence. Uses narrow + concat; cheap in op count
/// (one narrow per KV head, one concat per layer per K/V) and avoids
/// touching `Op::Attention`. Replaced by a true GQA op in Phase 2.
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

/// Load a weight by key and register it as a `Param` node. When
/// `transpose` is set, the safetensors `[out, in]` layout is swapped
/// to rlx's `[in, out]` matmul convention.
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

/// Register a zero-valued param of the given length. Used to supply
/// the `beta` argument to RMSNorm calls (Qwen3 has no bias term but
/// the IR op signature requires one).
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
// Trade-off: 7-9× less load memory, 2-4× slower per matmul on CPU
// (each call re-dequants the weight to a scratch buffer before
// sgemm). A future tile-streaming kernel would close the compute
// gap; today it's "loadable but slow."
// ────────────────────────────────────────────────────────────────

/// Companion to [`build_qwen3_graph_sized`] that keeps K-quant
/// weights packed in the arena. Pass an empty `packed` HashMap;
/// the function fills it with the GGUF-packed bytes for every
/// per-layer matmul weight + the LM-head weight. Non-K-quant
/// tensors (F32 norms, F16/BF16, Q4_0/Q5_0/Q8_0 legacy formats)
/// stay in `params` as F32.
///
/// Used together with [`Qwen3Generator::from_path_packed`] and
/// [`Qwen3RunnerBuilder::packed_weights`] for the low-memory
/// inference path.
#[allow(clippy::too_many_arguments)]
fn gather_last_token(
    g: &mut Graph,
    hidden: NodeId,
    batch: usize,
    last_token_idx: NodeId,
) -> NodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(hidden, idx_2d, 1)
}

pub fn build_qwen3_graph_sized_packed(
    cfg: &Qwen3Config,
    weights: &mut rlx_core::weight_loader::GgufLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use rlx_ir::quant::QuantScheme;

    if !cfg
        .num_attention_heads
        .is_multiple_of(cfg.num_key_value_heads)
    {
        return Err(anyhow!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    let mut g = Graph::new("qwen3_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "qwen3.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "qwen3.zero_beta.head_dim", dh);

    // RoPE tables sized to compile `seq`, not `max_position_embeddings`.
    let half = dh / 2;
    let rope_len = seq;
    let mut cos_data = vec![0f32; rope_len * half];
    let mut sin_data = vec![0f32; rope_len * half];
    for pos in 0..rope_len {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param("rope.cos", Shape::new(&[rope_len, half], f));
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param("rope.sin", Shape::new(&[rope_len, half], f));
    params.insert("rope.sin".into(), sin_data);

    // Token IDs and position indices MUST be I32 — feeding them as F32
    // bit-pattern-reinterprets the floats as ints inside the gather kernel,
    // producing stable garbled token streams (e.g. "< as as as…").
    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::I32));
    let last_token_idx = if with_lm_head && last_token_from_input {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::I32)))
    } else {
        None
    };

    // Embedding stays F32 — gather op needs dequant'd table.
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    // Helper closure: load a matmul weight either packed (K-quant)
    // or F32 (anything else). Returns (NodeId, Option<scheme>) —
    // `Some(scheme)` means caller emits `Op::DequantMatMul`,
    // `None` means caller emits regular `Op::MatMul`.
    //
    // The shape semantics: for the packed case, the param is a
    // U8 byte tensor; the *output* shape of the matmul is `[..., n]`
    // where n is what GGUF reports as the innermost dim (=
    // out_features in HF). Caller passes that n.
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        weights: &mut rlx_core::weight_loader::GgufLoader,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>, Vec<usize>)> {
        // Probe packed first; if not a K-quant, fall back to F32.
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape.clone()));
            // GGUF shape (after the safetensors-style reverse) is
            // `[out, in]` — same as HF safetensors `[out_features,
            // in_features]`. The matmul output dim is `out` = shape[0].
            Ok((id, Some(scheme), shape))
        } else {
            let nid = load_p(g, params, weights, key, /*transpose*/ true)?;
            // load_p with transpose=true gives back [in, out].
            let shape = params
                .get(key)
                .map(|_| Vec::<usize>::new()) // dummy; unused for the F32 path
                .unwrap_or_default();
            Ok((nid, None, shape))
        }
    }

    // Emit either DequantMatMul or MatMul depending on scheme.
    fn emit_proj(
        g: &mut Graph,
        input: NodeId,
        w: NodeId,
        scheme: Option<QuantScheme>,
        out_shape: Shape,
    ) -> NodeId {
        match scheme {
            Some(s) => g.add_node(Op::DequantMatMul { scheme: s }, vec![input, w], out_shape),
            None => g.mm(input, w),
        }
    }

    let kv_outputs: Vec<NodeId> = Vec::new();
    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        let q_dim = nh * dh;
        let kv_dim = nkv * dh;
        let (q_w, q_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let (k_w, k_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let (v_w, v_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let mut q = emit_proj(
            &mut g,
            normed_in,
            q_w,
            q_s,
            Shape::new(&[batch, seq, q_dim], f),
        );
        let mut k = emit_proj(
            &mut g,
            normed_in,
            k_w,
            k_s,
            Shape::new(&[batch, seq, kv_dim], f),
        );
        let mut v = emit_proj(
            &mut g,
            normed_in,
            v_w,
            v_s,
            Shape::new(&[batch, seq, kv_dim], f),
        );

        // Qwen 2 / 2.5 ship explicit bias vectors on Q/K/V (1-D
        // `[out_features]`); Qwen 3 does not. Cfg flag selects.
        if cfg.attention_bias {
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

        // Qwen 3 applies per-head RMS-norm on Q/K *before* RoPE
        // ("QK-norm"). Qwen 2 skips this entirely.
        let (q_normed, k_normed) = if cfg.qk_norm {
            let q_norm_g = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.q_norm.weight"),
                false,
            )?;
            let k_norm_g = load_p(
                &mut g,
                &mut params,
                weights,
                &format!("{lp}.self_attn.k_norm.weight"),
                false,
            )?;
            let qn = per_head_rms(
                &mut g,
                q,
                q_norm_g,
                zero_beta_headdim,
                batch,
                seq,
                nh,
                dh,
                eps,
            );
            let kn = per_head_rms(
                &mut g,
                k,
                k_norm_g,
                zero_beta_headdim,
                batch,
                seq,
                nkv,
                dh,
                eps,
            );
            (qn, kn)
        } else {
            (q, k)
        };

        // MLX `Op::Rope` rank-3 multi-head packing expects [B, H, S, D]. Run rope on BHSD, then
        // restore the external layout [B, S, H*D] used by attention kernels.
        let q4 = g.reshape_(
            q_normed,
            vec![batch as i64, seq as i64, nh as i64, dh as i64],
        );
        let q_bhsd = g.transpose_(q4, vec![0, 2, 1, 3]);
        let q_rope_bhsd = g.rope(q_bhsd, cos_id, sin_id, dh);
        let q_rope_bshd = g.transpose_(q_rope_bhsd, vec![0, 2, 1, 3]);
        let q_rope = g.reshape_(
            q_rope_bshd,
            vec![batch as i64, seq as i64, (nh * dh) as i64],
        );

        let k4 = g.reshape_(
            k_normed,
            vec![batch as i64, seq as i64, nkv as i64, dh as i64],
        );
        let k_bhsd = g.transpose_(k4, vec![0, 2, 1, 3]);
        let k_rope_bhsd = g.rope(k_bhsd, cos_id, sin_id, dh);
        let k_rope_bshd = g.transpose_(k_rope_bhsd, vec![0, 2, 1, 3]);
        let k_rope = g.reshape_(
            k_rope_bshd,
            vec![batch as i64, seq as i64, (nkv * dh) as i64],
        );

        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.attention_kind(
            q_rope,
            k_rep,
            v_rep,
            nh,
            dh,
            attn_mask_kind(cfg),
            attn_shape,
        );

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        let (gate_w, gate_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
        )?;
        let (up_w, up_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
        )?;
        let (down_w, down_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let inter = cfg.intermediate_size;
        let gate = emit_proj(
            &mut g,
            normed_post,
            gate_w,
            gate_s,
            Shape::new(&[batch, seq, inter], f),
        );
        let up = emit_proj(
            &mut g,
            normed_post,
            up_w,
            up_s,
            Shape::new(&[batch, seq, inter], f),
        );
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = emit_proj(
            &mut g,
            swiglu,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );
        h_id = g.add(post_attn, ffn_out);
        let _ = kv_outputs.len(); // silence unused for now
    }

    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

    let out = if with_lm_head {
        let head_input = if let Some(idx) = last_token_idx {
            gather_last_token(&mut g, hidden, batch, idx)
        } else {
            hidden
        };
        let logit_rows = if last_token_from_input { 1 } else { seq };
        // Tied: try to reuse the embed weight in packed form. If it's
        // a K-quant in the GGUF, register a SECOND packed param under
        // a distinct name (lm_head reads it; gather reads the F32
        // dequant'd version already in `params`).
        let (lm_head_w, lm_head_scheme) = if cfg.tie_word_embeddings {
            // Re-open the file to grab the packed bytes for embed.
            // The loader has already taken the F32 dequant copy for
            // gather; take_packed by-name would fail (already taken).
            // Cheapest fix: build the f32→packed copy NOT — that's
            // hard. Instead: fall back to F32 tied path (which is
            // what we did before — pre-transpose once at build time).
            let embed = params
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
            let vocab = cfg.vocab_size;
            let hidden_size = cfg.hidden_size;
            let mut transposed = vec![0f32; embed.len()];
            for v in 0..vocab {
                for hi in 0..hidden_size {
                    transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                }
            }
            let name = "qwen3.lm_head.tied_t";
            let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
            params.insert(name.to_string(), transposed);
            (id, None)
        } else {
            let (id, scheme, _) =
                load_proj(&mut g, &mut params, packed, weights, "lm_head.weight")?;
            (id, scheme)
        };
        emit_proj(
            &mut g,
            head_input,
            lm_head_w,
            lm_head_scheme,
            Shape::new(&[batch, logit_rows, cfg.vocab_size], f),
        )
    } else {
        hidden
    };

    g.set_outputs(vec![out]);
    Ok((g, params))
}
