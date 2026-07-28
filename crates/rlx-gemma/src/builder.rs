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

//! Gemma graph builders — thin wrappers over [`crate::flow::GemmaFlow`].

use crate::config::GemmaConfig;
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::Graph;
use rlx_ir::hir::HirModule;
use rlx_ir::infer::GraphExt;
use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;

type F32WeightMap = HashMap<String, Vec<f32>>;

/// Where a packed weight's bytes come from at attach time. Storing the *source*
/// instead of the bytes keeps the quantized model out of resident RSS: 1:1
/// weights and fused projections are re-materialized transiently by borrowing
/// straight from the GGUF mmap at each attach (see `upload_packed_borrowed`).
#[derive(Clone, Debug)]
pub enum PackedSrc {
    /// Kept resident — used only when a weight genuinely can't be borrowed.
    Owned(Vec<u8>),
    /// Borrow these GGUF tensors from the loader mmap and concat in order at
    /// attach. One key = a 1:1 weight; many keys = a fused proj (gate‖up,
    /// q‖k‖v). `nbytes` is the total U8 length (sum of the parts).
    Borrow { keys: Vec<String>, nbytes: usize },
    /// Materialized to F32 at drain (norms / unsupported quant); the graph param
    /// is F32 and rebuilt from the f32 cache, never uploaded as packed bytes.
    F32,
}

impl PackedSrc {
    /// Byte length of the packed U8 param node (0 for the F32 sentinel).
    pub fn nbytes(&self) -> usize {
        match self {
            PackedSrc::Owned(b) => b.len(),
            PackedSrc::Borrow { nbytes, .. } => *nbytes,
            PackedSrc::F32 => 0,
        }
    }
    /// True for the F32-materialized sentinel.
    pub fn is_f32(&self) -> bool {
        matches!(self, PackedSrc::F32)
    }
    /// Resident bytes for an [`PackedSrc::Owned`] entry (the embed table);
    /// `None` for borrow recipes / F32 sentinels.
    pub fn owned_bytes(&self) -> Option<&[u8]> {
        match self {
            PackedSrc::Owned(b) => Some(b),
            _ => None,
        }
    }
}

type PackedWeightMap = HashMap<String, (PackedSrc, QuantScheme, Vec<usize>)>;
type PackedDrainResult = (F32WeightMap, PackedWeightMap);

/// Decode-graph lm_head output mode for packed GGUF graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PackedDecodeLmOutput {
    /// Final hidden state only (host tied-lm argmax or sampling elsewhere).
    #[default]
    HiddenOnly,
    /// Full vocab logits D2H (~1 MiB/step for Gemma 3).
    FullLogits,
    /// In-graph tied lm_head + argmax; read back one f32 token id.
    GreedyToken,
}

pub fn build_gemma_graph_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head,
        with_kv_outputs,
        last_logits_only: false,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_graph_sized_last_logits(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_prefill_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_prefill_hir_dynamic_ext_inner(cfg, weights, batch, max_seq, with_kv_outputs, false)
}

/// Dynamic-seq prefill from fused `inputs_embeds` (multimodal).
pub fn build_gemma_prefill_hidden_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_prefill_hir_dynamic_ext_inner(cfg, weights, batch, max_seq, with_kv_outputs, true)
}

fn build_gemma_prefill_hir_dynamic_ext_inner(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_seq: usize,
    with_kv_outputs: bool,
    prefill_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    if batch != 1 {
        return Err(anyhow!("gemma: dynamic_seq prefill requires batch=1"));
    }
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq: max_seq,
        dynamic_seq: true,
        prefill_hidden,
        media_attn_bias: prefill_hidden && cfg.use_bidirectional_vision(),
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    crate::flow::build_gemma_prefill_flow(cfg, weights, &opts)
}

pub fn build_gemma_graph_sized_last_logits_hidden(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaPrefillOpts {
        batch,
        seq,
        dynamic_seq: false,
        prefill_hidden: true,
        media_attn_bias: cfg.use_bidirectional_vision(),
        with_lm_head: true,
        with_kv_outputs,
        last_logits_only: true,
        profile: None,
    };
    rlx_core::flow_util::graph_from_built(crate::flow::build_gemma_prefill_built(
        cfg, weights, &opts,
    )?)
}

pub fn build_gemma_decode_graph_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_gemma_decode_graph_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_gemma_decode_graph_sized_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_graph(cfg, weights, &opts)
}

pub fn build_gemma_decode_hir_sized(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_gemma_decode_hir_sized_ext(cfg, weights, batch, past_seq, false)
}

pub fn build_gemma_decode_hir_sized_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_flow(cfg, weights, &opts)
}

pub fn build_gemma_decode_hir_dynamic_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    max_past_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let opts = crate::flow::GemmaDecodeOpts {
        batch,
        past_seq: max_past_seq,
        dynamic_past: true,
        use_custom_mask: false,
        profile: None,
        aux_hidden_layer_ids: Vec::new(),
    };
    crate::flow::build_gemma_decode_flow(cfg, weights, &opts)
}

/// Packed K-quant prefill graph. Mirrors `build_gemma_graph_sized`
/// but emits `Op::DequantMatMul` against the GGUF-quantized weight
/// buffers (kept in their on-disk layout in `packed`) for every
/// projection that has a quantization scheme. Tensors that come back
/// from the loader as F32 (norms, embed) still go through `MatMul`.
/// Memory cost: O(quantized weight bytes) per layer in the arena
/// instead of O(F32 bytes).
///
/// Supports the full Gemma 1/2/3/4 surface area exposed by
/// `GemmaConfig`: per-layer head_dim / num_kv_heads / n_rot,
/// `attention_k_eq_v`, split RoPE per layer kind, alternating
/// sliding-window vs full-attention masks, attention soft-cap, and
/// final logit soft-cap. Layer-norm names follow the version-aware
/// pattern (`post_attention_layernorm` for V1, pre+post feedforward
/// for V2+).
///
/// `last_token_from_input`: when true, adds a `last_token_idx` graph input
/// and gathers one hidden row before `lm_head` (1× vocab logits). Required
/// for autoregressive packed generation — do not use fixed `seq-1` narrow.
fn gather_last_token_packed(
    g: &mut Graph,
    hidden: rlx_ir::NodeId,
    batch: usize,
    last_token_idx: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(hidden, idx_2d, 1)
}

fn slice_rope_table(table: &[f32], half: usize, rows: usize) -> Vec<f32> {
    let need = rows * half;
    if table.len() >= need {
        table[..need].to_vec()
    } else {
        table.to_vec()
    }
}

/// GGUF stores 2D projection weights as `[in_features, out_features]`; RLX
/// matmul expects `[out_features, in_features]`.
fn should_transpose_gemma_drain_weight(canonical: &str) -> bool {
    canonical.ends_with(".weight")
        && !canonical.contains("norm")
        && !canonical.starts_with("rope.")
        && (canonical.contains(".self_attn.")
            || canonical.contains(".mlp.")
            || canonical.ends_with("lm_head.weight"))
}

/// Drain GGUF weights + RoPE tables for packed session init (no layer graph).
pub fn drain_gemma_packed_weights(
    cfg: &GemmaConfig,
    loader: &mut rlx_core::weight_loader::GgufLoader,
) -> Result<PackedDrainResult> {
    drain_gemma_packed_weights_ext(cfg, loader, None)
}

/// Same as [`drain_gemma_packed_weights`] but with an optional cap on the
/// number of RoPE-table rows materialised at LOAD. Default (`None`) preserves
/// the legacy behaviour (`cfg.max_position_embeddings` rows, up to ~1 GB for
/// Gemma 4 12B); the session caller passes `Some(max_seq + 16)` so we only
/// allocate the rows the prefill bucket can actually read (a ≥ 99% saving for
/// typical `max_seq=128`). Decode uses single-row `rope_slice` so the table
/// size doesn't gate decode either.
pub fn drain_gemma_packed_weights_ext(
    cfg: &GemmaConfig,
    loader: &mut rlx_core::weight_loader::GgufLoader,
    max_rope_rows: Option<usize>,
) -> Result<PackedDrainResult> {
    use crate::rope::{build_rope_tables, resolve_global_inv_freq, resolve_inv_freq};

    let rope_rows = max_rope_rows.unwrap_or(cfg.max_position_embeddings);
    let rope_factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
    let inv_freq = resolve_inv_freq(cfg, rope_factors.as_deref());
    let (cos_data, sin_data) = build_rope_tables(&inv_freq, rope_rows);

    let arch = loader.arch_hint().unwrap_or("gemma").to_string();

    let mut f32_params: HashMap<String, Vec<f32>> = HashMap::new();
    // Force-dequant the embed table to F32 so the input-embedding gather can
    // continue using a host-side f32 path. We do NOT pre-transpose it for the
    // tied LM head — that step is replaced by DequantMatMul on the original
    // Q4K-packed `token_embd.weight` bytes (see graph builder below). On
    // Gemma 4 31B Q4_K_M that saves the ~5.6 GB f32 transpose plus the same
    // amount cloned into each prefill/decode graph constant.
    //
    // Task #36: prefer `take_packed` so we ALSO retain the original Q4K bytes
    // for the LM head's `DequantMatMul`. Falls back to `take()` for f16/f32
    // embed tables (small models) or if the scheme isn't dequant-supported.
    let mut embed_packed: Option<(Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> = None;
    // RLX_GEMMA_LAZY_EMBED_DISABLE=1 forces the legacy `take()`+f32-gather path
    // even when packed bytes are available, for parity bisection.
    let lazy_disabled = std::env::var("RLX_GEMMA_LAZY_EMBED_DISABLE")
        .ok()
        .as_deref()
        == Some("1");
    if lazy_disabled {
        if let Ok((data, _shape)) = loader.take("model.embed_tokens.weight") {
            f32_params.insert("model.embed_tokens.weight".into(), data);
        }
    } else {
        match loader.take_packed("model.embed_tokens.weight") {
            Ok(Some((bytes, scheme, shape))) => {
                // Task #37: when packed Q4K bytes are available the runtime
                // host-gathers embedding rows on demand (see
                // `packed_session.rs::gather_embed_rows`), so we DO NOT dequant
                // the full vocab×hidden f32 table at LOAD. For 12B Q4_K_M that's
                // ~3.8 GB never allocated.
                embed_packed = Some((bytes, scheme, shape));
            }
            _ => {
                if let Ok((data, _shape)) = loader.take("model.embed_tokens.weight") {
                    f32_params.insert("model.embed_tokens.weight".into(), data);
                }
            }
        }
    }

    let keys = loader.remaining_keys();
    let mut f32_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    // (canonical name, gguf borrow key, scheme, shape, packed byte length).
    // Zero-copy: record only metadata + the key to borrow at attach — the
    // multi-GB packed bytes are never materialized into the resident map.
    let mut packed_list: Vec<(
        String,
        String,
        rlx_ir::quant::QuantScheme,
        Vec<usize>,
        usize,
    )> = Vec::new();
    for key in keys {
        let canonical = rlx_core::weight_loader::gguf_to_hf_name_for_arch(&key, &arch)
            .unwrap_or_else(|| key.clone());
        // RMSNorm / QK-norm weights stay F32 in the graph (delta-gamma path);
        // mixed-quant GGUF may still expose them as Q5_0 etc.
        let force_f32 = canonical.contains("layernorm") || canonical.contains("_norm.weight");
        if !force_f32 {
            if let Some((scheme, shape)) = loader.packed_meta(&key) {
                let nbytes = loader
                    .tensor_bytes_borrowed(&key)
                    .ok_or_else(|| anyhow::anyhow!("packed {key}: bytes unavailable"))?
                    .len();
                packed_list.push((canonical, key, scheme, shape, nbytes));
                continue;
            }
        }
        let (data, shape) = if should_transpose_gemma_drain_weight(&canonical) {
            loader.take_transposed(&key)?
        } else {
            loader.take(&key)?
        };
        f32_params.insert(canonical.clone(), data);
        f32_shapes.insert(canonical, shape);
    }
    f32_params.insert("rope.cos".into(), cos_data);
    f32_params.insert("rope.sin".into(), sin_data);
    if let Some(global_inv) = resolve_global_inv_freq(cfg, rope_factors.as_deref()) {
        let (gcd, gsd) = build_rope_tables(&global_inv, rope_rows);
        f32_params.insert("rope.global.cos".into(), gcd);
        f32_params.insert("rope.global.sin".into(), gsd);
    }

    let mut packed = HashMap::new();
    for (canonical, gguf_key, scheme, shape, nbytes) in packed_list {
        packed.insert(
            canonical,
            (
                PackedSrc::Borrow {
                    keys: vec![gguf_key],
                    nbytes,
                },
                scheme,
                shape,
            ),
        );
    }
    // Task #36: the embed table stays materialized (one bounded tensor) so the
    // host-side lazy gather + tied LM-head DequantMatMul keep reading owned
    // bytes; every per-layer weight below is borrowed from the mmap instead.
    if let Some((bytes, scheme, shape)) = embed_packed {
        packed.insert(
            "model.embed_tokens.weight".into(),
            (PackedSrc::Owned(bytes), scheme, shape),
        );
    }
    // Weights materialized to F32 at drain (norms, unsupported quants) get F32
    // sentinels so cached graph rebuilds resolve shape + f32 bytes.
    for (canonical, shape) in f32_shapes {
        packed
            .entry(canonical)
            .or_insert_with(|| (PackedSrc::F32, QuantScheme::GgufQ4_0, shape));
    }

    // Per-layer projection fusion. Each row of a Q4K weight is an
    // independent block sequence, so collapsing N proj weights into one
    // by byte-concat along the output (`n`) axis lets the runtime issue
    // one matmul + N narrows instead of N matmuls — saves the dispatch
    // overhead llama.cpp hides via fused MSL kernels per layer. Every
    // backend (rlx-cpu / rlx-metal / rlx-mlx) gets the win for free via
    // its existing matmul + narrow lowerings.
    let fuse_gate_up = std::env::var("RLX_GEMMA_NO_FUSE_GATE_UP").as_deref() != Ok("1");
    let fuse_qkv = std::env::var("RLX_GEMMA_NO_FUSE_QKV").as_deref() != Ok("1");
    let num_layers = cfg.num_hidden_layers;
    // Fusion now concatenates the borrow *recipes* (ordered GGUF keys), not the
    // bytes: at attach the components are borrowed from the mmap and copied into
    // one scratch buffer in sequence — same fused layout, nothing resident.
    for layer in 0..num_layers {
        // FFN: gate_proj ‖ up_proj.
        if fuse_gate_up {
            let gk = format!("model.layers.{layer}.mlp.gate_proj.weight");
            let uk = format!("model.layers.{layer}.mlp.up_proj.weight");
            if let (
                Some((
                    PackedSrc::Borrow {
                        keys: gate_keys,
                        nbytes: gate_nb,
                    },
                    gate_scheme,
                    gate_shape,
                )),
                Some((
                    PackedSrc::Borrow {
                        keys: up_keys,
                        nbytes: up_nb,
                    },
                    up_scheme,
                    up_shape,
                )),
            ) = (packed.get(&gk).cloned(), packed.get(&uk).cloned())
            {
                let gate_n = gate_shape.first().copied().unwrap_or(0);
                let gate_k = gate_shape.get(1).copied().unwrap_or(0);
                let up_n = up_shape.first().copied().unwrap_or(0);
                let up_k = up_shape.get(1).copied().unwrap_or(0);
                if gate_scheme == up_scheme && gate_k > 0 && gate_k == up_k {
                    let mut fused_keys = gate_keys;
                    fused_keys.extend(up_keys);
                    packed.insert(
                        format!("model.layers.{layer}.mlp.gate_up.weight"),
                        (
                            PackedSrc::Borrow {
                                keys: fused_keys,
                                nbytes: gate_nb + up_nb,
                            },
                            gate_scheme,
                            vec![gate_n + up_n, gate_k],
                        ),
                    );
                    packed.remove(&gk);
                    packed.remove(&uk);
                }
            }
        }
        // Attention: q_proj ‖ k_proj [‖ v_proj]. With attention_k_eq_v the v
        // stream aliases k so the GGUF has no v_proj.weight; the fused key is
        // q‖k only and the graph builder uses k as v.
        if fuse_qkv {
            let qk_key = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let kk_key = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let vk_key = format!("model.layers.{layer}.self_attn.v_proj.weight");
            if let (
                Some((
                    PackedSrc::Borrow {
                        keys: q_keys,
                        nbytes: q_nb,
                    },
                    q_scheme,
                    q_shape,
                )),
                Some((
                    PackedSrc::Borrow {
                        keys: k_keys,
                        nbytes: k_nb,
                    },
                    k_scheme,
                    k_shape,
                )),
            ) = (packed.get(&qk_key).cloned(), packed.get(&kk_key).cloned())
            {
                let v_entry = packed.get(&vk_key).cloned();
                let q_n = q_shape.first().copied().unwrap_or(0);
                let q_k_dim = q_shape.get(1).copied().unwrap_or(0);
                let k_n = k_shape.first().copied().unwrap_or(0);
                let k_k_dim = k_shape.get(1).copied().unwrap_or(0);
                if q_scheme == k_scheme && q_k_dim > 0 && q_k_dim == k_k_dim {
                    let mut fused_keys = q_keys;
                    fused_keys.extend(k_keys);
                    let mut total_nb = q_nb + k_nb;
                    let (mut total_n, has_v) = (q_n + k_n, v_entry.is_some());
                    if let Some((v_src, v_scheme, v_shape)) = v_entry.as_ref() {
                        let v_n = v_shape.first().copied().unwrap_or(0);
                        let v_k_dim = v_shape.get(1).copied().unwrap_or(0);
                        match v_src {
                            PackedSrc::Borrow {
                                keys: v_keys,
                                nbytes: v_nb,
                            } if *v_scheme == q_scheme && v_k_dim == q_k_dim => {
                                fused_keys.extend(v_keys.clone());
                                total_nb += *v_nb;
                                total_n += v_n;
                            }
                            // Mixed quant / non-borrowable — skip fusion for
                            // this layer rather than emit a broken tensor.
                            _ => continue,
                        }
                    }
                    packed.insert(
                        format!("model.layers.{layer}.self_attn.qkv.weight"),
                        (
                            PackedSrc::Borrow {
                                keys: fused_keys,
                                nbytes: total_nb,
                            },
                            q_scheme,
                            vec![total_n, q_k_dim],
                        ),
                    );
                    packed.remove(&qk_key);
                    packed.remove(&kk_key);
                    if has_v {
                        packed.remove(&vk_key);
                    }
                }
            }
        }
    }

    Ok((f32_params, packed))
}

#[allow(clippy::too_many_arguments)]
/// gemma-3n `gelu_topk` gate activation:
/// `gelu_approx(max(0, gate − (mean + std·std_mul)))`, with `mean`/`std`
/// (population, ddof=0) reduced over the last (intermediate) axis per token.
/// Matches HF `Gemma3nTextMLP._gaussian_topk` + `nn.gelu_approx`.
fn gelu_topk_gate(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    gate: rlx_ir::NodeId,
    std_mul: f32,
    name: &str,
) -> rlx_ir::NodeId {
    use rlx_ir::{DType, Shape};
    let mean = g.mean(gate, vec![2], true); // [b, s, 1]
    let sq = g.mul(gate, gate);
    let msq = g.mean(sq, vec![2], true); // E[x²]
    let mean2 = g.mul(mean, mean); // E[x]²
    let var = g.sub(msq, mean2); // population variance
    let var = g.relu(var); // clamp float-noise negatives before sqrt
    let std = g.sqrt(var);
    let sm = g.param(&format!("{name}.stdmul"), Shape::new(&[1], DType::F32));
    params.insert(format!("{name}.stdmul"), vec![std_mul]);
    let std_scaled = g.mul(std, sm); // std·std_mul (scalar broadcast)
    let cutoff = g.add(mean, std_scaled); // mean + std·std_mul
    let shifted = g.sub(gate, cutoff); // gate − cutoff, [b,s,int]
    let relud = g.relu(shifted); // max(0, ·)
    g.gelu_approx(relud)
}

pub fn build_gemma_graph_sized_packed(
    cfg: &GemmaConfig,
    weights: &mut rlx_core::weight_loader::GgufLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    with_kv_outputs: bool,
    packed: &mut PackedWeightMap,
) -> Result<(Graph, F32WeightMap)> {
    build_gemma_graph_sized_packed_ext(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_token_from_input,
        with_kv_outputs,
        packed,
        None,
        None,
    )
}

/// Like [`build_gemma_graph_sized_packed`] but can rebuild from session weight caches.
#[allow(clippy::too_many_arguments)]
pub fn build_gemma_graph_sized_packed_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_token_from_input: bool,
    with_kv_outputs: bool,
    packed: &mut PackedWeightMap,
    known_packed: Option<&PackedWeightMap>,
    known_f32: Option<&F32WeightMap>,
) -> Result<(Graph, F32WeightMap)> {
    use crate::config::GemmaArch;
    use crate::rope::{build_rope_tables, resolve_inv_freq};
    use rlx_core::weight_loader::WeightLoader;
    use rlx_ir::op::{Activation, Op};
    use rlx_ir::quant::QuantScheme;
    use rlx_ir::{DType, NodeId, Shape};

    validate_cfg(cfg)?;

    // gemma-3n: 4-stream AltUp + Laurel wrapper needs a dedicated graph shape.
    // Prefill-only; ignores the packed/kv/last-token knobs (mlx-affine is F32).
    if cfg.has_altup() {
        let _ = (
            with_lm_head,
            last_token_from_input,
            with_kv_outputs,
            &known_packed,
            &known_f32,
        );
        return build_gemma3n_prefill_graph(cfg, weights, batch, seq);
    }

    let mut g = Graph::new("gemma_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps as f32;
    let num_layers = cfg.active_num_layers();

    // ── Helpers ────────────────────────────────────────────────────
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
    fn load_p_cached(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        key: &str,
        shape: &[usize],
        transpose: bool,
    ) -> Result<NodeId> {
        let (data, out_shape) = if let Some(cached) = known_f32.and_then(|m| m.get(key)) {
            if transpose {
                let rows = shape[0];
                let cols = shape[1];
                let mut t = vec![0f32; cached.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = cached[r * cols + c];
                    }
                }
                (t, vec![cols, rows])
            } else {
                (cached.clone(), shape.to_vec())
            }
        } else if transpose {
            weights.take_transposed(key)?
        } else {
            weights.take(key)?
        };
        let id = g.param(key, Shape::new(&out_shape, DType::F32));
        params.insert(key.to_string(), data);
        Ok(id)
    }
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut PackedWeightMap,
        weights: &mut dyn WeightLoader,
        known_packed: Option<&PackedWeightMap>,
        known_f32: Option<&F32WeightMap>,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>)> {
        if let Some((src, scheme, shape)) = known_packed.and_then(|m| m.get(key)) {
            if src.is_f32() {
                let cached = known_f32
                    .and_then(|m| m.get(key))
                    .ok_or_else(|| anyhow::anyhow!("f32 cache miss for drained proj {key}"))?;
                let id = g.param(key, Shape::new(shape, DType::F32));
                params.insert(key.to_string(), cached.clone());
                return Ok((id, None));
            }
            let id = g.param(key, Shape::new(&[src.nbytes()], DType::U8));
            return Ok((id, Some(*scheme)));
        }
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (PackedSrc::Owned(bytes), scheme, shape));
            Ok((id, Some(scheme)))
        } else {
            let (data, shape) = weights.take_transposed(key)?;
            let id = g.param(key, Shape::new(&shape, DType::F32));
            params.insert(key.to_string(), data);
            // Sentinel: proj was materialized to F32 on drain (e.g. Q4_0); rebuild from cache.
            packed.insert(
                key.to_string(),
                (PackedSrc::F32, QuantScheme::GgufQ4_0, shape),
            );
            Ok((id, None))
        }
    }
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
    /// Delta-gamma RMS norm: gamma = 1 + loaded_weight.
    fn gemma_rms(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        x: NodeId,
        weight_key: &str,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        _zero_beta: NodeId,
        h: usize,
        eps: f32,
    ) -> Result<NodeId> {
        let w = load_p_cached(g, params, weights, known_f32, weight_key, &[h], false)?;
        let ones = synth_const(
            g,
            params,
            &format!("{weight_key}.ones"),
            vec![1.0f32; h],
            &[h],
        );
        let gamma = g.add(ones, w);
        // Build a beta sized to `h` rather than reusing the caller's
        // hidden-sized zero_beta — otherwise Q/K norm (h=head_dim) gets
        // a beta of size hidden_size and the RmsNorm op's gamma/beta
        // shapes disagree, propagating NaN through the network. This
        // was the root cause for task #50's persistent all-NaN logits.
        let beta = synth_const(
            g,
            params,
            &format!("{weight_key}.beta"),
            vec![0.0f32; h],
            &[h],
        );
        Ok(g.rms_norm(x, gamma, beta, eps))
    }

    let zero_beta = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.zero_beta",
        vec![0.0f32; h],
        &[h],
    );

    // ── Default RoPE table (sliding layers + non-Gemma-4) ─────────
    let inv_freq = if known_f32.is_some() {
        resolve_inv_freq(cfg, None)
    } else {
        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
        resolve_inv_freq(cfg, rope_factors.as_deref())
    };
    let half = inv_freq.len();
    let rope_len = seq;
    let (cos_id, sin_id) = if let (Some(cos), Some(sin)) = (
        known_f32.and_then(|m| m.get("rope.cos")),
        known_f32.and_then(|m| m.get("rope.sin")),
    ) {
        (
            synth_const(
                &mut g,
                &mut params,
                "rope.cos",
                slice_rope_table(cos, half, rope_len),
                &[rope_len, half],
            ),
            synth_const(
                &mut g,
                &mut params,
                "rope.sin",
                slice_rope_table(sin, half, rope_len),
                &[rope_len, half],
            ),
        )
    } else {
        let rope_factors = weights.take("rope_freqs.weight").ok().map(|(d, _)| d);
        let inv = resolve_inv_freq(cfg, rope_factors.as_deref());
        let (cos_data, sin_data) = build_rope_tables(&inv, rope_len);
        (
            synth_const(&mut g, &mut params, "rope.cos", cos_data, &[rope_len, half]),
            synth_const(&mut g, &mut params, "rope.sin", sin_data, &[rope_len, half]),
        )
    };

    // ── Secondary "global" RoPE table for Gemma 4 full-attention ──
    let (global_cos, global_sin) = if let (Some(cos), Some(sin)) = (
        known_f32.and_then(|m| m.get("rope.global.cos")),
        known_f32.and_then(|m| m.get("rope.global.sin")),
    ) {
        // Drain sizes the cached cos/sin table to `rope_rows` (= `max_seq + 16`
        // per `drain_gemma_packed_weights_ext`), NOT `max_position_embeddings`.
        // The old inference `half_g = cos.len() / max_position_embeddings`
        // gave 0 for capped tables (18432 / 262144 = 0), registering
        // rope.global.cos as 0-sized → empty tensors propagated through every
        // FULL-attention layer's RoPE.
        //
        // The full-attention head_dim can differ from the SWA head_dim (Gemma
        // 4 12B: SWA=256 → half=128, FULL=512 → half_g=256), so `half` is the
        // wrong upper bound for the FULL layer. Recover the FULL half_g from
        // the model config the same way drain did: ask
        // `resolve_global_inv_freq` for the global inverse-frequency table —
        // its length is the FULL-layer `head_dim / 2`.
        let half_g = crate::rope::resolve_global_inv_freq(cfg, None)
            .map(|v| v.len())
            .unwrap_or(half);
        (
            Some(synth_const(
                &mut g,
                &mut params,
                "rope.global.cos",
                slice_rope_table(cos, half_g, rope_len),
                &[rope_len, half_g],
            )),
            Some(synth_const(
                &mut g,
                &mut params,
                "rope.global.sin",
                slice_rope_table(sin, half_g, rope_len),
                &[rope_len, half_g],
            )),
        )
    } else if let Some(global_inv) = crate::rope::resolve_global_inv_freq(cfg, None) {
        let half_g = global_inv.len();
        let (cd, sd) = build_rope_tables(&global_inv, rope_len);
        let c = synth_const(
            &mut g,
            &mut params,
            "rope.global.cos",
            cd,
            &[rope_len, half_g],
        );
        let s = synth_const(
            &mut g,
            &mut params,
            "rope.global.sin",
            sd,
            &[rope_len, half_g],
        );
        (Some(c), Some(s))
    } else {
        (None, None)
    };

    let vocab = cfg.vocab_size;
    // Task #37: when the embed table is in `known_packed` (Q4K bytes), bypass
    // the in-graph `gather(embed_w, input_ids)` and accept the gathered rows
    // directly. The runtime (`packed_session.rs::gather_embed_rows`) dequants
    // only the prompt-token rows host-side. Skips the ~3.8 GB f32 embed cache
    // and the per-bucket embed param upload.
    let embed_lazy = known_packed
        .map(|m| m.contains_key("model.embed_tokens.weight"))
        .unwrap_or(false)
        && !known_f32
            .map(|m| m.contains_key("model.embed_tokens.weight"))
            .unwrap_or(false);
    // Multimodal: when the caller marks `__media_bias__`, attention uses an
    // additive bias tensor `[batch, heads, seq, seq]` (`MaskKind::Bias`) instead
    // of the fused causal/sliding `MaskKind` — letting bidirectional image/audio
    // blocks open up. Valid for `seq <= sliding_window` (sliding == causal then),
    // so one bias serves all layer types. Gated entirely on the sentinel: callers
    // that don't set it are byte-identical to before.
    let media_bias_id = if known_packed
        .map(|m| m.contains_key("__media_bias__"))
        .unwrap_or(false)
    {
        Some(g.input("attn_bias", Shape::new(&[batch, nh, seq, seq], DType::F32)))
    } else {
        None
    };
    let last_token_idx = if with_lm_head && last_token_from_input {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::F32)))
    } else {
        None
    };
    let mut h_id = if embed_lazy {
        g.input("input_embeddings", Shape::new(&[batch, seq, h], DType::F32))
    } else {
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
        let embed_w = load_p_cached(
            &mut g,
            &mut params,
            weights,
            known_f32,
            "model.embed_tokens.weight",
            &[vocab, h],
            false,
        )?;
        g.gather_(embed_w, input_ids, 0)
    };
    // Gemma embed-scale (sqrt(hidden_size)) — emitted as a single
    // scalar multiply; matches GemmaFlow's EmbedScaleStage.
    let scale_val = (h as f32).sqrt();
    let embed_scale = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.embed_scale",
        vec![scale_val],
        &[1],
    );
    h_id = g.mul(h_id, embed_scale);

    let attn_score_scale = cfg.attn_score_scale();
    let attn_softcap = cfg.attn_logit_softcapping;
    let mut kv_outputs: Vec<(NodeId, NodeId)> = Vec::new();
    // Gemma 4 E2B KV sharing: the last fresh layer of each attention type
    // (sliding / full) stores its post-norm/post-RoPE K and post-v_norm V;
    // shared layers (>= first_kv_shared_layer) reuse them instead of computing
    // their own. Index 0 = sliding, 1 = full. Empty for non-E2B configs.
    let mut shared_k: [Option<NodeId>; 2] = [None, None];
    let mut shared_v: [Option<NodeId>; 2] = [None, None];
    // Gemma 4 E2B Per-Layer Embeddings: the runner precomputes the per-layer
    // input slices [batch, seq, num_layers * ple_w] (gather + dequant + project
    // + combine) and feeds them as a graph input; each layer slices its block.
    let per_layer_inputs = if cfg.has_ple() {
        Some(g.input(
            "per_layer_inputs",
            Shape::new(&[batch, seq, num_layers * cfg.ple_width()], f),
        ))
    } else {
        None
    };
    // Diagnostic tap: when RLX_TAP_L0=1, surface layer-0 intermediates
    // as additional graph outputs so we can bisect Metal's all-NaN bug.
    // Order (consumed in `packed_session::predict_logits`):
    //   1: embed*scale (h_id at layer-0 entry)
    //   2: input_layernorm(x)
    //   3: Q after per-head q_norm + reshape back to [B,S,q_dim]
    //   4: K after per-head k_norm + reshape back to [B,S,kv_dim]
    //   5: V after v_norm + reshape back
    //   6: Q after RoPE
    //   7: K after RoPE
    //   8: Attention output (pre-o_proj)
    //   9: attn_out after post_attention_norm
    //  10: residual h_id + attn_out
    //  11: layer 0 final h (post-FFN add)
    let tap_l0 = std::env::var("RLX_TAP_L0").ok().is_some();
    let tap_layer: usize = std::env::var("RLX_TAP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut l0_taps: Vec<NodeId> = Vec::new();
    // RLX_TAP_ALL: append every layer's final hidden state as a graph output,
    // for full-trajectory parity bisection against HF hidden_states.
    let tap_all = std::env::var("RLX_TAP_ALL").ok().is_some();
    let mut all_layer_taps: Vec<NodeId> = Vec::new();

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;
        // Gemma 4 E2B: KV-shared layers use a double-wide MLP
        // (intermediate_size × 2). Non-E2B configs return the base size,
        // leaving flagship/legacy behavior unchanged.
        let int_dim = cfg.layer_intermediate_size(layer);
        let is_shared_kv = cfg.is_kv_shared_layer(layer);
        let kv_type_idx = cfg.is_full_attention_layer(layer) as usize;

        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 1: embed*scale (layer-0 input)
        }

        // input_layernorm.
        let normed_in = gemma_rms(
            &mut g,
            &mut params,
            h_id,
            &format!("{lp}.input_layernorm.weight"),
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;
        if tap_l0 && layer == tap_layer {
            l0_taps.push(normed_in); // tap 2: input_layernorm(x)
        }

        // Q/K/V projections. v_proj is skipped when k_eq_v.
        let fused_qkv_key = format!("{lp}.self_attn.qkv.weight");
        let has_fused_qkv = known_packed
            .map(|m| m.contains_key(&fused_qkv_key))
            .unwrap_or(false);
        let (q, k, v) = if has_fused_qkv {
            let (qkv_w, qkv_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_qkv_key,
            )?;
            let v_present = !cfg.layer_k_eq_v(layer);
            let total_n = q_dim + kv_dim + if v_present { kv_dim } else { 0 };
            let combined = emit_proj(
                &mut g,
                normed_in,
                qkv_w,
                qkv_s,
                Shape::new(&[batch, seq, total_n], f),
            );
            let q = g.narrow_(combined, 2, 0, q_dim);
            let k = g.narrow_(combined, 2, q_dim, kv_dim);
            let v = if v_present {
                g.narrow_(combined, 2, q_dim + kv_dim, kv_dim)
            } else {
                k
            };
            (q, k, v)
        } else {
            let (q_w, q_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let q = emit_proj(
                &mut g,
                normed_in,
                q_w,
                q_s,
                Shape::new(&[batch, seq, q_dim], f),
            );
            // KV-shared layers (Gemma 4 E2B, >= first_kv_shared_layer) reuse
            // K/V from an earlier same-type layer and ship no usable
            // k_norm/k_proj/v_proj — skip their projections entirely; the real
            // K/V are substituted after the RoPE step below. `q` is a harmless
            // placeholder for k/v here.
            let (k, v) = if is_shared_kv {
                (q, q)
            } else {
                let (k_w, k_s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    known_packed,
                    known_f32,
                    &format!("{lp}.self_attn.k_proj.weight"),
                )?;
                let k = emit_proj(
                    &mut g,
                    normed_in,
                    k_w,
                    k_s,
                    Shape::new(&[batch, seq, kv_dim], f),
                );
                let v = if cfg.layer_k_eq_v(layer) {
                    k
                } else {
                    let (v_w, v_s) = load_proj(
                        &mut g,
                        &mut params,
                        packed,
                        weights,
                        known_packed,
                        known_f32,
                        &format!("{lp}.self_attn.v_proj.weight"),
                    )?;
                    emit_proj(
                        &mut g,
                        normed_in,
                        v_w,
                        v_s,
                        Shape::new(&[batch, seq, kv_dim], f),
                    )
                };
                (k, v)
            };
            (q, k, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q); // tap A (was 3): Q POST-projection, PRE per-head norm
            l0_taps.push(k); // tap B: K post-projection, pre per-head norm
            l0_taps.push(v); // tap C: V post-projection
        }

        // Gemma 4 per-head Q/K RMS norms + V RMS norm (see decode
        // builder for full rationale + task #50). Same fix applied
        // here so predict_logits (prefill path) doesn't produce
        // all-NaN logits.
        let (q, k, v) = if matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
            let q_norm_key = format!("{lp}.self_attn.q_norm.weight");
            let k_norm_key = format!("{lp}.self_attn.k_norm.weight");
            let q_4d = g.reshape_(
                q,
                vec![batch as i64, seq as i64, nh as i64, layer_dh as i64],
            );
            if tap_l0 && layer == tap_layer {
                l0_taps.push(q_4d); // tap D: Q reshape-only (4D pre-norm)
            }
            let q_normed = gemma_rms(
                &mut g,
                &mut params,
                q_4d,
                &q_norm_key,
                weights,
                known_f32,
                zero_beta,
                layer_dh,
                eps,
            )?;
            if tap_l0 && layer == tap_layer {
                l0_taps.push(q_normed); // tap E: Q after per-head RMS norm (4D)
            }
            let q = g.reshape_(q_normed, vec![batch as i64, seq as i64, q_dim as i64]);

            // KV-shared layers don't carry k_norm/v_norm and reuse stored K/V,
            // so skip the K/V norm entirely (the `k`/`v` placeholders are
            // discarded after RoPE below).
            let (k, v) = if is_shared_kv {
                (k, v)
            } else {
                let k_4d = g.reshape_(
                    k,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let k_normed = gemma_rms(
                    &mut g,
                    &mut params,
                    k_4d,
                    &k_norm_key,
                    weights,
                    known_f32,
                    zero_beta,
                    layer_dh,
                    eps,
                )?;
                let k = g.reshape_(k_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                // Gemma 4 applies per-head V RMS-norm; Gemma 3 matches llama.cpp (V unchanged).
                let v = if matches!(cfg.arch, GemmaArch::Gemma4) {
                    let v_4d = g.reshape_(
                        v,
                        vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                    );
                    let v_ones = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.self_attn.v_norm.ones"),
                        vec![1.0f32; layer_dh],
                        &[layer_dh],
                    );
                    let v_zeros = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.self_attn.v_norm.zeros"),
                        vec![0.0f32; layer_dh],
                        &[layer_dh],
                    );
                    let v_normed = g.rms_norm(v_4d, v_ones, v_zeros, eps);
                    g.reshape_(v_normed, vec![batch as i64, seq as i64, kv_dim as i64])
                } else {
                    v
                };
                (k, v)
            };
            (q, k, v)
        } else {
            (q, k, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q); // tap 3: Q post-norm (after per-head q_norm)
            l0_taps.push(k); // tap 4: K post-norm
            l0_taps.push(v); // tap 5: V post-norm
        }

        // RoPE — pick global slot for full-attention layers when split.
        let (layer_cos, layer_sin) = if cfg.is_full_attention_layer(layer) {
            match (global_cos, global_sin) {
                (Some(gc), Some(gs)) => (gc, gs),
                _ => (cos_id, sin_id),
            }
        } else {
            (cos_id, sin_id)
        };
        let q_rope = g.rope_n(q, layer_cos, layer_sin, layer_dh, layer_nrot);
        // E2B KV sharing: shared layers reuse the stored same-type (post-norm,
        // post-RoPE) K and (post-v_norm) V; fresh layers RoPE their own K and,
        // when sharing is active, store it for later same-type layers.
        let (k_rope, v) = if is_shared_kv {
            let sk = shared_k[kv_type_idx].ok_or_else(|| {
                anyhow!("KV-shared layer {layer}: no stored source K for type {kv_type_idx}")
            })?;
            let sv = shared_v[kv_type_idx]
                .ok_or_else(|| anyhow!("KV-shared layer {layer}: no stored source V"))?;
            (sk, sv)
        } else {
            let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);
            if cfg.num_kv_shared_layers > 0 {
                shared_k[kv_type_idx] = Some(k_rope);
                shared_v[kv_type_idx] = Some(v);
            }
            (k_rope, v)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(q_rope); // tap 6: Q post-RoPE
            l0_taps.push(k_rope); // tap 7: K post-RoPE
        }
        if with_kv_outputs {
            kv_outputs.push((k_rope, v));
        }

        let k_rep = repeat_kv_packed(&mut g, k_rope, layer_kv, layer_dh, group);
        let v_rep = repeat_kv_packed(&mut g, v, layer_kv, layer_dh, group);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(k_rep); // tap F: K_rep
            l0_taps.push(v_rep); // tap G: V_rep
        }

        // Gemma 2/3: llama.cpp scales Q before SDPA and passes 1.0 as attn scale.
        let (q_attn, layer_attn_scale) =
            if matches!(cfg.arch, GemmaArch::Gemma2 | GemmaArch::Gemma3) {
                if let Some(scale) = attn_score_scale {
                    let q_scale = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.attn.q_score_scale"),
                        vec![scale],
                        &[1],
                    );
                    (g.mul(q_rope, q_scale), Some(1.0f32))
                } else {
                    (q_rope, attn_score_scale)
                }
            } else {
                (q_rope, attn_score_scale)
            };

        // Per-layer mask.
        let (mask_kind, _, _) = cfg.layer_attn_options(layer);
        let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_attn));
        let attn = if let Some(bias) = media_bias_id {
            g.attention_bias_opts(
                q_attn,
                k_rep,
                v_rep,
                bias,
                nh,
                layer_dh,
                attn_shape,
                layer_attn_scale,
                attn_softcap,
            )
        } else {
            g.attention_kind_opts(
                q_attn,
                k_rep,
                v_rep,
                nh,
                layer_dh,
                mask_kind,
                attn_shape,
                layer_attn_scale,
                attn_softcap,
            )
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(attn); // tap 8: attention output (pre-o_proj)
        }

        let (o_w, o_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        // Gemma 3/4 sandwich-norm: post_attention_layernorm applied to
        // attn_out BEFORE the residual add. Without this the attention
        // contribution grows unbounded across 48 layers — task #50 NaN
        // root cause. Maps GGUF blk.N.post_attention_norm.weight via
        // gguf_to_hf_name_for_arch.
        // Post-attention norm: Gemma 2/3/4 all have it (only Gemma 1 lacks it),
        // matching the post_feedforward_layernorm gate below. Dropping it for
        // Gemma 2 corrupted the whole forward (hidden cos ~0.35 vs llama.cpp).
        let attn_out = if cfg.arch != GemmaArch::Gemma {
            gemma_rms(
                &mut g,
                &mut params,
                attn_out,
                &format!("{lp}.post_attention_layernorm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?
        } else {
            attn_out
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(attn_out); // tap 9: attn_out after post_attn_norm
        }
        let post_attn = g.add(h_id, attn_out);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(post_attn); // tap 10: residual h + attn_out
        }

        // Pre-FFN norm — Gemma 1 uses `post_attention_layernorm`;
        // Gemma 2/3/4 use `pre_feedforward_layernorm`.
        let pre_ffn_key = if cfg.arch == GemmaArch::Gemma {
            format!("{lp}.post_attention_layernorm.weight")
        } else {
            format!("{lp}.pre_feedforward_layernorm.weight")
        };
        let normed_post = gemma_rms(
            &mut g,
            &mut params,
            post_attn,
            &pre_ffn_key,
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;
        if tap_l0 && layer == tap_layer {
            l0_taps.push(normed_post); // tap 10b: pre-FFN rms norm
        }

        // GeGLU MLP.
        // Try fused {gate_up}_proj first — set up at drain time by
        // `drain_gemma_packed_weights`. One matmul + narrow halves
        // replaces two matmuls on the same input.
        let fused_gate_up_key = format!("{lp}.mlp.gate_up.weight");
        let fused_gate_up = known_packed
            .map(|m| m.contains_key(&fused_gate_up_key))
            .unwrap_or(false);
        let (gate, up) = if fused_gate_up {
            let (gu_w, gu_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_gate_up_key,
            )?;
            let combined = emit_proj(
                &mut g,
                normed_post,
                gu_w,
                gu_s,
                Shape::new(&[batch, seq, int_dim * 2], f),
            );
            let gate = g.narrow_(combined, 2, 0, int_dim);
            let up = g.narrow_(combined, 2, int_dim, int_dim);
            (gate, up)
        } else {
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let (up_w, up_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed_post,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            let up = emit_proj(
                &mut g,
                normed_post,
                up_w,
                up_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            (gate, up)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(gate); // tap 10c: gate proj
            l0_taps.push(up); // tap 10d: up proj
        }
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        // gemma-3n activation sparsity: sparse layers apply `gelu_topk` — a
        // per-token gate cutoff at `mean + std·√2·erfinv(2·sparsity−1)` before
        // the GELU (HF `Gemma3nTextMLP._gaussian_topk`). Dense layers (and every
        // gemma-2/3/4 layer) keep the plain approx-GELU.
        let gate_act = if let Some(std_mul) = cfg.layer_gaussian_std_multiplier(layer) {
            gelu_topk_gate(
                &mut g,
                &mut params,
                gate,
                std_mul,
                &format!("{lp}.mlp.act_sparsity"),
            )
        } else {
            g.gelu_approx(gate)
        };
        if tap_l0 && layer == tap_layer {
            l0_taps.push(gate_act); // tap 10e: gelu(gate)
        }
        let mlp_inner = g.mul(gate_act, up);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(mlp_inner); // tap 10f: gate*up
        }
        let mut ffn_out = emit_proj(
            &mut g,
            mlp_inner,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );
        if tap_l0 && layer == tap_layer {
            l0_taps.push(ffn_out); // tap 10g: down proj (pre post_ffn norm)
        }

        // Post-FFN norm for Gemma 2/3/4.
        if cfg.arch != GemmaArch::Gemma {
            let post_ffn_key = format!("{lp}.post_feedforward_layernorm.weight");
            ffn_out = gemma_rms(
                &mut g,
                &mut params,
                ffn_out,
                &post_ffn_key,
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
        }

        h_id = g.add(post_attn, ffn_out);
        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 12: residual after FFN (pre-PLE)
        }

        // Gemma 4 E2B Per-Layer Embeddings: inject this layer's per-layer input
        // slice after the MLP residual (HF `Gemma4TextDecoderLayer.forward`):
        //   res = h; h = post_norm( per_layer_projection(
        //             gelu(per_layer_input_gate(h)) * ple_slice ) ); h = res + h
        if let Some(ple_all) = per_layer_inputs {
            let pw = cfg.ple_width();
            let ple_slice = g.narrow_(ple_all, 2, layer * pw, pw); // [B,S,ple_w]
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_input_gate.weight"),
            )?;
            let gated = emit_proj(
                &mut g,
                h_id,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, pw], f),
            );
            let gated = g.gelu_approx(gated);
            let gated = g.mul(gated, ple_slice);
            let (proj_w, proj_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_projection.weight"),
            )?;
            let projected = emit_proj(
                &mut g,
                gated,
                proj_w,
                proj_s,
                Shape::new(&[batch, seq, h], f),
            );
            let projected = gemma_rms(
                &mut g,
                &mut params,
                projected,
                &format!("{lp}.post_per_layer_input_norm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
            h_id = g.add(h_id, projected);
            if tap_l0 && layer == tap_layer {
                l0_taps.push(h_id); // tap 13: after PLE (pre layer_scalar)
            }
        }

        // Gemma 4 per-layer output scalar: E2B uses `layer_scalar`; flagship
        // GGUF ships `layer_output_scale` (≈0.02–0.05). llama.cpp multiplies
        // every layer output by this before the next residual; skipping it lets
        // the hidden stream grow ~20–50× per layer → softcap-saturated garbage
        // logits on prefill (decode already applied this in the builder below).
        // gemma-4 mobile-QAT ships a per-layer `layer_scalar`; gemma-3n does not
        // (its per-layer output magnitude is governed by AltUp instead). Apply it
        // only when present so both checkpoints build.
        if cfg.has_ple() {
            let lsk = format!("{lp}.layer_scalar");
            if let Ok((data, shape)) = weights.take(&lsk) {
                let id = g.param(&lsk, Shape::new(&shape, DType::F32));
                params.insert(lsk, data);
                h_id = g.mul(h_id, id);
            }
            // else: gemma-3n has no per-layer output scalar (AltUp governs it).
        } else if matches!(cfg.arch, GemmaArch::Gemma4) {
            // [1] output_scale scalar → full [hidden] vector. A [1]-param
            // broadcast multiply mis-reads on the CUDA backend (garbage slot →
            // hidden explodes to ~1e11 → NaN → flat logits); the last-dim
            // [hidden] broadcast is the proven norm-gamma path. Bit-identical
            // to CPU (same scalar, just materialised across the row).
            let okey = format!("{lp}.self_attn.output_scale.weight");
            let sval = match known_f32.and_then(|m| m.get(&okey)) {
                Some(v) => v.first().copied().unwrap_or(1.0),
                None => weights.take(&okey)?.0.first().copied().unwrap_or(1.0),
            };
            let scale_w = synth_const(
                &mut g,
                &mut params,
                &format!("{okey}.bh"),
                vec![sval; cfg.hidden_size],
                &[cfg.hidden_size],
            );
            h_id = g.mul(h_id, scale_w);
        }
        if tap_l0 && layer == tap_layer {
            l0_taps.push(h_id); // tap 11: layer 0 final h
        }
        if tap_all {
            all_layer_taps.push(h_id);
        }
    }

    // model.norm + lm_head.
    let hidden = gemma_rms(
        &mut g,
        &mut params,
        h_id,
        "model.norm.weight",
        weights,
        known_f32,
        zero_beta,
        h,
        eps,
    )?;

    let out = if with_lm_head {
        // lm_head is normally tied to embed_tokens for Gemma. Use the
        // F32 transposed embed; quantized lm_head isn't ubiquitous.
        let head_input = if let Some(idx) = last_token_idx {
            gather_last_token_packed(&mut g, hidden, batch, idx)
        } else {
            hidden
        };
        let logit_rows = if last_token_from_input { 1 } else { seq };
        let vocab = cfg.vocab_size;
        // Task #36: when the embed bytes are still packed (Q4K/Q6K), issue a
        // single `Op::DequantMatMul` against them instead of building the
        // ~4 GB transposed f32 constant (and the ~4 GB clone into the graph).
        let packed_embed_scheme = known_packed
            .and_then(|m| m.get("model.embed_tokens.weight"))
            .map(|(_, scheme, _)| *scheme);
        let mut logits = if let Some(scheme) =
            packed_embed_scheme.filter(|_| cfg.tie_word_embeddings && with_lm_head)
        {
            let (w_id, _) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                "model.embed_tokens.weight",
            )?;
            let logits_shape = Shape::new(&[batch, logit_rows, vocab], f);
            g.add_node(
                Op::DequantMatMul { scheme },
                vec![head_input, w_id],
                logits_shape,
            )
        } else {
            let lm_head_w = if cfg.tie_word_embeddings {
                let embed = params
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
                let mut transposed = vec![0f32; embed.len()];
                for v in 0..vocab {
                    for hi in 0..h {
                        transposed[hi * vocab + v] = embed[v * h + hi];
                    }
                }
                synth_const(
                    &mut g,
                    &mut params,
                    "gemma.packed.lm_head.tied_t",
                    transposed,
                    &[h, vocab],
                )
            } else {
                load_p(&mut g, &mut params, weights, "lm_head.weight", true)?
            };
            g.mm(head_input, lm_head_w)
        };
        // Final logit soft-cap: tanh(x / cap) * cap.
        if let Some(cap) = cfg.final_logit_softcapping {
            let inv = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.softcap.inv.{cap}"),
                vec![1.0 / cap],
                &[1],
            );
            let cap_id = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.softcap.cap.{cap}"),
                vec![cap],
                &[1],
            );
            let scaled = g.mul(logits, inv);
            let scaled_shape = g.shape(scaled).clone();
            let t = g.add_node(Op::Activation(Activation::Tanh), vec![scaled], scaled_shape);
            logits = g.mul(t, cap_id);
        }
        let _ = logit_rows;
        logits
    } else {
        hidden
    };

    let mut outputs = vec![out];
    if with_kv_outputs {
        for (k, v) in kv_outputs {
            outputs.push(k);
            outputs.push(v);
        }
    }
    // Append layer-0 taps last so existing kv-output indexing isn't disturbed.
    // packed_session reads outputs in [logits, ...kv, ...taps] order.
    if tap_l0 {
        outputs.extend(l0_taps.iter().copied());
        eprintln!(
            "[rlx-gemma] RLX_TAP_L0: appended {} layer-0 taps as graph outputs",
            l0_taps.len()
        );
    }
    if tap_all {
        outputs.extend(all_layer_taps.iter().copied());
        eprintln!(
            "[rlx-gemma] RLX_TAP_ALL: appended {} per-layer hidden taps",
            all_layer_taps.len()
        );
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

/// Single-token decode graph with packed `Op::DequantMatMul` projections.
///
/// Expects runtime `rope_cos` / `rope_sin` rows (and optional global rows for
/// Gemma 4 full-attention layers). When `use_custom_mask` is true, supply a
/// bucketed mask of length `past_seq + 1`.
///
/// Precompute tied lm_head transpose once for packed decode bucket builds.
pub fn precompute_packed_decode_tied_lm_head(cfg: &GemmaConfig, embed: &[f32]) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    if embed.len() != vocab * h {
        return Err(anyhow!(
            "embed_tokens.weight len {} != vocab*hidden ({vocab}*{h})",
            embed.len()
        ));
    }
    let mut transposed = vec![0f32; embed.len()];
    for v in 0..vocab {
        for hi in 0..h {
            transposed[hi * vocab + v] = embed[v * h + hi];
        }
    }
    Ok(transposed)
}

#[allow(clippy::too_many_arguments)]
pub fn build_gemma_decode_graph_sized_packed(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    packed: &mut PackedWeightMap,
) -> Result<(Graph, F32WeightMap)> {
    build_gemma_decode_graph_sized_packed_ext(
        cfg,
        weights,
        batch,
        past_seq,
        use_custom_mask,
        PackedDecodeLmOutput::FullLogits,
        packed,
        None,
        None,
    )
}

/// Like [`build_gemma_decode_graph_sized_packed`] but reuses cached packed/F32 tensors
/// (avoids GGUF reload and tied-lm_head re-transpose on each decode bucket).
#[allow(clippy::too_many_arguments)]
pub fn build_gemma_decode_graph_sized_packed_ext(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    lm_output: PackedDecodeLmOutput,
    packed: &mut PackedWeightMap,
    known_packed: Option<&PackedWeightMap>,
    known_f32: Option<&F32WeightMap>,
) -> Result<(Graph, F32WeightMap)> {
    use crate::config::GemmaArch;
    use crate::rope::resolve_inv_freq;
    use rlx_core::weight_loader::WeightLoader;
    use rlx_ir::op::{Activation, Op};
    use rlx_ir::quant::QuantScheme;
    use rlx_ir::{DType, NodeId, Shape};

    validate_cfg(cfg)?;
    if batch != 1 {
        return Err(anyhow!("gemma packed decode requires batch=1"));
    }
    let seq = 1usize;

    let mut g = Graph::new("gemma_packed_decode");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps as f32;
    let num_layers = cfg.active_num_layers();

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
    fn load_p_cached(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        key: &str,
        shape: &[usize],
        transpose: bool,
    ) -> Result<NodeId> {
        let (data, out_shape) = if let Some(cached) = known_f32.and_then(|m| m.get(key)) {
            if transpose {
                let rows = shape[0];
                let cols = shape[1];
                let mut t = vec![0f32; cached.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = cached[r * cols + c];
                    }
                }
                (t, vec![cols, rows])
            } else {
                (cached.clone(), shape.to_vec())
            }
        } else if transpose {
            weights.take_transposed(key)?
        } else {
            weights.take(key)?
        };
        let id = g.param(key, Shape::new(&out_shape, DType::F32));
        params.insert(key.to_string(), data);
        Ok(id)
    }
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut PackedWeightMap,
        weights: &mut dyn WeightLoader,
        known_packed: Option<&PackedWeightMap>,
        known_f32: Option<&F32WeightMap>,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>)> {
        if let Some((src, scheme, shape)) = known_packed.and_then(|m| m.get(key)) {
            if src.is_f32() {
                let cached = known_f32
                    .and_then(|m| m.get(key))
                    .ok_or_else(|| anyhow::anyhow!("f32 cache miss for drained proj {key}"))?;
                let id = g.param(key, Shape::new(shape, DType::F32));
                params.insert(key.to_string(), cached.clone());
                return Ok((id, None));
            }
            let id = g.param(key, Shape::new(&[src.nbytes()], DType::U8));
            return Ok((id, Some(*scheme)));
        }
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (PackedSrc::Owned(bytes), scheme, shape));
            Ok((id, Some(scheme)))
        } else {
            let (data, shape) = weights.take_transposed(key)?;
            let id = g.param(key, Shape::new(&shape, DType::F32));
            params.insert(key.to_string(), data);
            // Sentinel: proj was materialized to F32 on drain (e.g. Q4_0); rebuild from cache.
            packed.insert(
                key.to_string(),
                (PackedSrc::F32, QuantScheme::GgufQ4_0, shape),
            );
            Ok((id, None))
        }
    }
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
    fn gemma_rms(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        x: NodeId,
        weight_key: &str,
        weights: &mut dyn WeightLoader,
        known_f32: Option<&HashMap<String, Vec<f32>>>,
        _zero_beta: NodeId,
        h: usize,
        eps: f32,
    ) -> Result<NodeId> {
        let w = load_p_cached(g, params, weights, known_f32, weight_key, &[h], false)?;
        let ones = synth_const(
            g,
            params,
            &format!("{weight_key}.ones"),
            vec![1.0f32; h],
            &[h],
        );
        let gamma = g.add(ones, w);
        // Build a beta sized to `h` rather than reusing the caller's
        // hidden-sized zero_beta — otherwise Q/K norm (h=head_dim) gets
        // a beta of size hidden_size and the RmsNorm op's gamma/beta
        // shapes disagree, propagating NaN through the network. This
        // was the root cause for task #50's persistent all-NaN logits.
        let beta = synth_const(
            g,
            params,
            &format!("{weight_key}.beta"),
            vec![0.0f32; h],
            &[h],
        );
        Ok(g.rms_norm(x, gamma, beta, eps))
    }

    let zero_beta = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.decode.zero_beta",
        vec![0.0f32; h],
        &[h],
    );

    let inv_freq = resolve_inv_freq(cfg, None);
    let half = inv_freq.len();
    let rope_cos = g.input("rope_cos", Shape::new(&[1, half], f));
    let rope_sin = g.input("rope_sin", Shape::new(&[1, half], f));

    let global_rope = crate::rope::resolve_global_inv_freq(cfg, None);
    let (global_cos_in, global_sin_in) = if let Some(global_inv) = global_rope {
        let half_g = global_inv.len();
        (
            Some(g.input("rope_cos_global", Shape::new(&[1, half_g], f))),
            Some(g.input("rope_sin_global", Shape::new(&[1, half_g], f))),
        )
    } else {
        (None, None)
    };

    let mask_id = if use_custom_mask {
        Some(g.input("mask", Shape::new(&[batch, past_seq + seq], f)))
    } else {
        None
    };

    let vocab = cfg.vocab_size;
    // Task #37: lazy-embed path — runtime host-gathers the single decode
    // token's row from packed Q4K bytes and feeds it as `input_embeddings`.
    // Skips the per-decode-bucket embed param upload (≈ 540 MB Q4K for 12B).
    let embed_lazy = known_packed
        .map(|m| m.contains_key("model.embed_tokens.weight"))
        .unwrap_or(false)
        && !known_f32
            .map(|m| m.contains_key("model.embed_tokens.weight"))
            .unwrap_or(false);
    let mut h_id = if embed_lazy {
        g.input("input_embeddings", Shape::new(&[batch, seq, h], DType::F32))
    } else {
        let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
        let embed_w = load_p_cached(
            &mut g,
            &mut params,
            weights,
            known_f32,
            "model.embed_tokens.weight",
            &[vocab, h],
            false,
        )?;
        g.gather_(embed_w, input_ids, 0)
    };
    let scale_val = (h as f32).sqrt();
    let embed_scale = synth_const(
        &mut g,
        &mut params,
        "gemma.packed.decode.embed_scale",
        vec![scale_val],
        &[1],
    );
    h_id = g.mul(h_id, embed_scale);

    let attn_score_scale = cfg.attn_score_scale();
    let attn_softcap = cfg.attn_logit_softcapping;
    let mut new_kv_outputs: Vec<(NodeId, NodeId)> = Vec::with_capacity(num_layers);

    // Gemma 4 E2B decode: Per-Layer-Embedding inputs for the current token
    // (seq=1 in decode), fed by the runner via `compute_per_layer_inputs`; and
    // the shared-KV store (last fresh same-type layer's post-cache K/V). Both
    // empty for non-E2B configs, leaving the flagship/legacy decode untouched.
    let per_layer_inputs_dec = if cfg.has_ple() {
        Some(g.input(
            "per_layer_inputs",
            Shape::new(&[batch, seq, num_layers * cfg.ple_width()], f),
        ))
    } else {
        None
    };
    let mut shared_k_dec: [Option<NodeId>; 2] = [None, None];
    let mut shared_v_dec: [Option<NodeId>; 2] = [None, None];

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let layer_dh = cfg.layer_head_dim(layer);
        let layer_kv = cfg.layer_num_kv_heads(layer);
        let layer_nrot = cfg.layer_n_rot(layer);
        let q_dim = nh * layer_dh;
        let kv_dim = layer_kv * layer_dh;
        let group = nh / layer_kv;
        // Gemma 4 E2B: KV-shared layers use a double-wide MLP
        // (intermediate_size × 2). Non-E2B configs return the base size,
        // leaving flagship/legacy behavior unchanged.
        let int_dim = cfg.layer_intermediate_size(layer);
        let is_shared_kv = cfg.is_kv_shared_layer(layer);
        let kv_type_idx = cfg.is_full_attention_layer(layer) as usize;

        let past_k = g.input(
            format!("past_k_{layer}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );
        let past_v = g.input(
            format!("past_v_{layer}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );

        let normed_in = gemma_rms(
            &mut g,
            &mut params,
            h_id,
            &format!("{lp}.input_layernorm.weight"),
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;

        let fused_qkv_key = format!("{lp}.self_attn.qkv.weight");
        let has_fused_qkv = known_packed
            .map(|m| m.contains_key(&fused_qkv_key))
            .unwrap_or(false);
        let (q, k, v) = if has_fused_qkv {
            let (qkv_w, qkv_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_qkv_key,
            )?;
            let v_present = !cfg.layer_k_eq_v(layer);
            let total_n = q_dim + kv_dim + if v_present { kv_dim } else { 0 };
            let combined = emit_proj(
                &mut g,
                normed_in,
                qkv_w,
                qkv_s,
                Shape::new(&[batch, seq, total_n], f),
            );
            let q = g.narrow_(combined, 2, 0, q_dim);
            let k = g.narrow_(combined, 2, q_dim, kv_dim);
            let v = if v_present {
                g.narrow_(combined, 2, q_dim + kv_dim, kv_dim)
            } else {
                k
            };
            (q, k, v)
        } else {
            let (q_w, q_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.self_attn.q_proj.weight"),
            )?;
            let q = emit_proj(
                &mut g,
                normed_in,
                q_w,
                q_s,
                Shape::new(&[batch, seq, q_dim], f),
            );
            // KV-shared layers (Gemma 4 E2B) reuse an earlier same-type layer's
            // cached K/V and ship no usable k_norm/k_proj/v_proj — skip them; the
            // real K/V are substituted after RoPE below. `q` is a placeholder.
            let (k, v) = if is_shared_kv {
                (q, q)
            } else {
                let (k_w, k_s) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    known_packed,
                    known_f32,
                    &format!("{lp}.self_attn.k_proj.weight"),
                )?;
                let k = emit_proj(
                    &mut g,
                    normed_in,
                    k_w,
                    k_s,
                    Shape::new(&[batch, seq, kv_dim], f),
                );
                let v = if cfg.layer_k_eq_v(layer) {
                    k
                } else {
                    let (v_w, v_s) = load_proj(
                        &mut g,
                        &mut params,
                        packed,
                        weights,
                        known_packed,
                        known_f32,
                        &format!("{lp}.self_attn.v_proj.weight"),
                    )?;
                    emit_proj(
                        &mut g,
                        normed_in,
                        v_w,
                        v_s,
                        Shape::new(&[batch, seq, kv_dim], f),
                    )
                };
                (k, v)
            };
            (q, k, v)
        };

        // Gemma 4 specific per-head Q/K RMS norms + plain V RMS norm.
        // Tensors:
        //   blk.N.attn_q_norm.weight [head_dim]
        //   blk.N.attn_k_norm.weight [head_dim]
        //   (V uses ggml_rms_norm with NO learnable weight)
        // Applied AFTER q/k/v projection, BEFORE RoPE. Without them,
        // Q·K·V grows unbounded → softmax overflow → NaN at layer 0.
        // V-norm matches llama.cpp gemma4.cpp:256 — plain RMSNorm with
        // gamma=1, beta=0 (just normalizes V to unit-RMS per head).
        let (q, k, v) = if matches!(cfg.arch, GemmaArch::Gemma3 | GemmaArch::Gemma4) {
            let q_norm_key = format!("{lp}.self_attn.q_norm.weight");
            let k_norm_key = format!("{lp}.self_attn.k_norm.weight");
            // q: [B, S, nh*dh] → [B, S, nh, dh] for per-head norm.
            let q_4d = g.reshape_(
                q,
                vec![batch as i64, seq as i64, nh as i64, layer_dh as i64],
            );
            let q_normed = gemma_rms(
                &mut g,
                &mut params,
                q_4d,
                &q_norm_key,
                weights,
                known_f32,
                zero_beta,
                layer_dh,
                eps,
            )?;
            let q = g.reshape_(q_normed, vec![batch as i64, seq as i64, q_dim as i64]);

            // Shared-KV layers carry no k_norm/v_norm and reuse cached K/V; skip.
            let (k, v) = if is_shared_kv {
                (k, v)
            } else {
                let k_4d = g.reshape_(
                    k,
                    vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                );
                let k_normed = gemma_rms(
                    &mut g,
                    &mut params,
                    k_4d,
                    &k_norm_key,
                    weights,
                    known_f32,
                    zero_beta,
                    layer_dh,
                    eps,
                )?;
                let k = g.reshape_(k_normed, vec![batch as i64, seq as i64, kv_dim as i64]);
                // Gemma 4 applies per-head V RMS-norm; Gemma 3 matches llama.cpp (V unchanged).
                let v = if matches!(cfg.arch, GemmaArch::Gemma4) {
                    let v_4d = g.reshape_(
                        v,
                        vec![batch as i64, seq as i64, layer_kv as i64, layer_dh as i64],
                    );
                    let v_ones = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.self_attn.v_norm.ones"),
                        vec![1.0f32; layer_dh],
                        &[layer_dh],
                    );
                    let v_zeros = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.self_attn.v_norm.zeros"),
                        vec![0.0f32; layer_dh],
                        &[layer_dh],
                    );
                    let v_normed = g.rms_norm(v_4d, v_ones, v_zeros, eps);
                    g.reshape_(v_normed, vec![batch as i64, seq as i64, kv_dim as i64])
                } else {
                    v
                };
                (k, v)
            };
            (q, k, v)
        } else {
            (q, k, v)
        };

        let (layer_cos, layer_sin) = if cfg.is_full_attention_layer(layer) {
            match (global_cos_in, global_sin_in) {
                (Some(gc), Some(gs)) => (gc, gs),
                _ => (rope_cos, rope_sin),
            }
        } else {
            (rope_cos, rope_sin)
        };
        let q_rope = g.rope_n(q, layer_cos, layer_sin, layer_dh, layer_nrot);
        // E2B KV sharing: a shared layer reuses the last fresh same-type layer's
        // full (post-cache) K/V — no own RoPE/concat. Its `past_k_{L}`/cache slot
        // still exists (kept as a redundant copy so the cache stays uniform and
        // packed_session needs no per-layer special-casing) but is unused here.
        // Fresh layers RoPE + concat their own K and, when sharing is active,
        // store the result for later same-type shared layers.
        let (new_k, new_v) = if is_shared_kv {
            let sk = shared_k_dec[kv_type_idx].ok_or_else(|| {
                anyhow!("decode KV-shared layer {layer}: no source K for type {kv_type_idx}")
            })?;
            let sv = shared_v_dec[kv_type_idx]
                .ok_or_else(|| anyhow!("decode KV-shared layer {layer}: no source V"))?;
            (sk, sv)
        } else {
            let k_rope = g.rope_n(k, layer_cos, layer_sin, layer_dh, layer_nrot);
            let nk = g.concat_(vec![past_k, k_rope], 1);
            let nv = g.concat_(vec![past_v, v], 1);
            if cfg.num_kv_shared_layers > 0 {
                shared_k_dec[kv_type_idx] = Some(nk);
                shared_v_dec[kv_type_idx] = Some(nv);
            }
            (nk, nv)
        };
        new_kv_outputs.push((new_k, new_v));

        let k_rep = repeat_kv_packed(&mut g, new_k, layer_kv, layer_dh, group);
        let v_rep = repeat_kv_packed(&mut g, new_v, layer_kv, layer_dh, group);

        let (q_attn, layer_attn_scale) =
            if matches!(cfg.arch, GemmaArch::Gemma2 | GemmaArch::Gemma3) {
                if let Some(scale) = attn_score_scale {
                    let q_scale = synth_const(
                        &mut g,
                        &mut params,
                        &format!("{lp}.attn.q_score_scale"),
                        vec![scale],
                        &[1],
                    );
                    (g.mul(q_rope, q_scale), Some(1.0f32))
                } else {
                    (q_rope, attn_score_scale)
                }
            } else {
                (q_rope, attn_score_scale)
            };

        let attn = if let Some(mask) = mask_id {
            let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_attn));
            g.attention_opts(
                q_attn,
                k_rep,
                v_rep,
                mask,
                nh,
                layer_dh,
                attn_shape,
                layer_attn_scale,
                attn_softcap,
            )
        } else {
            let (mask_kind, _, _) = cfg.layer_attn_options(layer);
            let attn_shape = rlx_ir::shape::attention_shape(g.shape(q_attn));
            g.attention_kind_opts(
                q_attn,
                k_rep,
                v_rep,
                nh,
                layer_dh,
                mask_kind,
                attn_shape,
                layer_attn_scale,
                attn_softcap,
            )
        };

        let (o_w, o_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        // Gemma 3/4 sandwich-norm: post_attention_layernorm applied to
        // attn_out BEFORE the residual add. Without this the attention
        // contribution grows unbounded across 48 layers — task #50 NaN
        // root cause. Maps GGUF blk.N.post_attention_norm.weight via
        // gguf_to_hf_name_for_arch.
        // Post-attention norm: Gemma 2/3/4 all have it (only Gemma 1 lacks it),
        // matching the post_feedforward_layernorm gate below. Dropping it for
        // Gemma 2 corrupted the whole forward (hidden cos ~0.35 vs llama.cpp).
        let attn_out = if cfg.arch != GemmaArch::Gemma {
            gemma_rms(
                &mut g,
                &mut params,
                attn_out,
                &format!("{lp}.post_attention_layernorm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?
        } else {
            attn_out
        };
        let post_attn = g.add(h_id, attn_out);

        let pre_ffn_key = if cfg.arch == GemmaArch::Gemma {
            format!("{lp}.post_attention_layernorm.weight")
        } else {
            format!("{lp}.pre_feedforward_layernorm.weight")
        };
        let normed_post = gemma_rms(
            &mut g,
            &mut params,
            post_attn,
            &pre_ffn_key,
            weights,
            known_f32,
            zero_beta,
            h,
            eps,
        )?;

        // Try fused {gate_up}_proj first — set up at drain time by
        // `drain_gemma_packed_weights`. One matmul + narrow halves
        // replaces two matmuls on the same input.
        let fused_gate_up_key = format!("{lp}.mlp.gate_up.weight");
        let fused_gate_up = known_packed
            .map(|m| m.contains_key(&fused_gate_up_key))
            .unwrap_or(false);
        let (gate, up) = if fused_gate_up {
            let (gu_w, gu_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &fused_gate_up_key,
            )?;
            let combined = emit_proj(
                &mut g,
                normed_post,
                gu_w,
                gu_s,
                Shape::new(&[batch, seq, int_dim * 2], f),
            );
            let gate = g.narrow_(combined, 2, 0, int_dim);
            let up = g.narrow_(combined, 2, int_dim, int_dim);
            (gate, up)
        } else {
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.gate_proj.weight"),
            )?;
            let (up_w, up_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.mlp.up_proj.weight"),
            )?;
            let gate = emit_proj(
                &mut g,
                normed_post,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            let up = emit_proj(
                &mut g,
                normed_post,
                up_w,
                up_s,
                Shape::new(&[batch, seq, int_dim], f),
            );
            (gate, up)
        };
        let (down_w, down_s) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            known_packed,
            known_f32,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        // gemma-3n activation sparsity (decode path — same as prefill).
        let gate_act = if let Some(std_mul) = cfg.layer_gaussian_std_multiplier(layer) {
            gelu_topk_gate(
                &mut g,
                &mut params,
                gate,
                std_mul,
                &format!("{lp}.mlp.act_sparsity"),
            )
        } else {
            g.gelu_approx(gate)
        };
        let mlp_inner = g.mul(gate_act, up);
        let mut ffn_out = emit_proj(
            &mut g,
            mlp_inner,
            down_w,
            down_s,
            Shape::new(&[batch, seq, h], f),
        );

        if cfg.arch != GemmaArch::Gemma {
            let post_ffn_key = format!("{lp}.post_feedforward_layernorm.weight");
            ffn_out = gemma_rms(
                &mut g,
                &mut params,
                ffn_out,
                &post_ffn_key,
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
        }

        let mut layer_out = g.add(post_attn, ffn_out);

        // Gemma 4 E2B Per-Layer Embeddings: inject the current token's per-layer
        // slice after the MLP residual (mirrors prefill / HF decoder forward).
        if let Some(ple_all) = per_layer_inputs_dec {
            let pw = cfg.ple_width();
            let ple_slice = g.narrow_(ple_all, 2, layer * pw, pw);
            let (gate_w, gate_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_input_gate.weight"),
            )?;
            let gated = emit_proj(
                &mut g,
                layer_out,
                gate_w,
                gate_s,
                Shape::new(&[batch, seq, pw], f),
            );
            let gated = g.gelu_approx(gated);
            let gated = g.mul(gated, ple_slice);
            let (proj_w, proj_s) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                known_packed,
                known_f32,
                &format!("{lp}.per_layer_projection.weight"),
            )?;
            let projected = emit_proj(
                &mut g,
                gated,
                proj_w,
                proj_s,
                Shape::new(&[batch, seq, h], f),
            );
            let projected = gemma_rms(
                &mut g,
                &mut params,
                projected,
                &format!("{lp}.post_per_layer_input_norm.weight"),
                weights,
                known_f32,
                zero_beta,
                h,
                eps,
            )?;
            layer_out = g.add(layer_out, projected);
        }

        // Per-layer output scalar that multiplies the combined layer output
        // before it becomes the next layer's residual (without it the stream
        // grows unbounded → inf/NaN at the LM head). Gemma 4 E2B ships this as
        // `layer_scalar`; the flagship GGUF path uses `self_attn.output_scale`.
        h_id = if cfg.has_ple() {
            let ls = load_p_cached(
                &mut g,
                &mut params,
                weights,
                known_f32,
                &format!("{lp}.layer_scalar"),
                &[1],
                false,
            )?;
            g.mul(layer_out, ls)
        } else if matches!(cfg.arch, GemmaArch::Gemma4) {
            // See prefill: a [1] output_scale broadcast mis-reads on CUDA; bake
            // the scalar into a [hidden] vector and use the last-dim broadcast.
            let okey = format!("{lp}.self_attn.output_scale.weight");
            let sval = match known_f32.and_then(|m| m.get(&okey)) {
                Some(v) => v.first().copied().unwrap_or(1.0),
                None => weights.take(&okey)?.0.first().copied().unwrap_or(1.0),
            };
            let scale_w = synth_const(
                &mut g,
                &mut params,
                &format!("{okey}.bh"),
                vec![sval; cfg.hidden_size],
                &[cfg.hidden_size],
            );
            g.mul(layer_out, scale_w)
        } else {
            layer_out
        };
    }

    let hidden = gemma_rms(
        &mut g,
        &mut params,
        h_id,
        "model.norm.weight",
        weights,
        known_f32,
        zero_beta,
        h,
        eps,
    )?;

    const TIED_LM_HEAD: &str = "gemma.packed.decode.lm_head.tied_t";
    let with_lm_head = !matches!(lm_output, PackedDecodeLmOutput::HiddenOnly);
    let mut outputs = if with_lm_head {
        // Task #36: prefer packed DequantMatMul on the original Q4K-packed embed
        // bytes — skips the ~4 GB transposed-f32 constant per decode bucket
        // (≈ 4 GB × num_decode_buckets if recompiled per bucket).
        let packed_embed_scheme = known_packed
            .and_then(|m| m.get("model.embed_tokens.weight"))
            .map(|(_, scheme, _)| *scheme);
        let mut logits =
            if let Some(scheme) = packed_embed_scheme.filter(|_| cfg.tie_word_embeddings) {
                let (w_id, _) = load_proj(
                    &mut g,
                    &mut params,
                    packed,
                    weights,
                    known_packed,
                    known_f32,
                    "model.embed_tokens.weight",
                )?;
                let logits_shape = Shape::new(&[batch, seq, vocab], f);
                g.add_node(
                    Op::DequantMatMul { scheme },
                    vec![hidden, w_id],
                    logits_shape,
                )
            } else {
                let lm_head_w = if cfg.tie_word_embeddings {
                    if let Some(tied) = known_f32.and_then(|m| m.get(TIED_LM_HEAD)) {
                        synth_const(&mut g, &mut params, TIED_LM_HEAD, tied.clone(), &[h, vocab])
                    } else {
                        let embed = params
                            .get("model.embed_tokens.weight")
                            .ok_or_else(|| {
                                anyhow!("missing model.embed_tokens.weight for tied lm_head")
                            })?
                            .clone();
                        synth_const(
                            &mut g,
                            &mut params,
                            TIED_LM_HEAD,
                            precompute_packed_decode_tied_lm_head(cfg, &embed)?,
                            &[h, vocab],
                        )
                    }
                } else {
                    load_p_cached(
                        &mut g,
                        &mut params,
                        weights,
                        known_f32,
                        "lm_head.weight",
                        &[vocab, h],
                        true,
                    )?
                };
                g.mm(hidden, lm_head_w)
            };
        if let Some(cap) = cfg.final_logit_softcapping {
            let inv = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.decode.softcap.inv.{cap}"),
                vec![1.0 / cap],
                &[1],
            );
            let cap_id = synth_const(
                &mut g,
                &mut params,
                &format!("gemma.packed.decode.softcap.cap.{cap}"),
                vec![cap],
                &[1],
            );
            let scaled = g.mul(logits, inv);
            let scaled_shape = g.shape(scaled).clone();
            let t = g.add_node(Op::Activation(Activation::Tanh), vec![scaled], scaled_shape);
            logits = g.mul(t, cap_id);
        }
        if lm_output == PackedDecodeLmOutput::GreedyToken {
            let logits_shape = g.shape(logits).clone();
            let vocab_axis = logits_shape.rank().saturating_sub(1);
            let argmax = g.add_node(
                Op::ArgMax {
                    axis: vocab_axis,
                    keep_dim: false,
                },
                vec![logits],
                Shape::new(&[batch, seq], f),
            );
            vec![argmax]
        } else {
            vec![logits]
        }
    } else {
        vec![hidden]
    };
    for (k, v) in new_kv_outputs {
        outputs.push(k);
        outputs.push(v);
    }
    g.set_outputs(outputs);
    Ok((g, params))
}

/// gemma-3n text-decoder prefill graph — the 4-stream **AltUp** + **Laurel**
/// wrapper around the Gemma-3 layer body.
///
/// Reproduces `mlx_lm/models/gemma3n.py` (`LanguageModel.__call__` +
/// `Gemma3nDecoderLayer.__call__` + `Gemma3nAltUp` + `Gemma3nLaurelBlock`)
/// operation-for-operation for **prefill** (batch=1, cache=None → every layer
/// computes its own K/V; the `num_kv_shared_layers` reuse is a decode-time
/// optimization and is a no-op for a fresh prefill). Loads F32 (via the
/// mlx-affine dequant loader). Inputs: `input_ids` `[b,s]` (f32 ids) and
/// `per_layer_inputs` `[b,s,num_layers*ple_w]` (host-precomputed PLE). Output:
/// last-axis-softcapped logits `[b,s,vocab]` (tied lm_head).
fn build_gemma3n_prefill_graph(
    cfg: &GemmaConfig,
    weights: &mut dyn rlx_core::weight_loader::WeightLoader,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use crate::rope::{build_rope_tables, default_inv_freq};
    use rlx_core::weight_loader::WeightLoader;
    use rlx_ir::op::{Activation, MaskKind, Op};
    use rlx_ir::{DType, NodeId, Shape};

    if batch != 1 {
        return Err(anyhow!("gemma-3n AltUp prefill requires batch=1"));
    }

    let mut g = Graph::new("gemma3n_altup");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let dh = cfg.head_dim();
    let q_dim = nh * dh;
    let kv_dim = n_kv * dh;
    let group = nh / n_kv;
    let eps = cfg.rms_norm_eps as f32;
    let num_layers = cfg.active_num_layers();
    let n_alt = cfg.altup_num_inputs;
    let active = cfg.altup_active_idx;
    let ple_w = cfg.ple_width();
    let clip = cfg.altup_coef_clip;
    let vocab = cfg.vocab_size;
    let window = cfg.sliding_window.unwrap_or(512);

    // ── Helpers ────────────────────────────────────────────────────
    fn synth(
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
    /// `gamma = 1 + loaded_weight` (loader returns `weight-1` for norm keys, so
    /// `1 + (w-1) = w` = the stored mlx `nn.RMSNorm` gain).
    fn norm_gamma(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        key: &str,
        dim: usize,
    ) -> Result<NodeId> {
        let (w, _shape) = weights.take(key)?;
        let wnode = g.param(key, Shape::new(&[dim], DType::F32));
        params.insert(key.to_string(), w);
        let ones = synth(g, params, &format!("{key}.ones"), vec![1.0f32; dim], &[dim]);
        Ok(g.add(ones, wnode))
    }
    fn gemma_rms(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        x: NodeId,
        key: &str,
        dim: usize,
        eps: f32,
    ) -> Result<NodeId> {
        let gamma = norm_gamma(g, params, weights, key, dim)?;
        let beta = synth(g, params, &format!("{key}.beta"), vec![0.0f32; dim], &[dim]);
        Ok(g.rms_norm(x, gamma, beta, eps))
    }
    /// Standard `x @ W.T` linear from a `[out,in]` (possibly quantized) weight.
    /// mlx-affine weights come back dequantized to F32 via `take_transposed`
    /// (→ `[in,out]`), so the projection is a plain `mm`.
    fn linear(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        x: NodeId,
        key: &str,
    ) -> Result<NodeId> {
        let (data, shape) = weights.take_transposed(key)?; // [in, out]
        let w = g.param(key, Shape::new(&shape, DType::F32));
        params.insert(key.to_string(), data);
        Ok(g.mm(x, w))
    }
    /// AltUp coefficient linear: the tiny `prediction_coefs`/`correction_coefs`
    /// weights are stored UNquantized as `[out,in]` f32. mlx clips them to
    /// `±coef_clip` then applies `m @ W.T`. We clip + transpose host-side.
    #[allow(clippy::too_many_arguments)]
    fn coef_lin(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        weights: &mut dyn WeightLoader,
        m: NodeId,
        key: &str,
        out: usize,
        inn: usize,
        clip: Option<f32>,
    ) -> Result<NodeId> {
        let (mut data, shape) = weights.take(key)?; // [out, inn]
        anyhow::ensure!(
            shape == [out, inn],
            "{key}: coef weight shape {shape:?} != [{out},{inn}]"
        );
        if let Some(c) = clip {
            for v in &mut data {
                *v = v.clamp(-c, c);
            }
        }
        let mut t = vec![0f32; data.len()];
        for r in 0..out {
            for cc in 0..inn {
                t[cc * out + r] = data[r * inn + cc];
            }
        }
        let w = g.param(key, Shape::new(&[inn, out], DType::F32));
        params.insert(key.to_string(), t);
        Ok(g.mm(m, w)) // [b,s,inn] @ [inn,out] → [b,s,out]
    }

    // ── Graph inputs ───────────────────────────────────────────────
    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], f));
    let per_layer_inputs = g.input(
        "per_layer_inputs",
        Shape::new(&[batch, seq, num_layers * ple_w], f),
    );

    // ── Shared small constants ─────────────────────────────────────
    let inv_h = synth(&mut g, &mut params, "g3n.inv_h", vec![1.0 / h as f32], &[1]);
    let inv_sqrt2 = synth(
        &mut g,
        &mut params,
        "g3n.inv_sqrt2",
        vec![0.5f32.sqrt()],
        &[1],
    );
    let one_scalar = synth(&mut g, &mut params, "g3n.one", vec![1.0f32], &[1]);
    let inv_n = synth(
        &mut g,
        &mut params,
        "g3n.inv_n",
        vec![1.0 / n_alt as f32],
        &[1],
    );

    // ── Embedding + AltUp entry ────────────────────────────────────
    // embed_tokens (quantized → dequant [vocab,h]); reused for tied lm_head.
    let (embed_data, embed_shape) = weights.take("model.embed_tokens.weight")?;
    let embed_w = g.param("model.embed_tokens.weight", Shape::new(&embed_shape, f));
    params.insert("model.embed_tokens.weight".into(), embed_data);
    let embed_scale = synth(
        &mut g,
        &mut params,
        "g3n.embed_scale",
        vec![(h as f32).sqrt()],
        &[1],
    );
    let gathered = g.gather_(embed_w, input_ids, 0); // [b,s,h]
    let h0 = g.mul(gathered, embed_scale); // active stream (idx 0)

    // target_magnitude = sqrt(mean(h0^2)) over hidden.
    let target_mag = {
        let sq = g.mul(h0, h0);
        let m = g.mean(sq, vec![2], true);
        g.sqrt(m)
    };
    let mut streams: Vec<NodeId> = Vec::with_capacity(n_alt);
    streams.push(h0);
    for j in 0..(n_alt - 1) {
        let proj = linear(
            &mut g,
            &mut params,
            weights,
            h0,
            &format!("model.altup_projections.{j}.weight"),
        )?;
        // magnitude-normalize: proj * (target_mag / mag)  (mlx maximum floor is
        // finfo.min ≈ -3.4e38 → a no-op; divide by mag directly).
        let sq = g.mul(proj, proj);
        let m = g.mean(sq, vec![2], true);
        let mag = g.sqrt(m);
        let ratio = g.div(target_mag, mag);
        let normed = g.mul(proj, ratio);
        streams.push(normed);
    }

    // ── RoPE tables (dual-θ: local 1e4 for sliding, global 1e6 for full) ──
    let local_inv = default_inv_freq(cfg.rope_local_base_freq, dh);
    let global_inv = default_inv_freq(cfg.rope_theta, dh);
    let half = local_inv.len();
    let (lc, ls) = build_rope_tables(&local_inv, seq);
    let (gc, gs) = build_rope_tables(&global_inv, seq);
    let local_cos = synth(&mut g, &mut params, "g3n.rope.local.cos", lc, &[seq, half]);
    let local_sin = synth(&mut g, &mut params, "g3n.rope.local.sin", ls, &[seq, half]);
    let global_cos = synth(&mut g, &mut params, "g3n.rope.global.cos", gc, &[seq, half]);
    let global_sin = synth(&mut g, &mut params, "g3n.rope.global.sin", gs, &[seq, half]);

    // ── Decoder layers ─────────────────────────────────────────────
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        let is_full = cfg.is_full_attention_layer(layer);
        let (cos, sin) = if is_full {
            (global_cos, global_sin)
        } else {
            (local_cos, local_sin)
        };
        let int_dim = cfg.layer_intermediate_size(layer);

        // Shared router (norm + modality_router) used by predict AND correct.
        let router_gamma = norm_gamma(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.altup.router_norm.weight"),
            h,
        )?;
        let router_beta = synth(
            &mut g,
            &mut params,
            &format!("{lp}.altup.router_norm.beta"),
            vec![0.0f32; h],
            &[h],
        );
        let (rw_data, rw_shape) =
            weights.take_transposed(&format!("{lp}.altup.modality_router.weight"))?;
        let router_w = g.param(
            format!("{lp}.altup.modality_router.weight"),
            Shape::new(&rw_shape, f),
        );
        params.insert(format!("{lp}.altup.modality_router.weight"), rw_data);
        // modalities(x) = tanh( modality_router( router_norm(x) * (1/h) ) )
        macro_rules! modalities {
            ($x:expr) => {{
                let normed = g.rms_norm($x, router_gamma, router_beta, eps);
                let scaled = g.mul(normed, inv_h);
                let routed = g.mm(scaled, router_w);
                g.tanh(routed)
            }};
        }

        // ── AltUp predict ──────────────────────────────────────────
        let m_pred = modalities!(streams[active]);
        let pred_lin = coef_lin(
            &mut g,
            &mut params,
            weights,
            m_pred,
            &format!("{lp}.altup.prediction_coefs.weight"),
            n_alt * n_alt,
            n_alt,
            clip,
        )?; // [b,s,n_alt^2]
        // predicted[c] = stream[c] + Σ_k stream[k] * pred_lin[..., c*n + k]
        let mut predictions: Vec<NodeId> = Vec::with_capacity(n_alt);
        for c in 0..n_alt {
            let mut acc = streams[c];
            for k in 0..n_alt {
                let coef = g.narrow_(pred_lin, 2, c * n_alt + k, 1); // [b,s,1]
                let term = g.mul(streams[k], coef);
                acc = g.add(acc, term);
            }
            predictions.push(acc);
        }
        let active_prediction = predictions[active];

        // ── Layer body on the active predicted stream ──────────────
        let ap_normed = gemma_rms(
            &mut g,
            &mut params,
            weights,
            active_prediction,
            &format!("{lp}.input_layernorm.weight"),
            h,
            eps,
        )?;

        // Laurel: x + post_laurel_norm(linear_right(linear_left(x))).
        let laurel_out = {
            let ll = linear(
                &mut g,
                &mut params,
                weights,
                ap_normed,
                &format!("{lp}.laurel.linear_left.weight"),
            )?;
            let lr = linear(
                &mut g,
                &mut params,
                weights,
                ll,
                &format!("{lp}.laurel.linear_right.weight"),
            )?;
            let ln = gemma_rms(
                &mut g,
                &mut params,
                weights,
                lr,
                &format!("{lp}.laurel.post_laurel_norm.weight"),
                h,
                eps,
            )?;
            g.add(ap_normed, ln)
        };

        // Attention (scale = 1.0, per-head q/k norm, no-scale v norm).
        let q = linear(
            &mut g,
            &mut params,
            weights,
            ap_normed,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let k = linear(
            &mut g,
            &mut params,
            weights,
            ap_normed,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let v = linear(
            &mut g,
            &mut params,
            weights,
            ap_normed,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let q4 = g.reshape_(q, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
        let qn = gemma_rms(
            &mut g,
            &mut params,
            weights,
            q4,
            &format!("{lp}.self_attn.q_norm.weight"),
            dh,
            eps,
        )?;
        let q = g.reshape_(qn, vec![batch as i64, seq as i64, q_dim as i64]);
        let k4 = g.reshape_(k, vec![batch as i64, seq as i64, n_kv as i64, dh as i64]);
        let kn = gemma_rms(
            &mut g,
            &mut params,
            weights,
            k4,
            &format!("{lp}.self_attn.k_norm.weight"),
            dh,
            eps,
        )?;
        let k = g.reshape_(kn, vec![batch as i64, seq as i64, kv_dim as i64]);
        // v_norm = RMSNoScale (gamma=1, beta=0).
        let v4 = g.reshape_(v, vec![batch as i64, seq as i64, n_kv as i64, dh as i64]);
        let v_ones = synth(
            &mut g,
            &mut params,
            &format!("{lp}.self_attn.v_norm.ones"),
            vec![1.0f32; dh],
            &[dh],
        );
        let v_zeros = synth(
            &mut g,
            &mut params,
            &format!("{lp}.self_attn.v_norm.zeros"),
            vec![0.0f32; dh],
            &[dh],
        );
        let vn = g.rms_norm(v4, v_ones, v_zeros, eps);
        let v = g.reshape_(vn, vec![batch as i64, seq as i64, kv_dim as i64]);

        let q = g.rope_n(q, cos, sin, dh, dh);
        let k = g.rope_n(k, cos, sin, dh, dh);
        let k_rep = repeat_kv_packed(&mut g, k, n_kv, dh, group);
        let v_rep = repeat_kv_packed(&mut g, v, n_kv, dh, group);
        let mask_kind = if is_full {
            MaskKind::Causal
        } else {
            MaskKind::SlidingWindow(window)
        };
        let attn_shape = rlx_ir::shape::attention_shape(g.shape(q));
        let attn = g.attention_kind_opts(
            q,
            k_rep,
            v_rep,
            nh,
            dh,
            mask_kind,
            attn_shape,
            Some(1.0f32),
            None,
        );
        let attn_out = linear(
            &mut g,
            &mut params,
            weights,
            attn,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = gemma_rms(
            &mut g,
            &mut params,
            weights,
            attn_out,
            &format!("{lp}.post_attention_layernorm.weight"),
            h,
            eps,
        )?;

        // attn_gated = active_prediction + attn;  attn_laurel = (gated+laurel)*2^-½
        let attn_gated = g.add(active_prediction, attn_out);
        let sum = g.add(attn_gated, laurel_out);
        let attn_laurel = g.mul(sum, inv_sqrt2);

        // FFN (GeGLU, gemma-3n gelu_topk gate on sparse layers).
        let attn_norm = gemma_rms(
            &mut g,
            &mut params,
            weights,
            attn_laurel,
            &format!("{lp}.pre_feedforward_layernorm.weight"),
            h,
            eps,
        )?;
        let gate = linear(
            &mut g,
            &mut params,
            weights,
            attn_norm,
            &format!("{lp}.mlp.gate_proj.weight"),
        )?;
        let up = linear(
            &mut g,
            &mut params,
            weights,
            attn_norm,
            &format!("{lp}.mlp.up_proj.weight"),
        )?;
        let gate_act = if let Some(std_mul) = cfg.layer_gaussian_std_multiplier(layer) {
            gelu_topk_gate(
                &mut g,
                &mut params,
                gate,
                std_mul,
                &format!("{lp}.mlp.act_sparsity"),
            )
        } else {
            g.gelu_approx(gate)
        };
        let _ = int_dim; // int_dim documented; shapes are inferred by mm.
        let inner = g.mul(gate_act, up);
        let down = linear(
            &mut g,
            &mut params,
            weights,
            inner,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let ffn = gemma_rms(
            &mut g,
            &mut params,
            weights,
            down,
            &format!("{lp}.post_feedforward_layernorm.weight"),
            h,
            eps,
        )?;
        let activated = g.add(attn_laurel, ffn);

        // ── AltUp correct ──────────────────────────────────────────
        let m_corr = modalities!(activated);
        let corr_lin = coef_lin(
            &mut g,
            &mut params,
            weights,
            m_corr,
            &format!("{lp}.altup.correction_coefs.weight"),
            n_alt,
            n_alt,
            clip,
        )?; // [b,s,n_alt]
        let innovation = g.sub(activated, predictions[active]);
        let mut corrected: Vec<NodeId> = Vec::with_capacity(n_alt);
        for c in 0..n_alt {
            let coef = g.narrow_(corr_lin, 2, c, 1); // [b,s,1]
            let coef1 = g.add(coef, one_scalar); // correction_coefs + 1
            let term = g.mul(innovation, coef1);
            corrected.push(g.add(predictions[c], term));
        }

        // ── Per-layer input injection (added to streams 1..n) ──────
        let first_prediction = corrected[active];
        let fp = if cfg.altup_correct_scale {
            let (cs_data, cs_shape) = weights.take(&format!("{lp}.altup.correct_output_scale"))?;
            let cs = g.param(
                format!("{lp}.altup.correct_output_scale"),
                Shape::new(&cs_shape, f),
            );
            params.insert(format!("{lp}.altup.correct_output_scale"), cs_data);
            g.mul(first_prediction, cs)
        } else {
            first_prediction
        };
        let gated = linear(
            &mut g,
            &mut params,
            weights,
            fp,
            &format!("{lp}.per_layer_input_gate.weight"),
        )?;
        let gated = g.gelu_approx(gated);
        let ple_slice = g.narrow_(per_layer_inputs, 2, layer * ple_w, ple_w);
        let gated = g.mul(gated, ple_slice);
        let projected = linear(
            &mut g,
            &mut params,
            weights,
            gated,
            &format!("{lp}.per_layer_projection.weight"),
        )?;
        let projected = gemma_rms(
            &mut g,
            &mut params,
            weights,
            projected,
            &format!("{lp}.post_per_layer_input_norm.weight"),
            h,
            eps,
        )?;
        // corrected[1:] += first_prediction; corrected[0] unchanged.
        let mut next = Vec::with_capacity(n_alt);
        for (c, cur) in corrected.iter().copied().enumerate() {
            if c == 0 {
                next.push(cur);
            } else {
                next.push(g.add(cur, projected));
            }
        }
        streams = next;
    }

    // ── AltUp exit: unembed streams 1.., magnitude-match, mean ─────
    let s0 = streams[active];
    let exit_target = {
        let sq = g.mul(s0, s0);
        let m = g.mean(sq, vec![2], true);
        g.sqrt(m)
    };
    let mut exit_streams: Vec<NodeId> = Vec::with_capacity(n_alt);
    exit_streams.push(s0);
    for j in 0..(n_alt - 1) {
        let proj = linear(
            &mut g,
            &mut params,
            weights,
            streams[j + 1],
            &format!("model.altup_unembed_projections.{j}.weight"),
        )?;
        let sq = g.mul(proj, proj);
        let m = g.mean(sq, vec![2], true);
        let mag = g.sqrt(m);
        let ratio = g.div(exit_target, mag);
        let normed = g.mul(proj, ratio);
        exit_streams.push(normed);
    }
    let mut acc = exit_streams[0];
    for s in exit_streams.iter().skip(1).copied() {
        acc = g.add(acc, s);
    }
    let h_final = g.mul(acc, inv_n); // mean over streams

    let hidden = gemma_rms(
        &mut g,
        &mut params,
        weights,
        h_final,
        "model.norm.weight",
        h,
        eps,
    )?;

    // Tied lm_head: logits = hidden @ embed_tokensᵀ (reuse embed param, in-graph
    // transpose — avoids a second ~2 GB f32 copy).
    let embed_t = g.transpose_(embed_w, vec![1, 0]); // [h, vocab]
    let mut logits = g.mm(hidden, embed_t); // [b,s,vocab]
    let _ = vocab;
    if let Some(cap) = cfg.final_logit_softcapping {
        let inv = synth(
            &mut g,
            &mut params,
            "g3n.softcap.inv",
            vec![1.0 / cap],
            &[1],
        );
        let cap_id = synth(&mut g, &mut params, "g3n.softcap.cap", vec![cap], &[1]);
        let scaled = g.mul(logits, inv);
        let scaled_shape = g.shape(scaled).clone();
        let t = g.add_node(Op::Activation(Activation::Tanh), vec![scaled], scaled_shape);
        logits = g.mul(t, cap_id);
    }
    g.set_outputs(vec![logits]);
    Ok((g, params))
}

fn repeat_kv_packed(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<rlx_ir::NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn validate_cfg(cfg: &GemmaConfig) -> Result<()> {
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
    if cfg.attention_bias {
        return Err(anyhow!("attention_bias=true not yet wired for gemma"));
    }
    Ok(())
}
