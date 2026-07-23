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

//! Pack HF per-expert MoE weights into IR expert stacks for `Op::GroupedMatMul`.
//!
//! HF stores each expert as a separate `nn.Linear` (`[out, in]`). The compiled
//! MoE path expects stacked `[n_routed, k, n]` tensors (Qwen35-style) with
//! each expert already transposed to matmul layout `[k, n]`.
//!
//! Shared experts stay a single SwiGLU whose intermediate width is
//! `moe_intermediate_size * n_shared_experts` (ungated add in the graph).
//!
//! Large matrices may be stored as F16/BF16 on the host ([`LmWeightPrecision`]);
//! [`PrecisionLoader`] widens to F32 when building graphs (F32/F16/BF16 IR).
//! For Q8_0/Q4_0, [`PackedLmWeights::ir_mat_blob`] keeps GGUF bytes for
//! `DequantMatMul` / `DequantGroupedMatMul` so compile params stay packed.

use crate::config::UnlimitedOcrConfig;
use crate::lm_precision::{
    LmWeightPrecision, ResolvedLmPrecision, bf16_bytes_to_f32, f16_bytes_to_f32, f32_to_bf16_bytes,
    f32_to_f16_bytes, f32_to_q4_0_bytes, f32_to_q8_0_bytes, precision_decision_message,
    q4_0_bytes_to_f32, q8_0_bytes_to_f32, resolve_lm_precision,
};
use crate::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_ir::QuantScheme;
use std::collections::HashMap;
use std::sync::Arc;

/// Packed GGUF bytes ready for `set_param_typed` + Dequant*MatMul.
#[derive(Clone, Debug)]
pub struct IrMatBlob {
    pub bytes: Vec<u8>,
    pub scheme: QuantScheme,
    /// Element shape after any transpose (not the U8 byte length).
    pub logical_shape: Vec<usize>,
}

/// Stacked expert gate/up weight key: shape `[n_routed, hidden, moe_ff]`.
pub fn expert_gate_exps_key(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.experts.gate_exps.weight")
}
/// Stacked expert up weight key: shape `[n_routed, hidden, moe_ff]`.
pub fn expert_up_exps_key(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.experts.up_exps.weight")
}
/// Stacked expert down weight key: shape `[n_routed, moe_ff, hidden]`.
pub fn expert_down_exps_key(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.experts.down_exps.weight")
}

/// Host-resident tensor: F32, half, or GGUF quant bytes (widened on load).
#[derive(Clone)]
pub enum HostTensor {
    F32 {
        data: Arc<Vec<f32>>,
        shape: Vec<usize>,
    },
    F16 {
        data: Arc<Vec<u8>>,
        shape: Vec<usize>,
    },
    Bf16 {
        data: Arc<Vec<u8>>,
        shape: Vec<usize>,
    },
    Q8_0 {
        data: Arc<Vec<u8>>,
        shape: Vec<usize>,
        nelems: usize,
    },
    Q4_0 {
        data: Arc<Vec<u8>>,
        shape: Vec<usize>,
        nelems: usize,
    },
}

impl HostTensor {
    fn from_f32(data: Vec<f32>, shape: Vec<usize>, store: ResolvedLmPrecision) -> Result<Self> {
        let nelems = data.len();
        Ok(match store {
            ResolvedLmPrecision::F32 => Self::F32 {
                data: Arc::new(data),
                shape,
            },
            ResolvedLmPrecision::F16 => Self::F16 {
                data: Arc::new(f32_to_f16_bytes(&data)),
                shape,
            },
            ResolvedLmPrecision::Bf16 => Self::Bf16 {
                data: Arc::new(f32_to_bf16_bytes(&data)),
                shape,
            },
            ResolvedLmPrecision::Q8_0 => Self::Q8_0 {
                data: Arc::new(f32_to_q8_0_bytes(&data)?),
                shape,
                nelems,
            },
            ResolvedLmPrecision::Q4_0 => Self::Q4_0 {
                data: Arc::new(f32_to_q4_0_bytes(&data)?),
                shape,
                nelems,
            },
        })
    }

    fn always_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::F32 {
            data: Arc::new(data),
            shape,
        }
    }

    fn to_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::F32 { data, shape } => Ok((data.as_ref().clone(), shape.clone())),
            Self::F16 { data, shape } => Ok((f16_bytes_to_f32(data)?, shape.clone())),
            Self::Bf16 { data, shape } => Ok((bf16_bytes_to_f32(data)?, shape.clone())),
            Self::Q8_0 {
                data,
                shape,
                nelems,
            } => Ok((q8_0_bytes_to_f32(data, *nelems)?, shape.clone())),
            Self::Q4_0 {
                data,
                shape,
                nelems,
            } => Ok((q4_0_bytes_to_f32(data, *nelems)?, shape.clone())),
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::F32 { data, .. } => data.len() * 4,
            Self::F16 { data, .. }
            | Self::Bf16 { data, .. }
            | Self::Q8_0 { data, .. }
            | Self::Q4_0 { data, .. } => data.len(),
        }
    }
}

/// Host-side LM weight cache used by compiled prefill/decode graph builds.
pub struct PackedLmWeights {
    cache: HashMap<String, HostTensor>,
    pub embed_tokens: Arc<Vec<f32>>,
    pub config: UnlimitedOcrConfig,
    pub requested_precision: LmWeightPrecision,
    pub resolved_precision: ResolvedLmPrecision,
}

impl PackedLmWeights {
    /// Load LM globals + all layers, packing experts at the resolved precision.
    pub fn from_store(store: &UnlimitedOcrWeightStore, cfg: &UnlimitedOcrConfig) -> Result<Self> {
        Self::from_store_with_precision(store, cfg, LmWeightPrecision::Auto)
    }

    pub fn from_store_with_precision(
        store: &UnlimitedOcrWeightStore,
        cfg: &UnlimitedOcrConfig,
        requested: LmWeightPrecision,
    ) -> Result<Self> {
        let resolved = resolve_lm_precision(requested, cfg);
        eprintln!(
            "[rlx-unlimited-ocr] {}",
            precision_decision_message(requested, resolved, cfg)
        );

        let mut tensors: HashMap<String, HostTensor> = HashMap::new();

        let mut globals = store.load_lm_globals()?;
        // Embeddings always F32 (host fuse / lookup).
        let (embed_data, embed_shape) = globals
            .take(UnlimitedOcrWeightPrefix::embed_tokens())
            .context("pack lm: embed_tokens")?;
        let embed = Arc::new(embed_data.clone());
        tensors.insert(
            UnlimitedOcrWeightPrefix::embed_tokens().into(),
            HostTensor::always_f32(embed_data, embed_shape),
        );
        // Final norm stays F32 (tiny, parity-sensitive).
        let (norm_data, norm_shape) = globals
            .take(UnlimitedOcrWeightPrefix::lm_norm())
            .context("pack lm: norm")?;
        tensors.insert(
            UnlimitedOcrWeightPrefix::lm_norm().into(),
            HostTensor::always_f32(norm_data, norm_shape),
        );
        // lm_head: Q4 softens to F16 (logit quality); Q8/F16/F32 as requested.
        let (head_data, head_shape) = globals
            .take(UnlimitedOcrWeightPrefix::lm_head())
            .context("pack lm: lm_head")?;
        tensors.insert(
            UnlimitedOcrWeightPrefix::lm_head().into(),
            HostTensor::from_f32(head_data, head_shape, soften_q4(resolved, SoftMat::LmHead))?,
        );

        for layer in 0..cfg.num_hidden_layers {
            let mut map = store.load_lm_layer(layer)?;
            if cfg.is_dense_layer(layer) {
                drain_layer_common(&mut map, &mut tensors, layer, resolved)?;
                let mlp_prec = soften_q4(resolved, SoftMat::DenseMlp);
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    take_into_prec(
                        &mut map,
                        &mut tensors,
                        UnlimitedOcrWeightPrefix::lm_dense_mlp(layer, proj),
                        mlp_prec,
                    )?;
                }
            } else {
                drain_layer_common(&mut map, &mut tensors, layer, resolved)?;
                // Router stays F32 (small, routing-sensitive).
                take_into_f32(
                    &mut map,
                    &mut tensors,
                    UnlimitedOcrWeightPrefix::lm_moe_gate(layer),
                )?;
                let shared_prec = soften_q4(resolved, SoftMat::SharedMlp);
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    take_into_prec(
                        &mut map,
                        &mut tensors,
                        UnlimitedOcrWeightPrefix::lm_moe_shared_expert(layer, proj),
                        shared_prec,
                    )?;
                }
                pack_routed_experts(store, cfg, layer, &mut map, &mut tensors, resolved)?;
            }
        }

        Ok(Self {
            cache: tensors,
            embed_tokens: embed,
            config: cfg.clone(),
            requested_precision: requested,
            resolved_precision: resolved,
        })
    }

    /// Non-consuming loader that widens half tensors to F32 for graph build.
    pub fn loader(&self) -> PrecisionLoader<'_> {
        PrecisionLoader { cache: &self.cache }
    }

    /// When true, LM graphs should emit Dequant* ops and attach U8 typed params.
    pub fn keeps_quants_in_ir(&self) -> bool {
        matches!(
            self.resolved_precision,
            ResolvedLmPrecision::Q8_0 | ResolvedLmPrecision::Q4_0
        )
    }

    /// Test/helper: build a pack from an expert-stacked [`WeightMap`].
    pub fn from_weight_map(
        map: &mut WeightMap,
        cfg: UnlimitedOcrConfig,
        resolved: ResolvedLmPrecision,
    ) -> Result<Self> {
        let mut cache: HashMap<String, HostTensor> = HashMap::new();
        let keys: Vec<String> = map.keys().map(|k| k.to_string()).collect();
        for key in keys {
            let (mut data, shape) = map.take(&key)?;
            let always_f32 = key.contains("layernorm")
                || key.contains("embed_tokens")
                || key.ends_with(".mlp.gate.weight")
                || key == UnlimitedOcrWeightPrefix::lm_norm();
            // `pack_experts_in_map` stores F32 `[E,K,N]`; GGUF DequantGrouped
            // needs BT `[n,k]` slabs — convert before quantizing.
            if matches!(
                resolved,
                ResolvedLmPrecision::Q8_0 | ResolvedLmPrecision::Q4_0
            ) && (key.contains("gate_exps")
                || key.contains("up_exps")
                || key.contains("down_exps"))
                && shape.len() == 3
            {
                let (n_e, k, n) = (shape[0], shape[1], shape[2]);
                data = kn_stack_to_bt(&data, n_e, k, n);
            }
            let tensor = if always_f32 {
                HostTensor::always_f32(data, shape)
            } else {
                let prec = if key.contains("lm_head") {
                    soften_q4(resolved, SoftMat::LmHead)
                } else if key.contains("self_attn") {
                    soften_q4(resolved, SoftMat::Attn)
                } else if key.contains(".mlp.shared_experts.") {
                    soften_q4(resolved, SoftMat::SharedMlp)
                } else if key.contains(".mlp.gate_proj")
                    || key.contains(".mlp.up_proj")
                    || key.contains(".mlp.down_proj")
                {
                    // Dense-layer MLP (not routed expert stacks).
                    soften_q4(resolved, SoftMat::DenseMlp)
                } else {
                    resolved
                };
                HostTensor::from_f32(data, shape, prec)?
            };
            cache.insert(key, tensor);
        }
        let embed = match cache.get(UnlimitedOcrWeightPrefix::embed_tokens()) {
            Some(HostTensor::F32 { data, .. }) => Arc::clone(data),
            _ => bail!("from_weight_map: missing F32 embed_tokens"),
        };
        Ok(Self {
            cache,
            embed_tokens: embed,
            config: cfg,
            requested_precision: match resolved {
                ResolvedLmPrecision::F32 => LmWeightPrecision::F32,
                ResolvedLmPrecision::F16 => LmWeightPrecision::F16,
                ResolvedLmPrecision::Bf16 => LmWeightPrecision::Bf16,
                ResolvedLmPrecision::Q8_0 => LmWeightPrecision::Q8_0,
                ResolvedLmPrecision::Q4_0 => LmWeightPrecision::Q4_0,
            },
            resolved_precision: resolved,
        })
    }

    /// Packed blob for IR when this key is Q8_0/Q4_0; `None` → use F32 `load_param`.
    ///
    /// GGUF `DequantMatMul` / `DequantGroupedMatMul` expect BT `[n, k]` (= HF
    /// `[out, in]`). Do **not** transpose — that is only for F32 `MatMul`.
    pub fn ir_mat_blob(&self, key: &str, _transpose: bool) -> Result<Option<IrMatBlob>> {
        let tensor = self
            .cache
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found in packed cache: {key}"))?;
        match tensor {
            HostTensor::Q8_0 {
                data,
                shape,
                nelems,
            } => Ok(Some(prepare_ir_blob(
                data,
                shape,
                *nelems,
                QuantScheme::GgufQ8_0,
            )?)),
            HostTensor::Q4_0 {
                data,
                shape,
                nelems,
            } => Ok(Some(prepare_ir_blob(
                data,
                shape,
                *nelems,
                QuantScheme::GgufQ4_0,
            )?)),
            HostTensor::F32 { .. } | HostTensor::F16 { .. } | HostTensor::Bf16 { .. } => Ok(None),
        }
    }

    pub fn host_nbytes(&self) -> usize {
        self.cache.values().map(HostTensor::nbytes).sum()
    }

    pub fn embed_tokens_lookup(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let table = self.embed_tokens.as_ref();
        let vocab = table.len() / hidden;
        let mut out = vec![0f32; ids.len() * hidden];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                (id as usize) < vocab,
                "embed_tokens: id {id} out of range (vocab={vocab})"
            );
            let src = &table[id as usize * hidden..(id as usize + 1) * hidden];
            out[i * hidden..(i + 1) * hidden].copy_from_slice(src);
        }
        Ok(out)
    }
}

/// [`WeightLoader`] over [`PackedLmWeights`] — clones/widens without consuming.
pub struct PrecisionLoader<'a> {
    cache: &'a HashMap<String, HostTensor>,
}

impl WeightLoader for PrecisionLoader<'_> {
    fn format_id(&self) -> &'static str {
        "unlimited-ocr-packed"
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.cache
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found in packed cache: {key}"))?
            .to_f32()
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut transposed = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((transposed, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
}

fn prepare_ir_blob(
    data: &Arc<Vec<u8>>,
    shape: &[usize],
    nelems: usize,
    scheme: QuantScheme,
) -> Result<IrMatBlob> {
    let _ = nelems;
    Ok(IrMatBlob {
        bytes: data.as_ref().clone(),
        scheme,
        logical_shape: shape.to_vec(),
    })
}

fn drain_layer_common(
    map: &mut WeightMap,
    out: &mut HashMap<String, HostTensor>,
    layer: usize,
    prec: ResolvedLmPrecision,
) -> Result<()> {
    take_into_f32(
        map,
        out,
        UnlimitedOcrWeightPrefix::lm_input_layernorm(layer),
    )?;
    take_into_f32(
        map,
        out,
        UnlimitedOcrWeightPrefix::lm_post_attention_layernorm(layer),
    )?;
    let attn_prec = soften_q4(prec, SoftMat::Attn);
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        take_into_prec(
            map,
            out,
            UnlimitedOcrWeightPrefix::lm_attn(layer, proj),
            attn_prec,
        )?;
    }
    Ok(())
}

/// Q4_0 softens quality-sensitive mats to F16; routed experts stay Q4.
#[derive(Clone, Copy)]
enum SoftMat {
    LmHead,
    Attn,
    DenseMlp,
    SharedMlp,
}

fn soften_q4(resolved: ResolvedLmPrecision, kind: SoftMat) -> ResolvedLmPrecision {
    match (resolved, kind) {
        (
            ResolvedLmPrecision::Q4_0,
            SoftMat::LmHead | SoftMat::Attn | SoftMat::DenseMlp | SoftMat::SharedMlp,
        ) => ResolvedLmPrecision::F16,
        _ => resolved,
    }
}

fn take_into_f32(
    map: &mut WeightMap,
    out: &mut HashMap<String, HostTensor>,
    key: String,
) -> Result<()> {
    let (data, shape) = map
        .take(&key)
        .with_context(|| format!("pack lm weight: {key}"))?;
    out.insert(key, HostTensor::always_f32(data, shape));
    Ok(())
}

fn take_into_prec(
    map: &mut WeightMap,
    out: &mut HashMap<String, HostTensor>,
    key: String,
    prec: ResolvedLmPrecision,
) -> Result<()> {
    let (data, shape) = map
        .take(&key)
        .with_context(|| format!("pack lm weight: {key}"))?;
    out.insert(key, HostTensor::from_f32(data, shape, prec)?);
    Ok(())
}

fn pack_routed_experts(
    store: &UnlimitedOcrWeightStore,
    cfg: &UnlimitedOcrConfig,
    layer: usize,
    map: &mut WeightMap,
    out: &mut HashMap<String, HostTensor>,
    prec: ResolvedLmPrecision,
) -> Result<()> {
    let n_e = cfg.n_routed_experts;
    let hidden = cfg.hidden_size;
    let ff = cfg.moe_intermediate_size;
    ensure!(
        store.count_experts(layer) == n_e,
        "layer {layer}: expert count {} != n_routed_experts {n_e}",
        store.count_experts(layer)
    );

    let mut gate_stack = vec![0f32; n_e * hidden * ff];
    let mut up_stack = vec![0f32; n_e * hidden * ff];
    let mut down_stack = vec![0f32; n_e * ff * hidden];

    for e in 0..n_e {
        let (gate_hf, gate_shape) = map
            .take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
                layer,
                e,
                "gate_proj",
            ))
            .with_context(|| format!("layer {layer} expert {e} gate"))?;
        let (up_hf, up_shape) = map
            .take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
                layer, e, "up_proj",
            ))
            .with_context(|| format!("layer {layer} expert {e} up"))?;
        let (down_hf, down_shape) = map
            .take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
                layer,
                e,
                "down_proj",
            ))
            .with_context(|| format!("layer {layer} expert {e} down"))?;

        ensure!(
            gate_shape == [ff, hidden],
            "layer {layer} expert {e} gate shape {gate_shape:?} != [{ff}, {hidden}]"
        );
        ensure!(
            up_shape == [ff, hidden],
            "layer {layer} expert {e} up shape {up_shape:?} != [{ff}, {hidden}]"
        );
        ensure!(
            down_shape == [hidden, ff],
            "layer {layer} expert {e} down shape {down_shape:?} != [{hidden}, {ff}]"
        );

        // F32 / half GroupedMatMul wants `[E, K, N]` (each expert `[k, n]`).
        // GGUF DequantGroupedMatMul wants BT slabs `[n, k]` (= HF `[out, in]`).
        let store_bt = matches!(prec, ResolvedLmPrecision::Q8_0 | ResolvedLmPrecision::Q4_0);
        let (gate, up, down) = if store_bt {
            (gate_hf, up_hf, down_hf)
        } else {
            (
                transpose_2d(&gate_hf, ff, hidden),
                transpose_2d(&up_hf, ff, hidden),
                transpose_2d(&down_hf, hidden, ff),
            )
        };

        let g_off = e * hidden * ff;
        let d_off = e * ff * hidden;
        gate_stack[g_off..g_off + gate.len()].copy_from_slice(&gate);
        up_stack[g_off..g_off + up.len()].copy_from_slice(&up);
        down_stack[d_off..d_off + down.len()].copy_from_slice(&down);
    }

    // Logical shape stays `[E, K, N]` for both layouts (bytes differ for Q BT).
    out.insert(
        expert_gate_exps_key(layer),
        HostTensor::from_f32(gate_stack, vec![n_e, hidden, ff], prec)?,
    );
    out.insert(
        expert_up_exps_key(layer),
        HostTensor::from_f32(up_stack, vec![n_e, hidden, ff], prec)?,
    );
    out.insert(
        expert_down_exps_key(layer),
        HostTensor::from_f32(down_stack, vec![n_e, ff, hidden], prec)?,
    );
    Ok(())
}

/// Row-major transpose `[rows, cols]` → `[cols, rows]`.
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// Convert stacked `[E, K, N]` (F32 GroupedMatMul) → BT slabs `[E]`×`[N, K]`.
fn kn_stack_to_bt(stack: &[f32], n_e: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_e * k * n];
    for e in 0..n_e {
        let src = &stack[e * k * n..(e + 1) * k * n];
        let bt = transpose_2d(src, k, n);
        out[e * k * n..(e + 1) * k * n].copy_from_slice(&bt);
    }
    out
}

/// Insert stacked expert tensors built from HF-layout per-expert matrices
/// already present in `map` (removes the per-expert keys). Always F32 (tests).
pub fn pack_experts_in_map(
    map: &mut WeightMap,
    layer: usize,
    n_routed: usize,
    hidden: usize,
    moe_ff: usize,
) -> Result<()> {
    let mut gate_stack = vec![0f32; n_routed * hidden * moe_ff];
    let mut up_stack = vec![0f32; n_routed * hidden * moe_ff];
    let mut down_stack = vec![0f32; n_routed * moe_ff * hidden];
    for e in 0..n_routed {
        let (gate_hf, _) = map.take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
            layer,
            e,
            "gate_proj",
        ))?;
        let (up_hf, _) = map.take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
            layer, e, "up_proj",
        ))?;
        let (down_hf, _) = map.take(&UnlimitedOcrWeightPrefix::lm_moe_expert(
            layer,
            e,
            "down_proj",
        ))?;
        let gate = transpose_2d(&gate_hf, moe_ff, hidden);
        let up = transpose_2d(&up_hf, moe_ff, hidden);
        let down = transpose_2d(&down_hf, hidden, moe_ff);
        let g_off = e * hidden * moe_ff;
        let d_off = e * moe_ff * hidden;
        gate_stack[g_off..g_off + gate.len()].copy_from_slice(&gate);
        up_stack[g_off..g_off + up.len()].copy_from_slice(&up);
        down_stack[d_off..d_off + down.len()].copy_from_slice(&down);
    }
    let mut kept: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let keys: Vec<String> = map.keys().map(|s| s.to_string()).collect();
    for k in keys {
        let (d, s) = map.take(&k)?;
        kept.insert(k, (d, s));
    }
    kept.insert(
        expert_gate_exps_key(layer),
        (gate_stack, vec![n_routed, hidden, moe_ff]),
    );
    kept.insert(
        expert_up_exps_key(layer),
        (up_stack, vec![n_routed, hidden, moe_ff]),
    );
    kept.insert(
        expert_down_exps_key(layer),
        (down_stack, vec![n_routed, moe_ff, hidden]),
    );
    *map = WeightMap::from_tensors(kept);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d_swaps_layout() {
        let m = vec![1., 2., 3., 4., 5., 6.];
        let t = transpose_2d(&m, 2, 3);
        assert_eq!(t, vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn stacked_key_names() {
        assert_eq!(
            expert_gate_exps_key(1),
            "model.layers.1.mlp.experts.gate_exps.weight"
        );
    }
}
