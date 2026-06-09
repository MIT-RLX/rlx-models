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

//! Build [`rlx_runtime::MoeExpertStore`] from Qwen3.5 MoE weights.

use crate::config::Qwen35Config;
use crate::weights::{MatWeight, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use anyhow::{Result, anyhow};
use rlx_cpu::moe_residency::{LayerHostBind, MoeHostBind};
use rlx_runtime::{ExpertStackF32, LayerMoeWeights, MoeExpertStore};

fn stack_from_mat(
    weight: &MatWeight,
    num_experts: usize,
    k: usize,
    n: usize,
    name: &str,
) -> Result<ExpertStackF32> {
    match weight {
        MatWeight::F32(data) => {
            if data.len() != num_experts * k * n {
                return Err(anyhow!(
                    "{name}: len {} != {num_experts}*{k}*{n}",
                    data.len()
                ));
            }
            Ok(ExpertStackF32::new(data.clone(), num_experts, k, n))
        }
        MatWeight::Packed { .. } => Err(anyhow!(
            "{name}: packed MoE expert store not implemented (use F32 weights)"
        )),
    }
}

fn layer_moe_weights(
    il: usize,
    cfg: &Qwen35Config,
    moe: &crate::weights::Qwen35MoeFfn,
) -> Result<LayerMoeWeights> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.expert_ffn_dim();
    let e = cfg.num_experts;
    Ok(LayerMoeWeights {
        layer_index: il,
        gate: stack_from_mat(&moe.gate_exps, e, n_embd, n_ff, "gate_exps")?,
        up: stack_from_mat(&moe.up_exps, e, n_embd, n_ff, "up_exps")?,
        down: stack_from_mat(&moe.down_exps, e, n_ff, n_embd, "down_exps")?,
    })
}

/// Collect MoE FFN layers from trunk weights (forward order).
pub fn build_moe_expert_store(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
) -> Result<MoeExpertStore> {
    if !cfg.is_moe() {
        return Err(anyhow!("not a MoE config"));
    }
    let mut layers = Vec::new();
    for (il, layer) in weights.trunk_layers.iter().enumerate() {
        let moe = match layer {
            Qwen35TrunkLayer::Linear(lin) => match &lin.ffn {
                Qwen35LayerFfn::Moe(m) => m,
                _ => continue,
            },
            Qwen35TrunkLayer::FullAttn(fa) => match &fa.ffn {
                Qwen35LayerFfn::Moe(m) => m,
                _ => continue,
            },
        };
        layers.push(layer_moe_weights(il, cfg, moe)?);
    }
    if layers.is_empty() {
        return Err(anyhow!("no MoE FFN layers in weights"));
    }
    Ok(MoeExpertStore { layers })
}

/// Host pointer table for CPU GroupedMatMul fallback (TIDE off-device path).
pub fn moe_host_bind_from_store(store: &MoeExpertStore) -> MoeHostBind {
    let layers = store
        .layers
        .iter()
        .map(|l| {
            let ptrs = |stack: &ExpertStackF32| -> Vec<*const f32> {
                (0..stack.num_experts)
                    .map(|e| stack.expert_slice(e).as_ptr())
                    .collect()
            };
            LayerHostBind {
                gate: ptrs(&l.gate),
                up: ptrs(&l.up),
                down: ptrs(&l.down),
                stride: l.gate.expert_stride(),
            }
        })
        .collect();
    MoeHostBind { layers }
}

/// Layer indices with MoE FFN (for param naming).
pub fn moe_layer_indices(weights: &Qwen35Weights) -> Vec<usize> {
    weights
        .trunk_layers
        .iter()
        .enumerate()
        .filter_map(|(il, layer)| {
            let is_moe = match layer {
                Qwen35TrunkLayer::Linear(l) => matches!(l.ffn, Qwen35LayerFfn::Moe(_)),
                Qwen35TrunkLayer::FullAttn(f) => matches!(f.ffn, Qwen35LayerFfn::Moe(_)),
            };
            is_moe.then_some(il)
        })
        .collect()
}
