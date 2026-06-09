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

// RLX — `MoeExpertStore` for LLaDA2 MoE layers (TIDE host migration).

use crate::config::LLaDA2MoeConfig;
use crate::weights::{LLaDA2Weights, LayerFfn};
use anyhow::{Result, anyhow};
use rlx_cpu::moe_residency::{LayerHostBind, MoeHostBind};
use rlx_runtime::{ExpertStackF32, LayerMoeWeights, MoeExpertStore};

fn stack(
    data: Vec<f32>,
    num_experts: usize,
    k: usize,
    n: usize,
    name: &str,
) -> Result<ExpertStackF32> {
    if data.len() != num_experts * k * n {
        return Err(anyhow!(
            "{name}: len {} != {num_experts}*{k}*{n}",
            data.len()
        ));
    }
    Ok(ExpertStackF32::new(data, num_experts, k, n))
}

pub fn build_moe_expert_store(
    cfg: &LLaDA2MoeConfig,
    weights: &LLaDA2Weights,
) -> Result<MoeExpertStore> {
    if cfg.num_experts == 0 {
        return Err(anyhow!("not a MoE config"));
    }
    let h = cfg.hidden_size;
    let ff = cfg.expert_ffn_dim();
    let e = cfg.num_experts;
    let mut layers = Vec::new();
    for (il, layer) in weights.layers.iter().enumerate() {
        let LayerFfn::Moe(m) = &layer.ffn else {
            continue;
        };
        layers.push(LayerMoeWeights {
            layer_index: il,
            gate: stack(m.gate_exps.clone(), e, h, ff, "gate_exps")?,
            up: stack(m.up_exps.clone(), e, h, ff, "up_exps")?,
            down: stack(m.down_exps.clone(), e, ff, h, "down_exps")?,
        });
    }
    if layers.is_empty() {
        return Err(anyhow!("no MoE layers"));
    }
    Ok(MoeExpertStore { layers })
}

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

/// Push full expert stacks into compiled params (HF key names).
pub fn apply_moe_store_to_compiled(
    store: &MoeExpertStore,
    compiled: &mut rlx_runtime::CompiledGraph,
) {
    for layer in &store.layers {
        let il = layer.layer_index;
        compiled.set_param(
            &format!("model.layers.{il}.mlp.gate_exps.weight"),
            layer.gate.as_slice(),
        );
        compiled.set_param(
            &format!("model.layers.{il}.mlp.up_exps.weight"),
            layer.up.as_slice(),
        );
        compiled.set_param(
            &format!("model.layers.{il}.mlp.down_exps.weight"),
            layer.down.as_slice(),
        );
    }
}

pub fn moe_layer_indices(weights: &LLaDA2Weights) -> Vec<usize> {
    weights
        .layers
        .iter()
        .enumerate()
        .filter_map(|(il, l)| matches!(l.ffn, LayerFfn::Moe(_)).then_some(il))
        .collect()
}
