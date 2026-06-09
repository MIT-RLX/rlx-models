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

// RLX — TIDE MoE offload for LLaDA2 (mirrors qwen35/moe_offload.rs).

use crate::config::LLaDA2MoeConfig;
use crate::tide::aggregate_offload_stats;
use crate::tide::{
    PredictiveOffloadParams, device_memory_for_offload, enable_predictive_expert_offload,
};
use crate::weights::{LLaDA2Weights, LayerFfn};
use rlx_runtime::Device;
use rlx_runtime::{
    ExpertPool, ExpertRefreshPolicy, MoeResidencyStats, merged_resident_mask,
    per_layer_resident_masks,
};

pub use crate::tide::MoeOffloadState;

pub fn expert_param_bytes_f32(cfg: &LLaDA2MoeConfig) -> usize {
    cfg.expert_param_bytes_f32()
}

pub fn count_moe_layers(weights: &LLaDA2Weights) -> usize {
    weights
        .layers
        .iter()
        .filter(|l| matches!(l.ffn, LayerFfn::Moe(_)))
        .count()
}

pub fn build_moe_offload(
    cfg: &LLaDA2MoeConfig,
    weights: &LLaDA2Weights,
    device: Device,
    max_gpu_experts_per_layer: Option<usize>,
    memory_budget_bytes: Option<usize>,
    jump_steps: Option<usize>,
    reserve_vram_gb: f64,
    collect_stats: bool,
) -> Option<MoeOffloadState> {
    if cfg.num_experts == 0 {
        return None;
    }
    let layer_count = count_moe_layers(weights).max(1);
    let num_experts = cfg.num_experts;
    let expert_bytes = expert_param_bytes_f32(cfg);
    let max_cap = max_gpu_experts_per_layer.unwrap_or(num_experts);
    if max_gpu_experts_per_layer.is_none() && memory_budget_bytes.is_none() {
        return None;
    }

    let mut params = PredictiveOffloadParams::new(max_cap, num_experts, layer_count, expert_bytes);
    params.reserve_vram_gb = reserve_vram_gb;
    params.jump_steps = jump_steps.unwrap_or(1);
    params.collect_stats = collect_stats;
    params.memory_budget_bytes = memory_budget_bytes;
    if params.device_memory.is_none() {
        params.device_memory = device_memory_for_offload(device);
    }

    let (pool_cfgs, info) = enable_predictive_expert_offload(&params)?;
    let refresh = ExpertRefreshPolicy::EveryDenoiseSteps(info.jump_steps);
    let pools = pool_cfgs.into_iter().map(ExpertPool::new).collect();

    Some(MoeOffloadState {
        pools,
        refresh,
        info,
        predictive_enabled: true,
        jump_steps: params.jump_steps,
        collect_stats,
    })
}

pub fn tide_stats(
    state: &MoeOffloadState,
    residency: Option<&MoeResidencyStats>,
) -> crate::tide::TideOffloadStats {
    aggregate_offload_stats(&state.pools, residency)
}

pub fn merged_mask(state: &MoeOffloadState) -> Vec<bool> {
    merged_resident_mask(&state.pools)
}

pub fn per_layer_masks(state: &MoeOffloadState) -> Vec<Vec<bool>> {
    per_layer_resident_masks(&state.pools)
}
