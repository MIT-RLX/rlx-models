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

// RLX — TIDE predictive expert offload (parity with modeling_llada2_moe.py).

use rlx_core::device_memory_for_moe_offload;
use rlx_runtime::{Device, ExpertPoolConfig};

/// `(free_bytes, total_bytes)` for TIDE VRAM budget sizing.
pub fn device_memory_for_offload(device: Device) -> Option<(usize, usize)> {
    device_memory_for_moe_offload(device)
}

/// Arguments matching TIDE `LLaDA2MoeModelLM.enable_predictive_expert_offload`.
#[derive(Debug, Clone)]
pub struct PredictiveOffloadParams {
    pub max_gpu_experts_per_layer: usize,
    pub reserve_vram_gb: f64,
    pub collect_stats: bool,
    pub jump_steps: usize,
    /// `(cuda_free_bytes, cuda_total_bytes)` when known; else budget from host RAM.
    pub device_memory: Option<(usize, usize)>,
    pub memory_budget_bytes: Option<usize>,
    pub num_experts: usize,
    pub num_sparse_moe_layers: usize,
    pub expert_param_bytes: usize,
}

impl PredictiveOffloadParams {
    pub fn new(
        max_gpu_experts_per_layer: usize,
        num_experts: usize,
        num_sparse_moe_layers: usize,
        expert_param_bytes: usize,
    ) -> Self {
        Self {
            max_gpu_experts_per_layer,
            reserve_vram_gb: 1.5,
            collect_stats: false,
            jump_steps: 1,
            device_memory: None,
            memory_budget_bytes: None,
            num_experts,
            num_sparse_moe_layers,
            expert_param_bytes,
        }
    }
}

/// Return value of TIDE `enable_predictive_expert_offload` (JSON-serializable keys).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictiveOffloadInfo {
    pub enabled: bool,
    pub gpu_expert_budget_per_layer: usize,
    pub num_sparse_moe_layers: usize,
    pub expert_param_bytes: usize,
    pub cuda_free_bytes: Option<usize>,
    pub cuda_total_bytes: Option<usize>,
    pub reserve_bytes: usize,
    pub jump_steps: usize,
    pub collect_stats: bool,
}

/// TIDE VRAM budget: `reserve = max(reserve_gb, 10% total)`, `budget = min(cap, usable // (expert_bytes * layers))`.
pub fn gpu_expert_budget_from_device_memory(
    free_bytes: usize,
    total_bytes: usize,
    expert_param_bytes: usize,
    num_moe_layers: usize,
    num_experts: usize,
    max_gpu_experts_per_layer: usize,
    reserve_vram_gb: f64,
) -> (usize, usize) {
    let reserve_gb_bytes = (reserve_vram_gb * (1024f64).powi(3)).max(0.0) as usize;
    let reserve_fraction_bytes = (0.1 * total_bytes as f64) as usize;
    let reserve_bytes = reserve_gb_bytes.max(reserve_fraction_bytes);
    let usable_bytes = free_bytes.saturating_sub(reserve_bytes);
    let max_budget = max_gpu_experts_per_layer.min(num_experts);
    let computed = if expert_param_bytes > 0 && num_moe_layers > 0 {
        usable_bytes / (expert_param_bytes.saturating_mul(num_moe_layers))
    } else {
        max_budget
    };
    (computed.min(max_budget), reserve_bytes)
}

/// Build [`PredictiveOffloadInfo`] + per-layer [`ExpertPoolConfig`] (one pool per MoE layer).
pub fn enable_predictive_expert_offload(
    params: &PredictiveOffloadParams,
) -> Option<(Vec<ExpertPoolConfig>, PredictiveOffloadInfo)> {
    let num_experts = params.num_experts;
    if params.num_sparse_moe_layers == 0 || num_experts == 0 {
        return None;
    }

    let (free_bytes, total_bytes) = if let Some(pair) = params.device_memory {
        pair
    } else if let Some(b) = params.memory_budget_bytes {
        (b, b)
    } else if let Some(total) = rlx_runtime::memory_estimate::available_unified_memory() {
        (total, total)
    } else {
        (usize::MAX / 2, usize::MAX)
    };

    let (gpu_budget, reserve_bytes) = gpu_expert_budget_from_device_memory(
        free_bytes,
        total_bytes,
        params.expert_param_bytes,
        params.num_sparse_moe_layers,
        num_experts,
        params.max_gpu_experts_per_layer,
        params.reserve_vram_gb,
    );

    if gpu_budget >= num_experts {
        return None;
    }

    let refresh = rlx_runtime::ExpertRefreshPolicy::EveryDenoiseSteps(params.jump_steps.max(1));
    let pools: Vec<_> = (0..params.num_sparse_moe_layers)
        .map(|_| ExpertPoolConfig::new(num_experts, gpu_budget, refresh))
        .collect();

    let info = PredictiveOffloadInfo {
        enabled: true,
        gpu_expert_budget_per_layer: gpu_budget,
        num_sparse_moe_layers: params.num_sparse_moe_layers,
        expert_param_bytes: params.expert_param_bytes,
        cuda_free_bytes: params.device_memory.map(|(f, _)| f).or(Some(free_bytes)),
        cuda_total_bytes: params.device_memory.map(|(_, t)| t).or(Some(total_bytes)),
        reserve_bytes,
        jump_steps: params.jump_steps.max(1),
        collect_stats: params.collect_stats,
    };

    Some((pools, info))
}
