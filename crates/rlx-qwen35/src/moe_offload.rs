// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// TIDE-style MoE offload helpers for Qwen3.5 MoE runners.
// Reference implementation: `/Users/Shared/TIDE` (`model/modeling_llada2_moe.py`).

use crate::config::Qwen35Config;
use crate::weights::{Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use rlx_llada2::tide::{PredictiveOffloadParams, enable_predictive_expert_offload};
use rlx_runtime::{ExpertPool, ExpertRefreshPolicy};

pub use rlx_llada2::tide::MoeOffloadState;

/// Bytes for one routed expert's gate+up+down matrices at F32 (budget sizing).
pub fn expert_param_bytes_f32(cfg: &Qwen35Config) -> usize {
    let n = cfg.hidden_size;
    let ff = cfg.expert_ffn_dim();
    3 * n * ff * std::mem::size_of::<f32>()
}

fn trunk_layer_is_moe(layer: &Qwen35TrunkLayer) -> bool {
    match layer {
        Qwen35TrunkLayer::Linear(lin) => matches!(lin.ffn, Qwen35LayerFfn::Moe(_)),
        Qwen35TrunkLayer::FullAttn(fa) => matches!(fa.ffn, Qwen35LayerFfn::Moe(_)),
    }
}

pub fn count_moe_ffn_layers(weights: &Qwen35Weights) -> usize {
    weights
        .trunk_layers
        .iter()
        .filter(|l| trunk_layer_is_moe(l))
        .count()
}

pub fn num_moe_ffn_layers(cfg: &Qwen35Config) -> usize {
    if !cfg.is_moe() {
        return 0;
    }
    cfg.mtp_layer_start().unwrap_or(cfg.num_hidden_layers)
}

/// Build per-layer pools using TIDE `enable_predictive_expert_offload` sizing.
pub fn build_moe_offload(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    max_gpu_experts_per_layer: Option<usize>,
    memory_budget_bytes: Option<usize>,
    jump_steps: Option<usize>,
    reserve_vram_gb: f64,
    collect_stats: bool,
) -> Option<MoeOffloadState> {
    if !cfg.is_moe() {
        return None;
    }
    let num_experts = cfg.num_experts;
    let layer_count = count_moe_ffn_layers(weights).max(1);
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

pub fn build_expert_pool(
    cfg: &Qwen35Config,
    max_gpu_experts_per_layer: Option<usize>,
    memory_budget_bytes: Option<usize>,
    jump_steps: Option<usize>,
) -> Option<ExpertPool> {
    build_moe_offload(
        cfg,
        &crate::synth::moe_synth_weights(cfg),
        max_gpu_experts_per_layer,
        memory_budget_bytes,
        jump_steps,
        1.5,
        false,
    )
    .and_then(|s| s.pools.into_iter().next())
}

pub fn decode_should_refresh(state: &MoeOffloadState, decode_step: usize) -> bool {
    state.should_refresh_forward(decode_step, false)
}
