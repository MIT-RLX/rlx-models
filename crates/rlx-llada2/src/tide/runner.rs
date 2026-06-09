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

// RLX — TIDE-facing runner API (`LLaDA2MoeModelLM` parity).

use crate::tide::{
    BlockDenoiseConfig, BlockDenoiseLoop, BlockDenoiseStepStats, PredictiveOffloadInfo,
    PredictiveOffloadParams, TideOffloadStats, enable_predictive_expert_offload,
};
use crate::{GenerateConfig, LLaDA2MoeConfig, LLaDA2Runner, LLaDA2RunnerBuilder, LLaDA2Weights};
use anyhow::Result;

/// TIDE reference model runner (LLaDA2 MoE + block diffusion + predictive offload).
pub struct TideRunner {
    inner: LLaDA2Runner,
}

impl TideRunner {
    pub fn builder() -> LLaDA2RunnerBuilder {
        LLaDA2Runner::builder()
    }

    pub fn from_llada2(inner: LLaDA2Runner) -> Self {
        Self { inner }
    }

    pub fn into_llada2(self) -> LLaDA2Runner {
        self.inner
    }

    pub fn llada2(&self) -> &LLaDA2Runner {
        &self.inner
    }

    pub fn llada2_mut(&mut self) -> &mut LLaDA2Runner {
        &mut self.inner
    }

    pub fn config(&self) -> &LLaDA2MoeConfig {
        self.inner.config()
    }

    /// TIDE `enable_predictive_expert_offload` (configure via [`Self::builder`] before `build`).
    pub fn predictive_offload_info(&self) -> Option<PredictiveOffloadInfo> {
        self.inner.predictive_offload_info()
    }

    pub fn predictive_offload_enabled(&self) -> bool {
        self.inner.predictive_offload_enabled()
    }

    pub fn jump_steps(&self) -> usize {
        self.inner.jump_steps()
    }

    /// TIDE `get_offload_stats()` — sum across MoE layers + last-forward residency.
    pub fn get_offload_stats(&mut self) -> TideOffloadStats {
        self.inner.get_offload_stats()
    }

    /// TIDE `generate(input_ids, ...)`.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<(Vec<u32>, Vec<BlockDenoiseStepStats>)> {
        self.inner.generate(gen_cfg, input_ids)
    }

    pub fn block_denoise_loop(
        &mut self,
        cfg: BlockDenoiseConfig,
    ) -> BlockDenoiseLoop<crate::runner::LLaDA2RunnerForward<'_>> {
        self.inner.block_denoise_loop(cfg)
    }
}

impl LLaDA2RunnerBuilder {
    /// TIDE `enable_predictive_expert_offload(max_gpu_experts_per_layer, ...)`.
    pub fn tide_enable_predictive_expert_offload(
        mut self,
        max_gpu_experts_per_layer: usize,
        reserve_vram_gb: f64,
        collect_stats: bool,
        jump_steps: usize,
    ) -> Self {
        self = self
            .enable_predictive_expert_offload(max_gpu_experts_per_layer)
            .reserve_vram_gb(reserve_vram_gb)
            .jump_steps(jump_steps)
            .moe_collect_stats(collect_stats);
        self
    }
}

/// Preview TIDE offload budget without building a runner (host/unified memory when no CUDA).
pub fn preview_predictive_offload(
    cfg: &LLaDA2MoeConfig,
    weights: &LLaDA2Weights,
    max_gpu_experts_per_layer: usize,
    reserve_vram_gb: f64,
    collect_stats: bool,
    jump_steps: usize,
) -> Option<PredictiveOffloadInfo> {
    let layer_count = crate::moe_offload::count_moe_layers(weights).max(1);
    let mut params = PredictiveOffloadParams::new(
        max_gpu_experts_per_layer,
        cfg.num_experts,
        layer_count,
        cfg.expert_param_bytes_f32(),
    );
    params.reserve_vram_gb = reserve_vram_gb;
    params.collect_stats = collect_stats;
    params.jump_steps = jump_steps;
    enable_predictive_expert_offload(&params).map(|(_, info)| info)
}
