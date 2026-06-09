// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Shared TIDE MoE offload state (Qwen3.5 + LLaDA2).

use rlx_runtime::{
    ExpertPool, ExpertRefreshPolicy, MoeExpertStore, MoeResidencyStats, merged_resident_mask,
    per_layer_resident_masks,
};

use super::{PredictiveOffloadInfo, TideOffloadStats, aggregate_offload_stats, refresh_experts};

/// Per-layer TIDE expert pools (one per MoE FFN in trunk order).
#[derive(Debug)]
pub struct MoeOffloadState {
    pub pools: Vec<ExpertPool>,
    pub refresh: ExpertRefreshPolicy,
    pub info: PredictiveOffloadInfo,
    pub predictive_enabled: bool,
    pub jump_steps: usize,
    pub collect_stats: bool,
}

impl MoeOffloadState {
    pub fn num_layers(&self) -> usize {
        self.pools.len()
    }

    pub fn merged_resident_mask(&self) -> Vec<bool> {
        merged_resident_mask(&self.pools)
    }

    pub fn per_layer_resident_masks(&self) -> Vec<Vec<bool>> {
        per_layer_resident_masks(&self.pools)
    }

    /// AR prefill or block-diffusion prefill block → always refresh; else `step % jump_steps == 0`.
    pub fn should_refresh_forward(&self, denoise_step: usize, is_prefill_block: bool) -> bool {
        if !self.predictive_enabled {
            return false;
        }
        if is_prefill_block {
            return true;
        }
        self.pools
            .first()
            .is_some_and(|p| p.should_refresh(rlx_runtime::MoEExecMode::Reuse, denoise_step, false))
    }

    /// Block diffusion: TIDE `generate` refresh line.
    pub fn should_refresh_block(
        &self,
        num_block: usize,
        prefill_blocks: usize,
        denoise_step: usize,
    ) -> bool {
        if !self.predictive_enabled {
            return false;
        }
        refresh_experts(
            true,
            self.jump_steps,
            num_block,
            prefill_blocks,
            denoise_step,
        )
    }

    /// Apply captured TopK indices per layer; returns true if any layer refreshed.
    pub fn refresh_from_capture(
        &mut self,
        layer_indices: &[Vec<u32>],
        denoise_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        let n = self.pools.len().min(layer_indices.len());
        if n == 0 {
            return false;
        }
        if !self.should_refresh_forward(denoise_step, is_prefill_block) {
            return false;
        }
        for (pool, idx) in self.pools.iter_mut().zip(&layer_indices[..n]) {
            pool.refresh_from_indices(idx);
        }
        true
    }

    pub fn refresh_from_capture_with_store(
        &mut self,
        store: &MoeExpertStore,
        captured: &[Vec<u32>],
        denoise_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        if !self.should_refresh_forward(denoise_step, is_prefill_block) {
            return false;
        }
        store.refresh_pools(&mut self.pools, captured, denoise_step, is_prefill_block)
    }

    pub fn tide_offload_stats(&self, residency: Option<&MoeResidencyStats>) -> TideOffloadStats {
        aggregate_offload_stats(&self.pools, residency)
    }
}
