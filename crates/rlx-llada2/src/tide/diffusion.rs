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

// RLX — block diffusion driver (delegates to [`super::generate`] for TIDE parity).

use crate::config::LLaDA2MoeConfig;
use crate::tide::stats::TideOffloadStats;
use crate::tide::{DenoiseStepCtx, GenerateConfig, run_block_diffusion};

/// Block diffusion generation options (TIDE `generate` defaults).
#[derive(Debug, Clone)]
pub struct BlockDenoiseConfig {
    pub temperature: f32,
    pub block_length: usize,
    pub steps: usize,
    pub gen_length: usize,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub eos_early_stop: bool,
    pub minimal_topk: usize,
    pub threshold: f32,
    pub mask_id: u32,
    pub eos_id: u32,
    pub do_sample: bool,
    pub predictive_offload_enabled: bool,
    pub jump_steps: usize,
    pub collect_stats: bool,
}

impl Default for BlockDenoiseConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            block_length: 32,
            steps: 32,
            gen_length: 2048,
            top_p: None,
            top_k: None,
            eos_early_stop: false,
            minimal_topk: 1,
            threshold: 0.9,
            mask_id: 156895,
            eos_id: 156892,
            do_sample: false,
            predictive_offload_enabled: false,
            jump_steps: 1,
            collect_stats: false,
        }
    }
}

impl BlockDenoiseConfig {
    pub fn to_generate_config(&self) -> GenerateConfig {
        GenerateConfig {
            temperature: self.temperature,
            block_length: self.block_length,
            steps: self.steps,
            gen_length: self.gen_length,
            top_p: self.top_p,
            top_k: self.top_k,
            eos_early_stop: self.eos_early_stop,
            minimal_topk: self.minimal_topk,
            threshold: self.threshold,
            eos_id: self.eos_id,
            mask_id: self.mask_id,
            do_sample: self.do_sample,
            predictive_offload_enabled: self.predictive_offload_enabled,
            jump_steps: self.jump_steps,
            collect_stats: self.collect_stats,
        }
    }
}

/// One denoise step record when `collect_stats` is enabled.
pub use crate::tide::generate::BlockDenoiseStepStats;

/// Forward callback: `refresh_experts` flag is set per TIDE policy before each call.
pub trait BlockDiffusionForward {
    fn forward_block(
        &mut self,
        token_ids: &[u32],
        seq_len: usize,
        refresh_experts: bool,
    ) -> Result<BlockForwardOutput, anyhow::Error>;
}

#[derive(Debug, Clone)]
pub struct BlockForwardOutput {
    pub x0: Vec<u32>,
    pub x0_p: Vec<f32>,
}

struct BlockSamplerAdapter<'a, F: BlockDiffusionForward> {
    forward: &'a mut F,
    block_length: usize,
}

impl<F: BlockDiffusionForward> crate::tide::generate::BlockDenoiseSampler
    for BlockSamplerAdapter<'_, F>
{
    fn sample_block(
        &mut self,
        x: &[u32],
        window_end: usize,
        block_length: usize,
        refresh_experts: bool,
        _gen_cfg: &GenerateConfig,
        _model_cfg: &LLaDA2MoeConfig,
        _step_ctx: DenoiseStepCtx,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let out = self.forward.forward_block(x, window_end, refresh_experts)?;
        let mut x0 = vec![0u32; block_length];
        let mut x0_p = vec![0f32; block_length];
        for i in 0..block_length.min(out.x0.len()) {
            x0[i] = out.x0[i];
            x0_p[i] = out.x0_p.get(i).copied().unwrap_or(0.0);
        }
        let _ = self.block_length;
        Ok((x0, x0_p))
    }
}

/// Driver for TIDE-style block masked diffusion (host-side token state).
pub struct BlockDenoiseLoop<F: BlockDiffusionForward> {
    pub cfg: BlockDenoiseConfig,
    pub model_cfg: LLaDA2MoeConfig,
    pub forward: F,
    pub offload_stats: Option<fn() -> TideOffloadStats>,
}

impl<F: BlockDiffusionForward> BlockDenoiseLoop<F> {
    pub fn new(cfg: BlockDenoiseConfig, model_cfg: LLaDA2MoeConfig, forward: F) -> Self {
        Self {
            cfg,
            model_cfg,
            forward,
            offload_stats: None,
        }
    }

    pub fn with_offload_stats(mut self, f: fn() -> TideOffloadStats) -> Self {
        self.offload_stats = Some(f);
        self
    }

    /// Run block diffusion from `prompt_ids`; returns generated suffix + optional stats.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<BlockDenoiseStepStats>), anyhow::Error> {
        let gen_cfg = self.cfg.to_generate_config();
        let mut adapter = BlockSamplerAdapter {
            forward: &mut self.forward,
            block_length: gen_cfg.block_length,
        };
        let stats_fn = self.offload_stats;
        run_block_diffusion(
            &mut adapter,
            &self.model_cfg,
            &gen_cfg,
            prompt_ids,
            move |_s| stats_fn.map(|f| f()).unwrap_or_default(),
        )
    }
}
