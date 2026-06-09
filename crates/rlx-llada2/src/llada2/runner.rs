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

// RLX — LLaDA2 MoE runner (block diffusion + TIDE offload, all backends).

use crate::builder::build_llada2_forward_graph;
use crate::capabilities::{default_memory_budget_bytes, validate_device};
use crate::compile_util::{compile_llada2_built, llada2_profile};
use crate::config::LLaDA2MoeConfig;
use crate::gate_op::ensure_group_limited_gate_registered;
use crate::load::load_llada2_from_dir;
use crate::mask::block_diffusion_attention_mask;
use crate::moe_offload::{self, MoeOffloadState};
use crate::moe_store::{
    apply_moe_store_to_compiled, build_moe_expert_store, moe_host_bind_from_store,
};
use crate::sampling::sample_logits;
use crate::tide::{
    BlockDenoiseConfig, BlockDenoiseLoop, BlockDenoiseSampler, BlockDiffusionForward,
    BlockForwardOutput, DenoiseStepCtx, GenerateConfig, run_block_diffusion,
};
use crate::weights::LLaDA2Weights;
use anyhow::{Result, anyhow};
use rlx_core::flow_util::built_from_graph;
use rlx_runtime::{CompiledGraph, Device, MoeExpertStore, MoeResidencyStats};

fn push_moe_residency(compiled: &mut CompiledGraph, layers: &[Vec<bool>]) {
    let refs: Vec<&[bool]> = layers.iter().map(|m| m.as_slice()).collect();
    compiled.set_moe_resident_experts_per_layer(&refs);
}

#[derive(Default)]
pub struct LLaDA2RunnerBuilder {
    inline: Option<(LLaDA2MoeConfig, LLaDA2Weights)>,
    weights_path: Option<std::path::PathBuf>,
    device: Option<Device>,
    batch: usize,
    max_seq: Option<usize>,
    max_gpu_experts_per_layer: Option<usize>,
    memory_budget_bytes: Option<usize>,
    jump_steps: Option<usize>,
    reserve_vram_gb: f64,
    moe_collect_stats: bool,
}

impl LLaDA2RunnerBuilder {
    pub fn inline_weights(mut self, cfg: LLaDA2MoeConfig, weights: LLaDA2Weights) -> Self {
        self.inline = Some((cfg, weights));
        self
    }

    pub fn weights_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.weights_path = Some(path.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn batch_seq(mut self, batch: usize, max_seq: usize) -> Self {
        self.batch = batch.max(1);
        self.max_seq = Some(max_seq.max(1));
        self
    }

    pub fn enable_predictive_expert_offload(mut self, max_per_layer: usize) -> Self {
        self.max_gpu_experts_per_layer = Some(max_per_layer);
        self
    }

    pub fn jump_steps(mut self, n: usize) -> Self {
        self.jump_steps = Some(n);
        self
    }

    pub fn reserve_vram_gb(mut self, gb: f64) -> Self {
        self.reserve_vram_gb = gb;
        self
    }

    pub fn moe_collect_stats(mut self, on: bool) -> Self {
        self.moe_collect_stats = on;
        self
    }

    pub fn memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    pub fn build(self) -> Result<LLaDA2Runner> {
        ensure_group_limited_gate_registered();

        let (cfg, weights) = match self.inline {
            Some(x) => x,
            None => {
                let path = self.weights_path.as_ref().ok_or_else(|| {
                    anyhow!("LLaDA2Runner: weights_path or inline_weights required")
                })?;
                load_llada2_from_dir(path.as_path())?
            }
        };

        let device = self.device.unwrap_or(Device::Cpu);
        validate_device(&cfg, device)?;
        let batch = self.batch.max(1);
        let seq = self.max_seq.unwrap_or(128).max(1);

        let (graph, params) = build_llada2_forward_graph(&cfg, &weights, batch, seq)?;
        let mut built = built_from_graph(graph, params)?;
        built.profile = llada2_profile();
        let mut compiled = compile_llada2_built(built, device)?;

        let moe_store = if cfg.num_experts > 0 {
            Some(build_moe_expert_store(&cfg, &weights)?)
        } else {
            None
        };

        let mem_budget = self
            .memory_budget_bytes
            .or_else(|| default_memory_budget_bytes(device));

        let moe = moe_offload::build_moe_offload(
            &cfg,
            &weights,
            device,
            self.max_gpu_experts_per_layer,
            mem_budget,
            self.jump_steps,
            self.reserve_vram_gb,
            self.moe_collect_stats,
        );

        if let Some(mo) = &moe {
            push_moe_residency(&mut compiled, &mo.per_layer_resident_masks());
            compiled.enable_moe_topk_capture(cfg.num_experts);
            if let Some(store) = &moe_store {
                apply_moe_store_to_compiled(store, &mut compiled);
            }
        }

        Ok(LLaDA2Runner {
            cfg,
            weights,
            compiled,
            device,
            batch,
            seq,
            block_length: 32,
            moe,
            moe_store,
        })
    }
}

pub struct LLaDA2Runner {
    pub cfg: LLaDA2MoeConfig,
    pub weights: LLaDA2Weights,
    compiled: CompiledGraph,
    device: Device,
    batch: usize,
    seq: usize,
    block_length: usize,
    moe: Option<MoeOffloadState>,
    moe_store: Option<MoeExpertStore>,
}

impl LLaDA2Runner {
    pub fn builder() -> LLaDA2RunnerBuilder {
        LLaDA2RunnerBuilder::default()
    }

    pub fn config(&self) -> &LLaDA2MoeConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn max_seq(&self) -> usize {
        self.seq
    }

    pub fn predictive_offload_enabled(&self) -> bool {
        self.moe.as_ref().is_some_and(|m| m.predictive_enabled)
    }

    pub fn jump_steps(&self) -> usize {
        self.moe.as_ref().map(|m| m.jump_steps).unwrap_or(1)
    }

    pub fn predictive_offload_info(&self) -> Option<crate::tide::PredictiveOffloadInfo> {
        self.moe.as_ref().map(|m| m.info.clone())
    }

    pub fn moe_offload(&self) -> Option<&MoeOffloadState> {
        self.moe.as_ref()
    }

    pub fn moe_store(&self) -> Option<&MoeExpertStore> {
        self.moe_store.as_ref()
    }

    pub fn sync_moe_residency(&self, compiled: &mut CompiledGraph) {
        if let Some(mo) = &self.moe {
            push_moe_residency(compiled, &mo.per_layer_resident_masks());
            if let Some(store) = &self.moe_store {
                apply_moe_store_to_compiled(store, compiled);
            }
        }
    }

    fn bind_moe_host_weights(&self) {
        if self.moe.is_none() {
            rlx_cpu::moe_residency::bind_host_weights(None);
            return;
        }
        if let Some(store) = &self.moe_store {
            rlx_cpu::moe_residency::bind_host_weights(Some(moe_host_bind_from_store(store)));
        } else {
            rlx_cpu::moe_residency::bind_host_weights(None);
        }
    }

    fn refresh_moe_after_forward(&mut self, step_ctx: DenoiseStepCtx, want_refresh: bool) {
        let Some(layers) = self.compiled.take_moe_topk_capture() else {
            return;
        };
        let Some(mo) = self.moe.as_mut() else {
            return;
        };
        let is_prefill = step_ctx.num_block == step_ctx.prefill_blocks;
        if !want_refresh || !mo.should_refresh_forward(step_ctx.denoise_step, is_prefill) {
            return;
        }
        let refreshed = if let Some(store) = self.moe_store.as_ref() {
            mo.refresh_from_capture_with_store(store, &layers, step_ctx.denoise_step, is_prefill)
        } else {
            mo.refresh_from_capture(&layers, step_ctx.denoise_step, is_prefill)
        };
        if refreshed {
            let masks = mo.per_layer_resident_masks();
            push_moe_residency(&mut self.compiled, &masks);
            if let Some(store) = &self.moe_store {
                apply_moe_store_to_compiled(store, &mut self.compiled);
            }
        }
    }

    fn forward_window_padded(
        &mut self,
        tokens: &[u32],
        window_len: usize,
        attn_mask: &[f32],
        position_ids: &[f32],
        step_ctx: DenoiseStepCtx,
        want_refresh: bool,
    ) -> Result<Vec<f32>> {
        let b = self.batch;
        let s = self.seq;
        let w = window_len.min(tokens.len()).min(s);
        let mut ids = vec![0f32; b * s];
        let mut pos = vec![0f32; b * s];
        for i in 0..w {
            ids[i] = tokens[i] as f32;
            pos[i] = position_ids.get(i).copied().unwrap_or(i as f32);
        }
        let mut full_mask = vec![f32::NEG_INFINITY; b * s * s];
        for r in 0..w {
            for c in 0..w {
                full_mask[r * s + c] = attn_mask[r * w + c];
            }
        }
        let logits = self.forward_logits(&ids, &pos, &full_mask)?;
        self.refresh_moe_after_forward(step_ctx, want_refresh);
        Ok(logits)
    }

    pub fn forward_logits(
        &mut self,
        input_ids: &[f32],
        position_ids: &[f32],
        attn_mask: &[f32],
    ) -> Result<Vec<f32>> {
        let b = self.batch;
        let s = self.seq;
        if input_ids.len() != b * s {
            return Err(anyhow!("input_ids len {} != {b}*{s}", input_ids.len()));
        }
        if attn_mask.len() != b * s * s {
            return Err(anyhow!(
                "attn_mask len {} != {b}*1*{s}*{s}",
                attn_mask.len()
            ));
        }
        self.bind_moe_host_weights();
        let outs = self.compiled.run(&[
            ("input_ids", input_ids),
            ("position_ids", position_ids),
            ("attn_mask", attn_mask),
        ]);
        Ok(outs.into_iter().next().unwrap_or_default())
    }

    pub fn block_denoise_loop(
        &mut self,
        cfg: BlockDenoiseConfig,
    ) -> BlockDenoiseLoop<LLaDA2RunnerForward<'_>> {
        self.block_length = cfg.block_length;
        let model_cfg = self.cfg.clone();
        BlockDenoiseLoop::new(cfg, model_cfg, LLaDA2RunnerForward { runner: self })
    }

    pub fn get_offload_stats(&mut self) -> crate::tide::TideOffloadStats {
        let residency = self
            .compiled
            .take_moe_residency_stats()
            .or_else(rlx_cpu::moe_residency::peek_stats);
        let residency_ref = residency.as_ref();
        self.offload_stats(residency_ref).unwrap_or_default()
    }

    pub fn reset_offload_step_stats(&mut self) {
        if let Some(mo) = self.moe.as_mut() {
            for pool in &mut mo.pools {
                pool.reset_step_stats();
            }
        }
        let _ = self.compiled.take_moe_residency_stats();
    }

    pub fn generate(
        &mut self,
        gen_cfg: &GenerateConfig,
        prompt_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<crate::tide::BlockDenoiseStepStats>)> {
        let max_window = (prompt_ids.len() + gen_cfg.gen_length).div_ceil(gen_cfg.block_length)
            * gen_cfg.block_length;
        if max_window > self.seq {
            return Err(anyhow!(
                "generate needs max_seq >= {max_window} (set .batch_seq(batch, max_seq) on builder)"
            ));
        }
        let cfg = self.cfg.clone();
        let collect = gen_cfg.collect_stats;
        run_block_diffusion(self, &cfg, gen_cfg, prompt_ids, |runner| {
            let stats = runner.get_offload_stats();
            if collect {
                runner.reset_offload_step_stats();
            }
            stats
        })
    }

    pub fn offload_stats(
        &self,
        residency: Option<&MoeResidencyStats>,
    ) -> Option<crate::tide::TideOffloadStats> {
        self.moe
            .as_ref()
            .map(|m| moe_offload::tide_stats(m, residency))
    }
}

impl BlockDenoiseSampler for LLaDA2Runner {
    fn sample_block(
        &mut self,
        x: &[u32],
        window_end: usize,
        block_length: usize,
        refresh_experts: bool,
        gen_cfg: &GenerateConfig,
        model_cfg: &LLaDA2MoeConfig,
        step_ctx: DenoiseStepCtx,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let mask = block_diffusion_attention_mask(1, window_end, block_length);
        let position_ids: Vec<f32> = (0..window_end).map(|i| i as f32).collect();
        let logits = self.forward_window_padded(
            &x[..window_end],
            window_end,
            &mask,
            &position_ids,
            step_ctx,
            refresh_experts,
        )?;
        let block_start = window_end.saturating_sub(block_length);
        let vocab = model_cfg.vocab_size;
        let mut x0 = vec![0u32; block_length];
        let mut x0_p = vec![0f32; block_length];
        for i in 0..block_length {
            let pos = block_start + i;
            if pos >= window_end {
                x0[i] = gen_cfg.mask_id;
                x0_p[i] = 0.0;
                continue;
            }
            let base = pos * vocab;
            let (tok, prob) = sample_logits(
                &logits[base..base + vocab],
                gen_cfg.temperature,
                gen_cfg.top_k,
                gen_cfg.top_p,
                gen_cfg.do_sample,
            );
            x0[i] = tok;
            x0_p[i] = prob;
        }
        Ok((x0, x0_p))
    }
}

pub struct LLaDA2RunnerForward<'a> {
    pub runner: &'a mut LLaDA2Runner,
}

impl BlockDiffusionForward for LLaDA2RunnerForward<'_> {
    fn forward_block(
        &mut self,
        token_ids: &[u32],
        seq_len: usize,
        refresh_experts: bool,
    ) -> Result<BlockForwardOutput, anyhow::Error> {
        let b = self.runner.batch;
        let s = self.runner.seq;
        let block = self.runner.block_length;
        let window = seq_len.min(token_ids.len()).min(s);
        let block_start = window.saturating_sub(block);

        let mut ids = vec![0f32; b * s];
        let mut pos = vec![0f32; b * s];
        for i in 0..window {
            ids[i] = token_ids[i] as f32;
            pos[i] = i as f32;
        }

        let mask = block_diffusion_attention_mask(b, window, block);
        let position_ids: Vec<f32> = (0..window).map(|i| i as f32).collect();
        let step_ctx = DenoiseStepCtx {
            num_block: 0,
            prefill_blocks: 0,
            denoise_step: 0,
        };
        let logits = self.runner.forward_window_padded(
            &token_ids[..window],
            window,
            &mask,
            &position_ids,
            step_ctx,
            refresh_experts,
        )?;

        let vocab = self.runner.cfg.vocab_size;
        let mut x0 = Vec::with_capacity(block);
        let mut x0_p = Vec::with_capacity(block);
        for i in 0..block {
            let tok_pos = block_start + i;
            if tok_pos >= window {
                x0.push(self.runner.cfg.mask_token_id);
                x0_p.push(0.0);
                continue;
            }
            let base = tok_pos * vocab;
            if base + vocab > logits.len() {
                break;
            }
            let (tok, conf) = sample_logits(&logits[base..base + vocab], 0.0, None, None, false);
            x0.push(tok);
            x0_p.push(conf);
        }
        Ok(BlockForwardOutput { x0, x0_p })
    }
}
