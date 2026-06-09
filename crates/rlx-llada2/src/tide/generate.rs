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

// RLX — block diffusion `generate()` (TIDE `LLaDA2MoeModelLM.generate`).

use crate::config::LLaDA2MoeConfig;
use crate::mask::block_diffusion_attention_mask;
use crate::sampling::sample_logits;
use crate::tide::{TideOffloadStats, refresh_experts};

/// One denoise step record when `collect_stats` is enabled.
#[derive(Debug, Clone, Default)]
pub struct BlockDenoiseStepStats {
    pub block: usize,
    pub step: usize,
    pub elapsed_ms: f64,
    pub active_tokens: usize,
    pub transferred_tokens: usize,
    pub offload_stats: TideOffloadStats,
}

/// Schedule of mask tokens to unmask per denoise step (TIDE `_get_num_transfer_tokens`).
pub fn num_transfer_tokens_schedule(block_length: usize, steps: usize) -> Vec<usize> {
    if steps == 0 {
        return Vec::new();
    }
    let base = block_length / steps;
    let remainder = block_length % steps;
    let mut schedule = vec![base; steps];
    for slot in schedule.iter_mut().take(remainder) {
        *slot += 1;
    }
    schedule
}

/// Generation options matching PyTorch `LLaDA2MoeModelLM.generate`.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub temperature: f32,
    pub block_length: usize,
    pub steps: usize,
    pub gen_length: usize,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub eos_early_stop: bool,
    pub minimal_topk: usize,
    /// Confidence threshold for unmasking (`eval_dinfer.py` default 0.9).
    pub threshold: f32,
    pub eos_id: u32,
    pub mask_id: u32,
    pub do_sample: bool,
    pub predictive_offload_enabled: bool,
    pub jump_steps: usize,
    pub collect_stats: bool,
}

impl GenerateConfig {
    pub fn from_model(cfg: &LLaDA2MoeConfig) -> Self {
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
            eos_id: cfg.eos_token_id,
            mask_id: cfg.mask_token_id,
            do_sample: false,
            predictive_offload_enabled: false,
            jump_steps: 1,
            collect_stats: false,
        }
    }
}

pub trait GenerateForward {
    fn forward_window(
        &mut self,
        tokens: &[u32],
        window_len: usize,
        attn_mask: &[f32],
        position_ids: &[f32],
        refresh_experts: bool,
    ) -> anyhow::Result<Vec<f32>>;
}

/// Per-denoise-step context for MoE expert refresh (TIDE `generate` loop).
#[derive(Debug, Clone, Copy, Default)]
pub struct DenoiseStepCtx {
    pub num_block: usize,
    pub prefill_blocks: usize,
    pub denoise_step: usize,
}

/// Sample `(x0, x0_p)` for the trailing block of the active window.
pub trait BlockDenoiseSampler {
    fn sample_block(
        &mut self,
        x: &[u32],
        window_end: usize,
        block_length: usize,
        refresh_experts: bool,
        gen_cfg: &GenerateConfig,
        model_cfg: &LLaDA2MoeConfig,
        step_ctx: DenoiseStepCtx,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)>;
}

impl<F: GenerateForward> BlockDenoiseSampler for F {
    fn sample_block(
        &mut self,
        x: &[u32],
        window_end: usize,
        block_length: usize,
        refresh_experts: bool,
        gen_cfg: &GenerateConfig,
        model_cfg: &LLaDA2MoeConfig,
        _step_ctx: DenoiseStepCtx,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let mask = block_diffusion_attention_mask(1, window_end, block_length);
        let position_ids: Vec<f32> = (0..window_end).map(|i| i as f32).collect();
        let logits = self.forward_window(
            &x[..window_end],
            window_end,
            &mask,
            &position_ids,
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

/// Run TIDE block diffusion; returns generated suffix (after prompt) + stats.
pub fn generate<S: BlockDenoiseSampler>(
    sampler: &mut S,
    cfg: &LLaDA2MoeConfig,
    gen_cfg: &GenerateConfig,
    prompt_ids: &[u32],
) -> anyhow::Result<(Vec<u32>, Vec<BlockDenoiseStepStats>)> {
    run_block_diffusion(sampler, cfg, gen_cfg, prompt_ids, |_| {
        TideOffloadStats::default()
    })
}

/// Run TIDE block diffusion; returns generated suffix (after prompt) + stats.
pub fn run_block_diffusion<S: BlockDenoiseSampler>(
    sampler: &mut S,
    cfg: &LLaDA2MoeConfig,
    gen_cfg: &GenerateConfig,
    prompt_ids: &[u32],
    mut offload_stats: impl FnMut(&mut S) -> TideOffloadStats,
) -> anyhow::Result<(Vec<u32>, Vec<BlockDenoiseStepStats>)> {
    let steps = gen_cfg
        .steps
        .min(gen_cfg.gen_length / gen_cfg.minimal_topk.max(1));
    let block_length = gen_cfg.block_length;
    let prompt_length = prompt_ids.len();
    let num_blocks = (prompt_length + gen_cfg.gen_length).div_ceil(block_length);
    let total_length = num_blocks * block_length;
    let prefill_blocks = prompt_length / block_length;

    let mut x = vec![gen_cfg.mask_id; total_length];
    x[..prompt_length].copy_from_slice(prompt_ids);

    let transfer_schedule = num_transfer_tokens_schedule(block_length, steps);
    let mut stats = Vec::new();

    for num_block in prefill_blocks..num_blocks {
        let window_end = (num_block + 1) * block_length;

        for step in 0..steps {
            let block_start = window_end.saturating_sub(block_length);
            let active_tokens = x[block_start..window_end]
                .iter()
                .filter(|&&t| t == gen_cfg.mask_id)
                .count();
            if active_tokens == 0 {
                break;
            }

            let refresh = refresh_experts(
                gen_cfg.predictive_offload_enabled,
                gen_cfg.jump_steps,
                num_block,
                prefill_blocks,
                step,
            );

            let t0 = std::time::Instant::now();
            let step_ctx = DenoiseStepCtx {
                num_block,
                prefill_blocks,
                denoise_step: step,
            };
            let (x0, x0_p) = sampler.sample_block(
                &x,
                window_end,
                block_length,
                refresh,
                gen_cfg,
                cfg,
                step_ctx,
            )?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let num_to_transfer = transfer_schedule
                .get(step)
                .copied()
                .unwrap_or(0)
                .min(active_tokens);

            let mut transfer = vec![false; block_length];
            let mut high_conf = 0usize;
            for i in 0..block_length {
                if x[block_start + i] != gen_cfg.mask_id {
                    continue;
                }
                if x0_p[i] > gen_cfg.threshold {
                    transfer[i] = true;
                    high_conf += 1;
                }
            }
            if high_conf < num_to_transfer {
                let mut ranked: Vec<(f32, usize)> = (0..block_length)
                    .filter(|&i| x[block_start + i] == gen_cfg.mask_id)
                    .map(|i| (x0_p[i], i))
                    .collect();
                ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (_, i) in ranked.into_iter().take(num_to_transfer) {
                    transfer[i] = true;
                }
            }

            let mut transferred = 0usize;
            for (i, &sel) in transfer.iter().enumerate() {
                if sel {
                    x[block_start + i] = x0[i];
                    transferred += 1;
                }
            }

            if gen_cfg.collect_stats {
                stats.push(BlockDenoiseStepStats {
                    block: num_block,
                    step,
                    elapsed_ms,
                    active_tokens,
                    transferred_tokens: transferred,
                    offload_stats: offload_stats(sampler),
                });
            }

            if gen_cfg.eos_early_stop
                && transfer
                    .iter()
                    .zip(x0.iter())
                    .any(|(&s, &t)| s && t == gen_cfg.eos_id)
            {
                if let Some(eos_pos) = x.iter().position(|&t| t == gen_cfg.eos_id) {
                    if x[prompt_length..eos_pos]
                        .iter()
                        .all(|&t| t != gen_cfg.mask_id)
                    {
                        return Ok((x[prompt_length..=eos_pos].to_vec(), stats));
                    }
                }
            }
        }

        if x[prompt_length..window_end].contains(&gen_cfg.eos_id) {
            break;
        }
    }

    let end = (prompt_length + gen_cfg.gen_length).min(x.len());
    let slice = &x[prompt_length..end];
    let eos_off = slice
        .iter()
        .position(|&t| t == gen_cfg.eos_id)
        .map(|p| p + 1)
        .unwrap_or(slice.len());
    Ok((slice[..eos_off].to_vec(), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_schedule_matches_tide() {
        assert_eq!(num_transfer_tokens_schedule(32, 32), vec![1; 32]);
        assert_eq!(num_transfer_tokens_schedule(10, 3), vec![4, 3, 3]);
    }

    #[test]
    fn from_model_threshold_matches_eval_dinfer() {
        let cfg = crate::llada2::synth::tiny_cfg();
        assert!((GenerateConfig::from_model(&cfg).threshold - 0.9).abs() < f32::EPSILON);
    }
}
