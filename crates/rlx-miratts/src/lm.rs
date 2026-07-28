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

//! MiraTTS Qwen2-0.5B LM (STEP 2). The LM is a Qwen2, which in RLX is exactly a
//! `rlx-qwen3` model with **`qk_norm = false`** (Qwen2 has no QK-norm) and
//! **`attention_bias = true`** (Qwen2 has qkv bias). It generates the acoustic
//! `<|speech_token_N|>` stream autoregressively from a prompt built by
//! [`crate::tokens`]. The AR loop mirrors `rlx-orpheus`'s codec generation.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_qwen3::{
    Qwen3Config, Qwen3Generator, Qwen3RunnerBuilder, SampleOpts, apply_repetition_penalty,
    sample_token_at,
};
use rlx_runtime::{Device, parse_device};

use crate::{MiraConfig, tokens};

fn resolve_lm_device(requested: Device) -> Device {
    if let Ok(v) = std::env::var("RLX_MIRATTS_LM_DEVICE") {
        if let Ok(d) = parse_device(v.trim()) {
            return d;
        }
    }
    // GPU graph compile for the 0.5B AR is slow/fragile on first use; keep LM on host.
    match requested {
        Device::Cuda | Device::Gpu | Device::Vulkan | Device::Metal | Device::Mlx => Device::Cpu,
        other => other,
    }
}

/// Build the `rlx-qwen3` config for the MiraTTS Qwen2-0.5B LM. The two flags
/// that distinguish Qwen2 from Qwen3 (`qk_norm`, `attention_bias`) are absent
/// from the model's `config.json`, so they're set explicitly here.
pub fn qwen2_config(m: &MiraConfig) -> Qwen3Config {
    Qwen3Config {
        vocab_size: m.vocab_size,
        hidden_size: m.hidden_size,
        intermediate_size: m.intermediate_size,
        num_hidden_layers: m.num_hidden_layers,
        num_attention_heads: m.num_attention_heads,
        num_key_value_heads: m.num_key_value_heads,
        head_dim: m.hidden_size / m.num_attention_heads,
        max_position_embeddings: 32768,
        rms_norm_eps: m.rms_norm_eps as f64,
        rope_theta: m.rope_theta as f64,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: m.tie_word_embeddings,
        attention_bias: true, // Qwen2 has qkv bias
        qk_norm: false,       // Qwen2 has no QK-norm
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// The MiraTTS language model — a Qwen2-0.5B via `rlx-qwen3`.
pub struct MiraLm {
    generator: Qwen3Generator,
}

impl MiraLm {
    /// Load `model.safetensors` + the Qwen2 config from a model directory.
    ///
    /// The LM defaults to **CPU** when the requested device is a GPU accelerator:
    /// compiling the Qwen2-0.5B AR graph for CUDA/wgpu is slow and often hangs
    /// the first bench cell. Codec + speaker encoder still use `device`.
    /// Override with `RLX_MIRATTS_LM_DEVICE=cuda|gpu|cpu`.
    pub fn load(dir: &Path, cfg: &MiraConfig, device: Device) -> Result<Self> {
        let lm_device = resolve_lm_device(device);
        if lm_device != device {
            eprintln!(
                "[miratts] LM on {lm_device:?} (codec/speaker default CPU; set RLX_MIRATTS_LM_DEVICE to override)"
            );
        }
        eprintln!(
            "[miratts] loading Qwen2 LM from {} on {lm_device:?}…",
            dir.display()
        );
        let runner = Qwen3RunnerBuilder::default()
            .weights(dir)
            .config_value(qwen2_config(cfg))
            .device(lm_device)
            .build()
            .context("build MiraTTS Qwen2 runner")?;
        let generator = runner
            .into_generator()
            .context("MiraTTS: f32 generator unavailable")?;
        eprintln!("[miratts] LM ready");
        Ok(Self { generator })
    }

    /// Deterministic greedy decode of `max_new` tokens from `prompt_ids` — used
    /// to validate the port against the HF transformers reference.
    pub fn generate_greedy(&mut self, prompt_ids: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut logits = self.generator.prefill_get_last_logits(prompt_ids)?;
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let next = argmax(&logits) as u32;
            out.push(next);
            logits = self.generator.decode_get_logits(next)?;
        }
        Ok(out)
    }

    /// Sample the acoustic `<|speech_token_N|>` stream with the MiraTTS
    /// generation config (temp 0.8, top_p 0.95, top_k 50, min_p 0.05,
    /// rep-penalty 1.2), stopping at eos / `<|end_acoustic_token|>`. Returns the
    /// raw acoustic codes (`id - SPEECH_BASE`), ready for the codec decoder.
    pub fn generate_speech_codes(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let opts = SampleOpts::temperature(0.8, seed)
            .with_top_k(50)
            .with_top_p(0.95)
            .with_min_p(0.05);
        eprintln!(
            "[miratts] AR speech codes: prompt_len={} max_new={max_new}",
            prompt_ids.len()
        );
        let mut logits = self.generator.prefill_get_last_logits(prompt_ids)?;
        let mut counts: HashMap<u32, u32> = HashMap::new();
        let mut codes = Vec::new();
        for step in 0..max_new {
            apply_repetition_penalty(&mut logits, &counts, 1.2);
            let next = sample_token_at(&logits, opts, step as u64) as u32;
            if tokens::is_stop(next) {
                break;
            }
            if (tokens::SPEECH_BASE..tokens::SPEECH_BASE + tokens::SPEECH_CODEBOOK).contains(&next)
            {
                codes.push(next - tokens::SPEECH_BASE);
            }
            *counts.entry(next).or_insert(0) += 1;
            if step == 0 || (step + 1) % 16 == 0 || step + 1 == max_new {
                eprintln!(
                    "[miratts] AR {}/{max_new} (codes={})",
                    step + 1,
                    codes.len()
                );
            }
            logits = self.generator.decode_get_logits(next)?;
        }
        Ok(codes)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_config_flags() {
        let c = qwen2_config(&MiraConfig::default());
        assert!(!c.qk_norm, "Qwen2 has no QK-norm");
        assert!(c.attention_bias, "Qwen2 has qkv bias");
        assert!(c.tie_word_embeddings);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.vocab_size, 166000);
    }
}
