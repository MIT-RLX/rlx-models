// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard compiled backbone session — real Qwen3.5 prefill/decode on the chosen device.

use anyhow::{Context, Result};
use rlx_qwen35::{Qwen35DecodeCache, Qwen35Runner, Qwen35RunnerBuilder};
use rlx_runtime::Device;
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::GepardConfig;
use crate::qwen35_adapter::load_gepard_qwen35_bundle;
use crate::weights::load_safetensors_bytes;

/// Per-stage timing from the last compiled synthesis run (milliseconds).
#[derive(Debug, Clone, Default)]
pub struct GepardTiming {
    pub prefill_ms: f64,
    pub ar_decode_ms: f64,
    pub frames: usize,
}

/// Compiled Gepard backbone — Qwen3.5 runner on the selected backend.
pub struct GepardCompiledSession {
    device: Device,
    hidden_size: usize,
    token_embd: Arc<[f32]>,
    /// Folded `(1 + w)` output RMSNorm gamma (HF Qwen3.5).
    output_norm: Vec<f32>,
    rms_eps: f32,
    runner: Qwen35Runner,
    max_seq: usize,
    pub last_timing: GepardTiming,
}

impl GepardCompiledSession {
    /// Build a compiled Qwen3.5 session from Gepard safetensors on `device`.
    pub fn new(device: Device, cfg: &GepardConfig, weights_dir: &Path) -> Result<Self> {
        let model_path = weights_dir.join("model.safetensors");
        let bytes = load_safetensors_bytes(&model_path)
            .with_context(|| format!("read {}", model_path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse {}", model_path.display()))?;

        let (qcfg, qweights) = load_gepard_qwen35_bundle(&st, &cfg.backbone)?;
        let hidden_size = cfg.backbone.hidden_size;
        let token_embd = qweights.token_embd.clone();
        let output_norm = qweights.output_norm.clone();
        let rms_eps = cfg.backbone.rms_norm_eps as f32;

        // TTS context is short (prompt + ~400–800 frames). Prefer a leaner CUDA
        // decode graph; override with RLX_GEPARD_MAX_SEQ. Default 1024 covers
        // the bench long paragraph (≈560 frames + prefill) without OOM from
        // dynamic past growth past a 512 ceiling.
        let max_seq = std::env::var("RLX_GEPARD_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024usize)
            .clamp(128, 2048);

        let runner_device = backbone_runner_device(device);
        if runner_device != device {
            eprintln!(
                "[gepard] Qwen3.5 AR: {device:?} → {runner_device:?} (NanoCodec stays on {device:?})"
            );
        }

        // One padded decode graph (bucketed) — matches eager after fixing
        // `slice_kv_from_bucket` to keep the new K/V at `upper`. Dynamic decode
        // (`RLX_GEPARD_DYNAMIC_DECODE=1`) recompiles per past_seq and OOMs long AR.
        let dynamic_decode = env_truthy("RLX_GEPARD_DYNAMIC_DECODE");
        let runner = Qwen35RunnerBuilder::default()
            .inline_weights(qcfg, qweights)
            .device(runner_device)
            .max_seq(max_seq)
            .batch(1)
            .dynamic_prefill(true)
            .dynamic_decode(dynamic_decode)
            .bucketed_decode(!dynamic_decode)
            .hidden_prefill(true)
            .force_host_embed(true)
            .fast_greedy_lm_head(true)
            .skip_warm(true)
            .build()
            .context("build Qwen35 runner for Gepard backbone")?;

        Ok(Self {
            device,
            hidden_size,
            token_embd,
            output_norm,
            rms_eps,
            runner,
            max_seq,
            last_timing: GepardTiming::default(),
        })
    }

    /// Final RMSNorm for decode/prefill trunks that export the pre-norm residual.
    /// Prefill-cache with `fast_greedy_lm_head` still emits post-layer `h` for some
    /// decode specializations; matching eager always applies this once on the host.
    fn apply_output_norm(&self, h: &[f32]) -> Vec<f32> {
        rms_norm_gamma(h, &self.output_norm, self.rms_eps)
    }

    /// Gather text token embeddings from the backbone table (host-side).
    pub fn embed_tokens(&self, ids: &[u32]) -> Vec<f32> {
        let h = self.hidden_size;
        let mut out = vec![0.0f32; ids.len() * h];
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * h;
            if src + h <= self.token_embd.len() {
                out[i * h..(i + 1) * h].copy_from_slice(&self.token_embd[src..src + h]);
            }
        }
        out
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Prefill text (+ optional prefix) hidden states; returns SOS hidden and decode cache.
    ///
    /// Uses sequential token steps (prefill token 0, then decode 1..N-1) so numerics match
    /// the eager CPU backbone, which is required for sampling-sensitive long utterances.
    pub fn prefill_hidden(
        &mut self,
        inputs: &[f32],
        n_prefill: usize,
    ) -> Result<(Vec<f32>, Qwen35DecodeCache)> {
        let hidden = self.hidden_size;
        if inputs.len() != n_prefill * hidden {
            anyhow::bail!(
                "prefill inputs len {} != n_prefill*hidden {}*{}",
                inputs.len(),
                n_prefill,
                hidden
            );
        }
        if n_prefill == 0 {
            anyhow::bail!("empty prefill");
        }
        if n_prefill > self.max_seq {
            anyhow::bail!(
                "prefill len {n_prefill} exceeds compiled max_seq={}",
                self.max_seq
            );
        }
        let t0 = Instant::now();
        let first = &inputs[..hidden];
        let (mut trunk, mut cache) = self.runner.prefill_hidden_state(first, 1)?;
        for t in 1..n_prefill {
            let emb = &inputs[t * hidden..(t + 1) * hidden];
            trunk = self.runner.decode_hidden_state(&mut cache, emb)?;
        }
        // Decode / sequential prefill trunks are pre-final-norm residuals;
        // apply the folded HF `(1+w)` RMSNorm once (matches eager backbone).
        let sos = self.apply_output_norm(&trunk);
        self.last_timing.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // Sequential token steps leave prompt_lens at 1; mark the full prompt so
        // bucketed masks treat the prefix as prompt (generated count stays correct).
        cache.prompt_lens = vec![n_prefill];
        // Prefill graphs are unused during the long AR decode; drop them on
        // VRAM-tight backends so decode buckets can allocate.
        if matches!(
            self.runner.device(),
            Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
        ) {
            self.runner.drop_prefill_cache();
        }
        Ok((sos, cache))
    }

    /// One AR decode step from an audio-frame embedding.
    pub fn decode_hidden(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        frame_embed: &[f32],
    ) -> Result<Vec<f32>> {
        let h = self.runner.decode_hidden_state(cache, frame_embed)?;
        Ok(self.apply_output_norm(&h))
    }

    /// Record AR loop wall time (call after the frame loop).
    pub fn record_ar_timing(&mut self, ar_ms: f64, frames: usize) {
        self.last_timing.ar_decode_ms = ar_ms;
        self.last_timing.frames = frames;
    }
}

fn rms_norm_gamma(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(gamma).map(|(v, g)| v / rms * g).collect()
}

fn env_truthy(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Backbone device for Qwen3.5 AR graphs.
///
/// Metal prefers MLX when available. Other backends run compiled AR on-device.
/// Force eager CPU AR with `RLX_GEPARD_EAGER_AR=1`.
pub(crate) fn backbone_runner_device(requested: Device) -> Device {
    if matches!(
        std::env::var("RLX_GEPARD_EAGER_AR").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    ) {
        return Device::Cpu;
    }
    match requested {
        Device::Metal if rlx_runtime::is_available(Device::Mlx) => Device::Mlx,
        other => other,
    }
}
