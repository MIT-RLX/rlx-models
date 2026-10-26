// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! The speech language model: a stock Qwen3-1.7B over a shared text/audio
//! vocabulary, speaker-conditioned through one extra head.
//!
//! `spk_proj` (`Linear(512 → 2048)` + `LayerNorm`) maps the voice's pyannote
//! x-vector to hidden size and *overwrites* position 0's input embedding at
//! prefill; every later step is an ordinary token embedding. Everything else
//! is `rlx-qwen3`'s native packed generator over the `Q2_0` GGUF that
//! [`crate::ternary`] transcodes from the shipped checkpoint, with the input
//! embeddings gathered on the host so position 0 can be replaced.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rlx_gguf::{GgmlType, GgufFile};
use rlx_qwen3::{Qwen3Config, Qwen3Generator};
use rlx_runtime::Device;

use crate::config::TokenMap;
use crate::sampling::{Rng, SamplerOpts, apply_penalties, sample};

/// Prefill length granularity (tokens).
const PREFILL_BUCKET: usize = 64;

/// File name of the transcoded checkpoint, written next to the ternary one.
pub const Q2_0_GGUF: &str = "model-q2_0.gguf";

/// `spk_proj`: Linear + LayerNorm, f32.
#[derive(Debug, Clone)]
pub struct SpeakerProjection {
    /// `[hidden, spk_dim]` row-major.
    weight: Vec<f32>,
    bias: Vec<f32>,
    ln_weight: Vec<f32>,
    ln_bias: Vec<f32>,
    spk_dim: usize,
    hidden: usize,
}

impl SpeakerProjection {
    pub fn from_ternary_checkpoint(path: &Path) -> Result<Self> {
        let mut t = crate::ternary::read_spk_proj(path)?;
        let mut take = |k: &str| t.remove(k).with_context(|| format!("missing spk_proj.{k}"));
        let (weight, shape) = take("0.weight")?;
        let (bias, _) = take("0.bias")?;
        let (ln_weight, _) = take("1.weight")?;
        let (ln_bias, _) = take("1.bias")?;
        let [hidden, spk_dim] = shape[..] else {
            anyhow::bail!("spk_proj.0.weight has shape {shape:?}");
        };
        ensure!(bias.len() == hidden && ln_weight.len() == hidden && ln_bias.len() == hidden);
        Ok(Self {
            weight,
            bias,
            ln_weight,
            ln_bias,
            spk_dim,
            hidden,
        })
    }

    pub fn spk_dim(&self) -> usize {
        self.spk_dim
    }

    /// `LayerNorm(W·e + b)` (eps 1e-5, `nn.LayerNorm`'s default).
    pub fn project(&self, emb: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            emb.len() == self.spk_dim,
            "speaker embedding has {} values, spk_proj wants {}",
            emb.len(),
            self.spk_dim
        );
        let mut y: Vec<f32> = self
            .weight
            .chunks_exact(self.spk_dim)
            .zip(&self.bias)
            .map(|(row, b)| row.iter().zip(emb).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        let n = self.hidden as f32;
        let mean = y.iter().sum::<f32>() / n;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ((v, g), b) in y.iter_mut().zip(&self.ln_weight).zip(&self.ln_bias) {
            *v = (*v - mean) * inv * g + b;
        }
        Ok(y)
    }
}

/// Token-embedding rows read straight from the GGUF (the same values the
/// generator's host gather uses).
struct EmbedTable {
    file: GgufFile,
    hidden: usize,
}

impl EmbedTable {
    fn open(gguf: &Path, hidden: usize) -> Result<Self> {
        let file =
            GgufFile::from_path_mmap(gguf).with_context(|| format!("mmap {}", gguf.display()))?;
        ensure!(
            file.get("token_embd.weight").is_some(),
            "no token_embd in {}",
            gguf.display()
        );
        Ok(Self { file, hidden })
    }

    fn rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.file.get("token_embd.weight").expect("checked at open");
        let bytes = self.file.tensor_bytes(t)?;
        let vocab = t.shape[1];
        let row_bytes = bytes.len() / vocab;
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            ensure!(id < vocab, "token {id} outside vocab {vocab}");
            out.extend(rlx_gguf::dequant_typed(
                t.dtype,
                &bytes[id * row_bytes..(id + 1) * row_bytes],
                self.hidden,
                "token_embd.weight",
            )?);
        }
        Ok(out)
    }
}

/// Resolve (transcoding on first use) the `Q2_0` GGUF for `lm_dir`.
pub fn ensure_q2_0_gguf(lm_dir: &Path, ternary: &Path, cfg: &Qwen3Config) -> Result<PathBuf> {
    let out = lm_dir.join(Q2_0_GGUF);
    // Files from before the embedding was kept BF16 stored it F16.
    let current = out.is_file()
        && GgufFile::from_path_mmap(&out)
            .ok()
            .and_then(|f| f.get("token_embd.weight").map(|t| t.dtype))
            == Some(GgmlType::BF16);
    if !current {
        eprintln!(
            "[kittentts2] transcoding {} → {} (one-time, lossless)",
            ternary.display(),
            out.display()
        );
        crate::ternary::transcode(ternary, cfg, &out)?;
    }
    Ok(out)
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    SpeechEnd,
    Stop,
    Budget,
}

/// The loaded speech LM.
pub struct SpeechLm {
    cfg: Qwen3Config,
    generator: Qwen3Generator,
    embed: EmbedTable,
    spk: SpeakerProjection,
}

impl SpeechLm {
    /// `lm_dir` holds `config.json`; `ternary` is `model-ternary.safetensors`.
    pub fn open(lm_dir: &Path, ternary: &Path, device: Device) -> Result<Self> {
        let cfg = Qwen3Config::from_file(&lm_dir.join("config.json"))?;
        let gguf = ensure_q2_0_gguf(lm_dir, ternary, &cfg)?;
        let gguf_s = gguf.to_str().context("non-UTF-8 GGUF path")?;
        if device == Device::Gpu {
            // The f32 tied head is 1.3 GB, over what wgpu can stage on a 2 GiB
            // storage binding (NVIDIA); project through the BF16 embedding
            // with a packed `DequantMatMul` instead. Exact: same values.
            rlx_ir::env::set("RLX_PACKED_LM_HEAD", "1");
            rlx_ir::env::set("RLX_QWEN3_KEEP_BF16", "1");
        }
        let mut generator = Qwen3Generator::new_native_packed_decode(cfg.clone(), gguf_s, device)?;
        // Prompts are a few hundred tokens and differ per chunk: 64-token
        // buckets keep both the compile count and the padded compute small
        // (the default power-of-two ladder pads 297 → 512).
        //
        // Not on wgpu: a bucketed graph picks the last real row with a
        // `Gather`, and any gather pins every output's ancestors in wgpu's
        // memory plan — ~2.7 GiB of unreused activations, past NVIDIA's 2 GiB
        // binding. An exact-length graph narrows instead (one compile per
        // prompt length).
        generator.set_prefill_bucket(Some(if device == Device::Gpu {
            1
        } else {
            PREFILL_BUCKET
        }));
        ensure!(
            generator.enable_host_embed()?,
            "{} token_embd cannot be gathered by row",
            gguf.display()
        );
        Ok(Self {
            embed: EmbedTable::open(&gguf, cfg.hidden_size)?,
            spk: SpeakerProjection::from_ternary_checkpoint(ternary)?,
            cfg,
            generator,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    pub fn speaker_projection(&self) -> &SpeakerProjection {
        &self.spk
    }

    /// Prompt embeddings with position 0 replaced by the projected speaker.
    pub fn prompt_embeddings(&self, prompt: &[u32], speaker: &[f32]) -> Result<Vec<f32>> {
        let mut e = self.embed.rows(prompt)?;
        let h = self.cfg.hidden_size;
        e[..h].copy_from_slice(&self.spk.project(speaker)?);
        Ok(e)
    }

    /// Last-position logits of a speaker-conditioned prefill.
    pub fn prefill_logits(&mut self, prompt: &[u32], speaker: &[f32]) -> Result<Vec<f32>> {
        let e = self.prompt_embeddings(prompt, speaker)?;
        self.generator.prefill_embeds_get_last_logits(prompt, &e)
    }

    /// Logits after the prompt and after each of `forced` (teacher forcing):
    /// `forced.len() + 1` rows. For checking a backend's decode against
    /// another's on identical inputs, and decode against prefill.
    pub fn forced_logits(
        &mut self,
        prompt: &[u32],
        speaker: &[f32],
        forced: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let mut rows = vec![self.prefill_logits(prompt, speaker)?];
        for &t in forced {
            self.generator.push_token(t);
            rows.push(self.generator.step_cached_logits()?);
        }
        Ok(rows)
    }

    /// Generate until `<speech_end>`, `[STOP]` or `max_new` tokens. Returns
    /// every generated id (audio and otherwise) and why it stopped.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        speaker: &[f32],
        tm: &TokenMap,
        opts: &SamplerOpts,
        max_new: usize,
        rng: &mut Rng,
        mut on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, StopReason)> {
        let t0 = std::time::Instant::now();
        let mut logits = self.prefill_logits(prompt, speaker)?;
        if rlx_ir::env::flag("RLX_KITTENTTS2_TIMING") {
            eprintln!(
                "[kittentts2] timing: prefill {} tokens {:.2}s",
                prompt.len(),
                t0.elapsed().as_secs_f32()
            );
        }
        let mut history = prompt.to_vec();
        let mut out = Vec::new();
        let mut timer = DecodeTimer::new();
        for step in 0..max_new {
            apply_penalties(&mut logits, &history, prompt.len(), tm, opts);
            let tok = sample(&logits, opts, rng);
            out.push(tok);
            history.push(tok);
            on_token(tok);
            if tok == tm.speech_end_id {
                return Ok((out, StopReason::SpeechEnd));
            }
            if tok == tm.stop_id {
                return Ok((out, StopReason::Stop));
            }
            if step + 1 == max_new {
                break;
            }
            self.generator.push_token(tok);
            logits = self.generator.step_cached_logits()?;
            timer.step();
        }
        Ok((out, StopReason::Budget))
    }
}

/// Logs decode throughput when `RLX_KITTENTTS2_TIMING` is set.
struct DecodeTimer {
    start: std::time::Instant,
    first: Option<f32>,
    steps: usize,
    on: bool,
}

impl DecodeTimer {
    fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
            first: None,
            steps: 0,
            on: rlx_ir::env::flag("RLX_KITTENTTS2_TIMING"),
        }
    }

    fn step(&mut self) {
        self.steps += 1;
        if self.first.is_none() {
            self.first = Some(self.start.elapsed().as_secs_f32());
        }
    }
}

impl Drop for DecodeTimer {
    fn drop(&mut self) {
        if self.on {
            let total = self.start.elapsed().as_secs_f32();
            let first = self.first.unwrap_or(0.0);
            let rest = self.steps.saturating_sub(1).max(1) as f32;
            eprintln!(
                "[kittentts2] timing: decode {total:.2}s ({} steps; first {first:.2}s, then {:.1} tok/s)",
                self.steps,
                rest / (total - first).max(1e-6)
            );
        }
    }
}
