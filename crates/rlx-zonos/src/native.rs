// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native Zonos: espeak → compiled AR (per Device) → DAC 44.1 kHz.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_dac::codec::DacCodec;
use rlx_dac::codes::DacCodes;
use rlx_runtime::Device;

use crate::conditioner::CondOpts;
use crate::config::{
    DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, MASKED_TOKEN_ID, N_CODEBOOKS, SAMPLE_RATE, ZonosFileConfig,
};
use crate::delay::{apply_delay_pattern, revert_delay_pattern};
use crate::engine::{self, BackboneEngine};
use crate::generate::{self, GenerateOpts};
use crate::phonemes;
use crate::weights::WeightMap;

#[derive(Debug, Clone)]
pub struct InferOpts {
    /// AR budget. `None` → [`engine::suggest_max_tokens`] from phoneme length + speaking rate.
    pub max_new_tokens: Option<usize>,
    pub greedy: bool,
    pub seed: u64,
    pub cfg_scale: f32,
    pub speaking_rate: f32,
    pub language: String,
    pub min_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    /// Optional speaker embedding `[128]` (raw LE f32 file via CLI, or in-memory).
    pub speaker: Option<Vec<f32>>,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            // Match Zyphra `generate` default (`min_p` sampling). Greedy is fine
            // for short prompts but truncates / mush-ends long paragraphs.
            greedy: false,
            seed: 1337,
            cfg_scale: 2.0,
            speaking_rate: 15.0,
            language: "en-us".into(),
            min_p: 0.1,
            temperature: 1.0,
            repetition_penalty: 3.0,
            speaker: None,
        }
    }
}

/// Load a 128-d speaker embedding from a little-endian f32 blob (512 bytes).
pub fn load_speaker_emb(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).with_context(|| format!("read speaker emb {}", path.display()))?;
    if bytes.len() != 128 * 4 {
        bail!(
            "speaker emb {}: expected 512 bytes (128×f32 LE), got {}",
            path.display(),
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(128);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub struct NativeZonos {
    dir: PathBuf,
    cfg: ZonosFileConfig,
    weights: WeightMap,
    dac: DacCodec,
    device: Device,
    /// Lazy compile — rebuilt if max_seq budget grows.
    engine: RefCell<Option<BackboneEngine>>,
    engine_upper: RefCell<usize>,
}

impl NativeZonos {
    pub fn open(dir: impl AsRef<Path>, dac_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let cfg = ZonosFileConfig::load(dir.join("config.json"))?;
        cfg.validate()?;
        let weights = WeightMap::load(dir.join("model.safetensors"))
            .with_context(|| format!("load weights under {}", dir.display()))?;
        // DAC decode backend. On CUDA/ROCm the rlx-dac decode path is unvalidated
        // and diverges (backbone is fine — correct on CPU); `RLX_ZONOS_DAC_DEVICE`
        // overrides (cpu|gpu). Localize/repair cuda DAC, then flip the default.
        let dac_device = match std::env::var("RLX_ZONOS_DAC_DEVICE").as_deref() {
            Ok("cpu") => Device::Cpu,
            Ok("gpu") | Ok("device") => device,
            _ => device,
        };
        if dac_device != device {
            eprintln!("[zonos] DAC decode on {dac_device:?} (backbone on {device:?})");
        }
        let dac = DacCodec::open_on(dac_dir.as_ref(), dac_device)
            .with_context(|| format!("open DAC at {}", dac_dir.as_ref().display()))?;
        Ok(Self {
            dir,
            cfg,
            weights,
            dac,
            device,
            engine: RefCell::new(None),
            engine_upper: RefCell::new(0),
        })
    }

    pub fn open_default(device: Device) -> Result<Self> {
        Self::open(DEFAULT_LOCAL_DIR, DEFAULT_DAC_DIR, device)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<i64>> {
        phonemes::phonemize_en(text)
    }

    pub fn decode_codes(&self, codes: &[Vec<i64>]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            codes.len() == N_CODEBOOKS,
            "expected {N_CODEBOOKS} codebooks"
        );
        let u32_codes: Vec<Vec<u32>> = codes
            .iter()
            .map(|row| row.iter().map(|&c| c.max(0) as u32).collect())
            .collect();
        let dac_codes = DacCodes::from_quantizer_layout(u32_codes);
        self.dac.decode_codes(&dac_codes).context("DAC decode")
    }

    pub fn decode_delayed_codes(&self, delayed: &[Vec<i64>]) -> Result<Vec<f32>> {
        let aligned = revert_delay_pattern(delayed);
        self.decode_codes(&aligned)
    }

    pub fn delay_codes(&self, codes: &[Vec<i64>]) -> Vec<Vec<i64>> {
        apply_delay_pattern(codes, MASKED_TOKEN_ID)
    }

    fn ensure_engine(&self, max_new_tokens: usize, phoneme_len: usize) -> Result<()> {
        let prefix_budget = phoneme_len + 8; // phonemes + conditioner scalars
        let need = engine::compile_upper_cap(
            self.device,
            engine::default_max_seq(max_new_tokens, prefix_budget),
        );
        let cur = *self.engine_upper.borrow();
        if self.engine.borrow().is_some() && cur >= need {
            return Ok(());
        }
        // Drop the previous engine before compiling a larger one so we do not
        // hold two CFG×2 decode graphs (Metal/MLX OOM on short→long upgrade).
        *self.engine.borrow_mut() = None;
        *self.engine_upper.borrow_mut() = 0;
        let eng = BackboneEngine::open(&self.cfg, &self.weights, self.device, need)?;
        *self.engine.borrow_mut() = Some(eng);
        *self.engine_upper.borrow_mut() = need;
        Ok(())
    }

    /// Full synthesis on `self.device` (compiled backbone + DAC).
    ///
    /// Set `RLX_ZONOS_EAGER=1` to force the host-f32 reference backbone.
    pub fn synthesize(&self, text: &str, opts: &InferOpts) -> Result<Vec<f32>> {
        let ids = self.encode_text(text)?;
        anyhow::ensure!(!ids.is_empty(), "empty phoneme ids");
        let max_new_tokens = opts
            .max_new_tokens
            .unwrap_or_else(|| engine::suggest_max_tokens(ids.len(), opts.speaking_rate));
        let max_new_tokens = {
            // Keep runtime budget inside the compile upper so AR does not trip
            // "past_seq >= compile upper" after the device cap.
            let prefix_budget = ids.len() + 8;
            let upper = engine::compile_upper_cap(
                self.device,
                engine::default_max_seq(max_new_tokens, prefix_budget),
            );
            max_new_tokens.min(upper.saturating_sub(prefix_budget + 16).max(128))
        };
        if opts.max_new_tokens.is_none() {
            eprintln!(
                "zonos: adaptive max_tokens={max_new_tokens} (phonemes={}, rate={})",
                ids.len(),
                opts.speaking_rate
            );
        }

        let gopts = GenerateOpts {
            max_new_tokens,
            cfg_scale: opts.cfg_scale,
            greedy: opts.greedy,
            seed: opts.seed,
            min_p: opts.min_p,
            temperature: opts.temperature,
            repetition_penalty: opts.repetition_penalty,
            cond: CondOpts {
                language: opts.language.clone(),
                speaking_rate: opts.speaking_rate,
                speaker: opts.speaker.clone(),
                ..CondOpts::default()
            },
        };

        // The compiled backbone diverges on CUDA/ROCm (garbage output; correct on
        // CPU/Metal/MLX) — a not-yet-fixed cuda kernel bug in flow.rs (GptJ RoPE /
        // GQA attention / mask). Fall back to the host-f32 reference there so the
        // output is correct (slower). `RLX_ZONOS_EAGER=1|0` forces on/off.
        let use_eager = match std::env::var("RLX_ZONOS_EAGER").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") => true,
            Ok("0") | Ok("false") | Ok("no") => false,
            _ => matches!(self.device, Device::Cuda | Device::Rocm),
        };
        let codes = if use_eager {
            generate::generate_codes_eager(&self.cfg, &self.weights, &ids, &gopts)
                .context("Zonos eager AR")?
        } else {
            self.ensure_engine(max_new_tokens, ids.len())?;
            let mut eng = self.engine.borrow_mut();
            let eng = eng.as_mut().context("engine missing after ensure")?;
            generate::generate_codes_compiled(&self.cfg, &self.weights, eng, &ids, &gopts)
                .context("Zonos compiled AR")?
        };
        anyhow::ensure!(
            !codes.is_empty() && !codes[0].is_empty(),
            "AR produced empty codes"
        );
        self.decode_codes(&codes)
    }

    pub fn write_wav(audio: &[f32], path: impl AsRef<Path>, sample_rate: u32) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path.as_ref(), spec)
            .with_context(|| format!("create {}", path.as_ref().display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_default_weights_when_present() {
        let dir = PathBuf::from(DEFAULT_LOCAL_DIR);
        if !dir.join("model.safetensors").is_file() {
            return;
        }
        if !PathBuf::from(DEFAULT_DAC_DIR)
            .join("model.safetensors")
            .is_file()
        {
            return;
        }
        let m = NativeZonos::open_default(Device::Cpu).expect("open");
        assert_eq!(m.sample_rate(), SAMPLE_RATE);
        assert_eq!(m.cfg.backbone.n_layer, 26);
        assert!(m.weights.tensors.contains_key("backbone.norm_f.weight"));
    }

    #[test]
    fn suggest_tokens_scales_with_phonemes() {
        let short = engine::suggest_max_tokens(20, 15.0);
        let long = engine::suggest_max_tokens(80, 15.0);
        assert!(long > short);
        assert!(short >= 128);
        // 222 phonemes @ rate 15 needs >12s; must not clamp to 86*12.
        let paragraph = engine::suggest_max_tokens(222, 15.0);
        assert!(
            paragraph > 86 * 12,
            "long text budget {paragraph} still clipped to ~12s"
        );
        let floor = generate::min_frames_before_eos(222, 15.0);
        assert!(
            floor > 86 * 12,
            "EOS floor {floor} should cover a paragraph"
        );
    }
}
