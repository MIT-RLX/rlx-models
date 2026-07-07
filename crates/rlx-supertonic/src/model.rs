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

//! Supertonic-3 runner: four chained ONNX subgraphs + Rust glue, mirroring the
//! reference `TextToSpeech` (`py/helper.py`):
//!
//! 1. `duration_predictor(text_ids, style_dp, text_mask)` → total duration (s)
//! 2. `text_encoder(text_ids, style_ttl, text_mask)` → text embedding `[1,256,T]`
//! 3. sample `noisy_latent ~ N(0,I)` of `[1,144,L]`, `L = ceil(dur·sr / chunk)`
//! 4. flow-matching ODE loop `total_step×`: `xt = vector_estimator(xt, …, step)`
//!    (the estimator integrates internally; the caller just feeds `xt` back)
//! 5. `vocoder(xt)` → waveform, trimmed to `dur·sr` samples.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::Device;

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};

use crate::config::StConfig;
use crate::tokenize::UnicodeIndexer;
use crate::voices::Voice;

/// Default flow-matching denoising steps (reference default).
pub const DEFAULT_TOTAL_STEP: usize = 8;
/// Default speaking-rate multiplier (reference default; >1 = faster).
pub const DEFAULT_SPEED: f32 = 1.05;

/// Peak amplitude below this is treated as silent (failed) output.
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter().filter(|s| s.is_finite()).map(|s| s.abs()).fold(0.0, f32::max)
}

/// Deterministic Gaussian source (xorshift128+ → Box–Muller).
pub struct Rng {
    s0: u64,
    s1: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self { s0: next() | 1, s1: next() | 1, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        self.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s1.wrapping_add(s0)
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn randn(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let mut u1 = self.next_unit();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_unit();
        }
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

/// Per-call synthesis options.
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub total_step: usize,
    pub speed: f32,
    pub seed: u64,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self { total_step: DEFAULT_TOTAL_STEP, speed: DEFAULT_SPEED, seed: 0 }
    }
}

/// A loaded Supertonic-3 model (four ONNX sessions).
pub struct Supertonic {
    device: Device,
    cfg: StConfig,
    indexer: UnicodeIndexer,
    #[cfg(feature = "onnx")]
    dp: Mutex<Session>,
    #[cfg(feature = "onnx")]
    text_enc: Mutex<Session>,
    #[cfg(feature = "onnx")]
    vector_est: Mutex<Session>,
    #[cfg(feature = "onnx")]
    vocoder: Mutex<Session>,
    ort_ep: String,
}

impl Supertonic {
    /// Load all four subgraphs from `<dir>/onnx/` on CPU.
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    /// Load on a specific device (ONNX Runtime EP with CPU fallback).
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let onnx_dir = if dir.join("onnx").is_dir() { dir.join("onnx") } else { dir.to_path_buf() };
        let cfg = StConfig::load(&onnx_dir)?;
        let indexer = UnicodeIndexer::load(&onnx_dir)?;

        #[cfg(not(feature = "onnx"))]
        {
            let _ = (device, &cfg, &indexer);
            anyhow::bail!("rlx-supertonic built without the `onnx` feature");
        }
        #[cfg(feature = "onnx")]
        {
            let load = |name: &str| -> Result<(Session, String)> {
                let path = onnx_dir.join(name);
                let built = rlx_kittentts::build_onnx_session(&path, device)
                    .with_context(|| format!("build session {name}"))?;
                Ok((built.session, built.ort_ep))
            };
            let (dp, ep) = load("duration_predictor.onnx")?;
            let (text_enc, _) = load("text_encoder.onnx")?;
            let (vector_est, _) = load("vector_estimator.onnx")?;
            let (vocoder, _) = load("vocoder.onnx")?;
            eprintln!("[supertonic] loaded 4 subgraphs on {device:?} (ep={ep})");
            Ok(Self {
                device,
                cfg,
                indexer,
                dp: Mutex::new(dp),
                text_enc: Mutex::new(text_enc),
                vector_est: Mutex::new(vector_est),
                vocoder: Mutex::new(vocoder),
                ort_ep: ep,
            })
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }
    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }
    pub fn ort_ep(&self) -> &str {
        &self.ort_ep
    }

    /// Synthesize `text` (single utterance, no sentence chunking) with `voice`.
    #[cfg(feature = "onnx")]
    pub fn synthesize(&self, text: &str, lang: &str, voice: &Voice, opts: &InferOpts) -> Result<Vec<f32>> {
        let ids = self.indexer.encode(text, lang)?;
        anyhow::ensure!(!ids.is_empty(), "text tokenized to empty sequence");
        let t = ids.len();
        let text_mask = vec![1.0f32; t];

        // 1. duration predictor → total seconds (input order: text_ids, style_dp, text_mask).
        let (_ds, dp_out) = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.dp.rows, voice.dp.cols], voice.dp.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.dp.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a, b, c]).context("duration_predictor")?)?
        };
        let duration = (dp_out[0] / opts.speed).max(0.05);

        // 2. text encoder → [1, 256, T] (order: text_ids, style_ttl, text_mask).
        let (emb_shape, text_emb) = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.text_enc.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a, b, c]).context("text_encoder")?)?
        };
        let emb_shape = shape_usize(&emb_shape);

        // 3. sample noisy latent [1, 144, L] and its (all-ones) mask.
        let l = self.cfg.latent_len(duration);
        let ch = self.cfg.latent_channels();
        let mut rng = Rng::new(opts.seed);
        let mut xt: Vec<f32> = (0..ch * l).map(|_| rng.randn()).collect();
        let latent_mask = vec![1.0f32; l];

        // 4. flow-matching ODE loop (estimator integrates internally). Input order:
        //    noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step.
        let total = opts.total_step.max(1);
        // Parity dump: write the exact ONNX inputs (incl. the sampled noise) so a
        // Python onnxruntime run can be compared bit-for-bit (see tests/parity).
        if let Some(dir) = std::env::var_os("RLX_ST_PARITY_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            dump_i64(&dir.join("ids.i64"), &ids);
            dump_f32(&dir.join("style_dp.f32"), &voice.dp.data);
            dump_f32(&dir.join("style_ttl.f32"), &voice.ttl.data);
            dump_f32(&dir.join("noise.f32"), &xt);
            let meta = format!(
                "{{\"t\":{t},\"l\":{l},\"ch\":{ch},\"duration\":{duration},\"total\":{total},\"dp_rows\":{},\"dp_cols\":{},\"ttl_rows\":{},\"ttl_cols\":{}}}",
                voice.dp.rows, voice.dp.cols, voice.ttl.rows, voice.ttl.cols
            );
            let _ = std::fs::write(dir.join("meta.json"), meta);
        }
        for step in 0..total {
            let nl = f32_t(&[1, ch, l], xt)?;
            let te = f32_t(&emb_shape, text_emb.clone())?;
            let st = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let lm = f32_t(&[1, 1, l], latent_mask.clone())?;
            let tm = f32_t(&[1, 1, t], text_mask.clone())?;
            let cs = f32_t(&[1], vec![step as f32])?;
            let ts = f32_t(&[1], vec![total as f32])?;
            let mut s = self.vector_est.lock().expect("ort poisoned");
            let out = s.run(ort::inputs![nl, te, st, lm, tm, cs, ts]).context("vector_estimator")?;
            xt = extract0(&out)?.1;
        }

        // 5. vocoder → waveform, trim to dur·sr.
        let wav = {
            let a = f32_t(&[1, ch, l], xt)?;
            let mut s = self.vocoder.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a]).context("vocoder")?)?.1
        };
        let n = ((duration * self.cfg.sample_rate as f32) as usize).min(wav.len());
        let audio = wav[..n.max(1)].to_vec();

        if let Some(dir) = std::env::var_os("RLX_ST_PARITY_DUMP") {
            dump_f32(&std::path::PathBuf::from(dir).join("audio_rlx.f32"), &audio);
        }

        let peak = peak_amplitude(&audio);
        anyhow::ensure!(peak >= MIN_AUDIBLE_PEAK, "synthesized audio is silent (peak={peak:.2e})");
        Ok(audio)
    }

    /// Write mono 16-bit PCM WAV at the model sample rate.
    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.cfg.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create WAV: {}", path.display()))?;
        for &s in audio {
            let v = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            w.write_sample(v).context("WAV write")?;
        }
        w.finalize().context("WAV finalize")?;
        Ok(())
    }
}

#[cfg(feature = "onnx")]
fn f32_t(shape: &[usize], data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::<f32>::from_array((shape.to_vec(), data)).context("build f32 tensor")
}

#[cfg(feature = "onnx")]
fn i64_t(shape: &[usize], data: Vec<i64>) -> Result<Tensor<i64>> {
    Tensor::<i64>::from_array((shape.to_vec(), data)).context("build i64 tensor")
}

#[cfg(feature = "onnx")]
fn shape_usize(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d.max(1) as usize).collect()
}

fn dump_f32(path: &Path, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _ = std::fs::write(path, bytes);
}
fn dump_i64(path: &Path, data: &[i64]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _ = std::fs::write(path, bytes);
}

/// Extract the first output of an ORT run as `(shape, f32 data)`.
#[cfg(feature = "onnx")]
fn extract0(outputs: &ort::session::SessionOutputs) -> Result<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().context("extract f32 output")?;
    Ok((shape.to_vec(), data.to_vec()))
}
