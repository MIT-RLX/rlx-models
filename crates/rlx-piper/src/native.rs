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

//! Native (ort-free) piper VITS runner over RLX.
//!
//! Piper's exported graph is a monolithic VITS model whose stochastic duration
//! predictor (`/dp/`) is a rational-quadratic-spline coupling flow with
//! boolean-mask indexing that no static-shape importer can rank. So the model is
//! **graph-split** with the dp reimplemented in Rust ([`crate::sdp`]):
//!
//! ```text
//! enc_p.onnx   [input, input_lengths]
//!            → m_p [1,192,T]  logs_p [1,192,T]  dp_in [1,192,T]
//!   ── Rust StochasticDurationPredictor (sdp.rs): dp_in → durations[T] ──
//!   ── Rust length regulator: repeat m_p/logs_p columns by durations → [1,192,T']
//!   ── z_p = m_p' + noise·exp(logs_p')·noise_scale ──
//! flow_dec.onnx [z_p [1,192,T'], y_mask [1,1,T']] → waveform [1,1,1,T'·hop]
//! ```
//!
//! Both subgraphs import through `rlx-onnx-import` → rlx-ir and run on any RLX
//! backend via rlx-tiny-tts's `compile_named`/`run_typed` harness. The bundle
//! (`enc_p.onnx`, `flow_dec.onnx`, `dp_weights.f32`, `dp_manifest.json`) is
//! produced by `scripts/split_piper.py`.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

use crate::config::{PiperConfig, find_voice};
use crate::model::peak_amplitude;
use crate::sdp::Sdp;

const ENC_P: &str = "enc_p";
const FLOW_DEC: &str = "flow_dec";
const CHANNELS: usize = 192;

/// Subdirectory (beside the voice `.onnx`) holding the native split bundle.
pub const SPLIT_SUBDIR: &str = "rlx-split";

/// Native (ort-free) piper runner: `enc_p` + `flow_dec` subgraphs on an RLX
/// backend, with the stochastic duration predictor + length regulator in Rust.
/// A drop-in for the ort [`crate::Piper`] with the same `synthesize` entry point.
pub struct NativeVits {
    model: TinyModel,
    sdp: Sdp,
    cfg: PiperConfig,
    device: Device,
    /// Duration-critical encoder stays on CPU by default (GPU drift near ceil
    /// boundaries changes frame counts → cos≈0). Override with
    /// `RLX_PIPER_ENC_DEVICE=gpu`.
    enc_device: Device,
}

impl NativeVits {
    /// Load a piper voice directory for native inference on `device`. Expects the
    /// voice `.onnx` + `.onnx.json` plus the split bundle in `<voice_dir>/rlx-split/`
    /// (produced by `scripts/split_piper.py`).
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let device = rlx_tiny_tts::resolve_tts_device(device);
        let (onnx, json) = find_voice(dir)?;
        let cfg = PiperConfig::load(&json)?;
        let split_dir = onnx.parent().unwrap_or(dir).join(SPLIT_SUBDIR);
        for f in [
            "enc_p.onnx",
            "flow_dec.onnx",
            "dp_weights.f32",
            "dp_manifest.json",
        ] {
            anyhow::ensure!(
                split_dir.join(f).is_file(),
                "native piper bundle missing {f} in {} (run scripts/split_piper.py)",
                split_dir.display()
            );
        }
        let sdp = Sdp::load(&split_dir)?;
        let bundle_cfg = BundleConfig {
            model: String::new(),
            sample_rate: cfg.sample_rate,
            add_blank: false,
            language: "EN".into(),
            speakers: Default::default(),
            default_speaker: None,
            noise_scale: cfg.noise_scale,
            noise_scale_w: cfg.noise_w,
            length_scale: cfg.length_scale,
            inter_channels: CHANNELS,
            gin_channels: 0,
        };
        let enc_device = match std::env::var("RLX_PIPER_ENC_DEVICE").as_deref() {
            Ok("gpu") | Ok("device") => device,
            _ => Device::Cpu,
        };
        Ok(Self {
            model: TinyModel::new(split_dir, bundle_cfg),
            sdp,
            cfg,
            device,
            enc_device,
        })
    }

    /// `load(dir, Cpu)`.
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load(dir, Device::Cpu)
    }

    pub fn device(&self) -> Device {
        self.device
    }
    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }
    pub fn config(&self) -> &PiperConfig {
        &self.cfg
    }

    /// Synthesize from text (espeak-ng G2P). `length_scale` > 1 slows speech.
    #[cfg(feature = "espeak")]
    pub fn synthesize(&self, text: &str, length_scale: Option<f32>) -> Result<Vec<f32>> {
        let ids = crate::tokenize::encode(text, &self.cfg)?;
        anyhow::ensure!(ids.len() > 2, "text produced no phonemes");
        self.run(&ids, length_scale)
    }

    /// Synthesize directly from espeak phonemes (skips G2P).
    pub fn synthesize_phonemes(
        &self,
        phonemes: &str,
        length_scale: Option<f32>,
    ) -> Result<Vec<f32>> {
        let ids = crate::tokenize::phonemes_to_ids(phonemes, &self.cfg);
        anyhow::ensure!(ids.len() > 2, "no phonemes");
        self.run(&ids, length_scale)
    }

    /// Full native pipeline: phoneme ids → 24 kHz waveform.
    pub fn run(&self, ids: &[i64], length_scale: Option<f32>) -> Result<Vec<f32>> {
        anyhow::ensure!(!ids.is_empty(), "empty phoneme id sequence");
        let seq = ids.len();
        let length_scale = length_scale.unwrap_or(self.cfg.length_scale);
        // Cross-backend cosine needs a deterministic mean path (no SDP / z noise).
        let deterministic = matches!(
            std::env::var("RLX_PIPER_DETERMINISTIC").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        );
        let noise_w = if deterministic { 0.0 } else { self.cfg.noise_w };
        let noise_scale = if deterministic {
            0.0
        } else {
            self.cfg.noise_scale
        };

        // ── 1. enc_p → m_p / logs_p / dp_in ─────────────────────────────────
        let mut enc = self
            .model
            .compile_named(ENC_P, self.enc_device, seq, &[("phonemes", seq)])
            .context("compile piper enc_p graph")?;
        let enc_out = enc.run_typed(&[
            ("input", &i64_bytes(ids), DType::I64),
            ("input_lengths", &i64_bytes(&[seq as i64]), DType::I64),
        ]);
        anyhow::ensure!(
            enc_out.len() >= 3,
            "enc_p produced {} outputs (need 3)",
            enc_out.len()
        );
        let m_p = as_f32(&enc_out[0].0); // [1,192,seq]
        let logs_p = as_f32(&enc_out[1].0); // [1,192,seq]
        let dp_in = as_f32(&enc_out[2].0); // [1,192,seq]
        anyhow::ensure!(
            m_p.len() == CHANNELS * seq,
            "m_p len {} != 192·{seq}",
            m_p.len()
        );

        // ── 2. Rust stochastic duration predictor → durations ───────────────
        let sdp_noise = gaussian(2 * seq, noise_w, 0x5eed_1234);
        let dur = self.sdp.durations(&dp_in, seq, length_scale, &sdp_noise);
        let frames: usize = dur.iter().sum();
        anyhow::ensure!(frames > 0, "total predicted duration is zero");

        // ── 3. Length regulate + z_p = m_p' + noise·exp(logs_p')·noise_scale ─
        let m_reg = length_regulate(&m_p, CHANNELS, seq, &dur, frames);
        let logs_reg = length_regulate(&logs_p, CHANNELS, seq, &dur, frames);
        let zp_noise = gaussian(CHANNELS * frames, 1.0, 0xabcd_ef01);
        let mut z_p = vec![0.0f32; CHANNELS * frames];
        for i in 0..z_p.len() {
            z_p[i] = m_reg[i] + zp_noise[i] * logs_reg[i].exp() * noise_scale;
        }
        let y_mask = vec![1.0f32; frames]; // [1,1,frames] all-ones

        // ── 4. flow_dec (main flow + HiFi-GAN) → waveform ───────────────────
        let mut dec = self
            .model
            .compile_named(
                FLOW_DEC,
                self.device,
                frames,
                &[("frames", frames), ("batch", 1)],
            )
            .context("compile piper flow_dec graph")?;
        let dec_out = dec.run_typed(&[
            ("/Add_output_0", &f32_bytes(&z_p), DType::F32),
            ("/Cast_2_output_0", &f32_bytes(&y_mask), DType::F32),
        ]);
        anyhow::ensure!(!dec_out.is_empty(), "flow_dec produced no output");
        let audio = as_f32(&dec_out[0].0);
        let peak = peak_amplitude(&audio);
        anyhow::ensure!(
            peak >= 1e-3,
            "synthesized audio is silent (peak={peak:.2e})"
        );
        Ok(audio)
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.cfg.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                .context("wav write")?;
        }
        w.finalize().context("wav finalize")?;
        Ok(())
    }
}

/// Repeat each column of `x [1, c, seq]` (row-major) `dur[i]` times → `[1, c, frames]`.
fn length_regulate(x: &[f32], c: usize, seq: usize, dur: &[usize], frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * frames];
    for ch in 0..c {
        let src = &x[ch * seq..ch * seq + seq];
        let dst = &mut out[ch * frames..ch * frames + frames];
        let mut f = 0;
        for (i, &d) in dur.iter().enumerate() {
            for _ in 0..d {
                dst[f] = src[i];
                f += 1;
            }
        }
    }
    out
}

/// Deterministic N(0, `scale`²) samples via a seeded LCG + Box-Muller. Keeps the
/// native path reproducible and ort-free (matching torch's randn statistically,
/// not bit-for-bit — durations feed a `ceil`, so small noise differences at most
/// shift a rare phoneme by one frame, perceptually identical).
fn gaussian(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0f32; n];
    }
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let u1 = next().max(1e-7);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out[i] = r * (std::f32::consts::TAU * u2).cos() * scale;
        if i + 1 < n {
            out[i + 1] = r * (std::f32::consts::TAU * u2).sin() * scale;
        }
        i += 2;
    }
    out
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
