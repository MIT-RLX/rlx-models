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

//! Piper VITS runner (single ONNX): `input` / `input_lengths` / `scales`.

use std::path::Path;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::Device;

use crate::config::{PiperConfig, find_voice};

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0, f32::max)
}

/// A loaded Piper voice.
pub struct Piper {
    device: Device,
    cfg: PiperConfig,
    ort_ep: String,
    #[cfg(feature = "onnx")]
    session: Mutex<Session>,
}

impl Piper {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let (onnx, json) = find_voice(dir)?;
        let cfg = PiperConfig::load(&json)?;
        #[cfg(not(feature = "onnx"))]
        {
            let _ = (device, &onnx, &cfg);
            anyhow::bail!("rlx-piper built without the `onnx` feature");
        }
        #[cfg(feature = "onnx")]
        {
            // Piper's VITS graph crashes ORT's CoreML EP (metal/mlx); the CPU EP
            // is the validated path, so fall back to it for GPU devices.
            let ort_device = if matches!(device, Device::Cpu) {
                device
            } else {
                eprintln!("[piper] CoreML EP is unstable for this VITS model; using CPU EP");
                Device::Cpu
            };
            let built = rlx_kittentts::build_onnx_session(&onnx, ort_device).context("session")?;
            eprintln!(
                "[piper] loaded {} on {ort_device:?} (ep={})",
                onnx.display(),
                built.ort_ep
            );
            Ok(Self {
                device,
                cfg,
                ort_ep: built.ort_ep,
                session: Mutex::new(built.session),
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
    pub fn config(&self) -> &PiperConfig {
        &self.cfg
    }

    /// Synthesize `text`. `length_scale` > 1 slows speech (`None` = config default).
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

    #[cfg(feature = "onnx")]
    fn run(&self, ids: &[i64], length_scale: Option<f32>) -> Result<Vec<f32>> {
        let n = ids.len();
        let scales = vec![
            self.cfg.noise_scale,
            length_scale.unwrap_or(self.cfg.length_scale),
            self.cfg.noise_w,
        ];
        let input = Tensor::<i64>::from_array(([1usize, n], ids.to_vec())).context("input")?;
        let lengths =
            Tensor::<i64>::from_array(([1usize], vec![n as i64])).context("input_lengths")?;
        let scales_t = Tensor::<f32>::from_array(([3usize], scales)).context("scales")?;

        let mut s = self.session.lock().expect("poisoned");
        let out = s
            .run(ort::inputs![input, lengths, scales_t])
            .context("piper inference")?;
        let (_shape, audio) = out[0].try_extract_tensor::<f32>().context("output")?;

        let audio = audio.to_vec();
        let peak = peak_amplitude(&audio);
        anyhow::ensure!(
            peak >= 1e-3,
            "synthesized audio is silent (peak={peak:.2e})"
        );
        Ok(audio)
    }

    #[cfg(not(feature = "onnx"))]
    fn run(&self, _ids: &[i64], _length_scale: Option<f32>) -> Result<Vec<f32>> {
        anyhow::bail!("rlx-piper built without the `onnx` feature")
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
