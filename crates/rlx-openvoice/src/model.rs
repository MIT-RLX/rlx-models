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

//! OpenVoice v2 runner: MeloTTS base (rlx-tiny-tts) + the tone-color converter
//! and extractor, all running **natively on RLX** (no ONNX Runtime). Both
//! `tone_extract.onnx` and `tone_color.onnx` are imported to rlx-ir via
//! `rlx-onnx-import` and compiled per-device through `TinyModel` — so OpenVoice
//! runs on every RLX backend (cpu/metal/mlx/wgpu/coreml/cuda). Zero-shot voice
//! cloning from a reference clip.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::config::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use rlx_tiny_tts::{InferOpts, TinyTts};

use crate::dsp::{SR, Spectrogram, resample};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32((bytes, _dt): &(Vec<u8>, DType)) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A throwaway config for `TinyModel`: only its ONNX dir + graph cache are used
/// here (we call `run_graph` directly, never the MeloTTS synthesize glue that
/// reads these fields), so the values are irrelevant.
fn tone_config() -> BundleConfig {
    BundleConfig {
        model: String::new(),
        sample_rate: SR,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: 192,
        gin_channels: 256,
    }
}

pub struct OpenVoice {
    melotts: TinyTts,
    /// Native runner for `tone_extract.onnx` + `tone_color.onnx` (rlx-ir).
    tone: TinyModel,
    device: Device,
    spec: Spectrogram,
}

impl OpenVoice {
    /// Load MeloTTS from `melotts_dir` (a tiny-tts bundle) and the OpenVoice tone
    /// graphs (`tone_extract.onnx`, `tone_color.onnx`) from `openvoice_dir`.
    /// Everything runs natively on `device` — no ONNX Runtime.
    pub fn load_on(melotts_dir: &Path, openvoice_dir: &Path, device: Device) -> Result<Self> {
        let device = rlx_tiny_tts::resolve_tts_device(device);
        let melotts = TinyTts::load(melotts_dir)
            .with_context(|| format!("load MeloTTS {}", melotts_dir.display()))?;
        anyhow::ensure!(
            openvoice_dir.join("tone_extract.onnx").is_file()
                && openvoice_dir.join("tone_color.onnx").is_file(),
            "missing tone_extract.onnx / tone_color.onnx in {}",
            openvoice_dir.display()
        );
        let tone = TinyModel::new(openvoice_dir.to_path_buf(), tone_config());
        Ok(Self {
            melotts,
            tone,
            device,
            spec: Spectrogram::new(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SR
    }

    /// Backend label (native rlx execution provider) — kept for CLI reporting.
    pub fn ort_ep(&self) -> String {
        format!("rlx-native/{:?}", self.device)
    }

    /// Extract a 256-d tone-color embedding from `audio` (at [`SR`] = 22.05 kHz),
    /// running `tone_extract` natively.
    fn extract_tone(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let (spec, t) = self.spec.magnitude(audio); // [N_FREQ, t] freq-major
        let nfreq = crate::dsp::N_FREQ;
        // tone_extract wants [1, t, N_FREQ] (time-major).
        let mut time_major = vec![0.0f32; t * nfreq];
        for f in 0..nfreq {
            for frame in 0..t {
                time_major[frame * nfreq + f] = spec[f * t + frame];
            }
        }
        if std::env::var("RLX_OV_STAGE").is_ok() {
            eprintln!("[ov] tone_extract t={t} nfreq={nfreq}");
        }
        let out = self.tone.run_graph(
            "tone_extract",
            self.device,
            t,
            &[("input", &f32_bytes(&time_major), DType::F32)],
        )?;
        anyhow::ensure!(!out.is_empty(), "tone_extract returned no output");
        let data = as_f32(&out[0]);
        anyhow::ensure!(
            data.len() >= 256,
            "tone_embedding too short: {}",
            data.len()
        );
        Ok(data[..256].to_vec())
    }

    /// Clone `text` into the voice of `reference` (PCM at `ref_sr`). `tau` controls
    /// the flow sampling temperature (OpenVoice default 0.3). Returns 22.05 kHz PCM.
    /// Runs entirely on RLX (base MeloTTS + tone conversion), no ONNX Runtime.
    pub fn synthesize(
        &self,
        text: &str,
        reference: &[f32],
        ref_sr: u32,
        tau: f32,
    ) -> Result<Vec<f32>> {
        // 1) base MeloTTS audio → 22.05 kHz
        let opts = InferOpts::from_config(self.melotts.config());
        let base = self
            .melotts
            .synthesize(text, &opts)
            .context("MeloTTS base synth")?;
        let base_22k = resample(&base.samples, base.sample_rate, SR);
        let ref_22k = resample(reference, ref_sr, SR);

        // 2) source + target tone-color embeddings (native tone_extract)
        let src_tone = self.extract_tone(&base_22k)?;
        let dest_tone = self.extract_tone(&ref_22k)?;

        // 3) source spectrogram [N_FREQ, t] (freq-major = tone_color's [1, 513, t])
        let (src_spec, t) = self.spec.magnitude(&base_22k);

        // Parity dump (dev): write the exact tone_color inputs so a reference
        // onnxruntime run can be compared against the native output.
        if let Ok(dir) = std::env::var("RLX_OV_DUMP") {
            let d = std::path::Path::new(&dir);
            let _ = std::fs::create_dir_all(d);
            let wf = |name: &str, v: &[f32]| {
                let _ = std::fs::write(d.join(name), f32_bytes(v));
            };
            wf("audio.f32", &src_spec);
            wf("src_tone.f32", &src_tone);
            wf("dest_tone.f32", &dest_tone);
            wf("tau.f32", &[tau]);
            let _ = std::fs::write(d.join("meta.txt"), format!("t={t} tau={tau}"));
        }

        if std::env::var("RLX_OV_STAGE").is_ok() {
            eprintln!("[ov] tone_color t={t}");
        }
        // 4) native tone_color conversion → waveform [1,1,samples]
        let out = self.tone.run_graph(
            "tone_color",
            self.device,
            t,
            &[
                ("audio", &f32_bytes(&src_spec), DType::F32),
                ("audio_length", &i64_bytes(&[t as i64]), DType::I64),
                ("src_tone", &f32_bytes(&src_tone), DType::F32),
                ("dest_tone", &f32_bytes(&dest_tone), DType::F32),
                ("tau", &f32_bytes(&[tau]), DType::F32),
            ],
        )?;
        anyhow::ensure!(!out.is_empty(), "tone_color returned no output");
        let wav = as_f32(&out[0]);
        if let Ok(dir) = std::env::var("RLX_OV_DUMP") {
            let _ = std::fs::write(
                std::path::Path::new(&dir).join("native_out.f32"),
                f32_bytes(&wav),
            );
        }
        Ok(wav)
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

/// Peak absolute amplitude (audibility check).
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
