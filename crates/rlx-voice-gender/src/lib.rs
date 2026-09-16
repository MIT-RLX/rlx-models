// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! On-device speech gender (ACF F0 default; optional ECAPA ONNX).

pub use rlx_f0::{SpeechGender, estimate_f0_hz, gender_from_f0};

/// Result of a gender estimate.
#[derive(Debug, Clone, Copy)]
pub struct GenderEstimate {
    pub gender: SpeechGender,
    pub f0_hz: Option<f32>,
    /// Neural male/female scores when ONNX path is used.
    pub male_score: Option<f32>,
    pub female_score: Option<f32>,
}

/// How to estimate gender from PCM.
#[derive(Debug, Clone)]
pub enum GenderEstimator {
    /// Autocorrelation F0 + band thresholds (iOS default).
    Acf { female_f0_hz: f32, male_f0_hz: f32 },
}

impl GenderEstimator {
    pub fn acf_default() -> Self {
        Self::Acf {
            female_f0_hz: 165.0,
            male_f0_hz: 145.0,
        }
    }

    pub fn estimate(&self, pcm: &[f32], sample_rate: u32) -> GenderEstimate {
        match self {
            Self::Acf {
                female_f0_hz,
                male_f0_hz,
            } => {
                let f0 = estimate_f0_hz(pcm, sample_rate);
                let gender = f0
                    .map(|f| gender_from_f0(f, *female_f0_hz, *male_f0_hz))
                    .unwrap_or(SpeechGender::Unknown);
                GenderEstimate {
                    gender,
                    f0_hz: f0,
                    male_score: None,
                    female_score: None,
                }
            }
        }
    }
}

#[cfg(feature = "onnx")]
pub mod onnx {
    //! ECAPA gender ONNX via `TinyModel` (optional).
    use super::*;
    use anyhow::{Context, Result, bail};
    use rlx_runtime::{DType, Device};
    use rlx_tiny_tts::config::BundleConfig;
    use rlx_tiny_tts::model::TinyModel;
    use std::path::{Path, PathBuf};

    /// JaesungHuh-style binary gender ONNX (input: raw 16 kHz mono float).
    pub struct OnnxGenderClassifier {
        model: TinyModel,
        device: Device,
        graph: String,
    }

    impl OnnxGenderClassifier {
        /// Load a single `.onnx` (or `.rlxp`) gender graph from `path`.
        pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
            let path = path.as_ref();
            let dir = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("gender onnx filename")?
                .to_string();
            let cfg = BundleConfig {
                model: String::new(),
                sample_rate: 16_000,
                add_blank: false,
                language: "EN".into(),
                speakers: Default::default(),
                default_speaker: None,
                noise_scale: 0.0,
                noise_scale_w: 0.0,
                length_scale: 1.0,
                inter_channels: 0,
                gin_channels: 0,
            };
            Ok(Self {
                model: TinyModel::new(dir, cfg),
                device,
                graph: stem,
            })
        }

        /// Run classification on 16 kHz mono PCM (model must embed mel / accept raw).
        pub fn predict(&mut self, pcm: &[f32]) -> Result<(f32, f32)> {
            if pcm.is_empty() {
                bail!("empty pcm");
            }
            let bytes: Vec<u8> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
            let outs = self
                .run_once("input", &bytes, pcm.len())
                .or_else(|_| self.run_once("audio", &bytes, pcm.len()))
                .or_else(|_| self.run_once("waveform", &bytes, pcm.len()))?;
            let logits = outs
                .first()
                .map(|(b, _)| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if logits.len() < 2 {
                bail!("gender onnx expected 2 logits, got {}", logits.len());
            }
            // Softmax → (male, female) assuming label order {0: male, 1: female}.
            let m = logits[0];
            let f = logits[1];
            let max = m.max(f);
            let em = (m - max).exp();
            let ef = (f - max).exp();
            let z = em + ef;
            Ok((em / z, ef / z))
        }

        fn run_once(
            &self,
            input_name: &str,
            bytes: &[u8],
            length: usize,
        ) -> Result<Vec<(Vec<u8>, DType)>> {
            self.model
                .run_named(
                    &self.graph,
                    self.device,
                    length,
                    &[],
                    &[(input_name, bytes, DType::F32)],
                )
                .with_context(|| format!("gender onnx run ({input_name})"))
        }

        pub fn estimate(&mut self, pcm: &[f32]) -> Result<GenderEstimate> {
            let (male, female) = self.predict(pcm)?;
            let gender = if female >= male {
                SpeechGender::Female
            } else {
                SpeechGender::Male
            };
            Ok(GenderEstimate {
                gender,
                f0_hz: estimate_f0_hz(pcm, 16_000),
                male_score: Some(male),
                female_score: Some(female),
            })
        }
    }

    /// Discover `gender.onnx` / `model_quantized.onnx` under `dir`.
    pub fn find_gender_onnx(dir: &Path) -> Option<PathBuf> {
        for name in [
            "model_quantized.onnx",
            "gender.onnx",
            "voice-gender.onnx",
            "model.onnx",
        ] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
            let p2 = dir.join("onnx").join(name);
            if p2.is_file() {
                return Some(p2);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acf_estimator_runs() {
        let sr = 16_000u32;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 120.0 * i as f32 / sr as f32).sin() * 0.4)
            .collect();
        let g = GenderEstimator::acf_default().estimate(&pcm, sr);
        assert_eq!(g.gender, SpeechGender::Male);
    }
}
