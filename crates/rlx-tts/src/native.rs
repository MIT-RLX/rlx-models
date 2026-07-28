//! Native RLX FastSpeech2 + WaveRNN over a private local bundle.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::fastspeech2::{FastSpeech2, VarianceControls};
use crate::frontend::{HydraLite, TextFrontend, parse_phone_string};
use crate::metrics::{apply_leading_silence_ms, apply_output_volume, apply_wavernn_mulaw_iir};
use crate::wavernn::{WaveRnn, WaveRnnOpts};
use crate::weights::Weights;
use crate::{AudioOutput, DEFAULT_VOICE_ID};

pub const DEFAULT_BUNDLE_DIR: &str = "weights/tts/rlx-tts";
/// Hugging Face model id for the packed bundle (`rlx-tts.rlxp`).
pub const HF_REPO: &str = "eugenehp/rlx-tts";
pub const BUNDLE_EXTRACT_HINT: &str = "\
RLX TTS bundle not found. Place weights under weights/tts/rlx-tts/ (or set RLX_TTS_BUNDLE):
  just fetch-rlx-tts          # → weights/tts/rlx-tts/rlx-tts.rlxp (eugenehp/rlx-tts)
  rlx-tts.rlxp               # preferred single-file pack
  rlx-tts.gguf               # legacy GGUF pack
  — or loose manifest.json + encoder/decoder/wavernn.safetensors + frontend/";

#[derive(Debug, Clone, Deserialize)]
pub struct BundleManifest {
    pub format: String,
    pub voice_identifier: String,
    pub sample_rate_hz: u32,
    pub mel_bins: u32,
    pub hop_length: u32,
    pub phone_vocab: u32,
    pub files: BundleFiles,
    #[serde(default)]
    pub predictor_biases: PredictorBiases,
    #[serde(default)]
    pub source_asset_dir: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PredictorBiases {
    #[serde(default)]
    pub duration: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub energy: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleFiles {
    pub encoder: String,
    pub decoder: String,
    pub wavernn: String,
}

#[derive(Debug, Clone)]
pub(crate) struct OutputPost {
    volume_global: f32,
    volume_peak_ratio: f32,
    volume_smoothing: usize,
    /// Leading silence in milliseconds (from bundle post config).
    leading_silence_ms: u32,
    /// Mel frames skipped at WaveRNN input.
    wr_leading_silence_frames: usize,
    /// Samples trimmed from WaveRNN PCM tail.
    wr_trailing_trim_samples: usize,
    /// μ-law+IIR alpha. `None` skips the post.
    wr_mulaw_iir_alpha: Option<f32>,
}

impl Default for OutputPost {
    fn default() -> Self {
        Self {
            volume_global: 0.8,
            volume_peak_ratio: 0.7,
            volume_smoothing: 120,
            leading_silence_ms: 50,
            wr_leading_silence_frames: 12,
            wr_trailing_trim_samples: 2860,
            wr_mulaw_iir_alpha: Some(0.86),
        }
    }
}

pub(crate) fn load_output_post(bundle_dir: &Path) -> OutputPost {
    let mut post = OutputPost::default();
    let path = ["post.cfg", "pipeline.cfg"]
        .iter()
        .map(|n| bundle_dir.join(n))
        .find(|p| p.is_file());
    let Some(path) = path else {
        return post;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return post;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return post;
    };
    let Some(pipeline) = v.get("pipeline").and_then(|p| p.as_array()) else {
        return post;
    };
    for stage in pipeline {
        let id = stage.get("id").and_then(|x| x.as_str()).unwrap_or("");
        let params = stage
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        match id {
            "waveform_volume_change" => {
                if let Some(g) = params.get("global").and_then(|x| x.as_f64()) {
                    post.volume_global = g as f32;
                }
                if let Some(p) = params.get("peak_ratio").and_then(|x| x.as_f64()) {
                    post.volume_peak_ratio = p as f32;
                }
                if let Some(w) = params.get("smoothing_window").and_then(|x| x.as_u64()) {
                    post.volume_smoothing = w as usize;
                }
            }
            "audio_resampling" => {
                if let Some(ms) = params.get("leading_silence").and_then(|x| x.as_u64()) {
                    post.leading_silence_ms = ms as u32;
                }
            }
            _ => {}
        }
    }
    post
}

/// Loaded RLX TTS model (FastSpeech2 + WaveRNN).
pub struct RlxTts {
    dir: PathBuf,
    manifest: BundleManifest,
    encoder: Weights,
    decoder: Weights,
    wavernn: Weights,
    frontend: HydraLite,
    post: OutputPost,
}

impl RlxTts {
    pub(crate) fn from_parts(
        dir: PathBuf,
        manifest: BundleManifest,
        encoder: Weights,
        decoder: Weights,
        wavernn: Weights,
        frontend: HydraLite,
        post: OutputPost,
    ) -> Self {
        Self {
            dir,
            manifest,
            encoder,
            decoder,
            wavernn,
            frontend,
            post,
        }
    }

    /// Open a directory bundle or a packed `rlx-tts.gguf`.
    pub fn open(bundle: impl AsRef<Path>) -> Result<Self> {
        crate::gguf_bundle::open_path(bundle.as_ref())
    }

    /// Directory bundle with `manifest.json` + safetensors on disk.
    pub(crate) fn open_dir_bundle(bundle_dir: &Path) -> Result<Self> {
        let dir = bundle_dir.to_path_buf();
        let man_path = dir.join("manifest.json");
        if !man_path.is_file() {
            bail!("{BUNDLE_EXTRACT_HINT}\n(missing {})", man_path.display());
        }
        let manifest: BundleManifest = serde_json::from_str(
            &std::fs::read_to_string(&man_path)
                .with_context(|| format!("read {}", man_path.display()))?,
        )?;
        let _ = &manifest.format;
        let encoder = Weights::load(&dir.join(&manifest.files.encoder))?;
        let decoder = Weights::load(&dir.join(&manifest.files.decoder))?;
        let (encoder, decoder) = if std::env::var_os("RLX_FS2_F16_PARAMS").is_some() {
            let mut encoder = encoder;
            let mut decoder = decoder;
            encoder.f16_round_params();
            decoder.f16_round_params();
            (encoder, decoder)
        } else {
            (encoder, decoder)
        };
        let wavernn = Weights::load(&dir.join(&manifest.files.wavernn))?;
        let frontend = HydraLite::open(&dir)?;
        let post = load_output_post(&dir);
        Ok(Self::from_parts(
            dir, manifest, encoder, decoder, wavernn, frontend, post,
        ))
    }

    /// Resolve default bundle: `RLX_TTS_BUNDLE`, then `weights/tts/rlx-tts`,
    /// then a few relative fallbacks (including legacy `.cache/rlx-tts`).
    pub fn open_default() -> Result<Self> {
        let mut candidates = Vec::new();
        if let Ok(d) = std::env::var("RLX_TTS_BUNDLE") {
            candidates.push(PathBuf::from(d));
        }
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        candidates.push(PathBuf::from(DEFAULT_BUNDLE_DIR));
        candidates.push(manifest_dir.join("../../weights/tts/rlx-tts"));
        candidates.push(PathBuf::from(".cache/rlx-tts"));
        candidates.push(manifest_dir.join("../../.cache/rlx-tts"));
        for c in &candidates {
            if let Ok(m) = crate::gguf_bundle::open_path(c) {
                return Ok(m);
            }
        }
        bail!("{BUNDLE_EXTRACT_HINT}")
    }

    pub fn bundle_dir(&self) -> &Path {
        &self.dir
    }

    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    pub fn sample_rate(&self) -> u32 {
        self.manifest.sample_rate_hz
    }

    pub fn synthesize_phones(
        &self,
        phone_ids: &[usize],
        ctrl: &VarianceControls,
        vocoder: &WaveRnnOpts,
    ) -> Result<AudioOutput> {
        let mel = self.infer_mel(phone_ids, ctrl)?;
        let wr = WaveRnn::new(&self.wavernn);
        let skip = self
            .post
            .wr_leading_silence_frames
            .min(mel.nrows().saturating_sub(1));
        let mut pcm = if skip > 0 {
            let view = mel.slice(ndarray::s![skip.., ..]);
            wr.infer(&view.to_owned(), vocoder)?
        } else {
            wr.infer(&mel, vocoder)?
        };
        drop(mel);
        if let Some(alpha) = self.post.wr_mulaw_iir_alpha {
            if std::env::var_os("RLX_TTS_NO_MULAW_IIR").is_none() {
                apply_wavernn_mulaw_iir(&mut pcm, alpha);
            }
        }
        let trim = self.post.wr_trailing_trim_samples;
        if trim > 0 && pcm.len() > trim {
            pcm.truncate(pcm.len() - trim);
        }
        if std::env::var_os("RLX_TTS_NO_VOLUME").is_none() {
            apply_output_volume(
                &mut pcm,
                self.post.volume_global,
                self.post.volume_peak_ratio,
                self.post.volume_smoothing,
            );
        }
        let pad_ms = if std::env::var_os("RLX_TTS_NO_LEADING_SILENCE").is_some()
            || (skip > 0 && std::env::var_os("RLX_TTS_FORCE_LEADING_SILENCE").is_none())
        {
            0
        } else {
            self.post.leading_silence_ms
        };
        apply_leading_silence_ms(&mut pcm, pad_ms, self.manifest.sample_rate_hz);
        Ok(AudioOutput {
            pcm,
            sample_rate: self.manifest.sample_rate_hz,
            channels: 1,
            voice_identifier: DEFAULT_VOICE_ID.to_string(),
        })
    }

    pub fn synthesize_phone_string(
        &self,
        phones: &str,
        ctrl: &VarianceControls,
        vocoder: &WaveRnnOpts,
    ) -> Result<AudioOutput> {
        let ids = parse_phone_string(phones, &self.frontend.map)?;
        self.synthesize_phones(&ids, ctrl, vocoder)
    }

    pub fn synthesize_text(
        &self,
        text: &str,
        ctrl: &VarianceControls,
        vocoder: &WaveRnnOpts,
    ) -> Result<AudioOutput> {
        let phones = self.frontend.text_to_phones(text)?;
        let ids = self.frontend.phones_to_ids(&phones)?;
        let mut ctrl = ctrl.clone();
        if ctrl.pause_min_frames == 0 {
            let ms = self.frontend.pause_min_duration_ms.max(0.0);
            ctrl.pause_min_frames = ((ms / 10.0).ceil() as usize).max(1);
        }
        self.synthesize_phones(&ids, &ctrl, vocoder)
    }

    pub fn infer_mel(
        &self,
        phone_ids: &[usize],
        ctrl: &VarianceControls,
    ) -> Result<ndarray::Array2<f32>> {
        FastSpeech2::new(&self.encoder, &self.decoder).infer(phone_ids, ctrl)
    }

    pub fn frontend(&self) -> &HydraLite {
        &self.frontend
    }

    pub fn encoder_weights(&self) -> &Weights {
        &self.encoder
    }

    pub fn decoder_weights(&self) -> &Weights {
        &self.decoder
    }

    pub fn wavernn_weights(&self) -> &Weights {
        &self.wavernn
    }
}

pub(crate) fn open_dir_bundle(path: &Path) -> Result<RlxTts> {
    RlxTts::open_dir_bundle(path)
}
