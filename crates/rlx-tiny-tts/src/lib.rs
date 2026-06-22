//! TinyTTS English text-to-speech for RLX.
//!
//! TinyTTS (<https://github.com/tronghieuit/tiny-tts>) is a MeloTTS / VITS2-style
//! model exported as four ONNX subgraphs — `text_encoder`, `duration_predictor`,
//! `flow` and `decoder` — with a small NumPy "glue" stage (monotonic alignment +
//! latent sampling) connecting them. RLX imports each ONNX graph into the rlx-ir
//! HIR and runs it on any backend (CPU / Metal / MLX / CUDA / ROCm / wgpu); the
//! glue stage is reimplemented in Rust (see [`glue`]).

pub mod audio;
pub mod config;
pub mod glue;
pub mod model;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use config::BundleConfig;
pub use model::{InferOpts, TinyModel};
pub use rlx_runtime::Device;

/// Reuse the byte-identical MeloTTS English frontend from rlx-inflect-nano.
pub use rlx_inflect_nano::frontend;

/// A loaded TinyTTS model: config + four compiled-on-demand ONNX graphs + frontend.
pub struct TinyTts {
    cfg: BundleConfig,
    dir: PathBuf,
    model: model::TinyModel,
    frontend: std::sync::OnceLock<frontend::English>,
}

/// Synthesized waveform.
pub struct Wav {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl TinyTts {
    /// Load an RLX TinyTTS bundle (see `scripts/export_tiny_tts.py`):
    /// `config.json`, `onnx/{text_encoder,duration_predictor,flow,decoder}.onnx`
    /// and a `frontend/` asset dir.
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let cfg_s = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let cfg = BundleConfig::from_json(&cfg_s)?;
        let model = model::TinyModel::new(dir.join("onnx"), cfg.clone());
        Ok(Self {
            cfg,
            dir: dir.to_path_buf(),
            model,
            frontend: std::sync::OnceLock::new(),
        })
    }

    pub fn config(&self) -> &BundleConfig {
        &self.cfg
    }

    /// Lazily load the English text frontend (CMUdict + g2p_en + tagger + BERT).
    pub fn frontend(&self) -> Result<&frontend::English> {
        if let Some(f) = self.frontend.get() {
            return Ok(f);
        }
        let f = frontend::English::load(&self.dir.join("frontend"))?;
        Ok(self.frontend.get_or_init(|| f))
    }

    /// Raw text → `(phone_ids, tone_ids, lang_ids)` with blank insertion.
    pub fn text_to_ids(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        self.frontend()?.text_to_ids(text, self.cfg.add_blank)
    }

    /// Full pipeline: raw text → waveform, running every graph on `device`.
    pub fn synthesize_on(&self, text: &str, device: Device, opts: &InferOpts) -> Result<Wav> {
        let (phone, tone, lang) = self.text_to_ids(text)?;
        let speaker = self.cfg.default_speaker();
        let samples = self
            .model
            .synthesize(device, &phone, &tone, &lang, speaker, opts)?;
        Ok(Wav {
            samples,
            sample_rate: self.cfg.sample_rate,
        })
    }

    /// Convenience: synthesize on the CPU backend.
    pub fn synthesize(&self, text: &str, opts: &InferOpts) -> Result<Wav> {
        self.synthesize_on(text, Device::Cpu, opts)
    }

    /// Best available accelerator (Metal → MLX → wgpu), else CPU.
    pub fn preferred_device() -> Device {
        [Device::Metal, Device::Mlx, Device::Gpu]
            .into_iter()
            .find(|&d| rlx_runtime::is_available(d))
            .unwrap_or(Device::Cpu)
    }
}
