//! TinyTTS English text-to-speech for RLX.
//!
//! TinyTTS (<https://github.com/tronghieuit/tiny-tts>) is a MeloTTS / VITS2-style
//! model with four subgraphs — `text_encoder`, `duration_predictor`, `flow`, and
//! `decoder` — plus a small NumPy "glue" stage (monotonic alignment + latent
//! sampling). Hub packs ship nested `graphs/*.rlxp`; RLX lowers each to rlx-ir
//! HIR and runs on any backend (CPU / Metal / MLX / CUDA / ROCm / wgpu). The
//! glue stage is reimplemented in Rust (see [`glue`]).

pub mod audio;
pub mod config;
pub mod coreml;
pub mod glue;
pub mod model;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use audio::{
    MIN_AUDIBLE_PEAK, MIN_AUDIBLE_SAMPLES, ensure_audible, normalize_audio, peak_amplitude,
    write_wav,
};
pub use config::BundleConfig;
pub use coreml::{ensure_coreml_units_for_tts, resolve_tts_device};
pub use model::{InferOpts, KernelVariant, TinyModel};
pub use rlx_runtime::Device;

/// Re-export the versatile bundle-loading types so callers can build an
/// [`AssetSource`] (directory / packed file / in-memory / config spec / custom
/// provider) without depending on `rlx-core` directly.
pub use rlx_core::asset_source::{self, AssetProvider, AssetSource, LocalDir, SourceSpec};

/// Reuse the byte-identical MeloTTS English frontend from rlx-inflect-nano.
pub use rlx_inflect_nano::frontend;

/// A loaded TinyTTS model: config + four compiled-on-demand subgraphs + frontend.
pub struct TinyTts {
    cfg: BundleConfig,
    dir: PathBuf,
    model: model::TinyModel,
    frontend: std::sync::OnceLock<frontend::English>,
    /// Keeps a materialized temp bundle alive for the model's lifetime when the
    /// source is not directory-backed (pack / in-memory). `None` for real dirs.
    _local: Option<LocalDir>,
}

/// Synthesized waveform.
pub struct Wav {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl TinyTts {
    /// Load a TinyTTS bundle from anywhere: a directory, a packed `.rlxp`
    /// (or legacy `.rlxpack`), an in-memory byte map, or any custom [`AssetSource`].
    ///
    /// ```no_run
    /// # use rlx_tiny_tts::{TinyTts, AssetSource};
    /// let a = TinyTts::load("weights/tiny-tts-rlx")?;         // directory
    /// let b = TinyTts::load("tiny-tts.rlxp")?;                // packed file
    /// let c = TinyTts::load(std::path::Path::new("bundle"))?; // &Path / PathBuf
    /// let bytes: std::sync::Arc<[u8]> = std::fs::read("tiny-tts.rlxp")?.into();
    /// let d = TinyTts::load(AssetSource::pack_bytes(bytes)?)?;// in-memory pack
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// The bundle contains `config.json`, nested `graphs/*.rlxp` (or legacy
    /// local `onnx/*.onnx` for rebuilds), and a `frontend/` asset dir.
    pub fn load(src: impl Into<AssetSource>) -> Result<Self> {
        Self::load_from_source(src.into())
    }

    /// Load from a resolved [`AssetSource`]. Directory sources load in place;
    /// pack / in-memory sources materialize the bundle to a temp directory that
    /// lives as long as the returned model (graph loaders and the g2p frontend
    /// both consume real paths).
    pub fn load_from_source(src: AssetSource) -> Result<Self> {
        let (mut this, keep) = rlx_core::asset_source::load_materialized(src, Self::load_from_dir)
            .context("materialize TinyTTS bundle assets")?;
        this._local = keep; // keep a materialized temp bundle alive (lazy graph/frontend reads)
        Ok(this)
    }

    /// Load a TinyTTS bundle described by a [`SourceSpec`] (typically parsed from
    /// a JSON config): `{"source":"dir","path":"…"}` or
    /// `{"source":"pack","path":"…"}`.
    pub fn load_from_spec(spec: &SourceSpec) -> Result<Self> {
        Self::load_from_source(AssetSource::from_spec(spec)?)
    }

    /// Load an RLX TinyTTS bundle from a filesystem directory: `config.json`,
    /// `graphs/{text_encoder,duration_predictor,flow,decoder}.rlxp` (or legacy
    /// `onnx/*.onnx`) and a `frontend/` asset dir. Prefer [`TinyTts::load`].
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let cfg_s = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let cfg = BundleConfig::from_json(&cfg_s)?;
        let model = model::TinyModel::new(dir.to_path_buf(), cfg.clone());
        Ok(Self {
            cfg,
            dir: dir.to_path_buf(),
            model,
            frontend: std::sync::OnceLock::new(),
            _local: None,
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
