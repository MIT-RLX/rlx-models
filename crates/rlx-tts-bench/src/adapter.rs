//! Thin TTS adapter trait shared by every model backend.

use std::path::PathBuf;

use anyhow::Result;
use rlx_runtime::Device;

#[derive(Debug, Clone)]
pub struct WeightHints {
    pub default_dir: PathBuf,
    pub env_keys: Vec<&'static str>,
    pub marker_files: Vec<&'static str>,
}

impl WeightHints {
    pub fn resolve_dir(&self) -> Option<PathBuf> {
        for key in &self.env_keys {
            if let Ok(v) = std::env::var(key) {
                let p = PathBuf::from(v);
                if p.is_file() || self.dir_ready(&p) {
                    return Some(p);
                }
            }
        }
        if self.default_dir.is_file() || self.dir_ready(&self.default_dir) {
            return Some(self.default_dir.clone());
        }
        None
    }

    pub fn available(&self) -> bool {
        self.resolve_dir().is_some()
    }

    pub fn dir_ready(&self, dir: &std::path::Path) -> bool {
        if dir.is_file() {
            return true;
        }
        if !dir.is_dir() {
            return false;
        }
        if self.marker_files.is_empty() {
            return true;
        }
        self.marker_files.iter().any(|m| dir.join(m).is_file())
    }

    pub fn missing_reason(&self) -> String {
        format!(
            "weights not found (tried env {:?}, default {})",
            self.env_keys,
            self.default_dir.display()
        )
    }
}

#[derive(Debug, Clone)]
pub struct CloneRequest<'a> {
    /// Reference WAV path (mono preferred).
    pub ref_wav: &'a std::path::Path,
    /// Optional transcript of the reference (F5 / LuxTTS).
    pub ref_text: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct SynthRequest<'a> {
    pub text: &'a str,
    pub phrase_id: &'a str,
    pub device: Device,
    pub clone: Option<CloneRequest<'a>>,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct SynthResult {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub wall_ms: f64,
    pub exec_label: String,
}

pub trait TtsAdapter: Send {
    fn id(&self) -> &'static str;
    fn weight_hints(&self) -> WeightHints;
    fn supports_clone(&self) -> bool;
    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult>;
}

/// Factory entry used by `list` / `run` without loading weights.
pub struct AdapterMeta {
    pub id: &'static str,
    pub supports_clone: bool,
    pub hints: WeightHints,
    pub feature: &'static str,
}

pub type AdapterFactory = fn(Device) -> Result<Box<dyn TtsAdapter>>;
