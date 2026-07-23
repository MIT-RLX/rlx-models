//! StyleTTS2-family TTS for RLX — Kokoro-82M.
//!
//! Runs the ort-free graph-split RLX path ([`NativeKokoro`]): the decoder runs
//! on the requested device and the encoder on CPU (set `RLX_KOKORO_ENC_DEVICE=gpu`
//! to move it, `RLX_KOKORO_NATIVE_ENC` is a no-op — the encoder is always native).
//!
//! ```bash
//! just styletts2
//! just styletts2-whisper
//! just styletts2-backends
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rlx_kokoro::{NativeKokoro, SAMPLE_RATE, write_wav};
use serde::{Deserialize, Serialize};

pub use rlx_kokoro::{SAMPLE_RATE as STYLETTS2_SAMPLE_RATE, peak_amplitude, resolve_native_device};
pub use rlx_runtime::{Device, is_available, parse_device};

/// Default Kokoro-82M bundle used for StyleTTS2-family synthesis.
pub fn default_model_dir() -> PathBuf {
    PathBuf::from("weights/tts/kokoro-82m")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTTS2Config {
    pub audio_sample_rate: u32,
    /// Named voice pack (Kokoro `.bin` under `voices/`), not raw-wav cloning.
    pub default_voice: String,
    pub default_speed: f32,
}

impl Default for StyleTTS2Config {
    fn default() -> Self {
        Self {
            audio_sample_rate: SAMPLE_RATE,
            default_voice: "af_heart".into(),
            default_speed: 1.0,
        }
    }
}

/// StyleTTS2-family session backed by Kokoro-82M (native graph-split RLX path).
pub struct StyleTTS2 {
    inner: NativeKokoro,
    config: StyleTTS2Config,
    requested: Device,
}

impl StyleTTS2 {
    /// Load Kokoro via the native graph-split RLX path.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        Self::load_native(model_dir, device)
    }

    /// Graph-split RLX path (see [`NativeKokoro`]).
    pub fn load_native(model_dir: &Path, device: Device) -> Result<Self> {
        let inner = NativeKokoro::load(model_dir, device)?;
        Ok(Self {
            inner,
            config: StyleTTS2Config::default(),
            requested: device,
        })
    }

    pub fn load_default(device: Device) -> Result<Self> {
        Self::load(&default_model_dir(), device)
    }

    /// Always `"native"` (the RLX graph-split path).
    pub fn path(&self) -> &'static str {
        "native"
    }

    pub fn generate(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        self.inner.generate_from_text(text, voice, speed)
    }

    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        self.generate(text, &self.config.default_voice, self.config.default_speed)
    }

    pub fn clone_voice(&self, _reference_audio: &[f32], _text: &str) -> Result<Vec<f32>> {
        bail!(
            "rlx-styletts2 uses Kokoro voice packs (not raw-wav cloning). \
             Pass a voice name to StyleTTS2::generate (e.g. af_heart). Available: {:?}",
            self.voice_names()
        )
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.inner.voice_names()
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn device(&self) -> Device {
        self.inner.device()
    }

    pub fn requested_device(&self) -> Device {
        self.requested
    }

    pub fn config(&self) -> &StyleTTS2Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut StyleTTS2Config {
        &mut self.config
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        write_wav(audio, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = StyleTTS2Config::default();
        assert_eq!(cfg.audio_sample_rate, 24_000);
        assert_eq!(cfg.default_voice, "af_heart");
    }
}
