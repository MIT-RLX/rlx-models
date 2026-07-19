//! Sesame CSM-1B — Llama-3.2-1B backbone + depth decoder → Mimi (24 kHz).
//!
//! Eager CPU AR for the LM; Mimi encode/decode on the requested `--device`.
//! Weights: HF transformers layout (`unsloth/csm-1b` or gated `sesame/csm-1b`).

pub mod cli;
pub mod config;
pub mod depth;
pub mod generate;
pub mod nn;
pub mod session;
pub mod tokenize;
pub mod weights;

pub use config::SesameConfig;
pub use generate::GenerateOpts;
pub use session::{SesameSession, SynthesisResult, load_wav_mono_24k, write_wav};
pub use tokenize::{default_mimi_dir, default_model_dir};

/// Compatibility wrapper used by early stubs / facade callers.
pub struct SesameCSM {
    session: SesameSession,
}

impl SesameCSM {
    pub fn open(
        model_dir: impl AsRef<std::path::Path>,
        device: rlx_runtime::Device,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            session: SesameSession::open_on(model_dir, default_mimi_dir(), device)?,
        })
    }

    pub fn open_defaults(device: rlx_runtime::Device) -> anyhow::Result<Self> {
        Ok(Self {
            session: SesameSession::open_defaults(device)?,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        context_audio: Option<&[f32]>,
    ) -> anyhow::Result<Vec<f32>> {
        let opts = GenerateOpts::default();
        let r = self
            .session
            .synthesize_with_context(text, context_audio, &opts)?;
        Ok(r.samples)
    }

    pub fn synthesize_dialogue(
        &mut self,
        text: &str,
        previous_audio: Option<&[f32]>,
        _speaker_continuity: bool,
    ) -> anyhow::Result<Vec<f32>> {
        self.synthesize(text, previous_audio)
    }

    pub fn device(&self) -> rlx_runtime::Device {
        self.session.device()
    }

    pub fn config(&self) -> &SesameConfig {
        self.session.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let c = SesameConfig::default();
        assert_eq!(c.audio_sample_rate(), 24_000);
        assert_eq!(c.num_hidden_layers, 16);
        assert_eq!(c.num_codebooks, 32);
        assert_eq!(c.frame_width(), 33);
    }
}
