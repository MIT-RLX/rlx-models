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

//! Maya1 — Maya Research's 3B expressive voice-design TTS for RLX (Apache-2.0).
//!
//! Maya1 is Orpheus-family: a Llama-3.2-3B `LlamaForCausalLM` that emits SNAC
//! 24 kHz codec tokens, with a **byte-identical SNAC token layout** to Orpheus
//! (`CODE_START=128257`, `CODE_END=128258`, SNAC offset `128266`, 7 tokens/frame,
//! `id = 128266 + slot*4096 + code`). The prompt turn tokens also match Orpheus
//! (`SOH=128259`, `BOS=128000`, `[EOT,EOH,SOA,SOS]=[128009,128260,128261,128257]`).
//!
//! So this crate reuses [`rlx_orpheus`]'s GGUF backbone + SNAC decoder +
//! [`rlx_orpheus::build_prompt_ids`] wholesale; the only difference is Maya1's
//! **body format**: `<description="<voice design>"> <text>`. Voice is controlled
//! by a natural-language description (age/gender/accent/pitch/emotion) rather
//! than a preset; inline emotion tags like `<laugh>`, `<whisper>` are supported.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_orpheus::{GenerationConfig, OrpheusTts, build_prompt_ids};
pub use rlx_runtime::{Device, parse_device};

/// Prebuilt GGUF quants (Apache-2.0). Also needs the SNAC decoder
/// (`hubertsiuzdak/snac_24khz`), fetched/resolved by rlx-orpheus.
pub const DEFAULT_HF_REPO: &str = "mradermacher/maya1-GGUF";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/maya1";
pub const SAMPLE_RATE: u32 = 24000;

/// Maya1's generation defaults (README: temperature 0.4, top_p 0.9).
pub fn maya1_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: 2048,
        temperature: 0.4,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.1,
        seed: 0,
        greedy: false,
    }
}

pub struct Maya1 {
    orpheus: OrpheusTts,
}

impl Maya1 {
    /// Load the Maya1 GGUF backbone + SNAC decoder (resolved via rlx-orpheus's
    /// env/default decoder path). `gguf` is a `maya1.*.gguf` file.
    pub fn load_on(gguf: &Path, device: Device) -> Result<Self> {
        let mut orpheus = OrpheusTts::load_with_env_decoder_on(gguf, device)
            .with_context(|| format!("load Maya1 backbone {}", gguf.display()))?;
        orpheus.config = maya1_config();
        Ok(Self { orpheus })
    }

    /// Load with an explicit SNAC decoder path.
    pub fn load_with_snac(gguf: &Path, snac: &Path, device: Device) -> Result<Self> {
        let mut orpheus = OrpheusTts::load_on(gguf, snac, device).context("load Maya1")?;
        orpheus.config = maya1_config();
        Ok(Self { orpheus })
    }

    /// Synthesize `text` in the voice given by a natural-language `description`.
    /// Returns 24 kHz mono PCM.
    pub fn synthesize(&self, description: &str, text: &str) -> Result<Vec<f32>> {
        let body = format!("<description=\"{description}\"> {text}");
        let prompt_ids = build_prompt_ids(self.orpheus.backbone.weights_path(), &body)
            .context("build Maya1 prompt")?;
        Ok(self.orpheus.synthesize_from_prompt_ids(&prompt_ids)?.samples)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Mutable access to sampling config (temperature, top_p, seed, …).
    pub fn config_mut(&mut self) -> &mut GenerationConfig {
        &mut self.orpheus.config
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("create {}", path.display()))?;
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
