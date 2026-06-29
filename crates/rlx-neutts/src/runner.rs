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

//! High-level NeuTTS runner — backbone + NeuCodec decoder.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::Device;

use crate::decoder::NeuCodecDecoder;
use crate::tokens;

#[cfg(feature = "llama")]
use crate::backbone::{BackboneModel, DEFAULT_N_CTX};

/// Best execution device for the convenience `load` / `load_with_decoder`
/// entry points: honour `RLX_DEVICE`, else the fastest compiled-in GPU backend
/// that's actually available, else CPU. This makes NeuTTS GPU-accelerated by
/// default instead of silently running the backbone on a single CPU core.
#[cfg(feature = "llama")]
fn best_device() -> Device {
    use rlx_runtime::device_ext::is_available;

    if let Ok(name) = std::env::var("RLX_DEVICE") {
        if let Some(d) = parse_device_name(&name) {
            if matches!(d, Device::Cpu) || is_available(d) {
                return d;
            }
        }
    }
    [
        Device::Metal,
        Device::Mlx,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
    ]
    .into_iter()
    .find(|d| is_available(*d))
    .unwrap_or(Device::Cpu)
}

#[cfg(feature = "llama")]
fn parse_device_name(s: &str) -> Option<Device> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(Device::Cpu),
        "metal" => Some(Device::Metal),
        "mlx" => Some(Device::Mlx),
        "cuda" => Some(Device::Cuda),
        "rocm" => Some(Device::Rocm),
        "gpu" | "vulkan" | "wgpu" => Some(Device::Gpu),
        _ => None,
    }
}

/// Generation hyper-parameters for the GGUF backbone.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
        }
    }
}

/// NeuTTS handle: GGUF backbone (optional) + NeuCodec decoder.
pub struct NeuTTS {
    #[cfg(feature = "llama")]
    pub backbone: BackboneModel,
    pub codec: NeuCodecDecoder,
    pub language: String,
    pub config: GenerationConfig,
}

impl NeuTTS {
    #[cfg(feature = "llama")]
    pub fn load_with_decoder(
        backbone_path: &Path,
        decoder_path: &Path,
        language: &str,
    ) -> Result<Self> {
        Self::load_with_decoder_on(backbone_path, decoder_path, language, best_device())
    }

    #[cfg(feature = "llama")]
    pub fn load_with_decoder_on(
        backbone_path: &Path,
        decoder_path: &Path,
        language: &str,
        device: Device,
    ) -> Result<Self> {
        eprintln!("[neutts] Loading backbone: {}", backbone_path.display());
        let backbone = BackboneModel::load_on(backbone_path, DEFAULT_N_CTX, device)
            .context("Failed to load backbone")?;

        eprintln!(
            "[neutts] Loading NeuCodec decoder: {}",
            decoder_path.display()
        );
        let codec = NeuCodecDecoder::from_file(decoder_path).with_context(|| {
            format!(
                "Failed to load NeuCodec decoder from {}",
                decoder_path.display()
            )
        })?;

        Ok(Self {
            backbone,
            codec,
            language: language.to_string(),
            config: GenerationConfig::default(),
        })
    }

    #[cfg(feature = "llama")]
    pub fn load(backbone_path: &Path, language: &str) -> Result<Self> {
        Self::load_on(backbone_path, language, best_device())
    }

    #[cfg(feature = "llama")]
    pub fn load_on(backbone_path: &Path, language: &str, device: Device) -> Result<Self> {
        let decoder_path = crate::decoder::decoder_weights_path()?;
        Self::load_with_decoder_on(backbone_path, &decoder_path, language, device)
    }

    #[cfg(not(feature = "llama"))]
    pub fn load_codec_only() -> Result<Self> {
        let codec = NeuCodecDecoder::new().context("Failed to initialise NeuCodec decoder")?;
        Ok(Self {
            codec,
            language: "en-us".to_string(),
            config: GenerationConfig::default(),
        })
    }

    #[cfg(feature = "llama")]
    pub fn infer_from_ipa(
        &self,
        input_ipa: &str,
        ref_codes: &[i32],
        ref_ipa: &str,
    ) -> Result<Vec<f32>> {
        let prompt = tokens::build_prompt(ref_ipa, input_ipa, ref_codes);
        let generated = self
            .backbone
            .generate(&prompt, self.config.max_new_tokens)
            .context("Backbone generation failed")?;

        let speech_ids = tokens::extract_ids(&generated);
        if speech_ids.is_empty() {
            anyhow::bail!(
                "No speech tokens in backbone output. Snippet: {:?}",
                &generated[..generated.len().min(200)]
            );
        }

        self.codec
            .decode(&speech_ids)
            .context("NeuCodec decode failed")
    }

    pub fn decode_tokens(&self, speech_ids: &[i32]) -> Result<Vec<f32>> {
        self.codec
            .decode(speech_ids)
            .context("NeuCodec decode failed")
    }
}
