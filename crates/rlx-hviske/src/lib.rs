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

//! # rlx-hviske
//!
//! **Hviske** is a family of Danish ASR models — Whisper-large-v3 finetunes by
//! Syv.ai. Architecturally they *are* Whisper-large-v3, so this crate is a thin
//! preset over [`rlx_whisper`]: it pins the correct large-v3 config, defaults the
//! decode language to Danish, and forwards backend feature flags so Hviske runs
//! on every RLX backend `rlx-whisper` supports (CPU/Metal/MLX/CoreML/CUDA/ROCm/
//! Vulkan/wgpu).
//!
//! ```no_run
//! use rlx_hviske::{HviskeVariant, danish_builder};
//! # use std::path::Path;
//! // Point the shared rlx-whisper builder at a downloaded Hviske checkpoint.
//! let runner = danish_builder()
//!     .weights(Path::new("hviske-v3/model.safetensors"))
//!     .config_path(Path::new("hviske-v3/config.json"))
//!     .tokenizer_path(Path::new("hviske-v3/tokenizer.json"))
//!     .build()?;
//! # anyhow::Ok(())
//! ```
//!
//! Status: preset + config, CPU smoke. Real-weight transcription and per-backend
//! parity are validated once a Hviske checkpoint is available (the compute path
//! is entirely `rlx-whisper`, which is already multi-backend).

use rlx_whisper::{WhisperConfig, WhisperRunner, WhisperRunnerBuilder};

/// Decode language for every Hviske model (ISO-639-1 Danish).
pub const LANGUAGE: &str = "da";

/// Native input sample rate (Whisper front-end).
pub const SAMPLE_RATE: u32 = 16_000;

/// Known Hviske releases. All share the Whisper-large-v3 topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HviskeVariant {
    /// Original `syvai/hviske`.
    V1,
    /// `syvai/hviske-v2`.
    V2,
    /// `syvai/hviske-v3` (current, recommended).
    V3,
}

impl HviskeVariant {
    /// Hugging Face repository id.
    pub fn repo_id(self) -> &'static str {
        match self {
            HviskeVariant::V1 => "syvai/hviske",
            HviskeVariant::V2 => "syvai/hviske-v2",
            HviskeVariant::V3 => "syvai/hviske-v3",
        }
    }

    /// Human-readable name.
    pub fn display_name(self) -> &'static str {
        match self {
            HviskeVariant::V1 => "Hviske",
            HviskeVariant::V2 => "Hviske v2",
            HviskeVariant::V3 => "Hviske v3",
        }
    }

    /// The recommended default.
    pub fn recommended() -> Self {
        HviskeVariant::V3
    }

    /// The Whisper config for this variant (all variants are large-v3). This is a
    /// reference/fallback config; a real checkout should prefer the shipped
    /// `config.json` via [`WhisperConfig::from_file`] for exact suppression lists.
    pub fn whisper_config(self) -> WhisperConfig {
        large_v3_config()
    }
}

/// Whisper-large-v3 architecture config (shared by all Hviske variants).
pub fn large_v3_config() -> WhisperConfig {
    // Start from the crate's tiny template so we inherit any non-dimensional
    // fields, then overwrite every dimension with the large-v3 values.
    let mut c = WhisperConfig::tiny();
    c.num_mel_bins = 128; // large-v3 uses 128 mel bins (v1/v2 used 80)
    c.max_source_positions = 1500;
    c.d_model = 1280;
    c.encoder_attention_heads = 20;
    c.encoder_layers = 32;
    c.vocab_size = 51866; // large-v3 vocabulary
    c.max_target_positions = 448;
    c.decoder_attention_heads = 20;
    c.decoder_layers = 32;
    // Suppression lists are decoding policy, not architecture — leave them to the
    // shipped config/tokenizer at load time.
    c.suppress_tokens = Vec::new();
    c.begin_suppress_tokens = vec![220, 50257];
    c
}

/// A [`WhisperRunnerBuilder`] pre-configured for Danish Hviske decoding
/// (language = `da`, timestamps on). The caller supplies the checkpoint paths and
/// device before calling `.build()`.
pub fn danish_builder() -> WhisperRunnerBuilder {
    WhisperRunner::builder().language(LANGUAGE).timestamps(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_have_hviske_repo_ids() {
        for v in [HviskeVariant::V1, HviskeVariant::V2, HviskeVariant::V3] {
            let id = v.repo_id();
            assert!(id.contains("hviske"), "repo id {id}");
            assert!(!v.display_name().is_empty());
        }
        assert_eq!(HviskeVariant::recommended(), HviskeVariant::V3);
    }

    #[test]
    fn config_matches_whisper_large_v3() {
        let c = large_v3_config();
        assert_eq!(c.num_mel_bins, 128);
        assert_eq!(c.d_model, 1280);
        assert_eq!(c.encoder_layers, 32);
        assert_eq!(c.decoder_layers, 32);
        assert_eq!(c.encoder_attention_heads, 20);
        assert_eq!(c.decoder_attention_heads, 20);
        assert_eq!(c.vocab_size, 51866);
        assert_eq!(c.max_source_positions, 1500);
        assert_eq!(c.max_target_positions, 448);
        // Derived head dim is 64 (1280 / 20) and the model is multilingual.
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.decoder_head_dim(), 64);
        assert!(c.is_multilingual());
    }

    #[test]
    fn all_variants_share_the_large_v3_config() {
        let a = HviskeVariant::V1.whisper_config();
        let b = HviskeVariant::V3.whisper_config();
        assert_eq!(a.d_model, b.d_model);
        assert_eq!(a.num_mel_bins, b.num_mel_bins);
    }

    #[test]
    fn language_and_sample_rate_defaults() {
        assert_eq!(LANGUAGE, "da");
        assert_eq!(SAMPLE_RATE, 16_000);
    }
}
