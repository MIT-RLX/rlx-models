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

//! Voice-package manifest (`roota.raw-fp16.v1`) and Piper phoneme-config schemas.
//!
//! Only the fields the runtime actually reads are modelled; the manifests carry a
//! lot of training provenance that inference does not need.

use std::collections::HashMap;

use serde::Deserialize;

/// Manifest format string this crate understands.
pub const MANIFEST_FORMAT: &str = "roota.raw-fp16.v1";

/// Top-level `manifest.json` of a voice package.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: String,
    #[serde(default)]
    pub package_name: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub voice: String,
    pub sample_rate: u32,
    #[serde(default)]
    pub hop_length: usize,
    #[serde(default)]
    pub inference: Inference,
    pub weights_file: String,
    #[serde(default = "neg_one")]
    pub weights_size_bytes: i64,
    #[serde(default)]
    pub weights_sha256: Option<String>,
    #[serde(default)]
    pub total_parameters: u64,
    pub components: HashMap<String, Component>,
    #[serde(default)]
    pub frontend: Frontend,
}

fn neg_one() -> i64 {
    -1
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Inference {
    #[serde(default = "one_f32")]
    pub duration_length_scale: f32,
}

fn one_f32() -> f32 {
    1.0
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Frontend {
    /// Phoneme config shipped alongside the manifest.
    #[serde(default)]
    pub included_config: Option<String>,
}

/// One component (`duration` / `acoustic` / `decoder`) of a voice package.
#[derive(Debug, Clone, Deserialize)]
pub struct Component {
    /// Left as raw JSON: each component has its own config schema.
    pub config: serde_json::Value,
    pub tensors: Vec<TensorEntry>,
}

/// One tensor's location inside the flat weights blob.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset_bytes: usize,
    pub nbytes: usize,
}

// ---------------------------------------------------------------------------
// Per-component configs
// ---------------------------------------------------------------------------

/// Duration student config (`architecture = "duration_conv"`).
#[derive(Debug, Clone, Deserialize)]
pub struct DurationConfig {
    #[serde(default)]
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden: usize,
    pub depth: usize,
    pub max_tokens: usize,
    pub max_duration: usize,
}

/// Acoustic student config (`architecture = "token_context"`).
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticConfig {
    #[serde(default)]
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden: usize,
    pub depth: usize,
    pub token_depth: usize,
    pub out_channels: usize,
}

/// Decoder student config (`variant = "piperlite"`).
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    #[serde(default)]
    pub variant: String,
    #[serde(default = "leaky_relu_name")]
    pub activation: String,
    #[serde(default = "one_usize")]
    pub res_layers: usize,
    pub channels: Vec<usize>,
    #[serde(default)]
    pub stage0_branches: Option<Vec<usize>>,
    #[serde(default)]
    pub stage1_branches: Option<Vec<usize>>,
    #[serde(default)]
    pub stage2_branches: Option<Vec<usize>>,
    #[serde(default)]
    pub post_filter_channels: usize,
    #[serde(default)]
    pub post_filter_layers: usize,
    #[serde(default = "nine_usize")]
    pub post_filter_kernel: usize,
    #[serde(default)]
    pub post_filter_scale: f32,
    #[serde(default)]
    pub pre_tanh_repair_channels: usize,
}

fn leaky_relu_name() -> String {
    "leaky_relu".to_string()
}

fn one_usize() -> usize {
    1
}

fn nine_usize() -> usize {
    9
}

impl DecoderConfig {
    /// Active residual-bank branches for stage `i`, defaulting to all three.
    pub fn stage_branches(&self, stage: usize) -> Vec<usize> {
        let field = match stage {
            0 => &self.stage0_branches,
            1 => &self.stage1_branches,
            _ => &self.stage2_branches,
        };
        field.clone().unwrap_or_else(|| vec![0, 1, 2])
    }
}

// ---------------------------------------------------------------------------
// Piper phoneme config
// ---------------------------------------------------------------------------

/// Framing ids fixed by the Piper `phoneme_id_map` convention.
pub const PAD_ID: i64 = 0;
/// Beginning-of-sequence id (`^`).
pub const BOS_ID: i64 = 1;
/// End-of-sequence id (`$`).
pub const EOS_ID: i64 = 2;

/// A voice's `piper-phoneme-config.json`: espeak voice + codepoint → id map.
#[derive(Debug, Clone)]
pub struct PhonemeTable {
    pub espeak_voice: String,
    pub id_map: HashMap<char, i64>,
}

#[derive(Debug, Deserialize)]
struct RawPhonemeConfig {
    #[serde(default)]
    phoneme_type: Option<String>,
    #[serde(default)]
    espeak: Option<RawEspeak>,
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct RawEspeak {
    #[serde(default)]
    voice: Option<String>,
}

impl PhonemeTable {
    /// Parse a Piper `*.onnx.json` / `piper-phoneme-config.json`.
    ///
    /// Rejects anything that does not have the exact shape the frontend relies
    /// on: single-codepoint keys, single-id values, and Piper's `_`/`^`/`$`
    /// framing ids — a silently mis-shaped table would produce plausible but
    /// wrong ids rather than an error.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let raw: RawPhonemeConfig = serde_json::from_str(text)?;
        match raw.phoneme_type.as_deref() {
            None | Some("espeak") => {}
            Some(other) => anyhow::bail!("unsupported phoneme_type {other:?}"),
        }
        let espeak_voice = raw
            .espeak
            .and_then(|e| e.voice)
            .ok_or_else(|| anyhow::anyhow!("phoneme config is missing espeak.voice"))?;
        if raw.phoneme_id_map.is_empty() {
            anyhow::bail!("phoneme config has an empty phoneme_id_map");
        }
        let mut id_map = HashMap::with_capacity(raw.phoneme_id_map.len());
        for (key, ids) in &raw.phoneme_id_map {
            let mut chars = key.chars();
            let (Some(ch), None) = (chars.next(), chars.next()) else {
                anyhow::bail!("multi-codepoint phoneme map key {key:?}");
            };
            if ids.len() != 1 {
                anyhow::bail!("multi-id phoneme map value {key:?} -> {ids:?}");
            }
            id_map.insert(ch, ids[0]);
        }
        for (sym, want) in [('_', PAD_ID), ('^', BOS_ID), ('$', EOS_ID)] {
            let got = id_map.get(&sym).copied();
            if got != Some(want) {
                anyhow::bail!("framing symbol {sym:?} maps to {got:?}, expected {want}");
            }
        }
        Ok(Self {
            espeak_voice,
            id_map,
        })
    }
}
