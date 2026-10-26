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

//! Model configuration in the upstream YAML format (`pocket_tts/config/*.yaml`).
//!
//! Every released model — the English revisions, the 24-layer and per-language
//! models, the drifting head and the small 2M/7M/18M models — is described by
//! one of these files, and they differ in more than sizes: grouped-query
//! attention, a narrower attention width (`attn_dim`), a low-rank text table
//! (`rank`), a projection before the sampler head (`z_dim`), four sampler-head
//! types and a decoder SEANet/transformer that differs from the encoder's. The
//! upstream configs are bundled (see [`ModelSpec::builtin`]) so a language name
//! is enough; [`ModelSpec::from_yaml`] reads any other.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Upstream configs, keyed by language name (the `--language` values).
const BUILTIN: &[(&str, &str)] = &[
    ("english", include_str!("../configs/english.yaml")),
    (
        "english_2026-01",
        include_str!("../configs/english_2026-01.yaml"),
    ),
    (
        "english_2026-04",
        include_str!("../configs/english_2026-04.yaml"),
    ),
    (
        "english_2026-04_24l",
        include_str!("../configs/english_2026-04_24l.yaml"),
    ),
    (
        "english_2026-09",
        include_str!("../configs/english_2026-09.yaml"),
    ),
    (
        "english_2026-09_2",
        include_str!("../configs/english_2026-09_2.yaml"),
    ),
    (
        "english_2026-09_24l",
        include_str!("../configs/english_2026-09_24l.yaml"),
    ),
    (
        "english_2026-10",
        include_str!("../configs/english_2026-10.yaml"),
    ),
    (
        "english_drifting_26-09",
        include_str!("../configs/english_drifting_26-09.yaml"),
    ),
    (
        "english_18m_2026-10",
        include_str!("../configs/english_18m_2026-10.yaml"),
    ),
    (
        "english_18m_refl_2026-10",
        include_str!("../configs/english_18m_refl_2026-10.yaml"),
    ),
    (
        "english_7m_2026-10",
        include_str!("../configs/english_7m_2026-10.yaml"),
    ),
    (
        "english_7m_refl_2026-10",
        include_str!("../configs/english_7m_refl_2026-10.yaml"),
    ),
    (
        "english_2m_2026-10",
        include_str!("../configs/english_2m_2026-10.yaml"),
    ),
    (
        "english_2m_refl_2026-10",
        include_str!("../configs/english_2m_refl_2026-10.yaml"),
    ),
    ("french", include_str!("../configs/french.yaml")),
    ("french_24l", include_str!("../configs/french_24l.yaml")),
    ("german", include_str!("../configs/german.yaml")),
    ("german_24l", include_str!("../configs/german_24l.yaml")),
    ("italian", include_str!("../configs/italian.yaml")),
    ("italian_24l", include_str!("../configs/italian_24l.yaml")),
    ("portuguese", include_str!("../configs/portuguese.yaml")),
    (
        "portuguese_24l",
        include_str!("../configs/portuguese_24l.yaml"),
    ),
    ("spanish", include_str!("../configs/spanish.yaml")),
    ("spanish_24l", include_str!("../configs/spanish_24l.yaml")),
    ("dutch", include_str!("../configs/dutch.yaml")),
    ("dutch_24l", include_str!("../configs/dutch_24l.yaml")),
];

/// Upstream `DEFAULT_EOS_THRESHOLD`.
pub const DEFAULT_EOS_THRESHOLD: f32 = -4.0;

/// Root of a pocket-tts YAML config.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub flow_lm: FlowLmSpec,
    pub mimi: MimiSpec,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    #[serde(default)]
    pub remove_semicolons: bool,
    #[serde(default = "yes")]
    pub append_terminal_punctuation: bool,
    #[serde(default = "yes")]
    pub capitalize_first_letter: bool,
    /// Per-character rewrites applied before tokenization (`""` deletes).
    #[serde(default, deserialize_with = "nullable_map")]
    pub replace_characters: BTreeMap<String, String>,
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<usize>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
    #[serde(default = "default_eos_threshold")]
    pub default_eos_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowLmSpec {
    pub flow: FlowSpec,
    pub transformer: FlowTransformerSpec,
    pub lookup_table: LookupTableSpec,
    /// Width of the backbone output fed to the sampler and EOS heads (a
    /// bias-free projection from `d_model`); the transformer output when `None`.
    #[serde(default)]
    pub z_dim: Option<usize>,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

/// How the sampler head turns noise into a latent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlowType {
    /// Lagrangian self-distillation: two time conditions (start, end), Euler steps.
    #[default]
    Lsd,
    /// Optimal-transport flow matching: one time condition, Euler steps.
    FlowMatching,
    /// No time condition; the head's output for the noise is the sample.
    Drifting,
    /// No time condition; the noise plus the head's output.
    Pointwise,
}

impl FlowType {
    /// Time conditions the head embeds.
    pub fn num_time_conds(self) -> usize {
        match self {
            FlowType::Lsd => 2,
            FlowType::FlowMatching => 1,
            FlowType::Drifting | FlowType::Pointwise => 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowSpec {
    pub dim: usize,
    pub depth: usize,
    #[serde(default, rename = "type")]
    pub kind: FlowType,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowTransformerSpec {
    pub hidden_scale: usize,
    pub max_period: f32,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub attn_dim: Option<usize>,
    #[serde(default)]
    pub num_kv_heads: Option<usize>,
    #[serde(default)]
    pub dim_feedforward: Option<usize>,
}

impl FlowTransformerSpec {
    pub fn attn_dim(&self) -> usize {
        self.attn_dim.unwrap_or(self.d_model)
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }
    pub fn head_dim(&self) -> usize {
        self.attn_dim() / self.num_heads
    }
    pub fn kv_dim(&self) -> usize {
        self.head_dim() * self.num_kv_heads()
    }
    pub fn ffn_dim(&self) -> usize {
        self.dim_feedforward
            .unwrap_or(self.d_model * self.hidden_scale)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LookupTableSpec {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
    #[serde(default)]
    pub rank: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SeanetSpec {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

impl SeanetSpec {
    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MimiTransformerSpec {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f32,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f32,
    pub dim_feedforward: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizerSpec {
    pub dimension: usize,
    pub output_dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MimiSpec {
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f32,
    pub seanet: SeanetSpec,
    pub transformer: MimiTransformerSpec,
    pub quantizer: QuantizerSpec,
    #[serde(default)]
    pub inner_dim: Option<usize>,
    #[serde(default)]
    pub outer_dim: Option<usize>,
    #[serde(default)]
    pub decoder_seanet: Option<SeanetSpec>,
    #[serde(default)]
    pub decoder_transformer: Option<MimiTransformerSpec>,
}

impl MimiSpec {
    pub fn decoder_seanet(&self) -> &SeanetSpec {
        self.decoder_seanet.as_ref().unwrap_or(&self.seanet)
    }
    pub fn decoder_transformer(&self) -> &MimiTransformerSpec {
        self.decoder_transformer
            .as_ref()
            .unwrap_or(&self.transformer)
    }
    /// Audio samples per latent frame (1920 at 24 kHz / 12.5 Hz).
    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f32 / self.frame_rate).round() as usize
    }
    /// Encoder-rate steps per latent frame (the up/down-sampling stride, 16).
    pub fn steps_per_frame(&self) -> usize {
        self.frame_size() / self.seanet.hop_length()
    }
}

impl ModelSpec {
    /// Parse a config in the upstream YAML format.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let spec: Self = serde_yaml::from_str(text).context("parse pocket-tts config")?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_yaml(&text).with_context(|| format!("config {}", path.display()))
    }

    /// The bundled upstream config for `language` (e.g. `english`, `french_24l`,
    /// `english_7m_2026-10`).
    pub fn builtin(language: &str) -> Result<Self> {
        match BUILTIN.iter().find(|(name, _)| *name == language) {
            Some((_, text)) => Self::from_yaml(text).with_context(|| format!("builtin {language}")),
            None => bail!(
                "unknown pocket-tts language {language:?}; available: {}",
                Self::builtin_languages().join(", ")
            ),
        }
    }

    pub fn builtin_languages() -> Vec<&'static str> {
        BUILTIN.iter().map(|(n, _)| *n).collect()
    }

    fn validate(&self) -> Result<()> {
        let t = &self.flow_lm.transformer;
        if !t.attn_dim().is_multiple_of(t.num_heads)
            || !t.num_heads.is_multiple_of(t.num_kv_heads())
        {
            bail!(
                "attention shape: attn_dim {} / heads {} / kv heads {}",
                t.attn_dim(),
                t.num_heads,
                t.num_kv_heads()
            );
        }
        if !t.head_dim().is_multiple_of(2) {
            bail!("RoPE needs an even head dim, got {}", t.head_dim());
        }
        let m = self.mimi.decoder_transformer();
        if !m.d_model.is_multiple_of(m.num_heads) || !(m.d_model / m.num_heads).is_multiple_of(2) {
            bail!(
                "mimi transformer d_model {} / heads {}",
                m.d_model,
                m.num_heads
            );
        }
        if !self
            .mimi
            .frame_size()
            .is_multiple_of(self.mimi.decoder_seanet().hop_length())
        {
            bail!("frame size is not a multiple of the decoder hop length");
        }
        Ok(())
    }

    /// Latent width (32).
    pub fn latent_dim(&self) -> usize {
        self.mimi.quantizer.dimension
    }

    /// Width the sampler and EOS heads see.
    pub fn cond_dim(&self) -> usize {
        self.flow_lm
            .z_dim
            .unwrap_or(self.flow_lm.transformer.d_model)
    }

    /// Path of the tokenizer inside a repository checkout, from the config's
    /// `hf://<org>/<repo>/<path>@<rev>` reference (`None` for other schemes).
    pub fn tokenizer_repo_path(&self) -> Option<String> {
        hf_repo_path(&self.flow_lm.lookup_table.tokenizer_path)
    }
}

/// `hf://org/repo/a/b.json@rev` → `a/b.json`.
pub(crate) fn hf_repo_path(reference: &str) -> Option<String> {
    let rest = reference.strip_prefix("hf://")?;
    let rest = rest.split('@').next()?;
    let mut parts = rest.splitn(3, '/');
    let _org = parts.next()?;
    let _repo = parts.next()?;
    parts.next().map(str::to_string)
}

fn yes() -> bool {
    true
}
fn default_temperature() -> f32 {
    0.3
}
fn default_eos_threshold() -> f32 {
    DEFAULT_EOS_THRESHOLD
}
fn default_max_period() -> f32 {
    10_000.0
}

/// `replace_characters:` with no entries parses as YAML null.
fn nullable_map<'de, D>(d: D) -> std::result::Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<BTreeMap<String, String>>::deserialize(d)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_config_parses() {
        for name in ModelSpec::builtin_languages() {
            let spec = ModelSpec::builtin(name).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(spec.latent_dim(), 32, "{name}");
            assert_eq!(spec.mimi.frame_size(), 1920, "{name}");
            assert_eq!(spec.mimi.steps_per_frame(), 16, "{name}");
        }
    }

    #[test]
    fn small_model_shapes() {
        let s = ModelSpec::builtin("english_2m_2026-10").unwrap();
        let t = &s.flow_lm.transformer;
        assert_eq!((t.attn_dim(), t.num_kv_heads(), t.head_dim()), (192, 1, 64));
        assert_eq!(t.ffn_dim(), 96);
        assert_eq!(s.flow_lm.flow.kind, FlowType::Pointwise);
        assert_eq!(s.cond_dim(), 64);
        assert_eq!(s.flow_lm.lookup_table.rank, Some(32));
        assert_eq!(s.mimi.decoder_transformer().context, 50);
        assert_eq!(s.default_eos_threshold, -1.0);
    }

    #[test]
    fn tokenizer_path_maps_into_the_repo() {
        let s = ModelSpec::builtin("english_drifting_26-09").unwrap();
        assert_eq!(
            s.tokenizer_repo_path().as_deref(),
            Some("languages/english_2026-09/tokenizer.json")
        );
    }
}
