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

//! ClinicalBERT configuration — wraps [`rlx_core::config::BertConfig`].

use anyhow::{Context, Result, bail};
use rlx_core::config::BertConfig;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Known ClinicalBERT checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClinicalBertVariant {
    /// Huang et al. 2019 — `medicalai/ClinicalBERT`.
    Huang,
    /// Alsentzer et al. 2019 — `emilyalsentzer/Bio_ClinicalBERT`.
    BioClinical,
    /// Alsentzer et al. 2019 (discharge summaries) — `emilyalsentzer/Bio_Discharge_Summary_BERT`.
    BioDischarge,
}

impl ClinicalBertVariant {
    pub fn hf_repo(&self) -> &'static str {
        match self {
            ClinicalBertVariant::Huang => "medicalai/ClinicalBERT",
            ClinicalBertVariant::BioClinical => "emilyalsentzer/Bio_ClinicalBERT",
            ClinicalBertVariant::BioDischarge => "emilyalsentzer/Bio_Discharge_Summary_BERT",
        }
    }

    pub fn preset(&self) -> BertConfig {
        match self {
            ClinicalBertVariant::Huang => clinicalbert_huang_preset(),
            ClinicalBertVariant::BioClinical => bio_clinicalbert_preset(),
            ClinicalBertVariant::BioDischarge => bio_discharge_summary_preset(),
        }
    }
}

/// ClinicalBERT configuration. Equivalent to [`BertConfig`], with helpers for
/// detecting checkpoint variants and locating sibling files.
#[derive(Debug, Clone)]
pub struct ClinicalBertConfig {
    pub bert: BertConfig,
    pub variant: Option<ClinicalBertVariant>,
}

impl ClinicalBertConfig {
    pub fn new(bert: BertConfig) -> Self {
        Self {
            bert,
            variant: None,
        }
    }

    pub fn with_variant(mut self, variant: ClinicalBertVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let cfg = BertConfig::from_file(path)
            .with_context(|| format!("rlx-clinicalbert: parse {path:?}"))?;
        let variant = probe_variant(path).ok().flatten();
        Ok(Self { bert: cfg, variant })
    }

    /// Resolve `config.json` next to a safetensors file or inside a directory.
    pub fn config_json_path(weights_or_dir: &Path) -> PathBuf {
        if weights_or_dir.is_dir() {
            return weights_or_dir.join("config.json");
        }
        weights_or_dir
            .parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_else(|| PathBuf::from("config.json"))
    }
}

/// Reference dims for [Huang et al. ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT).
///
/// Initialized from `bert-base-uncased`, further pretrained on MIMIC-III notes.
pub fn clinicalbert_huang_preset() -> BertConfig {
    BertConfig {
        vocab_size: 119_547,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        hidden_act: "gelu".into(),
    }
}

/// Reference dims for [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).
///
/// Initialized from BioBERT v1.0 (PubMed-pretrained), further pretrained on
/// the MIMIC-III note corpus.
pub fn bio_clinicalbert_preset() -> BertConfig {
    BertConfig {
        vocab_size: 28_996,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        hidden_act: "gelu".into(),
    }
}

/// Reference dims for [`emilyalsentzer/Bio_Discharge_Summary_BERT`](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT).
pub fn bio_discharge_summary_preset() -> BertConfig {
    bio_clinicalbert_preset()
}

#[derive(Debug, Deserialize)]
struct HfConfigProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default, alias = "_name_or_path")]
    name_or_path: Option<String>,
    #[serde(default)]
    vocab_size: Option<usize>,
}

/// Public wrapper around [`probe_variant`] for callers outside this module
/// (e.g. `crate::prepare::try_download_safetensors_pr` needs to decide which
/// HF PR branch to mirror from).
pub fn probe_variant_public(path: &Path) -> Result<Option<ClinicalBertVariant>> {
    probe_variant(path)
}

/// Heuristic — look at `_name_or_path` then fall back to vocab size + dims.
fn probe_variant(path: &Path) -> Result<Option<ClinicalBertVariant>> {
    let raw = std::fs::read_to_string(path)?;
    let probe: HfConfigProbe = serde_json::from_str(&raw)?;

    if let Some(name) = &probe.name_or_path {
        let lower = name.to_ascii_lowercase();
        if lower.contains("bio_discharge") || lower.contains("discharge_summary") {
            return Ok(Some(ClinicalBertVariant::BioDischarge));
        }
        if lower.contains("bio_clinicalbert") || lower.contains("biobert") {
            return Ok(Some(ClinicalBertVariant::BioClinical));
        }
        if lower.contains("medicalai/clinicalbert") || lower.contains("clinicalbert") {
            return Ok(Some(ClinicalBertVariant::Huang));
        }
    }

    match probe.vocab_size {
        Some(28_996) => Ok(Some(ClinicalBertVariant::BioClinical)),
        Some(119_547) => Ok(Some(ClinicalBertVariant::Huang)),
        _ => Ok(None),
    }
}

/// Fail fast when `config.json` is not BERT-shaped.
pub fn validate_hf_config(weights_or_dir: &Path) -> Result<()> {
    let cfg_path = ClinicalBertConfig::config_json_path(weights_or_dir);
    let raw =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path:?}"))?;
    let probe: HfConfigProbe =
        serde_json::from_str(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;

    match probe.model_type.as_deref() {
        Some("bert") => {}
        Some(other) => bail!(
            "rlx-clinicalbert: {cfg_path:?} has model_type={other:?}; expected `bert` \
             (ClinicalBERT is BERT-base-shaped)"
        ),
        None => bail!("rlx-clinicalbert: {cfg_path:?} missing model_type"),
    }

    if let Some(archs) = &probe.architectures {
        let ok = archs.iter().any(|a| {
            matches!(
                a.as_str(),
                "BertModel"
                    | "BertForMaskedLM"
                    | "BertForSequenceClassification"
                    | "BertForPreTraining"
            )
        });
        if !ok {
            bail!(
                "rlx-clinicalbert: {cfg_path:?} architectures={archs:?}; \
                 expected BertModel / BertForMaskedLM / BertForSequenceClassification"
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_bert_base_dims() {
        for cfg in [
            clinicalbert_huang_preset(),
            bio_clinicalbert_preset(),
            bio_discharge_summary_preset(),
        ] {
            assert_eq!(cfg.hidden_size, 768);
            assert_eq!(cfg.num_hidden_layers, 12);
            assert_eq!(cfg.num_attention_heads, 12);
            assert_eq!(cfg.intermediate_size, 3072);
            assert_eq!(cfg.head_dim(), 64);
        }
        assert_eq!(bio_clinicalbert_preset().vocab_size, 28_996);
        assert_eq!(clinicalbert_huang_preset().vocab_size, 119_547);
    }

    #[test]
    fn probes_variant_by_name() {
        let dir = std::env::temp_dir().join(format!(
            "rlx_clinicalbert_cfg_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{
                "_name_or_path": "emilyalsentzer/Bio_ClinicalBERT",
                "model_type": "bert",
                "architectures": ["BertModel"],
                "vocab_size": 28996,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 512
            }"#,
        )
        .unwrap();
        let cfg = ClinicalBertConfig::from_file(&path).unwrap();
        assert_eq!(cfg.variant, Some(ClinicalBertVariant::BioClinical));
        assert_eq!(cfg.bert.vocab_size, 28_996);
        validate_hf_config(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
