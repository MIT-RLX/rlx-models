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

//! ClinicalBERT — BERT pretrained on clinical text (MIMIC-III, BioMed papers).
//!
//! ClinicalBERT shares the BERT-base architecture (12 layers / 768 hidden /
//! 12 heads / 3072 FFN), so this crate is a thin specialization on top of
//! [`rlx_bert`]: it loads weights with the `bert.` prefix used by HF
//! checkpoints, registers ClinicalBERT presets, exposes the
//! [`ClinicalBertRunner`] surface, and adds optional pooling +
//! WordPiece tokenization.
//!
//! Variants:
//! * [`ClinicalBertVariant::Huang`] — `medicalai/ClinicalBERT` (Huang et al. 2019)
//! * [`ClinicalBertVariant::BioClinical`] — `emilyalsentzer/Bio_ClinicalBERT`
//! * [`ClinicalBertVariant::BioDischarge`] — `emilyalsentzer/Bio_Discharge_Summary_BERT`
//!
//! All three are BERT-base shaped — the only differences are the pretraining
//! corpus and the WordPiece vocabulary (`vocab_size`).
//!
//! The optional MLM head can run as a CPU post-process or be folded into the
//! compiled encoder graph; see [`MlmExecMode`] for the measured crossover and
//! the `Auto` policy.

pub mod builder;
pub mod classifier;
pub mod cli;
pub mod config;
pub mod runner;

#[cfg(any(feature = "pooler", feature = "mlm"))]
pub mod heads;

#[cfg(feature = "mlm")]
pub use heads::MlmHead;
#[cfg(feature = "pooler")]
pub use heads::PoolerHead;

#[cfg(feature = "hf-download")]
pub mod download;

#[cfg(feature = "prepare")]
pub mod prepare;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use builder::{build_clinicalbert_built, build_clinicalbert_graph};
pub use classifier::{LabeledFeature, LinearClassifier, TrainConfig, train_logreg};
pub use config::{
    ClinicalBertConfig, ClinicalBertVariant, bio_clinicalbert_preset, bio_discharge_summary_preset,
    clinicalbert_huang_preset,
};
pub use runner::{ClinicalBertRunner, ClinicalBertRunnerBuilder, MlmExecMode, Pooling};

#[cfg(feature = "hf-download")]
pub use download::{download_clinicalbert, fetch_clinicalbert};
#[cfg(feature = "prepare")]
pub use prepare::prepare_clinicalbert_dir;

#[cfg(feature = "tokenizer")]
pub use tokenizer::ClinicalBertTokenizer;

pub const FAMILY: &str = "ClinicalBERT";

/// HF model IDs covered by the built-in presets.
pub const HF_MODEL_IDS: &[(ClinicalBertVariant, &str)] = &[
    (ClinicalBertVariant::Huang, "medicalai/ClinicalBERT"),
    (
        ClinicalBertVariant::BioClinical,
        "emilyalsentzer/Bio_ClinicalBERT",
    ),
    (
        ClinicalBertVariant::BioDischarge,
        "emilyalsentzer/Bio_Discharge_Summary_BERT",
    ),
];
