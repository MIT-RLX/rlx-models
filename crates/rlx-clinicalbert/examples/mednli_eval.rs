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

//! Mini MedNLI-style sentence-pair demo: pooler features + logistic regression.

#[path = "support/common.rs"]
mod common;

use anyhow::{Context, Result};
use rlx_clinicalbert::{
    ClinicalBertRunner, ClinicalBertTokenizer, LabeledFeature, Pooling, TrainConfig, train_logreg,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights = PathBuf::from(common::require_flag(&args, "--weights")?);
    let device = common::parse_device(
        &common::parse_flag(&args, "--device")?.unwrap_or_else(|| "cpu".into()),
    )?;
    let seq: usize = common::parse_flag(&args, "--seq")?
        .unwrap_or_else(|| "64".into())
        .parse()
        .context("--seq")?;

    // Tiny synthetic NLI-style set (entailment / neutral / contradiction).
    let pairs: [(&str, &str, usize); 6] = [
        (
            "The patient has pneumonia.",
            "The patient has a lung infection.",
            0,
        ),
        (
            "The patient has pneumonia.",
            "The patient was discharged.",
            1,
        ),
        (
            "The patient has pneumonia.",
            "The patient has no lung disease.",
            2,
        ),
        ("Blood pressure is elevated.", "Hypertension was noted.", 0),
        (
            "Blood pressure is elevated.",
            "The patient is asymptomatic.",
            1,
        ),
        (
            "Blood pressure is elevated.",
            "Blood pressure is normal.",
            2,
        ),
    ];

    let tok = ClinicalBertTokenizer::from_dir_or_sibling(&weights)?;
    let pair_refs: Vec<(&str, &str)> = pairs.iter().map(|(a, b, _)| (*a, *b)).collect();
    let enc = tok.encode_pairs_batch(&pair_refs, seq)?;

    let mut runner = ClinicalBertRunner::builder()
        .weights(&weights)
        .device(device)
        .batch(pairs.len())
        .max_seq(seq)
        .pooling(Pooling::Cls)
        .with_pooler()
        .build()?;

    let hidden = runner.forward(
        &enc.input_ids,
        &enc.attention_mask,
        &enc.token_type_ids,
        &enc.position_ids,
    )?;
    let pooled = runner.pooler_output(&hidden)?;
    let hidden_size = runner.hidden_size();
    let num_classes = 3;

    let mut features: Vec<Vec<f32>> = Vec::with_capacity(pairs.len());
    for i in 0..pairs.len() {
        let off = i * hidden_size;
        features.push(pooled[off..off + hidden_size].to_vec());
    }
    let labeled: Vec<LabeledFeature<'_>> = pairs
        .iter()
        .zip(features.iter())
        .map(|((_, _, label), feat)| LabeledFeature {
            features: feat,
            label: *label,
        })
        .collect();

    let cfg = TrainConfig {
        epochs: 200,
        lr: 0.05,
        momentum: 0.9,
        l2: 1e-4,
        batch: pairs.len(),
    };
    let clf = train_logreg(hidden_size, num_classes, &labeled, &cfg, false)?;
    let acc = clf.accuracy(&labeled)?;
    println!("synthetic MedNLI demo accuracy={acc:.1}% (device={device:?})");
    Ok(())
}
