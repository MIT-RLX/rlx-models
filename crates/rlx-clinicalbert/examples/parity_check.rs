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

//! Run ClinicalBERT on HF parity fixtures and write RLX `.bin` outputs for
//! `parity/compare.py`.

#[path = "support/common.rs"]
mod common;

use anyhow::{Context, Result, bail};
use rlx_clinicalbert::{ClinicalBertRunner, MlmExecMode, Pooling};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights = PathBuf::from(common::require_flag(&args, "--weights")?);
    let parity_dir = PathBuf::from(
        common::parse_flag(&args, "--parity-dir")?
            .unwrap_or_else(|| "/tmp/rlx-clinicalbert-parity".into()),
    );
    let device = common::parse_device(
        &common::parse_flag(&args, "--device")?.unwrap_or_else(|| "cpu".into()),
    )?;

    let (meta, inputs) = common::load_parity_inputs(&parity_dir)?;
    if inputs.input_ids.len() != meta.seq {
        bail!(
            "inputs.json seq mismatch: meta.seq={} input_ids={}",
            meta.seq,
            inputs.input_ids.len()
        );
    }

    let input_ids = common::f32_vec_from_u64(&inputs.input_ids);
    let attention_mask = common::f32_vec_from_u64(&inputs.attention_mask);
    let token_type_ids = common::f32_vec_from_u64(&inputs.token_type_ids);
    let position_ids = common::position_ids(meta.seq);

    let mut runner = ClinicalBertRunner::builder()
        .weights(&weights)
        .device(device)
        .batch(1)
        .max_seq(meta.seq)
        .pooling(Pooling::Cls)
        .with_pooler()
        .mlm_mode(MlmExecMode::InGraph)
        .build()
        .context("build runner")?;

    let hidden = runner.forward(&input_ids, &attention_mask, &token_type_ids, &position_ids)?;
    let pooler = runner.pooler_output(&hidden)?;
    let mlm = runner.mlm_logits(&hidden)?;

    common::write_f32_bin(&parity_dir.join("hidden_states_rlx.bin"), &hidden)?;
    common::write_f32_bin(&parity_dir.join("pooler_output_rlx.bin"), &pooler)?;
    common::write_f32_bin(&parity_dir.join("mlm_logits_rlx.bin"), &mlm)?;

    println!(
        "wrote RLX outputs under {} (device={device:?})",
        parity_dir.display()
    );
    Ok(())
}
