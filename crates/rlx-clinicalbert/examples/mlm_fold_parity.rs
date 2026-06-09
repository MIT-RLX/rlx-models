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

//! Compare in-graph vs CPU MLM head logits on the same hidden states.

#[path = "support/common.rs"]
mod common;

use anyhow::{Result, bail};
use rlx_clinicalbert::{ClinicalBertRunner, MlmExecMode, Pooling};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights = PathBuf::from(common::require_flag(&args, "--weights")?);
    let device = common::parse_device(
        &common::parse_flag(&args, "--device")?.unwrap_or_else(|| "cpu".into()),
    )?;
    let parity_dir = PathBuf::from(
        common::parse_flag(&args, "--parity-dir")?
            .unwrap_or_else(|| "/tmp/rlx-clinicalbert-parity".into()),
    );

    let (meta, inputs) = common::load_parity_inputs(&parity_dir)?;
    let input_ids = common::f32_vec_from_u64(&inputs.input_ids);
    let attention_mask = common::f32_vec_from_u64(&inputs.attention_mask);
    let token_type_ids = common::f32_vec_from_u64(&inputs.token_type_ids);
    let position_ids = common::position_ids(meta.seq);

    let mut ingraph = ClinicalBertRunner::builder()
        .weights(&weights)
        .device(device)
        .batch(1)
        .max_seq(meta.seq)
        .pooling(Pooling::None)
        .mlm_mode(MlmExecMode::InGraph)
        .build()?;
    let hidden = ingraph.forward(&input_ids, &attention_mask, &token_type_ids, &position_ids)?;
    let ingraph_logits = ingraph.mlm_logits(&hidden)?;

    let mut cpu = ClinicalBertRunner::builder()
        .weights(&weights)
        .device(device)
        .batch(1)
        .max_seq(meta.seq)
        .pooling(Pooling::None)
        .mlm_mode(MlmExecMode::Cpu)
        .build()?;
    let hidden_cpu = cpu.forward(&input_ids, &attention_mask, &token_type_ids, &position_ids)?;
    let cpu_logits = cpu.mlm_logits(&hidden_cpu)?;

    if ingraph_logits.len() != cpu_logits.len() {
        bail!(
            "logits length mismatch: ingraph={} cpu={}",
            ingraph_logits.len(),
            cpu_logits.len()
        );
    }
    let err = common::max_abs_diff(&ingraph_logits, &cpu_logits);
    println!("mlm ingraph vs cpu max_abs={err:.6e} (device={device:?})");
    if err > 1e-3 {
        bail!("MLM fold parity failed: max_abs={err:.6e} > 1e-3");
    }
    println!("MLM fold parity ok");
    Ok(())
}
