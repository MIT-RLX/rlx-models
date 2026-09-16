// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Dump the FPGA testbench vectors: Q15 features in, Q15 probabilities out.
//!
//! The RTL in `rlx-ten-vad-fpga` is checked bit-for-bit against these, so
//! "the hardware is correct" reduces to "the hardware agrees with
//! `rlx_ten_vad_core::fixed`", which the parity suite ties to the published
//! model.
//!
//! ```text
//! cargo run -p rlx-ten-vad --example dump_fpga_golden
//! ```

use rlx_ten_vad_core::fixed::{FixedNet, ONE};
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};
use std::fmt::Write as _;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read(root.join("tests/fixtures/reference_features.f32"))?;
    let rows: Vec<Vec<f32>> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>()
        .chunks_exact(FEATURE_LEN)
        .map(<[f32]>::to_vec)
        .collect();

    let mut net = FixedNet::embedded();
    let mut stack = vec![0i32; CONTEXT_FRAMES * FEATURE_LEN];
    let (mut feats, mut probs) = (String::new(), String::new());
    for row in &rows {
        stack.copy_within(FEATURE_LEN.., 0);
        for (slot, &v) in stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]
            .iter_mut()
            .zip(row)
        {
            *slot = (v * ONE as f32).round() as i32;
        }
        for &v in &stack {
            writeln!(feats, "{:08x}", v as u32)?;
        }
        writeln!(probs, "{:08x}", net.forward(&stack) as u32)?;
    }

    let tb = root.join("../rlx-ten-vad-fpga/tb");
    std::fs::create_dir_all(&tb)?;
    std::fs::write(tb.join("golden_features.mem"), &feats)?;
    std::fs::write(tb.join("golden_probs.mem"), &probs)?;
    println!(
        "{} frames -> {} feature words, {} probabilities",
        rows.len(),
        rows.len() * CONTEXT_FRAMES * FEATURE_LEN,
        rows.len()
    );
    Ok(())
}
