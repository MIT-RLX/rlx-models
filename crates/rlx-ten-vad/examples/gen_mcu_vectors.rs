// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regenerate the MCU firmware's golden vectors.
//!
//! `rlx-ten-vad-mcu` self-checks its integer net against `probs_q15.bin`, which
//! is what proves the datapath is bit-exact on RISC-V rather than merely close.
//! Those files existed with no generator in the tree, so any deliberate change
//! to the integer datapath — a different accumulator width, a different
//! activation scale — left the firmware failing with no way to tell a real
//! regression from a stale fixture.
//!
//! This reads the committed `feats_q15.bin` (the input side, unchanged) and
//! recomputes the expected probabilities with the host `FixedNet`, in one
//! continuous run because the LSTM state carries across frames.
//!
//! Run: cargo run --release -p rlx-ten-vad --example gen_mcu_vectors

use rlx_ten_vad_core::fixed::FixedNet;
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-ten-vad-mcu/data");
    let stack_len = CONTEXT_FRAMES * FEATURE_LEN;

    let raw = std::fs::read(dir.join("feats_q15.bin"))?;
    anyhow::ensure!(
        raw.len() % (stack_len * 4) == 0,
        "feats_q15.bin is {} bytes, not a whole number of {stack_len}-element stacks",
        raw.len()
    );
    let frames = raw.len() / (stack_len * 4);
    let feats: Vec<i32> = raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let old = std::fs::read(dir.join("probs_q15.bin")).unwrap_or_default();
    let mut net = FixedNet::embedded();
    let mut out = Vec::with_capacity(frames * 4);
    let mut changed = 0usize;
    for f in 0..frames {
        let p = net.forward(&feats[f * stack_len..(f + 1) * stack_len]);
        if old.len() == frames * 4 {
            let was = i32::from_le_bytes(old[f * 4..f * 4 + 4].try_into().unwrap());
            if was != p {
                changed += 1;
            }
        }
        out.extend_from_slice(&p.to_le_bytes());
    }

    std::fs::write(dir.join("probs_q15.bin"), &out)?;
    println!("wrote {frames} probabilities; {changed} differ from the previous file");
    println!("LSTM_IN_SHIFT = {}", rlx_ten_vad_core::fixed::LSTM_IN_SHIFT);
    Ok(())
}
