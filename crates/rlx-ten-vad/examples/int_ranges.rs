// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What accumulator width the integer net actually needs.
//!
//! The LSTM hot loop accumulates `i16 x Q15` products in `i64`. On a 32-bit
//! core that is roughly twice the instruction count of an `i32` accumulate, and
//! the net is 31% of the ESP32-C3 frame budget — so whether the `i64` is
//! *demanded* or merely *safe* is worth a number rather than an assumption.
//!
//! Worst case says i64: 144 terms of 31 bits each is 39 bits. But the bound
//! that matters is `max_r ||W_r||_1 * max|x|`, and trained weights are nothing
//! like worst case. This measures the real one over the distillation set.
//!
//! Run: cargo run --release -p rlx-ten-vad --example int_ranges

use rlx_ten_vad_core::fixed::{FixedNet, LSTM_IN_SHIFT, ONE};
use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::weights::embedded_net;
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let raw = std::fs::read(root.join("target/distill/features.f32"))?;
    let feats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = feats.len() / FEATURE_LEN;
    println!("scanning {n} frames");

    let mut f32net = Net::new(embedded_net());
    let mut net = FixedNet::embedded();
    let (mut flips, mut worst, mut scored) = (0usize, 0.0f32, 0usize);
    let mut fstack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    // The net takes Q15 features, as the frontend hands them over on-device.
    let mut stack = vec![0i32; CONTEXT_FRAMES * FEATURE_LEN];
    for t in 0..n {
        stack.copy_within(FEATURE_LEN.., 0);
        for (slot, &v) in stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]
            .iter_mut()
            .zip(&feats[t * FEATURE_LEN..(t + 1) * FEATURE_LEN])
        {
            *slot = (v * ONE as f32).round() as i32;
        }
        let q = net.forward(&stack) as f32 / ONE as f32;

        fstack.copy_within(FEATURE_LEN.., 0);
        fstack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]
            .copy_from_slice(&feats[t * FEATURE_LEN..(t + 1) * FEATURE_LEN]);
        let r = f32net.forward(&fstack);

        scored += 1;
        worst = worst.max((q - r).abs());
        if (q >= 0.5) != (r >= 0.5) {
            flips += 1;
        }
    }

    println!(
        "\nLSTM_IN_SHIFT = {LSTM_IN_SHIFT}: vs the f32 net over {scored} frames — \
         max|Δ| {worst:.3e}, flips {flips}"
    );
    let r = net.ranges();
    let bits = r.partial_bits();
    println!(
        "\n  max |partial sum|   {:>14}   ({bits} bits)",
        r.max_partial
    );
    println!(
        "  max |pre-activation| {:>13}   ({:.1} in Q15 units)",
        r.max_pre,
        f64::from(r.max_pre) / 32768.0
    );
    println!(
        "  max |cell|           {:>13}   ({:.1} in Q15 units)",
        r.max_cell,
        f64::from(r.max_cell) / 32768.0
    );
    println!("  max |activation|     {:>13}", r.max_act);
    println!(
        "\n  i32 accumulate needs <= 31 bits: {}",
        if bits <= 31 { "FITS" } else { "does NOT fit" }
    );
    println!("  headroom to 31 bits: {} bits", 31i64 - i64::from(bits));
    Ok(())
}
