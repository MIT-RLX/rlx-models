// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! What the pitch feature is worth, and how precisely it has to be computed.
//!
//! Feature 40 of 41 is an LPC pitch estimate, and on an FPU-less MCU it costs
//! 3.74 M instructions per frame — 81% of the whole DSP frontend. Porting it to
//! fixed point is the expensive option. This measures the cheap question first:
//! how much precision does the network actually need from it?
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example pitch_value
//! ```

use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::weights::embedded_net;
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};
use std::path::Path;

const PITCH: usize = FEATURE_LEN - 1;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Score the reference clip with feature 40 passed through `f`.
fn score(rows: &[Vec<f32>], want: &[f32], mut f: impl FnMut(f32) -> f32) -> (f32, f64, usize) {
    let mut net = Net::new(embedded_net());
    let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    let got: Vec<f32> = rows
        .iter()
        .map(|row| {
            stack.copy_within(FEATURE_LEN.., 0);
            let cur = &mut stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
            cur.copy_from_slice(row);
            cur[PITCH] = f(row[PITCH]);
            net.forward(&stack)
        })
        .collect();
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = got
        .iter()
        .map(|a| f64::from(*a).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = want
        .iter()
        .map(|b| f64::from(*b).powi(2))
        .sum::<f64>()
        .sqrt();
    (
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
        1.0 - dot / (na * nb),
        got.iter()
            .zip(want)
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count(),
    )
}

/// Score with every *mel* feature (0..39) perturbed by `f` — the other side of
/// the budget: how exact the FFT and filterbank have to be.
fn score_mel(rows: &[Vec<f32>], want: &[f32], mut f: impl FnMut(f32) -> f32) -> (f32, f64, usize) {
    let mut net = Net::new(embedded_net());
    let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    let got: Vec<f32> = rows
        .iter()
        .map(|row| {
            stack.copy_within(FEATURE_LEN.., 0);
            let cur = &mut stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
            cur.copy_from_slice(row);
            for v in cur[..PITCH].iter_mut() {
                *v = f(*v);
            }
            net.forward(&stack)
        })
        .collect();
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = got
        .iter()
        .map(|a| f64::from(*a).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = want
        .iter()
        .map(|b| f64::from(*b).powi(2))
        .sum::<f64>()
        .sqrt();
    (
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
        1.0 - dot / (na * nb),
        got.iter()
            .zip(want)
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count(),
    )
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rows: Vec<Vec<f32>> = read_f32(&root.join("tests/fixtures/reference_features.f32"))
        .chunks_exact(FEATURE_LEN)
        .map(<[f32]>::to_vec)
        .collect();
    let want = read_f32(&root.join("tests/fixtures/reference_probs_onnx.f32"));

    let lo = rows.iter().map(|r| r[PITCH]).fold(f32::MAX, f32::min);
    let hi = rows.iter().map(|r| r[PITCH]).fold(f32::MIN, f32::max);
    println!(
        "pitch feature (normalised) spans {lo:.3} .. {hi:.3} over {} frames\n",
        rows.len()
    );

    let report = |label: &str, r: (f32, f64, usize)| {
        println!(
            "  {label:34} max|Δ|={:.3e}  1-cos={:.3e}  flips={}/250",
            r.0, r.1, r.2
        );
    };

    report("exact (sanity)", score(&rows, &want, |v| v));
    // Normalised features are (x - mean)/std, so 0 is "the average pitch".
    report("replaced by its mean (0.0)", score(&rows, &want, |_| 0.0));
    report(
        "sign only",
        score(&rows, &want, |v| if v >= 0.0 { 1.0 } else { -1.0 }),
    );

    println!();
    for bits in [1u32, 2, 3, 4, 6, 8] {
        let levels = 1u32 << bits;
        let step = (hi - lo) / (levels - 1) as f32;
        report(
            &format!("{bits}-bit ({levels} levels, step {step:.3})"),
            score(&rows, &want, |v| lo + ((v - lo) / step).round() * step),
        );
    }

    // Pitch moves on a ~10 ms timescale, so recomputing it every 16 ms frame
    // may be wasted work. Holding the previous value costs nothing to try and
    // divides the estimator's cost by the hold length.
    println!();
    for hold in [2usize, 3, 4, 6, 8] {
        let mut net = Net::new(embedded_net());
        let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
        let mut held = 0.0f32;
        let got: Vec<f32> = rows
            .iter()
            .enumerate()
            .map(|(t, row)| {
                if t % hold == 0 {
                    held = row[PITCH];
                }
                stack.copy_within(FEATURE_LEN.., 0);
                let cur = &mut stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
                cur.copy_from_slice(row);
                cur[PITCH] = held;
                net.forward(&stack)
            })
            .collect();
        let dot: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let na: f64 = got
            .iter()
            .map(|a| f64::from(*a).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = want
            .iter()
            .map(|b| f64::from(*b).powi(2))
            .sum::<f64>()
            .sqrt();
        let flips = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count();
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        report(
            &format!("recomputed every {hold} frames ({} ms)", hold * 16),
            (max, 1.0 - dot / (na * nb), flips),
        );
    }

    // The 40 mel features: how exact must the FFT and filterbank be?
    let mlo = rows
        .iter()
        .flat_map(|r| &r[..PITCH])
        .fold(f32::MAX, |a, &b| a.min(b));
    let mhi = rows
        .iter()
        .flat_map(|r| &r[..PITCH])
        .fold(f32::MIN, |a, &b| a.max(b));
    println!("\nmel features span {mlo:.2} .. {mhi:.2}");
    for bits in [4u32, 6, 8, 10, 12] {
        let levels = (1u32 << bits) - 1;
        let step = (mhi - mlo) / levels as f32;
        report(
            &format!("all 40 mel at {bits} bits (step {step:.4})"),
            score_mel(&rows, &want, |v| mlo + ((v - mlo) / step).round() * step),
        );
    }

    println!();
    for rel in [1e-1f32, 3e-2, 1e-2, 1e-3] {
        let span = hi - lo;
        report(
            &format!(
                "absolute error up to {:.3} ({:.0}% of span)",
                rel * span,
                rel * 100.0
            ),
            score(&rows, &want, |v| {
                v + rel * span * if v > 0.0 { 1.0 } else { -1.0 }
            }),
        );
    }
}
