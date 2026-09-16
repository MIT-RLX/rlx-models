// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Where the per-frame time goes: FFT, the rest of the mel path, pitch, net.
//!
//! Worth knowing before moving any of it onto an accelerator — the pitch
//! estimator consumes the same power spectrum the mel bands do, so the FFT
//! cannot be offloaded independently of it.
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example frontend_profile
//! ```

use rlx_ten_vad_core::frontend::{Frontend, pre_emphasis};
use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::{FFT_SIZE, HOP_SIZE, weights::embedded_net};
use std::time::Instant;

fn bench(label: &str, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters / 10 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("  {label:28} {us:8.3} us/frame");
    us
}

fn main() {
    const N: usize = 4000;
    // Speech-like input, so the pitch estimator's Viterbi does real work.
    let pcm: Vec<f32> = (0..HOP_SIZE * 64)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            ((t * 130.0 * std::f32::consts::TAU).sin() * 0.6
                + (t * 260.0 * std::f32::consts::TAU).sin() * 0.3)
                * 8000.0
        })
        .collect();

    println!("per-frame cost (16 ms of audio per frame)\n");

    let mut fe = Frontend::new(rlx_ten_vad_core::weights::embedded());
    let mut prev = 0.0f32;
    let mut emph = vec![0.0f32; HOP_SIZE];
    let mut at = 0usize;
    let full = bench("frontend (mel + pitch)", N, || {
        let raw = &pcm[at..at + HOP_SIZE];
        at = (at + HOP_SIZE) % (pcm.len() - HOP_SIZE);
        pre_emphasis(raw, &mut prev, &mut emph);
        fe.push(raw, &emph);
    });

    // The FFT alone, exactly as the frontend calls it: a 768-sample window
    // zero-padded to 1024, straight to a power spectrum.
    let window: Vec<f32> = (0..rlx_ten_vad_core::WINDOW_SIZE)
        .map(|i| pcm[i % pcm.len()])
        .collect();
    let mut spectrum = vec![0.0f32; FFT_SIZE / 2 + 1];
    let fft = bench("  of which: 1024-pt FFT", N, || {
        rlx_ten_vad_core::ooura::power_spectrum(&window, &mut spectrum);
        std::hint::black_box(&spectrum);
    });

    // The pitch estimator alone, fed the same spectrum the mel path uses.
    let mut pe = rlx_ten_vad_core::pitch::PitchEstimator::new();
    let mut sig = vec![0.0f32; HOP_SIZE];
    let mut k = 0usize;
    let pitch = bench("  of which: pitch estimator", N, || {
        sig.copy_from_slice(&pcm[k..k + HOP_SIZE]);
        k = (k + HOP_SIZE) % (pcm.len() - HOP_SIZE);
        std::hint::black_box(pe.process(&sig, &spectrum));
    });

    let mut net = Net::new(embedded_net());
    let feat = vec![0.1f32; 3 * 41];
    let nn = bench("network (f32 scalar)", N, || {
        std::hint::black_box(net.forward(&feat));
    });

    println!();
    println!("  FFT    {:5.1}% of the frontend", 100.0 * fft / full);
    println!("  pitch  {:5.1}% of the frontend", 100.0 * pitch / full);
    println!(
        "  mel+rest {:3.1}% of the frontend",
        100.0 * (full - fft - pitch).max(0.0) / full
    );
    println!();
    println!(
        "  frontend is {:.0}% of the frame",
        100.0 * full / (full + nn)
    );
    println!(
        "  real-time budget is 16000 us; frame costs {:.1} us ({:.0}x)",
        full + nn,
        16_000.0 / (full + nn)
    );
}
