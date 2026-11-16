// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Cost per 40 ms hop: streaming (4-frame hops) against recomputing the window (5-frame
//! hops, which cannot stream), for the lite / mid / full networks at 1.2 s and 2 s windows.
//!
//!   cargo run --release -p rlx-wakeword-core --example stream_bench

use std::time::Instant;

use rlx_wakeword_core::{WakeCnn, WakeCnnConfig, WakeCnnWeights};

fn main() {
    let mid = WakeCnnConfig {
        n_mels: 32,
        c1: 32,
        c2: 64,
        c3: 64,
        k: 3,
        hidden: 128,
        pool_chunks: 4,
        res_blocks: 0,
        attn: false,
        pcen: false,
    };
    for (name, cfg) in [
        ("lite", WakeCnnConfig::lite()),
        ("mid", mid),
        ("full", WakeCnnConfig::full()),
    ] {
        for window in [120usize, 200] {
            let mut w = WakeCnnWeights::he_init(cfg.clone(), 1);
            let h = w.cfg.hidden;
            w.fc2_w = vec![0.01; 3 * h];
            w.fc2_b = vec![0.0; 3];
            let n_mels = w.cfg.n_mels;
            let frames: Vec<f32> = (0..(window + 4000) * n_mels)
                .map(|i| ((i * 31 % 97) as f32) / 50.0 - 1.0)
                .collect();
            let mut t = [0.0f64; 2];
            for (j, hop) in [4usize, 5].into_iter().enumerate() {
                let mut cnn = WakeCnn::new(w.clone()).with_window_frames(window);
                cnn.push_mel_frames_all(&frames[..window * n_mels]);
                let (mut at, mut n) = (window, 0);
                let t0 = Instant::now();
                while at + hop <= frames.len() / n_mels && n < 600 {
                    std::hint::black_box(
                        cnn.push_mel_frames_all(&frames[at * n_mels..(at + hop) * n_mels]),
                    );
                    at += hop;
                    n += 1;
                }
                t[j] = t0.elapsed().as_secs_f64() / n as f64;
            }
            println!(
                "{name:4} window {window}: streaming {:7.1} µs/hop, full window {:7.1} µs/hop ({:.0}x)",
                t[0] * 1e6,
                t[1] * 1e6,
                t[1] / t[0]
            );
        }
    }
}
