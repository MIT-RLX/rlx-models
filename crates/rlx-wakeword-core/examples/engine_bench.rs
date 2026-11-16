// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Time per 40 ms hop of a streaming wake network (the shape Reachy Mini's runs: `mid`,
//! log-mel + PCEN input, 4 residual blocks, attention pooling, 5 outputs, 2 s window), with each
//! engine option off and on, and scoring every hop or every other one.
//!
//!   cargo run --release -p rlx-wakeword-core --example engine_bench

use std::time::Instant;

use rlx_wakeword_core::cnn::EngineOpts;
use rlx_wakeword_core::{WakeCnn, WakeCnnConfig, WakeCnnWeights};

fn main() {
    let cfg = WakeCnnConfig {
        n_mels: 32,
        c1: 32,
        c2: 64,
        c3: 64,
        k: 3,
        hidden: 128,
        pool_chunks: 4,
        res_blocks: 4,
        attn: true,
        pcen: false,
        dual: true,
        causal: false,
    };
    let five = |cfg: WakeCnnConfig| {
        let mut w = WakeCnnWeights::he_init(cfg, 7);
        w.fc2_w = w.fc2_w.iter().cycle().take(5 * 128).copied().collect();
        w.fc2_b = vec![0.0; 5];
        w
    };
    // The fully streaming (causal) form, over a window long enough to pool the same ~2 s.
    let causal = five(WakeCnnConfig { causal: true, ..cfg.clone() });
    let w = five(cfg);
    let mut s = 0x1234_5678u32;
    let frames: Vec<f32> = (0..4 * 3000 * 32)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            0.5 + (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0
        })
        .collect();
    let all = EngineOpts::default();
    let none = EngineOpts { reuse_pad: false, stem_blocked: false, fast_math: false };
    let variants: [(&str, EngineOpts, usize, bool); 9] = [
        ("baseline (all off)", none, 1, false),
        ("+ reuse pad buffer", EngineOpts { reuse_pad: true, ..none }, 1, false),
        ("+ stem 4-channel blocking", EngineOpts { stem_blocked: true, ..none }, 1, false),
        ("+ fast exp/ln (PCEN, attention)", EngineOpts { fast_math: true, ..none }, 1, false),
        ("all three", all, 1, false),
        ("baseline, every other hop", none, 2, false),
        ("all three, every other hop", all, 2, false),
        ("causal streaming, all three", all, 1, true),
        ("causal streaming, every other hop", all, 2, true),
    ];
    // Each variant timed in 3 interleaved rounds, the best kept (other load only adds time).
    let mut best = [f64::MAX; 9];
    let mut sinks = [0.0f32; 9];
    for _round in 0..3 {
        for (v, (_, opts, every, is_causal)) in variants.iter().enumerate() {
            let (net_w, win) = if *is_causal { (&causal, 332) } else { (&w, 200) };
            let mut net = WakeCnn::new(net_w.clone()).with_window_frames(win).with_opts(*opts);
            let hop = 4 * 32;
            // Fill the window, then time 2000 hops (40 ms each: 80 s of audio).
            net.push_mel_frames_all(&frames[..win * 32]);
            let mut sink = 0.0f32;
            let t = Instant::now();
            for (i, f) in frames[win * 32..].chunks_exact(hop).take(2000).enumerate() {
                if i % every == every - 1 {
                    sink += net.push_mel_frames_all(f).map_or(0.0, |p| p[0]);
                } else {
                    net.queue_frames(f);
                }
            }
            best[v] = best[v].min(t.elapsed().as_secs_f64() * 1e6 / 2000.0);
            sinks[v] = sink;
        }
    }
    for (v, (name, _, _, _)) in variants.iter().enumerate() {
        let us = best[v];
        println!(
            "{name:34} {us:7.1} µs per hop  ({:.2}% of a core, {:.2}× baseline)  [checksum {:.3}]",
            us / 400.0,
            best[0] / us,
            sinks[v]
        );
    }
}
