// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Phase-4 gate: an end-to-end SnapViT run (local scores → xNES → elastic
//! pruning) produces a monotone sparsity/retention continuum from one score,
//! xNES never underperforms the local-only baseline, and the pruned masks run.

use rlx_runtime::Device;
use rlx_vit_elastic::snapvit::{self, CalibImage, SnapVitParams};
use rlx_vit_elastic::vit::runner::VitRunner;
use rlx_vit_elastic::vit::{VitConfig, prepare_from_weightmap, synthetic_checkpoint};

fn synth_image(seed: u32, side: usize) -> CalibImage {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let rgb = (0..side * side * 3)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect();
    CalibImage {
        rgb,
        h: side,
        w: side,
    }
}

#[test]
fn snapvit_end_to_end_elastic() {
    let cfg = VitConfig::synthetic();
    let loaded = prepare_from_weightmap(synthetic_checkpoint(&cfg, 9), &cfg).unwrap();

    let mut params = SnapVitParams::new(cfg.img_size);
    params.ssl.crops.n_local = 2; // 4 crops
    params.ssl.crops.blur_prob = 0.0;
    params.pca_dim = 0; // D=32; raw cosine
    params.xnes.population = 6;
    params.xnes.iterations = 3;
    params.xnes.sparsities = vec![0.2, 0.4];
    params.elastic_sparsities = vec![0.2, 0.4, 0.6];

    let calib: Vec<CalibImage> = (0..3).map(|i| synth_image(200 + i, 48)).collect();
    let fit: Vec<CalibImage> = (0..4).map(|i| synth_image(300 + i, 48)).collect();

    let res = snapvit::run(&cfg, &loaded, &calib, &fit, &params, Device::Cpu).unwrap();

    // xNES never worse than the pure-local (c=1) baseline.
    assert!(res.best_fitness.is_finite() && res.baseline_fitness.is_finite());
    assert!(
        res.best_fitness >= res.baseline_fitness - 1e-4,
        "xNES underperformed baseline: {} < {}",
        res.best_fitness,
        res.baseline_fitness
    );
    assert_eq!(res.xnes_history.len(), 3);

    // Elastic continuum: monotone param reduction, sane retention ordering.
    assert_eq!(res.elastic.len(), 3);
    for e in &res.elastic {
        assert!(e.fitness.is_finite());
        assert!(e.param_reduction >= 0.0 && e.param_reduction < 1.0);
    }
    assert!(res.elastic[0].param_reduction <= res.elastic[1].param_reduction);
    assert!(res.elastic[1].param_reduction <= res.elastic[2].param_reduction);
    assert!(
        res.elastic[2].heads_pruned + res.elastic[2].ffn_pruned > 0,
        "nothing pruned at 0.6"
    );
    // Less pruning retains at least as much representation (allow small noise).
    assert!(
        res.elastic[0].fitness >= res.elastic[2].fitness - 0.05,
        "retention not monotone: {} (0.2) vs {} (0.6)",
        res.elastic[0].fitness,
        res.elastic[2].fitness
    );

    // The exported 0.4-sparsity sub-network runs on a fresh runner.
    let loaded2 = prepare_from_weightmap(synthetic_checkpoint(&cfg, 9), &cfg).unwrap();
    let mut runner = VitRunner::from_loaded(cfg.clone(), loaded2, Device::Cpu, 1).unwrap();
    let e04 = &res.elastic[1];
    runner
        .set_masks(e04.head_mask.clone(), e04.ffn_mask.clone())
        .unwrap();
    let probe = synth_image(777, cfg.img_size);
    let (emb, _) = runner.predict_image(&probe.rgb, probe.h, probe.w).unwrap();
    assert!(emb[0].iter().all(|v| v.is_finite()));
    let norm: f32 = emb[0].iter().map(|v| v * v).sum();
    assert!(norm > 1e-6, "pruned model embedding collapsed");
}
