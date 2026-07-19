// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Phase-3 gate: the local Hessian-diagonal scores are finite, non-degenerate,
//! and rank structures sensibly — masking the highest-scored attention head
//! perturbs the embedding more than masking the lowest-scored one.

use rlx_runtime::Device;
use rlx_vit_elastic::snapvit::mask::zero_head;
use rlx_vit_elastic::snapvit::{CalibImage, SnapVitConfig, compute_local_scores};
use rlx_vit_elastic::vit::runner::VitRunner;
use rlx_vit_elastic::vit::{VitConfig, prepare_from_weightmap, synthetic_checkpoint};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += (x * x) as f64;
        nb += (y * y) as f64;
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

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
fn local_scores_are_sane_and_rank_heads() {
    let cfg = VitConfig::synthetic(); // plain ViT: hidden 32, 4 heads, 2 layers, gelu
    let loaded = prepare_from_weightmap(synthetic_checkpoint(&cfg, 3), &cfg).unwrap();

    let mut sc = SnapVitConfig::new(cfg.img_size);
    sc.crops.n_local = 2; // 2 global + 2 local = 4 crops (keeps the test fast)
    sc.crops.blur_prob = 0.0;

    let images: Vec<CalibImage> = (0..3).map(|i| synth_image(100 + i, 48)).collect();
    let scores = compute_local_scores(&cfg, &loaded, &images, &sc, Device::Cpu).unwrap();

    let n_heads = cfg.num_hidden_layers * cfg.num_attention_heads;
    let n_ffn = cfg.num_hidden_layers * cfg.ffn_inner();
    assert_eq!(scores.head.len(), n_heads);
    assert_eq!(scores.ffn.len(), n_ffn);
    assert!(scores.head.iter().all(|&s| s.is_finite() && s >= 0.0));
    assert!(scores.ffn.iter().all(|&s| s.is_finite() && s >= 0.0));

    let hmax = scores.head.iter().cloned().fold(f32::MIN, f32::max);
    let hmin = scores.head.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        hmax > hmin,
        "head scores degenerate (all equal): {:?}",
        scores.head
    );
    assert!(hmax > 0.0, "head scores all zero");

    // Ordering: mask the top vs bottom head, measure embedding perturbation.
    let top = scores
        .head
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let bot = scores
        .head
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;

    let loaded2 = prepare_from_weightmap(synthetic_checkpoint(&cfg, 3), &cfg).unwrap();
    let mut runner = VitRunner::from_loaded(cfg.clone(), loaded2, Device::Cpu, 1).unwrap();
    let probe = synth_image(999, cfg.img_size);

    runner.reset_masks();
    let (base, _) = runner.predict_image(&probe.rgb, probe.h, probe.w).unwrap();

    let mut hm_top = rlx_vit_elastic::snapvit::ones_head_mask(&cfg);
    zero_head(&cfg, &mut hm_top, top);
    runner
        .set_masks(hm_top, rlx_vit_elastic::snapvit::ones_ffn_mask(&cfg))
        .unwrap();
    let (emb_top, _) = runner.predict_image(&probe.rgb, probe.h, probe.w).unwrap();
    let drop_top = 1.0 - cosine(&base[0], &emb_top[0]);

    let mut hm_bot = rlx_vit_elastic::snapvit::ones_head_mask(&cfg);
    zero_head(&cfg, &mut hm_bot, bot);
    runner
        .set_masks(hm_bot, rlx_vit_elastic::snapvit::ones_ffn_mask(&cfg))
        .unwrap();
    let (emb_bot, _) = runner.predict_image(&probe.rgb, probe.h, probe.w).unwrap();
    let drop_bot = 1.0 - cosine(&base[0], &emb_bot[0]);

    assert!(
        drop_top > 1e-6,
        "masking the top head changed nothing (drop {drop_top})"
    );
    assert!(
        drop_top >= 0.5 * drop_bot,
        "top-scored head ({top}, drop {drop_top}) should perturb >= bottom ({bot}, drop {drop_bot})"
    );
}
