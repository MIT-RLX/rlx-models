// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Phase-5 gate: GLARE adapter-only continual pre-training runs on RLX, reduces
//! the three-term consistency loss, and the EMA teacher tracks the student.

use rlx_runtime::Device;
use rlx_vit_elastic::glare::{GlareConfig, GlareTrainer};
use rlx_vit_elastic::snapvit::CalibImage;
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
fn glare_trains_adapter_and_reduces_loss() {
    let cfg = VitConfig::synthetic(); // plain ViT: hidden 32, 4 heads, 2 layers
    let loaded = prepare_from_weightmap(synthetic_checkpoint(&cfg, 21), &cfg).unwrap();

    let mut gc = GlareConfig::small(cfg.hidden_size);
    gc.lr = 0.01;
    gc.n_regions = 2;

    let steps = 40;
    let mut trainer = GlareTrainer::new(&cfg, &loaded, &gc, steps, Device::Cpu).unwrap();

    // Snapshot an initial teacher param to verify EMA movement later.
    let up0 = trainer.teacher_params()["glare.head.last.weight"].clone();

    let images: Vec<CalibImage> = (0..4)
        .map(|i| synth_image(400 + i, cfg.img_size * 2))
        .collect();
    let losses = trainer.train(&images, steps).unwrap();

    assert_eq!(losses.len(), steps);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite GLARE loss: {losses:?}"
    );

    let first: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = losses[steps - 5..].iter().sum::<f32>() / 5.0;
    assert!(
        last < first,
        "GLARE loss did not decrease: first5-mean {first} -> last5-mean {last} ({losses:?})"
    );

    // EMA teacher moved from its initial value.
    let up1 = &trainer.teacher_params()["glare.head.last.weight"];
    let delta: f32 = up0.iter().zip(up1).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        delta > 1e-6,
        "EMA teacher did not track the student (delta {delta})"
    );
}

// GLARE continual pre-training (adapter + cross-attention + head backward, EMA
// teacher) run natively on a GPU device — finite, non-increasing loss.
#[cfg(any(
    feature = "metal",
    feature = "mlx",
    feature = "gpu",
    feature = "cuda",
    feature = "vulkan"
))]
fn glare_native_on(dev: Device) {
    let cfg = VitConfig::synthetic();
    let loaded = prepare_from_weightmap(synthetic_checkpoint(&cfg, 21), &cfg).unwrap();
    let mut gc = GlareConfig::small(cfg.hidden_size);
    gc.lr = 0.01;
    gc.n_regions = 2;
    let steps = 20;
    let mut trainer = GlareTrainer::new(&cfg, &loaded, &gc, steps, dev).unwrap();
    let images: Vec<CalibImage> = (0..4)
        .map(|i| synth_image(400 + i, cfg.img_size * 2))
        .collect();
    let losses = trainer.train(&images, steps).unwrap();
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "{dev:?} GLARE loss NaN: {losses:?}"
    );
    let first: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = losses[steps - 5..].iter().sum::<f32>() / 5.0;
    eprintln!("{dev:?}_GLARE first5={first:.4} last5={last:.4}");
    assert!(
        last <= first + 1e-3,
        "{dev:?} GLARE loss increased: {first} -> {last}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn glare_native_metal() {
    glare_native_on(Device::Metal);
}

#[cfg(feature = "mlx")]
#[test]
fn glare_native_mlx() {
    glare_native_on(Device::Mlx);
}

#[cfg(feature = "gpu")]
#[test]
fn glare_native_wgpu() {
    glare_native_on(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn glare_native_cuda() {
    glare_native_on(Device::Cuda);
}

#[cfg(feature = "vulkan")]
#[test]
fn glare_native_vulkan() {
    glare_native_on(Device::Vulkan);
}
