// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The vision tower on **real** `google/diffusiongemma-26B-A4B-it` weights.
//!
//! The full checkpoint is 51 GB and its routed experts alone are ~91 GB as f32,
//! but the tower is only ~1.15 GB and can be pulled tensor-by-tensor with HTTP
//! Range requests:
//!
//! ```sh
//! python3 scripts/diffusiongemma_fetch_subset.py /w/dg-vision --subset vision
//! python3 scripts/diffusiongemma_real_vision.py /w/dg-vision
//! RLX_DG_REAL_VISION_DIR=/w/dg-vision \
//!     cargo test -p rlx-diffusiongemma --test real_vision -- --nocapture
//! ```
//!
//! Skips when the env var is unset. Unlike the synthetic parity fixture, this
//! runs the trained weights at the real geometry — 27 layers, 1152 hidden,
//! 16 heads × 72, a 10240-entry position table — so it exercises magnitudes and
//! numerics the tiny config cannot.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::preprocess::{ImagePreprocessConfig, preprocess_image};
use rlx_diffusiongemma::vision::{
    ENCODER_TAP, PATCH_EMBED_TAP, PIXELS_INPUT, POOL_INPUT, POOLED_TAP, POS_X_INPUT, POS_Y_INPUT,
    ROPE_COS_INPUT, ROPE_SIN_INPUT, SOFT_TOKENS_OUTPUT, VALID_INPUT,
};
use rlx_diffusiongemma::{
    DiffusionGemmaConfig, build_vision_flow, grid_positions, vision_pool_matrix, vision_rope_tables,
};
use rlx_runtime::Device;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("RLX_DG_REAL_VISION_DIR").ok()?);
    assert!(
        d.join("real_meta.json").is_file(),
        "RLX_DG_REAL_VISION_DIR={d:?} has no real_meta.json — \
         run scripts/diffusiongemma_real_vision.py first"
    );
    Some(d)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let raw = std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("reading {name}.bin: {e}"));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// Real activations run to magnitude ~2000, so an absolute tolerance says
/// nothing. Compare relative to the tensor's own scale, and report the spread
/// so a few outliers are distinguishable from systematic drift.
fn report(label: &str, got: &[f32], want: &[f32], rel_tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let cos = cosine(got, want);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mad = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mean_abs: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / got.len() as f64;
    let rel = mad / scale;
    let mean_rel = mean_abs as f32 / scale;
    println!(
        "  {label:<22} cos {cos:.8}  rel max {rel:.2e}  rel mean {mean_rel:.2e}  (|x|max {scale:.4})"
    );
    assert!(cos > 0.99999, "{label}: cosine {cos:.8}");
    assert!(
        rel <= rel_tol,
        "{label}: relative max {rel:.2e} > {rel_tol:.2e} (abs {mad:.3e} on scale {scale:.4})"
    );
}

#[test]
fn vision_tower_on_real_weights_matches_torch() {
    let Some(d) = dir() else {
        eprintln!("skipping: set RLX_DG_REAL_VISION_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("real_meta.json")).unwrap()).unwrap();
    let g = |k: &str| meta[k].as_u64().unwrap() as usize;
    let (patches, soft_len) = (g("patches"), g("soft_tokens"));
    let (cols, rows) = (g("grid_cols"), g("grid_rows"));

    let cfg = DiffusionGemmaConfig::from_file(d.join("config.json")).expect("real config");
    let v = cfg.vision_config.as_ref().expect("vision config");
    assert_eq!(
        (v.hidden_size, v.num_hidden_layers, v.head_dim),
        (1152, 27, 72),
        "this is the real tower geometry"
    );

    let wm = WeightMap::from_safetensors_dir(&d).expect("real vision weights");
    println!("loaded {} real tensors", wm.len());

    let positions = grid_positions(cols, rows);
    assert_eq!(positions.len(), patches);
    let (cos, sin) = vision_rope_tables(v, &positions);
    let pool = vision_pool_matrix(&positions, v.pooling_kernel_size, soft_len);
    report("pool matrix", &pool, &read_f32(&d, "real_pool"), 1e-6);

    let pixels = read_f32(&d, "real_pixels");
    let pos_x: Vec<f32> = positions.iter().map(|p| p.0 as f32).collect();
    let pos_y: Vec<f32> = positions.iter().map(|p| p.1 as f32).collect();
    let valid = vec![1f32; patches];

    let built = build_vision_flow(&cfg, &wm, patches, soft_len).expect("build real vision flow");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile real vision flow");
    let outs = compiled.run(&[
        (PIXELS_INPUT, pixels.as_slice()),
        (POS_X_INPUT, pos_x.as_slice()),
        (POS_Y_INPUT, pos_y.as_slice()),
        (ROPE_COS_INPUT, cos.as_slice()),
        (ROPE_SIN_INPUT, sin.as_slice()),
        (VALID_INPUT, valid.as_slice()),
        (POOL_INPUT, pool.as_slice()),
    ]);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();

    println!("real vision tower, {patches} patches -> {soft_len} soft tokens:");
    report(
        "patch_embed",
        by[PATCH_EMBED_TAP],
        &read_f32(&d, "real_patch_embed"),
        1e-5,
    );
    report(
        "encoder_out",
        by[ENCODER_TAP],
        &read_f32(&d, "real_encoder_out"),
        2e-2,
    );
    report("pooled", by[POOLED_TAP], &read_f32(&d, "real_pooled"), 2e-2);
    report(
        "soft_tokens",
        by[SOFT_TOKENS_OUTPUT],
        &read_f32(&d, "real_soft_tokens"),
        2e-2,
    );

    // Trained weights produce real structure, not a collapsed constant.
    let soft = by[SOFT_TOKENS_OUTPUT];
    assert!(soft.iter().all(|x| x.is_finite()));
    let absmax = soft.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(absmax > 1.0, "soft tokens look collapsed (absmax {absmax})");
}

/// The Rust preprocessor, end to end on the same real image: resize, rescale
/// and patchify must reproduce the tensor the torch reference built with PIL.
#[test]
fn preprocessor_on_a_real_image_matches_the_reference_tensor() {
    let Some(d) = dir() else {
        eprintln!("skipping: set RLX_DG_REAL_VISION_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("real_meta.json")).unwrap()).unwrap();
    let g = |k: &str| meta[k].as_u64().unwrap() as usize;
    let (src_h, src_w) = (g("src_h"), g("src_w"));

    let cfg = ImagePreprocessConfig {
        patch_size: g("patch_size"),
        max_soft_tokens: g("max_soft_tokens"),
        pooling_kernel_size: g("pooling_kernel_size"),
    };
    let img = std::fs::read(d.join("real_image.bin")).expect("real_image.bin");
    assert_eq!(img.len(), src_h * src_w * 3);

    let p = preprocess_image(&img, src_h, src_w, cfg).expect("preprocess");
    assert_eq!(p.size, (g("target_h"), g("target_w")), "resize target");
    assert_eq!(p.grid, (g("grid_cols"), g("grid_rows")));
    assert_eq!(p.num_patches, g("patches"));
    assert_eq!(p.num_soft_tokens, g("soft_tokens"));

    // The reference tensor is unpadded; ours is padded out to the budget.
    let want = read_f32(&d, "real_pixels");
    let n = p.num_patches * cfg.patch_dim();
    assert_eq!(want.len(), n);
    let got = &p.pixels[..n];
    let mad = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("  preprocessed pixels    max|Δ| {mad:.3e} over {n} values");
    // A 1-LSB resampler difference is 1/255; anything larger is a real bug.
    assert!(mad <= 1.0 / 255.0 + 1e-6, "pixel mismatch: max|Δ| {mad}");
    assert!(
        p.pixels[n..].iter().all(|&x| x == 0.0),
        "padding must be zero"
    );
}
