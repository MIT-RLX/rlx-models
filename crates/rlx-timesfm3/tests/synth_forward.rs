// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use ndarray::Array3;
use rlx_timesfm3::{TimesFM3Config, TimesFM3Model};

#[test]
fn synth_forward_finite() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg.clone(), 99);
    let ctx: Vec<f32> = (0..cfg.input_patch_len * 4)
        .map(|i| (i as f32 * 0.07).sin())
        .collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
    let out = model.decode(arr.view(), 32, None, None, None);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(out.shape()[2], 32);
}
