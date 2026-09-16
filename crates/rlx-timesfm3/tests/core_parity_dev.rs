//! Strict host vs compiled parity (requires `dev` feature).

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timesfm3::host::preprocess::build_decode_inputs;
use rlx_timesfm3::{TimesFM3Config, TimesFM3Model, TimesFM3Session, compare_core};

/// Bit-identical on CPU.
const EXACT: f32 = 0.0;
/// Compiled transformer core vs host on synthetic tiny (~1% max on CPU today).
const COMPILED_CORE_MAX: f32 = 1.2e-2;

#[test]
fn resblock_exact_parity_cpu() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg, 99);
    let ctx: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
    let (values, masks, pit, cpm, _, _) =
        build_decode_inputs(&model.cfg, arr.view(), 12, None, None, None);
    let prep = model
        .prepare_core(values.view(), masks.view(), pit.view(), Some(cpm.row(0)))
        .unwrap();
    let (_, _, diff) = rlx_timesfm3::parity::compare_resblock(&model, &prep, Device::Cpu).unwrap();
    assert_eq!(diff, EXACT, "resblock max diff {diff}");
}

#[test]
fn compiled_core_tracks_host_cpu() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg, 99);
    let ctx: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
    let (values, masks, pit, cpm, _, _) =
        build_decode_inputs(&model.cfg, arr.view(), 12, None, None, None);
    let prep = model
        .prepare_core(values.view(), masks.view(), pit.view(), Some(cpm.row(0)))
        .unwrap();
    let (_host, _compiled, diff) = compare_core(&model, &prep, Device::Cpu).unwrap();
    assert!(
        diff <= COMPILED_CORE_MAX,
        "compiled core max diff {diff} (limit {COMPILED_CORE_MAX})"
    );
}

#[test]
fn decode_exact_parity_cpu() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg, 99);
    let ctx: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();

    let host = model.decode(arr.view(), 12, None, None, None);

    let mut session = TimesFM3Session::from_model(
        TimesFM3Model::synth(TimesFM3Config::synth_tiny(), 99),
        Device::Cpu,
    );
    let session_out = session.decode(arr.view(), 12, None, None, None).unwrap();

    let max_diff = host
        .iter()
        .zip(session_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_diff, EXACT, "decode max diff {max_diff}");
}

#[test]
fn compiled_core_tracks_host_seeds_and_contexts() {
    for seed in [1_u64, 7, 42, 99, 123] {
        for ctx_len in [64_usize, 128, 256] {
            let cfg = TimesFM3Config::synth_tiny();
            let model = TimesFM3Model::synth(cfg, seed);
            let ctx: Vec<f32> = (0..ctx_len)
                .map(|i| (i as f32 * 0.03 + seed as f32).sin())
                .collect();
            let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();
            let (values, masks, pit, cpm, _, _) =
                build_decode_inputs(&model.cfg, arr.view(), 8, None, None, None);
            let prep = model
                .prepare_core(values.view(), masks.view(), pit.view(), Some(cpm.row(0)))
                .unwrap();
            let (_, _, diff) = compare_core(&model, &prep, Device::Cpu).unwrap();
            assert!(
                diff <= COMPILED_CORE_MAX,
                "seed={seed} ctx={ctx_len} compiled core diff {diff} (limit {COMPILED_CORE_MAX})"
            );
        }
    }
}
