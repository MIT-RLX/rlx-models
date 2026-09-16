//! Host vs compiled-core smoke check (loose tolerance).

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timesfm3::{TimesFM3Config, TimesFM3Model, TimesFM3Session};

#[test]
fn host_near_compiled_decode_cpu() {
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg, 99);
    let ctx: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let arr = Array3::from_shape_vec((1, 1, ctx.len()), ctx).unwrap();

    let host = model.decode(arr.view(), 12, None, None, None);
    let mut session = TimesFM3Session::from_model(
        TimesFM3Model::synth(TimesFM3Config::synth_tiny(), 99),
        Device::Cpu,
    );
    let compiled = session.decode(arr.view(), 12, None, None, None).unwrap();

    let max_diff = host
        .iter()
        .zip(compiled.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-2,
        "host vs compiled decode max diff {max_diff}"
    );
}
