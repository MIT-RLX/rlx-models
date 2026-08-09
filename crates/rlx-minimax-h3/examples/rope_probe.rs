//! Isolate `rope_n` (partial RoPE) across backends.

use rlx_flow::{CompileProfile, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, seq: usize, nh: usize, hd: usize, n_rot: usize) -> Vec<f32> {
    let f = DType::F32;
    let half = n_rot / 2;
    let inner = nh * hd;
    let flow = ModelFlow::new("rope_probe")
        .with_profile(CompileProfile::encoder())
        .input("x", Shape::new(&[1, seq, inner], f))
        .input("cos", Shape::new(&[seq, half], f))
        .input("sin", Shape::new(&[seq, half], f))
        .stage(plugin_named("r", move |emit, _h| {
            let x = emit.flow_input("x")?;
            let cos = emit.flow_input("cos")?;
            let sin = emit.flow_input("sin")?;
            let mut gb = HirMut::new(emit.hir());
            let y = if std::env::var("ROPE_WORKAROUND").is_ok() && n_rot != hd {
                // Slice the rotated channels of every head into their own
                // contiguous tensor, rotate with a FULL rope whose head_dim is
                // n_rot, and glue the untouched tail back on.
                let x4 = gb.reshape_(x.hir_id(), vec![1, seq as i64, nh as i64, hd as i64]);
                let rot = gb.narrow_(x4, 3, 0, n_rot);
                let pass = gb.narrow_(x4, 3, n_rot, hd - n_rot);
                let rot3 = gb.reshape_(rot, vec![1, seq as i64, (nh * n_rot) as i64]);
                let rotated = gb.rope(rot3, cos.hir_id(), sin.hir_id(), n_rot);
                let rotated4 = gb.reshape_(rotated, vec![1, seq as i64, nh as i64, n_rot as i64]);
                let joined = gb.concat_(vec![rotated4, pass], 3);
                gb.reshape_(joined, vec![1, seq as i64, inner as i64])
            } else {
                gb.rope_n(x.hir_id(), cos.hir_id(), sin.hir_id(), hd, n_rot)
            };
            Ok(Some(emit.wrap(y, Shape::new(&[1, seq, inner], f))))
        }))
        .output("out");
    let mut empty = rlx_core::weight_map::WeightMap::from_tensors(Default::default());
    let built = flow
        .build_with(&mut rlx_core::flow_util::WeightMapSource(&mut empty), None)
        .unwrap();
    let typed = built.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built).unwrap();
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut c = Session::new(device).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut c, params, &typed);

    let x: Vec<f32> = (0..seq * inner)
        .map(|i| ((i % 23) as f32 / 23.0) - 0.5)
        .collect();
    let cos: Vec<f32> = (0..seq * half).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin: Vec<f32> = (0..seq * half).map(|i| (i as f32 * 0.1).sin()).collect();
    c.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let s = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / s))
}

fn main() {
    #[allow(unused_mut)]
    let mut devices: Vec<(&str, Device)> = Vec::new();
    #[cfg(feature = "metal")]
    devices.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    devices.push(("mlx", Device::Mlx));
    #[cfg(feature = "gpu")]
    devices.push(("wgpu", Device::Gpu));

    // (head_dim, n_rot): full rotation vs partial.
    for (hd, n_rot) in [
        (16usize, 16usize),
        (16, 12),
        (128, 128),
        (128, 96),
        (64, 48),
    ] {
        let r = run(Device::Cpu, 6, 2, hd, n_rot);
        let mut line = format!(
            "hd={hd:3} n_rot={n_rot:3} {}:",
            if hd == n_rot { "full   " } else { "partial" }
        );
        for (name, d) in &devices {
            let g = run(*d, 6, 2, hd, n_rot);
            line.push_str(&format!("  {name}={:.2e}", rel(&r, &g)));
        }
        println!("{line}");
    }
}
