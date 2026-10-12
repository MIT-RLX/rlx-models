//! Which op kind breaks under bf16? Enable BF16 for one kind at a time.
use rlx_compile::precision::{AutoMixedPrecision, OpKind, Precision, PrecisionPolicy};
use rlx_fusion::pass::Pass;
use rlx_ir::DType;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let dev = std::env::args().nth(1).map_or(Device::Cpu, |d| {
        rlx_runtime::parse_device(&d).expect("device")
    });
    eprintln!("device: {dev:?}");
    let cfg = rlx_qwen35::synth::tiny_cfg();
    let w = rlx_qwen35::synth::synth_weights(&cfg);
    let (g, params, _) = rlx_qwen35::build_qwen35_graph_sized_ext(
        &cfg, w, 1, 64, false, false, false, false, None, true, false, true, false,
    )?;
    let ids: Vec<f32> = (0..64).map(|i| (i % 20 + 1) as f32).collect();
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, 64, None, cfg.rope_dim_count / 2);

    let kinds = [
        ("elementwise", OpKind::Elementwise),
        ("reduction", OpKind::Reduction),
        ("compute", OpKind::Compute),
        ("datamovement", OpKind::DataMovement),
    ];
    let base: std::cell::OnceCell<Vec<f32>> = std::cell::OnceCell::new();
    let check = |label: &str, policy: PrecisionPolicy| {
        let gg = AutoMixedPrecision::new(policy).run(g.clone());
        let bf = gg
            .nodes()
            .iter()
            .filter(|n| n.shape.dtype() == DType::BF16)
            .count();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut c = Session::new(dev).compile(gg);
            for (n, v) in &params {
                c.set_param(n, v);
            }
            let out = c.run(&[
                ("input_ids", &ids[..]),
                ("rope_cos", &cos[..]),
                ("rope_sin", &sin[..]),
            ]);
            out[0].clone()
        }));
        match r {
            Ok(v) => {
                let finite = v.iter().all(|x| x.is_finite());
                let note = match base.get() {
                    None => {
                        let _ = base.set(v.clone());
                        "baseline".to_string()
                    }
                    Some(b) => {
                        let d = b
                            .iter()
                            .zip(&v)
                            .map(|(a, c)| (a - c).abs())
                            .fold(0f32, f32::max);
                        format!("max|d| vs f32 = {d:.4}")
                    }
                };
                if finite {
                    println!("  {label:14} {bf:4} bf16 nodes  finite   {note}");
                } else {
                    println!("  {label:14} {bf:4} bf16 nodes  *** NaN/Inf ***");
                }
            }
            Err(_) => println!("  {label:14} {bf:4} bf16 nodes  *** PANIC ***"),
        }
    };

    check("f32 baseline", PrecisionPolicy::AlwaysF32);
    for (label, kind) in kinds {
        let mut m: HashMap<OpKind, Precision> = HashMap::new();
        m.insert(kind, Precision::BF16);
        check(label, PrecisionPolicy::Custom(m));
    }
    check("AutoMixedBf16", PrecisionPolicy::AutoMixedBf16);
    Ok(())
}
