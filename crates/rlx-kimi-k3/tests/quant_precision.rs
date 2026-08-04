//! Measure the PRECISION of the `RLX_KIMI_QUANT` backbone weight-quant modes
//! (per-channel int8 / per-tensor int8 / int4-g64 fake-quant) against the bf16
//! baseline, on a REAL Kimi-K3 KDA layer. Reports max|Δ|, relative-L2, and SNR
//! (dB) of the layer output — the direct precision cost of each scheme the
//! dataflow recording said the weights can tolerate. Skips if unmounted.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Philox4x32, Shape};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

const CKPT: &str = "/Volumes/FOUR/kimi";

fn dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: 7168,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

fn run(w: &KdaWeights, d: KdaDims, x: &[f32], mode: &str) -> Vec<f32> {
    unsafe { std::env::set_var("RLX_KIMI_QUANT", mode) };
    let mut hir = HirModule::new("kda");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_kda_layer(&mut g, &mut p, "kda", hin, w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    c.run(&[("h", x)]).remove(0)
}

fn err(base: &[f32], q: &[f32]) -> (f32, f32, f32) {
    let mut sd = 0f64;
    let mut sb = 0f64;
    let mut mx = 0f32;
    for (b, v) in base.iter().zip(q) {
        let e = (b - v).abs();
        mx = mx.max(e);
        sd += (e as f64) * (e as f64);
        sb += (*b as f64) * (*b as f64);
    }
    let rel = (sd / sb.max(1e-30)).sqrt() as f32;
    let snr = if sd > 0.0 {
        10.0 * (sb / sd).log10() as f32
    } else {
        f32::INFINITY
    };
    (mx, rel, snr)
}

#[test]
fn kda_weight_quant_precision() {
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip: {CKPT} not mounted");
        return;
    }
    let seq = 8;
    let d = dims(seq);
    let kc = KimiK3Config::load(Path::new(CKPT).join("config.json")).unwrap();
    // layer 0 is KDA.
    let mut ck = CheckpointLoader::open(CKPT).unwrap();
    let _ = &kc;
    let w = ck
        .load_kda("language_model.model.layers.0", d)
        .expect("load kda 0");

    let mut rng = Philox4x32::new(0xBEEF);
    let mut x = vec![0f32; seq * d.hidden];
    rng.fill_normal(&mut x);

    let base = run(&w, d, &x, "off");
    unsafe { std::env::remove_var("RLX_KIMI_QUANT") };
    assert!(base.iter().all(|v| v.is_finite()));

    // per-mode: compression is theoretical (int8 ~2× vs bf16, int4 ~4×).
    eprintln!(
        "\nKDA layer-0 weight-quant PRECISION (vs bf16 baseline), seq={seq}:\n\
         {:<8} {:>10} {:>10} {:>9} {:>8}",
        "mode", "max|Δ|", "rel-L2", "SNR(dB)", "~x vs bf16"
    );
    for (mode, comp) in [("int8", "2.0"), ("int8t", "2.0"), ("int4", "3.6")] {
        let q = run(&w, d, &x, mode);
        unsafe { std::env::remove_var("RLX_KIMI_QUANT") };
        let (mx, rel, snr) = err(&base, &q);
        assert!(
            q.iter().all(|v| v.is_finite()),
            "{mode} produced non-finite"
        );
        eprintln!("{mode:<8} {mx:>10.3e} {rel:>10.3e} {snr:>9.2} {comp:>8}");
        // int8 per-channel must be high-fidelity (the recording said outliers are per-channel).
        if mode == "int8" {
            assert!(rel < 5e-2, "int8 per-channel rel-L2 {rel:.3e} too high");
        }
    }
}
