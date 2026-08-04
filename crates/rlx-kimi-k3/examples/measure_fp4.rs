//! `measure_fp4` — can a 4-bit scheme actually REPLACE per-channel int8 in the
//! Kimi-K3 backbone? Symmetric int4-g64 fails (~24% at every depth) because the
//! per-channel outliers crush its group scale. This compares the real fixes on a
//! hotspot (L0) and a mild layer (L6), each loaded once:
//!   int8   — per-channel int8 (the bar to beat, ~2–4%)
//!   int4   — symmetric int4-g64 (the broken baseline, ~24%)
//!   mxfp4  — FP4 e2m1 + e8m0 block-32 (the model's own expert format)
//!   int4mix— top-1/8 outlier channels int8, rest int4-g64 (~4.5 avg bits)
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example measure_fp4 [-- model_dir]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Philox4x32, Shape};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

fn dims(hidden: usize, seq: usize) -> KdaDims {
    KdaDims {
        hidden,
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
    let y = c.run(&[("h", x)]).remove(0);
    unsafe { std::env::remove_var("RLX_KIMI_QUANT") };
    y
}

fn err(base: &[f32], q: &[f32]) -> (f32, f32) {
    let (mut sd, mut sb) = (0f64, 0f64);
    for (b, v) in base.iter().zip(q) {
        let e = (*b - *v) as f64;
        sd += e * e;
        sb += (*b as f64) * (*b as f64);
    }
    let rel = (sd / sb.max(1e-30)).sqrt() as f32;
    let snr = if sd > 0.0 {
        10.0 * (sb / sd).log10() as f32
    } else {
        f32::INFINITY
    };
    (rel, snr)
}

fn main() -> Result<(), String> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Volumes/FOUR/kimi".into());
    if !Path::new(&model_dir).join("config.json").exists() {
        eprintln!("skip: {model_dir}/config.json not found");
        return Ok(());
    }
    let kc =
        KimiK3Config::load(Path::new(&model_dir).join("config.json")).map_err(|e| e.to_string())?;
    let tc = &kc.text_config;
    let (hidden, seq) = (tc.hidden_size, 8usize);
    let d = dims(hidden, seq);
    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let mut rng = Philox4x32::new(0xF4F4);

    let modes = ["int8", "int4", "mxfp4", "int4mix", "nf4"];
    let bits = ["8.0", "4.1", "4.25", "~4.5", "4.1"];
    eprintln!(
        "\nKimi-K3 backbone 4-bit-fix comparison (real KDA layers, seq={seq})\n\
         {:>4} {:<8} {:>8} {:>7} {:>10}",
        "L", "mode", "relL2", "SNRdB", "~bits/w"
    );
    for &i in &[0usize, 6] {
        let w = match ck.load_kda(&format!("language_model.model.layers.{i}"), d) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  L{i} load failed: {e}");
                continue;
            }
        };
        let mut x = vec![0f32; seq * hidden];
        rng.fill_normal(&mut x);
        let base = run(&w, d, &x, "off");
        for (m, b) in modes.iter().zip(bits) {
            let (r, s) = err(&base, &run(&w, d, &x, m));
            eprintln!("{i:>4} {m:<8} {r:>8.3e} {s:>7.2} {b:>10}");
        }
        eprintln!();
    }
    eprintln!(
        "goal: a 4-bit row with relL2 near int8's → a real ~2× streaming win without int8's byte cost."
    );
    Ok(())
}
