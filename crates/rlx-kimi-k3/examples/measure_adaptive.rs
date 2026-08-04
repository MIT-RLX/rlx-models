//! `measure_adaptive` — sweep representative REAL KDA layers across depth and
//! measure the per-layer PRECISION cost of per-channel int8 vs int4-g64, then show
//! the `RLX_KIMI_QUANT=adaptive` schedule's per-layer choice (int8 on the
//! fp32-sensitive AttnRes-snapshot / near-head hotspots, int4 on the mild
//! mid-depth layers) and the error IT would incur. This validates the adaptive
//! depth schedule with direct output error, not just the recording's outlier
//! ratios — the data behind mixing int4 into the backbone.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example measure_adaptive [-- model_dir]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Philox4x32, Shape};
use rlx_kimi_k3::common::is_quant_hotspot;
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

/// (relative-L2, SNR-dB) of `q` vs `base`.
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
    let (hidden, n) = (tc.hidden_size, tc.num_hidden_layers);
    let seq = 8;
    let d = dims(hidden, seq);

    // representative KDA layers spanning depth (skip the MLA slots).
    let sample: Vec<usize> = [0usize, 6, 12, 18, 24, 36, 48, 60, 72, 84, n - 1]
        .into_iter()
        .filter(|&i| i < n && tc.is_kda_layer(i))
        .collect();

    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let mut rng = Philox4x32::new(0xADAB7);

    eprintln!(
        "\nKimi-K3 per-layer backbone-quant PRECISION sweep ({} KDA layers, seq={seq}), n_layers={n}\n\
         adaptive: int8 on hotspots (i%12==0 or i>=n-4), else int4-g64\n\n\
         {:>4}  {:>8}  {:>10} {:>8}   {:>10} {:>8}   {:<8} {:>10} {:>8}",
        sample.len(),
        "L",
        "role",
        "int8 relL2",
        "SNR",
        "int4 relL2",
        "SNR",
        "adaptive",
        "relL2",
        "SNR"
    );

    let (mut sum_i8, mut sum_ad, mut cnt) = (0f64, 0f64, 0usize);
    for &i in &sample {
        let w = match ck.load_kda(&format!("language_model.model.layers.{i}"), d) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  L{i:<3} load failed: {e}");
                continue;
            }
        };
        let mut x = vec![0f32; seq * hidden];
        rng.fill_normal(&mut x);

        let base = run(&w, d, &x, "off");
        let (r8, s8) = err(&base, &run(&w, d, &x, "int8"));
        let (r4, s4) = err(&base, &run(&w, d, &x, "int4"));

        let hot = is_quant_hotspot(i, n);
        let (ad_name, ad_r, ad_s) = if hot {
            ("int8", r8, s8)
        } else {
            ("int4", r4, s4)
        };
        eprintln!(
            "{i:>4}  {:>8}  {r8:>10.3e} {s8:>8.2}   {r4:>10.3e} {s4:>8.2}   {ad_name:<8} {ad_r:>10.3e} {ad_s:>8.2}",
            if hot { "HOTSPOT" } else { "mild" }
        );
        sum_i8 += r8 as f64;
        sum_ad += ad_r as f64;
        cnt += 1;
    }
    if cnt > 0 {
        let n_int4 = sample.iter().filter(|&&i| !is_quant_hotspot(i, n)).count();
        eprintln!(
            "\nmean relL2: all-int8 {:.3e}   adaptive {:.3e}   ({}/{} sampled layers on int4)\n\
             adaptive backbone bytes ≈ int8 on {} hotspots + int4 on {} mild layers → ~2×→~3× vs bf16 where int4 applies",
            sum_i8 / cnt as f64,
            sum_ad / cnt as f64,
            n_int4,
            cnt,
            cnt - n_int4,
            n_int4
        );
    }
    Ok(())
}
