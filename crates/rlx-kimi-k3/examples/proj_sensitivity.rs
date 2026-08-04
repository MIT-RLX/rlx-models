//! `proj_sensitivity` — the real fix for 4-bit KDA. Per-matrix int4 is fine
//! (~27 dB, see `awq_quant`); the ~24% full-layer error is ERROR AMPLIFICATION
//! through the gated-delta-net recurrence when ALL 8 projections are int4. So the
//! fix is per-PROJECTION: int8 only the projections whose error the recurrence
//! amplifies, int4 the rest. This quantizes ONE projection at a time to int4
//! (others f32), runs the REAL layer-0 KDA over REAL normed activations, and ranks
//! each projection's contribution to the full-layer output error — then validates a
//! mixed recipe (int8 on the worst-K, int4 elsewhere) against uniform int4/int8.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example proj_sensitivity [-- model_dir]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::common::{WeightQuant, fake_quant_weight};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

const EMB: &str = "language_model.model.embed_tokens.weight";

/// (name, [K,N]) of every KDA matmul weight in `KdaWeights`.
fn proj_dims(hidden: usize, proj: usize, hd: usize, h: usize) -> Vec<(&'static str, usize, usize)> {
    vec![
        ("q_proj", hidden, proj),
        ("k_proj", hidden, proj),
        ("v_proj", hidden, proj),
        ("g_proj", hidden, proj),
        ("f_a", hidden, hd),
        ("f_b", hd, proj),
        ("b_proj", hidden, h),
        ("o_proj", proj, hidden),
    ]
}

/// Clone `base`, applying `plan[name]` (default None) to each matmul field.
fn quantized(
    base: &KdaWeights,
    dims: &[(&str, usize, usize)],
    plan: &HashMap<&str, WeightQuant>,
) -> KdaWeights {
    let mut w = base.clone();
    let q = |name: &str, v: &mut Vec<f32>, k: usize, n: usize| {
        if let Some(&m) = plan.get(name) {
            if m != WeightQuant::None {
                *v = fake_quant_weight(v, k, n, m);
            }
        }
    };
    for &(name, k, n) in dims {
        match name {
            "q_proj" => q(name, &mut w.q_proj, k, n),
            "k_proj" => q(name, &mut w.k_proj, k, n),
            "v_proj" => q(name, &mut w.v_proj, k, n),
            "g_proj" => q(name, &mut w.g_proj, k, n),
            "f_a" => q(name, &mut w.f_a, k, n),
            "f_b" => q(name, &mut w.f_b, k, n),
            "b_proj" => q(name, &mut w.b_proj, k, n),
            "o_proj" => q(name, &mut w.o_proj, k, n),
            _ => {}
        }
    }
    w
}

fn run_layer(w: &KdaWeights, d: KdaDims, x: &[f32]) -> Vec<f32> {
    unsafe { std::env::set_var("RLX_KIMI_QUANT", "off") };
    let mut hir = HirModule::new("kda");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_kda_layer(&mut g, &mut p, "kda", hin, w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    c.run(&[("h", x)]).remove(0)
}

fn rel(base: &[f32], q: &[f32]) -> f32 {
    let (mut sd, mut sb) = (0f64, 0f64);
    for (b, v) in base.iter().zip(q) {
        let e = (*b - *v) as f64;
        sd += e * e;
        sb += (*b as f64) * (*b as f64);
    }
    (sd / sb.max(1e-30)).sqrt() as f32
}

fn rmsnorm(x: &mut [f32], m: usize, k: usize, w: &[f32]) {
    for r in 0..m {
        let row = &mut x[r * k..r * k + k];
        let ms: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        for (v, &g) in row.iter_mut().zip(w) {
            *v = (*v as f64 * inv) as f32 * g;
        }
    }
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
    let (hidden, vocab) = (tc.hidden_size, tc.vocab_size);
    let (h, hd, seq) = (96usize, 128usize, 8usize);
    let proj = h * hd;
    let d = KdaDims {
        hidden,
        num_heads: h,
        head_dim: hd,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    };

    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let base_w = ck
        .load_kda("language_model.model.layers.0", d)
        .map_err(|e| e.to_string())?;
    let ln = ck
        .tensor_f32("language_model.model.layers.0.input_layernorm.weight")
        .map_err(|e| e.to_string())?;

    // real normed activations for a seq of real tokens.
    let toks: Vec<u32> = (0..seq as u32)
        .map(|i| (i.wrapping_mul(2657).wrapping_add(13)) % vocab as u32)
        .collect();
    let mut x = ck
        .gather_embed(EMB, &toks, hidden)
        .map_err(|e| e.to_string())?;
    rmsnorm(&mut x, seq, hidden, &ln);

    let dims = proj_dims(hidden, proj, hd, h);
    let base = run_layer(&base_w, d, &x);

    // (A) single-projection int4 sensitivity: which one, alone, hurts the layer most.
    eprintln!("\nKDA layer-0 per-projection int4 sensitivity (real normed input, seq={seq})");
    eprintln!("full-layer relL2 with ONLY that projection int4-g64 (rest f32):\n");
    let mut single: Vec<(&str, f32)> = Vec::new();
    for &(name, _, _) in &dims {
        let plan = HashMap::from([(name, WeightQuant::Int4G64)]);
        let w = quantized(&base_w, &dims, &plan);
        let r = rel(&base, &run_layer(&w, d, &x));
        single.push((name, r));
    }
    single.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, r) in &single {
        eprintln!("  {name:<8} {r:>10.3e}");
    }

    // (B) uniform baselines + mixed recipe (int8 on the worst-K, int4 on the rest).
    let all = |q: WeightQuant| -> HashMap<&str, WeightQuant> {
        dims.iter().map(|&(n, _, _)| (n, q)).collect()
    };
    let r_all4 = rel(
        &base,
        &run_layer(
            &quantized(&base_w, &dims, &all(WeightQuant::Int4G64)),
            d,
            &x,
        ),
    );
    let r_all8 = rel(
        &base,
        &run_layer(&quantized(&base_w, &dims, &all(WeightQuant::Int8Ch)), d, &x),
    );

    eprintln!("\nuniform:  all-int4 {r_all4:.3e}   all-int8 {r_all8:.3e}");
    eprintln!("\nmixed recipe (int8 on worst-K sensitive, int4 rest):");
    for kk in [1usize, 2, 3] {
        let mut plan = all(WeightQuant::Int4G64);
        let (mut i8w, mut totw) = (0f64, 0f64);
        // weight-size per projection ~ K*N; compute avg bits.
        let szs: HashMap<&str, f64> = dims
            .iter()
            .map(|&(n, k, nn)| (n, (k * nn) as f64))
            .collect();
        for &(name, _) in single.iter().take(kk) {
            plan.insert(name, WeightQuant::Int8Ch);
        }
        for &(n, _, _) in &dims {
            let s = szs[n];
            totw += s;
            if plan[n] == WeightQuant::Int8Ch {
                i8w += s;
            }
        }
        let avg_bits = 4.0 + 4.0 * (i8w / totw);
        let names: Vec<&str> = single.iter().take(kk).map(|(n, _)| *n).collect();
        let r = rel(&base, &run_layer(&quantized(&base_w, &dims, &plan), d, &x));
        eprintln!(
            "  int8 on {:<26} relL2 {r:>10.3e}   (~{avg_bits:.2} avg bits)",
            format!("{names:?}")
        );
    }
    eprintln!("\ngoal: a mixed recipe near all-int8 ({r_all8:.3e}) at well under 8 bits.");
    Ok(())
}
