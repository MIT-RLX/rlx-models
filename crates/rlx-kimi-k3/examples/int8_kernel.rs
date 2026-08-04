//! `int8_kernel` — the NATIVE int8 matmul (not fake-quant): store the backbone
//! weight as int8 `[K,N]` + one f32 scale per output column and run the tested
//! `Op::DequantMatMul{Int8Block, block_size=K}` kernel, which dequantizes on the
//! fly. This is what turns the backbone-quant *precision* result into an actual
//! bandwidth win: the weight is ¼ the f32 bytes (½ of bf16). Verifies parity vs
//! the f32 matmul + the fake-quant int8 (same numbers), and measures weight RAM +
//! matmul timing on a REAL Kimi-K3 `q_proj`.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example int8_kernel [-- model_dir]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Philox4x32, QuantScheme, Shape};
use rlx_kimi_k3::common::{WeightQuant, fake_quant_weight};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::KdaDims;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Per-output-channel symmetric int8: `w_q[k,n]=round(w[k,n]/s[n])`, `s[n]=amax_n/127`.
fn quantize_int8_col(w: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    let mut scale = vec![0f32; n];
    let mut wq = vec![0i8; k * n];
    for col in 0..n {
        let mut amax = 0f32;
        for row in 0..k {
            amax = amax.max(w[row * n + col].abs());
        }
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[col] = s;
        for row in 0..k {
            wq[row * n + col] = (w[row * n + col] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (wq, scale)
}

fn f32_matmul(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut hir = HirModule::new("mm");
    let mut g = HirMut::new(&mut hir);
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut p = HashMap::new();
    p.insert("w".to_string(), w.to_vec());
    let wi = g.param("w", Shape::new(&[k, n], DType::F32));
    let out = g.mm(xi, wi);
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    // warm + timed
    let _ = c.run(&[("x", x)]);
    let t = Instant::now();
    let y = c.run(&[("x", x)]).remove(0);
    eprintln!("  f32 matmul:  {:?}", t.elapsed());
    y
}

fn int8_kernel(x: &[f32], m: usize, k: usize, wq: &[i8], scale: &[f32], n: usize) -> Vec<f32> {
    let mut hir = HirModule::new("dq");
    let mut g = HirMut::new(&mut hir);
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wi = g.param("w", Shape::new(&[k, n], DType::I8));
    let si = g.param("scale", Shape::new(&[1, n], DType::F32));
    let zi = g.param("zp", Shape::new(&[1, n], DType::F32));
    let out = g.0.dequant_matmul(
        xi,
        wi,
        Some(si),
        Some(zi),
        QuantScheme::Int8Block {
            block_size: k as u32,
        },
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![out]);
    // scale + zp are f32 params (baked); the i8 weight is fed typed post-compile.
    let mut p = HashMap::new();
    p.insert("scale".to_string(), scale.to_vec());
    p.insert("zp".to_string(), vec![0f32; n]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let wbytes: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
    c.set_param_typed("w", &wbytes, DType::I8);
    let _ = c.run(&[("x", x)]);
    let t = Instant::now();
    let y = c.run(&[("x", x)]).remove(0);
    eprintln!("  int8 kernel: {:?}", t.elapsed());
    y
}

fn med3<F: FnMut()>(mut f: F) -> f64 {
    let mut ts = [0f64; 3];
    for t in ts.iter_mut() {
        let s = Instant::now();
        f();
        *t = s.elapsed().as_secs_f64();
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[1]
}

/// Compile an f32 matmul once on `dev`; return a closure that runs it (for timing).
fn f32_runner(
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    dev: Device,
) -> impl FnMut(&[f32]) -> Vec<f32> {
    let mut hir = HirModule::new("mm");
    let mut g = HirMut::new(&mut hir);
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut p = HashMap::new();
    p.insert("w".to_string(), w.to_vec());
    let wi = g.param("w", Shape::new(&[k, n], DType::F32));
    let out = g.mm(xi, wi);
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), dev).unwrap();
    move |x: &[f32]| c.run(&[("x", x)]).remove(0)
}

fn int8_runner(
    m: usize,
    k: usize,
    wq: &[i8],
    scale: &[f32],
    n: usize,
    dev: Device,
) -> impl FnMut(&[f32]) -> Vec<f32> {
    let mut hir = HirModule::new("dq");
    let mut g = HirMut::new(&mut hir);
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wi = g.param("w", Shape::new(&[k, n], DType::I8));
    let si = g.param("scale", Shape::new(&[1, n], DType::F32));
    let zi = g.param("zp", Shape::new(&[1, n], DType::F32));
    let out = g.0.dequant_matmul(
        xi,
        wi,
        Some(si),
        Some(zi),
        QuantScheme::Int8Block {
            block_size: k as u32,
        },
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![out]);
    let mut p = HashMap::new();
    p.insert("scale".to_string(), scale.to_vec());
    p.insert("zp".to_string(), vec![0f32; n]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), dev).unwrap();
    let wbytes: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
    c.set_param_typed("w", &wbytes, DType::I8);
    move |x: &[f32]| c.run(&[("x", x)]).remove(0)
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
    let hidden = kc.text_config.hidden_size;
    let d = KdaDims {
        hidden,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 1,
    };
    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let w = ck
        .load_kda("language_model.model.layers.0", d)
        .map_err(|e| e.to_string())?;
    let (k, n) = (hidden, w.q_proj.len() / hidden);
    let m = 8usize;
    eprintln!("q_proj [{k},{n}] real layer-0, M={m}");

    let _ = m;
    let (wq, scale) = quantize_int8_col(&w.q_proj, k, n);
    let mut rng = Philox4x32::new(0x18);

    // parity once (M=8), then a timing sweep over M (M=1 = decode GEMV, M>1 = prefill).
    let w_f32_mb = (k * n * 4) as f64 / 1e6;
    let w_i8_mb = (k * n + n * 4) as f64 / 1e6;
    {
        let mut x = vec![0f32; 8 * k];
        rng.fill_normal(&mut x);
        let reff = f32_matmul(&x, 8, k, &w.q_proj, n);
        let fq = fake_quant_weight(&w.q_proj, k, n, WeightQuant::Int8Ch);
        let fakeq = f32_matmul(&x, 8, k, &fq, n);
        let real = int8_kernel(&x, 8, k, &wq, &scale, n);
        eprintln!(
            "\nparity (M=8): int8 vs f32 relL2 {:.3e}  |  int8 vs fake-int8 relL2 {:.3e} (0=same math)",
            rel(&reff, &real),
            rel(&fakeq, &real),
        );
    }
    eprintln!(
        "weight RAM: f32 {w_f32_mb:.1} MB → int8 {w_i8_mb:.1} MB ({:.2}× smaller)\n",
        w_f32_mb / w_i8_mb
    );

    for (label, dev) in [("CPU (AMX)", Device::Cpu), ("Metal (GPU)", Device::Metal)] {
        eprintln!("\ntiming on {label} — f32 matmul vs int8 DequantMatMul, median of 3:");
        eprintln!(
            "{:>4}  {:>12}  {:>12}  {:>8}  {:>10}",
            "M", "f32", "int8", "int8/f32", "int8 parity"
        );
        for &mm in &[1usize, 4, 16, 64] {
            let mut x = vec![0f32; mm * k];
            rng.fill_normal(&mut x);
            let mut rf = f32_runner(mm, k, &w.q_proj, n, dev);
            let mut ri = int8_runner(mm, k, &wq, &scale, n, dev);
            let yf = rf(&x);
            let yi = ri(&x); // warm + capture for parity
            let par = rel(&yf, &yi);
            let tf = med3(|| {
                rf(&x);
            });
            let ti = med3(|| {
                ri(&x);
            });
            eprintln!(
                "{mm:>4}  {:>10.2}ms  {:>10.2}ms  {:>7.2}×  {par:>10.2e}",
                tf * 1e3,
                ti * 1e3,
                ti / tf
            );
        }
    }
    Ok(())
}
