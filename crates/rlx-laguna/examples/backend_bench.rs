//! Laguna XS packed GGUF — RLX backend speed + numeric parity bench.
//!
//! Builds a real `Op::DequantMatMul` graph on `blk.0.attn_q.weight` from the
//! local XS Q4_K_M checkpoint and runs it on every available RLX device.
//! Reference is host [`rlx_cpu::gguf_matmul`] (same kernel as packed generate).
//!
//! ```bash
//! cargo run -p rlx-laguna --example backend_bench --release --features apple-silicon -- \
//!   --weights .cache/laguna-xs/Laguna-XS-2.1-Q4_K_M.gguf
//! ```

use anyhow::{Context, Result, bail};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill, packed_gguf_compile_guard,
    packed_gguf_execution_device,
};
use rlx_core::weight_loader::GgufLoader;
use rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch;
use rlx_ir::{DType, Graph, Op, Shape, quant::QuantScheme};
use rlx_runtime::{Device, Session, is_available};
use std::path::PathBuf;
use std::time::Instant;

const WARM: usize = 3;
const ITERS: usize = 10;
const SEQ: usize = 8;

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().max(1) as f32;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / n
}

struct Row {
    device: String,
    exec: String,
    ms: f64,
    max_abs: f32,
    mean_abs: f32,
    cosine: f32,
    note: String,
}

fn candidate_devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    for d in [
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Vulkan,
        Device::Cuda,
        Device::Ane,
    ] {
        if is_available(d) && !v.contains(&d) {
            v.push(d);
        }
    }
    v
}

fn run_device(
    device: Device,
    bytes: &[u8],
    scheme: QuantScheme,
    x: &[f32],
    hidden: usize,
    out_dim: usize,
    seq: usize,
    reference: &[f32],
) -> Result<Row> {
    let exec = packed_gguf_execution_device(device);
    let key = "blk.0.attn_q.weight";
    let mut g = Graph::new("laguna_q_spot");
    let x_id = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let w_id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
    let y_id = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_id, w_id],
        Shape::new(&[1, seq, out_dim], DType::F32),
    );
    g.set_outputs(vec![y_id]);

    let opts = compile_options_for_packed_gguf_prefill(exec);
    let mut compiled = packed_gguf_compile_guard(exec, || {
        Session::new(exec).compile_with(g.clone(), &opts)
    });
    compiled.set_param_typed(key, bytes, DType::U8);

    for _ in 0..WARM {
        let _ = compiled.run(&[("x", x)]);
    }
    let t0 = Instant::now();
    let mut last = Vec::new();
    for _ in 0..ITERS {
        last = compiled.run(&[("x", x)])[0].clone();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

    if last.len() != reference.len() {
        bail!(
            "{device:?}: output len {} != ref {}",
            last.len(),
            reference.len()
        );
    }
    let note = if exec != device {
        format!("exec redirected → {exec:?}")
    } else {
        String::new()
    };
    Ok(Row {
        device: format!("{device:?}"),
        exec: format!("{exec:?}"),
        ms,
        max_abs: max_abs(&last, reference),
        mean_abs: mean_abs(&last, reference),
        cosine: cosine(&last, reference),
        note,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights = flag_value(&args, "--weights").unwrap_or_else(|| {
        ".cache/laguna-xs/Laguna-XS-2.1-Q4_K_M.gguf".into()
    });
    let path = PathBuf::from(&weights);
    if !path.is_file() {
        bail!("weights not found: {}", path.display());
    }

    println!("Laguna packed DequantMatMul backend bench");
    println!("  weights: {}", path.display());
    println!("  spot: blk.0.attn_q.weight  seq={SEQ}  warm={WARM} iters={ITERS}");
    println!();

    let loader = GgufLoader::from_file(path.to_str().unwrap())
        .with_context(|| format!("open {}", path.display()))?;
    let t = loader
        .file()
        .get("blk.0.attn_q.weight")
        .ok_or_else(|| anyhow::anyhow!("missing blk.0.attn_q.weight"))?;
    let mut shape = t.shape.clone();
    shape.reverse(); // [out, in] after reverse of ggml [in, out]
    if shape.len() != 2 {
        bail!("attn_q shape {shape:?}");
    }
    let out_dim = shape[0];
    let hidden = shape[1];
    let scheme = rlx_core::ggml_type_to_quant_scheme(t.dtype)
        .ok_or_else(|| anyhow::anyhow!("attn_q dtype {:?} not packed", t.dtype))?;
    let bytes = loader
        .tensor_bytes_borrowed("blk.0.attn_q.weight")
        .ok_or_else(|| anyhow::anyhow!("bytes"))?
        .to_vec();
    println!(
        "  tensor: {:?} scheme={scheme:?} logical=[{out_dim},{hidden}] packed_bytes={}",
        t.dtype,
        bytes.len()
    );

    let x: Vec<f32> = (0..SEQ * hidden)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let mut reference = vec![0.0f32; SEQ * out_dim];
    gguf_matmul_bt_dispatch(&x, &bytes, &mut reference, SEQ, hidden, out_dim, scheme);

    // Host kernel timing (exact path used by packed generate).
    for _ in 0..WARM {
        let mut tmp = vec![0.0f32; SEQ * out_dim];
        gguf_matmul_bt_dispatch(&x, &bytes, &mut tmp, SEQ, hidden, out_dim, scheme);
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut tmp = vec![0.0f32; SEQ * out_dim];
        gguf_matmul_bt_dispatch(&x, &bytes, &mut tmp, SEQ, hidden, out_dim, scheme);
    }
    let host_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

    let mut rows = vec![Row {
        device: "HostKernel".into(),
        exec: "rlx_cpu::gguf_matmul".into(),
        ms: host_ms,
        max_abs: 0.0,
        mean_abs: 0.0,
        cosine: 1.0,
        note: "packed-generate matmul reference".into(),
    }];

    for d in candidate_devices() {
        match run_device(d, &bytes, scheme, &x, hidden, out_dim, SEQ, &reference) {
            Ok(r) => rows.push(r),
            Err(e) => rows.push(Row {
                device: format!("{d:?}"),
                exec: "-".into(),
                ms: f64::NAN,
                max_abs: f32::NAN,
                mean_abs: f32::NAN,
                cosine: f32::NAN,
                note: format!("ERROR: {e:#}"),
            }),
        }
    }

    // Rank by speed (finite ms) and by precision (highest cosine, then lowest max_abs).
    let mut by_speed: Vec<_> = rows
        .iter()
        .filter(|r| r.ms.is_finite())
        .collect();
    by_speed.sort_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap());
    let fastest = by_speed.first().map(|r| r.device.as_str()).unwrap_or("-");

    let mut by_prec: Vec<_> = rows
        .iter()
        .filter(|r| r.cosine.is_finite() && r.device != "HostKernel")
        .collect();
    by_prec.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap()
            .then(a.max_abs.partial_cmp(&b.max_abs).unwrap())
    });
    let most_precise = by_prec
        .first()
        .map(|r| r.device.as_str())
        .unwrap_or("HostKernel");

    println!();
    println!(
        "| {:<12} | {:<22} | {:>10} | {:>10} | {:>10} | {:>8} | notes |",
        "device", "exec", "ms/iter", "max_abs", "mean_abs", "cosine"
    );
    println!(
        "|--------------|------------------------|------------|------------|------------|----------|-------|"
    );
    for r in &rows {
        let ms = if r.ms.is_finite() {
            format!("{:.3}", r.ms)
        } else {
            "—".into()
        };
        let ma = if r.max_abs.is_finite() {
            format!("{:.3e}", r.max_abs)
        } else {
            "—".into()
        };
        let me = if r.mean_abs.is_finite() {
            format!("{:.3e}", r.mean_abs)
        } else {
            "—".into()
        };
        let cos = if r.cosine.is_finite() {
            format!("{:.6}", r.cosine)
        } else {
            "—".into()
        };
        println!(
            "| {:<12} | {:<22} | {:>10} | {:>10} | {:>10} | {:>8} | {} |",
            r.device, r.exec, ms, ma, me, cos, r.note
        );
    }
    println!();
    println!("Fastest:      **{fastest}**");
    println!("Most precise: **{most_precise}** (vs host `gguf_matmul` reference; HostKernel is exact by definition)");
    println!();
    println!(
        "Note: e2e generate uses packed KV-cached decode (`--device metal|mlx` optional); \
         this bench measures a single RLX `DequantMatMul` on a real Laguna Q4_K weight."
    );
    Ok(())
}
