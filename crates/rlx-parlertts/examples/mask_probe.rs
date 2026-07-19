// RLX — versatile ML compiler + runtime. SPDX-License-Identifier: GPL-3.0
//! Per-node native-vs-ort compare of the T5 padding-mask chain
//! (Mul → Slice → Add_3) on `text_encoder_probe.onnx` (intermediates exposed as
//! outputs). The FIRST intermediate that diverges is the importer bug.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ort::value::Tensor;
use rlx_ir::DType;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{AotCache, CompileOptions, Device};
use tokenizers::Tokenizer;

const N: usize = 128;

fn i64_le(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn main() -> Result<()> {
    unsafe { std::env::set_var("RLX_DBG_BINF", "1") };
    let dir = std::env::var("RLX_PARLER_DIR").unwrap_or_else(|_| "weights/tts/parlertts".into());
    let dir = Path::new(&dir);
    let path = dir.join("onnx/text_encoder_probe.onnx");
    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut ids: Vec<i64> = tok
        .encode("The quick brown fox.", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .iter()
        .map(|&i| i as i64)
        .collect();
    ids.push(1);
    let real = ids.len();
    ids.resize(N, 0);
    let mask: Vec<i64> = (0..N).map(|i| if i < real { 1 } else { 0 }).collect();

    // --- native ---
    let named: HashMap<String, usize> = [("sequence_length", N), ("t", N), ("batch_size", 1)]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let opts = ImportOptions {
        sequence_length: N,
        named_lengths: named,
        strict: false,
        ..Default::default()
    };
    let (hir, mut params, _r, manifest) = build_hir_from_onnx_file(&path, opts)?;
    let out_names: Vec<String> = manifest.outputs.iter().map(|o| o.name.clone()).collect();
    let cache = AotCache::new(std::env::temp_dir().join("rlx_mask_probe"));
    // RLX_NO_PASSES: disable fusion/DCE/const-fold so exposing intermediates as
    // graph outputs does NOT perturb the compiled graph (non-confounded probe).
    let mut copts = CompileOptions::default();
    let key = if std::env::var_os("RLX_NO_PASSES").is_some() {
        copts.dce = false;
        copts.constant_folding = false;
        copts.fusion_opts.skip_fusion = true;
        "mask_probe_nopass"
    } else {
        "mask_probe"
    };
    let mut g = cache
        .compile_hir_cached(key, Device::Cpu, hir, &copts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for (n, d) in params.drain() {
        g.set_param(&n, &d);
    }
    g.finalize_params();
    let nout = g.run_typed(&[
        ("input_ids", &i64_le(&ids), DType::I64),
        ("attention_mask", &i64_le(&mask), DType::I64),
    ]);
    let as_dtype = |b: &[u8], d: DType| -> Vec<f32> {
        match d {
            DType::I64 => b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            DType::I32 => b
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            _ => as_f32(b),
        }
    };
    let nat: HashMap<String, Vec<f32>> = out_names
        .iter()
        .cloned()
        .zip(nout.iter().map(|(b, d)| as_dtype(b, *d)))
        .collect();

    // --- ort ---
    let mut sess = ort::session::Session::builder()?.commit_from_file(&path)?;
    let o = sess.run(ort::inputs![
        "input_ids" => Tensor::<i64>::from_array(([1usize, N], ids.clone()))?,
        "attention_mask" => Tensor::<i64>::from_array(([1usize, N], mask.clone()))?
    ])?;
    let mut ortv: HashMap<String, Vec<f32>> = HashMap::new();
    for name in &out_names {
        if let Ok((_, d)) = o[name.as_str()].try_extract_tensor::<f32>() {
            ortv.insert(name.clone(), d.to_vec());
        } else if let Ok((_, d)) = o[name.as_str()].try_extract_tensor::<i64>() {
            ortv.insert(name.clone(), d.iter().map(|&x| x as f32).collect());
        }
    }

    println!("native outputs: {:?}", out_names);
    for name in &out_names {
        if let (Some(a), Some(b)) = (nat.get(name), ortv.get(name)) {
            let short = name.rsplit('/').next().unwrap_or(name);
            let mx = a.iter().cloned().fold(f32::MIN, f32::max);
            let mn = a.iter().cloned().fold(f32::MAX, f32::min);
            let rowc = if a.len() == 128 * 1024 && b.len() == a.len() {
                let r = real * 1024;
                format!(
                    " | real-rows(0..{real})={:.6} pad-rows={:.6}",
                    cosine(&a[..r], &b[..r]),
                    cosine(&a[r..], &b[r..])
                )
            } else {
                String::new()
            };
            println!(
                "  {short:22} native[{}] vs ort[{}]  cosine={:.6}{rowc}  native range[{mn:.2e},{mx:.2e}]",
                a.len(),
                b.len(),
                cosine(a, b)
            );
            // Per-128-key-row divergence scan (softmax [.,.,q,128]).
            if std::env::var_os("RLX_SCAN").is_some() && a.len() == b.len() && a.len() % 128 == 0 {
                let nrows = a.len() / 128;
                let mut worst = (1.0f64, 0usize);
                let mut nbad = 0;
                for r in 0..nrows {
                    let (aa, bb) = (&a[r * 128..r * 128 + 128], &b[r * 128..r * 128 + 128]);
                    let c = cosine(aa, bb);
                    if c < 0.999 {
                        nbad += 1;
                    }
                    if c < worst.0 {
                        worst = (c, r);
                    }
                }
                let r = std::env::var("RLX_SCAN_ROW")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(worst.1);
                println!(
                    "      SCAN {nrows} rows, {nbad} with cos<0.999; worst row {} cos={:.5} (head {}, query {})",
                    worst.1,
                    worst.0,
                    worst.1 / 128,
                    worst.1 % 128
                );
                let (aa, bb) = (&a[r * 128..r * 128 + 8], &b[r * 128..r * 128 + 8]);
                println!(
                    "        nat[{r}]: {:?}",
                    aa.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>()
                );
                println!(
                    "        ort[{r}]: {:?}",
                    bb.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>()
                );
            }
            if std::env::var_os("RLX_ROW").is_some() && a.len() == b.len() {
                // row 0 (head0, query0): first 6 + last 6 keys, native vs ort
                let k = 128.min(a.len());
                let show = |v: &[f32]| {
                    format!(
                        "{:?} … {:?}",
                        &v[..6.min(k)]
                            .iter()
                            .map(|x| format!("{x:.3e}"))
                            .collect::<Vec<_>>(),
                        &v[k.saturating_sub(6)..k]
                            .iter()
                            .map(|x| format!("{x:.3e}"))
                            .collect::<Vec<_>>()
                    )
                };
                println!("      nat: {}", show(a));
                println!("      ort: {}", show(b));
            }
        } else {
            println!(
                "  {name}: MISSING (native={} ort={})",
                nat.contains_key(name),
                ortv.contains_key(name)
            );
        }
    }
    Ok(())
}
