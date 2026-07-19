// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Self-consistency gate for the native T3 **KV-cache decode**: prove that
//! `prefill(T) + export_kv → decode(1 token, past=that KV)` yields the SAME
//! last-position logits as `prefill(T+1)`. Validates the KV handoff
//! (prefill-present [K_rope,V] per layer → decode past_k_i/past_v_i, dynamic
//! past length) before wiring it into the AR loop. All native rlx-llama32, CPU,
//! no onnxruntime.
//!
//! ```text
//! cargo run -p rlx-chatterbox --example native_t3_kv_parity --features native
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::WeightMap;
use rlx_ir::DType;
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};

const H: usize = 1024;
const VOCAB: usize = 8194;
const NL: usize = 30;

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_wm(nat: &Path) -> Result<WeightMap> {
    WeightMap::from_file(nat.join("t3_lm.safetensors").to_str().unwrap())
        .context("load t3_lm.safetensors")
}

fn compile(
    cache: &AotCache,
    key: &str,
    device: Device,
    hir: rlx_ir::hir::HirModule,
    mut params: std::collections::HashMap<String, Vec<f32>>,
) -> Result<CompiledGraph> {
    let mut g = cache
        .compile_hir_cached(key, device, hir, &CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("compile {key}: {e}"))?;
    for (name, data) in params.drain() {
        g.set_param(&name, &data);
    }
    g.finalize_params();
    Ok(g)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn main() -> Result<()> {
    let nat: PathBuf =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/chatterbox/native");
    let cfg = Llama32Config::from_file(&nat.join("t3_config.json"))?;
    let device =
        rlx_runtime::parse_device(&std::env::var("RLX_DEVICE").unwrap_or_else(|_| "cpu".into()))
            .unwrap_or(Device::Cpu);
    println!("[kv] device = {device:?}");
    let cache = AotCache::new(std::env::temp_dir().join(format!("rlx_t3_kv_parity_{device:?}")));

    let t = 6usize; // prompt length; then one decode step to position T
    // deterministic embeds [T+1, H]
    let mk = |n: usize| -> Vec<f32> {
        (0..n * H)
            .map(|i| (((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5) * 0.2)
            .collect()
    };
    let emb_full = mk(t + 1); // rows 0..T = prompt, row T = the "new" token
    let emb_prompt = emb_full[..t * H].to_vec();
    let emb_new = emb_full[t * H..(t + 1) * H].to_vec();

    // --- 1) prefill(T) with export_kv ---
    let mut wm = load_wm(&nat)?;
    let prefill = Llama32Flow::new(&cfg)
        .prefill()
        .batch(1)
        .seq(t)
        .inputs_embeds()
        .lm_head()
        .export_kv()
        .build(&mut wm)?;
    let (hir, params) = prefill.into_parts()?;
    let mut gp = compile(&cache, &format!("kv_prefill_s{t}"), device, hir, params)?;
    let outs = gp.run_typed(&[("inputs_embeds", &f32_le(&emb_prompt), DType::F32)]);
    // outs[0] = logits [1,T,VOCAB]; outs[1..] = [L0_K, L0_V, L1_K, L1_V, ...] each [1,T,H]
    anyhow::ensure!(
        outs.len() == 1 + 2 * NL,
        "prefill outputs {} != {}",
        outs.len(),
        1 + 2 * NL
    );
    let past_k: Vec<Vec<u8>> = (0..NL).map(|i| outs[1 + 2 * i].0.clone()).collect();
    let past_v: Vec<Vec<u8>> = (0..NL).map(|i| outs[2 + 2 * i].0.clone()).collect();
    println!(
        "[kv] prefill T={t}: {} outputs, KV row = [1,{t},{H}] ({} f32)",
        outs.len(),
        as_f32(&past_k[0]).len()
    );

    // --- 2) BUCKETED masked decode: ONE static graph at past=`upper` serves any
    //        real past length via a keep-mask. This is the shape the AR loop uses
    //        (one compile + one 2GB set_param, reused every step). past_k is the
    //        real KV padded to `upper` rows; mask keeps [0,past_seq)+new@upper. ---
    let upper = 12usize; // bucket > t to exercise padding
    let mut wm2 = load_wm(&nat)?;
    let decode = Llama32Flow::new(&cfg)
        .decode()
        .batch(1)
        .past(upper)
        .custom_mask()
        .inputs_embeds()
        .export_kv()
        .lm_head()
        .build(&mut wm2)?;
    let (hir, params) = decode.into_parts()?;
    let mut gd = compile(
        &cache,
        &format!("kv_decode_bucket{upper}"),
        device,
        hir,
        params,
    )?;
    // pad real KV [1,t,H] → [1,upper,H]
    let pad = |real: &[u8]| -> Vec<u8> {
        let mut v = as_f32(real);
        v.resize(upper * H, 0.0);
        f32_le(&v)
    };
    let mask = rlx_runtime::attn_mask::bucket_decode_mask(t, upper); // [upper+1]
    let mut din: Vec<(String, Vec<u8>, DType)> = Vec::new();
    din.push(("inputs_embeds".into(), f32_le(&emb_new), DType::F32));
    din.push(("mask".into(), f32_le(&mask), DType::F32));
    din.push(("position".into(), f32_le(&[t as f32]), DType::F32));
    for i in 0..NL {
        din.push((format!("past_k_{i}"), pad(&past_k[i]), DType::F32));
        din.push((format!("past_v_{i}"), pad(&past_v[i]), DType::F32));
    }
    let dref: Vec<(&str, &[u8], DType)> = din
        .iter()
        .map(|(n, b, d)| (n.as_str(), b.as_slice(), *d))
        .collect();
    let douts = gd.run_typed(&dref);
    let dlogits = as_f32(&douts[0].0); // [1,1,VOCAB]
    anyhow::ensure!(
        dlogits.len() == VOCAB,
        "decode logits {} != {VOCAB}",
        dlogits.len()
    );
    // sanity: the new token's KV comes back at output row `upper`
    let new_k0 = as_f32(&douts[1].0);
    anyhow::ensure!(
        new_k0.len() == (upper + 1) * H,
        "decode KV len {} != {}",
        new_k0.len(),
        (upper + 1) * H
    );

    // --- 3) reference: prefill(T+1), read row T ---
    let mut wm3 = load_wm(&nat)?;
    let prefill2 = Llama32Flow::new(&cfg)
        .prefill()
        .batch(1)
        .seq(t + 1)
        .inputs_embeds()
        .lm_head()
        .build(&mut wm3)?;
    let (hir, params) = prefill2.into_parts()?;
    let mut gp2 = compile(
        &cache,
        &format!("kv_prefill_s{}", t + 1),
        device,
        hir,
        params,
    )?;
    let o2 = gp2.run_typed(&[("inputs_embeds", &f32_le(&emb_full), DType::F32)]);
    let full = as_f32(&o2[0].0); // [1,T+1,VOCAB]
    let ref_row = &full[t * VOCAB..(t + 1) * VOCAB];

    let cos = cosine(&dlogits, ref_row);
    let maxabs = dlogits
        .iter()
        .zip(ref_row)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "[kv] decode-vs-prefill(T+1)[row T]: cosine={cos:.8} max_abs={maxabs:.5} argmax {} vs {}",
        argmax(&dlogits),
        argmax(ref_row)
    );
    if cos > 0.9999 && argmax(&dlogits) == argmax(ref_row) {
        println!("[kv] ✅ KV-cache decode matches full prefill — handoff correct");
        Ok(())
    } else {
        anyhow::bail!("KV parity FAILED (cos {cos:.6})")
    }
}
