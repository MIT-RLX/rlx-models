// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Numeric parity gate for the **hand-authored native T3 LM** (rlx-llama32 graph,
//! NOT ONNX-imported). Builds the T3 Llama prefill in `inputs_embeds` mode from
//! the extracted `t3_lm.safetensors`, runs it on the deterministic
//! `parity_inputs_embeds.bin`, and compares the logits to the numpy reference
//! (`native_t3_reference.py`). Both sides use the SAME extracted weights, so this
//! validates the rlx graph math (rope / RMSNorm / MHA / SwiGLU / head) end to end
//! on CPU with no onnxruntime.
//!
//! ```text
//! python3 crates/rlx-chatterbox/scripts/native_t3_reference.py
//! cargo run -p rlx-chatterbox --example native_t3_parity
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::WeightMap;
use rlx_ir::DType;
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_runtime::{AotCache, CompileOptions, Device};

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let nat: PathBuf =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/chatterbox/native");
    let dev = std::env::var("RLX_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let device = rlx_runtime::parse_device(&dev).unwrap_or(Device::Cpu);
    println!("[parity] device = {device:?}");
    let t = 8usize; // matches native_t3_reference.py
    let vocab = 8194usize;

    let cfg = Llama32Config::from_file(&nat.join("t3_config.json"))?;
    println!(
        "[cfg] layers={} hidden={} heads={} head_dim={} vocab={} rope_theta={} scaling={:?}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.head_dim(),
        cfg.vocab_size,
        cfg.rope_theta,
        cfg.rope_scaling.is_some(),
    );

    // Hand-authored native T3 prefill — inputs_embeds entry, full-sequence logits.
    let mut wm = WeightMap::from_file(nat.join("t3_lm.safetensors").to_str().unwrap())?;
    let built = Llama32Flow::new(&cfg)
        .prefill()
        .batch(1)
        .seq(t)
        .inputs_embeds()
        .lm_head()
        .build(&mut wm)?;
    let (hir, mut params) = built.into_parts()?;

    let cache = AotCache::new(std::env::temp_dir().join(format!("rlx_t3_parity_{device:?}")));
    let mut g = cache
        .compile_hir_cached("t3_prefill_s8", device, hir, &CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("compile: {e}"))?;
    for (name, data) in params.drain() {
        g.set_param(&name, &data);
    }
    g.finalize_params();

    let emb = read_f32(&nat.join("parity_inputs_embeds.bin"))?;
    let emb_bytes: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
    let out = g.run_typed(&[("inputs_embeds", &emb_bytes, DType::F32)]);
    let got = as_f32(&out[0].0);
    anyhow::ensure!(
        got.len() == t * vocab,
        "native logits len {} != {}*{}",
        got.len(),
        t,
        vocab
    );

    let refl = read_f32(&nat.join("parity_ref_logits.bin"))?;
    anyhow::ensure!(refl.len() == t * vocab, "ref logits len {}", refl.len());

    // Cosine + max-abs over all positions; argmax agreement per row.
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(&refl) {
        dot += a as f64 * b as f64;
        na += (a as f64).powi(2);
        nb += (b as f64).powi(2);
        maxabs = maxabs.max((a - b).abs() as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());

    let mut agree = 0;
    for row in 0..t {
        let am = |v: &[f32]| {
            v[row * vocab..(row + 1) * vocab]
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0
        };
        if am(&got) == am(&refl) {
            agree += 1;
        }
    }

    println!("[parity] cosine={cos:.8}  max_abs={maxabs:.4}  argmax_agree={agree}/{t}");
    let last_native = got[(t - 1) * vocab..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("[parity] native argmax(last)={last_native}");

    if cos > 0.9999 && agree == t {
        println!("[parity] ✅ native T3 LM matches numpy reference");
        Ok(())
    } else {
        anyhow::bail!("parity FAILED (cos {cos:.6}, agree {agree}/{t})")
    }
}
