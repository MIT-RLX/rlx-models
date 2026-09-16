// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CPU vs available backends on a NLLB/M2M100-shaped synthetic enc–dec graph.
//
//   cargo test -p rlx-models --test nllb_backend_parity --features nllb --release

#![allow(dead_code)]

mod compile_support;

use rlx_core::flow_util::compile_built;
use rlx_models::weight_map::WeightMap;
use rlx_nllb::config::NllbConfig;
use rlx_nllb::flow;
use rlx_nllb::weight_source::CloningWeightSource;
use rlx_nllb::weights::lang as lk;
use rlx_runtime::Device;
use std::collections::HashMap;

fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 0.02 - 0.01
        })
        .collect()
}

fn put(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, shape: &[usize], seed: u32) {
    let n: usize = shape.iter().product();
    map.insert(key.to_string(), (rand_vec(n, seed), shape.to_vec()));
}

fn synth_weights(cfg: &NllbConfig) -> WeightMap {
    let mut t = HashMap::new();
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let ffn = cfg.encoder_ffn_dim;
    let pos = cfg.max_position_embeddings + NllbConfig::POS_OFFSET;
    put(&mut t, lk::SHARED, &[v, d], 1);
    put(&mut t, &lk::enc_embed_positions(), &[pos, d], 2);
    put(&mut t, &lk::dec_embed_positions(), &[pos, d], 3);
    put(&mut t, &lk::enc_layernorm_embedding_w(), &[d], 4);
    put(&mut t, &lk::enc_layernorm_embedding_b(), &[d], 5);
    put(&mut t, &lk::dec_layernorm_embedding_w(), &[d], 6);
    put(&mut t, &lk::dec_layernorm_embedding_b(), &[d], 7);
    put(&mut t, &lk::enc_final_layer_norm_w(), &[d], 8);
    put(&mut t, &lk::enc_final_layer_norm_b(), &[d], 9);
    put(&mut t, &lk::dec_final_layer_norm_w(), &[d], 10);
    put(&mut t, &lk::dec_final_layer_norm_b(), &[d], 11);

    let mut seed = 100u32;
    for layer in 0..cfg.encoder_layers {
        let p = |s: &str| lk::enc_layer(layer, s);
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ] {
            put(&mut t, &p(&format!("{name}.weight")), &[d, d], seed);
            seed += 1;
            put(&mut t, &p(&format!("{name}.bias")), &[d], seed);
            seed += 1;
        }
        put(&mut t, &p("self_attn_layer_norm.weight"), &[d], seed);
        seed += 1;
        put(&mut t, &p("self_attn_layer_norm.bias"), &[d], seed);
        seed += 1;
        put(&mut t, &p("fc1.weight"), &[ffn, d], seed);
        seed += 1;
        put(&mut t, &p("fc1.bias"), &[ffn], seed);
        seed += 1;
        put(&mut t, &p("fc2.weight"), &[d, ffn], seed);
        seed += 1;
        put(&mut t, &p("fc2.bias"), &[d], seed);
        seed += 1;
        put(&mut t, &p("final_layer_norm.weight"), &[d], seed);
        seed += 1;
        put(&mut t, &p("final_layer_norm.bias"), &[d], seed);
        seed += 1;
    }
    for layer in 0..cfg.decoder_layers {
        let p = |s: &str| lk::dec_layer(layer, s);
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "encoder_attn.q_proj",
            "encoder_attn.k_proj",
            "encoder_attn.v_proj",
            "encoder_attn.out_proj",
        ] {
            put(&mut t, &p(&format!("{name}.weight")), &[d, d], seed);
            seed += 1;
            put(&mut t, &p(&format!("{name}.bias")), &[d], seed);
            seed += 1;
        }
        put(&mut t, &p("self_attn_layer_norm.weight"), &[d], seed);
        seed += 1;
        put(&mut t, &p("self_attn_layer_norm.bias"), &[d], seed);
        seed += 1;
        put(&mut t, &p("encoder_attn_layer_norm.weight"), &[d], seed);
        seed += 1;
        put(&mut t, &p("encoder_attn_layer_norm.bias"), &[d], seed);
        seed += 1;
        put(&mut t, &p("fc1.weight"), &[ffn, d], seed);
        seed += 1;
        put(&mut t, &p("fc1.bias"), &[ffn], seed);
        seed += 1;
        put(&mut t, &p("fc2.weight"), &[d, ffn], seed);
        seed += 1;
        put(&mut t, &p("fc2.bias"), &[d], seed);
        seed += 1;
        put(&mut t, &p("final_layer_norm.weight"), &[d], seed);
        seed += 1;
        put(&mut t, &p("final_layer_norm.bias"), &[d], seed);
        seed += 1;
    }
    WeightMap::from_tensors(t)
}

fn run_decoder_hidden(device: Device) -> Vec<f32> {
    let cfg = NllbConfig::tiny();
    let weights = synth_weights(&cfg);
    let enc_seq = 4usize;
    let dec_seq = 3usize;

    let mut src = CloningWeightSource(&weights);
    let enc_built = flow::build_encoder_built(&cfg, &mut src, 1, enc_seq).expect("encoder build");
    let mut enc_g = compile_built(enc_built, device).expect("encoder compile");
    let embeds = vec![0.01f32; enc_seq * cfg.d_model];
    let enc_out = enc_g.run(&[("inputs_embeds", embeds.as_slice())]);
    let enc_hidden = enc_out.into_iter().next().expect("enc out");

    let mut src = CloningWeightSource(&weights);
    let dec_built = flow::build_decoder_hidden_built(&cfg, &mut src, 1, dec_seq, enc_seq)
        .expect("decoder build");
    let mut dec_g = compile_built(dec_built, device).expect("decoder compile");
    let dec_embeds = vec![0.02f32; dec_seq * cfg.d_model];
    let dec_out = dec_g.run(&[
        ("decoder_inputs_embeds", dec_embeds.as_slice()),
        ("encoder_hidden", enc_hidden.as_slice()),
    ]);
    dec_out.into_iter().next().expect("dec out")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn assert_backend_matches_cpu(name: &str, device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip nllb {name}: {device:?} not available");
        return;
    }
    let cpu = run_decoder_hidden(Device::Cpu);
    let other = run_decoder_hidden(device);
    let c = cosine(&cpu, &other);
    eprintln!("nllb cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "nllb cpu vs {name} cosine {c}");
}

#[test]
fn nllb_decoder_hidden_cpu_smoke() {
    let out = run_decoder_hidden(Device::Cpu);
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(!out.is_empty());
}

#[test]
fn nllb_decoder_hidden_matches_metal() {
    #[cfg(feature = "metal")]
    assert_backend_matches_cpu("metal", Device::Metal);
    #[cfg(not(feature = "metal"))]
    eprintln!("skip nllb metal: feature not enabled");
}

#[test]
fn nllb_decoder_hidden_matches_mlx() {
    #[cfg(feature = "mlx")]
    assert_backend_matches_cpu("mlx", Device::Mlx);
    #[cfg(not(feature = "mlx"))]
    eprintln!("skip nllb mlx: feature not enabled");
}

#[test]
fn nllb_decoder_hidden_matches_wgpu() {
    #[cfg(feature = "gpu")]
    assert_backend_matches_cpu("wgpu", Device::Gpu);
    #[cfg(not(feature = "gpu"))]
    eprintln!("skip nllb wgpu: feature not enabled");
}
