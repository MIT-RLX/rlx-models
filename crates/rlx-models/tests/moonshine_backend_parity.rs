// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CPU vs available backends on a Moonshine-shaped synthetic enc–dec graph.
//
//   cargo test -p rlx-models --test moonshine_backend_parity --features moonshine --release

#![allow(dead_code)]

mod compile_support;

use rlx_core::flow_util::compile_built;
use rlx_models::weight_map::WeightMap;
use rlx_moonshine::config::MoonshineConfig;
use rlx_moonshine::flow;
use rlx_moonshine::weight_source::CloningWeightSource;
use rlx_moonshine::weights::MoonshineWeightPrefix;
use rlx_runtime::Device;
use std::collections::HashMap;

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.001 + scale * ((i % 97) as f32) * 0.01)
        .collect()
}

fn synth_weights(cfg: &MoonshineConfig) -> (WeightMap, MoonshineWeightPrefix) {
    let pfx = MoonshineWeightPrefix {
        encoder: "model.encoder".into(),
        decoder: "model.decoder".into(),
        proj_out: None,
    };
    let d = cfg.hidden_size;
    let v = cfg.vocab_size;
    let e_ff = cfg.encoder_intermediate_size;
    let d_ff = cfg.decoder_intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u32;
    let mut z = |n: usize| {
        seed = seed.wrapping_add(1);
        ramp(n, 0.01 * (seed as f32))
    };
    let ones = |n: usize| vec![1.0f32; n];

    t.insert(pfx.enc_conv1_w(), (z(d * 127), vec![d, 1, 127]));
    t.insert(pfx.enc_conv2_w(), (z((2 * d) * d * 7), vec![2 * d, d, 7]));
    t.insert(pfx.enc_conv2_b(), (z(2 * d), vec![2 * d]));
    t.insert(pfx.enc_conv3_w(), (z(d * (2 * d) * 3), vec![d, 2 * d, 3]));
    t.insert(pfx.enc_conv3_b(), (z(d), vec![d]));
    t.insert(pfx.enc_groupnorm_w(), (ones(d), vec![d]));
    t.insert(pfx.enc_groupnorm_b(), (vec![0.0; d], vec![d]));
    t.insert(pfx.enc_ln_w(), (ones(d), vec![d]));

    for i in 0..cfg.encoder_num_hidden_layers {
        for name in ["q_proj", "k_proj", "v_proj"] {
            t.insert(
                pfx.enc_layer(i, &format!("self_attn.{name}.weight")),
                (z(d * d), vec![d, d]),
            );
        }
        t.insert(
            pfx.enc_layer(i, "self_attn.o_proj.weight"),
            (z(d * d), vec![d, d]),
        );
        t.insert(
            pfx.enc_layer(i, "input_layernorm.weight"),
            (ones(d), vec![d]),
        );
        t.insert(
            pfx.enc_layer(i, "post_attention_layernorm.weight"),
            (ones(d), vec![d]),
        );
        t.insert(
            pfx.enc_layer(i, "mlp.fc1.weight"),
            (z(e_ff * d), vec![e_ff, d]),
        );
        t.insert(pfx.enc_layer(i, "mlp.fc1.bias"), (z(e_ff), vec![e_ff]));
        t.insert(
            pfx.enc_layer(i, "mlp.fc2.weight"),
            (z(d * e_ff), vec![d, e_ff]),
        );
        t.insert(pfx.enc_layer(i, "mlp.fc2.bias"), (z(d), vec![d]));
    }

    t.insert(pfx.dec_embed_tokens(), (z(v * d), vec![v, d]));
    t.insert(pfx.dec_norm_w(), (ones(d), vec![d]));

    for i in 0..cfg.decoder_num_hidden_layers {
        for blk in ["self_attn", "encoder_attn"] {
            for name in ["q_proj", "k_proj", "v_proj"] {
                t.insert(
                    pfx.dec_layer(i, &format!("{blk}.{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
            }
            t.insert(
                pfx.dec_layer(i, &format!("{blk}.o_proj.weight")),
                (z(d * d), vec![d, d]),
            );
        }
        for n in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "final_layernorm.weight",
        ] {
            t.insert(pfx.dec_layer(i, n), (ones(d), vec![d]));
        }
        t.insert(
            pfx.dec_layer(i, "mlp.fc1.weight"),
            (z((2 * d_ff) * d), vec![2 * d_ff, d]),
        );
        t.insert(
            pfx.dec_layer(i, "mlp.fc1.bias"),
            (z(2 * d_ff), vec![2 * d_ff]),
        );
        t.insert(
            pfx.dec_layer(i, "mlp.fc2.weight"),
            (z(d * d_ff), vec![d, d_ff]),
        );
        t.insert(pfx.dec_layer(i, "mlp.fc2.bias"), (z(d), vec![d]));
    }

    (WeightMap::from_tensors(t), pfx)
}

fn run_decoder_hidden(device: Device) -> Vec<f32> {
    let cfg = MoonshineConfig::synth_tiny();
    let (weights, pfx) = synth_weights(&cfg);
    let audio_len = 1024;
    let enc_seq = MoonshineConfig::feat_extract_output_length(audio_len);

    let mut src = CloningWeightSource(&weights);
    let enc_built =
        flow::build_encoder_built(&cfg, &mut src, &pfx, 1, audio_len).expect("encoder build");
    let mut enc_g = compile_built(enc_built, device).expect("encoder compile");
    let pcm = vec![0.01f32; audio_len];
    let enc_out = enc_g.run(&[("pcm", pcm.as_slice())]);
    let enc_hidden = enc_out.into_iter().next().expect("enc out");

    let mut src = CloningWeightSource(&weights);
    let dec_built = flow::build_decoder_hidden_built(&cfg, &mut src, &pfx, 1, 4, enc_seq)
        .expect("decoder build");
    let mut dec_g = compile_built(dec_built, device).expect("decoder compile");
    let embeds = vec![0.01f32; 4 * cfg.hidden_size];
    let dec_out = dec_g.run(&[
        ("decoder_inputs_embeds", embeds.as_slice()),
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
        eprintln!("skip moonshine {name}: {device:?} not available");
        return;
    }
    let cpu = run_decoder_hidden(Device::Cpu);
    let other = run_decoder_hidden(device);
    let cpu_finite = cpu.iter().filter(|x| x.is_finite()).count();
    let other_finite = other.iter().filter(|x| x.is_finite()).count();
    let cpu_norm: f64 = cpu.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let other_norm: f64 = other
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "moonshine cpu vs {name}: len {}/{} finite {}/{} L2 {:.4}/{:.4}",
        cpu.len(),
        other.len(),
        cpu_finite,
        other_finite,
        cpu_norm,
        other_norm
    );
    assert_eq!(
        cpu.len(),
        other.len(),
        "moonshine cpu vs {name} length mismatch"
    );
    assert!(
        other_finite == other.len(),
        "moonshine {name} produced non-finite values"
    );
    let c = cosine(&cpu, &other);
    eprintln!("moonshine cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "moonshine cpu vs {name} cosine {c}");
}

#[test]
fn moonshine_decoder_hidden_cpu_smoke() {
    let out = run_decoder_hidden(Device::Cpu);
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(!out.is_empty());
    let norm: f64 = out.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    assert!(
        norm > 1e-6,
        "cpu decoder hidden collapsed to ~0 (norm={norm})"
    );
}

#[test]
fn moonshine_decoder_hidden_matches_metal() {
    #[cfg(feature = "metal")]
    assert_backend_matches_cpu("metal", Device::Metal);
    #[cfg(not(feature = "metal"))]
    eprintln!("skip moonshine metal: feature not enabled");
}

#[test]
fn moonshine_decoder_hidden_matches_mlx() {
    #[cfg(feature = "mlx")]
    assert_backend_matches_cpu("mlx", Device::Mlx);
    #[cfg(not(feature = "mlx"))]
    eprintln!("skip moonshine mlx: feature not enabled");
}

#[test]
fn moonshine_decoder_hidden_matches_wgpu() {
    #[cfg(feature = "gpu")]
    assert_backend_matches_cpu("wgpu", Device::Gpu);
    #[cfg(not(feature = "gpu"))]
    eprintln!("skip moonshine wgpu: feature not enabled");
}
