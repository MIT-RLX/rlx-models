// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! Print Nanbeige looped-Transformer availability + synth prefill cosine vs CPU
//! on every RLX backend enabled at build time.
//!
//! ```sh
//! cargo run -p rlx-nanbeige --example backend_matrix --features all-backends --release
//! ```

use anyhow::Result;
use rlx_core::flow_bridge::compile_graph_with_profile;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_llama32::{
    Llama32Config, STANDARD_DEVICES, build_llama32_graph_sized_last_logits, validate_device,
};
use rlx_nanbeige::nanbeige42_3b_preset;
use rlx_runtime::Device;
use std::collections::HashMap;

fn tiny_looped() -> Llama32Config {
    let mut c = nanbeige42_3b_preset();
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 96;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = Some(8);
    c.num_loops = 2;
    c
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn synth(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q = cfg.q_proj_dim();
    let kv = cfg.kv_proj_dim();
    let ff = cfg.intermediate_size;
    let mut t = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.physical_layers() {
        let lp = format!("model.layers.{i}");
        t.insert(format!("{lp}.input_layernorm.weight"), (vec![1.0; h], vec![h]));
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q * h, 0.01), vec![q, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv * h, 0.01), vec![kv, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv * h, 0.01), vec![kv, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q, 0.01), vec![h, q]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(ff * h, 0.01), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(ff * h, 0.01), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * ff, 0.01), vec![h, ff]),
        );
    }
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
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

fn run(device: Device) -> Result<Vec<f32>> {
    let cfg = tiny_looped();
    validate_device(&cfg, device, false)?;
    let mut wm = synth(&cfg);
    let (graph, params) = build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, 4, false)?;
    let profile = CompileProfile::llama32_prefill();
    let mut compiled = compile_graph_with_profile(device, graph, &profile)?;
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let outs = compiled.run(&[
        ("input_ids", &[1.0f32, 2.0, 3.0, 4.0][..]),
        ("last_token_idx", &[3.0f32][..]),
    ]);
    Ok(outs[0].to_vec())
}

fn main() -> Result<()> {
    let cfg = tiny_looped();
    println!(
        "nanbeige backend matrix — physical={} loops={} kv_layers={} (synth)",
        cfg.physical_layers(),
        cfg.num_loops,
        cfg.kv_layers()
    );
    let cpu = run(Device::Cpu)?;
    println!(
        "{:<10} {:>6}  cosine_vs_cpu",
        "device", "avail"
    );
    println!(
        "{:<10} {:>6}  {:.8}  (reference)",
        "cpu",
        "yes",
        1.0
    );
    for &dev in STANDARD_DEVICES {
        if matches!(dev, Device::Cpu) {
            continue;
        }
        let avail = rlx_runtime::is_available(dev);
        if !avail {
            println!("{dev:<10?} {:>6}  (skip)", "no");
            continue;
        }
        match run(dev) {
            Ok(logits) => {
                let c = cosine(&cpu, &logits);
                let ok = if c > 0.99 { "ok" } else { "FAIL" };
                println!("{dev:<10?} {:>6}  {c:.8}  {ok}", "yes");
                if c <= 0.99 {
                    anyhow::bail!("{dev:?} cosine {c} <= 0.99");
                }
            }
            Err(e) => {
                println!("{dev:<10?} {:>6}  ERROR: {e:#}", "yes");
                return Err(e);
            }
        }
    }
    println!("all available backends matched CPU (cosine > 0.99)");
    Ok(())
}
