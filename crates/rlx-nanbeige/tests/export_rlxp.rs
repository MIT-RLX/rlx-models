// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A Nanbeige checkpoint (its layers run twice over shared weights) exports
//! as a package that decodes what `Llama32Runner` does on it: two physical
//! layers, `num_loops = 2`, random weights.

#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::check::{check_generate, write_checkpoint};
use rlx_core::export::{Checkpoint, ExportOptions};
use rlx_nanbeige::Llama32Runner;
use rlx_runtime::Device;
use serde_json::json;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

#[test]
fn a_looped_nanbeige_exports_what_the_runner_decodes() {
    let (vocab, h, inter, layers, heads, kv, hd) = (96, 32, 64, 2, 4, 2, 8);
    let config = json!({
        "model_type": "nanbeige", "architectures": ["NanbeigeForCausalLM"],
        "vocab_size": vocab, "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": hd, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5, "rope_theta": 70000.0, "hidden_act": "silu",
        "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2,
        "num_loops": 2,
    });
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 3u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 7;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put("model.embed_tokens.weight".into(), vec![vocab, h]);
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        put(format!("{p}.input_layernorm.weight"), vec![h]);
        put(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        put(format!("{p}.self_attn.q_proj.weight"), vec![heads * hd, h]);
        put(format!("{p}.self_attn.k_proj.weight"), vec![kv * hd, h]);
        put(format!("{p}.self_attn.v_proj.weight"), vec![kv * hd, h]);
        put(format!("{p}.self_attn.o_proj.weight"), vec![h, heads * hd]);
        put(format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        put(format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        put(format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }
    put("model.norm.weight".into(), vec![h]);
    put("lm_head.weight".into(), vec![vocab, h]);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("nanbeige");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    // The generic exporter refuses it: it would run the layers once.
    assert!(!rlx_core::export::standard::supports(
        &Checkpoint::open(&dir).unwrap()
    ));
    let out = tmp.path().join("nanbeige.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 64],
        ..ExportOptions::default()
    };
    let report = rlx_nanbeige::export::export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = Llama32Runner::builder()
        .weights(dir.join("model.safetensors"))
        .device(Device::Cpu)
        .build()
        .unwrap();
    let long: Vec<u32> = (0..30).map(|i| 3 + (i * 11) % 90).collect();
    let prompts = [vec![5u32, 9, 13, 17], vec![40, 41], long];
    let c = check_generate(&mut runner, m.lm().unwrap(), &prompts, 16, &[2]).unwrap();
    eprintln!("{c}");
    assert!(c.passed(), "{c}");
}
