// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A Llama checkpoint exported by the config-driven exporter
//! (`rlx_core::export::standard`, no family code) decodes what
//! `Llama32Runner` does: a two-layer model with random weights and Llama 3's
//! scaled RoPE, written as a checkpoint.

use std::collections::HashMap;

use rlx_core::export::check::{check_generate, write_checkpoint};
use rlx_core::export::{Checkpoint, ExportOptions, detect};
use rlx_llama32::Llama32Runner;
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
fn a_llama_exported_by_its_config_alone_decodes_what_the_runner_does() {
    let (vocab, h, inter, layers, heads, kv, hd) = (96, 32, 64, 2, 4, 2, 8);
    let config = json!({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "vocab_size": vocab, "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": hd, "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "hidden_act": "silu",
        "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2,
        "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 16},
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
    let dir = tmp.path().join("llama");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    // No family exporter is registered here: the checkpoint goes to the
    // config-driven one.
    let exporter = detect(&Checkpoint::open(&dir).unwrap()).unwrap();
    assert_eq!(exporter.name, "standard");
    let out = tmp.path().join("llama.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 64],
        ..ExportOptions::default()
    };
    let report = (exporter.export)(&dir, &out, &opts).unwrap();
    assert_eq!(report.referenced, t.len(), "{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = Llama32Runner::builder()
        .weights(dir.join("model.safetensors"))
        .device(Device::Cpu)
        .build()
        .unwrap();
    // Positions past 16 are where the scaled frequencies differ.
    let long: Vec<u32> = (0..30).map(|i| 3 + (i * 11) % 90).collect();
    let prompts = [vec![5u32, 9, 13, 17], vec![40, 41], long];
    let c = check_generate(&mut runner, m.lm().unwrap(), &prompts, 24, &[2]).unwrap();
    eprintln!("{c}");
    assert!(c.passed(), "{c}");
    assert!(c.tokens > 30, "{c}");
}
