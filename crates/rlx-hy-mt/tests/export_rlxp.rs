// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An HY-MT (Hunyuan dense) checkpoint, its Q/K norms under Hunyuan's names
//! (`query_layernorm` / `key_layernorm`), exports as a package that decodes
//! what the runner does on it: two layers, random weights.

#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::{check_generate, write_checkpoint};
use rlx_qwen3::{Qwen3ConfigSource, Qwen3Runner};
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
fn an_hy_mt_checkpoint_exports_what_the_runner_decodes() {
    let (vocab, h, inter, layers, heads, kv, hd) = (96, 32, 64, 2, 4, 2, 8);
    let config = json!({
        "model_type": "hunyuan_v1_dense", "architectures": ["HunYuanDenseV1ForCausalLM"],
        "vocab_size": vocab, "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "attention_head_dim": hd,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "use_qk_norm": true, "tie_word_embeddings": true,
        "bos_token_id": 1, "eos_token_id": 2,
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
        put(format!("{p}.self_attn.query_layernorm.weight"), vec![hd]);
        put(format!("{p}.self_attn.key_layernorm.weight"), vec![hd]);
        put(format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        put(format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        put(format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }
    put("model.norm.weight".into(), vec![h]);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("hy-mt");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    let out = tmp.path().join("hy-mt.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 64],
        ..ExportOptions::default()
    };
    let report = rlx_hy_mt::export::export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    // The runner HyMtRunner builds, past its 1.8B / 7B size check.
    let cfg = rlx_hy_mt::config::qwen3_config_from_hf(&dir).unwrap();
    let mut runner = Qwen3Runner::builder()
        .weights(dir.join("model.safetensors"))
        .config(Qwen3ConfigSource::Explicit(cfg))
        .device(Device::Cpu)
        .build()
        .unwrap();
    let long: Vec<u32> = (0..30).map(|i| 3 + (i * 11) % 90).collect();
    let prompts = [vec![5u32, 9, 13, 17], vec![40, 41], long];
    let c = check_generate(&mut runner, m.lm().unwrap(), &prompts, 16, &[2]).unwrap();
    eprintln!("{c}");
    assert!(c.passed(), "{c}");
}
