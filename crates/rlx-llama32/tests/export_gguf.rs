// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A Llama GGUF exported by this crate's exporter (GPT-J RoPE, the GGUF's
//! own `rope_freqs`) decodes what `Llama32Runner` does on that GGUF: a
//! two-layer model with random weights, written as a GGUF.

#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::{check_generate, write_checkpoint};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};
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
fn a_llama_gguf_exported_decodes_what_the_runner_does() {
    let (vocab, h, inter, layers, heads, kv, hd) = (96usize, 32, 64, 2, 4, 2, 8);
    let mut w = GgufWriter::new();
    w.set_arch("llama");
    w.set_meta("general.name", MetaValue::String("tiny-llama".into()));
    for (k, v) in [
        ("llama.vocab_size", vocab),
        ("llama.embedding_length", h),
        ("llama.feed_forward_length", inter),
        ("llama.block_count", layers),
        ("llama.attention.head_count", heads),
        ("llama.attention.head_count_kv", kv),
        ("llama.rope.dimension_count", hd),
        ("llama.context_length", 8192),
    ] {
        w.set_meta(k, MetaValue::U32(v as u32));
    }
    w.set_meta("llama.rope.freq_base", MetaValue::F32(500_000.0));
    w.set_meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    // Tensors by their GGUF names, shapes row-major (the file holds them
    // innermost first).
    let mut put = |name: String, shape: Vec<usize>, data: Vec<f32>| {
        let ne: Vec<usize> = shape.iter().rev().copied().collect();
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.add_tensor_bytes(name, ne, GgmlType::F32, bytes).unwrap();
    };
    let mut seed = 3u64;
    let mut rand = |n: usize| {
        seed += 7;
        fill(n, seed)
    };
    put("token_embd.weight".into(), vec![vocab, h], rand(vocab * h));
    for i in 0..layers {
        let p = format!("blk.{i}");
        for (t, shape) in [
            ("attn_norm", vec![h]),
            ("ffn_norm", vec![h]),
            ("attn_q", vec![heads * hd, h]),
            ("attn_k", vec![kv * hd, h]),
            ("attn_v", vec![kv * hd, h]),
            ("attn_output", vec![h, heads * hd]),
            ("ffn_gate", vec![inter, h]),
            ("ffn_up", vec![inter, h]),
            ("ffn_down", vec![h, inter]),
        ] {
            let n = shape.iter().product();
            put(format!("{p}.{t}.weight"), shape, rand(n));
        }
    }
    put("output_norm.weight".into(), vec![h], rand(h));
    put("output.weight".into(), vec![vocab, h], rand(vocab * h));
    // Llama 3's scaled RoPE, as its GGUFs carry it: a divisor per frequency.
    let factors: Vec<f32> = (0..hd / 2).map(|i| 1.0 + 3.0 * i as f32).collect();
    put("rope_freqs.weight".into(), vec![hd / 2], factors);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("llama");
    // A tokenizer.json beside the GGUF (the exporter takes it over the
    // GGUF's metadata).
    write_checkpoint(&dir, &json!({}), &HashMap::new(), vocab).unwrap();
    std::fs::remove_file(dir.join("config.json")).unwrap();
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    let gguf = dir.join("tiny.gguf");
    w.write_to_path(&gguf).unwrap();

    let out = tmp.path().join("llama.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 64],
        ..ExportOptions::default()
    };
    let report = rlx_llama32::export::export_rlxp(&gguf, &out, &opts).unwrap();
    eprintln!("{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = Llama32Runner::builder()
        .weights(&gguf)
        .device(Device::Cpu)
        .build()
        .unwrap();
    let long: Vec<u32> = (0..30).map(|i| 3 + (i * 11) % 90).collect();
    let prompts = [vec![5u32, 9, 13, 17], vec![40, 41], long];
    let c = check_generate(&mut runner, m.lm().unwrap(), &prompts, 24, &[2]).unwrap();
    eprintln!("{c}");
    assert!(c.passed(), "{c}");
    assert!(c.tokens > 30, "{c}");
}
