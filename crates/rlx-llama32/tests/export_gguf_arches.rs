// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Every Llama-shaped GGUF architecture `Llama32Runner` builds besides
//! Llama's own (OLMo-2, Nemotron, GLM-4, ChatGLM, Granite, Exaone, Phi-3,
//! Mistral 3, Muse Glimmer) exports as a package that decodes what the runner does on
//! that GGUF: four layers, random weights, written as a GGUF with its
//! matrices in Q8_0 (the runner's per-arch blocks are its packed path).

#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::{check_generate, write_checkpoint};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};
use rlx_llama32::Llama32Runner;
use rlx_runtime::Device;
use serde_json::json;

const VOCAB: usize = 96;
const H: usize = 32;
const INTER: usize = 64;
const LAYERS: usize = 4;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 8;

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

/// One architecture: its extra metadata and the tensors of a layer.
struct Arch {
    name: &'static str,
    meta: Vec<(&'static str, MetaValue)>,
    layer: Vec<(&'static str, Vec<usize>)>,
    /// `output_norm.bias` and the like.
    top: Vec<(&'static str, Vec<usize>)>,
    rope_dims: usize,
}

fn attn() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("attn_q", vec![HEADS * HD, H]),
        ("attn_k", vec![KV * HD, H]),
        ("attn_v", vec![KV * HD, H]),
        ("attn_output", vec![H, HEADS * HD]),
    ]
}

fn swiglu() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("ffn_gate", vec![INTER, H]),
        ("ffn_up", vec![INTER, H]),
        ("ffn_down", vec![H, INTER]),
    ]
}

fn arches() -> Vec<Arch> {
    let pre = vec![("attn_norm", vec![H]), ("ffn_norm", vec![H])];
    let cat = |parts: &[Vec<(&'static str, Vec<usize>)>]| parts.concat();
    vec![
        Arch {
            name: "granite",
            meta: vec![
                ("granite.embedding_scale", MetaValue::F32(12.0)),
                ("granite.residual_scale", MetaValue::F32(0.22)),
                ("granite.attention.scale", MetaValue::F32(0.125)),
                ("granite.logit_scale", MetaValue::F32(8.0)),
            ],
            layer: cat(&[pre.clone(), attn(), swiglu()]),
            top: vec![],
            rope_dims: HD,
        },
        Arch {
            name: "exaone",
            meta: vec![],
            layer: cat(&[pre.clone(), attn(), swiglu()]),
            top: vec![],
            rope_dims: HD,
        },
        Arch {
            name: "olmo2",
            meta: vec![],
            layer: cat(&[
                attn(),
                swiglu(),
                vec![
                    ("attn_q_norm", vec![HEADS * HD]),
                    ("attn_k_norm", vec![KV * HD]),
                    ("post_attention_norm", vec![H]),
                    ("post_ffw_norm", vec![H]),
                ],
            ]),
            top: vec![],
            rope_dims: HD,
        },
        Arch {
            name: "nemotron",
            meta: vec![],
            layer: cat(&[
                pre.clone(),
                vec![("attn_norm.bias", vec![H]), ("ffn_norm.bias", vec![H])],
                attn(),
                vec![("ffn_up", vec![INTER, H]), ("ffn_down", vec![H, INTER])],
            ]),
            top: vec![("output_norm.bias", vec![H])],
            rope_dims: HD / 2,
        },
        Arch {
            name: "mistral3",
            meta: vec![],
            layer: cat(&[pre.clone(), attn(), swiglu()]),
            top: vec![],
            rope_dims: HD,
        },
        Arch {
            name: "glm4",
            meta: vec![],
            layer: cat(&[
                pre.clone(),
                attn(),
                vec![
                    ("ffn_up", vec![2 * INTER, H]),
                    ("ffn_down", vec![H, INTER]),
                    ("post_attention_norm", vec![H]),
                    ("post_ffw_norm", vec![H]),
                ],
            ]),
            top: vec![],
            rope_dims: HD / 2,
        },
        Arch {
            name: "chatglm",
            meta: vec![],
            layer: cat(&[
                pre.clone(),
                attn(),
                vec![("ffn_up", vec![2 * INTER, H]), ("ffn_down", vec![H, INTER])],
            ]),
            top: vec![],
            rope_dims: HD / 2,
        },
        Arch {
            name: "phi3",
            meta: vec![],
            layer: cat(&[
                pre.clone(),
                vec![
                    ("attn_qkv", vec![(HEADS + 2 * KV) * HD, H]),
                    ("attn_output", vec![H, HEADS * HD]),
                    ("ffn_up", vec![2 * INTER, H]),
                    ("ffn_down", vec![H, INTER]),
                ],
            ]),
            top: vec![],
            rope_dims: HD / 2,
        },
        Arch {
            name: "muse-glimmer",
            meta: vec![
                // Layers 0–2 sliding (RoPE), 3 full (NoPE): with four layers
                // and an 8-row window, the prompts reach past it.
                ("muse-glimmer.attention.sliding_window", MetaValue::U32(8)),
                (
                    "muse-glimmer.attention.sliding_window_pattern",
                    MetaValue::U32(4),
                ),
                ("muse-glimmer.final_logit_softcapping", MetaValue::F32(20.0)),
                ("muse-glimmer.logit_scale", MetaValue::F32(0.5)),
            ],
            layer: cat(&[
                pre.clone(),
                attn(),
                swiglu(),
                vec![
                    ("attn_q_norm", vec![HD]),
                    ("attn_k_norm", vec![HD]),
                    ("attn_gate", vec![HEADS * HD, H]),
                    ("post_attention_norm", vec![H]),
                    ("post_ffw_norm", vec![H]),
                ],
            ]),
            top: vec![],
            rope_dims: HD,
        },
    ]
}

fn write_gguf(arch: &Arch, path: &std::path::Path) {
    let a = arch.name;
    let mut w = GgufWriter::new();
    w.set_arch(a);
    w.set_meta("general.name", MetaValue::String(format!("tiny-{a}")));
    for (k, v) in [
        ("vocab_size", VOCAB),
        ("embedding_length", H),
        ("feed_forward_length", INTER),
        ("block_count", LAYERS),
        ("attention.head_count", HEADS),
        ("attention.head_count_kv", KV),
        ("attention.key_length", HD),
        ("attention.value_length", HD),
        ("rope.dimension_count", arch.rope_dims),
        ("context_length", 8192),
    ] {
        w.set_meta(format!("{a}.{k}"), MetaValue::U32(v as u32));
    }
    w.set_meta(format!("{a}.rope.freq_base"), MetaValue::F32(10_000.0));
    w.set_meta(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        MetaValue::F32(1e-5),
    );
    w.set_meta(
        format!("{a}.attention.layer_norm_epsilon"),
        MetaValue::F32(1e-5),
    );
    for (k, v) in &arch.meta {
        w.set_meta(*k, v.clone());
    }
    let mut seed = 3u64;
    let mut put = |name: String, shape: &[usize]| {
        seed += 7;
        let n = shape.iter().product();
        let data = fill(n, seed);
        let ne: Vec<usize> = shape.iter().rev().copied().collect();
        if shape.len() == 2 {
            let bytes = rlx_gguf::quantize(&data, GgmlType::Q8_0).unwrap();
            w.add_tensor_bytes(name, ne, GgmlType::Q8_0, bytes).unwrap();
        } else {
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.add_tensor_bytes(name, ne, GgmlType::F32, bytes).unwrap();
        }
    };
    put("token_embd.weight".into(), &[VOCAB, H]);
    for i in 0..LAYERS {
        for (t, shape) in &arch.layer {
            let t = if t.contains('.') {
                t.to_string()
            } else {
                format!("{t}.weight")
            };
            put(format!("blk.{i}.{t}"), shape);
        }
    }
    put("output_norm.weight".into(), &[H]);
    for (t, shape) in &arch.top {
        put(t.to_string(), shape);
    }
    put("output.weight".into(), &[VOCAB, H]);
    w.write_to_path(path).unwrap();
}

fn check_arch(arch: &Arch) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join(arch.name);
    write_checkpoint(&dir, &json!({}), &HashMap::new(), VOCAB).unwrap();
    std::fs::remove_file(dir.join("config.json")).unwrap();
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    let gguf = dir.join("tiny.gguf");
    write_gguf(arch, &gguf);

    let out = tmp.path().join("model.rlxp");
    let opts = ExportOptions {
        uppers: vec![32, 64],
        ..ExportOptions::default()
    };
    let report = rlx_llama32::export::export_rlxp(&gguf, &out, &opts)
        .unwrap_or_else(|e| panic!("{}: export: {e:#}", arch.name));
    eprintln!("{}: {report}", arch.name);

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = Llama32Runner::builder()
        .weights(&gguf)
        .device(Device::Cpu)
        .build()
        .unwrap_or_else(|e| panic!("{}: runner: {e:#}", arch.name));
    let long: Vec<u32> = (0..30).map(|i| 3 + (i * 11) % 90).collect();
    let prompts = [vec![5u32, 9, 13, 17], vec![40, 41], long];
    let c = check_generate(&mut runner, m.lm().unwrap(), &prompts, 12, &[2]).unwrap();
    eprintln!("{}: {c}", arch.name);
    assert!(c.passed(), "{}: {c}", arch.name);
}

#[test]
fn every_llama_shaped_gguf_arch_exports_what_the_runner_decodes() {
    let only = rlx_ir::env::var("RLX_TEST_ARCH");
    let mut failed = Vec::new();
    for arch in arches() {
        if only.as_deref().is_some_and(|o| o != arch.name) {
            continue;
        }
        if let Err(e) = std::panic::catch_unwind(|| check_arch(&arch)) {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            failed.push(format!("{}: {msg}", arch.name));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
