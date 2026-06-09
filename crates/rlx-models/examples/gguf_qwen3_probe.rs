// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Verify the HF → GGUF name mapping resolves correctly against a
// real qwen3-architecture GGUF. Usage:
//   cargo run --release --example gguf_qwen3_probe -- <path-to-file.gguf>
//
// Independent of the rlx-* graph machinery — only exercises the
// loader-level translation + the K-quant decoders. Useful for
// validating that "I downloaded an unsloth Qwen3 GGUF" works before
// trying to actually build the graph against it.

use anyhow::Result;
use rlx_models::weight_loader::{GgufLoader, WeightLoader, hf_to_gguf_name};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_qwen3_probe <file.gguf>");
    let mut loader = GgufLoader::from_file(&path)?;

    let hf_names = std::iter::once("model.embed_tokens.weight".to_string())
        .chain(std::iter::once("model.norm.weight".to_string()))
        .chain(
            [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]
            .iter()
            .map(|tail| format!("model.layers.0.{tail}")),
        )
        .collect::<Vec<_>>();

    let remaining_set: std::collections::HashSet<String> =
        loader.remaining_keys().into_iter().collect();

    println!("HF → GGUF resolution against {path}:");
    let mut hits = 0usize;
    let mut decoded = 0usize;
    let mut decode_fail: Vec<String> = Vec::new();
    for hf in &hf_names {
        let gguf = hf_to_gguf_name(hf);
        let exists = gguf.as_ref().is_some_and(|g| remaining_set.contains(g));
        let mark = if exists { "ok " } else { "no " };
        if exists {
            hits += 1;
        }
        // Try to actually decode tiny ones to prove the dequant path.
        let mut note = String::new();
        if exists {
            match loader.take(hf) {
                Ok((data, shape)) => {
                    decoded += 1;
                    let n = data.len();
                    let any_nan = data.iter().any(|x| x.is_nan());
                    note = format!(" shape={shape:?} n={n} any_nan={any_nan}");
                }
                Err(e) => {
                    decode_fail.push(format!("{hf}: {e}"));
                    note = " decode_err".to_string();
                }
            }
        }
        println!(
            "  {mark} {hf:55} → {:?}{note}",
            gguf.as_deref().unwrap_or("(no mapping)")
        );
    }
    println!();
    println!("Resolved: {hits}/{}", hf_names.len());
    println!("Decoded:  {decoded}/{hits}");
    if !decode_fail.is_empty() {
        println!("Decode failures (likely unsupported quant for this build):");
        for f in &decode_fail {
            println!("  {f}");
        }
    }

    let mtp = loader.mtp_keys();
    println!();
    println!("MTP heads in file: {}", mtp.len());
    for n in mtp.iter().take(5) {
        println!("  {n}");
    }
    Ok(())
}
