// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// End-to-end Qwen3-from-GGUF: take a Q4_K_M GGUF, build the qwen3
// graph via the HF↔GGUF name mapper, run prefill, and report the
// top-1 prediction for the last position. Proves the dequant +
// name-mapping path composes with the existing graph builder.
//
// Usage:
//   cargo run --release --example qwen3_gguf_inference -- \
//       /path/to/Qwen3-X-Q4_K_M.gguf
//
// The example loads the Qwen3 config from the GGUF metadata when
// possible, falling back to defaults for the 0.6B model.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized_last_logits};
use rlx_models::weight_loader::GgufLoader;
use rlx_runtime::Device;

fn extract_config(meta: &GgufFile) -> Result<Qwen3Config> {
    let get_u32 = |k: &str| -> Result<u32> {
        meta.metadata
            .get(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow!("missing GGUF metadata: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        meta.metadata.get(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        meta.metadata.get(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };
    let hidden_size = get_u32("qwen3.embedding_length")? as usize;
    let num_attention_heads = get_u32("qwen3.attention.head_count")? as usize;
    let head_dim_default = hidden_size.checked_div(num_attention_heads).unwrap_or(128);
    Ok(Qwen3Config {
        vocab_size: get_u32("qwen3.vocab_size").unwrap_or(151_936) as usize,
        hidden_size,
        intermediate_size: get_u32("qwen3.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("qwen3.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("qwen3.attention.head_count_kv")? as usize,
        head_dim: get_u32("qwen3.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim_default),
        attention_bias: false,
        qk_norm: true,
        max_position_embeddings: get_u32("qwen3.context_length").unwrap_or(40960) as usize,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: get_bool("qwen3.tie_word_embeddings").unwrap_or(true),
        rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rms_norm_eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        use_sliding_window: false,
        hidden_act: "silu".into(),
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    })
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: qwen3_gguf_inference <file.gguf>");

    println!("[1/4] reading GGUF header from {path}");
    let raw = GgufFile::from_path(&path)?;
    println!(
        "      tensors={} metadata_keys={} arch={:?}",
        raw.tensors.len(),
        raw.metadata.len(),
        raw.metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
    );

    println!("[2/4] extracting config");
    let cfg = extract_config(&raw)?;
    println!(
        "      hidden={} layers={} q_heads={} kv_heads={} head_dim={} vocab={} tied={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.tie_word_embeddings,
    );

    println!("[3/4] building qwen3 graph (B=1, L=8, last-only logits)");
    let mut loader = GgufLoader::from_file(&path)?;
    let (graph, params) = build_qwen3_graph_sized_last_logits(
        &cfg,
        &mut loader,
        1,
        8,
        /*with_kv_outputs*/ false,
    )?;
    println!(
        "      graph: {} nodes; params loaded: {} tensors",
        graph.len(),
        params.len()
    );

    println!("[4/4] compiling on Metal + running prefill");
    let mut compiled = rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(
        Device::Metal,
        graph,
        params,
    )?;
    // Token pool reused from the matrix harness — arbitrary ids within
    // vocab; we only care that the forward runs and produces a valid
    // logit distribution.
    let ids: Vec<f32> = (0..8u32).map(|i| (100 + i * 17) as f32).collect();
    let outputs = compiled.run(&[("input_ids", ids.as_slice())]);
    let logits = outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no logits returned"))?;
    let n = logits.len();
    if n != cfg.vocab_size {
        anyhow::bail!("unexpected logits length {n}, expected {}", cfg.vocab_size);
    }
    let (top1, top1_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, v)| (i, *v))
        .unwrap();
    let any_nan = logits.iter().any(|x| x.is_nan());
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "      logits.len={n} min={min:.3} max={max:.3} any_nan={any_nan} top1_id={top1} top1_logit={top1_val:.3}"
    );
    if any_nan {
        anyhow::bail!("NaN in logits — pipeline broken");
    }

    // Optional safetensors parity check: if RLX_QWEN3_WEIGHTS points
    // at the corresponding F32 safetensors file, run the same input
    // through that path and compare top-1 + cosine. Q4_K_M is lossy
    // so we expect cosine ~0.97+ and most top-1s to agree, not
    // bit-exact parity.
    if let Some(st_path) = rlx_ir::env::var("RLX_QWEN3_WEIGHTS") {
        println!("\n[parity] comparing against safetensors weights at {st_path}");
        let mut st_loader = rlx_models::weight_map::WeightMap::from_file(&st_path)?;
        let (graph2, params2) =
            build_qwen3_graph_sized_last_logits(&cfg, &mut st_loader, 1, 8, false)?;
        let mut compiled2 = rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(
            Device::Metal,
            graph2,
            params2,
        )?;
        let outputs2 = compiled2.run(&[("input_ids", ids.as_slice())]);
        let st_logits = outputs2.into_iter().next().unwrap();
        if st_logits.len() != logits.len() {
            anyhow::bail!("safetensors logits length mismatch");
        }
        let (st_top1, _) = st_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        let dot: f32 = logits.iter().zip(&st_logits).map(|(a, b)| a * b).sum();
        let na: f32 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = st_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(f32::MIN_POSITIVE);
        let max_abs = logits
            .iter()
            .zip(&st_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "      st_top1={st_top1} gguf_top1={top1} top1_match={} cosine={cos:.5} max|Δ|={max_abs:.3}",
            st_top1 == top1
        );
    }

    println!("\nOK — GGUF → graph → Metal forward end-to-end succeeded.");
    Ok(())
}
