// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// End-to-end qwen3 inference with PACKED GGUF weights: K-quant
// matmul weights stay in the arena as raw bytes; the graph emits
// `Op::DequantMatMul { scheme }` so the kernel dequants per call.
// Cuts the host RAM footprint by ~7-9× vs the default F32-load
// path — the path to 27 B-class Qwen models fitting on a 32 GB
// Mac.
//
// Usage:
//   cargo run --release -p rlx-models \
//       --example qwen3_packed_inference -- /path/to/Qwen3-X-Q4_K_M.gguf
//
// Optional: set RLX_QWEN3_PARITY to also build the F32-load path
// for the same file and compare the last-position logits (cosine,
// top-1 match). Skips parity on machines where the F32 path
// wouldn't fit.

use anyhow::{Context, Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized_packed};
use rlx_models::weight_loader::GgufLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;

fn extract_config(raw: &GgufFile) -> Result<Qwen3Config> {
    let g_u32 = |k: &str| -> Result<u32> {
        raw.metadata
            .get(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow!("missing GGUF metadata: {k}"))
    };
    let g_f32 = |k: &str| -> Option<f32> {
        raw.metadata.get(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let g_bool = |k: &str| -> Option<bool> {
        raw.metadata.get(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };
    let hidden_size = g_u32("qwen3.embedding_length")? as usize;
    let num_attention_heads = g_u32("qwen3.attention.head_count")? as usize;
    let head_dim_default = hidden_size.checked_div(num_attention_heads).unwrap_or(128);
    Ok(Qwen3Config {
        vocab_size: g_u32("qwen3.vocab_size").unwrap_or(151_936) as usize,
        hidden_size,
        intermediate_size: g_u32("qwen3.feed_forward_length")? as usize,
        num_hidden_layers: g_u32("qwen3.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: g_u32("qwen3.attention.head_count_kv")? as usize,
        head_dim: g_u32("qwen3.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim_default),
        attention_bias: false,
        qk_norm: true,
        max_position_embeddings: g_u32("qwen3.context_length").unwrap_or(40_960) as usize,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: g_bool("qwen3.tie_word_embeddings").unwrap_or(true),
        rope_theta: g_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rms_norm_eps: g_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
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
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: qwen3_packed_inference <file.gguf>"))?
        .into();

    let raw = GgufFile::from_path(&path).with_context(|| format!("opening {path:?}"))?;
    let cfg = extract_config(&raw)?;
    eprintln!(
        "[packed] file={path:?} tensors={} hidden={} layers={} vocab={} tied={}",
        raw.tensors.len(),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.vocab_size,
        cfg.tie_word_embeddings,
    );

    // Estimate the load-time footprint we're avoiding. Sum sizes of
    // the K-quant tensors (which will stay packed) vs. their f32
    // dequant equivalent.
    let mut packed_bytes: u64 = 0;
    let mut f32_dequant_bytes: u64 = 0;
    for t in raw.tensors.values() {
        let n = t.n_elements() as u64;
        let f32_size = n * 4;
        let packed_size = match t.dtype {
            rlx_gguf::GgmlType::Q4K => (n / 256) * 144,
            rlx_gguf::GgmlType::Q5K => (n / 256) * 176,
            rlx_gguf::GgmlType::Q6K => (n / 256) * 210,
            rlx_gguf::GgmlType::Q8K => (n / 256) * 292,
            rlx_gguf::GgmlType::F32 => f32_size,
            rlx_gguf::GgmlType::F16 | rlx_gguf::GgmlType::BF16 => n * 2,
            _ => f32_size,
        };
        packed_bytes += packed_size;
        f32_dequant_bytes += f32_size;
    }
    let to_gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
    eprintln!(
        "[packed] memory estimate:  packed={:.2} GB  vs  full-f32-dequant={:.2} GB  →  {:.1}× smaller",
        to_gb(packed_bytes),
        to_gb(f32_dequant_bytes),
        f32_dequant_bytes as f64 / packed_bytes as f64,
    );

    // Build the prefill graph in packed mode.
    let batch = 1usize;
    let seq = 8usize;
    let mut loader = GgufLoader::from_file(path.to_str().unwrap())?;
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_qwen3_graph_sized_packed(
        &cfg,
        &mut loader,
        batch,
        seq,
        /*with_lm_head*/ true,
        /*last_logits_only*/ true,
        &mut packed,
    )?;
    eprintln!(
        "[packed] graph: {} nodes, {} f32 params, {} packed K-quant params",
        graph.len(),
        params.len(),
        packed.len()
    );
    let packed_total: usize = packed.values().map(|(b, _, _)| b.len()).sum();
    let f32_total: usize = params.values().map(|v| v.len() * 4).sum();
    eprintln!(
        "[packed] arena projection:  packed={:.2} GB  f32={:.2} GB",
        packed_total as f64 / 1024.0 / 1024.0 / 1024.0,
        f32_total as f64 / 1024.0 / 1024.0 / 1024.0
    );

    // Compile on CPU — Metal lowering for Op::DequantMatMul is TBD.
    eprintln!("[packed] compiling on CPU");
    let t_compile = std::time::Instant::now();
    let mut compiled =
        rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(Device::Cpu, graph, params)?;
    for (name, (bytes, _scheme, _shape)) in &packed {
        compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
    }
    eprintln!(
        "[packed] compiled + params uploaded in {:?}",
        t_compile.elapsed()
    );

    // Forward pass — deterministic input ids.
    let ids: Vec<f32> = (0..batch * seq).map(|i| (100 + i * 17) as f32).collect();
    let t_run = std::time::Instant::now();
    let out = compiled.run(&[("input_ids", ids.as_slice())]);
    let logits = out.into_iter().next().ok_or_else(|| anyhow!("no logits"))?;
    eprintln!(
        "[packed] forward in {:?}; logits.len={} min={:.3} max={:.3} top1_id={} top1_logit={:.3}",
        t_run.elapsed(),
        logits.len(),
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    // Optional parity vs F32-load path.
    if rlx_ir::env::flag("RLX_QWEN3_PARITY") {
        eprintln!("[parity] building F32-load reference path…");
        use rlx_models::qwen3::build_qwen3_graph_sized_last_logits;
        let mut loader_ref = GgufLoader::from_file(path.to_str().unwrap())?;
        let (g_ref, p_ref) = build_qwen3_graph_sized_last_logits(
            &cfg,
            &mut loader_ref,
            batch,
            seq,
            /*with_kv_outputs*/ false,
        )?;
        let mut c_ref =
            rlx_models::flow_util::compile_graph_legacy_with_params(Device::Cpu, g_ref, p_ref)?;
        let out_ref = c_ref.run(&[("input_ids", ids.as_slice())]);
        let logits_ref = out_ref.into_iter().next().unwrap();
        let dot: f32 = logits.iter().zip(&logits_ref).map(|(a, b)| a * b).sum();
        let na: f32 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = logits_ref.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(f32::MIN_POSITIVE);
        let max_abs = logits
            .iter()
            .zip(&logits_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top1_packed = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        let top1_ref = logits_ref
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        eprintln!(
            "[parity] cosine={cos:.5} max|Δ|={max_abs:.3} top1_packed={top1_packed} top1_ref={top1_ref} match={}",
            top1_packed == top1_ref
        );
    }

    eprintln!("OK — packed-weights forward pass succeeded.");
    Ok(())
}
