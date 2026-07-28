//! Run an mlx-community 4-bit Qwen3 checkpoint through the packed
//! `Op::DequantMatMul { MlxAffine }` path (weights stay 4-bit in the arena)
//! and check parity against an mlx-lm greedy oracle.
//!
//! Usage:
//!   cargo run --release -p rlx-qwen3 --example mlx_community_run -- <model_dir>
//!
//! Oracle (mlx-lm, prompt "The capital of France is"):
//!   prefill argmax = 12095 (" Paris")

use anyhow::Result;
use rlx_qwen3::{Qwen3Config, Qwen3Runner, build_qwen3_graph_sized_packed};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const PROMPT: [u32; 5] = [785, 6722, 315, 9625, 374];

/// Diagnostic: build the packed graph with `with_lm_head=false` and return the
/// final normed hidden `[1, seq, hidden]` so wgpu-vs-CPU divergence can be
/// bisected away from the LM head. `RLX_DIAG_STAGE` (read inside the builder)
/// can further swap the output to an earlier layer-0 stage.
fn probe_hidden(cfg: &Qwen3Config, dir: &Path, device: Device) -> Result<Vec<f32>> {
    use rlx_core::flow_bridge::{
        compile_options_for_packed_gguf_prefill_with_profile, packed_gguf_compile_guard,
        packed_gguf_execution_device,
    };
    let actual = PROMPT.len();
    // Build at a (possibly larger) bucket to exercise active-extent trimming.
    let seq = rlx_ir::env::var("RLX_DIAG_BUCKET")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(actual)
        .max(actual);
    let with_lm_head = rlx_ir::env::flag("RLX_DIAG_LMHEAD");
    let last_from_input = rlx_ir::env::flag("RLX_DIAG_LAST");
    let mut loader = rlx_core::weight_loader::MlxLoader::open(dir.to_str().unwrap())?;
    let exec = packed_gguf_execution_device(device);
    let mut packed = HashMap::new();
    let (graph, params) = build_qwen3_graph_sized_packed(
        cfg,
        &mut loader,
        1,
        seq,
        with_lm_head,
        last_from_input,
        &mut packed,
    )?;
    let opts = compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        exec,
    );
    let mut compiled =
        packed_gguf_compile_guard(exec, || Session::new(exec).compile_with(graph, &opts));
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    for (n, (b, _, _)) in &packed {
        compiled.set_param_typed(n, b, rlx_ir::DType::U8);
    }
    // Pad the prompt up to the bucket; active-extent trims compute to `actual`.
    let mut ids: Vec<f32> = PROMPT.iter().map(|&t| t as f32).collect();
    ids.resize(seq, 0.0);
    let last = [(actual - 1) as f32];
    let inputs: Vec<(&str, &[f32])> = if last_from_input {
        vec![("input_ids", &ids), ("last_token_idx", &last)]
    } else {
        vec![("input_ids", &ids)]
    };
    let out = rlx_core::run_packed_prefill(&mut compiled, exec, actual, seq, &inputs);
    Ok(out.into_iter().next().unwrap_or_default())
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/Shared/rlx-models/.mlx-test/qwen3-0.6b-4bit".into());
    let dir = Path::new(&dir);

    let device = match std::env::args().nth(2).as_deref() {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("vulkan") => Device::Vulkan,
        Some("cuda") => Device::Cuda,
        Some("rocm") => Device::Rocm,
        Some("coreml") | Some("ane") => Device::Ane,
        _ => Device::Cpu,
    };
    eprintln!("device = {device:?}");

    // Registry auto-dispatch: a quantized mlx-community dir routes to MlxLoader.
    let via_registry = rlx_core::weight_loader::load_from_path(dir.to_str().unwrap())?;
    eprintln!(
        "registry dispatch: format_id = {}  ({} tensors)",
        via_registry.format_id(),
        via_registry.len()
    );
    drop(via_registry);

    let mut cfg = Qwen3Config::from_file(&dir.join("config.json"))?;
    // Diagnostic: truncate layers to isolate per-op backend divergence.
    if let Some(n) = rlx_ir::env::var("RLX_DIAG_NLAYERS").and_then(|s| s.parse::<usize>().ok()) {
        cfg.num_hidden_layers = n.min(cfg.num_hidden_layers);
        eprintln!(
            "DIAG: num_hidden_layers overridden to {}",
            cfg.num_hidden_layers
        );
    }
    if rlx_ir::env::flag("RLX_DIAG_HIDDEN") {
        let h = probe_hidden(&cfg, dir, device)?;
        let out =
            std::env::var("RLX_DIAG_OUT").unwrap_or_else(|_| ".mlx-test/diag_hidden.bin".into());
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
        for v in &h {
            f.write_all(&v.to_le_bytes())?;
        }
        eprintln!(
            "DIAG hidden: len={} stage={:?} -> {out}",
            h.len(),
            std::env::var("RLX_DIAG_STAGE").ok()
        );
        return Ok(());
    }

    eprintln!(
        "cfg: layers={} hidden={} heads={} kv={} head_dim={} inter={} vocab={} tie={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.tie_word_embeddings,
    );

    // "The capital of France is" (Qwen3 tokenizer, no chat template).
    let prompt_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let n_new = 6usize;
    let max_seq = prompt_ids.len() + n_new + 2;

    eprintln!("building packed MLX runner ({device:?}), max_seq={max_seq} ...");
    let mut runner = Qwen3Runner::from_mlx_packed(cfg, dir, max_seq, device)?;

    // Prefill parity: last-token logits.
    let logits = runner.predict_logits(&prompt_ids)?;
    let (argmax, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("prefill argmax = {argmax}  (oracle = 12095 \" Paris\")");
    let out_path = std::env::var("RLX_DIAG_OUT").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/.mlx-test/rlx_prefill_last_logits.bin".into()
    });
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for v in &logits {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;

    // Greedy generation.
    let generated = runner.generate_packed(&prompt_ids, n_new, |_| {})?;
    eprintln!("rlx greedy ids = {generated:?}");

    println!(
        "PREFILL_ARGMAX={argmax} PREFILL_OK={} FIRST_TOKEN_MATCH={}",
        argmax == 12095,
        generated.first() == Some(&12095)
    );
    Ok(())
}
