//! Model-agnostic packed-prefill runner for huggingface.co/mlx-community
//! dense decoders — NO per-model crate. Reads `config.json`, infers a
//! [`DecoderSpec`] (topology flags probed from the tensor names), builds the
//! generic packed graph, and runs one prefill.
//!
//! Prompt ids are read from `<dir>/oracle.json` (`prompt_ids`) when present
//! so the run lines up with an mlx-lm oracle (see
//! `scripts/mlx_oracle_dump.py`); otherwise pass `--ids a,b,c`.
//!
//! Usage:
//!   cargo run --release -p rlx-models-core --example mlx_auto_run -- <dir> [device] [--ids 1,2,3]
//!
//! Validated: mlx-community/Llama-3.2-1B-Instruct-4bit → prefill argmax 12366.

use anyhow::{Result, anyhow};
use rlx_models_core::standard_decoder::{DecoderSpec, build_standard_decoder_packed};
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn parse_device(s: Option<&str>) -> Device {
    match s {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("vulkan") => Device::Vulkan,
        Some("cuda") => Device::Cuda,
        Some("rocm") => Device::Rocm,
        Some("coreml") | Some("ane") => Device::Ane,
        _ => Device::Cpu,
    }
}

/// Prompt ids from `--ids a,b,c`, else `<dir>/oracle.json`.prompt_ids, else
/// the Qwen3 France prompt as a fallback.
fn prompt_ids(args: &[String], dir: &Path) -> (Vec<u32>, Option<i64>) {
    if let Some(pos) = args.iter().position(|a| a == "--ids") {
        if let Some(list) = args.get(pos + 1) {
            let ids = list
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();
            return (ids, None);
        }
    }
    if let Ok(bytes) = std::fs::read(dir.join("oracle.json")) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let ids: Vec<u32> = v
                .get("prompt_ids")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as u32))
                        .collect()
                })
                .unwrap_or_default();
            let oracle = v.get("prefill_argmax").and_then(|x| x.as_i64());
            if !ids.is_empty() {
                return (ids, oracle);
            }
        }
    }
    (vec![785, 6722, 315, 9625, 374], None)
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let dir = raw
        .first()
        .cloned()
        .unwrap_or_else(|| ".mlx-test/llama32-1b-4bit".into());
    let dir = Path::new(&dir);
    let device = parse_device(raw.get(1).map(|s| s.as_str()));
    let (ids, oracle_argmax) = prompt_ids(&raw, dir);

    eprintln!("device = {device:?}, dir = {}", dir.display());

    // Probe the loader read-only for topology inference, then build the spec.
    let mut loader = MlxLoader::open(dir.to_str().ok_or_else(|| anyhow!("non-utf8 dir"))?)?;
    let spec = DecoderSpec::from_config_json(dir, &loader)?;
    eprintln!(
        "spec: arch={} layers={} hidden={} heads={} kv={} head_dim={} inter={} vocab={} \
         qk_norm={} attn_bias={} tie={} rope_theta={} rope_scaling={:?} act={}",
        spec.arch,
        spec.num_hidden_layers,
        spec.hidden_size,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.intermediate_size,
        spec.vocab_size,
        spec.qk_norm,
        spec.attention_bias,
        spec.tie_word_embeddings,
        spec.rope_theta,
        spec.rope_scaling,
        spec.hidden_act,
    );

    // Diagnostic: RLX_DIAG_HIDDEN builds with_lm_head=false so the output is the
    // trunk hidden `[1, seq, hidden]` (or an earlier layer-0 stage via
    // RLX_DIAG_STAGE), to bisect a backend divergence away from the LM head.
    let diag_hidden = rlx_ir::env::flag("RLX_DIAG_HIDDEN");
    // wgpu (`Device::Gpu`) on NVIDIA/Vulkan miscomputes the large tied-LM-head
    // F32 matmul (`[1,hidden]×[hidden,vocab]`, ~1 GiB weight) → all-zero logits
    // (driver defect, isolated in rlx-wgpu). Work around it by running the whole
    // transformer trunk on the GPU and computing only the final LM-head
    // projection on the host — cheap (one matmul) and the trunk still runs on
    // device. Opt out with RLX_WGPU_INGRAPH_HEAD=1. Other backends are unaffected.
    // `RLX_FORCE_HOST_HEAD=1` exercises the host-LM-head path on any device
    // (e.g. CPU, whose trunk is trusted) to validate the untied projection
    // independently of the wgpu driver defect that motivated it.
    let host_head = (matches!(device, Device::Gpu) || rlx_ir::env::flag("RLX_FORCE_HOST_HEAD"))
        && !rlx_ir::env::flag("RLX_WGPU_INGRAPH_HEAD")
        && !diag_hidden
        && std::env::var("RLX_DIAG_STAGE").is_err();
    if host_head {
        eprintln!(
            "[mlx_auto_run] trunk on {device:?}, LM head on host \
             ({} projection; wgpu large-F32-matmul driver-defect workaround)",
            if spec.tie_word_embeddings {
                "tied embed"
            } else {
                "untied lm_head.weight"
            },
        );
    }
    let with_lm_head = !diag_hidden && !host_head;
    let actual = ids.len();
    let seq = actual;
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_standard_decoder_packed(
        &spec,
        &mut loader,
        1,
        seq,
        with_lm_head,
        /*last_from_input*/ with_lm_head,
        /*embeds_input*/ false,
        &mut packed,
    )?;

    let exec = rlx_models_core::flow_bridge::packed_gguf_execution_device(device);
    if exec != device {
        eprintln!("[mlx_auto_run] {device:?} routes packed prefill to {exec:?}");
    }
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        exec,
    );
    let mut compiled = rlx_models_core::flow_bridge::packed_gguf_compile_guard(exec, || {
        Session::new(exec).compile_with(graph, &opts)
    });
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    for (n, (b, _, _)) in &packed {
        compiled.set_param_typed(n, b, rlx_ir::DType::U8);
    }

    let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let last = [(actual - 1) as f32];
    let inputs: Vec<(&str, &[f32])> = if with_lm_head {
        vec![
            ("input_ids", ids_f32.as_slice()),
            ("last_token_idx", last.as_slice()),
        ]
    } else {
        vec![("input_ids", ids_f32.as_slice())]
    };
    let out = rlx_models_core::run_packed_prefill(&mut compiled, exec, actual, seq, &inputs);
    let raw = out.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    // Host LM head: `raw` is the trunk hidden `[1, seq, hidden]`; project the
    // last token's hidden through the LM-head weight `[vocab, hidden]` on the
    // CPU. `logits[v] = <hidden_last, head_row_v>`. Tied models reuse the embed
    // table (already in `params`); untied models take `lm_head.weight` from the
    // loader (dequantized to f32) — it was left untaken because the graph was
    // built with_lm_head=false.
    let logits = if host_head {
        let h = spec.hidden_size;
        let vocab = spec.vocab_size;
        let last_off = (actual - 1) * h;
        let hidden_last = raw
            .get(last_off..last_off + h)
            .ok_or_else(|| anyhow!("trunk hidden too short: {} < {}", raw.len(), last_off + h))?;
        let untied_w: Option<Vec<f32>> = if spec.tie_word_embeddings {
            None
        } else {
            Some(
                loader
                    .take("lm_head.weight")
                    .map(|(w, _shape)| w)
                    .map_err(|e| anyhow!("host LM head (untied) needs lm_head.weight: {e}"))?,
            )
        };
        let head_w: &[f32] = match &untied_w {
            Some(w) => w.as_slice(),
            None => params
                .get("model.embed_tokens.weight")
                .map(|v| v.as_slice())
                .ok_or_else(|| anyhow!("host LM head (tied) needs model.embed_tokens.weight"))?,
        };
        if head_w.len() < vocab * h {
            return Err(anyhow!(
                "LM-head weight {} < vocab*hidden {}",
                head_w.len(),
                vocab * h
            ));
        }
        let mut logits = vec![0f32; vocab];
        for (v, slot) in logits.iter_mut().enumerate() {
            let row = &head_w[v * h..v * h + h];
            *slot = hidden_last.iter().zip(row).map(|(a, b)| a * b).sum();
        }
        logits
    } else {
        raw
    };

    // Report raw stats (not argmax) whenever the graph output isn't full logits:
    // RLX_DIAG_HIDDEN (with_lm_head=false trunk) OR RLX_DIAG_STAGE=head_input
    // (with_lm_head=true but output overridden to the gathered last-token hidden).
    let stats_mode = diag_hidden || std::env::var("RLX_DIAG_STAGE").is_ok();
    if stats_mode {
        // Trunk-hidden bisection: report stats so a zeroed/collapsed stage on
        // a specific backend is obvious without a host oracle.
        let n = logits.len();
        let nonzero = logits.iter().filter(|&&x| x != 0.0).count();
        let l2 = logits
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let stage = std::env::var("RLX_DIAG_STAGE").unwrap_or_else(|_| "final_norm".into());
        let head: Vec<f32> = logits.iter().take(6).copied().collect();
        eprintln!("DIAG hidden stage={stage} len={n} nonzero={nonzero} l2={l2:.4} head={head:?}");
        let dev_tag = format!("{device:?}").to_lowercase();
        let out_path = dir.join(format!("rlx_hidden_{dev_tag}.bin"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        for v in &logits {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
        println!("DIAG_HIDDEN stage={stage} nonzero={nonzero} l2={l2:.4}");
        return Ok(());
    }

    let vocab = spec.vocab_size;
    if logits.len() < vocab {
        return Err(anyhow!("logits short: {} < {vocab}", logits.len()));
    }
    let logits = &logits[..vocab];
    let (argmax, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let out_path = dir.join("rlx_prefill_last_logits.bin");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for v in logits {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;

    match oracle_argmax {
        Some(o) => {
            eprintln!("prefill argmax = {argmax}  (oracle = {o})");
            println!(
                "PREFILL_ARGMAX={argmax} ORACLE={o} MATCH={}",
                argmax as i64 == o
            );
        }
        None => {
            eprintln!("prefill argmax = {argmax}  (no oracle.json)");
            println!("PREFILL_ARGMAX={argmax}");
        }
    }
    Ok(())
}
