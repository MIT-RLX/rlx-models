// RLX — versatile ML compiler + runtime. GPLv3.
//! **Ready-to-run** DeepSeek-V4 real-checkpoint validation harness. Loads an
//! mlx-community `DeepSeek-V4-Flash-*` directory (`config.json` +
//! affine/MXFP4 safetensors), parses it with [`DeepseekV4Spec::from_config`],
//! builds the prefill graph with [`build_deepseek_v4_prefill`], compiles on CPU,
//! runs a prefill, and reports the last-token argmax + finiteness (and, given
//! `RLX_DSV4_ORACLE_ARGMAX`, checks it). This is the one command that upgrades
//! deepseek_v4 WiredDeferred → Validated — but it needs a machine with enough
//! RAM: the smallest checkpoint (2-bit) is ~96 GB, the 4-bit ~151 GB, so it
//! CANNOT run on a 64 GB box. All subsystem cores + the assembled forward are
//! already cos-exact / finite-validated on synthetic configs; this closes the
//! remaining real-weights-at-scale gap wherever the hardware exists.
//!
//!   RLX_DSV4_DIR=/path/to/DeepSeek-V4-Flash-4bit \
//!   cargo run --release -p rlx-models-core --example dsv4_validate -- 0,1,2,3

use anyhow::{Context, Result};
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_prefill};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn main() -> Result<()> {
    let dir = std::env::var("RLX_DSV4_DIR")
        .context("set RLX_DSV4_DIR to a mlx-community/DeepSeek-V4-Flash-* directory")?;
    let ids: Vec<u32> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0u32, 1, 2, 3]);
    let seq = ids.len();

    let cfg_bytes = std::fs::read(format!("{dir}/config.json")).context("read config.json")?;
    let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes).context("parse config.json")?;
    let spec = DeepseekV4Spec::from_config(&cfg)?;
    eprintln!(
        "[dsv4] {} layers · dim {} · heads {} · {} experts top-{} · hc_mult {} · index_topk {} · hash {}",
        spec.n_layers,
        spec.dim,
        spec.n_heads,
        spec.n_routed_experts,
        spec.n_activated_experts,
        spec.hc_mult,
        spec.index_topk,
        spec.n_hash_layers
    );

    let mut loader = MlxLoader::open(&dir).context("open mlx dir")?;
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_deepseek_v4_prefill(&spec, &mut loader, seq, &mut packed)?;
    eprintln!(
        "[dsv4] graph built: {} f32 params · {} packed tensors",
        params.len(),
        packed.len()
    );

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    for (n, (bytes, _scheme, _shape)) in &packed {
        // Quant scheme + shape are carried by the graph's Dequant* nodes; the
        // runtime just needs the packed U8 bytes.
        compiled.set_param_typed(n, bytes, rlx_ir::DType::U8);
    }
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let logits = compiled
        .run(&[("input_ids", ids_f.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    let vocab = spec.vocab_size;
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let finite = logits.iter().all(|v| v.is_finite());
    println!("[dsv4] prefill seq={seq} → last-token argmax = {argmax}  finite = {finite}");

    if let Ok(oracle) = std::env::var("RLX_DSV4_ORACLE_ARGMAX") {
        let oracle: usize = oracle
            .trim()
            .parse()
            .context("RLX_DSV4_ORACLE_ARGMAX must be an int")?;
        if argmax == oracle {
            println!("✅ argmax matches oracle {oracle} — deepseek_v4 VALIDATED on real weights");
        } else {
            anyhow::bail!("argmax {argmax} != oracle {oracle}");
        }
    } else {
        println!("(set RLX_DSV4_ORACLE_ARGMAX=<mlx-lm argmax> to assert an exact oracle match)");
    }
    Ok(())
}
