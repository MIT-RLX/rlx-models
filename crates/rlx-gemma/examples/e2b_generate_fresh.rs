// RLX — versatile ML compiler + runtime. GPLv3.
//! Hypothesis test: does MLX leak state across `compiled.run()` calls?
//! Re-compile the prefill graph fresh per token-step and compare vs the
//! single-compile loop in `e2b_generate.rs`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

const PROMPT_IDS: &[u32] = &[818, 5279, 529, 7001, 563];
const HF_REFERENCE: &[u32] = &[7001, 563, 7001, 563, 7001, 563, 7001, 563, 7001, 563];
const BUCKET: usize = 16;

fn ckpt_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    Ok(std::fs::read_dir(&base)?
        .flatten()
        .next()
        .context("snapshot")?
        .path())
}

fn pick(arg: Option<&str>) -> Result<Device> {
    match arg.unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "gpu" => Ok(Device::Gpu),
        other => bail!("unknown device {other:?}"),
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let device = pick(std::env::args().nth(1).as_deref())?;
    let label = format!("{device:?}");
    let dir = ckpt_dir()?;
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let vocab = cfg.vocab_size;
    let loader = GemmaQatLoader::open(&dir)?;
    let mut ids = vec![0u32; BUCKET];
    for (i, &t) in PROMPT_IDS.iter().enumerate() {
        ids[i] = t;
    }

    let mut generated: Vec<u32> = Vec::new();
    for step in 0..HF_REFERENCE.len() {
        let cur = PROMPT_IDS.len() + step;
        let ple = loader.compute_per_layer_inputs(&cfg, &ids)?;
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        // Build a fresh graph + compile per step (no shared MlxExecutable state).
        let mut bld = GemmaQatLoader::open(&dir)?;
        let mut packed = HashMap::new();
        let (graph, params) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
            &cfg,
            &mut bld,
            1,
            BUCKET,
            true,
            false,
            false,
            &mut packed,
            None,
            None,
        )?;
        let mut compiled = compile_graph_gemma_prefill_with_params(device, graph, params)?;
        let outs = compiled.run(&[
            ("input_ids", ids_f32.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ]);
        let logits = &outs[0];
        let next = argmax(&logits[(cur - 1) * vocab..cur * vocab]);
        println!("   [{label}] step#{step}: tok={next}");
        generated.push(next);
        if cur < BUCKET {
            ids[cur] = next;
        }
    }
    let matches = generated
        .iter()
        .zip(HF_REFERENCE)
        .take_while(|(a, b)| a == b)
        .count();
    println!("\n   [{label}] rlx={generated:?}");
    println!("   [{label}] hf ={HF_REFERENCE:?}");
    println!("   [{label}] match={matches}/{}", HF_REFERENCE.len());
    Ok(())
}
