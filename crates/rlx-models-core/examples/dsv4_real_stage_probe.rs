// RLX — versatile ML compiler + runtime. GPLv3.
//! Builds one DeepSeek-V4 pipeline STAGE from a REAL mlx-community checkpoint on
//! this machine — proving the config parses to the real spec, the downloaded
//! shards carry exactly this stage's tensors, and `build_deepseek_v4_stage`
//! constructs from them. Uses `MlxLoader::open_lazy` (mmap) + `StructureLoader`
//! so it reads only tensor SHAPES (peak RAM = one tensor), not the ~40 GB of
//! weights — a fast integrity+wiring check before the full cross-machine run.
//!
//!   RLX_DSV4_DIR=/path/to/DeepSeek-V4-Flash-2bit-DQ \
//!   RLX_DSV4_LAYERS=0:18 RLX_DSV4_FIRST=1 RLX_DSV4_LAST=0 \
//!   cargo run --release -p rlx-models-core --example dsv4_real_stage_probe

use anyhow::{Context, Result};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::StructureLoader;
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_stage};
use rlx_models_core::weight_loader::MlxLoader;
use std::collections::HashMap;

fn env_flag(k: &str) -> bool {
    matches!(std::env::var(k).as_deref(), Ok("1") | Ok("true"))
}

fn main() -> Result<()> {
    let dir = std::env::var("RLX_DSV4_DIR").context("set RLX_DSV4_DIR to the checkpoint dir")?;
    let lr = std::env::var("RLX_DSV4_LAYERS").unwrap_or_else(|_| "0:18".into());
    let (a, b) = lr.split_once(':').context("RLX_DSV4_LAYERS=A:B")?;
    let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
    let first = env_flag("RLX_DSV4_FIRST");
    let last = env_flag("RLX_DSV4_LAST");
    let seq: usize = std::env::var("RLX_DSV4_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
    let spec = DeepseekV4Spec::from_config(&cfg)?;
    println!(
        "config: {} layers, dim {}, {} heads, {} experts top-{}, hc_mult {}, index_topk {}",
        spec.n_layers,
        spec.dim,
        spec.n_heads,
        spec.n_routed_experts,
        spec.n_activated_experts,
        spec.hc_mult,
        spec.index_topk
    );
    println!("building stage layers {a}..{b} (first={first} last={last}) from {dir}");

    let mut loader = MlxLoader::open_lazy(&dir).context("open_lazy checkpoint")?;
    let mut structure = StructureLoader::new(&mut loader);
    let mut packed = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
    let t0 = std::time::Instant::now();
    let (graph, params) =
        build_deepseek_v4_stage(&spec, &mut structure, seq, a..b, first, last, &mut packed)?;
    let held: usize = params.values().map(|v| v.len()).sum();
    let out = graph.node(*graph.outputs.first().unwrap());
    println!(
        "✅ stage built in {:.1?}: {} nodes, {} params ({} deferred via manifest), {} packed-defer; \
         held only {} f32 (masks/rope, not weights); boundary out shape {:?}",
        t0.elapsed(),
        graph.len(),
        params.len(),
        structure.manifest.len(),
        packed.len(),
        held,
        out.shape.dims(),
    );
    Ok(())
}
