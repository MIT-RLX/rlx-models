// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a MiniCPM-V 4.6 / 4.7 checkpoint directory as an `.rlxp` package
//! rlx runs alone: the vision graphs at each crop bucket the runner pads to,
//! a decode step of the Qwen3.5 trunk with its state inside (the KV caches
//! written in place at `position`, the recurrent states replaced), taking a
//! token or an image row and its M-RoPE positions, and a program answering
//! `run` with what [`crate::MiniCpmVRunner::chat`] does around them — from
//! this crate's own files: the pictures sliced (`preprocess.rs`), the chat
//! prompt and placeholders (`prompt.rs`), each crop's feeds (`feeds.rs`),
//! the canvas positions (`positions.rs`) — the prompt run through the step
//! token by token, then the reply decoded, the caches kept in the runtime.
//!
//! `Model.run([image0, …], {prompt, max_tokens, thinking})` (pictures `u8
//! [h, w, 3]`) returns `text`, the reply as UTF-8 bytes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::{Checkpoint, Package, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::{DType, Graph};
use serde_json::{Value, json};

use crate::config::MiniCpmVConfig;

use rlx_qwen35::export_vl::{CAP, PAST};
/// The longest context exported when the options name none.
const DEFAULT_CONTEXT: usize = 4096;
/// Crop buckets exported (patches). The slicer aims each crop at
/// `scale_resolution²` pixels — 1024 patches — and its rounding to the
/// patch grid can land past that (1056 for a 4:3 photo), so the largest
/// leaves half again as much.
const BUCKETS: [usize; 6] = [256, 512, 768, 1024, 1280, 1536];

type Params = HashMap<String, Vec<f32>>;

/// The vision graph for crops of up to `bucket` patches, as the runner
/// compiles it (its tower in f32).
fn vision_graph(dir: &Path, cfg: &MiniCpmVConfig, bucket: usize) -> Result<(Graph, Params)> {
    let mut wm = crate::runner::load_vision_weights(dir)?;
    let built = crate::vision::build_vision(cfg, &mut wm, bucket, DType::F32)?;
    let (mut g, p) = rlx_core::flow_util::graph_from_built(built)?;
    g.name = format!("vision_{bucket}");
    let out = g.outputs[0];
    g.node_mut(out).name = Some("features".into());
    Ok((g, p))
}

/// Export the MiniCPM-V directory `dir` (`config.json`,
/// `preprocessor_config.json`, `model.safetensors`, `tokenizer.json`) as
/// the package `out`. `opts.max_context` bounds prompt and reply together
/// (default 4096).
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let mcfg = MiniCpmVConfig::from_dir(dir)?;
    let mut cfg = rlx_qwen35::Qwen35Config::from_hf_config_json(&dir.join("config.json"))?;
    cfg.mrope_interleaved = true;
    let rows = opts.max_context.unwrap_or(DEFAULT_CONTEXT) + 1;
    let half = cfg.rope_dim_count / 2;
    let ck = Checkpoint::open(dir)?;
    let mut refr = ck.referencer(opts.embed_weights)?;

    let (step, states) =
        rlx_qwen35::export_vl::host_embed_step(dir, &cfg, rows, &mut |p| refr.take(p))?;
    let mut extra = Params::new();
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, rows, None, half);
    extra.insert("rope_cos_rows".into(), cos);
    extra.insert("rope_sin_rows".into(), sin);
    extra.insert(
        "mrope_sections".into(),
        rlx_qwen35::mrope_section_masks(&cfg, half)?,
    );
    extra.insert("one".into(), vec![1.0]);
    extra.insert("decode.iota".into(), (0..rows).map(|i| i as f32).collect());
    let mut loader = ck.loader()?;
    let (table, _) = loader
        .take("model.language_model.embed_tokens.weight")
        .context("no model.language_model.embed_tokens.weight")?;
    drop(loader);
    extra.insert("embed_tokens".into(), table);
    refr.take(extra)?;

    let mut graphs = vec![step];
    for &b in &BUCKETS {
        let (g, p) = vision_graph(dir, &mcfg, b)?;
        refr.take(p)?;
        graphs.push(g);
    }

    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        "minicpm-v",
        "image-text-to-text",
        "rlx-minicpm-v",
    );
    pkg.weights_as_stored();
    pkg.graphs = graphs;
    pkg.dims = vec![("cap".into(), CAP), ("past".into(), PAST)];
    pkg.reference(refr, out)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;

    let read = |name: &str| -> Result<Value> {
        let p = dir.join(name);
        Ok(if p.is_file() {
            serde_json::from_str(&std::fs::read_to_string(&p)?)?
        } else {
            Value::Null
        })
    };
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let config = json!({
        "tokenizer": tokenizer,
        "config": read("config.json")?,
        "preprocessor": read("preprocessor_config.json")?,
        "eos": crate::runner::read_eos(dir),
        "hidden": cfg.hidden_size,
        "context": rows - 1,
        "buckets": BUCKETS,
        "states": states.iter().map(|(n, d)| json!({"name": n, "dims": d})).collect::<Vec<_>>(),
    });
    program::entry(
        &mut pkg,
        "chat",
        module,
        &["run"],
        "image-text-to-text",
        config,
    );
    pkg.metadata("context_length", json!(rows - 1));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("minicpm-v", "image-text-to-text", export_rlxp)
            .claims(&["minicpmv4_6", "minicpmv4_7"]),
    );
}
