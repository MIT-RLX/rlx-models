// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a kev checkpoint (its run directory and the Qwen3.5 base it was
//! trained on) as an `.rlxp` package rlx runs alone: the backbone with the
//! LoRA folded in as [`crate::DecisionModel`] runs it — one causal row per
//! question, the final-norm hidden state gathered at each `<decide>` and
//! `</opt>` — as one graph symbolic in the rows and their length, and a
//! program answering `run` with the protocol around it, built from this
//! crate's own `api.rs`, `encode.rs`, `json.rs` and `pointer.rs`: the
//! request read in order, encoded, its rows shaped as the runner shapes
//! them, the pointer head over the gathered states, each answer.
//!
//! `Model.run([], {request: "<the /v1/systemone body, as text>"})` returns
//! `decision` (the response JSON as UTF-8 bytes) and `probabilities` (every
//! question's distribution, concatenated). The request goes as text so its
//! questions and options keep their order.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::export::{Checkpoint, Package, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Graph, Shape};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use serde_json::json;

use crate::encode::{HfTokenizer, SPECIAL, TokenizerLike};
use crate::lora::LoraAdapter;
use crate::model::{INFER_MAX_BRANCH, INFER_MAX_STATE, SHAPE_BUCKET, ShapePolicy};

/// Rows (questions, padded to a power of two).
const ROWS: u32 = 0;
/// Their length (a multiple of [`SHAPE_BUCKET`]).
const SEQ: u32 = 1;
/// The positions read from them.
const READS: u32 = 2;

type Params = HashMap<String, Vec<f32>>;

fn from_hir(hir: HirModule) -> Result<Graph> {
    Graph::from_hir(hir).map_err(|e| anyhow!("{e}"))
}

fn splice(
    into: &mut Graph,
    src: &Graph,
    bind: &[(&str, rlx_ir::NodeId)],
) -> Result<Vec<rlx_ir::NodeId>> {
    let s = rlx_ir::compose::Splice::new().bind_all(bind.iter().map(|(n, id)| (*n, *id)));
    Ok(into
        .splice(src, &s)
        .map_err(|e| anyhow!("joining {}: {e:?}", src.name))?
        .outputs)
}

/// `rows` × `seq` tokens (`input_ids` f32) → the final-norm hidden state at
/// `reads` i32 `[n]` flat positions: `states` `[n, hidden]`. The RoPE rows
/// are gathered from held tables (`rope_rows` long) at `positions`.
fn rows_graph(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    rows: usize,
    seq: usize,
    n: usize,
    rope_rows: usize,
) -> Result<(Graph, Params)> {
    let h = cfg.hidden_size;
    let half = cfg.rope_dim_count / 2;
    let f = DType::F32;
    let (hir, params, packed) = rlx_qwen35::build_qwen35_prefill_flow_ext(
        cfg, weights, rows, seq, false, false, false, true, false, true,
    )
    .with_context(|| format!("building the trunk at {rows}×{seq}"))?;
    if !packed.is_empty() {
        bail!("the merged weights came back packed");
    }
    let trunk = from_hir(hir)?;
    let mut out = Graph::new("rows");

    let mut hir = HirModule::new("rope");
    let mut g = HirMut::new(&mut hir);
    let pos = g.input("positions", Shape::new(&[seq], DType::I32));
    let cos_t = g.param("rope_cos_rows", Shape::new(&[rope_rows, half], f));
    let sin_t = g.param("rope_sin_rows", Shape::new(&[rope_rows, half], f));
    let cos = g.gather_(cos_t, pos, 0);
    let sin = g.gather_(sin_t, pos, 0);
    hir.set_outputs(vec![cos, sin]);
    let rope = splice(&mut out, &from_hir(hir)?, &[])?;
    let hidden = splice(
        &mut out,
        &trunk,
        &[("rope_cos", rope[0]), ("rope_sin", rope[1])],
    )?;

    let mut hir = HirModule::new("reads");
    let mut g = HirMut::new(&mut hir);
    let x = g.input("hidden", Shape::new(&[rows, seq, h], f));
    let reads = g.input("reads", Shape::new(&[n], DType::I32));
    let x = g.reshape_(x, vec![(rows * seq) as i64, h as i64]);
    let states = g.gather_(x, reads, 0);
    hir.set_outputs(vec![states]);
    let res = splice(&mut out, &from_hir(hir)?, &[("hidden", hidden[0])])?;
    out.node_mut(res[0]).name = Some("states".into());
    out.set_outputs(res);
    Ok((out, params))
}

/// [`rows_graph`] symbolic in [`ROWS`], [`SEQ`] and [`READS`]: built at
/// sizes no other dim has, generalized, checked against another build.
fn symbolic_rows(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    rope_rows: usize,
) -> Result<(Graph, Params)> {
    let b = SHAPE_BUCKET;
    let sizes = [
        (3usize, 5 * b, 7usize),
        (5, 7 * b, 11),
        (7, 3 * b, 13),
        (2, 9 * b, 17),
    ];
    let mut first = None;
    let mut graphs = Vec::new();
    for &(r, s, n) in &sizes {
        let (g, p) = rows_graph(cfg, weights, r, s, n, rope_rows)?;
        graphs.push(g);
        if first.is_none() {
            first = Some(p);
        }
    }
    let refs: Vec<&Graph> = graphs.iter().collect();
    let syms = [
        (ROWS, sizes.iter().map(|s| s.0).collect::<Vec<_>>()),
        (SEQ, sizes.iter().map(|s| s.1).collect()),
        (READS, sizes.iter().map(|s| s.2).collect()),
    ];
    // The convolutions pad the length by their kernel less one (`seq + 3`):
    // past the default offset.
    let mut g = rlx_ir::dynamic::generalize_many(&refs, &syms, Some(16))
        .map_err(|e| anyhow!("the trunk is not symbolic in its rows and length: {e}"))?;
    // Checked at another shape and at the smallest the runner uses.
    for (r, s, n) in [(4, 6 * b, 19), (1, b, 3)] {
        let (want, _) = rows_graph(cfg, weights, r, s, n, rope_rows)?;
        let mut bind = rlx_ir::DimBinding::new();
        bind.set(ROWS, r);
        bind.set(SEQ, s);
        bind.set(READS, n);
        let (got, want) = (
            format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
            format!("{want}"),
        );
        if got != want {
            let (a, b) = got
                .lines()
                .zip(want.lines())
                .find(|(a, b)| a != b)
                .unwrap_or_default();
            bail!(
                "the trunk generalized over four shapes is not the one built at {r}×{s}: `{a}` \
                 where it builds `{b}`"
            );
        }
    }
    // Params a build sized to its rows (the convolutions' zero pads): one
    // slice each, expanded.
    let mut params = first.expect("four builds");
    let g_at = graphs.swap_remove(0);
    drop(graphs);
    g = rlx_core::export::graph::broadcast_params(&g, &g_at, &mut params, &[])?;
    drop(g_at);
    g.name = "rows".into();
    Ok((g, params))
}

/// Export the kev run directory `run_dir` (`head.pt`,
/// `adapter_model.safetensors`, `tokenizer.json`) over the base
/// `base_dir` (a `Qwen/Qwen3.5-*-Base` snapshot) as the package `out`.
pub fn export_with_base(
    run_dir: &Path,
    base_dir: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let ckpt = crate::checkpoint::Checkpoint::open(run_dir)?;
    let head = ckpt.pointer_head()?;
    let adapter = LoraAdapter::open(run_dir)?;
    let merged = crate::weights::load_merged(base_dir, adapter)?;
    let (cfg, weights) = (merged.cfg, merged.weights);
    let ShapePolicy::Bucketed { max_rows, max_seq } = ShapePolicy::default() else {
        bail!("the default shape policy is bucketed");
    };

    let (graph, mut params) = symbolic_rows(&cfg, &weights, max_seq)?;
    drop(weights);
    let half = cfg.rope_dim_count / 2;
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, max_seq, None, half);
    params.insert("rope_cos_rows".into(), cos);
    params.insert("rope_sin_rows".into(), sin);

    // The base's tensors where a param equals one; the folded projections held.
    let base = Checkpoint::open(base_dir)?;
    let mut refr = base.referencer(opts.embed_weights)?;
    refr.take(params)?;
    let mut pkg = Package::new(
        "kev",
        &opts.version,
        "kev",
        "text-classification",
        "rlx-kev",
    );
    pkg.weights_as_stored();
    pkg.function(graph);
    pkg.dims = vec![
        ("rows".into(), ROWS),
        ("seq".into(), SEQ),
        ("reads".into(), READS),
    ];
    pkg.reference(refr, out)?;

    // The five Qwen specials kev's encoder reads, by id.
    let hf = HfTokenizer::from_file(&ckpt.tokenizer_path())?;
    let specials: serde_json::Map<String, serde_json::Value> = SPECIAL
        .iter()
        .filter_map(|n| Some((n.to_string(), json!(hf.token_to_id(n)?))))
        .collect();
    let tok = std::fs::read(ckpt.tokenizer_path()).context("reading the run's tokenizer.json")?;
    let tokenizer = pkg.sidecar("tokenizer.json", "application/json", tok);
    let mut bytes = Vec::new();
    for v in [&head.q_weight, &head.q_bias, &head.k_weight, &head.k_bias] {
        bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    }
    let head_file = pkg.sidecar("pointer_head.f32", "application/octet-stream", bytes);
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let config = json!({
        "tokenizer": tokenizer,
        "specials": specials,
        "head": head_file,
        "hidden": head.hidden,
        "head_dim": head.head_dim,
        "temperature": head.temperature,
        "max_state": INFER_MAX_STATE,
        "max_branch": INFER_MAX_BRANCH,
        "shape_bucket": SHAPE_BUCKET,
        "max_rows": max_rows,
        "max_seq": max_seq,
    });
    program::entry(
        &mut pkg,
        "decide",
        module,
        &["run"],
        "text-classification",
        config,
    );
    pkg.metadata("base", json!(base_dir.file_name().and_then(|n| n.to_str())));
    pkg.write(out, opts)
}

/// [`export_with_base`] with the base found beside the run: `--weights`
/// names the run directory, its base `base/` inside it or the
/// `base_model_name_or_path` of its `adapter_config.json` next to it.
pub fn export_rlxp(run_dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("adapter_config.json"))
            .context("reading adapter_config.json")?,
    )?;
    let named = cfg["base_model_name_or_path"].as_str().unwrap_or_default();
    let leaf = Path::new(named).file_name().unwrap_or_default();
    let candidates = [
        run_dir.join("base"),
        run_dir.parent().map(|p| p.join(leaf)).unwrap_or_default(),
        Path::new(named).to_path_buf(),
    ];
    let Some(base) = candidates.iter().find(|p| p.join("config.json").is_file()) else {
        bail!(
            "no base model for {}: put {named} beside it (or `base/` inside it)",
            run_dir.display()
        );
    };
    export_with_base(run_dir, base, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family kev`): a kev run is a LoRA adapter and a head.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("kev", "text-classification", export_rlxp));
}
