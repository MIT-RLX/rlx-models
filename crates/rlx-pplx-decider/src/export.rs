// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a pplx-decider GGUF as an `.rlxp` package rlx runs alone: a graph
//! per sequence bucket ([`crate::seq_bucket`]) doing what
//! [`crate::PplxDecider::marker_hidden`] and the readout do — the token
//! embeddings and RoPE rows gathered, the Qwen3.5 trunk with its
//! full-attention layers bidirectional (its Q8_0 projections read from the
//! GGUF as stored), the final-norm hidden state at each marker, the
//! readout's 255 rows applied — symbolic in the markers; and a program
//! answering `run` with the protocol around it (`protocol.rs`, built from
//! this crate's own file): the prompt rendered, tokenized by the package's
//! tokenizer, split into question chunks past the longest bucket, and each
//! answer's distribution.
//!
//! With the mmproj GGUF beside the backbone, the package holds the vision
//! tower too — one graph symbolic in the image's merged grid, the position
//! table resampled to it in-graph — and the program preprocesses each
//! image as the runner does (`vision.rs`), places its rows over the image
//! tokens and gives them their M-RoPE positions.
//!
//! `Model.run([], {request: "<the request JSON, as text>"})` returns
//! `decision`, the reference's `{"answers": …, "usage": …}` JSON as UTF-8
//! bytes. The request goes as text so its questions keep their order; its
//! `images` are `data:image/…;base64,` URLs (a package reads no paths).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::export::{Checkpoint, Package, Referencer, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Graph, Shape};
use rlx_qwen35::{ATTN_KEY_MASK, Qwen35Config, Qwen35Flow, Qwen35Weights};
use serde_json::json;

use crate::model::{DeciderMeta, gguf_tokenizer, seq_bucket};
use crate::protocol::{MARKER, MAX_OPTIONS};

/// Markers (questions) in one sequence.
const MARKERS: u32 = 0;
/// Image rows a sequence places over its image tokens (one, unread, when
/// it has none).
const IMAGE_ROWS: u32 = 1;
/// M-RoPE sections: time, height, width and the fourth.
const SECTIONS: usize = 4;
/// The longest bucket exported when the options list none.
const DEFAULT_LONGEST: usize = 4096;

type Params = HashMap<String, Vec<f32>>;
type Packed = rlx_qwen35::PackedParams;

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

/// The trunk at `seq`: `hidden` `[1, seq, h]`, `input_ids`, `rope_cos` /
/// `rope_sin` `[seq, half]` and the key mask in, the final-norm hidden
/// state out — the decider's own build.
fn trunk(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    seq: usize,
) -> Result<(Graph, Params, Packed)> {
    let (built, packed) = Qwen35Flow::new(cfg)
        .weights(weights)
        .batch(1)
        .seq(seq)
        .runtime_mrope()
        .export_normed_hidden()
        .bidirectional_attention(true)
        .build_with_weights(weights)
        .with_context(|| format!("building the decider graph at seq {seq}"))?;
    let (g, p) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((g, p, packed))
}

/// The marker rows' readout at `seq` tokens and `m` markers: `normed`
/// `[1, seq, h]` and `markers` i32 `[m]` → `logits` `[m, 255]`.
fn readout_graph(h: usize, seq: usize, m: usize) -> Result<Graph> {
    let f = DType::F32;
    let mut hir = HirModule::new("decide_out");
    let mut g = HirMut::new(&mut hir);
    let x = g.input("normed", Shape::new(&[1, seq, h], f));
    let markers = g.input("markers", Shape::new(&[m], DType::I32));
    let readout = g.param("readout", Shape::new(&[h, MAX_OPTIONS], f));
    let x = g.reshape_(x, vec![seq as i64, h as i64]);
    let rows = g.gather_(x, markers, 0);
    let logits = g.mm(rows, readout);
    hir.set_outputs(vec![logits]);
    from_hir(hir)
}

/// [`readout_graph`] symbolic in [`MARKERS`], checked at another count.
fn symbolic_readout(h: usize, seq: usize) -> Result<Graph> {
    let sizes = [3usize, 5, 11];
    let built: Vec<Graph> = sizes
        .iter()
        .map(|&m| readout_graph(h, seq, m))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().collect();
    let g = rlx_ir::dynamic::generalize_many(&refs, &[(MARKERS, sizes.to_vec())], None)
        .map_err(|e| anyhow!("the readout is not symbolic in its markers: {e}"))?;
    let want = readout_graph(h, seq, 7)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(MARKERS, 7);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{want}") {
        bail!("the readout generalized over three marker counts is not the one built at a fourth");
    }
    Ok(g)
}

/// The trunk's inputs at `seq` tokens and `m` image rows: each token's row
/// — `rows` i32 `[seq]` into the token embeddings of `ids` i32 `[seq]`
/// followed by `image_rows` `[m, h]` — and its M-RoPE row, each section's
/// columns gathered at that section's position (`positions` i32 `[4, seq]`).
fn inputs_graph(
    cfg: &Qwen35Config,
    vocab: usize,
    seq: usize,
    m: usize,
    rope_rows: usize,
) -> Result<Graph> {
    let h = cfg.hidden_size;
    let half = cfg.rope_dim_count / 2;
    let f = DType::F32;
    let mut hir = HirModule::new("decide_in");
    let mut g = HirMut::new(&mut hir);
    let ids = g.input("ids", Shape::new(&[seq], DType::I32));
    let image = g.input("image_rows", Shape::new(&[m, h], f));
    let rows = g.input("rows", Shape::new(&[seq], DType::I32));
    let pos = g.input("positions", Shape::new(&[SECTIONS, seq], DType::I32));
    let table = g.param("token_embd.weight", Shape::new(&[vocab, h], f));
    let cos_t = g.param("rope_cos_rows", Shape::new(&[rope_rows, half], f));
    let sin_t = g.param("rope_sin_rows", Shape::new(&[rope_rows, half], f));
    let sections = g.param("mrope_sections", Shape::new(&[SECTIONS, half], f));
    let text = g.gather_(table, ids, 0);
    let all = g.concat_(vec![text, image], 0);
    let hidden = g.gather_(all, rows, 0);
    let hidden = g.reshape_(hidden, vec![1, seq as i64, h as i64]);
    let (mut cos, mut sin) = (None, None);
    for k in 0..SECTIONS {
        let p = g.narrow_(pos, 0, k, 1);
        let p = g.reshape_(p, vec![seq as i64]);
        let mask = g.narrow_(sections, 0, k, 1);
        let c = g.gather_(cos_t, p, 0);
        let c = g.mul(c, mask);
        let s = g.gather_(sin_t, p, 0);
        let s = g.mul(s, mask);
        cos = Some(cos.map_or(c, |a| g.add(a, c)));
        sin = Some(sin.map_or(s, |a| g.add(a, s)));
    }
    hir.set_outputs(vec![hidden, cos.expect("sections"), sin.expect("sections")]);
    from_hir(hir)
}

/// [`inputs_graph`] symbolic in [`IMAGE_ROWS`], checked at another count.
fn symbolic_inputs(
    cfg: &Qwen35Config,
    vocab: usize,
    seq: usize,
    rope_rows: usize,
) -> Result<Graph> {
    let sizes = [3usize, 5, 11];
    let built: Vec<Graph> = sizes
        .iter()
        .map(|&m| inputs_graph(cfg, vocab, seq, m, rope_rows))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().collect();
    // The rows table is `seq + m` long: an offset as long as the sequence.
    let g = rlx_ir::dynamic::generalize_many(&refs, &[(IMAGE_ROWS, sizes.to_vec())], Some(1 << 16))
        .map_err(|e| anyhow!("the inputs are not symbolic in the image rows: {e}"))?;
    let want = inputs_graph(cfg, vocab, seq, 7, rope_rows)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(IMAGE_ROWS, 7);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{want}") {
        bail!(
            "the inputs generalized over three image-row counts are not the ones built at a fourth"
        );
    }
    Ok(g)
}

/// `decide_{seq}`: `ids`, `rows` i32 `[seq]`, `positions` i32 `[4, seq]`,
/// `image_rows` `[image_rows, h]`, `input_ids` and the key mask f32
/// `[1, seq]`, `markers` i32 `[markers]` → `logits` `[markers, 255]`.
fn decide_graph(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    seq: usize,
    rope_rows: usize,
) -> Result<(Graph, Params, Packed)> {
    let h = cfg.hidden_size;
    let vocab = weights.embd_elems() / h;
    let (trunk, params, packed) = trunk(cfg, weights, seq)?;
    let mut out = Graph::new(format!("decide_{seq}"));

    let ins = splice(&mut out, &symbolic_inputs(cfg, vocab, seq, rope_rows)?, &[])?;
    let normed = splice(
        &mut out,
        &trunk,
        &[
            ("hidden", ins[0]),
            ("rope_cos", ins[1]),
            ("rope_sin", ins[2]),
        ],
    )?;
    let res = splice(
        &mut out,
        &symbolic_readout(h, seq)?,
        &[("normed", normed[0])],
    )?;
    out.node_mut(res[0]).name = Some("logits".into());
    out.set_outputs(res);
    Ok((out, params, packed))
}

/// The sequence buckets up to `longest`, as the runner pads to them.
fn buckets(longest: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut n = 1;
    loop {
        let b = seq_bucket(n);
        if b > longest {
            break;
        }
        out.push(b);
        n = b + 1;
    }
    out
}

/// Export the decider GGUF `gguf` as the package `out`. `opts.uppers`'s
/// largest is the longest sequence (default 4096): a longer request is
/// split into question chunks, as the runner does past its `max_length`.
pub fn export_rlxp(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let gguf = crate::model::resolve_gguf(gguf)?;
    let path = gguf.to_str().context("GGUF path is not UTF-8")?;
    let mut loader = GgufLoader::from_file(path)?;
    let meta = DeciderMeta::from_gguf(loader.file())?;
    let mut cfg = Qwen35Config::from_gguf(loader.file())?;
    cfg.mrope_interleaved = true;
    if cfg.nextn_predict_layers != 0 {
        bail!("unexpected MTP layers in a decider checkpoint");
    }
    let longest = opts
        .uppers_or(&[DEFAULT_LONGEST])
        .into_iter()
        .max()
        .unwrap_or(DEFAULT_LONGEST);
    let sizes = buckets(longest);
    let Some(&longest) = sizes.last() else {
        bail!("no sequence bucket fits {longest} tokens");
    };
    let h = cfg.hidden_size;
    let half = cfg.rope_dim_count / 2;
    let readout = crate::model::read_readout(&loader, &meta.token_ids, h)?;
    let weights = Qwen35Weights::from_loader_packed(&mut loader, &cfg)?;

    let mut graphs = Vec::new();
    let mut params = Params::new();
    let mut packed = Packed::new();
    for &seq in &sizes {
        let (g, p, k) = decide_graph(&cfg, &weights, seq, longest)?;
        graphs.push(g);
        if params.is_empty() {
            params = p;
            packed = k;
        }
    }

    // The readout `[h, 255]`, the RoPE rows of every position a bucket reaches.
    let mut t = vec![0f32; h * MAX_OPTIONS];
    for (o, row) in readout.chunks_exact(h).enumerate() {
        for (k, &v) in row.iter().enumerate() {
            t[k * MAX_OPTIONS + o] = v;
        }
    }
    params.insert("readout".into(), t);
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, longest, None, half);
    params.insert("rope_cos_rows".into(), cos);
    params.insert("rope_sin_rows".into(), sin);
    params.insert(
        "mrope_sections".into(),
        rlx_qwen35::mrope_section_masks(&cfg, half)?,
    );
    // The table the ids gather from, as the runtime reads it from the GGUF.
    let mut tables = GgufLoader::from_file(path)?;
    let (embd, _) = tables.take("token_embd.weight")?;
    params.insert("token_embd.weight".into(), embd);
    drop(tables);

    let packed_keys: HashSet<String> = packed.values().map(|(k, _, _)| k.clone()).collect();
    let lens: HashSet<usize> = params.values().map(Vec::len).collect();
    let mut refr = Referencer::open_gguf_only(&gguf, opts.embed_weights, &|name, n| {
        !packed_keys.contains(name) || lens.contains(&n)
    })?;
    refr.take(std::mem::take(&mut params))?;
    for (param, (key, _, _)) in &packed {
        refr.packed(param, key);
    }
    let ck = Checkpoint::open(&gguf)?;
    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        "pplx-decider",
        "text-classification",
        "rlx-pplx-decider",
    );
    pkg.graphs = graphs;
    pkg.dims = vec![
        ("markers".into(), MARKERS),
        ("image_rows".into(), IMAGE_ROWS),
    ];
    pkg.reference(refr, out)?;

    // The vision tower, from the mmproj GGUF beside the backbone; held (the
    // runner widens it to f32 too).
    let vision = match crate::model::sibling_mmproj(&gguf) {
        Some(mmproj) => {
            // The decider's own choice: the fused attention, measured faster.
            if rlx_ir::env::var_result("RLX_QWEN35_VISION_SDPA_DECOMP").is_err() {
                rlx_ir::env::set("RLX_QWEN35_VISION_SDPA_DECOMP", "0");
            }
            let mp = mmproj.to_str().context("mmproj path is not UTF-8")?;
            let mut loader = GgufLoader::from_file(mp)?;
            let vcfg = rlx_qwen35::MmProjConfig::from_gguf(loader.file())?;
            let vw = rlx_qwen35::MmProjWeights::from_loader(&vcfg, &mut loader)?;
            if vcfg.llm_hidden_size != h
                || vcfg.patch_size != crate::vision::PATCH
                || vcfg.n_merge != crate::vision::MERGE
            {
                bail!("{} does not project onto this backbone", mmproj.display());
            }
            let side = (vw.position_embd.len() / vcfg.n_embd).isqrt();
            let (g, params) = rlx_qwen35::export_vl::symbolic_vision(&vcfg, &vw)?;
            pkg.function(g);
            let dims = pkg.param_dims();
            let mut names: Vec<&String> = params.keys().collect();
            names.sort();
            for n in names {
                let d = dims
                    .get(n)
                    .cloned()
                    .unwrap_or_else(|| vec![params[n].len()]);
                pkg.hold_f32(n.clone(), d, &params[n]);
            }
            pkg.dims
                .push(("grid_h".into(), rlx_qwen35::export_vl::GRID_H));
            pkg.dims
                .push(("grid_w".into(), rlx_qwen35::export_vl::GRID_W));
            json!({
                "side": side,
                "head_dim": vcfg.n_embd / vcfg.n_head,
                "max_pixels": meta.max_pixels,
                "min_pixels": crate::protocol::MIN_PIXELS,
            })
        }
        None => serde_json::Value::Null,
    };

    let tokenizer = gguf_tokenizer(&gguf)?;
    let marker_id = tokenizer
        .token_to_id(MARKER)
        .ok_or_else(|| anyhow!("the tokenizer has no {MARKER}"))?;
    let json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow!("serialize tokenizer: {e}"))?;
    let tokenizer_id = pkg.sidecar("tokenizer.json", "application/json", json.into_bytes());

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let config = json!({
        "tokenizer": tokenizer_id,
        "marker_id": marker_id,
        "codes": meta.codes,
        "escape_tokens": meta.escape_tokens,
        "temperature": meta.temperature,
        "key_mask": ATTN_KEY_MASK,
        "image_pad_id": tokenizer.token_to_id("<|image_pad|>"),
        "hidden": h,
        "vision": vision,
        "buckets": sizes,
        "functions": sizes.iter().map(|s| format!("decide_{s}")).collect::<Vec<_>>(),
    });
    program::entry(
        &mut pkg,
        "decide",
        module,
        &["run"],
        "text-classification",
        config,
    );
    pkg.metadata("protocol_version", json!(crate::protocol::PROTOCOL_VERSION));
    pkg.metadata("max_length", json!(longest));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family pplx-decider`): its GGUF says `qwen35`, which
/// the Qwen3.5 exporter claims.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "pplx-decider",
        "text-classification",
        export_rlxp,
    ));
}
