// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a d1 (LFM2.5-VL) checkpoint directory as an `.rlxp` package rlx
//! runs alone, for text states: a graph per length bucket (as
//! `Lfm2GgufRunner::hidden_from_embeds` pads, a power of two from 64)
//! doing what [`crate::D1`] reads per question — the token embeddings, one
//! causal pass of the LFM2 text model, the last position's final-normed
//! hidden state against the tied head's rows of the answer tokens, symbolic
//! in those tokens — and a program answering `run` with d1's protocol
//! around it (`prompt.rs`, built from this crate's own file): the prompt
//! rendered and tokenized by the package's tokenizer, one pass per
//! question, each answer's distribution.
//!
//! The package holds the vision half too: the SigLIP2 NaFlex tower at the
//! processor's patch maximum (padding masked) and the projector, the
//! program processing each picture as the runner does (`processor.rs`,
//! the position grid resampled by SigLIP2's own `pos_grid.rs`) and placing
//! its features over the image tokens.
//!
//! `Model.run([], {request: "{\"state\": …, \"questions\": {…}, \"images\": […]}"})`
//! returns `decision`, `system_one`'s JSON as UTF-8 bytes. The request goes
//! as text so its questions keep their order; `images` are `data:` URLs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::export::{Checkpoint, Package, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::standard_decoder::{DenseStore, Lfm2Spec, build_lfm2_prefill_stored};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Graph, Shape};
use serde_json::json;

/// The answer tokens a pass is scored on.
const OPTIONS: u32 = 0;
/// Image rows a pass places over its image tokens (one, unread, when it
/// has none).
const IMAGE_ROWS: u32 = 1;
/// The longest bucket exported when the options list none.
const DEFAULT_LONGEST: usize = 4096;
const TABLE: &str = "model.embed_tokens.weight";

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

/// The scoring tail at `seq` tokens and `k` answer tokens: `hidden`
/// `[seq, h]`, `last` i32 `[1]` and `options` i32 `[k]` → `scores` `[k]`.
fn score_graph(spec: &Lfm2Spec, seq: usize, k: usize) -> Result<Graph> {
    let (h, v) = (spec.hidden_size, spec.vocab_size);
    let f = DType::F32;
    let mut hir = HirModule::new("score");
    let mut g = HirMut::new(&mut hir);
    let hidden = g.input("hidden", Shape::new(&[seq, h], f));
    let last = g.input("last", Shape::new(&[1], DType::I32));
    let options = g.input("options", Shape::new(&[k], DType::I32));
    let table = g.param(TABLE, Shape::new(&[v, h], f));
    let row = g.gather_(hidden, last, 0);
    let rows = g.gather_(table, options, 0);
    let rows_t = g.transpose_(rows, vec![1, 0]);
    let scores = g.mm(row, rows_t);
    let scores = g.reshape_(scores, vec![k as i64]);
    hir.set_outputs(vec![scores]);
    from_hir(hir)
}

/// [`score_graph`] symbolic in [`OPTIONS`], checked at another count.
fn symbolic_score(spec: &Lfm2Spec, seq: usize) -> Result<Graph> {
    let sizes = [3usize, 5, 11];
    let built: Vec<Graph> = sizes
        .iter()
        .map(|&k| score_graph(spec, seq, k))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().collect();
    let g = rlx_ir::dynamic::generalize_many(&refs, &[(OPTIONS, sizes.to_vec())], None)
        .map_err(|e| anyhow!("the scoring is not symbolic in its answer tokens: {e}"))?;
    let want = score_graph(spec, seq, 7)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(OPTIONS, 7);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{want}") {
        bail!("the scoring generalized over three counts is not the one built at a fourth");
    }
    Ok(g)
}

/// Each token's input row at `seq` tokens and `m` image rows: `rows` i32
/// `[seq]` into the embeddings of `ids` i32 `[seq]` followed by
/// `image_rows` `[m, h]`.
fn embed_graph(h: usize, v: usize, seq: usize, m: usize) -> Result<Graph> {
    let mut hir = HirModule::new("embed");
    let mut g = HirMut::new(&mut hir);
    let ids = g.input("ids", Shape::new(&[seq], DType::I32));
    let image = g.input("image_rows", Shape::new(&[m, h], DType::F32));
    let rows = g.input("rows", Shape::new(&[seq], DType::I32));
    let table = g.param(TABLE, Shape::new(&[v, h], DType::F32));
    let x = g.gather_(table, ids, 0);
    let x = g.concat_(vec![x, image], 0);
    let x = g.gather_(x, rows, 0);
    let x = g.reshape_(x, vec![1, seq as i64, h as i64]);
    hir.set_outputs(vec![x]);
    from_hir(hir)
}

/// [`embed_graph`] symbolic in [`IMAGE_ROWS`], checked at another count.
fn symbolic_embed(h: usize, v: usize, seq: usize) -> Result<Graph> {
    let sizes = [3usize, 5, 11];
    let built: Vec<Graph> = sizes
        .iter()
        .map(|&m| embed_graph(h, v, seq, m))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().collect();
    // The rows table is `seq + m` long: an offset as long as the pass.
    let g = rlx_ir::dynamic::generalize_many(&refs, &[(IMAGE_ROWS, sizes.to_vec())], Some(1 << 16))
        .map_err(|e| anyhow!("the embedding is not symbolic in the image rows: {e}"))?;
    let want = embed_graph(h, v, seq, 7)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(IMAGE_ROWS, 7);
    if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{want}") {
        bail!(
            "the embedding generalized over three image-row counts is not the one built at a fourth"
        );
    }
    Ok(g)
}

/// The vision half as the runner compiles it: the tower over the
/// processor's patch maximum (`pixel_values`, `pos_embed`, `key_mask`) and
/// the projector over its unshuffled rows (`image_features`), with their
/// params and what the program needs to feed them.
fn vision_graphs(dir: &Path) -> Result<(Graph, Graph, Params, serde_json::Value, Vec<f32>)> {
    use crate::vision::{PROJECTOR, ProcessorConfig, TOWER, VisionConfig, build_projector};
    let raw = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
    let config: serde_json::Value = serde_json::from_str(&raw)?;
    let cfg = VisionConfig::from_config(&config)?;
    let proc = ProcessorConfig::from_dir(dir)?;
    let names = rlx_core::SafetensorsSource::File(dir.to_path_buf())
        .loader()?
        .remaining_keys();
    let want: std::collections::HashSet<String> = names
        .into_iter()
        .filter(|n| n.starts_with(TOWER) || n.starts_with(PROJECTOR))
        .collect();
    let mut weights = rlx_core::weight_map::WeightMap::from_safetensors_dir_selected(dir, &want)?;
    let (grid, _) = weights.take(&format!("{TOWER}.embeddings.position_embedding.weight"))?;
    let seq = proc.max_num_patches();
    let spec = rlx_siglip2::flow::NaflexEncoderSpec {
        prefix: TOWER.into(),
        width: cfg.hidden_size,
        heads: cfg.num_heads,
        layers: cfg.num_layers,
        patch_dim: 3 * cfg.patch_size * cfg.patch_size,
        seq,
        eps: cfg.layer_norm_eps,
        weight_dtype: DType::F32,
    };
    let (mut tower, mut params) = rlx_core::flow_util::graph_from_built(
        rlx_siglip2::flow::build_naflex_encoder(&spec, &mut weights)?,
    )?;
    tower.name = "tower".into();
    let ds = proc.downsample_factor;
    let rows = seq / (ds * ds);
    let (mut projector, p) = rlx_core::flow_util::graph_from_built(build_projector(
        &cfg,
        ds,
        rows,
        DType::F32,
        &mut weights,
    )?)?;
    projector.name = "projector".into();
    params.extend(p);
    let settings = json!({
        "hidden": cfg.hidden_size,
        "patch": cfg.patch_size,
        "pos_side": cfg.pos_side,
        "text_hidden": cfg.text_hidden,
        "seq": seq,
        "proj_rows": rows,
        "processor": {
            "downsample_factor": proc.downsample_factor,
            "do_image_splitting": proc.do_image_splitting,
            "min_tiles": proc.min_tiles,
            "max_tiles": proc.max_tiles,
            "use_thumbnail": proc.use_thumbnail,
            "min_image_tokens": proc.min_image_tokens,
            "max_image_tokens": proc.max_image_tokens,
            "encoder_patch_size": proc.encoder_patch_size,
            "tile_size": proc.tile_size,
            "max_pixels_tolerance": proc.max_pixels_tolerance,
            "use_image_special_tokens": proc.use_image_special_tokens,
        },
    });
    Ok((tower, projector, params, settings, grid))
}

/// `pass_{seq}`: `ids`, `rows` i32 `[seq]`, `image_rows` `[image_rows, h]`,
/// `last` i32 `[1]`, `options` i32 `[options]` → `scores` `[options]`.
fn pass_graph(dir: &Path, seq: usize) -> Result<(Graph, Params)> {
    let (spec, mut loader) = rlx_lfm::lfm2_gguf::hf_text_model(dir)?;
    let mut packed = HashMap::new();
    let (trunk, params) = build_lfm2_prefill_stored(
        &spec,
        loader.as_mut(),
        seq,
        &mut packed,
        true,
        &mut DenseStore::f32(),
    )?;
    if !packed.is_empty() {
        bail!("this checkpoint's weights are quantized: export from the bf16 one");
    }
    let (h, v) = (spec.hidden_size, spec.vocab_size);
    let mut out = Graph::new(format!("pass_{seq}"));
    let embeds = splice(&mut out, &symbolic_embed(h, v, seq)?, &[])?;
    let hidden = splice(&mut out, &trunk, &[("inputs_embeds", embeds[0])])?;
    let scores = splice(
        &mut out,
        &symbolic_score(&spec, seq)?,
        &[("hidden", hidden[0])],
    )?;
    out.node_mut(scores[0]).name = Some("scores".into());
    out.set_outputs(scores);
    Ok((out, params))
}

/// Export the d1 checkpoint directory `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`) as the
/// package `out`. `opts.uppers`'s largest is the longest pass (default 4096).
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let longest = opts
        .uppers_or(&[DEFAULT_LONGEST])
        .into_iter()
        .max()
        .unwrap_or(DEFAULT_LONGEST);
    let mut sizes = Vec::new();
    let mut seq = 64;
    while seq <= longest {
        sizes.push(seq);
        seq *= 2;
    }
    if sizes.is_empty() {
        bail!("no length bucket fits {longest} tokens");
    }

    let mut refr = ck.referencer(opts.embed_weights)?;
    let mut graphs = Vec::new();
    for (i, &seq) in sizes.iter().enumerate() {
        let (g, mut params) = pass_graph(dir, seq)?;
        if i == 0 {
            let (_, mut loader) = rlx_lfm::lfm2_gguf::hf_text_model(dir)?;
            let (table, _) = loader.take(TABLE)?;
            params.insert(TABLE.into(), table);
        } else {
            // The RoPE rows are per length; the rest the first build gave.
            params.retain(|k, _| k.starts_with("lfm.rope."));
        }
        let rope: Params = params
            .iter()
            .filter(|(k, _)| k.starts_with("lfm.rope."))
            .map(|(k, v)| (format!("{k}.{seq}"), v.clone()))
            .collect();
        params.retain(|k, _| !k.starts_with("lfm.rope."));
        params.extend(rope);
        refr.take(params)?;
        // The per-length RoPE tables get the length in their name.
        let mut g = g;
        for n in g.nodes_mut() {
            if let rlx_ir::Op::Param { name } = &mut n.op
                && name.starts_with("lfm.rope.")
            {
                *name = format!("{name}.{seq}");
            }
        }
        graphs.push(g);
    }

    // The vision half: its tensors referenced from the checkpoint as well.
    let (tower, projector, vparams, vision, grid) = vision_graphs(dir)?;
    refr.take(vparams)?;
    graphs.push(tower);
    graphs.push(projector);

    let mut pkg = Package::for_checkpoint(&ck, opts, "d1", "text-classification", "rlx-d1");
    pkg.weights_as_stored();
    pkg.graphs = graphs;
    pkg.dims = vec![
        ("options".into(), OPTIONS),
        ("image_rows".into(), IMAGE_ROWS),
    ];
    pkg.reference(refr, out)?;
    let grid_bytes: Vec<u8> = grid.iter().flat_map(|x| x.to_le_bytes()).collect();
    let grid_file = pkg.sidecar("position_grid.f32", "application/octet-stream", grid_bytes);
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let bos = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|tc| tc.get("bos_token")?.as_str().map(String::from))
        .unwrap_or_default();

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("building the d1 program")?;
    let image_token = {
        let tok = crate::HfTokenizer::from_file(&dir.join("tokenizer.json"))?;
        let ids = crate::TextTokenizer::encode(&tok, crate::vision::IMAGE_TOKEN)?;
        anyhow::ensure!(ids.len() == 1, "`<image>` is not a single token");
        ids[0]
    };
    let config = json!({
        "tokenizer": tokenizer,
        "bos": bos,
        "buckets": sizes,
        "hidden": rlx_lfm::lfm2_gguf::hf_text_model(dir)?.0.hidden_size,
        "image_token_id": image_token,
        "vision": vision,
        "position_grid": grid_file,
    });
    program::entry(
        &mut pkg,
        "decide",
        module,
        &["run"],
        "text-classification",
        config,
    );
    pkg.metadata("max_length", json!(longest));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family d1`): its `config.json` says `lfm2_vl`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("d1", "text-classification", export_rlxp));
}
