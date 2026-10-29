// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! The graphs a Qwen3.5 vision-language package is made of, for the
//! exporters built on this trunk (Qwen3.5's own multimodal export, MiniCPM-V,
//! pplx-decider, Fara): the vision tower symbolic in an image's merged grid,
//! and a decode step with its state inside that takes a token or an image
//! row and its M-RoPE positions — what a package's program drives a prompt
//! and its reply through, token by token, the caches kept in the runtime.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Graph, Op, Shape};

/// The step's cache capacity, and the past rows it was built at.
pub const CAP: u32 = 0;
pub const PAST: u32 = 1;
/// An image's merged grid: rows and columns of `patch · merge` pixels.
pub const GRID_H: u32 = 2;
pub const GRID_W: u32 = 3;
/// M-RoPE sections a position has.
pub const SECTIONS: usize = 4;

/// Params by name.
pub type Params = HashMap<String, Vec<f32>>;

/// `hir` as a graph.
pub fn from_hir(hir: HirModule) -> Result<Graph> {
    Graph::from_hir(hir).map_err(|e| anyhow!("{e}"))
}

/// `src` copied into `into`, its leaves `bind` fed from nodes there: its
/// outputs.
pub fn splice(
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

/// The vision tower at a `gh × gw` merged grid, as the runner builds it,
/// its size-dependent tensors taken in: `spatial_merge_idx` and `attn_mask`
/// as inputs, the position embeddings resampled in-graph from the square
/// table (`ry` `[hp, side]` and `rx` `[wp, side]`, the bilinear weights)
/// and put in merge order.
pub fn vision_graph(
    cfg: &crate::MmProjConfig,
    w: &crate::MmProjWeights,
    gh: usize,
    gw: usize,
) -> Result<(Graph, Params)> {
    let (n, m, ps) = (cfg.n_embd, cfg.n_merge, cfg.patch_size);
    let (hp, wp) = (gh * m, gw * m);
    let n_pos = hp * wp;
    let side = (w.position_embd.len() / n).isqrt();
    if side * side * n != w.position_embd.len() {
        bail!("the vision position table is not square");
    }
    let built = crate::build_qwen35_vision_built(cfg, w, wp * ps, hp * ps)?;
    let (mut tower, mut params) = rlx_core::flow_util::graph_from_built(built)?;
    for node in tower.nodes_mut() {
        if let rlx_ir::Op::Param { name } = &node.op
            && matches!(
                name.as_str(),
                "spatial_merge_idx" | "position_embd" | "attn_mask"
            )
        {
            node.op = rlx_ir::Op::Input { name: name.clone() };
        }
    }
    for k in ["spatial_merge_idx", "position_embd", "attn_mask"] {
        params.remove(k);
    }
    let f = DType::F32;
    let mut hir = HirModule::new("vision_pos");
    let mut g = HirMut::new(&mut hir);
    let ry = g.input("ry", Shape::new(&[hp, side], f));
    let rx = g.input("rx", Shape::new(&[wp, side], f));
    let idx = g.input("spatial_merge_idx", Shape::new(&[n_pos], f));
    let table = g.param("v.position_table", Shape::new(&[side, side * n], f));
    let a = g.mm(ry, table);
    let a = g.reshape_(a, vec![hp as i64, side as i64, n as i64]);
    let a = g.transpose_(a, vec![0, 2, 1]);
    let a = g.reshape_(a, vec![(hp * n) as i64, side as i64]);
    let rx_t = g.transpose_(rx, vec![1, 0]);
    let a = g.mm(a, rx_t);
    let a = g.reshape_(a, vec![hp as i64, n as i64, wp as i64]);
    let a = g.transpose_(a, vec![0, 2, 1]);
    let a = g.reshape_(a, vec![n_pos as i64, n as i64]);
    let pos = g.gather_(a, idx, 0);
    hir.set_outputs(vec![pos]);
    let mut out = Graph::new("vision");
    let pos = splice(&mut out, &from_hir(hir)?, &[])?;
    let emb = splice(&mut out, &tower, &[("position_embd", pos[0])])?;
    out.node_mut(emb[0]).name = Some("embeddings".into());
    out.set_outputs(emb);
    params.insert("v.position_table".into(), w.position_embd.clone());
    Ok((out, params))
}

/// [`vision_graph`] symbolic in [`GRID_H`] and [`GRID_W`].
pub fn symbolic_vision(
    cfg: &crate::MmProjConfig,
    w: &crate::MmProjWeights,
) -> Result<(Graph, Params)> {
    let sizes = [(3usize, 5usize), (5, 7), (7, 4), (4, 9)];
    let mut params = None;
    let mut built = Vec::new();
    for &(gh, gw) in &sizes {
        let (g, p) = vision_graph(cfg, w, gh, gw)?;
        built.push(g);
        params.get_or_insert(p);
    }
    let refs: Vec<&Graph> = built.iter().collect();
    let syms = [
        (GRID_H, sizes.iter().map(|s| s.0).collect::<Vec<_>>()),
        (GRID_W, sizes.iter().map(|s| s.1).collect()),
    ];
    let mut g = rlx_ir::dynamic::generalize_many(&refs, &syms, Some(16))
        .map_err(|e| anyhow!("the vision tower is not symbolic in its grid: {e}"))?;
    drop(built);
    for (gh, gw) in [(6usize, 3usize), (2, 2)] {
        let (want, _) = vision_graph(cfg, w, gh, gw)?;
        let mut bind = rlx_ir::DimBinding::new();
        bind.set(GRID_H, gh);
        bind.set(GRID_W, gw);
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
                "the vision tower generalized over four grids is not the one built at {gh}×{gw}: `{a}` where it builds `{b}`"
            );
        }
    }
    g.name = "vision".into();
    // The tower's params share names with the backbone's (`blk.0.…`): its own
    // namespace.
    for node in g.nodes_mut() {
        if let rlx_ir::Op::Param { name } = &mut node.op {
            *name = format!("mmproj.{name}");
        }
    }
    let params = params
        .expect("four builds")
        .into_iter()
        .map(|(k, v)| (format!("mmproj.{k}"), v))
        .collect();
    Ok((g, params))
}

/// The trunk's decode step at `past` cached rows, as the runner builds it:
/// one row of input embeddings, RoPE rows fed, a per-key mask.
fn raw_step(dir: &Path, past: usize) -> Result<(Graph, Params)> {
    let (mut cfg, weights) = crate::hf::hf_text_model(dir)?;
    cfg.mrope_interleaved = true;
    raw_step_from(&cfg, weights, past)
}

/// [`raw_step`] over a trunk already loaded (a checkpoint that nests the
/// trunk under its own names: FireRedAudio's `backbone_llm.*`).
pub fn raw_step_from(
    cfg: &crate::Qwen35Config,
    weights: crate::Qwen35Weights,
    past: usize,
) -> Result<(Graph, Params)> {
    let opts = crate::Qwen35DecodeOpts {
        batch: 1,
        past_seq: past,
        seq: 1,
        verify_all: false,
        dynamic_past: false,
        use_custom_mask: true,
        bias_mask: false,
        force_host_embed: true,
        enable_mtp_head: false,
        fast_mtp: false,
        fast_greedy_lm_head: false,
        profile: None,
        residual_out: false,
        greedy_token: false,
        multi_row: false,
        mtp_kv: false,
        keys_mask: false,
    };
    let (built, packed) = crate::build_qwen35_decode_built(cfg, weights, &opts)?;
    if !packed.is_empty() {
        bail!("the trunk's weights came back packed");
    }
    rlx_core::flow_util::graph_from_built(built)
}

/// The step with its state inside, symbolic in its capacity, a token's
/// embedding (or an image row) and its M-RoPE rows computed in front:
/// `token` i32 `[1]`, `image_row` `[1, h]`, `use_image` `[1]` (1: the row),
/// `positions` i32 `[4]` (the token's position per section), `position`
/// `[1]` (its row in the caches) → `logits` and each state's next value
/// (`<state>.next`).
///
/// Its params go to `take` as soon as they are built (a referencer drops
/// the values of those the checkpoint holds), so the export holds at most
/// one build's weights: the build whose params are dropped comes first.
pub fn host_embed_step(
    dir: &Path,
    cfg: &crate::Qwen35Config,
    rows: usize,
    take: &mut dyn FnMut(Params) -> Result<()>,
) -> Result<(Graph, Vec<(String, Vec<i64>)>)> {
    host_embed_step_with(&|past| raw_step(dir, past), cfg, rows, take)
}

/// [`host_embed_step`] with the trunk's step at `past` cached rows built by
/// `raw` (e.g. [`raw_step_from`] over the caller's own loading).
pub fn host_embed_step_with(
    raw: &dyn Fn(usize) -> Result<(Graph, Params)>,
    cfg: &crate::Qwen35Config,
    rows: usize,
    take: &mut dyn FnMut(Params) -> Result<()>,
) -> Result<(Graph, Vec<(String, Vec<i64>)>)> {
    let (b, _) = raw(52)?;
    let (a, params) = raw(36)?;
    take(params)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(CAP, 37, 53), (PAST, 36, 52)])
        .map_err(|e| anyhow!("the step is not symbolic in its cache: {e}"))?;
    drop((a, b));
    let spec = rlx_ir::stateful::StatefulDecode {
        position: "position".into(),
        tables: vec![],
        mask: Some(("mask".into(), "decode.iota".into(), rows)),
        cache_symbol: CAP,
        tokens: 1,
        starts: None,
        rope_offset: None,
        windows: vec![],
    };
    let mut step = rlx_ir::stateful::stateful_decode(&g, &spec).map_err(anyhow::Error::msg)?;
    // Name the outputs: the logits, then each state's next value, the states
    // in the order the step declares them.
    let states: Vec<(String, Vec<i64>)> = step
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name }
                if name.starts_with("conv_state_")
                    || name.starts_with("ssm_state_")
                    || name.starts_with("past_k_")
                    || name.starts_with("past_v_") =>
            {
                let dims = n
                    .shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        rlx_ir::Dim::Static(x) => *x as i64,
                        rlx_ir::Dim::Dynamic(_) => -1,
                    })
                    .collect();
                Some((name.clone(), dims))
            }
            _ => None,
        })
        .collect();
    if states.len() + 1 != step.outputs.len() {
        bail!(
            "{} states but {} outputs after the logits",
            states.len(),
            step.outputs.len() - 1
        );
    }
    let outs = step.outputs.clone();
    step.node_mut(outs[0]).name = Some("logits".into());
    for (k, (name, _)) in states.iter().enumerate() {
        step.node_mut(outs[k + 1]).name = Some(format!("{name}.next"));
    }

    // In front: the embedding (or image row), and the M-RoPE rows.
    let h = cfg.hidden_size;
    let half = cfg.rope_dim_count / 2;
    let f = DType::F32;
    let mut hir = HirModule::new("step_in");
    let mut gb = HirMut::new(&mut hir);
    let token = gb.input("token", Shape::new(&[1], DType::I32));
    let row = gb.input("image_row", Shape::new(&[1, h], f));
    let use_image = gb.input("use_image", Shape::new(&[1], f));
    let pos = gb.input("positions", Shape::new(&[SECTIONS], DType::I32));
    let table = gb.param("embed_tokens", Shape::new(&[cfg.vocab_size, h], f));
    let cos_t = gb.param("rope_cos_rows", Shape::new(&[rows, half], f));
    let sin_t = gb.param("rope_sin_rows", Shape::new(&[rows, half], f));
    let sections = gb.param("mrope_sections", Shape::new(&[SECTIONS, half], f));
    let text = gb.gather_(table, token, 0);
    let one = gb.param("one", Shape::new(&[1], f));
    let keep = gb.sub(one, use_image);
    let text = gb.mul(text, keep);
    let image = gb.mul(row, use_image);
    let embeds = gb.add(text, image);
    let embeds = gb.reshape_(embeds, vec![1, 1, h as i64]);
    let (mut cos, mut sin) = (None, None);
    for k in 0..SECTIONS {
        let p = gb.narrow_(pos, 0, k, 1);
        let mask = gb.narrow_(sections, 0, k, 1);
        let c = gb.gather_(cos_t, p, 0);
        let c = gb.mul(c, mask);
        let s = gb.gather_(sin_t, p, 0);
        let s = gb.mul(s, mask);
        cos = Some(cos.map_or(c, |a| gb.add(a, c)));
        sin = Some(sin.map_or(s, |a| gb.add(a, s)));
    }
    hir.set_outputs(vec![embeds, cos.expect("sections"), sin.expect("sections")]);
    let mut out = Graph::new("step");
    let ins = splice(&mut out, &from_hir(hir)?, &[])?;
    let res = splice(
        &mut out,
        &step,
        &[
            ("inputs_embeds", ins[0]),
            ("rope_cos", ins[1]),
            ("rope_sin", ins[2]),
        ],
    )?;
    // The step's variables (the KV caches) follow it through the splice.
    out.set_outputs(res);
    Ok((out, states))
}

/// The params [`host_embed_step`] computes rather than reads: the text RoPE
/// rows and section masks its M-RoPE rows come from, the mask's positions,
/// and the constant one. The embedding table (`embed_tokens`) is the
/// caller's, as its checkpoint names it.
pub fn step_tables(cfg: &crate::Qwen35Config, rows: usize) -> Result<Params> {
    let half = cfg.rope_dim_count / 2;
    let mut p = Params::new();
    let (cos, sin) = crate::mrope_prefill_feeds(cfg, rows, None, half);
    p.insert("rope_cos_rows".into(), cos);
    p.insert("rope_sin_rows".into(), sin);
    p.insert(
        "mrope_sections".into(),
        crate::mrope_section_masks(cfg, half)?,
    );
    p.insert("one".into(), vec![1.0]);
    p.insert("decode.iota".into(), (0..rows).map(|i| i as f32).collect());
    Ok(p)
}

/// The vision tower of an HF Qwen3.5 checkpoint (`model.visual.*`), its
/// config and weights as the runner reads them.
pub fn hf_vision(dir: &Path) -> Result<(crate::MmProjConfig, crate::MmProjWeights)> {
    let vcfg = crate::MmProjConfig::from_hf_config_json(&dir.join("config.json"))?;
    let mut loader = rlx_core::SafetensorsMmapLoader::open(dir)?;
    let vw = crate::MmProjWeights::from_hf_visual(&vcfg, &mut loader)?;
    Ok((vcfg, vw))
}

/// What a program needs of the vision tower besides its graph.
pub fn vision_settings(cfg: &crate::MmProjConfig, w: &crate::MmProjWeights) -> serde_json::Value {
    serde_json::json!({
        "patch": cfg.patch_size,
        "merge": cfg.n_merge,
        "side": (w.position_embd.len() / cfg.n_embd).isqrt(),
        "head_dim": cfg.n_embd / cfg.n_head,
        "min_pixels": cfg.image_min_pixels,
        "max_pixels": cfg.image_max_pixels,
        "mean": cfg.image_mean,
        "std": cfg.image_std,
    })
}

/// The states [`host_embed_step`] lists, as a program's config takes them.
pub fn states_json(states: &[(String, Vec<i64>)]) -> serde_json::Value {
    serde_json::Value::Array(
        states
            .iter()
            .map(|(n, d)| serde_json::json!({"name": n, "dims": d}))
            .collect(),
    )
}

/// The longest context [`export_vl_rlxp`] exports when the options name none.
pub const DEFAULT_CONTEXT: usize = 4096;

/// Export an HF Qwen3.5 checkpoint with its vision tower (`model.visual.*`)
/// as an `.rlxp` package rlx runs alone: the tower symbolic in the image's
/// grid, the stateful decode step, and a program answering `run` with what
/// [`crate::Qwen35Runner::generate_multimodal_with_opts`] does — the image
/// preprocessed and encoded, the prompt assembled around it (this crate's
/// own `preprocess.rs` and `multimodal.rs`), the prompt and the reply run
/// through the step. `Model.run([image0, …], {prompt, max_tokens})` (the
/// prompt holding one `<__media__>` per picture) returns `text`.
pub fn export_vl_rlxp(
    dir: &Path,
    out: &Path,
    opts: &rlx_core::export::ExportOptions,
) -> Result<rlx_core::export::ExportReport> {
    use rlx_core::export::{Checkpoint, Package, program};
    let mut cfg = crate::Qwen35Config::from_hf_config_json(&dir.join("config.json"))?;
    cfg.mrope_interleaved = true;
    let rows = opts.max_context.unwrap_or(DEFAULT_CONTEXT) + 1;
    let ck = Checkpoint::open(dir)?;
    let mut refr = ck.referencer(opts.embed_weights)?;

    let (step, states) = host_embed_step(dir, &cfg, rows, &mut |p| refr.take(p))?;
    let mut extra = step_tables(&cfg, rows)?;
    let mut loader = ck.loader()?;
    let (table, _) = loader.take("model.language_model.embed_tokens.weight")?;
    drop(loader);
    extra.insert("embed_tokens".into(), table);
    refr.take(extra)?;

    let (vcfg, vw) = hf_vision(dir)?;
    let vision = vision_settings(&vcfg, &vw);
    let (tower, vparams) = symbolic_vision(&vcfg, &vw)?;
    drop(vw);
    refr.take(vparams)?;

    let mut pkg = Package::for_checkpoint(&ck, opts, "qwen35", "image-text-to-text", "rlx-qwen35");
    pkg.weights_as_stored();
    pkg.graphs = vec![step, tower];
    pkg.dims = vec![
        ("cap".into(), CAP),
        ("past".into(), PAST),
        ("grid_h".into(), GRID_H),
        ("grid_w".into(), GRID_W),
    ];
    pkg.reference(refr, out)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let config = serde_json::json!({
        "tokenizer": tokenizer,
        "stop_ids": ck.stop_ids(),
        "hidden": cfg.hidden_size,
        "context": rows - 1,
        "states": states_json(&states),
        "vision": vision,
    });
    program::entry(
        &mut pkg,
        "chat",
        module,
        &["run"],
        "image-text-to-text",
        config,
    );
    pkg.metadata("context_length", serde_json::json!(rows - 1));
    pkg.write(out, opts)
}
