// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export VibeVoice-ASR-Streaming as an `.rlxp` package rlx runs alone: the
//! speech encoder over one segment (chunk and lookahead, the last
//! zero-padded) as four graphs — `acoustic` and `semantic` (the two VAE
//! encoders' bodies) and their connectors (`acoustic_connector`,
//! `semantic_connector`; each set's params prefixed, as both are built by
//! the same builder) — `embed` (a token's row), and the Qwen2 decoder's
//! `step` over one embedding row with its caches inside, symbolic in their
//! rows. The checkpoint's tensors are referenced, bf16 as stored. A program
//! answers `transcribe`: what the official `streaming_generate` does with
//! its defaults (`split_then_encode`) — the prompt stepped, then per
//! segment `<|speech_start|>`, its features and `<|speech_end|>`, greedy
//! tokens until `<|text_chunk_end|>` (or the end of text), that token fed
//! back, each chunk's text cleaned of the special tokens.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::{Checkpoint, Package, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::config::{
    COMPRESS_RATIO, STREAMING_PROMPT_PREFIX, TARGET_SR, TOK_ENDOFTEXT, TOK_IM_END, TOK_SPEECH_END,
    TOK_SPEECH_START, TOK_TEXT_CHUNK_END, VibeAsrConfig,
};
use crate::load_streaming::{StreamingLmLoader, StreamingWeightStore};

type Params = HashMap<String, Vec<f32>>;

/// The decoder's cache rows, a symbol.
const CAP: u32 = 0;
/// The positions the RoPE tables reach.
const ROWS: usize = 8192;
/// The checkpoint's embedding table.
const EMBED_KEY: &str = "model.language_model.embed_tokens.weight";

/// `g` and `params` with every param named `prefix` + its name.
fn prefixed(g: &Graph, params: Vec<(String, Vec<f32>)>, prefix: &str) -> Result<(Graph, Params)> {
    let g = rlx_ir::stateful::rebuild(g, |out, _, node, _| match &node.op {
        Op::Param { name } => Ok(Some(
            out.param(format!("{prefix}{name}"), node.shape.clone()),
        )),
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    Ok((
        g,
        params
            .into_iter()
            .map(|(n, v)| (format!("{prefix}{n}"), v))
            .collect(),
    ))
}

fn named(mut g: Graph, name: &str, out: &str) -> Graph {
    g.name = name.into();
    let o = g.outputs[0];
    g.node_mut(o).name = Some(out.into());
    g
}

/// `embed`: `token` i32 `[1]` → `row` `[1, 1, hidden]`.
fn embed(vocab: usize, hidden: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("embed");
    let token = g.input("token", Shape::new(&[1], DType::I32));
    let table = g.param(EMBED_KEY, Shape::new(&[vocab, hidden], f));
    let row = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, token],
        Shape::new(&[1, hidden], f),
    );
    let row = g.reshape_(row, vec![1, 1, hidden as i64]);
    g.node_mut(row).name = Some("row".into());
    g.set_outputs(vec![row]);
    g
}

/// `step`: the decoder over one row of `inputs_embeds` `[1, 1, hidden]` at
/// `position`, symbolic in its caches' rows, its state inside → `logits` and
/// each cache with the row written (`past_k_{l}.next`, `past_v_{l}.next`).
/// The 52-row build's params are dropped before the 36-row one's are taken
/// (a 7B decoder in f32 is 28 GB).
fn step(
    cfg: &rlx_qwen3::Qwen3Config,
    store: &StreamingWeightStore,
    take: &mut dyn FnMut(Params) -> Result<()>,
) -> Result<Graph> {
    let at = |upper: usize| -> Result<(Graph, Params)> {
        let mut wm = store.load_language_model_weights()?;
        let mut loader = StreamingLmLoader::new(&mut wm);
        let built = crate::stream_lm::build_stream_decode_embeds_built(
            cfg,
            &mut loader,
            1,
            upper,
            false,
            true,
        )?;
        rlx_core::flow_util::graph_from_built(built)
    };
    let (b, _) = at(52)?;
    let (a, params) = at(36)?;
    take(params)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(CAP, 37, 53), (1, 36, 52)])
        .map_err(|e| anyhow::anyhow!("the decode step is not symbolic in its cache: {e}"))?;
    drop((a, b));
    let spec = rlx_ir::stateful::StatefulDecode {
        position: "position".into(),
        tables: vec![
            ("rope_cos".into(), "rope_cos.table".into(), ROWS),
            ("rope_sin".into(), "rope_sin.table".into(), ROWS),
        ],
        mask: Some(("mask".into(), "decode.iota".into(), ROWS)),
        cache_symbol: CAP,
        tokens: 1,
        starts: None,
        rope_offset: None,
        windows: Vec::new(),
    };
    let mut g = rlx_ir::stateful::stateful_decode(&g, &spec)
        .map_err(|e| anyhow::anyhow!("the decode step's state: {e}"))?;
    let o = g.outputs.clone();
    g.node_mut(o[0]).name = Some("logits".into());
    for (i, &id) in o[1..].iter().enumerate() {
        let kind = if i % 2 == 0 { "k" } else { "v" };
        g.node_mut(id).name = Some(format!("past_{kind}_{}.next", i / 2));
    }
    g.name = "step".into();
    Ok(g)
}

/// Export the VibeVoice-ASR-Streaming directory `dir` (`config.json`,
/// `preprocessor_config.json`, the shards, `tokenizer.json`) as the package
/// `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = VibeAsrConfig::from_model_dir(dir)?;
    let qcfg = crate::lm::qwen3_config(&cfg.lm);
    let store = StreamingWeightStore::open(dir)?;
    let ck = Checkpoint::open(dir)?;
    let mut refr = ck.referencer(opts.embed_weights)?;

    let step = step(&qcfg, &store, &mut |p| refr.take(p))?;
    let mut tables = Params::new();
    let half = qcfg.head_dim / 2;
    let (mut cos, mut sin) = (
        Vec::with_capacity(ROWS * half),
        Vec::with_capacity(ROWS * half),
    );
    for p in 0..ROWS {
        let (c, s) = crate::stream_lm::rope_slice(&qcfg, p);
        cos.extend(c);
        sin.extend(s);
    }
    tables.insert("rope_cos.table".into(), cos);
    tables.insert("rope_sin.table".into(), sin);
    tables.insert("decode.iota".into(), (0..ROWS).map(|i| i as f32).collect());
    let mut wm = store.load_keys(&[EMBED_KEY])?;
    tables.insert(EMBED_KEY.into(), wm.take(EMBED_KEY)?.0);
    drop(wm);
    refr.take(tables)?;

    // The speech encoder over one segment.
    let sched = &cfg.streaming;
    let segment = sched.chunk_samples() + sched.lookahead_samples();
    anyhow::ensure!(
        segment.is_multiple_of(COMPRESS_RATIO),
        "a segment of {segment} samples is not whole frames"
    );
    let (acoustic, semantic) = store.load_vae_pair()?;
    let mut graphs = vec![embed(qcfg.vocab_size, qcfg.hidden_size), step];
    let mut frames = 0;
    for (name, w) in [("acoustic", &acoustic), ("semantic", &semantic)] {
        let (bg, bp, n, _) = crate::vae::build_latent_graph_with_act(w, segment, cfg.vae_ffn)?;
        let (bg, bp) = prefixed(&bg, bp, &format!("{name}."))?;
        refr.take(bp)?;
        graphs.push(named(bg, name, "latent"));
        let (cg, cp, _) = crate::vae::build_connector_graph(&w.connector, n)?;
        let (cg, cp) = prefixed(&cg, cp, &format!("{name}_connector."))?;
        refr.take(cp)?;
        graphs.push(named(cg, &format!("{name}_connector"), "features"));
        frames = n;
    }
    drop((acoustic, semantic));

    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        "vibevoice-asr-streaming",
        "automatic-speech-recognition",
        "rlx-vibevoice-asr",
    );
    pkg.model["license"] = json!("mit");
    pkg.weights_as_stored();
    pkg.graphs = graphs;
    pkg.dims = vec![("cap".into(), CAP)];
    pkg.reference(refr, out)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build VibeVoice-ASR's program")?;
    let config = json!({
        "tokenizer": tokenizer,
        "hidden": qcfg.hidden_size,
        "layers": qcfg.num_hidden_layers,
        "kv": qcfg.kv_proj_dim(),
        "rows": ROWS,
        "sample_rate": TARGET_SR,
        "chunk": sched.chunk_samples(),
        "segment": segment,
        "frames": frames,
        "prompt": format!("{STREAMING_PROMPT_PREFIX}\n"),
        "prompt_prefix": STREAMING_PROMPT_PREFIX,
        "speech_start": TOK_SPEECH_START,
        "speech_end": TOK_SPEECH_END,
        "chunk_end": TOK_TEXT_CHUNK_END,
        "stop": [TOK_TEXT_CHUNK_END, TOK_IM_END, TOK_ENDOFTEXT],
        "max_tokens": 256,
    });
    program::entry(
        &mut pkg,
        "stream",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        config,
    );
    pkg.metadata("sample_rate", json!(TARGET_SR));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "vibevoice-asr-streaming",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
