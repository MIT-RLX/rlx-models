// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Export Voxtral (`mistralai/Voxtral-Mini-3B-2507`) as an `.rlxp` package
//! rlx runs alone: 16 kHz speech in, its transcript out, through the one
//! exporter ([`rlx_core::export`]).
//!
//! Its graphs are the runner's: `mel` (Whisper's log-mel of the
//! reflect-padded signal, symbolic in its frames); `encoder` (the Whisper
//! encoder over `chunks` 30 s windows) and `projector` (each four encoder
//! rows through the multimodal projector, `[1, rows, hidden]`); `embed` (the
//! prompt's token embeddings, the audio's rows written at its
//! placeholders); `prompt` (the Llama decoder over the prompt, its last
//! row's logits and each layer's keys and values at the start of caches of
//! `cap` rows) and `step` (one token at `position`, under the mask the
//! program gives, its key and value written in). The program (`program/`)
//! pads the signal to whole windows, builds the transcription prompt
//! (`options.language`, else `en`), decodes greedily to `</s>` and renders
//! the ids through `tekken.json`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::audio::{whisper_log_mel, whisper_mel_params};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_core::flow_util::graph_from_built;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_llama32::Llama32Flow;
use serde_json::json;

use crate::config::VoxtralConfig;
use crate::encoder::build_voxtral_encoder_built;
use crate::lm_flow::build_voxtral_prefill_built;
use crate::load::VoxtralWeightStore;
use crate::projector::build_voxtral_projector_built;
use crate::weights::{LanguageModelPrefixLoader, VoxtralWeightPrefix};

/// The symbols: the caches' rows, the mel's frames, the 30 s windows, the
/// audio rows, the prompt's rows.
const CAP: u32 = 0;
const FRAMES: u32 = 10;
const CHUNKS: u32 = 11;
const TOKENS: u32 = 12;
const PROMPT: u32 = 13;
/// Positions the package's RoPE tables reach (a prompt and its transcript).
const ROWS: usize = 8192;
const N_FFT: usize = 400;
const HOP: usize = 160;
const SAMPLE_RATE: usize = 16_000;
/// Mel frames in one 30 s window.
const WINDOW_FRAMES: usize = 3000;
const EMBED: &str = "model.embed_tokens.weight";
/// The languages the transcription prompt names, by code.
const LANGUAGES: [&str; 8] = ["en", "fr", "de", "es", "pt", "hi", "nl", "it"];

type Built = (Graph, HashMap<String, Vec<f32>>);

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

fn name(g: &mut Graph, id: NodeId, n: &str) {
    g.node_mut(id).name = Some(n.into());
}

/// `encoder` at `n` windows: `mel` `[n, n_mels, 3000]` → `hidden` `[n,
/// 1500, d_model]`.
fn encoder(cfg: &VoxtralConfig, store: &VoxtralWeightStore, n: usize) -> Result<Built> {
    let mut wm = store.load_audio_weights()?;
    let (mut g, params) = graph_from_built(build_voxtral_encoder_built(
        &cfg.audio_config,
        &mut wm,
        n,
        WINDOW_FRAMES,
    )?)?;
    let out = g.outputs[0];
    name(&mut g, out, "hidden");
    g.name = "encoder".into();
    Ok((g, params))
}

/// `projector` at `n` windows: `encoder_hidden` `[n, 1500, d_model]` →
/// `audio` `[1, n · 375, hidden]`.
fn projector(cfg: &VoxtralConfig, store: &VoxtralWeightStore, n: usize) -> Result<Built> {
    let enc_seq = cfg.audio_config.encoder_seq_len(WINDOW_FRAMES);
    let mut wm = store.load_projector_weights()?;
    let (mut g, params) =
        graph_from_built(build_voxtral_projector_built(cfg, &mut wm, n, enc_seq)?)?;
    let rows = n * enc_seq / 4;
    let h = cfg.text_config.hidden_size;
    let out = g.outputs[0];
    let out = g.reshape_(out, vec![1, rows as i64, h as i64]);
    name(&mut g, out, "audio");
    g.set_outputs(vec![out]);
    g.name = "projector".into();
    Ok((g, params))
}

/// `embed` at `t` rows and `a` audio rows: `ids` `[t]` i32, `audio` `[1,
/// a, hidden]` and `at` `[1]` (its first placeholder's row) → `embeds` `[1,
/// t, hidden]`.
fn embed(cfg: &VoxtralConfig, t: usize, a: usize) -> Graph {
    let (h, vocab) = (cfg.text_config.hidden_size, cfg.text_config.vocab_size);
    let mut g = Graph::new("embed");
    let ids = g.input("ids", Shape::new(&[t], DType::I32));
    let audio = g.input("audio", f32s(&[1, a, h]));
    let at = g.input("at", f32s(&[1]));
    let table = g.param(EMBED, f32s(&[vocab, h]));
    let rows = g.gather_(table, ids, 0);
    let rows = g.reshape_(rows, vec![1, t as i64, h as i64]);
    let out = g.add_node(
        Op::DynamicUpdateSlice { axis: 1 },
        vec![rows, audio, at],
        f32s(&[1, t, h]),
    );
    name(&mut g, out, "embeds");
    g.set_outputs(vec![out]);
    g
}

/// `prompt` at `t` rows and `cap` cache rows: `inputs_embeds` `[1, t,
/// hidden]` → `logits` `[1, vocab]` (the last row's) and each layer's keys
/// and values at the start of zero caches `[1, cap, kv]` (`past_k_{l}`,
/// `past_v_{l}`).
fn prompt(cfg: &VoxtralConfig, store: &VoxtralWeightStore, t: usize, cap: usize) -> Result<Built> {
    let mut wm = store.load_language_model_weights()?;
    let (mut g, params) =
        graph_from_built(build_voxtral_prefill_built(cfg, &mut wm, 1, t, true, true)?)?;
    let o = g.outputs.clone();
    name(&mut g, o[0], "logits");
    let kv = cfg.llama_config().kv_proj_dim();
    let origin = constant(&mut g, 0.0);
    let zero = constant(&mut g, 0.0);
    let zero = g.reshape_(zero, vec![1, 1, 1]);
    let mut outs = vec![o[0]];
    for (i, &rows) in o[1..].iter().enumerate() {
        let z = g.add_node(
            Op::Expand {
                target_shape: vec![1, cap as i64, kv as i64],
            },
            vec![zero],
            f32s(&[1, cap, kv]),
        );
        let full = g.add_node(
            Op::DynamicUpdateSlice { axis: 1 },
            vec![z, rows, origin],
            f32s(&[1, cap, kv]),
        );
        let kind = if i % 2 == 0 { "k" } else { "v" };
        name(&mut g, full, &format!("past_{kind}_{}", i / 2));
        outs.push(full);
    }
    g.set_outputs(outs);
    g.name = "prompt".into();
    Ok((g, params))
}

/// `step` at `cap` cache rows: `input_ids` `[1, 1]`, `position` `[1]` and
/// `mask` `[1, cap + 1]` (1 on the cached rows and the last, the token's)
/// → `logits` and each layer's caches with the token's key and value
/// written at `position` (`past_k_{l}.next`, `past_v_{l}.next`).
fn step(cfg: &VoxtralConfig, store: &VoxtralWeightStore, cap: usize) -> Result<Built> {
    let mut wm = store.load_language_model_weights()?;
    let built = {
        let mut loader = LanguageModelPrefixLoader::new(&mut wm);
        Llama32Flow::for_decode(cfg.llama_config(), 1, cap)
            .custom_mask()
            .export_kv()
            .build(&mut loader)?
    };
    let (mut g, params) = graph_from_built(built)?;
    let o = g.outputs.clone();
    name(&mut g, o[0], "logits");
    let pos = g.input_id("position").context("no position")?;
    let mut outs = vec![o[0]];
    for (i, &full) in o[1..].iter().enumerate() {
        let kind = if i % 2 == 0 { "k" } else { "v" };
        let input = format!("past_{kind}_{}", i / 2);
        let cache = g.input_id(&input).with_context(|| format!("no {input}"))?;
        let rows = g.node(full).shape.dims()[1].unwrap_static();
        let row = g.narrow_(full, 1, rows - 1, 1);
        let shape = g.node(cache).shape.clone();
        let next = g.add_node(
            Op::DynamicUpdateSlice { axis: 1 },
            vec![cache, row, pos],
            shape,
        );
        name(&mut g, next, &format!("{input}.next"));
        outs.push(next);
    }
    g.set_outputs(outs);
    g.name = "step".into();
    Ok((g, params))
}

/// Export the Voxtral directory `dir` (`config.json`, the safetensors
/// shards, `tekken.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let mut cfg = VoxtralConfig::from_file(&dir.join("config.json"))?;
    cfg.validate()?;
    // The decoder's tables reach `ROWS` positions, not the config's.
    cfg.text_config.max_position_embeddings = cfg.text_config.max_position_embeddings.min(ROWS);
    let tekken = dir.join("tekken.json");
    if !tekken.is_file() {
        bail!("{}: no tekken.json", dir.display());
    }
    let store = VoxtralWeightStore::open(dir)?;
    let ck = Checkpoint::open(dir)?;
    let mut refr = ck.referencer(opts.embed_weights)?;
    let n_mels = cfg.audio_config.num_mel_bins;
    let mut pkg = Package::new(
        "voxtral-mini-3b",
        &opts.version,
        "voxtral",
        "automatic-speech-recognition",
        "rlx-voxtral",
    );
    pkg.model["license"] = json!("apache-2.0");
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();

    // The front end.
    let (mg, _) = symbolic_in_lifted_many(&[(FRAMES, vec![37, 53, 20])], &[0], |n| {
        Ok(whisper_log_mel(N_FFT, HOP, n_mels, n[0]))
    })?;
    pkg.function(mg);
    held.extend(whisper_mel_params(
        N_FFT,
        &rlx_whisper::mel::mel_filters(n_mels),
    ));

    // The encoder and projector over whole windows.
    let windows = vec![5, 7, 3];
    // Its batch-broadcast biases are params as long as the windows: held
    // as one slice, expanded.
    let eg = symbolic_in(CHUNKS, (5, 7, 3), |n| Ok(encoder(&cfg, &store, n)?.0))?;
    let (at, mut ep) = encoder(&cfg, &store, 3)?;
    let eg = broadcast_params(&eg, &at, &mut ep, &[])?;
    pkg.function(eg);
    refr.take(ep)?;
    let (pg, _) = symbolic_in_lifted_many(&[(CHUNKS, windows)], &[0], |n| {
        Ok(projector(&cfg, &store, n[0])?.0)
    })?;
    pkg.function(pg);
    refr.take(projector(&cfg, &store, 3)?.1)?;

    // The prompt's embedding, the decoder over it, and the step.
    let (bg, _) = symbolic_in_lifted_many(
        &[(PROMPT, vec![37, 53, 20]), (TOKENS, vec![9, 13, 6])],
        &[0, 0],
        |n| Ok(embed(&cfg, n[0], n[1])),
    )?;
    pkg.function(bg);
    let (pg, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![37, 53, 20]), (CAP, vec![70, 101, 44])],
        &[0, 0],
        |n| Ok(prompt(&cfg, &store, n[0], n[1])?.0),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    pkg.function(pg);
    refr.take(prompt(&cfg, &store, 20, 44)?.1)?;
    let (sg, lifted) = symbolic_in_lifted_many(&[(CAP, vec![37, 53, 20])], &[0], |n| {
        Ok(step(&cfg, &store, n[0])?.0)
    })?;
    anyhow::ensure!(lifted.is_empty(), "the step lifted tables");
    pkg.function(sg);
    refr.take(step(&cfg, &store, 20)?.1)?;
    {
        let key = VoxtralWeightPrefix::lm_embed_tokens();
        let mut wm = store.load_keys(&[key])?;
        refr.take(HashMap::from([(EMBED.to_string(), wm.take(key)?.0)]))?;
    }
    pkg.dims = vec![
        ("cap".into(), CAP),
        ("frames".into(), FRAMES),
        ("chunks".into(), CHUNKS),
        ("tokens".into(), TOKENS),
        ("prompt".into(), PROMPT),
    ];

    // Checkpoint tensors referenced, the rest held: the mel's bases and
    // filters, the RoPE tables.
    pkg.reference(refr, out)?;
    let dims = pkg.param_dims();
    for (n, v) in &held {
        let d = dims
            .get(n)
            .cloned()
            .with_context(|| format!("no graph takes {n}"))?;
        pkg.hold_f32(n.clone(), d, v);
    }

    // The text front end, and the program.
    pkg.sidecar("tekken.json", "application/json", std::fs::read(&tekken)?);
    let mut languages = serde_json::Map::new();
    for code in LANGUAGES {
        languages.insert(
            code.into(),
            json!(crate::embed::language_code_token_id(Some(dir), code)?),
        );
    }
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Voxtral's program")?;
    program::entry(
        &mut pkg,
        "transcribe",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        json!({
            "tekken": "sidecar:tekken.json",
            "sample_rate": SAMPLE_RATE,
            "hop": HOP,
            "n_fft": N_FFT,
            "n_mels": n_mels,
            "window_frames": WINDOW_FRAMES,
            "rows_per_window": cfg.audio_config.audio_token_count(WINDOW_FRAMES),
            "layers": cfg.text_config.num_hidden_layers,
            "rows": ROWS,
            "max_tokens": 512,
            "audio": cfg.audio_token_id,
            "eos": cfg.eos_token_id,
            "languages": languages,
            "language": "en",
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "voxtral",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
