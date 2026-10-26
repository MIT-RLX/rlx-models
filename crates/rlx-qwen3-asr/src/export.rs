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

//! Export Qwen3-ASR as an `.rlxp` package rlx runs alone: 16 kHz speech in,
//! its transcript out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs are the runner's: `mel` (the Whisper log-mel of the
//! reflect-padded signal, symbolic in its frames, `[n_mels, frames]`);
//! `encoder` (the audio tower over the mel cut into `chunks` of
//! `2·n_window` frames, symbolic in them and in the audio `tokens` it
//! keeps, each token's attention window an input); `embed` (the prompt's
//! token embeddings, the audio's rows written at its placeholders);
//! `prompt` (the Qwen3 decoder over the prompt, its last row's logits and
//! each layer's keys and values at the start of caches of `cap` rows) and
//! `step` (one token at `position`, its key and value written in, RoPE rows
//! and mask from the position). The program (`program/`) pads the signal,
//! lays the mel out in chunks, builds the chat prompt (`options.context` its
//! system text, `options.language` forcing the language as the reference
//! does), decodes greedily and returns the text after `<asr_text>`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::audio::{whisper_log_mel, whisper_mel_params};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_qwen3::Qwen3Config;
use serde_json::json;

use crate::audio::AudioGeometry;
use crate::config::Qwen3AsrConfig;
use crate::encoder::build_encoder_windowed;
use crate::lm_flow::{build_asr_prefill_built, rope_slice};
use crate::load::AsrWeightStore;
use crate::tokenizer::AsrTokenizer;
use crate::weights::{
    KEY_EMBED_TOKENS, KEY_LM_HEAD, LanguageModelPrefixLoader, PREFIX_LANGUAGE_MODEL,
};

/// The symbols: the step's cache rows (and its bucket, 1), the mel's
/// frames, the encoder's chunks and audio tokens, the prompt's rows.
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
const EMBED: &str = "model.embed_tokens.weight";
/// `<|endoftext|>`, `<|im_end|>`: where the transcript ends.
const EOS: [u32; 2] = [151643, 151645];
const IM_START: u32 = 151644;
const IM_END: u32 = 151645;

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

/// `encoder` at `c` chunks and `a` audio tokens: `mel` `[1, n_mels, c ·
/// chunk]` (each chunk `chunk` frames, the last zero-padded) and `window`
/// `[a]` → `audio` `[1, a, hidden]`.
fn encoder(cfg: &Qwen3AsrConfig, store: &AsrWeightStore, c: usize, a: usize) -> Result<Built> {
    let chunk = cfg.audio.chunk_frames();
    let mut geom = AudioGeometry::new(&cfg.audio, c * chunk)?;
    geom.num_audio_tokens = a;
    geom.windows = vec![a];
    let mut wm = store.load_audio_weights()?;
    let (mut g, params) = graph_from_built(build_encoder_windowed(&cfg.audio, &mut wm, &geom)?)?;
    let out = g.outputs[0];
    name(&mut g, out, "audio");
    g.name = "encoder".into();
    Ok((g, params))
}

/// `embed` at `t` rows and `a` audio tokens: `ids` `[t]` i32, `audio`
/// `[1, a, hidden]` and `at` `[1]` (its first placeholder's row) → `embeds`
/// `[1, t, hidden]`: each id's embedding, the audio's rows from `at`.
fn embed(cfg: &Qwen3Config, t: usize, a: usize) -> Graph {
    let h = cfg.hidden_size;
    let mut g = Graph::new("embed");
    let ids = g.input("ids", Shape::new(&[t], DType::I32));
    let audio = g.input("audio", f32s(&[1, a, h]));
    let at = g.input("at", f32s(&[1]));
    let table = g.param(EMBED, f32s(&[cfg.vocab_size, h]));
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

/// The decoder's tensors as the flow builders name them.
fn lm_weights(store: &AsrWeightStore) -> Result<WeightMap> {
    store.load_prefixes(&[PREFIX_LANGUAGE_MODEL, KEY_LM_HEAD])
}

/// `prompt` at `t` rows and `cap` cache rows: `inputs_embeds` `[1, t,
/// hidden]` → `logits` `[1, vocab]` (the last row's) and each layer's keys
/// and values at the start of zero caches `[1, cap, kv]`, named as the
/// step's cache inputs (`past_k_{l}`, `past_v_{l}`).
fn prompt(cfg: &Qwen3Config, store: &AsrWeightStore, t: usize, cap: usize) -> Result<Built> {
    let mut wm = lm_weights(store)?;
    let built = {
        let mut loader = LanguageModelPrefixLoader::new(&mut wm);
        build_asr_prefill_built(cfg, &mut loader, 1, t, false)?
    };
    let (mut g, params) = graph_from_built(built)?;
    let o = g.outputs.clone();
    name(&mut g, o[0], "logits");
    let kv = cfg.kv_proj_dim();
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

/// `step`: the Qwen3 decode step symbolic in its cache rows (`cap`), with
/// its state inside ([`rlx_ir::stateful::stateful_decode`]): `input_ids`
/// and `position` → `logits` and each cache with the token's row written at
/// the position (`past_k_{l}.next`, `past_v_{l}.next`).
fn step(cfg: &Qwen3Config, store: &AsrWeightStore) -> Result<Built> {
    let at = |upper: usize| -> Result<Built> {
        let mut wm = lm_weights(store)?;
        let mut loader = LanguageModelPrefixLoader::new(&mut wm);
        rlx_qwen3::builder::build_qwen3_decode_graph_sized_ext(cfg, &mut loader, 1, upper, true)
    };
    let (a, params) = at(36)?;
    let (b, _) = at(52)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(CAP, 37, 53), (1, 36, 52)])
        .map_err(|e| anyhow::anyhow!("the decode step is not symbolic in its cache: {e}"))?;
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
    name(&mut g, o[0], "logits");
    for (i, &id) in o[1..].iter().enumerate() {
        let kind = if i % 2 == 0 { "k" } else { "v" };
        name(&mut g, id, &format!("past_{kind}_{}.next", i / 2));
    }
    g.name = "step".into();
    Ok((g, params))
}

/// Export the Qwen3-ASR checkpoint in `dir` (`config.json`,
/// `model.safetensors`, `vocab.json`, `merges.txt`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = Qwen3AsrConfig::from_file(&dir.join("config.json"))?;
    cfg.validate()?;
    let store = AsrWeightStore::open(dir)?;
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        bail!("{}: no model.safetensors", dir.display());
    }
    // The decoder's tables reach `ROWS` positions, not the config's.
    let mut text = cfg.text.clone();
    text.max_position_embeddings = text.max_position_embeddings.min(ROWS);
    let chunk = cfg.audio.chunk_frames();
    let n_mels = cfg.audio.num_mel_bins;
    let size = if dir.to_string_lossy().contains("1.7B") {
        "1.7b"
    } else {
        "0.6b"
    };
    let mut pkg = Package::new(
        &format!("qwen3-asr-{size}"),
        &opts.version,
        "qwen3-asr",
        "automatic-speech-recognition",
        "rlx-qwen3-asr",
    );
    pkg.model["license"] = json!("apache-2.0");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    // The front end.
    let (mg, lifted) = symbolic_in_lifted_many(&[(FRAMES, vec![37, 53, 20])], &[0], |n| {
        Ok(whisper_log_mel(N_FFT, HOP, n_mels, n[0]))
    })?;
    anyhow::ensure!(lifted.is_empty(), "the mel lifted tables");
    pkg.function(mg);
    params.extend(whisper_mel_params(
        N_FFT,
        &crate::audio::mel_filterbank(SAMPLE_RATE as f64, N_FFT, n_mels),
    ));

    // The audio tower: chunks and tokens at sizes no other dim has (a
    // token count is past the full chunks' and within the last's).
    let t_pc = crate::audio::downsample_time(chunk);
    let sizes = [(5, 4 * t_pc + 8), (7, 6 * t_pc + 3), (3, 2 * t_pc + 6)];
    let (eg, lifted) = symbolic_in_lifted_many(
        &[
            (CHUNKS, sizes.iter().map(|s| s.0).collect()),
            (TOKENS, sizes.iter().map(|s| s.1).collect()),
        ],
        &[0, 0],
        |n| Ok(encoder(&cfg, &store, n[0], n[1])?.0),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the encoder lifted tables");
    pkg.function(eg);
    params.extend(encoder(&cfg, &store, sizes[2].0, sizes[2].1)?.1);

    // The prompt's embedding, the decoder over it, and the step.
    let (bg, _) = symbolic_in_lifted_many(
        &[(PROMPT, vec![37, 53, 20]), (TOKENS, vec![9, 13, 6])],
        &[0, 0],
        |n| Ok(embed(&text, n[0], n[1])),
    )?;
    pkg.function(bg);
    let (pg, lifted) = symbolic_in_lifted_many(
        &[(PROMPT, vec![37, 53, 20]), (CAP, vec![70, 101, 44])],
        &[0, 0],
        |n| Ok(prompt(&text, &store, n[0], n[1])?.0),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the prompt graph lifted tables");
    pkg.function(pg);
    params.extend(prompt(&text, &store, 20, 44)?.1);
    let (sg, sp) = step(&text, &store)?;
    pkg.function(sg);
    params.extend(sp);
    {
        let mut wm = store.load_keys(&[KEY_EMBED_TOKENS])?;
        params.insert(EMBED.into(), wm.take(KEY_EMBED_TOKENS)?.0);
    }
    pkg.dims = vec![
        ("cap".into(), CAP),
        ("frames".into(), FRAMES),
        ("chunks".into(), CHUNKS),
        ("tokens".into(), TOKENS),
        ("prompt".into(), PROMPT),
    ];

    // RoPE rows and the mask's iota for every position the step reaches.
    let half = text.head_dim / 2;
    let (mut cos, mut sin) = (
        Vec::with_capacity(ROWS * half),
        Vec::with_capacity(ROWS * half),
    );
    for pos in 0..ROWS {
        let (c, s) = rope_slice(&text, pos);
        cos.extend(c);
        sin.extend(s);
    }
    params.insert("rope_cos.table".into(), cos);
    params.insert("rope_sin.table".into(), sin);
    params.insert("decode.iota".into(), (0..ROWS).map(|i| i as f32).collect());

    // Checkpoint tensors referenced (a fingerprint match, transposed or
    // not), the rest held: the mel's bases, the positions, the tables.
    let source = WeightMap::from_file(weights.to_str().context("path")?)?;
    pkg.reference_map(&params, &source, "safetensors", &weights, out, false)?;
    drop(source);

    // The text front end, and the program.
    let tok = AsrTokenizer::from_model_dir(dir)?;
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        tok.to_json()?.into_bytes(),
    );
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build Qwen3-ASR's program")?;
    program::entry(
        &mut pkg,
        "transcribe",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        json!({
            "tokenizer": "sidecar:tokenizer.json",
            "sample_rate": SAMPLE_RATE,
            "hop": HOP,
            "n_fft": N_FFT,
            "n_mels": n_mels,
            "chunk": chunk,
            "window": t_pc * (cfg.audio.n_window_infer / chunk).max(1),
            "layers": text.num_hidden_layers,
            "rows": ROWS,
            "max_tokens": 512,
            "im_start": IM_START,
            "im_end": IM_END,
            "audio_start": cfg.audio_start_token_id,
            "audio_pad": cfg.audio_token_id,
            "audio_end": cfg.audio_end_token_id,
            "asr_text": 151704,
            "eos": EOS,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "qwen3-asr",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
