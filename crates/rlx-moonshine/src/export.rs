// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Moonshine (v1) checkpoint as an `.rlxp` package rlx runs alone:
//! `Model.transcribe(audio)`. A `transcribe` entry: the encoder over the raw
//! PCM, one graph symbolic in its length (the conv frontend's frames are
//! quotients of it; its RoPE rows read from one held table), then a decoder
//! with no step — each token, the whole prefix through the full-prefix
//! decoder at the bucket that holds it, the encoder's states its context,
//! the tied head on its last row — greedy until the end token, as
//! [`crate::MoonshineRunner::transcribe`] decodes.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::export::graph::{Pipeline, Stage, broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::weight_source::CloningWeightSource;
use crate::{MoonshineConfig, MoonshineWeightPrefix};

/// The audio and the encoded-frames symbols.
const SAMPLES: u32 = 0;
const ENCODED: u32 = 1;
/// RoPE rows held for the encoder: past 3 minutes of audio.
const ENCODER_ROWS: usize = 8192;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// The encoder for `samples` of audio: `audio` `[samples]` → `encoded`
/// `[1, frames, d]`, the params the builder names after a length renamed
/// apart from it.
fn encoder(
    cfg: &MoonshineConfig,
    weights: &WeightMap,
    pfx: &MoonshineWeightPrefix,
    samples: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let t = MoonshineConfig::feat_extract_output_length(samples);
    let mut src = CloningWeightSource(weights);
    let built = crate::flow::build_encoder_built(cfg, &mut src, pfx, 1, samples)?;
    let (inner, mut params) = graph_from_built(built)?;
    let mut g = Graph::new("encode");
    let audio = g.input("audio", f32s(&[samples]));
    let pcm = g.reshape_(audio, vec![1, samples as i64]);
    let spliced = g
        .splice(&inner, &rlx_ir::Splice::new().bind("pcm", pcm))
        .map_err(|e| anyhow!("splicing the encoder: {e}"))?;
    let out = spliced.outputs[0];
    g.node_mut(out).name = Some("encoded".into());
    g.set_outputs(vec![out]);
    // Params the builder names after a length (`rope_cos_enc_76`,
    // `conv_bias_ones_118`) take one name at every length: the stem and its
    // place among those of that stem (several convs each have one).
    let mut seen: HashMap<String, usize> = HashMap::new();
    let renames: Vec<(String, String)> = g
        .nodes()
        .iter()
        .filter_map(|n| {
            let Op::Param { name } = &n.op else {
                return None;
            };
            let (stem, num) = name.rsplit_once('_')?;
            let num: usize = num.parse().ok()?;
            if !n.shape.dims().contains(&rlx_ir::Dim::Static(num)) {
                return None;
            }
            let k = seen.entry(stem.to_string()).or_insert(0);
            *k += 1;
            Some((name.clone(), format!("{stem}.{}", *k - 1)))
        })
        .collect();
    let _ = t;
    for (from, to) in renames {
        for n in g.nodes_mut() {
            if matches!(&n.op, Op::Param { name } if *name == from) {
                n.op = Op::Param { name: to.clone() };
                n.name = Some(to.clone());
            }
        }
        if let Some(v) = params.remove(&from) {
            params.insert(to, v);
        }
    }
    Ok((g, params))
}

/// The encoder's RoPE table, `rows` long, as the builder fills it.
fn encoder_rope(cfg: &MoonshineConfig, rows: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let hd = cfg.enc_head_dim();
    let n_rot = cfg.n_rot(hd);
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; rows * half], vec![0f32; rows * half]);
    for pos in 0..rows {
        for i in 0..n_rot / 2 {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / n_rot as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos[pos * half + i] = c as f32;
            sin[pos * half + i] = s as f32;
        }
    }
    (cos, sin, half)
}

/// The decoder over a prefix of `cap` tokens and `enc` encoded frames:
/// `tokens` `[1, cap]`, `last` `[1]`, `encoder_states` `[1, enc, d]` →
/// `logits` `[1, vocab]` at row `last`.
fn decoder(
    cfg: &MoonshineConfig,
    weights: &WeightMap,
    pfx: &MoonshineWeightPrefix,
    cap: usize,
    enc: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (d, vocab) = (cfg.hidden_size, cfg.vocab_size);
    let mut src = CloningWeightSource(weights);
    let built = crate::flow::build_decoder_hidden_built(cfg, &mut src, pfx, 1, cap, enc)?;
    let (inner, params) = graph_from_built(built)?;
    let mut g = Graph::new(format!("decode_{cap}"));
    let tokens = g.input("tokens", Shape::new(&[1, cap], DType::I32));
    let last = g.input("last", Shape::new(&[1], DType::I32));
    let states = g.input("encoder_states", f32s(&[1, enc, d]));
    let table = g.param(pfx.dec_embed_tokens(), f32s(&[vocab, d]));
    let flat = g.reshape(tokens, vec![cap as i64], Shape::new(&[cap], DType::I32));
    let rows = g.gather(table, flat, 0, f32s(&[cap, d]));
    let embeds = g.reshape(rows, vec![1, cap as i64, d as i64], f32s(&[1, cap, d]));
    let spliced = g
        .splice(
            &inner,
            &rlx_ir::Splice::new()
                .bind("decoder_inputs_embeds", embeds)
                .bind("encoder_hidden", states),
        )
        .map_err(|e| anyhow!("splicing the decoder: {e}"))?;
    let hidden = g.reshape(
        spliced.outputs[0],
        vec![cap as i64, d as i64],
        f32s(&[cap, d]),
    );
    let row = g.gather(hidden, last, 0, f32s(&[1, d]));
    // A tied head reads the embedding's own param node: two params of one
    // name would bind one of them.
    let head = match &pfx.proj_out {
        Some(name) => g.param(name.clone(), f32s(&[vocab, d])),
        None => table,
    };
    let head = g.transpose_(head, vec![1, 0]);
    let logits = g.mm(row, head);
    g.node_mut(logits).name = Some("logits".into());
    g.set_outputs(vec![logits]);
    Ok((g, params))
}

/// The encoder as one graph symbolic in the audio's length ([`SAMPLES`]),
/// and its params (its RoPE rows a held table, read to the length).
fn symbolic_encoder(
    cfg: &MoonshineConfig,
    weights: &WeightMap,
    pfx: &MoonshineWeightPrefix,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    // The frames are three nested quotients of the samples past the first
    // conv's 127 (by 64, 3, 2: each `⌊(L − c) / k⌋ + 1`): fit them through
    // lengths whose remainders of `L − 127` by 384 differ, the smallest and
    // the largest among them, and check at another.
    let lengths: Vec<usize> = [0usize, 383, 1, 382, 191, 192, 96, 288]
        .iter()
        .enumerate()
        .map(|(i, r)| (60 + 5 * i) * 384 + 127 + r)
        .collect();
    let built: Vec<(Graph, HashMap<String, Vec<f32>>)> = lengths
        .iter()
        .map(|&n| encoder(cfg, weights, pfx, n))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().map(|(g, _)| g).collect();
    let enc_g = rlx_ir::dynamic::generalize_many(&refs, &[(SAMPLES, lengths.clone())], Some(4096))
        .map_err(|e| anyhow!("the encoder is not symbolic in the audio's length: {e}"))?;
    let check_len = 47 * 384 + 123;
    let (check, _) = encoder(cfg, weights, pfx, check_len)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(SAMPLES, check_len);
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&enc_g, &bind)),
        format!("{check}"),
    );
    if got != want {
        let (a, b) = got
            .lines()
            .zip(want.lines())
            .find(|(a, b)| a != b)
            .unwrap_or_default();
        bail!(
            "the encoder generalized from eight lengths is not the one built at another: \
             `{a}` where it builds `{b}`"
        );
    }
    let (at, mut params) = built.into_iter().next().expect("eight builds");
    let (cos, sin, half) = encoder_rope(cfg, ENCODER_ROWS);
    let enc_g = broadcast_params(
        &enc_g,
        &at,
        &mut params,
        &[
            ("rope_cos_enc.0", &cos, half),
            ("rope_sin_enc.0", &sin, half),
        ],
    )?;

    Ok((enc_g, params))
}

/// Export the Moonshine checkpoint in `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg = MoonshineConfig::from_dir(dir)?;
    let (_, weights) = rlx_core::weights::open_map(dir.join("model.safetensors"))?;
    let pfx = MoonshineWeightPrefix::detect(&weights);

    let (enc_g, mut params) = symbolic_encoder(&cfg, &weights, &pfx)?;

    // A full-prefix decoder per bucket the runner pads to, each symbolic in
    // the encoded frames.
    let max_len = cfg.max_position_embeddings.max(2);
    let caps: Vec<usize> = crate::runner::DECODER_BUCKETS
        .iter()
        .copied()
        .filter(|&b| b < max_len)
        .chain(std::iter::once(max_len))
        .collect();
    let mut functions = Vec::new();
    let mut decoders = Vec::new();
    for &cap in &caps {
        let g = symbolic_in(ENCODED, (5, 7, 4), |t| {
            Ok(decoder(&cfg, &weights, &pfx, cap, t)?.0)
        })
        .with_context(|| format!("the decoder at {cap} tokens"))?;
        let (_, p) = decoder(&cfg, &weights, &pfx, cap, 4)?;
        params.extend(p);
        functions.push(g.name.clone());
        decoders.push(g);
    }
    for name in [pfx.dec_embed_tokens()]
        .into_iter()
        .chain(pfx.proj_out.clone())
    {
        let (v, _) = weights.get(&name).ok_or_else(|| anyhow!("no {name}"))?;
        params.insert(name, v.to_vec());
    }

    let stages = vec![
        Stage::new(enc_g)
            .input("audio", "audio")
            .outputs(&["encoded"]),
    ];
    let mut p = Pipeline::new("moonshine", "speech-recognition", stages);
    p.dims = vec![("samples".into(), SAMPLES), ("encoded".into(), ENCODED)];
    p.functions = decoders;
    p.exported_by = "rlx-moonshine".into();
    let mut pkg = p.package(params, Some(&ck), out, opts)?;
    let pipeline = pkg.model["entries"]
        .as_object_mut()
        .and_then(|e| e.remove("pipeline"))
        .ok_or_else(|| anyhow!("the pipeline wrote no entry"))?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    pkg.entry(
        "transcribe",
        json!({
            "sample_rate": crate::SAMPLE_RATE,
            "stages": pipeline["stages"],
            "decoder": {
                "prefill": {
                    "functions": functions,
                    "inputs": {
                        "tokens": "tokens",
                        "last": "last",
                        "encoder_states": {"context": "encoded"},
                    },
                    "state": [],
                    "logits": 0,
                },
                "vocab_size": cfg.vocab_size,
                // No step: each token, the whole prefix through a function.
                "inputs": {},
                "tokenizer": tokenizer,
                "stop_ids": [cfg.eos_token_id],
            },
            "tokenizer": tokenizer,
            "prompt_ids": [cfg.decoder_start_token_id],
            "stop_ids": [cfg.eos_token_id],
            // The runner decodes to the decoder's positions.
            "max_tokens": max_len - 1,
        }),
    );
    pkg.model["task"] = json!("speech-recognition");
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("moonshine", "speech-recognition", export_rlxp).claims(&["moonshine"]),
    );
}
