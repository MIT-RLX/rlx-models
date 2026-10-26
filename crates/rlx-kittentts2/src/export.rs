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

//! Export KittenTTS 2 as an `.rlxp` package rlx runs alone, through the one
//! exporter ([`rlx_core::export`]).
//!
//! The speech LM is the `Q2_0` GGUF's language-model package
//! ([`rlx_qwen3::export::export_gguf_media`]): its `generate` entry, a
//! prompt's first row the speaker's — a medium (`speaker`, its placeholder
//! the start token) whose stage is `spk_proj` (Linear, LayerNorm) and whose
//! prefill takes embedded rows (`prefill_embeds`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::decoder::{MediaParts, prompt_function};
use rlx_core::export::program;
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::{DType, Graph, GraphExt, HirGraphExt, Op, Shape};
use rlx_qwen3::{Qwen3Config, Qwen3PrefillOpts, build_qwen3_prefill_embeds_built};
use serde_json::json;

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::config::Kitten2Config;

/// The embedding graph's and the media prefill's symbols, past the step's.
const EMBED_N: u32 = 10;
const MEDIA_SEQ: u32 = 11;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `input_ids` `[n]` → their rows of the embedding table.
fn embed_graph(cfg: &Qwen3Config, n: usize) -> Graph {
    let mut g = Graph::new("embed");
    let ids = g.input("input_ids", Shape::new(&[n], DType::I32));
    let table = g.param(
        "model.embed_tokens.weight",
        f32s(&[cfg.vocab_size, cfg.hidden_size]),
    );
    let rows = g.gather(table, ids, 0, f32s(&[n, cfg.hidden_size]));
    g.set_outputs(vec![rows]);
    g
}

/// `speaker` `[spk_dim]` → `LayerNorm(W·e + b)` `[1, hidden]`: the row
/// standing for the prompt's start token.
fn spk_proj_graph(hidden: usize, spk_dim: usize) -> Graph {
    let mut g = Graph::new("spk_proj");
    let e = g.input("speaker", f32s(&[spk_dim]));
    let e = g.reshape_(e, vec![1, spk_dim as i64]);
    let w = g.param("spk_proj.0.weight", f32s(&[hidden, spk_dim]));
    let b = g.param("spk_proj.0.bias", f32s(&[hidden]));
    let gamma = g.param("spk_proj.1.weight", f32s(&[hidden]));
    let beta = g.param("spk_proj.1.bias", f32s(&[hidden]));
    let wt = g.transpose_(w, vec![1, 0]);
    let y = g.mm(e, wt);
    let y = g.add(y, b);
    let y = g.ln(y, gamma, beta, 1e-5);
    g.node_mut(y).name = Some("speaker_embeds".into());
    g.set_outputs(vec![y]);
    g
}

/// The prefill over embedded rows at `seq`: `inputs_embeds` → the last
/// row's logits, then each layer's K and V.
fn prefill_graph(
    cfg: &Qwen3Config,
    wm: &mut dyn WeightLoader,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let opts = Qwen3PrefillOpts {
        with_lm_head: true,
        with_kv_outputs: true,
        last_logits_only: true,
        ..Qwen3PrefillOpts::static_prefill(1, seq)
    };
    let built = build_qwen3_prefill_embeds_built(cfg, wm, &opts)?;
    rlx_core::flow_util::graph_from_built(built)
}

/// The params named `names` of `g` cut to their first `rows` rows of
/// `half` (RoPE tables spanning every position the model knows).
fn cut_tables(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    names: &[&str],
    rows: usize,
    half: usize,
) {
    for name in names {
        let Some(data) = params.get_mut(*name) else {
            continue;
        };
        if data.len() <= rows * half {
            continue;
        }
        data.truncate(rows * half);
        let ids: Vec<_> = g
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::Param { name: p } if p == name))
            .map(|n| n.id)
            .collect();
        for id in ids {
            let node = g.node_mut(id);
            node.shape = Shape::new(&[rows, half], node.shape.dtype());
        }
    }
}

/// The speaker medium: its stage, the embedding and the prefill.
fn speaker_media(
    cfg: &Qwen3Config,
    spk: HashMap<String, (Vec<f32>, Vec<usize>)>,
    start_id: u32,
    weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
    rows: usize,
    take: &mut dyn FnMut(HashMap<String, Vec<f32>>) -> Result<()>,
) -> Result<Option<MediaParts>> {
    let (hidden, spk_dim) = match spk.get("0.weight") {
        Some((_, s)) if s.len() == 2 => (s[0], s[1]),
        _ => anyhow::bail!("spk_proj.0.weight is not a matrix"),
    };
    let embed = rlx_ir::dynamic::generalize(
        &embed_graph(cfg, 37),
        &embed_graph(cfg, 53),
        &[(EMBED_N, 37, 53)],
    )
    .map_err(anyhow::Error::msg)?;
    let half = cfg.head_dim / 2;
    let (mut prefill, mut kept) = prompt_function(
        |seq| {
            let mut wm = weights()?;
            prefill_graph(cfg, wm.as_mut(), seq)
        },
        MEDIA_SEQ,
        &["rope.cos".into(), "rope.sin".into()],
        &mut |p| take(p),
    )
    .map_err(anyhow::Error::msg)?;
    prefill.name = "prefill_embeds".into();
    cut_tables(
        &mut prefill,
        &mut kept,
        &["rope.cos", "rope.sin"],
        rows,
        half,
    );
    take(kept)?;
    take(
        spk.into_iter()
            .map(|(k, (v, _))| (format!("spk_proj.{k}"), v))
            .collect(),
    )?;
    let mut outputs = vec!["logits".to_string()];
    for i in 0..cfg.num_hidden_layers {
        outputs.push(format!("past_k_{i}"));
        outputs.push(format!("past_v_{i}"));
    }
    let sizes: Vec<usize> = (6..)
        .map(|p| 1usize << p)
        .take_while(|&s| s < rows)
        .collect();
    let entry = json!({
        "embed": {"function": "embed", "inputs": {"input_ids": "ids"}, "outputs": ["embeds"]},
        "prefill": {"function": "prefill_embeds",
                    "inputs": {"inputs_embeds": "prompt_embeds", "last": "last"},
                    "outputs": outputs, "sizes": sizes},
        "kinds": {"speaker": {
            "token": start_id,
            "input": "speaker",
            "stages": [{"function": "spk_proj", "inputs": {"speaker": "speaker"},
                        "outputs": ["speaker_embeds"]}],
            "output": "speaker_embeds",
            "contract": {"kind": "tensor", "dtype": "f32", "shape": [spk_dim]},
        }},
    });
    Ok(Some(MediaParts {
        graphs: vec![embed, prefill, spk_proj_graph(hidden, spk_dim)],
        dims: vec![("ids".into(), EMBED_N), ("prompt".into(), MEDIA_SEQ)],
        entry,
    }))
}

/// Export the KittenTTS 2 repository `repo` (`config.json`, `lm/`, its
/// ternary checkpoint, transcoded to `Q2_0` once) as the package `out`.
pub fn export_rlxp(repo: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = Kitten2Config::from_dir(repo)?;
    let lm_dir = repo.join(&cfg.lm_dir);
    let ternary = repo.join(
        cfg.lm_packed
            .as_deref()
            .context("config declares no lm_packed checkpoint")?,
    );
    let qcfg = Qwen3Config::from_file(&lm_dir.join("config.json"))?;
    let gguf = crate::lm::ensure_q2_0_gguf(&lm_dir, &ternary, &qcfg)?;
    let spk = crate::ternary::read_spk_proj(&ternary)?;
    let start_id = cfg.token_map.start_id;
    let media: rlx_qwen3::export::MediaBuilder = Box::new(move |weights, rows, take| {
        speaker_media(&qcfg, spk.clone(), start_id, weights, rows, take)
    });
    let s3_path = crate::pipeline::resolve_s3gen(repo)?;
    let s3 = crate::s3gen::S3Gen::from_weights(
        std::sync::Arc::new(crate::s3gen::nn::S3GenWeights::open(&s3_path)?),
        rlx_runtime::Device::Cpu,
    );
    let voices = crate::voices::VoiceLibrary::open(
        &repo.join(&cfg.voices),
        &repo.join(
            cfg.s3gen_voices_path()
                .context("config declares no S3Gen voice conditioning")?,
        ),
    )?;
    rlx_qwen3::export::export_gguf_media(&gguf, out, opts, None, Some(media), &mut |pkg| {
        pkg.model["family"] = json!("kittentts2");
        pkg.model["task"] = json!("text-to-speech");
        let w = s3.weights();

        // S3Gen: the encoder, the flow's estimator, HiFT's f0 and decoder.
        let mut held = HashMap::new();
        for (g, p) in s3_functions(w)? {
            pkg.function(g);
            held.extend(p);
        }
        pkg.dims.extend([
            ("s3_tokens".to_string(), S3_TOKENS),
            ("s3_frames".to_string(), S3_FRAMES),
        ]);
        let dims = pkg.param_dims();
        let mut names: Vec<&String> = held.keys().collect();
        names.sort();
        for n in names {
            let d = dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
            pkg.hold_f32(n.clone(), d, &held[n]);
        }

        // The flow's time conditioning for each of its steps.
        let steps = crate::s3gen::CFM_STEPS;
        let temb: Vec<Vec<f32>> = (0..steps)
            .map(|k| {
                crate::s3gen::estimator::time_embedding(
                    w,
                    k as f32 / steps as f32,
                    (k + 1) as f32 / steps as f32,
                )
            })
            .collect::<Result<_>>()?;

        // Each built-in voice: its prompt for the LM, its conditioning for
        // the flow (the speaker's affine applied).
        let mut voice_names = Vec::new();
        for name in voices.names().map(String::from).collect::<Vec<_>>() {
            let v = voices.load(&name)?;
            let body = json!({
                "transcript": v.transcript,
                "reference_tokens": v.reference_tokens,
                "speaker_embedding": v.speaker_embedding,
                "prompt_token": v.s3.prompt_token,
                "prompt_feat": v.s3.prompt_feat,
                "prompt_frames": v.s3.prompt_frames,
                "spks": s3.speaker(&v.s3.xvector)?,
            });
            pkg.sidecar(
                &format!("voices/{name}.json"),
                "application/json",
                serde_json::to_vec(&body)?,
            );
            voice_names.push(name);
        }

        let module = program::build_for(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
            "wasm32-wasip1",
        )?;
        let g = &cfg.generation;
        program::entry(
            pkg,
            "speak",
            module,
            &["synthesize", "run"],
            "text-to-speech",
            json!({
                "token_map": token_map_json(&cfg.token_map),
                "repetition_window": g.repetition_window,
                "use_reference_prompt": g.use_reference_prompt,
                "emotion_control": g.emotion_control,
                "chunk_chars": g.chunk_chars,
                "chunk_min_chars": g.chunk_min_chars,
                "presets": cfg.decode_presets.iter().map(|(k, p)| (k.clone(), json!({
                    "temperature": p.temperature, "top_p": p.top_p,
                    "top_k": p.top_k, "min_p": p.min_p,
                }))).collect::<serde_json::Map<_, _>>(),
                "default_preset": cfg.default_preset,
                "default_voice": cfg.default_voice,
                "voices": voice_names,
                "sample_rate": cfg.sample_rate,
                "temb": temb,
                "source": {
                    "weight": w.get("mel2wav.m_source.l_linear.weight")?,
                    "bias": w.get("mel2wav.m_source.l_linear.bias")?[0],
                },
            }),
        );
        pkg.metadata("sample_rate", json!(cfg.sample_rate));
        pkg.metadata("voices", json!(voice_names));
        Ok(())
    })
}

fn token_map_json(tm: &crate::config::TokenMap) -> serde_json::Value {
    json!({
        "audio_id_base": tm.audio_id_base,
        "num_audio_tokens": tm.num_audio_tokens,
        "speech_start_id": tm.speech_start_id,
        "speech_end_id": tm.speech_end_id,
        "text_start_id": tm.text_start_id,
        "text_end_id": tm.text_end_id,
        "start_id": tm.start_id,
        "stop_id": tm.stop_id,
        "final_seg_id": tm.final_seg_id,
        "reference_speech_start_id": tm.reference_speech_start_id,
        "reference_speech_end_id": tm.reference_speech_end_id,
        "reference_text_start_id": tm.reference_text_start_id,
        "reference_text_end_id": tm.reference_text_end_id,
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("kittentts2", "text-to-speech", export_rlxp));
}

/// The longest token run the encoder's relative positions reach.
const REL_POS: usize = 4096;

/// S3Gen's stages' symbols.
const S3_TOKENS: u32 = 20;
const S3_FRAMES: u32 = 21;

/// An S3Gen stage built by `build` (on a fresh `Ctx`) at a length, lowered
/// to a graph named `name`, with its params.
fn s3_stage(
    name: &str,
    build: impl FnOnce(&mut crate::s3gen::nn::Ctx) -> Result<rlx_ir::hir::HirNodeId>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut hir = rlx_ir::hir::HirModule::new(name);
    let mut g = rlx_ir::hir::HirMut::new(&mut hir);
    let mut ctx = crate::s3gen::nn::Ctx::new(&mut g);
    let out = build(&mut ctx)?;
    let mut params: HashMap<String, Vec<f32>> = ctx.into_params().into_iter().collect();
    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    // The tables the builder computes (`p<n>`: relative positions, a
    // source's windows) constants, not weights: one as long as a length is
    // then the fit's to make symbolic.
    let computed = |p: &str| {
        p.strip_prefix('p')
            .is_some_and(|n| n.parse::<usize>().is_ok())
    };
    let mut graph = rlx_ir::stateful::rebuild(&graph, |out, _, node, _| {
        let Op::Param { name: p } = &node.op else {
            return Ok(None);
        };
        if !computed(p) {
            return Ok(None);
        }
        let data = params.get(p).ok_or_else(|| format!("no data for {p}"))?;
        Ok(Some(out.add_node(
            Op::Constant {
                data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            },
            vec![],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)?;
    params.retain(|k, _| !computed(k));
    graph.name = name.into();
    Ok((graph, params))
}

/// S3Gen's four stages, each at the length `n` (tokens or mel frames).
pub fn s3_build(
    w: &crate::s3gen::nn::S3GenWeights,
    stage: &str,
    n: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use crate::s3gen::{encoder, estimator, hift};
    const MEL: usize = 80;
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    match stage {
        "s3_encoder" => {
            let d = w.shape("flow.input_embedding.weight")?[1];
            let vocab = w.shape("flow.input_embedding.weight")?[0];
            let (g, mut params) = s3_stage(stage, |ctx| {
                // The tokens' rows of the input embedding, then the encoder.
                let tokens = ctx.g.input("tokens", Shape::new(&[n], DType::I32));
                let table = ctx.param("flow.input_embedding.weight", &[vocab, d], || {
                    w.get("flow.input_embedding.weight").unwrap().to_vec()
                });
                let x = ctx.g.gather_(table, tokens, 0);
                encoder::build(ctx, w, x)
            })?;
            // Its relative positions (`[2m − 1, d]`, offsets m − 1 … 1 − m,
            // for the tokens and, past the upsampling, twice as many) the
            // middle rows of one table as long as the package reaches.
            let bytes = |m: usize| -> Vec<u8> {
                encoder::rel_pos_table(m, d)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect()
            };
            let tables = [(n, bytes(n)), (2 * n, bytes(2 * n))];
            let mut found = 0;
            // One param node, however many tables read it.
            let mut table_node = None;
            let g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
                let Op::Constant { data } = &node.op else {
                    return Ok(None);
                };
                let Some(&(m, _)) = tables.iter().find(|(_, b)| b == data) else {
                    return Ok(None);
                };
                found += 1;
                // The table's own layout (`[2m − 1, d]`, or with unit axes).
                let dims: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|x| x.unwrap_static())
                    .collect();
                let axis = dims
                    .iter()
                    .position(|&x| x == 2 * m - 1)
                    .ok_or("the relative positions have no axis as long")?;
                let mut long = dims.clone();
                long[axis] = 2 * REL_POS - 1;
                let all =
                    *table_node.get_or_insert_with(|| out.param("s3_encoder.rel_pos", f32s(&long)));
                Ok(Some(out.add_node(
                    Op::Narrow {
                        axis,
                        start: REL_POS - m,
                        len: 2 * m - 1,
                    },
                    vec![all],
                    node.shape.clone(),
                )))
            })
            .map_err(anyhow::Error::msg)?;
            anyhow::ensure!(found > 0, "the encoder's relative positions not found");
            params.insert(
                "s3_encoder.rel_pos".into(),
                encoder::rel_pos_table(REL_POS, d),
            );
            Ok((g, params))
        }
        "s3_estimator" => s3_stage(stage, |ctx| {
            let s = |c: usize| f32s(&[1, c, n, 1]);
            let x = ctx.g.input("x", s(MEL));
            let mu = ctx.g.input("mu", s(MEL));
            let cond = ctx.g.input("cond", s(MEL));
            let spks = ctx.g.input("spks", f32s(&[1, MEL, 1, 1]));
            let temb = ctx.g.input("temb", f32s(&[1, estimator::time_dim(w)?]));
            estimator::build(ctx, w, x, mu, spks, cond, temb)
        }),
        "hift_f0" => s3_stage(stage, |ctx| {
            let mel = ctx.g.input("mel", f32s(&[1, MEL, n, 1]));
            hift::build_f0(ctx, w, mel)
        }),
        "hift_decode" => s3_stage(stage, |ctx| {
            let mel = ctx.g.input("mel", f32s(&[1, MEL, n, 1]));
            let frames = n * hift::UPSAMPLE / hift::HOP + 1;
            let s = ctx
                .g
                .input("s_stft", f32s(&[1, 2 * (hift::N_FFT / 2 + 1), frames, 1]));
            hift::build_decode(ctx, w, mel, s)
        }),
        other => anyhow::bail!("no S3Gen stage {other}"),
    }
}

/// S3Gen's stages symbolic in their lengths, and their params.
pub fn s3_functions(
    w: &crate::s3gen::nn::S3GenWeights,
) -> Result<Vec<(Graph, HashMap<String, Vec<f32>>)>> {
    let mut out = Vec::new();
    for (stage, sym, sizes) in [
        ("s3_encoder", S3_TOKENS, vec![37usize, 46, 61, 53]),
        ("s3_estimator", S3_FRAMES, vec![74, 92, 122, 106]),
        ("hift_f0", S3_FRAMES, vec![74, 92, 122, 106]),
        ("hift_decode", S3_FRAMES, vec![74, 92, 122, 106]),
    ] {
        let held = std::cell::RefCell::new(None);
        let (g, tables) =
            rlx_core::export::graph::symbolic_in_lifted_many(&[(sym, sizes)], &[8192], |n| {
                let (g, p) = s3_build(w, stage, n[0])?;
                held.borrow_mut().get_or_insert(p);
                Ok(g)
            })
            .with_context(|| format!("{stage} symbolic"))?;
        let mut params = held.take().context("no build")?;
        params.extend(tables);
        out.push((g, params));
    }
    Ok(out)
}
