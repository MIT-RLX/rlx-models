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

//! Export ChatterBox as an `.rlxp` package rlx runs alone: text and a
//! reference voice in, 24 kHz speech out (`Model::synthesize`), through the
//! one exporter ([`rlx_core::export`]).
//!
//! Its ONNX graphs (`onnx/`, the export-time reference) are converted to
//! rlx IR once, each symbolic in its lengths, imported as
//! [`NativeChatterBox`](crate::native::NativeChatterBox) imports them:
//! `speech_encoder` (the reference, in 960-sample units — one speech token
//! at 24 kHz), `embed_tokens`, `flow_encoder`, `cfm_estimator` and
//! `hift_head`. The T3 language model is rlx-llama32's graph over
//! `native/t3_lm.safetensors` (`inputs_embeds` in, the logits out, the
//! whole context each token). The program (`program/`) is the runner's
//! host logic: the reference resampled and padded, the prompt embeds, the
//! sampler, the CFM Euler solver with classifier-free guidance, HiFT's
//! n_fft-16 inverse STFT and the onset polish.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::Graph;
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use serde_json::json;

use crate::common::{
    CFG_RATE, ISTFT_HOP, ISTFT_NFFT, N_FLOW_STEPS, SAMPLE_RATE, SPEECH_VOCAB, START_SPEECH,
    START_TEXT, STOP_TEXT, SynthOpts,
};

/// The reference's unit: one speech token, 960 samples at 24 kHz.
pub const REFERENCE_UNIT: usize = 960;

/// Each function's symbols, apart (a function's derived symbols follow
/// its own).
const UNITS: u32 = 10;
const IDS: u32 = 20;
const TOKENS: u32 = 30;
const PROMPT_TOKENS: u32 = 31;
const PROMPT_FRAMES: u32 = 32;
const FRAMES: u32 = 40;
const MEL: u32 = 50;
const CONTEXT: u32 = 60;

pub type Params = RefCell<Option<HashMap<String, Vec<f32>>>>;

/// `onnx/{component}.onnx` at `seq` (and `named`'s lengths) as rlx IR, as
/// the runner imports it; its params named `component.` + theirs, kept in
/// `held` once; its outputs named as the ONNX names them.
pub fn import(
    dir: &Path,
    component: &str,
    seq: usize,
    named: &[(&str, usize)],
    max_wav: usize,
    held: &Params,
) -> Result<Graph> {
    let path = dir.join("onnx").join(format!("{component}.onnx"));
    let mut named_lengths: HashMap<String, usize> =
        named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    named_lengths.entry("sequence_length".into()).or_insert(seq);
    named_lengths
        .entry("num_speech_tokens".into())
        .or_insert(seq);
    named_lengths.entry("batch_size".into()).or_insert(1);
    named_lengths
        .entry("past_sequence_length".into())
        .or_insert(0);
    named_lengths
        .entry("total_sequence_length".into())
        .or_insert(seq);
    let opts = ImportOptions {
        sequence_length: seq,
        named_lengths,
        max_waveform_samples: max_wav,
        strict: false,
        decompose_conv_transpose: true,
        ..Default::default()
    };
    let (hir, params, report, manifest) =
        build_hir_from_onnx_file(&path, opts).with_context(|| format!("import {component}"))?;
    anyhow::ensure!(
        report.stubbed == 0 && report.unsupported.is_empty(),
        "{component}: {} ops stubbed ({:?}), unsupported {:?}",
        report.stubbed,
        report.stubbed_nodes,
        report.unsupported
    );
    let g = hir
        .lower_to_mir()
        .map_err(|e| anyhow::anyhow!("lower {component}: {e:?}"))?
        .into_graph();
    // A param of one value throughout (a `ConstantOfShape`'s fill) a
    // constant, as `import_rlx_graph` bakes them: one as long as a length
    // is then the fit's to make symbolic.
    let uniform = |data: &[f32]| data.first().is_some_and(|f| data.iter().all(|x| x == f));
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let rlx_ir::Op::Param { name } = &node.op else {
            return Ok(None);
        };
        Ok(Some(match params.get(name) {
            Some(d) if uniform(d) => out.add_node(
                rlx_ir::Op::Constant {
                    data: d.iter().flat_map(|x| x.to_le_bytes()).collect(),
                },
                vec![],
                node.shape.clone(),
            ),
            _ => out.param(format!("{component}.{name}"), node.shape.clone()),
        }))
    })
    .map_err(anyhow::Error::msg)?;
    rlx_ir::dynamic::sync_graph_shapes(&mut g);
    let outs = g.outputs.clone();
    for (id, o) in outs.iter().zip(&manifest.outputs) {
        g.node_mut(*id).name = Some(o.name.clone());
    }
    g.name = component.into();
    held.borrow_mut().get_or_insert_with(|| {
        params
            .into_iter()
            .filter(|(_, v)| !uniform(v))
            .map(|(k, v)| (format!("{component}.{k}"), v))
            .collect()
    });
    Ok(g)
}

/// `component` symbolic in `syms` (each with its builds' sizes), its
/// params and the tables the fit lifted.
fn lifted(
    syms: &[(u32, Vec<usize>)],
    longest: &[usize],
    build: impl Fn(&[usize], &Params) -> Result<Graph>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let held = RefCell::new(None);
    let (g, tables) = symbolic_in_lifted_many(syms, longest, |n| build(n, &held))?;
    let mut params = held.take().context("no build ran")?;
    params.extend(tables);
    Ok((g, params))
}

/// The ONNX components, each symbolic in its lengths, and their params.
pub fn onnx_functions(dir: &Path) -> Result<Vec<(Graph, HashMap<String, Vec<f32>>)>> {
    let mut out = Vec::new();
    // speech_encoder: `units` × 960 samples, 51 to 100 (2 to 4 s) — its
    // import lowers a reduction differently at exactly 50, and past 100 its
    // speaker branch pads the features to a multiple of 200 frames (and the
    // tokenizer stops at 6.25 s): a graph per 100 units, not one symbolic.
    out.push(lifted(&[(UNITS, vec![52, 54, 56])], &[1024], |n, h| {
        import(dir, "speech_encoder", 100, &[], n[0] * REFERENCE_UNIT, h)
    })?);
    // embed_tokens: any number of ids.
    out.push(lifted(&[(IDS, vec![5, 7, 9])], &[4096], |n, h| {
        import(dir, "embed_tokens", n[0], &[], 48_000, h)
    })?);
    // flow_encoder: symbolic in all the tokens (the prompt's and the
    // generated, `tokens`), the prompt's (`prompt_tokens`) and the prompt's
    // mel frames (`prompt_frames`, about twice its tokens) — its attention's
    // relative positions `2 · tokens − 1` long, the generated
    // `tokens − prompt_tokens`. The builds step them independently (8, 5,
    // 11: no dim of one a whole multiple of another's), the generated never
    // as many as the prompt's (the import confuses equal lengths).
    out.push(lifted(
        &[
            (TOKENS, vec![9, 17, 11, 18, 14]),
            (PROMPT_TOKENS, vec![4, 9, 5, 7, 6]),
            (PROMPT_FRAMES, vec![8, 19, 9, 15, 13]),
        ],
        &[4096, 4096, 8192],
        |n, h| {
            let (total, np, lp) = (n[0], n[1], n[2]);
            let nt = total - np;
            import(
                dir,
                "flow_encoder",
                nt,
                &[("Np", np), ("Nt", nt), ("Lp", lp)],
                0,
                h,
            )
        },
    )?);
    // cfm_estimator and hift_head: the mel frames.
    out.push(lifted(&[(FRAMES, vec![10, 14, 18])], &[8192], |n, h| {
        import(dir, "cfm_estimator", n[0], &[("T", n[0])], 0, h)
    })?);
    out.push(lifted(&[(MEL, vec![10, 14, 18])], &[8192], |n, h| {
        import(dir, "hift_head", n[0], &[("T", n[0])], 0, h)
    })?);
    Ok(out)
}

/// The longest context the T3 graph's RoPE tables reach.
const MAX_CONTEXT: usize = 4096;

/// T3: rlx-llama32's prefill over `native/t3_lm.safetensors`, symbolic in
/// the context (`inputs_embeds` `[1, context, 1024]` → `logits`), its RoPE
/// tables as long as [`MAX_CONTEXT`]; and its params.
pub fn t3_function(dir: &Path) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let native = dir.join("native");
    let cfg = Llama32Config::from_file(&native.join("t3_config.json"))
        .context("load native/t3_config.json")?;
    let st = native.join("t3_lm.safetensors");
    let build = |n: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let mut wm = WeightMap::from_file(st.to_str().context("t3_lm path")?)
            .with_context(|| format!("load {}", st.display()))?;
        let built = Llama32Flow::new(&cfg)
            .prefill()
            .batch(1)
            .seq(n)
            .inputs_embeds()
            .lm_head()
            .build(&mut wm)
            .context("build T3's prefill")?;
        let (mut g, params) = rlx_core::flow_util::graph_from_built(built)?;
        let out = g.outputs[0];
        g.node_mut(out).name = Some("logits".into());
        g.name = "t3".into();
        Ok((g, params))
    };
    let t3 = symbolic_in(CONTEXT, (5, 7, 3), |n| Ok(build(n)?.0))?;
    let (at, mut params) = build(3)?;
    let inv_freq = rlx_llama32::rope::resolve_inv_freq(&cfg, None);
    let half = inv_freq.len();
    let (cos, sin) = rlx_llama32::rope::build_rope_tables(&inv_freq, MAX_CONTEXT);
    let t3 = broadcast_params(
        &t3,
        &at,
        &mut params,
        &[("rope.cos", &cos, half), ("rope.sin", &sin, half)],
    )?;
    Ok((t3, params))
}

/// A WAV's mono samples (the channels averaged) and its rate.
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = raw
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

/// What `speech_encoder` makes of `reference` (at `sr`), as the runner
/// conditions on it — resampled to 24 kHz, at least 2 s — each output's
/// name, shape and little-endian bytes, the graph imported at its length.
fn conditioning(
    dir: &Path,
    reference: &[f32],
    sr: u32,
) -> Result<Vec<(String, Vec<usize>, Vec<u8>)>> {
    let mut pcm = crate::common::resample(reference, sr, SAMPLE_RATE);
    if pcm.len() < 48_000 {
        pcm.resize(48_000, 0.0);
    }
    let held = RefCell::new(None);
    let g = import(dir, "speech_encoder", 100, &[], pcm.len(), &held)?;
    let names = [
        "audio_features",
        "audio_tokens",
        "speaker_embeddings",
        "speaker_features",
    ]
    .map(String::from);
    let shapes: Vec<Vec<usize>> = g
        .outputs
        .iter()
        .map(|&o| {
            g.node(o)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect()
        })
        .collect();
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    for (n, d) in held.take().context("the encoder's params")? {
        c.set_param(&n, &d);
    }
    c.finalize_params();
    let bytes: Vec<u8> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
    let outs = c.run_typed(&[("audio_values", &bytes, rlx_ir::DType::F32)]);
    Ok(names
        .into_iter()
        .zip(outs)
        .zip(shapes)
        .map(|((n, (b, _)), s)| (n, s, b))
        .collect())
}

/// Export the ChatterBox directory `dir` (`tokenizer.json`, `onnx/`,
/// `native/t3_lm.safetensors` with its config, `default_voice.wav`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let mut pkg =
        Package::for_checkpoint(&ck, opts, "chatterbox", "text-to-speech", "rlx-chatterbox");
    let mut held = HashMap::new();
    for (g, params) in onnx_functions(dir)? {
        pkg.function(g);
        held.extend(params);
    }
    let (t3, t3_params) = t3_function(dir)?;
    pkg.function(t3);
    pkg.dims = vec![
        ("units".into(), UNITS),
        ("ids".into(), IDS),
        ("tokens".into(), TOKENS),
        ("prompt_tokens".into(), PROMPT_TOKENS),
        ("prompt_frames".into(), PROMPT_FRAMES),
        ("frames".into(), FRAMES),
        ("mel".into(), MEL),
        ("context".into(), CONTEXT),
    ];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = dims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    // T3's weights referenced from its safetensors, its tables held.
    let st = dir.join("native/t3_lm.safetensors");
    let source = WeightMap::from_file(st.to_str().context("t3_lm path")?)?;
    pkg.reference_map(&t3_params, &source, "safetensors", &st, out, false)?;

    let tokenizer = pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(dir.join("tokenizer.json"))?,
    );
    // The default voice's conditioning, the encoder run by rlx at its own
    // length (any length: a graph made for it).
    let (voice, voice_sr) = read_wav(&dir.join("default_voice.wav"))?;
    let cond = conditioning(dir, &voice, voice_sr)?;
    let mut shapes = serde_json::Map::new();
    for (name, shape, bytes) in cond {
        pkg.sidecar(
            &format!("voices/default/{name}"),
            "application/octet-stream",
            bytes,
        );
        shapes.insert(name, json!(shape));
    }
    let o = SynthOpts::default();
    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "tokenizer": tokenizer,
            "voice": "/package/voices/default",
            "voice_shapes": shapes,
            "reference_unit": REFERENCE_UNIT,
            "start_text": START_TEXT,
            "stop_text": STOP_TEXT,
            "start_speech": START_SPEECH,
            "speech_vocab": SPEECH_VOCAB,
            "hidden": 1024,
            "exaggeration": o.exaggeration,
            "temperature": o.temperature,
            "top_p": o.top_p,
            "top_k": o.top_k,
            "repetition_penalty": o.repetition_penalty,
            "max_frames": o.max_frames,
            "seed": o.seed,
            "greedy": o.greedy,
            "cfg_rate": CFG_RATE,
            "flow_steps": N_FLOW_STEPS,
            "istft_nfft": ISTFT_NFFT,
            "istft_hop": ISTFT_HOP,
            "sample_rate": SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("chatterbox", "text-to-speech", export_rlxp));
}
