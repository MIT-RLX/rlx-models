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

//! Export Nemotron 3.5 ASR (a cache-aware FastConformer RNN-T `.nemo`) as an
//! `.rlxp` package rlx runs alone: `Model.transcribe(audio)`.
//!
//! Its graphs are the runner's: `features` (NeMo's frontend from the
//! checkpoint's own window and filterbank, zero-padded as NeMo pads,
//! [`rlx_core::export::audio::nemo_features_with`]); `encoder` (the
//! FastConformer, symbolic in the mel frames: its relative positions and
//! its `chunked_limited` mask read from tables); `enc_proj` (the language
//! prompt kernel and the joint's encoder projection over every frame);
//! `pred_step` (the prediction LSTMs on one label) and `joint` (one frame's
//! logits and their argmax). The program (`program/`) runs the greedy
//! RNN-T search the runner runs and reads the SentencePiece pieces.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::export::audio::{
    NemoFrontend, NemoPadding, SAMPLES, audio_iota, nemo_features_with, stft_bases,
};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_nemo::NemoModel;
use serde_json::json;

use crate::config::AsrConfig;
use crate::encoder::{EncoderPart, build_encoder_part_hir, chunked_limited_bias, rel_pos_encoding};
use crate::mel::Frontend;
use crate::tokenizer::SpmTokenizer;
use crate::weights::{NemoWeights, keys};

/// The symbols: the audio's samples, the mel frames, the encoder frames.
const MEL: u32 = 1;
const ENC: u32 = 2;
/// The longest encoder output the tables cover (~82 s).
const MAX_ENC: usize = 1024;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, v: f32, dims: &[usize]) -> NodeId {
    let one = g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&vec![1; dims.len()]),
    );
    g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![one],
        f32s(dims),
    )
}

fn name(g: &mut Graph, id: NodeId, n: &str) {
    g.node_mut(id).name = Some(n.into());
}

type Built = (Graph, HashMap<String, Vec<f32>>);

/// The encoder's `part` at `n` frames (mel frames for the subsampling,
/// encoder frames for the layers): [`build_encoder_part_hir`]'s graph, its
/// relative positions and attention mask rows of tables (`enc.rel_pos`
/// `[1, 2·MAX_ENC − 1, d]`, `enc.att_bias` `[MAX_ENC, MAX_ENC]`), its zero
/// pads constants.
fn encoder(cfg: &AsrConfig, model: &NemoModel, n: usize, part: EncoderPart) -> Result<Built> {
    let mut w = NemoWeights::new(model);
    let (hir, params, t) = build_encoder_part_hir(cfg, &mut w, n, part)?;
    let built = rlx_core::flow_util::built_from_hir(hir, params)?;
    let (g, mut params) = rlx_core::flow_util::graph_from_built(built)?;
    let d = cfg.d_model;
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| {
        let Op::Param { name } = &node.op else {
            return Ok(None);
        };
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|x| x.unwrap_static())
            .collect();
        if name == "_enc.rel_pos_encoding" {
            let table = out.param("enc.rel_pos", f32s(&[1, 2 * MAX_ENC - 1, d]));
            return Ok(Some(out.narrow_(table, 1, MAX_ENC - t, 2 * t - 1)));
        }
        if name == "_enc.att_context_bias" {
            let table = out.param("enc.att_bias", f32s(&[MAX_ENC, MAX_ENC]));
            let rows = out.narrow_(table, 0, 0, t);
            let block = out.narrow_(rows, 1, 0, t);
            return Ok(Some(out.reshape_(block, vec![1, 1, t as i64, t as i64])));
        }
        match params.get(name) {
            Some(v) if v.iter().all(|x| *x == 0.0) && v.len() > 1 => {
                Ok(Some(constant(out, 0.0, &dims)))
            }
            _ => Ok(None),
        }
    })
    .map_err(anyhow::Error::msg)?;
    params.retain(|n, v| {
        n != "_enc.rel_pos_encoding"
            && n != "_enc.att_context_bias"
            && !(v.len() > 1 && v.iter().all(|x| *x == 0.0))
    });
    let out = g.outputs[0];
    let (output, function) = match part {
        EncoderPart::Subsampling => ("x", "subsampling"),
        _ => ("enc", "encoder"),
    };
    name(&mut g, out, output);
    g.name = function.into();
    Ok((g, params))
}

/// Whether the model reads a language one-hot (a prompt kernel, or a joint
/// whose encoder input is wider than the encoder).
fn uses_language(cfg: &AsrConfig, shapes: &HashMap<&str, Vec<usize>>) -> bool {
    shapes.contains_key(keys::PROMPT_K0_W) || shapes[keys::JOINT_ENC_W][1] > cfg.d_model
}

/// `enc_proj` at `t` encoder frames: `enc` `[1, t, d]` and the language
/// one-hot `lang` `[1, langs]` → `proj` `[t, joint_hidden]` — the prompt
/// kernel's `relu(W0 · [enc ‖ lang] + b0)` through `W2`, then the joint's
/// encoder projection.
fn enc_proj(cfg: &AsrConfig, shapes: &HashMap<&str, Vec<usize>>, t: usize) -> Graph {
    let d = cfg.d_model;
    let langs = cfg.num_languages;
    let mut g = Graph::new("enc_proj");
    let enc = g.input("enc", f32s(&[1, t, d]));
    let enc = g.reshape_(enc, vec![t as i64, d as i64]);
    let lang_rows = uses_language(cfg, shapes).then(|| {
        let lang = g.input("lang", f32s(&[1, langs]));
        g.add_node(
            Op::Expand {
                target_shape: vec![t as i64, langs as i64],
            },
            vec![lang],
            f32s(&[t, langs]),
        )
    });
    let linear = |g: &mut Graph, x: NodeId, w: &str, b: &str| -> NodeId {
        let ws = &shapes[w];
        let (o, i) = (ws[0], ws[1]);
        let p = g.param(w, f32s(&[o, i]));
        let p = g.transpose_(p, vec![1, 0]);
        let y = g.mm(x, p);
        let bp = g.param(b, f32s(&[o]));
        let bp = g.reshape_(bp, vec![1, o as i64]);
        let bp = g.add_node(
            Op::Expand {
                target_shape: vec![t as i64, o as i64],
            },
            vec![bp],
            f32s(&[t, o]),
        );
        g.add(y, bp)
    };
    let mut x = enc;
    if let (true, Some(lang_rows)) = (shapes.contains_key(keys::PROMPT_K0_W), lang_rows) {
        let cat = g.concat_(vec![enc, lang_rows], 1);
        let h = linear(&mut g, cat, keys::PROMPT_K0_W, keys::PROMPT_K0_B);
        let h = g.relu(h);
        x = linear(&mut g, h, keys::PROMPT_K2_W, keys::PROMPT_K2_B);
    }
    if let (true, Some(lang_rows)) = (shapes[keys::JOINT_ENC_W][1] > d, lang_rows) {
        x = g.concat_(vec![x, lang_rows], 1);
    }
    let proj = linear(&mut g, x, keys::JOINT_ENC_W, keys::JOINT_ENC_B);
    name(&mut g, proj, "proj");
    g.set_outputs(vec![proj]);
    g
}

/// `pred_step`: label `id` `[1]` i32 (`on` `[1, 1]` 0 for the start of
/// the sequence, which feeds zeros) through the prediction LSTMs from
/// `h{l}` / `c{l}` `[1, hidden]` → `out` (the top layer's hidden state) and
/// `h{l}.next` / `c{l}.next`.
fn pred_step(shapes: &HashMap<&str, Vec<usize>>, layers: usize) -> Graph {
    let mut g = Graph::new("pred_step");
    let es = &shapes[keys::PRED_EMBED];
    let id = g.input("id", Shape::new(&[1], DType::I32));
    let on = g.input("on", f32s(&[1, 1]));
    let table = g.param(keys::PRED_EMBED, f32s(&[es[0], es[1]]));
    let x = g.gather_(table, id, 0);
    let mut x = g.mul(x, on);
    let mut outs = vec![];
    for l in 0..layers {
        let ih = keys::pred_lstm("weight_ih", l);
        let hh = keys::pred_lstm("weight_hh", l);
        let (g4, input) = (shapes[ih.as_str()][0], shapes[ih.as_str()][1]);
        let hid = g4 / 4;
        let h = g.input(format!("h{l}"), f32s(&[1, hid]));
        let c = g.input(format!("c{l}"), f32s(&[1, hid]));
        let wih = g.param(&ih, f32s(&[g4, input]));
        let wih = g.transpose_(wih, vec![1, 0]);
        let whh = g.param(&hh, f32s(&[g4, hid]));
        let whh = g.transpose_(whh, vec![1, 0]);
        let bih = g.param(keys::pred_lstm("bias_ih", l), f32s(&[1, g4]));
        let bhh = g.param(keys::pred_lstm("bias_hh", l), f32s(&[1, g4]));
        let a = g.mm(x, wih);
        let b = g.mm(h, whh);
        let gates = g.add(bih, bhh);
        let gates = g.add(gates, a);
        let gates = g.add(gates, b);
        let gate = |g: &mut Graph, k: usize| g.narrow_(gates, 1, k * hid, hid);
        let (i, f, gg, o) = (
            gate(&mut g, 0),
            gate(&mut g, 1),
            gate(&mut g, 2),
            gate(&mut g, 3),
        );
        let i = g.sigmoid(i);
        let f = g.sigmoid(f);
        let gg = g.tanh(gg);
        let o = g.sigmoid(o);
        let fc = g.mul(f, c);
        let ig = g.mul(i, gg);
        let c2 = g.add(fc, ig);
        let tc = g.tanh(c2);
        let h2 = g.mul(o, tc);
        name(&mut g, h2, &format!("h{l}.next"));
        name(&mut g, c2, &format!("c{l}.next"));
        outs.push(h2);
        outs.push(c2);
        x = h2;
    }
    let out = g.reshape_(x, vec![1, x_width(&g, x) as i64]);
    name(&mut g, out, "out");
    let mut all = vec![out];
    all.extend(outs);
    g.set_outputs(all);
    g
}

fn x_width(g: &Graph, x: NodeId) -> usize {
    g.node(x).shape.dims()[1].unwrap_static()
}

/// `joint` at `t` frames: frame `at` `[1]` i32 of `proj` `[t, hidden]` and
/// the prediction `pred` `[1, pred_hidden]` → `label` `[1]`, the argmax of
/// the token logits of `joint_net(relu(proj[at] + W·pred + b))` (the last
/// token class is the blank); with `durations` TDT heads (its last outputs)
/// also `duration`, their argmax.
fn joint(shapes: &HashMap<&str, Vec<usize>>, durations: usize, t: usize) -> Graph {
    let ps = &shapes[keys::JOINT_PRED_W];
    let ns = &shapes[keys::JOINT_NET_W];
    let (hj, hp, outs) = (ps[0], ps[1], ns[0]);
    let classes = outs - durations;
    let mut g = Graph::new("joint");
    let proj = g.input("proj", f32s(&[t, hj]));
    let at = g.input("at", Shape::new(&[1], DType::I32));
    let pred = g.input("pred", f32s(&[1, hp]));
    let row = g.gather_(proj, at, 0);
    let w = g.param(keys::JOINT_PRED_W, f32s(&[hj, hp]));
    let w = g.transpose_(w, vec![1, 0]);
    let pp = g.mm(pred, w);
    let b = g.param(keys::JOINT_PRED_B, f32s(&[1, hj]));
    let pp = g.add(pp, b);
    let h = g.add(row, pp);
    let h = g.relu(h);
    let nw = g.param(keys::JOINT_NET_W, f32s(&[outs, hj]));
    let nw = g.transpose_(nw, vec![1, 0]);
    let logits = g.mm(h, nw);
    let nb = g.param(keys::JOINT_NET_B, f32s(&[1, outs]));
    let logits = g.add(logits, nb);
    let argmax = |g: &mut Graph, x: NodeId, n: &str| {
        let id = g.add_node(
            Op::ArgMax {
                axis: 1,
                keep_dim: false,
            },
            vec![x],
            f32s(&[1]),
        );
        name(g, id, n);
        id
    };
    let tokens = if durations > 0 {
        g.narrow_(logits, 1, 0, classes)
    } else {
        logits
    };
    let mut results = vec![argmax(&mut g, tokens, "label")];
    if durations > 0 {
        let d = g.narrow_(logits, 1, classes, durations);
        results.push(argmax(&mut g, d, "duration"));
    }
    g.set_outputs(results);
    g
}

/// A FastConformer transducer as [`transducer_rlxp`] exports it.
pub struct Transducer<'a> {
    /// The package's name and family.
    pub name: &'a str,
    /// The crate that exported.
    pub exported_by: &'a str,
    pub license: &'a str,
    /// TDT durations (frames a label advances, by the joint's duration head);
    /// empty for a plain RNN-T, whose label stays on its frame.
    pub durations: Vec<i32>,
    /// The search keeps the state the start-of-sequence step leaves (NeMo's
    /// greedy search) rather than starting the first label from zeros.
    pub sos_carry: bool,
}

/// Export the Nemotron ASR `.nemo` at `path` (beside it, the Hugging Face
/// `model.safetensors` of the same weights, which the package then
/// references) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    transducer_rlxp(
        path,
        out,
        opts,
        &Transducer {
            name: "nemotron-asr",
            exported_by: "rlx-nemotron-asr",
            license: "nvidia-open-model-license",
            durations: Vec::new(),
            sos_carry: false,
        },
    )
}

/// Export the FastConformer transducer `.nemo` at `path` (this crate's
/// encoder and prediction net; `spec` its search) as the package `out`.
pub fn transducer_rlxp(
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
    spec: &Transducer,
) -> Result<ExportReport> {
    let model = NemoModel::open(path).with_context(|| format!("open {}", path.display()))?;
    let cfg = AsrConfig::from_nemo(model.config())?;
    let fe = Frontend::from_model(&model, &cfg)
        .context("the frontend's filterbank and window (preprocessor.featurizer)")?;
    let tok = model
        .tokenizers()
        .iter()
        .find(|t| t.name.ends_with(".model"))
        .context("no SentencePiece model in the .nemo")
        .and_then(|t| SpmTokenizer::from_model_bytes(&t.bytes))?;
    let mut pkg = Package::new(
        spec.name,
        &opts.version,
        spec.name,
        "automatic-speech-recognition",
        spec.exported_by,
    );
    pkg.model["license"] = json!(spec.license);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    // The frontend.
    let mut window = vec![0f32; cfg.n_fft];
    let left = cfg.n_fft.saturating_sub(fe.window.len()) / 2;
    for (i, &v) in fe.window.iter().enumerate() {
        if left + i < cfg.n_fft {
            window[left + i] = v;
        }
    }
    let mut features = nemo_features_with(
        &NemoFrontend {
            n_fft: cfg.n_fft,
            hop: cfg.hop_length,
            n_mels: cfg.n_mels,
            window: window.clone(),
            preemphasis: Some(0.97),
            per_feature: cfg.normalize == "per_feature",
        },
        NemoPadding {
            zeros: true,
            mask_tail: true,
        },
    )?;
    features.name = "features".into();
    pkg.function(features);
    let n_freq = cfg.n_fft / 2 + 1;
    let mut filters = vec![0f32; n_freq * cfg.n_mels];
    for m in 0..cfg.n_mels {
        for k in 0..n_freq {
            filters[k * cfg.n_mels + m] = fe.fb[m * n_freq + k];
        }
    }
    params.extend(stft_bases(cfg.n_fft, &window));
    params.insert("mel.filters".into(), filters);
    params.insert("audio.iota".into(), audio_iota());

    // The subsampling, symbolic in the mel frames: its output is a quotient
    // of them, pinned by lengths of every remainder by 8; the layers,
    // symbolic in the encoder frames.
    let sizes: Vec<usize> = (0..8).map(|r| 61 + 9 * r).collect();
    let (sg, lifted) = symbolic_in_lifted_many(&[(MEL, sizes)], &[MAX_ENC * 8], |n| {
        Ok(encoder(&cfg, &model, n[0], EncoderPart::Subsampling)?.0)
    })?;
    ensure!(lifted.is_empty(), "the subsampling lifted tables");
    pkg.function(sg);
    params.extend(encoder(&cfg, &model, 64, EncoderPart::Subsampling)?.1);
    pkg.function(symbolic_in(ENC, (13, 17, 9), |t| {
        Ok(encoder(&cfg, &model, t, EncoderPart::Layers)?.0)
    })?);
    params.extend(encoder(&cfg, &model, 9, EncoderPart::Layers)?.1);
    params.insert("enc.rel_pos".into(), rel_pos_encoding(MAX_ENC, cfg.d_model));
    if cfg.chunked_attention {
        params.insert(
            "enc.att_bias".into(),
            chunked_limited_bias(MAX_ENC, cfg.att_context_size),
        );
    }

    // The decoder.
    let names: Vec<String> = [
        keys::PRED_EMBED,
        keys::JOINT_ENC_W,
        keys::JOINT_ENC_B,
        keys::JOINT_PRED_W,
        keys::JOINT_PRED_B,
        keys::JOINT_NET_W,
        keys::JOINT_NET_B,
        keys::PROMPT_K0_W,
        keys::PROMPT_K0_B,
        keys::PROMPT_K2_W,
        keys::PROMPT_K2_B,
    ]
    .iter()
    .map(|s| s.to_string())
    .chain((0..cfg.pred_rnn_layers).flat_map(|l| {
        ["weight_ih", "weight_hh", "bias_ih", "bias_hh"].map(|p| keys::pred_lstm(p, l))
    }))
    .collect();
    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    for n in &names {
        if let Ok(t) = model.tensor(n) {
            shapes.insert(n.as_str(), t.shape.clone());
            params.insert(n.clone(), t.data);
        }
    }
    pkg.function(symbolic_in(ENC, (5, 7, 3), |t| {
        Ok(enc_proj(&cfg, &shapes, t))
    })?);
    pkg.function(pred_step(&shapes, cfg.pred_rnn_layers));
    let durations = spec.durations.len();
    pkg.function(symbolic_in(ENC, (5, 7, 3), |t| {
        Ok(joint(&shapes, durations, t))
    })?);
    pkg.dims = vec![
        ("samples".into(), SAMPLES),
        ("mel".into(), MEL),
        ("frames".into(), ENC),
    ];

    // The checkpoint's tensors referenced from the Hugging Face file beside
    // the `.nemo` (a fingerprint match, transposed or not), the rest held.
    let st = path.with_file_name("model.safetensors");
    if st.is_file() {
        let source = WeightMap::from_file(st.to_str().context("path")?)?;
        pkg.reference_map(&params, &source, "safetensors", &st, out, false)?;
    } else {
        let dims = pkg.param_dims();
        let mut ns: Vec<&String> = params.keys().collect();
        ns.sort();
        for n in ns {
            let d = dims
                .get(n)
                .cloned()
                .unwrap_or_else(|| vec![params[n].len()]);
            pkg.hold_f32(n.clone(), d, &params[n]);
        }
    }

    // The program: the search, the pieces, the language prompt.
    let lang_map: serde_json::Map<String, serde_json::Value> = model
        .config()
        .get_str_i64_map("model_defaults.prompt_dictionary")
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, json!(v)))
        .collect();
    let pieces: Vec<String> = (0..tok.vocab_size())
        .map(|i| tok.piece(i).unwrap_or("").to_string())
        .collect();
    let stages = (cfg.subsampling_factor as f64).log2().round() as usize;
    let pad = if cfg.causal_downsampling { 3 } else { 2 };
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build the transducer program")?;
    program::entry(
        &mut pkg,
        "transcribe",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        json!({
            "sample_rate": cfg.sample_rate,
            "hop": cfg.hop_length,
            "subsampling": vec![pad; stages],
            "languages": if uses_language(&cfg, &shapes) { cfg.num_languages } else { 0 },
            "language_map": lang_map,
            "language": if lang_map.contains_key("auto") { "auto" } else { "en-US" },
            "pred_layers": cfg.pred_rnn_layers,
            "pred_hidden": cfg.pred_hidden,
            "blank": shapes[keys::JOINT_NET_W][0] - durations - 1,
            "durations": spec.durations,
            "sos_carry": spec.sos_carry,
            "max_symbols_per_step": cfg.max_symbols_per_step,
            "pieces": pieces,
            "space": "\u{2581}",
        }),
    );
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter, taken by name (`--family nemotron-asr`):
/// a `.nemo` names no `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "nemotron-asr",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
