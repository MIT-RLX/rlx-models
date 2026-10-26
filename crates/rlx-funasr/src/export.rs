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

//! Export FunASR SenseVoiceSmall and Paraformer as `.rlxp` packages rlx runs
//! alone: `Model.transcribe(audio)`. Paraformer's graphs are its `encoder`
//! (the SAN-M encoder and the CIF predictor's alphas) and its `decoder`
//! (symbolic in the frames and the tokens, each token's argmax); the
//! program fires the CIF (`cif.rs`) between them.
//!
//! Its graph is [`crate::sensevoice::SenseVoice`]'s, `encoder`: the four
//! prompt rows (language, event, emotion, text normalization) gathered from
//! the checkpoint's `embed.weight` before the audio's features, the SAN-M
//! encoder (its sinusoidal positions rows of one table), the CTC head and
//! each frame's argmax; symbolic in the frames. The program (`program/`)
//! makes the features with the crate's own Kaldi fbank, LFR and CMVN
//! (`frontend.rs`), collapses the CTC ids and reads the pieces.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::paraformer::Paraformer;
use crate::sensevoice::SenseVoice;

/// The audio's LFR frames, and (Paraformer) the tokens the CIF fires.
const FRAMES: u32 = 0;
const TOKENS: u32 = 1;
/// Rows of the position table: the longest input (~6 min of audio).
const MAX_ROWS: usize = 6000;
/// The prompt rows before the audio.
const PROMPT: usize = 4;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `g` as a package's graph: each sinusoidal position param (`[1, t,
/// depth]`) rows of one table per depth (`positions.{depth}`, `MAX_ROWS`
/// rows), each all-zero param (a pad) a constant; `params` without them.
fn ready(g: &Graph, params: &mut HashMap<String, Vec<f32>>) -> Result<Graph> {
    let held = params.clone();
    let g = rlx_ir::stateful::rebuild(g, |out, _, node, _| match &node.op {
        Op::Param { name } if name.contains("sinpos") => {
            let dims: Vec<usize> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let (t, depth) = (dims[1], dims[2]);
            let table = out.param(format!("positions.{depth}"), f32s(&[1, MAX_ROWS, depth]));
            Ok(Some(out.narrow_(table, 1, 0, t)))
        }
        Op::Param { name } => {
            let dims: Vec<usize> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            match held.get(name) {
                Some(v) if v.len() > 1 && v.iter().all(|x| *x == 0.0) => {
                    Ok(Some(zeros(out, &dims)))
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    params.retain(|n, v| !n.contains("sinpos") && !(v.len() > 1 && v.iter().all(|x| *x == 0.0)));
    Ok(g)
}

fn zeros(g: &mut Graph, dims: &[usize]) -> rlx_ir::NodeId {
    let zero = g.add_node(
        Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&vec![1; dims.len()]),
    );
    g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![zero],
        f32s(dims),
    )
}

/// The position tables [`ready`] reads, for the depths its graphs use.
fn position_tables(graphs: &[&Graph]) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    for g in graphs {
        for n in g.nodes() {
            if let Op::Param { name } = &n.op
                && let Some(depth) = name.strip_prefix("positions.")
            {
                let depth: usize = depth.parse().expect("a depth");
                out.entry(name.clone())
                    .or_insert_with(|| crate::sanm::sinusoidal_pos(MAX_ROWS, depth));
            }
        }
    }
    out
}

fn argmax(g: &mut Graph, logits: rlx_ir::NodeId, rows: usize, vocab: usize) -> rlx_ir::NodeId {
    let logits = g.reshape_(logits, vec![rows as i64, vocab as i64]);
    let ids = g.add_node(
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        vec![logits],
        f32s(&[rows]),
    );
    g.node_mut(ids).name = Some("ids".into());
    ids
}

/// SenseVoice's `encoder` at `f` audio frames: `prompt` `[4]` i32 (rows of
/// `embed.weight`) and `feats` `[1, f, input]` → `ids` `[f + 4]`, each
/// frame's CTC argmax.
fn encoder(sv: &SenseVoice, f: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let t = f + PROMPT;
    let input = sv.config().encoder.input_size;
    let (g, mut params) = rlx_core::flow_util::graph_from_built(sv.logits_model(t)?)?;
    let g = ready(&g, &mut params)?;
    let rows = sv
        .weights()
        .get("embed.weight")
        .map(|(_, s)| s[0])
        .context("no embed.weight")?;
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Input { name } if name == "feats" => {
            let prompt = out.input("prompt", Shape::new(&[PROMPT], DType::I32));
            let table = out.param("embed.weight", f32s(&[rows, input]));
            let p = out.gather_(table, prompt, 0);
            let p = out.reshape_(p, vec![1, PROMPT as i64, input as i64]);
            let feats = out.input("feats", f32s(&[1, f, input]));
            Ok(Some(out.concat_(vec![p, feats], 1)))
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let logits = g.outputs[0];
    let ids = argmax(&mut g, logits, t, sv.config().vocab_size);
    g.set_outputs(vec![ids]);
    g.name = "encoder".into();
    Ok((g, params))
}

/// Paraformer's `encoder` at `t` frames: `feats` `[1, t, input]` → `enc`
/// `[1, t, d]` and the CIF predictor's `alphas` `[t]` — `relu(conv1d)`
/// (`l_order` / `r_order` frames of context, a matmul a tap), `cif_output`,
/// `max(σ · smooth − noise, 0)`.
fn para_encoder(pf: &Paraformer, t: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let cfg = pf.config();
    let d = cfg.encoder.output_size;
    let pc = &cfg.predictor;
    let k = pc.l_order + pc.r_order + 1;
    let (g, mut params) = rlx_core::flow_util::graph_from_built(pf.encoder_model(t)?)?;
    let mut g = ready(&g, &mut params)?;
    let enc = g.outputs[0];
    g.node_mut(enc).name = Some("enc".into());
    let x = g.reshape_(enc, vec![t as i64, d as i64]);
    let mut parts = Vec::new();
    if pc.l_order > 0 {
        parts.push(zeros(&mut g, &[pc.l_order, d]));
    }
    parts.push(x);
    if pc.r_order > 0 {
        parts.push(zeros(&mut g, &[pc.r_order, d]));
    }
    let padded = g.concat_(parts, 0);
    let mut acc = None;
    for kk in 0..k {
        let rows = g.narrow_(padded, 0, kk, t);
        let w = g.param(format!("predictor.tap{kk}"), f32s(&[d, d]));
        let y = g.mm(rows, w);
        acc = Some(match acc {
            None => y,
            Some(a) => g.add(a, y),
        });
    }
    let b = g.param("predictor.cif_conv1d.bias", f32s(&[1, d]));
    let h = g.add(acc.expect("a tap"), b);
    let h = g.relu(h);
    let ow = g.param("predictor.cif_output.weight.t", f32s(&[d, 1]));
    let a = g.mm(h, ow);
    let ob = g.param("predictor.cif_output.bias", f32s(&[1]));
    let a = g.add(a, ob);
    let a = g.sigmoid(a);
    let sf = g.add_node(
        Op::Constant {
            data: pc.smooth_factor.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let nt = g.add_node(
        Op::Constant {
            data: pc.noise_threshold.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let a = g.mul(a, sf);
    let a = g.sub(a, nt);
    let a = g.relu(a);
    let alphas = g.reshape_(a, vec![t as i64]);
    g.node_mut(alphas).name = Some("alphas".into());
    g.set_outputs(vec![enc, alphas]);
    g.name = "encoder".into();
    // The conv's taps, `[out, in, k]` → `[in, out]` each.
    let w = pf
        .weights()
        .get("predictor.cif_conv1d.weight")
        .map(|(v, _)| v.to_vec())
        .context("no predictor.cif_conv1d.weight")?;
    for kk in 0..k {
        let mut tap = vec![0f32; d * d];
        for o in 0..d {
            for i in 0..d {
                tap[i * d + o] = w[o * d * k + i * k + kk];
            }
        }
        params.insert(format!("predictor.tap{kk}"), tap);
    }
    for key in ["predictor.cif_conv1d.bias", "predictor.cif_output.bias"] {
        let v = pf
            .weights()
            .get(key)
            .map(|(v, _)| v.to_vec())
            .context("predictor")?;
        params.insert(key.into(), v);
    }
    let ow = pf
        .weights()
        .get("predictor.cif_output.weight")
        .map(|(v, _)| v.to_vec())
        .context("no predictor.cif_output.weight")?;
    params.insert("predictor.cif_output.weight.t".into(), ow);
    Ok((g, params))
}

/// Paraformer's `decoder` at `t` frames and `l` tokens: `memory` `[1, t,
/// d]`, `tgt` `[1, l, d]` (the CIF embeddings) → `ids` `[l]`.
fn para_decoder(pf: &Paraformer, t: usize, l: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (g, mut params) = rlx_core::flow_util::graph_from_built(pf.decoder_model(t, l)?)?;
    let mut g = ready(&g, &mut params)?;
    let logits = g.outputs[0];
    let ids = argmax(&mut g, logits, l, pf.config().vocab_size);
    g.set_outputs(vec![ids]);
    g.name = "decoder".into();
    Ok((g, params))
}

/// The package's held tensors.
fn hold(pkg: &mut Package, params: &HashMap<String, Vec<f32>>) {
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
}

/// The program's frontend config.
fn frontend_json(fe: &crate::frontend::FrontendConfig) -> serde_json::Value {
    json!({
        "sample_rate": fe.sample_rate,
        "n_mels": fe.n_mels,
        "frame_length_ms": fe.frame_length_ms,
        "frame_shift_ms": fe.frame_shift_ms,
        "low_freq": fe.low_freq,
        "high_freq": fe.high_freq,
        "preemphasis": fe.preemphasis,
        "lfr_m": fe.lfr_m,
        "lfr_n": fe.lfr_n,
    })
}

/// Export the FunASR model directory `dir` (`model.pt`, `config.yaml`,
/// `am.mvn`, the tokens): SenseVoiceSmall or Paraformer, as its
/// `config.yaml` says, as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let yaml = std::fs::read_to_string(dir.join("config.yaml"))?;
    let model = yaml
        .lines()
        .find_map(|l| l.strip_prefix("model:"))
        .map(str::trim)
        .unwrap_or("");
    let cmvn = crate::frontend::load_configured_cmvn(dir);
    let tok = crate::tokenizer::Tokenizer::from_dir(dir)?;
    let mut pkg;
    let config = if model == "SenseVoiceSmall" {
        let sv = SenseVoice::open(dir, rlx_runtime::Device::Cpu)?;
        let cfg = sv.config().clone();
        pkg = Package::new(
            "sensevoice-small",
            &opts.version,
            "sensevoice",
            "automatic-speech-recognition",
            "rlx-funasr",
        );
        let g = symbolic_in(FRAMES, (5, 7, 3), |f| Ok(encoder(&sv, f)?.0))?;
        let mut params = position_tables(&[&g]);
        params.extend(encoder(&sv, 3)?.1);
        if let Some((data, _)) = sv.weights().get("embed.weight") {
            params.insert("embed.weight".into(), data.to_vec());
        }
        pkg.function(g);
        pkg.dims = vec![("frames".into(), FRAMES)];
        hold(&mut pkg, &params);
        let pieces: Vec<String> = (0..cfg.vocab_size as u32)
            .map(|i| tok.piece(i).unwrap_or("").to_string())
            .collect();
        json!({
            "kind": "sensevoice",
            "frontend": frontend_json(&cfg.frontend),
            "cmvn": cmvn.map(|c| json!({"neg_mean": c.neg_mean, "inv_std": c.inv_std})),
            "languages": {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13},
            "textnorm": {"withitn": 14, "woitn": 15},
            "blank": cfg.blank_id,
            "pieces": pieces,
        })
    } else {
        let pf = Paraformer::open(dir, rlx_runtime::Device::Cpu)?;
        let cfg = pf.config().clone();
        pkg = Package::new(
            "paraformer",
            &opts.version,
            "paraformer",
            "automatic-speech-recognition",
            "rlx-funasr",
        );
        let e = symbolic_in(FRAMES, (13, 17, 9), |t| Ok(para_encoder(&pf, t)?.0))?;
        let (d, lifted) = symbolic_in_lifted_many(
            &[(FRAMES, vec![13, 17, 9]), (TOKENS, vec![3, 4, 6])],
            &[MAX_ROWS, MAX_ROWS],
            |n| Ok(para_decoder(&pf, n[0], n[1])?.0),
        )?;
        anyhow::ensure!(lifted.is_empty(), "the decoder lifted tables");
        let mut params = position_tables(&[&e, &d]);
        params.extend(para_encoder(&pf, 9)?.1);
        params.extend(para_decoder(&pf, 9, 3)?.1);
        pkg.function(e);
        pkg.function(d);
        pkg.dims = vec![("frames".into(), FRAMES), ("tokens".into(), TOKENS)];
        hold(&mut pkg, &params);
        let pieces: Vec<String> = (0..cfg.vocab_size as u32)
            .map(|i| tok.piece(i).unwrap_or("").to_string())
            .collect();
        let pc = &cfg.predictor;
        json!({
            "kind": "paraformer",
            "frontend": frontend_json(&cfg.frontend),
            "cmvn": cmvn.map(|c| json!({"neg_mean": c.neg_mean, "inv_std": c.inv_std})),
            "cif": {
                "idim": pc.idim, "l_order": pc.l_order, "r_order": pc.r_order,
                "threshold": pc.threshold, "tail_threshold": pc.tail_threshold,
                "smooth_factor": pc.smooth_factor, "noise_threshold": pc.noise_threshold,
            },
            "dim": cfg.encoder.output_size,
            "blank": cfg.blank_id,
            "pieces": pieces,
        })
    };
    pkg.model["license"] = json!("funasr-model-license");
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build the FunASR program")?;
    let rate = config["frontend"]["sample_rate"].clone();
    program::entry(
        &mut pkg,
        "transcribe",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        config,
    );
    pkg.metadata("sample_rate", rate);
    pkg.write(out, opts)
}

/// Register this crate's exporter, taken by name (`--family funasr`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "funasr",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
