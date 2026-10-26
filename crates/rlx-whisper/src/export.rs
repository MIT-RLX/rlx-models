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

//! Export a Hugging Face Whisper checkpoint as an `.rlxp` package that rlx
//! runs with no rlx-models code: a `transcribe` entry whose stages are the
//! audio front end (waveform → log-mel, as a graph), the encoder and the
//! cross-attention K / V, then the decode step. The front end can be its own
//! package, shared by every model that takes the same features.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::{Checkpoint, Package, package_dir, relative_path};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::model::Tensor;
use serde_json::{Value, json};

use crate::{
    FusedDecoderWeights, FusedEncoderWeights, WhisperConfig, WhisperGraphCtx, WhisperWeightPrefix,
};

/// Samples in one window: 30 s at 16 kHz.
const SAMPLES: usize = 480_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
/// Mel frames in one window.
const FRAMES: usize = SAMPLES / HOP;

pub use rlx_core::export::{ExportOptions, ExportReport};

/// Decode-cache capacities when the options list none (the longest is the
/// transcript's token cap).
const DEFAULT_UPPERS: [usize; 2] = [64, 224];

/// Whisper's log-mel front end as a graph: `pcm [480000]` → `mel [1, n_mels,
/// 3000]`, from primitives. Frames are 400 samples every 160, built without
/// an index table: the reflect-padded signal as rows of 160, three shifted row
/// windows side by side (480 samples), the first 400 kept. Hann window, FFT,
/// power, mel filters, log10, the floor 8 below the whole spectrogram's max,
/// `(x + 4) / 4`, over the 3000 frames Whisper keeps.
pub fn log_mel_graph(n_mels: usize) -> Graph {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("log_mel");
    let pcm = g.input("pcm", f(&[SAMPLES]));
    let window = g.param("frontend.window", f(&[N_FFT]));
    let filters = g.param("frontend.mel_filters", f(&[n_mels, N_FFT / 2 + 1]));
    // Centered frames: reflect-padded by N_FFT/2 each side (`center=True`),
    // zeros after to fill 3003 rows.
    let rows = FRAMES + 3;
    let reflected = g.pad_(
        pcm,
        vec![[N_FFT / 2, N_FFT / 2]],
        rlx_ir::op::PadMode::Reflect,
    );
    let after_len = rows * HOP - SAMPLES - N_FFT;
    let after = g.constant(0.0, DType::F32);
    let after = g.add_node(
        Op::Expand {
            target_shape: vec![after_len as i64],
        },
        vec![after],
        f(&[after_len]),
    );
    let padded = g.concat_(vec![reflected, after], 0);
    let r = g.reshape_(padded, vec![rows as i64, HOP as i64]);
    let shifted: Vec<_> = (0..3).map(|k| g.narrow_(r, 0, k, FRAMES)).collect();
    let wide = g.concat_(shifted, 1);
    let frames = g.narrow_(wide, 1, 0, N_FFT);
    let windowed = g.mul(frames, window);
    let zero = g.constant(0.0, DType::F32);
    let imag = g.mul(windowed, zero);
    let block = g.concat_(vec![windowed, imag], 1);
    let spectrum = g.add_node(
        Op::Fft {
            inverse: false,
            norm: rlx_ir::fft::FftNorm::Backward,
        },
        vec![block],
        f(&[FRAMES, 2 * N_FFT]),
    );
    // Power, mel filters, log10, then Whisper's floor 8 below the max of
    // the whole spectrogram (`Op::LogMel` floors each frame at its own).
    let n_bins = N_FFT / 2 + 1;
    let re = g.narrow_(spectrum, 1, 0, n_bins);
    let im = g.narrow_(spectrum, 1, N_FFT, n_bins);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let filters_t = g.transpose_(filters, vec![1, 0]);
    let mel = g.mm(power, filters_t);
    let mel = g.clamp_(mel, 1e-10, f32::MAX);
    let ln = g.log(mel);
    let inv_ln10 = g.constant(1.0 / std::f64::consts::LN_10, DType::F32);
    let log10 = g.mul(ln, inv_ln10);
    let max = g.reduce(
        log10,
        rlx_ir::op::ReduceOp::Max,
        vec![0, 1],
        true,
        f(&[1, 1]),
    );
    let eight = g.constant(8.0, DType::F32);
    let floor = g.sub(max, eight);
    let floored = g.binary(
        rlx_ir::op::BinaryOp::Max,
        log10,
        floor,
        f(&[FRAMES, n_mels]),
    );
    let four = g.constant(4.0, DType::F32);
    let shifted = g.add(floored, four);
    let quarter = g.constant(0.25, DType::F32);
    let mel = g.mul(shifted, quarter);
    let t = g.transpose_(mel, vec![1, 0]);
    let out = g.reshape_(t, vec![1, n_mels as i64, FRAMES as i64]);
    g.set_outputs(vec![out]);
    g
}

/// The front end's own tensors (window, filters).
fn frontend_tensors(n_mels: usize) -> Vec<Tensor> {
    vec![
        Tensor::f32(
            "frontend.window",
            vec![N_FFT],
            &crate::mel::hann_window(N_FFT),
        ),
        Tensor::f32(
            "frontend.mel_filters",
            vec![n_mels, N_FFT / 2 + 1],
            &crate::mel::mel_filterbank(16_000.0, N_FFT, n_mels),
        ),
    ]
}

fn ids(v: Option<&Value>) -> Vec<u64> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

/// Export the Hugging Face Whisper checkpoint in `dir` as the package
/// `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = WhisperConfig::from_file(&dir.join("config.json"))?;
    let st = dir.join("model.safetensors");
    let st_str = st.to_str().context("weights path is not UTF-8")?;
    let mut w = WeightMap::snapshot_from_path(st_str)?;
    let pfx = WhisperWeightPrefix::detect_with(|k| w.contains_key(k));
    let fused = FusedDecoderWeights::from_checkpoint(&w, &cfg, &pfx)?;
    let fused_enc = FusedEncoderWeights::from_checkpoint(&w, &cfg, &pfx)?;
    fused.merge_into_tensors(&mut w);
    fused_enc.merge_into_tensors(&mut w);
    let enc_seq = cfg.encoder_seq_len(FRAMES);
    let ctx = WhisperGraphCtx {
        cfg: cfg.clone(),
        pfx,
        weights: std::sync::Arc::new(w),
        enc_seq,
        mel_frames: FRAMES,
        graph_opts: Default::default(),
        fused: Some(fused),
        fused_enc: Some(fused_enc),
    };
    let mut uppers = opts.uppers_or(&DEFAULT_UPPERS);
    uppers.sort();

    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut graphs = Vec::new();
    for (name, built) in [
        ("encode", ctx.build_encoder_sized(1, FRAMES)?),
        ("cross", ctx.build_cross_sized(1, enc_seq)?),
    ] {
        let (mut g, p) = graph_from_built(built)?;
        g.name = name.into();
        params.extend(p);
        graphs.push(g);
    }
    // One decode step symbolic in its cache, else one per capacity.
    let step = |upper: usize| -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        graph_from_built(ctx.build_decode_step_sized(1, upper, enc_seq)?)
    };
    let mut symbolic_failure = None;
    let (a, pa) = step(36)?;
    let (b, _) = step(52)?;
    let decode_graphs = match rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53), (1, 36, 52)]) {
        Ok(mut g) => {
            g.name = "decode".into();
            vec![g]
        }
        Err(e) => {
            symbolic_failure = Some(e);
            let mut v = Vec::new();
            for &u in &uppers {
                let (mut g, _) = step(u)?;
                g.name = format!("decode_{u}");
                v.push(g);
            }
            v
        }
    };
    params.extend(pa);
    let decode_names: Vec<String> = decode_graphs.iter().map(|g| g.name.clone()).collect();
    graphs.extend(decode_graphs);

    // The decode step's inputs; its states in input order, its outputs after
    // the logits in the same order (each layer's K then V).
    let dec = graphs.last().unwrap();
    let mut inputs = serde_json::Map::new();
    let mut states = Vec::new();
    for n in dec.nodes() {
        let Op::Input { name } = &n.op else { continue };
        let feed = match name.as_str() {
            "token_id" => json!("token"),
            "pos_ix" => json!("position"),
            "mask" => json!("decode_mask"),
            c if c.starts_with("cross_") => json!({"context": c}),
            p if p.starts_with("past_") => {
                states.push(p.to_string());
                continue;
            }
            other => bail!("decode input '{other}' has no feed"),
        };
        inputs.insert(name.clone(), feed);
    }
    if states.len() + 1 != dec.outputs.len() {
        bail!(
            "{} caches but {} outputs after the logits",
            states.len(),
            dec.outputs.len() - 1
        );
    }
    let state: Vec<Value> = states
        .iter()
        .enumerate()
        .map(|(i, s)| json!([s, i + 1, "append"]))
        .collect();
    let vocab = match dec.node(dec.outputs[0]).shape.dims().last() {
        Some(rlx_ir::Dim::Static(v)) => *v,
        _ => cfg.vocab_size,
    };
    let cross = graphs.iter().find(|g| g.name == "cross").unwrap();
    let cross_in = cross
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .context("the cross graph has no input")?;
    let cross_out: Vec<String> = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
        .collect();
    if cross_out.len() != cross.outputs.len() {
        bail!(
            "the cross graph has {} outputs, not {}",
            cross.outputs.len(),
            cross_out.len()
        );
    }

    let ck = Checkpoint::open(dir)?;
    let mut pkg =
        Package::for_checkpoint(&ck, opts, "whisper", "speech-recognition", "rlx-whisper");

    // The front end: in this package, or its own.
    let frontend = log_mel_graph(cfg.num_mel_bins);
    let mut front_stage = json!({
        "function": "log_mel",
        "inputs": {"pcm": "audio"},
        "outputs": ["mel"],
    });
    match &opts.frontend_package {
        Some(path) => {
            let mut fe = Package::new("", &opts.version, "whisper", "", "rlx-whisper");
            fe.function(frontend);
            fe.held = frontend_tensors(cfg.num_mel_bins);
            // It runs alone too, as a pipeline: 30 s of audio in, its mel.
            fe.model = json!({
                "name": format!("whisper-log-mel-{}", cfg.num_mel_bins),
                "task": "feature-extraction",
                "entries": {"pipeline": {
                    "task": "feature-extraction",
                    "inputs": [{
                        "name": "audio",
                        "kind": "audio",
                        "dtype": "f32",
                        "shape": [SAMPLES],
                        "sample_rate": 16000,
                        "channels": 1,
                    }],
                    "stages": [front_stage.clone()],
                    "outputs": ["mel"],
                }},
                "metadata": {"input": "16 kHz mono, 30 s", "output": "log-mel [1, n_mels, 3000]"},
            });
            fe.write(path, opts)?;
            let rel = relative_path(
                package_dir(out),
                &path.canonicalize().unwrap_or(path.clone()),
            );
            front_stage["package"] = json!(rel.display().to_string());
        }
        None => {
            graphs.insert(0, frontend);
            pkg.held.extend(frontend_tensors(cfg.num_mel_bins));
        }
    }

    let gcfg = ck.json("generation_config.json");
    let lang = |l: &str| gcfg["lang_to_id"][l].as_u64();
    let mut prompt = vec![gcfg["decoder_start_token_id"].as_u64().unwrap_or(50258)];
    if gcfg["is_multilingual"].as_bool().unwrap_or(false) {
        prompt.extend(lang("<|en|>"));
        prompt.extend(gcfg["task_to_id"]["transcribe"].as_u64());
    }
    prompt.extend(gcfg["no_timestamps_token_id"].as_u64());
    let eos = gcfg["eos_token_id"].as_u64().unwrap_or(50257);

    pkg.graphs = graphs;
    // Weights: referenced where they equal a checkpoint tensor.
    let source = WeightMap::from_file(st_str)?;
    pkg.reference_map(
        &params,
        &source,
        "safetensors",
        &st,
        out,
        opts.embed_weights,
    )?;
    drop(source);
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let entry = json!({
            "sample_rate": 16000,
            "stages": [
                front_stage,
                {"function": "encode", "inputs": {"mel": "mel"}, "outputs": ["encoded"]},
                {"function": "cross", "inputs": {cross_in: "encoded"}, "outputs": cross_out},
            ],
            "decoder": {
                "functions": decode_names,
                "sizes": uppers.iter().map(|u| json!({"cap": u + 1, "upper": u})).collect::<Vec<_>>(),
                "inputs": inputs,
                "state": state,
                "logits": 0,
                "vocab_size": vocab,
            },
            "tokenizer": tokenizer,
            "prompt_ids": prompt,
            "stop_ids": [eos],
            "suppress_ids": ids(gcfg.get("suppress_tokens")),
            "begin_suppress_ids": ids(gcfg.get("begin_suppress_tokens")),
            "max_tokens": uppers.last().copied().unwrap_or(224) - prompt.len(),
    });
    pkg.entry("transcribe", entry);
    if symbolic_failure.is_none() {
        pkg.dims = vec![("cap".into(), 0), ("upper".into(), 1)];
    }
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        ..report
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `whisper` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("whisper", "speech-recognition", export_rlxp).claims(&["whisper"]),
    );
}
