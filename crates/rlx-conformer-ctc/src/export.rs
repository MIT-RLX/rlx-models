// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a NeMo Conformer-CTC `.nemo` as an `.rlxp` package rlx runs alone:
//! `Model.transcribe(audio)`. Its graphs are NeMo's frontend
//! ([`rlx_core::export::audio::nemo_features_with`]: zero-padded, the
//! frame past the audio masked, the checkpoint's own window and
//! filterbank) and the encoder at each mel-length bucket the runner pads
//! to ([`crate::bucket_mel_frames`]), the CTC head and each frame's argmax
//! appended; the shared CTC program ([`rlx_core::export::program::ctc`])
//! pads, collapses and reads the SentencePiece pieces, as
//! [`crate::ConformerCtc::transcribe`] does.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::export::audio::{
    NemoFrontend, NemoPadding, SAMPLES, audio_iota, nemo_features_with, stft_bases,
};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::export::{Package, program};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_nemo::NemoModel;
use serde_json::json;

use crate::config::AsrConfig;
use crate::encoder::build_encoder_hir;
use crate::mel::Frontend;
use crate::tokenizer::SpmTokenizer;
use crate::weights::{NemoWeights, keys};

/// The mel-length buckets exported: the runner's ladder up to ~30 s.
pub const BUCKETS: [usize; 6] = [256, 512, 768, 1024, 1536, 2048];

/// The encoder at `frames` mel frames, its output through the CTC head
/// (`ctc.weight` `[d_model, classes]`, `ctc.bias`) to each frame's argmax.
fn encoder(
    cfg: &AsrConfig,
    model: &NemoModel,
    frames: usize,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut w = NemoWeights::new(model);
    let (hir, params, t) = build_encoder_hir(cfg, &mut w, frames)?;
    let built = rlx_core::flow_util::built_from_hir(hir, params)?;
    let (mut g, params) = rlx_core::flow_util::graph_from_built(built)?;
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let (d, c) = (cfg.d_model, cfg.num_classes);
    let enc = g.outputs[0];
    let enc = g.reshape_(enc, vec![t as i64, d as i64]);
    let wt = g.param("ctc.weight", f32s(&[d, c]));
    let b = g.param("ctc.bias", f32s(&[c]));
    let logits = g.mm(enc, wt);
    let logits = g.add(logits, b);
    let ids = g.add_node(
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        vec![logits],
        f32s(&[t]),
    );
    g.node_mut(ids).name = Some("ids".into());
    g.set_outputs(vec![ids]);
    g.name = format!("encode_{frames}");
    Ok((g, params))
}

/// Export the Conformer-CTC checkpoint `path` (`.nemo`) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
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

    // The window centred into n_fft, as the runner frames.
    let mut window = vec![0f32; cfg.n_fft];
    let left = cfg.n_fft.saturating_sub(fe.window.len()) / 2;
    for (i, &v) in fe.window.iter().enumerate() {
        if left + i < cfg.n_fft {
            window[left + i] = v;
        }
    }
    let features = nemo_features_with(
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
    let n_freq = cfg.n_fft / 2 + 1;
    let mut filters = vec![0f32; n_freq * cfg.n_mels];
    for m in 0..cfg.n_mels {
        for k in 0..n_freq {
            filters[k * cfg.n_mels + m] = fe.fb[m * n_freq + k];
        }
    }

    let mut builds = Vec::new();
    for &frames in &BUCKETS {
        let (g, p) = encoder(&cfg, &model, frames)?;
        builds.push((frames, g, p));
    }
    let (encoders, mut params) = rlx_core::export::graph::per_size(builds);

    // The CTC head, `[classes, d, 1]` → `[d, classes]`.
    let (wd, sh) = {
        let t = model.tensor(keys::CTC_W)?;
        (t.data, t.shape)
    };
    ensure!(
        sh.first() == Some(&cfg.num_classes),
        "CTC weight {sh:?}, not {} classes",
        cfg.num_classes
    );
    let (c, d) = (cfg.num_classes, cfg.d_model);
    let mut wt = vec![0f32; d * c];
    for ci in 0..c {
        for di in 0..d {
            wt[di * c + ci] = wd[ci * d + di];
        }
    }
    params.insert("ctc.weight".into(), wt);
    params.insert("ctc.bias".into(), model.tensor(keys::CTC_B)?.data);
    params.extend(stft_bases(cfg.n_fft, &window));
    params.insert("mel.filters".into(), filters);
    params.insert("audio.iota".into(), audio_iota());

    let mut pkg = Package::new(
        "conformer-ctc",
        &opts.version,
        "conformer-ctc",
        "automatic-speech-recognition",
        "rlx-conformer-ctc",
    );
    let mut features = features;
    features.name = "features".into();
    pkg.function(features);
    for g in encoders {
        pkg.function(g);
    }
    pkg.dims = vec![("samples".into(), SAMPLES)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let shape = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), shape, &params[n]);
    }
    let stages = (cfg.subsampling_factor as f64).log2().round() as usize;
    let pieces: Vec<String> = (0..tok.vocab_size())
        .map(|i| tok.piece(i).unwrap_or("").to_string())
        .collect();
    program::ctc(
        &mut pkg,
        json!({
            "sample_rate": cfg.sample_rate,
            "features": "features",
            "encoders": BUCKETS.iter().map(|b| json!({"frames": b, "function": format!("encode_{b}")})).collect::<Vec<_>>(),
            "pad_value": 0.0,
            "subsampling": vec![[3, 2, 1]; stages],
            "blank": cfg.blank_id,
            "pieces": pieces,
            "space": "\u{2581}",
            "strip_specials": true,
        }),
    )?;
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family conformer-ctc`): a `.nemo` names no
/// `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "conformer-ctc",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}
