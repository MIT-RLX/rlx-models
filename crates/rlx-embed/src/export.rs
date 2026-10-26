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

//! Export a Hugging Face BERT / XLM-R / NomicBERT encoder as an `.rlxp` package that
//! rlx runs with no rlx-models code: the encoder graph at a few sequence
//! lengths, an `embed` entry, the tokenizer, the model card, and its
//! weights referenced from the checkpoint rather than copied.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_bert::flow::build_bert_built;
use rlx_core::config::{BertConfig, NomicBertConfig};
use rlx_core::export::{Checkpoint, Package};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_nomic::flow::build_nomic_built;
use serde_json::{Value, json};

use crate::bert::roberta_position_offset;

pub use rlx_core::export::{ExportOptions, ExportReport};

/// Sequence lengths an encoder is built at when the options list none.
const DEFAULT_SEQS: [usize; 6] = [16, 32, 64, 128, 256, 512];

/// Export the checkpoint in `dir` (`config.json`, `model.safetensors`,
/// `tokenizer.json`, optional `README.md`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export(dir, out, opts, false)
}

/// Export a sequence-classification checkpoint (a cross-encoder reranker:
/// `BertForSequenceClassification`, `XLMRobertaForSequenceClassification`)
/// as a package with a `score` entry: each (query, passage) pair through
/// the encoder and its classifier head, one logit per label out.
/// `opts.pooling` is unused.
pub fn export_reranker_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export(dir, out, opts, true)
}

/// The tensor of the first of `keys` the checkpoint has, and its dims.
fn tensor(file: &WeightMap, keys: &[&str]) -> Option<(Vec<f32>, Vec<usize>)> {
    keys.iter()
        .find_map(|k| file.get(k).map(|(d, s)| (d.to_vec(), s.to_vec())))
}

/// The classifier head on the encoder's `[CLS]` row: BERT's pooler (dense +
/// tanh) and `classifier`, or RoBERTa's `classifier.dense` + tanh +
/// `classifier.out_proj`. The graph's output becomes `[batch, labels]`
/// logits; the head's params join `params` (weights transposed to
/// `[in, out]`, which the package references from the checkpoint).
fn add_classifier_head(
    g: &mut rlx_ir::Graph,
    params: &mut HashMap<String, Vec<f32>>,
    file: &WeightMap,
    batch: usize,
    hidden: usize,
) -> Result<usize> {
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Op, Shape};
    let out = *g.outputs.first().context("the encoder has no output")?;
    let seq = g.node(out).shape.dims().len();
    if seq != 3 {
        bail!("the encoder's output has rank {seq}, not [batch, seq, hidden]");
    }
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let cls = g.add_node(
        Op::Narrow {
            axis: 1,
            start: 0,
            len: 1,
        },
        vec![out],
        f32s(&[batch, 1, hidden]),
    );
    let mut x = g.reshape(
        cls,
        vec![batch as i64, hidden as i64],
        f32s(&[batch, hidden]),
    );
    // `x · Wᵀ + b` for a torch `[out, in]` weight, optionally through tanh.
    let mut dense = |g: &mut rlx_ir::Graph,
                     x: rlx_ir::NodeId,
                     name: &str,
                     (w, wd): (Vec<f32>, Vec<usize>),
                     b: Option<(Vec<f32>, Vec<usize>)>,
                     tanh: bool|
     -> Result<(rlx_ir::NodeId, usize)> {
        let [o, i] = wd[..] else {
            bail!("{name}.weight is {wd:?}, not [out, in]");
        };
        if i != hidden {
            bail!("{name}.weight is {wd:?}; the encoder is {hidden} wide");
        }
        let mut t = vec![0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                t[c * o + r] = w[r * i + c];
            }
        }
        let wn = format!("{name}.weight.t");
        params.insert(wn.clone(), t);
        let wp = g.param(wn, f32s(&[i, o]));
        let mut y = g.matmul(x, wp, f32s(&[batch, o]));
        if let Some((b, _)) = b {
            let bn = format!("{name}.bias");
            params.insert(bn.clone(), b);
            let bp = g.param(bn, f32s(&[o]));
            y = g.binary(BinaryOp::Add, y, bp, f32s(&[batch, o]));
        }
        if tanh {
            y = g.activation(Activation::Tanh, y, f32s(&[batch, o]));
        }
        Ok((y, o))
    };
    let labels;
    if let Some(w) = tensor(file, &["classifier.out_proj.weight"]) {
        // RoBERTa: dense + tanh, then out_proj.
        let dw = tensor(file, &["classifier.dense.weight"])
            .context("classifier.out_proj without classifier.dense")?;
        let db = tensor(file, &["classifier.dense.bias"]);
        (x, _) = dense(g, x, "classifier.dense", dw, db, true)?;
        let b = tensor(file, &["classifier.out_proj.bias"]);
        (x, labels) = dense(g, x, "classifier.out_proj", w, b, false)?;
    } else {
        let w = tensor(file, &["classifier.weight"]).context(
            "the checkpoint has no `classifier.weight` (not a sequence-classification model?)",
        )?;
        if let Some(pw) = tensor(file, &["bert.pooler.dense.weight", "pooler.dense.weight"]) {
            let pb = tensor(file, &["bert.pooler.dense.bias", "pooler.dense.bias"]);
            (x, _) = dense(g, x, "pooler.dense", pw, pb, true)?;
        }
        let b = tensor(file, &["classifier.bias"]);
        (x, labels) = dense(g, x, "classifier", w, b, false)?;
    }
    g.set_outputs(vec![x]);
    Ok(labels)
}

/// A NomicBERT config from `config.json`, under its own keys or the
/// GPT-style ones Hugging Face's `nomic_bert` checkpoints use.
fn nomic_config(raw: &Value) -> Result<NomicBertConfig> {
    if let Ok(c) = serde_json::from_value::<NomicBertConfig>(raw.clone()) {
        return Ok(c);
    }
    let u = |k: &str| {
        raw.get(k)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .with_context(|| format!("nomic_bert config.json: no {k}"))
    };
    let hidden = u("n_embd")?;
    let heads = u("n_head")?;
    Ok(NomicBertConfig {
        vocab_size: u("vocab_size")?,
        hidden_size: hidden,
        num_hidden_layers: u("n_layer")?,
        num_attention_heads: heads,
        intermediate_size: u("n_inner").unwrap_or(4 * hidden),
        max_position_embeddings: u("n_positions")?,
        type_vocab_size: u("type_vocab_size").unwrap_or(2),
        layer_norm_eps: raw
            .get("layer_norm_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(1e-12),
        head_dim: hidden / heads,
        rotary_emb_base: raw
            .get("rotary_emb_base")
            .and_then(Value::as_f64)
            .unwrap_or(1000.0),
    })
}

fn export(dir: &Path, out: &Path, opts: &ExportOptions, reranker: bool) -> Result<ExportReport> {
    let config_path = dir.join("config.json");
    let weights_path = dir.join("model.safetensors");
    let raw_config: Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let nomic = raw_config.get("model_type").and_then(Value::as_str) == Some("nomic_bert");
    if nomic && reranker {
        bail!("a NomicBERT reranker is not exported");
    }
    // NomicBERT (rotary, SwiGLU) or BERT / XLM-R.
    let (bert, nomic_cfg) = if nomic {
        (None, Some(nomic_config(&raw_config)?))
    } else {
        (Some(BertConfig::from_file(&config_path)?), None)
    };
    let hidden = match (&bert, &nomic_cfg) {
        (Some(b), _) => b.hidden_size,
        (_, Some(n)) => n.hidden_size,
        _ => unreachable!("one config"),
    };
    let offset = if nomic {
        0
    } else {
        roberta_position_offset(&config_path)
    };
    let family = raw_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("bert")
        .to_string();
    let max_positions = raw_config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(512) as usize;
    let seqs: Vec<usize> = opts
        .seqs_or(&DEFAULT_SEQS)
        .into_iter()
        .filter(|&s| s + offset <= max_positions)
        .collect();
    if seqs.is_empty() {
        bail!("no sequence length fits {max_positions} positions");
    }
    let weights_str = weights_path.to_str().context("weights path is not UTF-8")?;

    // The encoder at each length (and a reranker's head); the params are
    // the same in every one.
    let file = WeightMap::from_file(weights_str)?;
    let mut labels = 0;
    let mut build = |seq: usize| -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
        let mut wm = WeightMap::from_file(weights_str)?;
        let (mut g, mut params) = graph_from_built(match (&bert, &nomic_cfg) {
            (Some(b), _) => build_bert_built(b, &mut wm, opts.texts, seq)?,
            (_, Some(n)) => build_nomic_built(n, &mut wm, opts.texts, seq)?,
            _ => unreachable!("one config"),
        })?;
        if reranker {
            labels = add_classifier_head(&mut g, &mut params, &file, opts.texts, hidden)?;
        }
        Ok((g, params))
    };
    // One symbolic graph, if the encoder is symbolic in `seq`: built at two
    // sizes no other dim has, and generalized. Else one graph per length.
    let mut symbolic_failure = None;
    let mut symbolic = None;
    if opts.symbolic {
        let (a, pa) = build(37)?;
        let (b, _) = build(53)?;
        match rlx_ir::dynamic::generalize(&a, &b, &[(0, 37, 53)]) {
            Ok(mut g) => {
                g.name = "encode".into();
                symbolic = Some((g, pa));
            }
            Err(e) => symbolic_failure = Some(e),
        }
    }
    let (graphs, params) = match symbolic {
        Some((g, p)) => (vec![g], p),
        None => {
            // One graph per length; a param that depends on the length (a
            // rotary table) is one per length.
            let mut builds = Vec::new();
            for &seq in &seqs {
                let (mut graph, p) = build(seq)?;
                graph.name = format!("encode_{seq}");
                builds.push((seq, graph, p));
            }
            rlx_core::export::graph::per_size(builds)
        }
    };
    let ck = Checkpoint::open(dir)?;
    let task = if reranker {
        "text-ranking"
    } else {
        "embedding"
    };
    let mut pkg = Package::for_checkpoint(&ck, opts, &family, task, "rlx-embed");
    let functions: Vec<String> = graphs.iter().map(|g| g.name.clone()).collect();
    pkg.graphs = graphs;
    // Each param that is a checkpoint tensor, as stored or transposed, is
    // referenced; the rest (derived at build time) are held.
    pkg.reference_map(
        &params,
        &file,
        "safetensors",
        &weights_path,
        out,
        opts.embed_weights,
    )?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    let sizes: Vec<Value> = seqs.iter().map(|s| json!({"seq": s})).collect();
    if reranker {
        pkg.entry(
            "score",
            json!({
                "functions": functions,
                "sizes": sizes,
                "tokenizer": tokenizer,
                "inputs": {
                    "input_ids": "token_ids",
                    "attention_mask": "attention_mask",
                    "token_type_ids": "token_types",
                    "position_ids": "positions",
                },
                "position_offset": offset,
                "label": 0,
            }),
        );
        pkg.metadata("labels", json!(labels));
    } else {
        pkg.entry(
            "embed",
            json!({
                "functions": functions,
                "sizes": sizes,
                "tokenizer": tokenizer,
                "inputs": {
                    "input_ids": "token_ids",
                    "attention_mask": "attention_mask",
                    "token_type_ids": "zeros",
                    "position_ids": "positions",
                },
                "position_offset": offset,
                "pooling": opts.pooling.as_deref().unwrap_or("mean"),
                "normalize": true,
            }),
        );
        pkg.metadata("hidden_size", json!(hidden));
    }
    if symbolic_failure.is_none() && opts.symbolic {
        pkg.dims = vec![("seq".into(), 0)];
    }
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        ..report
    })
}

/// Register this crate's exporters ([`rlx_core::export::register_exporter`]):
/// a BERT / XLM-R encoder as an `embed` package, and one with a
/// sequence-classification head as a `score` package.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("embed", "embedding", export_rlxp).claims(&[
        "bert",
        "xlm-roberta",
        "roberta",
        "nomic_bert",
    ]));
    register_exporter(
        Exporter::new("rerank", "text-ranking", export_reranker_rlxp).architectures(&[
            "BertForSequenceClassification",
            "XLMRobertaForSequenceClassification",
            "RobertaForSequenceClassification",
        ]),
    );
}
