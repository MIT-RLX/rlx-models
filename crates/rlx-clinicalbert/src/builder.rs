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

//! ClinicalBERT graph construction — delegates to [`rlx_bert::flow::BertFlow`].

use anyhow::{Context, Result, bail};
use rlx_bert::flow::BertFlow;
use rlx_core::config::BertConfig;
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::BuiltModel;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use std::collections::HashMap;

/// Build the ClinicalBERT encoder flow.
///
/// ClinicalBERT shares the BERT-base layout: token / position / type
/// embeddings → 12 transformer layers (multi-head self-attention + GeLU FFN)
/// with post-LN. [`BertFlow`] auto-detects the `bert.` weight prefix used by
/// HuggingFace `BertModel` checkpoints (including all ClinicalBERT variants).
pub fn build_clinicalbert_built(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    BertFlow::new(cfg, batch, seq).build(weights)
}

/// Build a ClinicalBERT encoder graph + parameter map sized for `(batch, seq)`.
///
/// Returns `(graph, params)` where `params` maps IR param names to F32 weight
/// data ready for `Session::set_param`.
///
/// Output: `hidden_states [batch, seq, hidden_size]`.
pub fn build_clinicalbert_graph(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    graph_from_built(build_clinicalbert_built(cfg, weights, batch, seq)?)
}

/// Build the encoder graph with the HF `BertLMPredictionHead` appended as a
/// second output (`dense(H→H) + GeLU + LN + tied decoder(H→V) + bias`).
///
/// The decoder weight is taken from `cls.predictions.decoder.weight` when
/// present, otherwise tied to `bert.embeddings.word_embeddings.weight` (the
/// common HF layout). Returns a [`BuiltModel`] whose graph emits
/// `[hidden_states, mlm_logits]`. See [`crate::MlmExecMode`] for when this
/// path beats the CPU post-process head.
pub fn build_clinicalbert_with_mlm_built(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    // Detect prefix (Bio_ClinicalBERT and Huang both use `bert.`).
    let prefix = if weights.has("bert.embeddings.word_embeddings.weight") {
        "bert."
    } else {
        ""
    };

    // Clone the input embedding matrix BEFORE the encoder build — the
    // BertFlow consumes it via `take(..)`, so the tied decoder needs a
    // private copy stashed away first. Only needed when there's no
    // explicit `cls.predictions.decoder.weight` (the common HF layout).
    let tied_decoder_w: Option<(Vec<f32>, Vec<usize>)> =
        if weights.has("cls.predictions.decoder.weight") {
            None
        } else {
            let key = format!("{prefix}embeddings.word_embeddings.weight");
            let (data, shape) = weights
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: tied MLM decoder needs {key}"))?;
            Some((data.to_vec(), shape.to_vec()))
        };

    // 1. Build the encoder. After this WeightMap still holds the cls.*
    //    tensors but no longer has the embedding matrix.
    let built_encoder = BertFlow::new(cfg, batch, seq).build(weights)?;
    let profile = built_encoder.profile().clone();
    let (mut graph, mut params) = graph_from_built(built_encoder)?;

    let hidden_id = *graph.outputs.last().ok_or_else(|| {
        anyhow::anyhow!("build_clinicalbert_with_mlm_built: encoder has no outputs")
    })?;

    // 2. Pull MLM head weights from the WeightMap.
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let eps = cfg.layer_norm_eps as f32;
    let f = DType::F32;

    let (transform_w, transform_w_shape) =
        weights
            .take("cls.predictions.transform.dense.weight")
            .context("loading cls.predictions.transform.dense.weight")?;
    let transform_b = weights
        .take("cls.predictions.transform.dense.bias")
        .context("loading cls.predictions.transform.dense.bias")?
        .0;
    let ln_w = weights
        .take("cls.predictions.transform.LayerNorm.weight")
        .context("loading cls.predictions.transform.LayerNorm.weight")?
        .0;
    let ln_b = weights
        .take("cls.predictions.transform.LayerNorm.bias")
        .context("loading cls.predictions.transform.LayerNorm.bias")?
        .0;
    let decoder_b = if weights.has("cls.predictions.bias") {
        weights
            .take("cls.predictions.bias")
            .context("loading cls.predictions.bias")?
            .0
    } else if weights.has("cls.predictions.decoder.bias") {
        weights
            .take("cls.predictions.decoder.bias")
            .context("loading cls.predictions.decoder.bias")?
            .0
    } else {
        bail!("rlx-clinicalbert: MLM bias missing (cls.predictions.bias / .decoder.bias)");
    };
    if decoder_b.len() != v {
        bail!(
            "rlx-clinicalbert: MLM bias length {} != vocab_size {v}",
            decoder_b.len()
        );
    }

    // Tied decoder weight: use the explicit `cls.predictions.decoder.weight`
    // when present (rare), otherwise the embedding-matrix clone we stashed
    // before the encoder build above.
    let (decoder_w_raw, decoder_w_shape) = if weights.has("cls.predictions.decoder.weight") {
        weights
            .take("cls.predictions.decoder.weight")
            .context("loading cls.predictions.decoder.weight")?
    } else {
        tied_decoder_w
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: tied decoder clone missing"))?
    };

    // HF Linear stores weights as `[out, in]`; IR matmul wants `[in, out]`.
    let transform_w_t = transpose(&transform_w, transform_w_shape[0], transform_w_shape[1]);
    let decoder_w_t = transpose(&decoder_w_raw, decoder_w_shape[0], decoder_w_shape[1]);

    // 3. Register MLM head params on the graph.
    let transform_w_id = graph.param("mlm.transform.weight_t", Shape::new(&[h, h], f));
    let transform_b_id = graph.param("mlm.transform.bias", Shape::new(&[h], f));
    let ln_w_id = graph.param("mlm.transform.LayerNorm.weight", Shape::new(&[h], f));
    let ln_b_id = graph.param("mlm.transform.LayerNorm.bias", Shape::new(&[h], f));
    let decoder_w_id = graph.param("mlm.decoder.weight_t", Shape::new(&[h, v], f));
    let decoder_b_id = graph.param("mlm.decoder.bias", Shape::new(&[v], f));
    params.insert("mlm.transform.weight_t".into(), transform_w_t);
    params.insert("mlm.transform.bias".into(), transform_b);
    params.insert("mlm.transform.LayerNorm.weight".into(), ln_w);
    params.insert("mlm.transform.LayerNorm.bias".into(), ln_b);
    params.insert("mlm.decoder.weight_t".into(), decoder_w_t);
    params.insert("mlm.decoder.bias".into(), decoder_b);

    // 4. Append the head ops: matmul + bias + GeLU + LayerNorm + matmul + bias.
    let bsh = Shape::new(&[batch, seq, h], f);
    let bsv = Shape::new(&[batch, seq, v], f);

    let mm1 = graph.matmul(hidden_id, transform_w_id, bsh.clone());
    let mm1_bias = graph.binary(BinaryOp::Add, mm1, transform_b_id, bsh.clone());
    let gelu = graph.activation(Activation::Gelu, mm1_bias, bsh.clone());
    // LayerNorm normalizes the last (feature) axis: axis = -1 = h-axis index.
    let normalized = graph.layer_norm(gelu, ln_w_id, ln_b_id, -1, eps, bsh.clone());
    let mm2 = graph.matmul(normalized, decoder_w_id, bsv.clone());
    let mlm_logits = graph.binary(BinaryOp::Add, mm2, decoder_b_id, bsv);

    // 5. Emit both encoder and head outputs. Compile profile + typed_params
    //    carry over from the encoder build.
    graph.set_outputs(vec![hidden_id, mlm_logits]);
    if std::env::var("RLX_CLINICALBERT_DEBUG").is_ok() {
        eprintln!(
            "[rlx-clinicalbert::builder] graph.outputs = {:?}",
            graph
                .outputs
                .iter()
                .map(|&id| {
                    let shape = &graph.node(id).shape;
                    (id, shape.dims().to_vec(), shape.num_elements())
                })
                .collect::<Vec<_>>()
        );
    }
    let mut built = BuiltModel::from_graph(graph, params)?;
    built.profile = profile;
    Ok(built)
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
