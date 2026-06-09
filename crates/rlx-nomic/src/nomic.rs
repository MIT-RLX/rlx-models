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

//! NomicBERT graph builder — RoPE + SwiGLU + no bias.
//!
//! Production builds go through [`crate::flow::NomicFlow`] (native
//! [`ModelFlow`]). The functions below remain for diagnostics only.

use anyhow::Result;
use rlx_core::config::NomicBertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

type DiagnosticGraphParts = (Graph, HashMap<String, Vec<f32>>, Vec<String>);

/// Build a NomicBERT encoder IR graph.
pub fn build_nomic_graph_sized(
    cfg: &NomicBertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(crate::flow::build_nomic_built(cfg, weights, batch, seq)?)
}

/// Diagnostic builder — same as `build_nomic_graph_sized` but exposes
/// intermediate tensors at every transformer-stage boundary as outputs.
/// Returns (graph, params, checkpoint_names) where outputs\[i\] holds the
/// tensor at checkpoint_names\[i\]. Used by examples/tests to bisect
/// numerical issues (NaN/Inf) without instrumenting the executor.
///
/// `max_layers` caps the number of transformer layers built (0 = full
/// model). Restricting to 1–2 layers keeps the output volume manageable
/// when chasing the *first* divergence.
pub fn build_nomic_diagnostic_graph(
    cfg: &NomicBertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
    max_layers: usize,
) -> Result<DiagnosticGraphParts> {
    let mut g = Graph::new("nomic_diagnose");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut checkpoints: Vec<NodeId> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim;
    let int_dim = cfg.intermediate_size;
    let eps = cfg.layer_norm_eps as f32;

    let word_emb = load_p(
        &mut g,
        &mut params,
        weights,
        "embeddings.word_embeddings.weight",
        &[cfg.vocab_size, h],
        false,
    )?;
    let tt_emb = load_p(
        &mut g,
        &mut params,
        weights,
        "embeddings.token_type_embeddings.weight",
        &[cfg.type_vocab_size, h],
        false,
    )?;
    let emb_ln_g = load_p(&mut g, &mut params, weights, "emb_ln.weight", &[h], false)?;
    let emb_ln_b = load_p(&mut g, &mut params, weights, "emb_ln.bias", &[h], false)?;

    let half = dh / 2;
    let mut cos_data = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin_data = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rotary_emb_base.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (sn, cs) = angle.sin_cos();
            cos_data[pos * half + i] = cs as f32;
            sin_data[pos * half + i] = sn as f32;
        }
    }
    let cos_id = g.param(
        "rope.cos",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param(
        "rope.sin",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let attn_mask = g.input("attention_mask", Shape::new(&[batch, seq], f));
    let token_type_ids = g.input("token_type_ids", Shape::new(&[batch, seq], DType::F32));

    let word_out = g.gather_(word_emb, input_ids, 0);
    checkpoints.push(word_out);
    names.push("word_emb_lookup".into());
    let tt_out = g.gather_(tt_emb, token_type_ids, 0);
    checkpoints.push(tt_out);
    names.push("token_type_emb_lookup".into());
    let emb_sum = g.add(word_out, tt_out);
    checkpoints.push(emb_sum);
    names.push("emb_sum_word_plus_tt".into());
    let hidden0 = g.ln(emb_sum, emb_ln_g, emb_ln_b, eps);
    checkpoints.push(hidden0);
    names.push("after_embedding_ln".into());

    let layers = if max_layers == 0 {
        cfg.num_hidden_layers
    } else {
        max_layers.min(cfg.num_hidden_layers)
    };
    let mut h_id = hidden0;
    for layer_idx in 0..layers {
        let lp = format!("encoder.layers.{layer_idx}");
        let qkv_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.Wqkv.weight"),
            &[h, 3 * h],
            true,
        )?;
        let qkv = g.mm(h_id, qkv_w);
        checkpoints.push(qkv);
        names.push(format!("L{layer_idx}_qkv_mm"));

        let last_ax = g.shape(qkv).rank() - 1;
        let q = g.narrow_(qkv, last_ax, 0, h);
        let k = g.narrow_(qkv, last_ax, h, h);
        let v = g.narrow_(qkv, last_ax, 2 * h, h);
        checkpoints.push(q);
        names.push(format!("L{layer_idx}_q_narrow"));
        checkpoints.push(k);
        names.push(format!("L{layer_idx}_k_narrow"));
        checkpoints.push(v);
        names.push(format!("L{layer_idx}_v_narrow"));

        let q_rope = g.rope(q, cos_id, sin_id, dh);
        let k_rope = g.rope(k, cos_id, sin_id, dh);
        checkpoints.push(q_rope);
        names.push(format!("L{layer_idx}_q_rope"));
        checkpoints.push(k_rope);
        names.push(format!("L{layer_idx}_k_rope"));

        let attn = g.attention_(q_rope, k_rope, v, attn_mask, nh, dh);
        checkpoints.push(attn);
        names.push(format!("L{layer_idx}_attention"));

        let out_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.out_proj.weight"),
            &[h, h],
            true,
        )?;
        let attn_out = g.mm(attn, out_w);
        checkpoints.push(attn_out);
        names.push(format!("L{layer_idx}_out_proj"));

        let ln1_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.weight"),
            &[h],
            false,
        )?;
        let ln1_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.bias"),
            &[h],
            false,
        )?;
        let res1 = g.add(attn_out, h_id);
        let normed1 = g.ln(res1, ln1_g, ln1_b, eps);
        checkpoints.push(normed1);
        names.push(format!("L{layer_idx}_after_ln1"));

        let fc11_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc11.weight"),
            &[h, int_dim],
            true,
        )?;
        let fc12_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc12.weight"),
            &[h, int_dim],
            true,
        )?;
        let fc2_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc2.weight"),
            &[int_dim, h],
            true,
        )?;

        let up = g.mm(normed1, fc11_w);
        let gate_mm = g.mm(normed1, fc12_w);
        let gate = g.silu(gate_mm);
        let swiglu = g.mul(up, gate);
        checkpoints.push(swiglu);
        names.push(format!("L{layer_idx}_swiglu"));
        let ffn_out = g.mm(swiglu, fc2_w);
        checkpoints.push(ffn_out);
        names.push(format!("L{layer_idx}_ffn_out"));

        let ln2_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.weight"),
            &[h],
            false,
        )?;
        let ln2_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.bias"),
            &[h],
            false,
        )?;
        let res2 = g.add(ffn_out, normed1);
        h_id = g.ln(res2, ln2_g, ln2_b, eps);
        checkpoints.push(h_id);
        names.push(format!("L{layer_idx}_layer_out"));
    }

    g.set_outputs(checkpoints);
    Ok((g, params, names))
}

fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    key: &str,
    _shape: &[usize],
    transpose: bool,
) -> Result<NodeId> {
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let name = key.to_string();
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = g.param(&name, ir_shape);
    params.insert(name, data);
    Ok(id)
}
