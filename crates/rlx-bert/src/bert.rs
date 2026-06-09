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

//! BERT graph builder — constructs RLX IR from config + weights.

use anyhow::Result;
use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::*;
use std::collections::HashMap;

/// Build a BERT encoder IR graph from config and weights.
///
/// Returns the graph and a map of param_name → weight data.
/// The graph expects inputs: `input_ids [B,S]`, `attention_mask [B,S]`, `token_type_ids [B,S]`.
/// Output: `hidden_states [B, S, H]`.
/// Build a BERT encoder IR graph.
///
/// `batch` and `seq` are the concrete dimensions for this compilation.
/// The graph will be compiled for exactly these dimensions.
/// Call again with different dims to recompile for a different size.
pub fn build_bert_graph(
    cfg: &BertConfig,
    weights: &mut WeightMap,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_bert_graph_sized(cfg, weights, 1, 1)
}

pub fn build_bert_graph_sized(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(crate::flow::build_bert_built(cfg, weights, batch, seq)?)
}

/// Load a parameter: register in graph + store weight data.
#[allow(dead_code)]
fn load_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    key: &str,
    _expected_shape: &[usize],
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

/// Fuse Q/K/V weights into single [H, 3H] matrix (BERT-style keys).
#[allow(dead_code)]
fn load_fused_qkv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    layer_prefix: &str,
    h: usize,
    _nh: usize,
    _dh: usize,
) -> Result<(NodeId, NodeId)> {
    let (wq, _) =
        weights.take_transposed(&format!("{layer_prefix}.attention.self.query.weight"))?;
    let (wk, _) = weights.take_transposed(&format!("{layer_prefix}.attention.self.key.weight"))?;
    let (wv, _) =
        weights.take_transposed(&format!("{layer_prefix}.attention.self.value.weight"))?;

    let bq = weights
        .take(&format!("{layer_prefix}.attention.self.query.bias"))?
        .0;
    let bk = weights
        .take(&format!("{layer_prefix}.attention.self.key.bias"))?
        .0;
    let bv = weights
        .take(&format!("{layer_prefix}.attention.self.value.bias"))?
        .0;

    // Concatenate: [H, H] + [H, H] + [H, H] → [H, 3H]
    let mut fused_w = vec![0f32; h * 3 * h];
    let mut fused_b = vec![0f32; 3 * h];
    for row in 0..h {
        fused_w[row * 3 * h..row * 3 * h + h].copy_from_slice(&wq[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + h..row * 3 * h + 2 * h].copy_from_slice(&wk[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + 2 * h..row * 3 * h + 3 * h]
            .copy_from_slice(&wv[row * h..(row + 1) * h]);
    }
    fused_b[..h].copy_from_slice(&bq);
    fused_b[h..2 * h].copy_from_slice(&bk);
    fused_b[2 * h..].copy_from_slice(&bv);

    let w_name = format!("{layer_prefix}.attention.qkv.weight");
    let b_name = format!("{layer_prefix}.attention.qkv.bias");
    let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
    let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
    params.insert(w_name, fused_w);
    params.insert(b_name, fused_b);

    Ok((w_id, b_id))
}

/// mpnet-style QKV fusion (different key names).
#[allow(dead_code)]
fn load_fused_qkv_mpnet(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    layer_prefix: &str,
    h: usize,
    nh: usize,
    dh: usize,
) -> Result<(NodeId, NodeId)> {
    // Try mpnet keys
    let q_key = format!("{layer_prefix}.attention.attn.q.weight");
    if weights.has(&q_key) {
        let (wq, _) = weights.take_transposed(&q_key)?;
        let (wk, _) =
            weights.take_transposed(&format!("{layer_prefix}.attention.attn.k.weight"))?;
        let (wv, _) =
            weights.take_transposed(&format!("{layer_prefix}.attention.attn.v.weight"))?;
        let bq = weights
            .take(&format!("{layer_prefix}.attention.attn.q.bias"))?
            .0;
        let bk = weights
            .take(&format!("{layer_prefix}.attention.attn.k.bias"))?
            .0;
        let bv = weights
            .take(&format!("{layer_prefix}.attention.attn.v.bias"))?
            .0;

        let mut fused_w = vec![0f32; h * 3 * h];
        let mut fused_b = vec![0f32; 3 * h];
        for row in 0..h {
            fused_w[row * 3 * h..row * 3 * h + h].copy_from_slice(&wq[row * h..(row + 1) * h]);
            fused_w[row * 3 * h + h..row * 3 * h + 2 * h]
                .copy_from_slice(&wk[row * h..(row + 1) * h]);
            fused_w[row * 3 * h + 2 * h..row * 3 * h + 3 * h]
                .copy_from_slice(&wv[row * h..(row + 1) * h]);
        }
        fused_b[..h].copy_from_slice(&bq);
        fused_b[h..2 * h].copy_from_slice(&bk);
        fused_b[2 * h..].copy_from_slice(&bv);

        let w_name = format!("{layer_prefix}.attention.qkv.weight");
        let b_name = format!("{layer_prefix}.attention.qkv.bias");
        let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
        let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
        params.insert(w_name, fused_w);
        params.insert(b_name, fused_b);
        return Ok((w_id, b_id));
    }

    // Fallback: already-fused QKV
    let fused_key = format!("{layer_prefix}.attention.self.qkv.weight");
    if weights.has(&fused_key) {
        let (data, _) = weights.take_transposed(&fused_key)?;
        let bias = weights
            .take(&format!("{layer_prefix}.attention.self.qkv.bias"))?
            .0;
        let w_name = format!("{layer_prefix}.attention.qkv.weight");
        let b_name = format!("{layer_prefix}.attention.qkv.bias");
        let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
        let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
        params.insert(w_name, data);
        params.insert(b_name, bias);
        return Ok((w_id, b_id));
    }

    // Fallback to BERT style
    load_fused_qkv(g, params, weights, layer_prefix, h, nh, dh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tiny_bert_graph() {
        // Create a minimal config
        let cfg = BertConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 256,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        };

        // Create fake weights
        let h = cfg.hidden_size;
        let int = cfg.intermediate_size;
        let mut tensors = HashMap::new();
        let add = |m: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
            let size: usize = shape.iter().product();
            m.insert(k.to_string(), (vec![0.01f32; size], shape));
        };

        // Embeddings
        add(
            &mut tensors,
            "embeddings.word_embeddings.weight",
            vec![100, h],
        );
        add(
            &mut tensors,
            "embeddings.position_embeddings.weight",
            vec![32, h],
        );
        add(
            &mut tensors,
            "embeddings.token_type_embeddings.weight",
            vec![2, h],
        );
        add(&mut tensors, "embeddings.LayerNorm.weight", vec![h]);
        add(&mut tensors, "embeddings.LayerNorm.bias", vec![h]);

        // Layer 0 — attention
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.query.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.query.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.key.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.key.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.value.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.value.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.dense.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.dense.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.LayerNorm.weight",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.LayerNorm.bias",
            vec![h],
        );

        // Layer 0 — FFN
        add(
            &mut tensors,
            "encoder.layer.0.intermediate.dense.weight",
            vec![int, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.intermediate.dense.bias",
            vec![int],
        );
        add(
            &mut tensors,
            "encoder.layer.0.output.dense.weight",
            vec![h, int],
        );
        add(&mut tensors, "encoder.layer.0.output.dense.bias", vec![h]);
        add(
            &mut tensors,
            "encoder.layer.0.output.LayerNorm.weight",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.output.LayerNorm.bias",
            vec![h],
        );

        let mut wm = WeightMap::from_tensors(tensors);
        let (graph, params) = build_bert_graph(&cfg, &mut wm).unwrap();

        println!("{graph}");
        println!("Nodes: {}, Params: {}", graph.len(), params.len());

        // Verify graph is valid
        let errors = rlx_ir::verify::verify(&graph);
        assert!(errors.is_empty(), "verification errors: {errors:?}");

        // Should have params for all weights
        assert!(
            params.len() >= 15,
            "expected 15+ params, got {}",
            params.len()
        );

        // Output should exist
        assert!(!graph.outputs.is_empty());
    }
}
