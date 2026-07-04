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

//! BERT graph builder — lowers a [`BertConfig`] + weights into an RLX IR graph.
//!
//! Thin shim over [`crate::flow::BertFlow`]: it assembles the encoder with the
//! shared `rlx_flow::ModelFlow` DSL and lowers the resulting `BuiltModel` to a
//! concrete [`Graph`]. See the crate README for the full architecture and the
//! end-to-end run path.

use anyhow::Result;
use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Graph;
use std::collections::HashMap;

/// Build a BERT encoder IR graph for concrete `batch` × `seq` dimensions.
///
/// Returns the lowered [`Graph`] plus a `param_name → f32 weights` map ready to
/// bind on a compiled session. The graph takes inputs `input_ids`,
/// `attention_mask`, `token_type_ids`, and `position_ids` (each `[batch, seq]`)
/// and produces `hidden_states` `[batch, seq, hidden_size]`.
///
/// Dimensions are baked into the graph — call again with different dims to
/// build for another shape. To get a `BuiltModel` instead (e.g. to fold a task
/// head on before lowering), use [`crate::build_bert_built`] or
/// [`crate::flow::BertFlow`].
pub fn build_bert_graph_sized(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    rlx_core::flow_util::graph_from_built(crate::flow::build_bert_built(cfg, weights, batch, seq)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tiny_bert_graph() {
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
        let mut wm = crate::test_support::tiny_bert_weights(&cfg);
        let (graph, params) = build_bert_graph_sized(&cfg, &mut wm, 1, 1).unwrap();

        // The lowered graph passes IR verification and has a full param set + output.
        let errors = rlx_ir::verify::verify(&graph);
        assert!(errors.is_empty(), "verification errors: {errors:?}");
        assert!(
            params.len() >= 15,
            "expected 15+ params, got {}",
            params.len()
        );
        assert!(!graph.outputs.is_empty());
    }
}
