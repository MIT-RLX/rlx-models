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

//! RLX-backed text and image embedding models.
//!
//! Migrated from `burnembed` — compiles BERT / NomicBERT / NomicVision graphs
//! via `rlx-runtime` and exposes tier-0 inference helpers.
//!
//! ```rust,ignore
//! use rlx_models::embed::{Pooling, RlxBertModel, BertTokenizer, embed_with_rlx};
//!
//! let tok = BertTokenizer::from_dir(model_dir, 512)?;
//! let mut model = RlxBertModel::load(&config, &weights)?;
//! let vecs = embed_with_rlx(&mut model, &tok, &["hello", "world"], Pooling::Mean)?;
//! ```

mod arch;
mod bert;
mod nomic;
mod pooling;
mod registry;
mod reranker;
mod runtime;
mod text;
mod tokenizer;
mod vision;

pub use arch::{Arch, default_pooling, detect_arch};
pub use bert::RlxBertModel;
pub use nomic::RlxNomicModel;
pub use pooling::{Pooling, l2_normalize_in_place, pool_embeddings};
pub use registry::{
    EmbeddingModel, ImageEmbeddingModel, ImageModelInfo, ModelArch, ModelInfo, models_map,
};
pub use reranker::RlxReranker;
pub use runtime::{RlxEmbed, compile_model, compile_model_cpu};
pub use text::embed_with_rlx;
pub use tokenizer::{BertTokenizer, TokenizedBatch};
pub use vision::{RlxVisionModel, assemble_vision_hidden};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn tiny_bert_cfg() -> rlx_core::config::BertConfig {
        rlx_core::config::BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        }
    }

    fn tiny_bert_weights(cfg: &rlx_core::config::BertConfig) -> WeightMap {
        let h = cfg.hidden_size;
        let int_dim = cfg.intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "embeddings.word_embeddings.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        t.insert(
            "embeddings.position_embeddings.weight".into(),
            (
                z(cfg.max_position_embeddings * h),
                vec![cfg.max_position_embeddings, h],
            ),
        );
        t.insert(
            "embeddings.token_type_embeddings.weight".into(),
            (z(cfg.type_vocab_size * h), vec![cfg.type_vocab_size, h]),
        );
        t.insert("embeddings.LayerNorm.weight".into(), (z(h), vec![h]));
        t.insert("embeddings.LayerNorm.bias".into(), (z(h), vec![h]));
        let lp = "encoder.layer.0";
        t.insert(
            format!("{lp}.attention.self.query.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.query.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.self.key.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.key.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.self.value.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.value.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.output.dense.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.output.dense.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.output.LayerNorm.weight"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.attention.output.LayerNorm.bias"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.intermediate.dense.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.intermediate.dense.bias"),
            (z(int_dim), vec![int_dim]),
        );
        t.insert(
            format!("{lp}.output.dense.weight"),
            (z(h * int_dim), vec![h, int_dim]),
        );
        t.insert(format!("{lp}.output.dense.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.output.LayerNorm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.output.LayerNorm.bias"), (z(h), vec![h]));
        t.insert("pooler.dense.weight".into(), (z(h * h), vec![h, h]));
        t.insert("pooler.dense.bias".into(), (z(h), vec![h]));
        WeightMap::from_tensors(t)
    }

    #[test]
    fn rlx_bert_graph_builds() {
        let cfg = tiny_bert_cfg();
        let mut wm = tiny_bert_weights(&cfg);
        let (graph, params) = rlx_bert::bert::build_bert_graph_sized(&cfg, &mut wm, 1, 4).unwrap();
        assert_eq!(graph.outputs.len(), 1);
        assert!(!params.is_empty());
    }

    #[test]
    fn registry_lists_models() {
        assert!(!EmbeddingModel::list_supported().is_empty());
        assert!(EmbeddingModel::AllMiniLML6V2.get_info().is_some());
    }
}
