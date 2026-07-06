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

//! Test-only fixtures. Compiled under `#[cfg(test)]` only.

use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

/// Build a minimal all-zeros BERT checkpoint (BERT-style keys) sized from
/// `cfg` — embeddings plus `cfg.num_hidden_layers` encoder layers. Enough
/// tensors for the graph builder to succeed; shapes match the HF layout so
/// `take`/`take_transposed` line up.
pub(crate) fn tiny_bert_weights(cfg: &BertConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let int = cfg.intermediate_size;

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut add = |key: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        tensors.insert(key, (vec![0.0f32; n], shape));
    };

    // Embeddings.
    add(
        "embeddings.word_embeddings.weight".into(),
        vec![cfg.vocab_size, h],
    );
    add(
        "embeddings.position_embeddings.weight".into(),
        vec![cfg.max_position_embeddings, h],
    );
    add(
        "embeddings.token_type_embeddings.weight".into(),
        vec![cfg.type_vocab_size, h],
    );
    add("embeddings.LayerNorm.weight".into(), vec![h]);
    add("embeddings.LayerNorm.bias".into(), vec![h]);

    // Encoder layers.
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("encoder.layer.{i}");
        for proj in ["query", "key", "value"] {
            add(format!("{lp}.attention.self.{proj}.weight"), vec![h, h]);
            add(format!("{lp}.attention.self.{proj}.bias"), vec![h]);
        }
        add(format!("{lp}.attention.output.dense.weight"), vec![h, h]);
        add(format!("{lp}.attention.output.dense.bias"), vec![h]);
        add(format!("{lp}.attention.output.LayerNorm.weight"), vec![h]);
        add(format!("{lp}.attention.output.LayerNorm.bias"), vec![h]);
        add(format!("{lp}.intermediate.dense.weight"), vec![int, h]);
        add(format!("{lp}.intermediate.dense.bias"), vec![int]);
        add(format!("{lp}.output.dense.weight"), vec![h, int]);
        add(format!("{lp}.output.dense.bias"), vec![h]);
        add(format!("{lp}.output.LayerNorm.weight"), vec![h]);
        add(format!("{lp}.output.LayerNorm.bias"), vec![h]);
    }

    WeightMap::from_tensors(tensors)
}

/// A `WeightMap` containing exactly `keys` with empty payloads — for
/// presence/detection checks (`weights.has(...)`) where only the names matter.
pub(crate) fn weights_with_keys(keys: &[&str]) -> WeightMap {
    let tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = keys
        .iter()
        .map(|k| (k.to_string(), (Vec::new(), Vec::new())))
        .collect();
    WeightMap::from_tensors(tensors)
}
