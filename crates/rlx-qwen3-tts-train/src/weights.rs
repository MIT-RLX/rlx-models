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

//! Talker backbone + LoRA weights (mmap selective load, single open).

use anyhow::{Context, Result};
use rlx_qwen3_tts::load::{Qwen3TtsWeightStore, remap_talker_weights};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct WeightStore(pub std::collections::HashMap<String, Vec<f32>>);

impl WeightStore {
    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }
}

pub fn init_lora_param(
    name: &str,
    rank: usize,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn: usize,
) -> Vec<f32> {
    let (rows, cols) = lora_shape(name, rank, h, q_dim, kv_dim, ffn);
    let scale = 1.0 / (rank as f32).sqrt();
    (0..rows * cols)
        .map(|i| {
            let x = ((i as f32 * 0.713) % 1.0) - 0.5;
            x * scale * 0.02
        })
        .collect()
}

fn lora_shape(
    name: &str,
    rank: usize,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn: usize,
) -> (usize, usize) {
    if name.ends_with("_a") {
        if name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("gate_proj")
            || name.contains("up_proj")
        {
            (rank, h)
        } else if name.contains("o_proj") {
            (rank, q_dim)
        } else if name.contains("down_proj") {
            (rank, ffn)
        } else {
            (rank, h)
        }
    } else if name.contains("q_proj") {
        (rank, q_dim)
    } else if name.contains("k_proj") || name.contains("v_proj") {
        (rank, kv_dim)
    } else if name.contains("o_proj") {
        (rank, h)
    } else if name.contains("gate_proj") || name.contains("up_proj") {
        (rank, ffn)
    } else {
        (rank, h)
    }
}

fn graph_param_to_hf_key(param: &str) -> String {
    format!("talker.{param}")
}

/// Load only tensors referenced by the LoRA graph (last *n* layers), not the full talker.
pub fn load_talker_backbone_from_store(
    store: &Qwen3TtsWeightStore,
    param_names: &[String],
) -> Result<WeightStore> {
    let mut hf: HashSet<String> = HashSet::new();
    for name in param_names {
        if name.starts_with("lora.") || name.starts_with("rope.") || name == "__zero" {
            continue;
        }
        hf.insert(graph_param_to_hf_key(name));
    }
    let mut wm = store.load_selected(&hf)?;
    let map = remap_talker_weights(&mut wm)?;
    let mut out = WeightStore::default();
    for name in param_names {
        if name.starts_with("lora.") || name.starts_with("rope.") || name == "__zero" {
            continue;
        }
        let (data, shape) = map
            .get(name.as_str())
            .with_context(|| format!("missing talker param {name}"))?;
        let (data, _) = if shape.len() == 2 {
            hf_linear_to_rlx(data, shape)?
        } else {
            (data.clone(), shape.clone())
        };
        out.0.insert(name.clone(), data);
    }
    Ok(out)
}

/// HF safetensors linear weights are `[out_features, in_features]`; RLX matmul uses `[in, out]`.
fn hf_linear_to_rlx(data: &[f32], shape: &[usize]) -> Result<(Vec<f32>, Vec<usize>)> {
    let (out_f, in_f) = (shape[0], shape[1]);
    let mut t = vec![0f32; out_f * in_f];
    for o in 0..out_f {
        for i in 0..in_f {
            t[i * out_f + o] = data[o * in_f + i];
        }
    }
    Ok((t, vec![in_f, out_f]))
}

pub fn load_lora_backbone_for_graph(
    model_dir: &Path,
    graph_param_names: &[String],
) -> Result<WeightStore> {
    let store = Qwen3TtsWeightStore::open(model_dir)?;
    load_talker_backbone_from_store(&store, graph_param_names)
}
