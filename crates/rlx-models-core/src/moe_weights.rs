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

//! GGUF MoE expert-stack loader.
//!
//! Companion to `rlx_flow::blocks::MoeFfnStage`. Loads the per-layer
//! `ffn_{gate,up,down}_exps.weight` stacked tensors that llama.cpp's
//! GGUF converters ship for Mixtral / Qwen3-MoE / Gemma 4 MoE / etc.,
//! and validates the `[num_experts, k, n]` shape contract the MoE
//! block expects.
//!
//! Per-family code paths (e.g. `rlx-qwen35::weights`) keep their own
//! loaders that also handle GGUF K-quant packed slabs; this module is
//! the **f32-dequant generic** path that's portable across families
//! that don't (yet) need packed routing.

use anyhow::{Context, Result, anyhow};

use crate::weight_loader::WeightLoader;

/// One layer's stacked MoE FFN tensors. Shapes match the contract of
/// [`rlx_flow::blocks::MoeFfnStage`] (and the underlying
/// `Op::GroupedMatMul`):
///
/// * `gate` / `up`: `[num_experts, n_embd, n_ff]`
/// * `down`:       `[num_experts, n_ff, n_embd]`
#[derive(Debug, Clone)]
pub struct MoeLayerWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub router: Vec<f32>,
    pub num_experts: usize,
    pub n_embd: usize,
    pub n_ff: usize,
}

/// GGUF-tensor naming for one MoE layer. Defaults follow llama.cpp's
/// `qwen2moe` / `qwen3moe` / `gemma4moe` converters:
///
/// * router: `blk.{layer}.ffn_gate_inp.weight`
/// * gate:   `blk.{layer}.ffn_gate_exps.weight`
/// * up:     `blk.{layer}.ffn_up_exps.weight`
/// * down:   `blk.{layer}.ffn_down_exps.weight`
#[derive(Debug, Clone)]
pub struct MoeLayerKeys {
    pub router: String,
    pub gate: String,
    pub up: String,
    pub down: String,
}

impl MoeLayerKeys {
    /// llama.cpp default convention.
    pub fn llama_cpp(layer_idx: usize) -> Self {
        let p = format!("blk.{layer_idx}");
        Self {
            router: format!("{p}.ffn_gate_inp.weight"),
            gate: format!("{p}.ffn_gate_exps.weight"),
            up: format!("{p}.ffn_up_exps.weight"),
            down: format!("{p}.ffn_down_exps.weight"),
        }
    }

    /// HuggingFace convention (`model.layers.{i}.block_sparse_moe.*` /
    /// `mlp.experts.*`). When the loader doesn't carry an MoE-specific
    /// HF→GGUF tensor-name resolver, callers usually want
    /// [`Self::llama_cpp`] instead since GGUF-on-disk uses the llama.cpp
    /// names.
    pub fn hf_block_sparse(layer_idx: usize) -> Self {
        let p = format!("model.layers.{layer_idx}.block_sparse_moe");
        Self {
            router: format!("{p}.gate.weight"),
            // HF stores per-expert separately — this loader expects the
            // stacked variant. Callers who only have HF tensors should
            // pre-stack them with `stack_expert_tensors` first.
            gate: format!("{p}.experts.gate_proj.weight"),
            up: format!("{p}.experts.up_proj.weight"),
            down: format!("{p}.experts.down_proj.weight"),
        }
    }
}

/// Load `[num_experts, k, n]` f32 expert stack from a loader, verifying
/// shape.
pub fn load_expert_stack(
    loader: &mut dyn WeightLoader,
    key: &str,
    num_experts: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let (data, shape) = loader
        .take(key)
        .with_context(|| format!("MoE expert stack `{key}`"))?;
    let expected = vec![num_experts, k, n];
    if shape != expected {
        return Err(anyhow!(
            "MoE expert stack `{key}`: expected shape {expected:?}, got {shape:?}"
        ));
    }
    let expected_len = num_experts * k * n;
    if data.len() != expected_len {
        return Err(anyhow!(
            "MoE expert stack `{key}`: shape {shape:?} declares \
             {expected_len} elements but loader returned {}",
            data.len()
        ));
    }
    Ok(data)
}

/// Load router weight `[n_embd, num_experts]` f32.
pub fn load_router(
    loader: &mut dyn WeightLoader,
    key: &str,
    n_embd: usize,
    num_experts: usize,
) -> Result<Vec<f32>> {
    let (data, shape) = loader
        .take(key)
        .with_context(|| format!("MoE router `{key}`"))?;
    let expected = vec![n_embd, num_experts];
    if shape != expected {
        return Err(anyhow!(
            "MoE router `{key}`: expected shape {expected:?}, got {shape:?}"
        ));
    }
    if data.len() != n_embd * num_experts {
        return Err(anyhow!(
            "MoE router `{key}`: data len {} != n_embd*num_experts ({})",
            data.len(),
            n_embd * num_experts
        ));
    }
    Ok(data)
}

/// Convenience: load all 4 tensors for one MoE layer at once.
pub fn load_layer(
    loader: &mut dyn WeightLoader,
    keys: &MoeLayerKeys,
    num_experts: usize,
    n_embd: usize,
    n_ff: usize,
) -> Result<MoeLayerWeights> {
    let router = load_router(loader, &keys.router, n_embd, num_experts)?;
    let gate = load_expert_stack(loader, &keys.gate, num_experts, n_embd, n_ff)?;
    let up = load_expert_stack(loader, &keys.up, num_experts, n_embd, n_ff)?;
    let down = load_expert_stack(loader, &keys.down, num_experts, n_ff, n_embd)?;
    Ok(MoeLayerWeights {
        gate,
        up,
        down,
        router,
        num_experts,
        n_embd,
        n_ff,
    })
}

/// Stack `num_experts` rank-2 per-expert tensors into one contiguous
/// `[num_experts, k, n]` slab in GroupedMatMul layout (expert dim
/// outermost). Used by HF-style checkpoints that ship per-expert
/// tensors separately and need to be packed for the MoE block.
pub fn stack_expert_tensors(
    per_expert: &[(Vec<f32>, Vec<usize>)],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let num_experts = per_expert.len();
    if num_experts == 0 {
        return Err(anyhow!("stack_expert_tensors: empty input"));
    }
    let first_shape = &per_expert[0].1;
    if first_shape.len() != 2 {
        return Err(anyhow!(
            "stack_expert_tensors: first expert tensor must be rank-2, got {first_shape:?}"
        ));
    }
    let k = first_shape[0];
    let n = first_shape[1];
    let per = k * n;
    let mut out = Vec::with_capacity(num_experts * per);
    for (idx, (data, shape)) in per_expert.iter().enumerate() {
        if shape.as_slice() != [k, n] {
            return Err(anyhow!(
                "stack_expert_tensors: expert {idx} shape {shape:?} != first expert shape {first_shape:?}"
            ));
        }
        if data.len() != per {
            return Err(anyhow!(
                "stack_expert_tensors: expert {idx} data len {} != {per}",
                data.len()
            ));
        }
        out.extend_from_slice(data);
    }
    Ok((out, vec![num_experts, k, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loader::WeightLoader;
    use crate::weight_map::WeightMap;
    use std::collections::HashMap;

    /// In-memory `WeightLoader` for tests — just a HashMap of
    /// `(name, (data, shape))`.
    struct MapLoader {
        tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    }

    impl WeightLoader for MapLoader {
        fn len(&self) -> usize {
            self.tensors.len()
        }
        fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.tensors
                .remove(key)
                .ok_or_else(|| anyhow!("missing weight: {key}"))
        }
        fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.take(key)
        }
        fn remaining_keys(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    fn synth_data(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as u64 + seed) % 7) as f32 * 0.01)
            .collect()
    }

    #[test]
    fn load_layer_round_trip() {
        let num_experts = 4;
        let n_embd = 8;
        let n_ff = 16;
        let keys = MoeLayerKeys::llama_cpp(0);

        let mut tensors = HashMap::new();
        tensors.insert(
            keys.router.clone(),
            (
                synth_data(n_embd * num_experts, 1),
                vec![n_embd, num_experts],
            ),
        );
        tensors.insert(
            keys.gate.clone(),
            (
                synth_data(num_experts * n_embd * n_ff, 2),
                vec![num_experts, n_embd, n_ff],
            ),
        );
        tensors.insert(
            keys.up.clone(),
            (
                synth_data(num_experts * n_embd * n_ff, 3),
                vec![num_experts, n_embd, n_ff],
            ),
        );
        tensors.insert(
            keys.down.clone(),
            (
                synth_data(num_experts * n_ff * n_embd, 4),
                vec![num_experts, n_ff, n_embd],
            ),
        );

        let mut loader = MapLoader { tensors };
        let w = load_layer(&mut loader, &keys, num_experts, n_embd, n_ff).expect("load_layer");
        assert_eq!(w.num_experts, num_experts);
        assert_eq!(w.gate.len(), num_experts * n_embd * n_ff);
        assert_eq!(w.up.len(), num_experts * n_embd * n_ff);
        assert_eq!(w.down.len(), num_experts * n_ff * n_embd);
        assert_eq!(w.router.len(), n_embd * num_experts);
    }

    #[test]
    fn shape_mismatch_errors() {
        let mut tensors = HashMap::new();
        // Wrong shape: missing num_experts dim
        tensors.insert(
            "blk.0.ffn_gate_exps.weight".into(),
            (synth_data(16, 0), vec![8, 2]),
        );
        let mut loader = MapLoader { tensors };
        let err = load_expert_stack(&mut loader, "blk.0.ffn_gate_exps.weight", 4, 8, 2)
            .expect_err("should error on wrong shape");
        assert!(format!("{err:#}").contains("expected shape"));
    }

    #[test]
    fn stack_expert_tensors_basic() {
        let per: Vec<(Vec<f32>, Vec<usize>)> =
            (0..3).map(|i| (vec![i as f32; 6], vec![2, 3])).collect();
        let (stacked, shape) = stack_expert_tensors(&per).expect("stack");
        assert_eq!(shape, vec![3, 2, 3]);
        assert_eq!(stacked.len(), 18);
        assert_eq!(&stacked[..6], &[0.0; 6]);
        assert_eq!(&stacked[6..12], &[1.0; 6]);
        assert_eq!(&stacked[12..18], &[2.0; 6]);
    }

    #[test]
    fn keys_use_llama_cpp_convention_by_default() {
        let k = MoeLayerKeys::llama_cpp(5);
        assert_eq!(k.router, "blk.5.ffn_gate_inp.weight");
        assert_eq!(k.gate, "blk.5.ffn_gate_exps.weight");
        assert_eq!(k.up, "blk.5.ffn_up_exps.weight");
        assert_eq!(k.down, "blk.5.ffn_down_exps.weight");
    }

    // Quiets the unused-import warning when only one test references it.
    #[allow(dead_code)]
    fn _kept(_m: WeightMap) {}
}
