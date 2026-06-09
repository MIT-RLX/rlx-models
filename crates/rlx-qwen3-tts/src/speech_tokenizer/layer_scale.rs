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

//! Bake HF `layer_scale` into Qwen3 `o_proj` / `down_proj` rows for compiled pre_transformer.

use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

const PT_PREFIX: &str = "decoder.pre_transformer";

fn scale_out_features(data: &mut [f32], shape: &[usize], scale: &[f32]) -> Result<()> {
    ensure!(shape.len() == 2, "expected 2d weight");
    let (out, inp) = (shape[0], shape[1]);
    ensure!(
        scale.len() == out,
        "scale len {} != out {}",
        scale.len(),
        out
    );
    for o in 0..out {
        let s = scale[o];
        let base = o * inp;
        for i in 0..inp {
            data[base + i] *= s;
        }
    }
    Ok(())
}

/// Fold per-layer `self_attn_layer_scale` / `mlp_layer_scale` into remapped Qwen3 weights.
pub fn bake_layer_scales_into_qwen3_weights(
    weights: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    hf_map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    num_layers: usize,
) -> Result<()> {
    for i in 0..num_layers {
        let lp = format!("{PT_PREFIX}.layers.{i}");
        let attn_scale = hf_map
            .get(&format!("{lp}.self_attn_layer_scale.scale"))
            .with_context(|| format!("{lp}.self_attn_layer_scale.scale"))?;
        let ffn_scale = hf_map
            .get(&format!("{lp}.mlp_layer_scale.scale"))
            .with_context(|| format!("{lp}.mlp_layer_scale.scale"))?;
        let wo_key = format!("model.layers.{i}.self_attn.o_proj.weight");
        let down_key = format!("model.layers.{i}.mlp.down_proj.weight");
        if let Some((data, shape)) = weights.get_mut(&wo_key) {
            scale_out_features(data, shape, &attn_scale.0)?;
        }
        if let Some((data, shape)) = weights.get_mut(&down_key) {
            scale_out_features(data, shape, &ffn_scale.0)?;
        }
    }
    Ok(())
}
