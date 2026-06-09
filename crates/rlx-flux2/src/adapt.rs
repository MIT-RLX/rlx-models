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

//! Checkpoint naming adapters — BFL single-file NVFP4 → diffusers/MAX keys.

use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

/// BFL → MAX/diffusers prefix mapping (see modular `flux2_modulev3/nvfp4_weight_adapter.py`).
pub const BFL_TO_MAX: &[(&str, &str)] = &[
    ("img_in.", "x_embedder."),
    ("txt_in.", "context_embedder."),
    (
        "time_in.in_layer.",
        "time_guidance_embed.timestep_embedder.linear_1.",
    ),
    (
        "time_in.out_layer.",
        "time_guidance_embed.timestep_embedder.linear_2.",
    ),
    (
        "guidance_in.in_layer.",
        "time_guidance_embed.guidance_embedder.linear_1.",
    ),
    (
        "guidance_in.out_layer.",
        "time_guidance_embed.guidance_embedder.linear_2.",
    ),
    ("final_layer.adaLN_modulation.1.", "norm_out.linear."),
    ("final_layer.linear.", "proj_out."),
    (".lin.", ".linear."),
    ("double_blocks.", "transformer_blocks."),
    ("single_blocks.", "single_transformer_blocks."),
    (".img_attn.qkv.", ".attn.qkv_proj."),
    (".txt_attn.qkv.", ".attn.add_qkv_proj."),
    (".img_attn.proj.", ".attn.to_out.0."),
    (".txt_attn.proj.", ".attn.to_add_out."),
    (".img_attn.norm.query_norm.scale", ".attn.norm_q.weight"),
    (".img_attn.norm.key_norm.scale", ".attn.norm_k.weight"),
    (
        ".txt_attn.norm.query_norm.scale",
        ".attn.norm_added_q.weight",
    ),
    (".txt_attn.norm.key_norm.scale", ".attn.norm_added_k.weight"),
    (".norm.query_norm.scale", ".attn.norm_q.weight"),
    (".norm.key_norm.scale", ".attn.norm_k.weight"),
    (".img_mlp.0.", ".ff.linear_in."),
    (".img_mlp.2.", ".ff.linear_out."),
    (".txt_mlp.0.", ".ff_context.linear_in."),
    (".txt_mlp.2.", ".ff_context.linear_out."),
    (".linear1.", ".attn.to_qkv_mlp_proj."),
    (".linear2.", ".attn.to_out."),
];

/// Normalize a checkpoint tensor name (HF prefixes + optional BFL → diffusers remap).
pub fn remap_checkpoint_key(key: &str, bfl: bool) -> String {
    let mut k = normalize_flux2_key(key);
    if bfl {
        for (before, after) in BFL_TO_MAX {
            k = k.replace(before, after);
        }
    }
    k
}

/// Strip common HuggingFace / pipeline prefixes from tensor names.
pub fn normalize_flux2_key(key: &str) -> String {
    let mut k = key.to_string();
    for prefix in [
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "pipe.transformer.",
    ] {
        if let Some(rest) = k.strip_prefix(prefix) {
            k = rest.to_string();
        }
    }
    // Flow-map / Diamond Maps dual-time embedder (diffusers `DualTimeEmbedder`).
    if k.contains("time_text_embed.second_embedder.") {
        k = k.replace(
            "time_text_embed.second_embedder.",
            "time_guidance_embed_target.",
        );
    }
    if k.contains("time_text_embed.original_embedder.") {
        k = k.replace("time_text_embed.original_embedder.", "time_guidance_embed.");
    }
    k
}

fn is_bfl_checkpoint(mut keys: impl Iterator<Item = impl AsRef<str>>) -> bool {
    keys.any(|k| k.as_ref().starts_with("double_blocks."))
}

/// Swap first/second halves of AdaLN linear weights (BFL vs diffusers chunk order).
fn swap_adaln_halves(data: &mut [f32]) {
    let half = data.len() / 2;
    let (a, b) = data.split_at_mut(half);
    a.swap_with_slice(b);
}

/// Rename BFL NVFP4 keys; swap `norm_out.linear` halves when needed.
pub fn adapt_bfl_weights(mut wm: WeightMap) -> WeightMap {
    let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    if !is_bfl_checkpoint(keys.iter()) {
        return wm;
    }
    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for key in keys {
        let (mut data, shape) = wm.take(&key).expect("key present");
        let mut max_name = key;
        for (before, after) in BFL_TO_MAX {
            max_name = max_name.replace(before, after);
        }
        if max_name == "norm_out.linear.weight" || max_name == "norm_out.linear.bias" {
            swap_adaln_halves(&mut data);
        }
        out.insert(max_name, (data, shape));
    }
    WeightMap::from_tensors(out)
}

const STACKED_QKV: &[(&str, &str, &str, &str)] = &[
    (
        ".attn.qkv_proj.",
        ".attn.to_q.",
        ".attn.to_k.",
        ".attn.to_v.",
    ),
    (
        ".attn.add_qkv_proj.",
        ".attn.add_q_proj.",
        ".attn.add_k_proj.",
        ".attn.add_v_proj.",
    ),
];

/// Split fused QKV projections into separate `to_q` / `to_k` / `to_v` tensors.
pub fn split_stacked_qkv(mut wm: WeightMap) -> WeightMap {
    let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    let stacked = keys
        .iter()
        .any(|k| k.contains(".attn.qkv_proj.") || k.contains(".attn.add_qkv_proj."));
    if !stacked {
        return wm;
    }
    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for key in keys {
        let (data, shape) = wm.take(&key).expect("key present");
        if let Some((stacked, q, k, v)) = STACKED_QKV
            .iter()
            .find(|(stacked, _, _, _)| key.contains(stacked))
        {
            if key.ends_with(".weight") && shape.len() == 2 {
                let chunk = shape[0] / 3;
                for (infix, i) in [(q, 0usize), (k, 1), (v, 2)] {
                    let name = key.replace(stacked, infix);
                    let row0 = i * chunk;
                    let mut slice = vec![0.0f32; chunk * shape[1]];
                    for r in 0..chunk {
                        let src = (row0 + r) * shape[1];
                        let dst = r * shape[1];
                        slice[dst..dst + shape[1]].copy_from_slice(&data[src..src + shape[1]]);
                    }
                    out.insert(name, (slice, vec![chunk, shape[1]]));
                }
            } else if key.ends_with(".bias") && shape.len() == 1 {
                let chunk = shape[0] / 3;
                for (infix, i) in [(q, 0usize), (k, 1), (v, 2)] {
                    let name = key.replace(stacked, infix);
                    out.insert(
                        name,
                        (data[i * chunk..(i + 1) * chunk].to_vec(), vec![chunk]),
                    );
                }
            } else {
                out.insert(key, (data, shape));
            }
        } else {
            out.insert(key, (data, shape));
        }
    }
    WeightMap::from_tensors(out)
}

/// Full load-time adaptation pipeline.
pub fn prepare_weight_map(mut wm: WeightMap) -> WeightMap {
    let mut renamed = HashMap::new();
    for key in wm.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
        let nk = normalize_flux2_key(&key);
        renamed.insert(nk, wm.take(&key).expect("present"));
    }
    let wm = adapt_bfl_weights(WeightMap::from_tensors(renamed));
    split_stacked_qkv(wm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_time_embedder_key_remap() {
        let k = normalize_flux2_key(
            "transformer.time_text_embed.second_embedder.timestep_embedder.linear_1.weight",
        );
        assert!(k.starts_with("time_guidance_embed_target."));
        let k2 = normalize_flux2_key(
            "transformer.time_text_embed.original_embedder.timestep_embedder.linear_1.weight",
        );
        assert!(k2.starts_with("time_guidance_embed."));
    }
}
