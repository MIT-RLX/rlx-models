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

//! Weight loading + OpenPI→RLX key remapping for VLASH checkpoints.
//!
//! The published `lerobot/pi0_base` / `lerobot/pi05_base` checkpoints ship a
//! single bf16 `model.safetensors` in **OpenPI** naming, e.g.
//! `paligemma_with_expert.paligemma.model.language_model.layers.3.self_attn.q_proj.weight`
//! and top-level `action_in_proj.weight`, `action_out_proj.weight`, … The
//! reference `from_pretrained` (`modeling_pi0.py`) applies a prefix rewrite
//! that folds these into the policy `state_dict`.
//!
//! Rather than reproduce the exact PyTorch module nesting (which varies with
//! the `transformers` version that saved the checkpoint), [`canonical_key`]
//! rewrites each checkpoint key to a **stable RLX name** anchored on
//! meaningful substrings (`vision_model.encoder.layers.`, `language_model` +
//! `.layers.`, `multi_modal_projector.linear.`, …). The graph builders in
//! `vision.rs` / `joint_layer.rs` / `suffix.rs` load params by these canonical
//! names, so the crate is robust to nesting differences and to the two
//! (π₀ vs π₀.₅) suffix-embedder layouts.
//!
//! Canonical namespace:
//! ```text
//!   vision.embeddings.patch_embedding.{weight,bias}
//!   vision.embeddings.position_embedding.weight
//!   vision.encoder.layers.{i}.{layer_norm1,layer_norm2}.{weight,bias}
//!   vision.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
//!   vision.encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias}
//!   vision.post_layernorm.{weight,bias}
//!   vision.projector.{weight,bias}                 (multi_modal_projector.linear)
//!   vlm.embed_tokens.weight                         (Gemma-2B token embeddings)
//!   vlm.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
//!   vlm.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   vlm.layers.{i}.mlp.{gate,up,down}_proj.weight
//!   vlm.norm.weight
//!   expert.layers.{i}.{…same as vlm…}
//!   expert.norm.weight
//!   suffix.action_in_proj.{weight,bias}
//!   suffix.state_proj.{weight,bias}
//!   suffix.action_time_mlp_in.{weight,bias}         (π₀)
//!   suffix.action_time_mlp_out.{weight,bias}        (π₀)
//!   suffix.time_mlp_in.{weight,bias}                (π₀.₅)
//!   suffix.time_mlp_out.{weight,bias}               (π₀.₅)
//!   suffix.state_mlp_in.{weight,bias}               (π₀.₅, state_cond)
//!   suffix.state_mlp_out.{weight,bias}              (π₀.₅, state_cond)
//!   action_out_proj.{weight,bias}
//!   norm.state.{mean,std}    norm.action.{mean,std} (dataset stats buffers)
//! ```

use anyhow::Result;
use rlx_core::weight_map::WeightMap;

/// Substring after the first occurrence of `anchor` (exclusive), if present.
fn after<'a>(key: &'a str, anchor: &str) -> Option<&'a str> {
    key.find(anchor).map(|i| &key[i + anchor.len()..])
}

/// Rewrite one OpenPI/HF checkpoint key to its stable RLX canonical name.
///
/// Returns `None` for keys we intentionally drop (e.g. a tied `lm_head` the
/// action path never uses). Unrecognized keys are returned unchanged so the
/// loader's "unexpected key" hygiene can surface genuinely novel tensors.
pub fn canonical_key(raw: &str) -> Option<String> {
    // ---- normalization-stat buffers (dots already replaced by underscores) ----
    let low = raw;
    if low.contains("observation_state") || low.contains("observation.state") {
        if low.ends_with(".mean") {
            return Some("norm.state.mean".to_string());
        }
        if low.ends_with(".std") {
            return Some("norm.state.std".to_string());
        }
    }
    if (low.contains("buffer_action")
        || low.ends_with("action.mean")
        || low.ends_with("action.std"))
        && low.contains("action")
    {
        // Distinguish state vs action already handled above; here handle action.
        if low.ends_with(".mean") && !low.contains("state") {
            return Some("norm.action.mean".to_string());
        }
        if low.ends_with(".std") && !low.contains("state") {
            return Some("norm.action.std".to_string());
        }
    }

    // ---- suffix embedder + action head (top-level OpenPI keys) ----
    for (src, dst) in [
        ("action_time_mlp_in.", "suffix.action_time_mlp_in."),
        ("action_time_mlp_out.", "suffix.action_time_mlp_out."),
        ("action_in_proj.", "suffix.action_in_proj."),
        ("action_out_proj.", "action_out_proj."),
        ("state_mlp_in.", "suffix.state_mlp_in."),
        ("state_mlp_out.", "suffix.state_mlp_out."),
        ("state_proj.", "suffix.state_proj."),
        ("time_mlp_in.", "suffix.time_mlp_in."),
        ("time_mlp_out.", "suffix.time_mlp_out."),
    ] {
        // Only fire on top-level occurrences (avoid matching inside longer paths).
        if let Some(rest) = raw.strip_prefix(src) {
            return Some(format!("{dst}{rest}"));
        }
    }

    // ---- action expert (Gemma-300M) ----
    if raw.contains("gemma_expert") {
        if let Some(rest) = after(raw, ".layers.") {
            // Layer keys pass through verbatim: standard `input_layernorm.weight`
            // (π₀) or adaRMS `input_layernorm.dense.{weight,bias}` (π₀.₅).
            return Some(format!("expert.layers.{rest}"));
        }
        // Final norm: `model.norm.weight` (π₀) or `model.norm.dense.{weight,bias}` (π₀.₅).
        if let Some(rest) = after(raw, "model.norm.") {
            return Some(format!("expert.norm.{rest}"));
        }
        if raw.contains("embed_tokens") {
            return Some("expert.embed_tokens.weight".to_string());
        }
        return None; // drop any stray expert tensors (e.g. tied lm_head)
    }

    // ---- PaliGemma VLM (vision tower + Gemma-2B text) ----
    if raw.contains("paligemma") {
        // Vision tower.
        if raw.contains("vision_model") {
            if let Some(rest) = after(raw, "encoder.layers.") {
                return Some(format!("vision.encoder.layers.{rest}"));
            }
            if raw.contains("embeddings.patch_embedding.weight") {
                return Some("vision.embeddings.patch_embedding.weight".to_string());
            }
            if raw.contains("embeddings.patch_embedding.bias") {
                return Some("vision.embeddings.patch_embedding.bias".to_string());
            }
            if raw.contains("embeddings.position_embedding.weight") {
                return Some("vision.embeddings.position_embedding.weight".to_string());
            }
            if raw.contains("post_layernorm.weight") {
                return Some("vision.post_layernorm.weight".to_string());
            }
            if raw.contains("post_layernorm.bias") {
                return Some("vision.post_layernorm.bias".to_string());
            }
        }
        if raw.contains("multi_modal_projector.linear.weight") {
            return Some("vision.projector.weight".to_string());
        }
        if raw.contains("multi_modal_projector.linear.bias") {
            return Some("vision.projector.bias".to_string());
        }
        // Text embeddings are tied to the LM head — the checkpoint ships only
        // `paligemma.lm_head.weight` [vocab, hidden]; use it as embed_tokens.
        if raw.contains("lm_head") {
            return Some("vlm.embed_tokens.weight".to_string());
        }
        // Gemma-2B text backbone.
        if raw.contains("language_model") {
            if let Some(rest) = after(raw, ".layers.") {
                return Some(format!("vlm.layers.{rest}"));
            }
            if raw.contains("embed_tokens") {
                return Some("vlm.embed_tokens.weight".to_string());
            }
            if raw.ends_with(".norm.weight") {
                return Some("vlm.norm.weight".to_string());
            }
        }
        return None; // drop VLM lm_head / rotary buffers / anything else unused
    }

    // Unknown: pass through so callers can detect it.
    Some(raw.to_string())
}

/// Load a checkpoint `model.safetensors` (file or directory) and remap every
/// key to its RLX canonical name (see [`canonical_key`]). Dropped keys
/// (`canonical_key` → `None`) are removed from the map.
pub fn load_remapped(path: &str) -> Result<WeightMap> {
    let p = std::path::Path::new(path);
    let mut wm = if p.is_dir() {
        WeightMap::from_safetensors_dir(p)?
    } else {
        WeightMap::from_file(path)?
    };
    remap(&mut wm);
    Ok(wm)
}

/// Apply [`canonical_key`] to every tensor in `wm` in place, dropping any key
/// that maps to `None`.
pub fn remap(wm: &mut WeightMap) {
    // Collect drops first (remap_keys can't remove).
    let drops: Vec<String> = wm
        .keys()
        .filter(|k| canonical_key(k).is_none())
        .map(|k| k.to_string())
        .collect();
    for d in &drops {
        // take() removes; ignore the value.
        let _ = wm.take(d);
    }
    wm.remap_keys(|k| canonical_key(&k).unwrap_or(k));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_representative_openpi_keys() {
        let cases = [
            (
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.weight",
                Some("vision.encoder.layers.5.self_attn.q_proj.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
                Some("vision.embeddings.patch_embedding.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight",
                Some("vision.embeddings.position_embedding.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias",
                Some("vision.post_layernorm.bias"),
            ),
            (
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
                Some("vision.projector.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.language_model.layers.3.self_attn.o_proj.weight",
                Some("vlm.layers.3.self_attn.o_proj.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight",
                Some("vlm.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.language_model.norm.weight",
                Some("vlm.norm.weight"),
            ),
            (
                "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
                Some("vlm.embed_tokens.weight"),
            ),
            (
                "paligemma_with_expert.gemma_expert.model.layers.7.input_layernorm.weight",
                Some("expert.layers.7.input_layernorm.weight"),
            ),
            (
                "paligemma_with_expert.gemma_expert.model.norm.weight",
                Some("expert.norm.weight"),
            ),
            (
                "action_in_proj.weight",
                Some("suffix.action_in_proj.weight"),
            ),
            ("action_in_proj.bias", Some("suffix.action_in_proj.bias")),
            ("action_out_proj.weight", Some("action_out_proj.weight")),
            (
                "action_time_mlp_in.weight",
                Some("suffix.action_time_mlp_in.weight"),
            ),
            (
                "action_time_mlp_out.bias",
                Some("suffix.action_time_mlp_out.bias"),
            ),
            ("state_proj.weight", Some("suffix.state_proj.weight")),
            ("time_mlp_in.weight", Some("suffix.time_mlp_in.weight")),
            ("state_mlp_out.weight", Some("suffix.state_mlp_out.weight")),
        ];
        for (raw, want) in cases {
            assert_eq!(
                canonical_key(raw).as_deref(),
                want,
                "canonical_key({raw:?}) mismatch"
            );
        }
    }

    #[test]
    fn distinguishes_action_in_vs_action_time_mlp() {
        // action_in_proj must NOT be swallowed by an action_time_mlp rule and
        // vice-versa (both start with "action_").
        assert_eq!(
            canonical_key("action_in_proj.weight").as_deref(),
            Some("suffix.action_in_proj.weight")
        );
        assert_eq!(
            canonical_key("action_time_mlp_in.weight").as_deref(),
            Some("suffix.action_time_mlp_in.weight")
        );
    }

    #[test]
    fn norm_stats_buffers() {
        assert_eq!(
            canonical_key("normalize_inputs.buffer_observation_state.mean").as_deref(),
            Some("norm.state.mean")
        );
        assert_eq!(
            canonical_key("normalize_targets.buffer_action.std").as_deref(),
            Some("norm.action.std")
        );
        assert_eq!(
            canonical_key("unnormalize_outputs.buffer_action.mean").as_deref(),
            Some("norm.action.mean")
        );
    }
}
