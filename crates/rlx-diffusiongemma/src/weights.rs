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

//! Checkpoint adaptation.
//!
//! DiffusionGemma already ships its experts stacked, but in PyTorch's
//! `nn.Linear` orientation:
//!
//! ```text
//!   experts.gate_up_proj  [E, 2·moe_inter, hidden]
//!   experts.down_proj     [E, hidden, moe_inter]
//! ```
//!
//! [`rlx_ir::op::Op::GroupedMatMul`] wants `[E, K, N]`:
//!
//! ```text
//!   experts.gate_up_proj  [E, hidden, 2·moe_inter]
//!   experts.down_proj     [E, moe_inter, hidden]
//! ```
//!
//! Transposing here rather than in-graph is deliberate: a constant-folded
//! in-graph transpose keeps *both* copies of the bank resident. For
//! diffusiongemma-26B the routed experts are
//! `30 · 128 · 3 · 704 · 2816 · 4 B ≈ 91 GB` as f32, so doubling them is not an
//! option — and it is why a memory-constrained deployment wants the bf16 or
//! paged expert path rather than a plain f32 `WeightMap`.
//!
//! The encoder and decoder stacks share every tensor except `layer_scalar`, so
//! nothing here duplicates weights per stack; the two graphs simply read the
//! same keys.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;

use crate::config::DiffusionGemmaConfig;
use crate::flow::{ENCODER_SCALAR_PREFIX, LAYER_PREFIX};

/// Sentinel entry recording that [`prepare_checkpoint`] already ran on a map.
///
/// Shape alone cannot decide this: when a bank happens to be square
/// (`2·moe_inter == hidden`) the pre- and post-transpose shapes are identical,
/// so a second pass would transpose it back. The real model is never square,
/// but small test and ablation configs are.
pub const PREPARED_MARKER: &str = "diffusiongemma.experts_pretransposed";

/// `(gate_up, down)` bank names for layer `i`.
pub fn expert_bank_keys(layer: usize) -> (String, String) {
    (
        format!("{LAYER_PREFIX}.{layer}.experts.gate_up_proj"),
        format!("{LAYER_PREFIX}.{layer}.experts.down_proj"),
    )
}

/// Rewrite a stock HF checkpoint in `wm` into the layout the graphs read.
///
/// Idempotent — banks already in `[E, K, N]` layout are left alone, so prefill
/// and denoiser builds can share one map.
pub fn prepare_checkpoint(cfg: &DiffusionGemmaConfig, wm: &mut WeightMap) -> Result<()> {
    if wm.has(PREPARED_MARKER) {
        return Ok(());
    }
    let t = &cfg.text_config;
    for layer in 0..t.num_hidden_layers {
        prepare_layer_experts(cfg, wm, layer)?;
    }
    wm.insert(PREPARED_MARKER.to_string(), vec![1.0], vec![1]);
    Ok(())
}

/// Pre-transpose one layer's expert banks.
///
/// [`prepare_checkpoint`] walks every layer; this is the single-layer form, for
/// partial checkpoints that hold one subsystem (see the real-weight tests,
/// which fetch a single layer rather than all 51 GB).
pub fn prepare_layer_experts(
    cfg: &DiffusionGemmaConfig,
    wm: &mut WeightMap,
    layer: usize,
) -> Result<()> {
    let t = &cfg.text_config;
    let (h, i, e) = (t.hidden_size, t.moe_intermediate_size, t.num_experts);
    let (gu_key, dn_key) = expert_bank_keys(layer);
    transpose_bank(wm, &gu_key, e, 2 * i, h)?;
    transpose_bank(wm, &dn_key, e, h, i)?;
    Ok(())
}

/// Transpose a stacked bank from `[E, rows, cols]` to `[E, cols, rows]`.
fn transpose_bank(wm: &mut WeightMap, key: &str, e: usize, rows: usize, cols: usize) -> Result<()> {
    let shape = wm
        .get(key)
        .map(|(_, s)| s.to_vec())
        .with_context(|| format!("DiffusionGemma checkpoint is missing {key}"))?;
    // Accept a bank some other loader already stored in `[E, K, N]` layout —
    // but only when the two orientations are distinguishable. For a square bank
    // this test would match the *source* layout too and skip a needed
    // transpose; repeat calls are handled by `PREPARED_MARKER` instead.
    if rows != cols && shape == [e, cols, rows] {
        return Ok(());
    }
    anyhow::ensure!(
        shape == [e, rows, cols],
        "{key}: expected [{e}, {rows}, {cols}] (or the transposed [{e}, {cols}, {rows}]), \
         checkpoint has {shape:?}"
    );
    let (data, _) = wm.take(key)?;
    let mut out = vec![0f32; data.len()];
    for ei in 0..e {
        let src = &data[ei * rows * cols..(ei + 1) * rows * cols];
        let dst = &mut out[ei * rows * cols..(ei + 1) * rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
    }
    wm.insert(key.to_string(), out, vec![e, cols, rows]);
    Ok(())
}

/// Every tensor the two graphs load, in checkpoint naming. Useful to validate a
/// checkpoint before paying for a build.
pub fn required_keys(cfg: &DiffusionGemmaConfig) -> Vec<String> {
    let t = &cfg.text_config;
    let mut keys = vec![
        "model.decoder.embed_tokens.weight".to_string(),
        "model.decoder.norm.weight".to_string(),
        "model.decoder.self_conditioning.pre_norm.weight".to_string(),
        "model.decoder.self_conditioning.gate_proj.weight".to_string(),
        "model.decoder.self_conditioning.up_proj.weight".to_string(),
        "model.decoder.self_conditioning.down_proj.weight".to_string(),
    ];
    for l in 0..t.num_hidden_layers {
        let p = format!("{LAYER_PREFIX}.{l}");
        keys.extend([
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
            format!("{p}.pre_feedforward_layernorm.weight"),
            format!("{p}.pre_feedforward_layernorm_2.weight"),
            format!("{p}.post_feedforward_layernorm.weight"),
            format!("{p}.post_feedforward_layernorm_1.weight"),
            format!("{p}.post_feedforward_layernorm_2.weight"),
            format!("{p}.layer_scalar"),
            format!("{ENCODER_SCALAR_PREFIX}.{l}.layer_scalar"),
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.mlp.gate_proj.weight"),
            format!("{p}.mlp.up_proj.weight"),
            format!("{p}.mlp.down_proj.weight"),
            format!("{p}.router.proj.weight"),
            format!("{p}.router.scale"),
            format!("{p}.router.per_expert_scale"),
            format!("{p}.experts.gate_up_proj"),
            format!("{p}.experts.down_proj"),
        ]);
        // Only sliding layers ship a `v_proj`; full-attention layers alias V→K.
        if !t.layer_k_eq_v(l) {
            keys.push(format!("{p}.self_attn.v_proj.weight"));
        }
    }
    keys
}

/// Names present in the checkpoint but not loaded by the text graphs — the
/// vision tower and its projector, which live behind the `vision` entry points.
pub fn is_vision_key(key: &str) -> bool {
    key.starts_with("model.encoder.vision_tower.") || key.starts_with("model.encoder.embed_vision.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// `hidden = 6`, `2·moe_inter = 4` — deliberately non-square, so the
    /// transpose is observable.
    fn tiny_cfg() -> DiffusionGemmaConfig {
        DiffusionGemmaConfig::from_json(
            r#"{"model_type":"diffusion_gemma","canvas_length":4,
                "text_config":{"vocab_size":16,"hidden_size":6,"intermediate_size":6,
                  "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                  "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":8,
                  "layer_types":["sliding_attention","full_attention"],
                  "sliding_window":8,"num_experts":3,"top_k_experts":2,
                  "moe_intermediate_size":2}}"#,
        )
        .unwrap()
    }

    /// Square banks (`2·moe_inter == hidden`), where shape alone cannot tell
    /// pre- from post-transpose.
    fn square_cfg() -> DiffusionGemmaConfig {
        DiffusionGemmaConfig::from_json(
            r#"{"model_type":"diffusion_gemma","canvas_length":4,
                "text_config":{"vocab_size":16,"hidden_size":4,"intermediate_size":6,
                  "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                  "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":8,
                  "layer_types":["full_attention"],
                  "sliding_window":8,"num_experts":3,"top_k_experts":2,
                  "moe_intermediate_size":2}}"#,
        )
        .unwrap()
    }

    fn banks(e: usize, i: usize, h: usize, layers: usize) -> WeightMap {
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for l in 0..layers {
            let (gu, dn) = expert_bank_keys(l);
            t.insert(
                gu,
                (
                    (0..e * 2 * i * h).map(|x| x as f32).collect(),
                    vec![e, 2 * i, h],
                ),
            );
            t.insert(
                dn,
                ((0..e * h * i).map(|x| x as f32).collect(), vec![e, h, i]),
            );
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn transposes_expert_banks_into_grouped_matmul_layout() {
        let c = tiny_cfg();
        let (h, i, e) = (6usize, 2usize, 3usize);
        let mut wm = banks(e, i, h, 2);
        prepare_checkpoint(&c, &mut wm).unwrap();

        // gate_up: [E, 2i=4, h=6] → [E, h=6, 2i=4]; src[r,c] = r*6 + c.
        let (gu, shape) = wm.get(&expert_bank_keys(0).0).unwrap();
        assert_eq!(shape, &[e, h, 2 * i]);
        assert_eq!(gu[0], 0.0, "dst[0,0] = src[0,0]");
        assert_eq!(gu[1], 6.0, "dst[0,1] = src[1,0]");
        assert_eq!(gu[2 * i], 1.0, "dst[1,0] = src[0,1]");
        assert_eq!(gu[h * 2 * i], (2 * i * h) as f32, "expert 1 slab offset");

        // down: [E, h=6, i=2] → [E, i=2, h=6]; src[r,c] = r*2 + c.
        let (dn, dshape) = wm.get(&expert_bank_keys(0).1).unwrap();
        assert_eq!(dshape, &[e, i, h]);
        assert_eq!(dn[0], 0.0);
        assert_eq!(dn[1], 2.0, "dst[0,1] = src[1,0]");
    }

    #[test]
    fn is_idempotent() {
        let c = tiny_cfg();
        let (h, i, e) = (6usize, 2usize, 3usize);
        let mut wm = banks(e, i, h, 2);
        prepare_checkpoint(&c, &mut wm).unwrap();
        let once = wm.get(&expert_bank_keys(0).0).unwrap().0.to_vec();
        prepare_checkpoint(&c, &mut wm).expect("second call must be a no-op");
        assert_eq!(wm.get(&expert_bank_keys(0).0).unwrap().1, &[e, h, 2 * i]);
        assert_eq!(wm.get(&expert_bank_keys(0).0).unwrap().0, once);
    }

    /// Regression: with square banks a shape-only idempotency check would
    /// transpose right back on the second call.
    #[test]
    fn is_idempotent_even_when_the_banks_are_square() {
        let c = square_cfg();
        let (h, i, e) = (4usize, 2usize, 3usize);
        assert_eq!(2 * i, h, "this test is only meaningful when square");
        let mut wm = banks(e, i, h, 1);
        prepare_checkpoint(&c, &mut wm).unwrap();
        let once = wm.get(&expert_bank_keys(0).0).unwrap().0.to_vec();
        prepare_checkpoint(&c, &mut wm).unwrap();
        assert_eq!(
            wm.get(&expert_bank_keys(0).0).unwrap().0,
            once,
            "a second prepare must not transpose the bank back"
        );
        assert_ne!(once[1], 1.0, "the first prepare did transpose");
        assert_eq!(once[1], 4.0);
    }

    #[test]
    fn names_the_missing_tensor() {
        let c = tiny_cfg();
        let mut wm = WeightMap::from_tensors(HashMap::new());
        let err = prepare_checkpoint(&c, &mut wm).unwrap_err();
        assert!(
            format!("{err:#}").contains("experts.gate_up_proj"),
            "unhelpful error: {err:#}"
        );
    }

    #[test]
    fn required_keys_track_the_v_proj_asymmetry() {
        let c = tiny_cfg();
        let keys = required_keys(&c);
        // Layer 0 is sliding → has v_proj; layer 1 is full → aliases V to K.
        assert!(keys.contains(&format!("{LAYER_PREFIX}.0.self_attn.v_proj.weight")));
        assert!(!keys.contains(&format!("{LAYER_PREFIX}.1.self_attn.v_proj.weight")));
        // The encoder's untied per-layer scalar is required too.
        assert!(keys.contains(&format!("{ENCODER_SCALAR_PREFIX}.0.layer_scalar")));
    }

    #[test]
    fn vision_keys_are_recognised() {
        assert!(is_vision_key(
            "model.encoder.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight"
        ));
        assert!(is_vision_key(
            "model.encoder.embed_vision.embedding_projection.weight"
        ));
        assert!(!is_vision_key(
            "model.decoder.layers.0.mlp.gate_proj.weight"
        ));
        // The encoder's per-layer scalar is a *text* tensor despite the prefix.
        assert!(!is_vision_key(
            "model.encoder.language_model.layers.0.layer_scalar"
        ));
    }
}
