// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Checkpoint → canonical matmul-ready parameters.
//!
//! Accepts two on-disk formats and normalizes both to the timm-canonical
//! block-parameter names the graph in [`super::forward`] declares:
//!   - **timm** (`vit_base_patch16_224.dino`, UNI2-h): keys already canonical.
//!   - **HF `ViTModel`** (`facebook/dino-vitb16`): `vit.*` names with separate
//!     `query`/`key`/`value` matrices — concatenated into a single `qkv`,
//!     layers/embeddings renamed.
//!
//! All 2-D weights are transposed to the row-major `[in, out]` layout the
//! `x @ W` matmuls expect (via [`WeightMap::take_transposed`]); the packed
//! SwiGLU `fc1` is split into value/gate halves on the host (mirrors
//! `rlx_uni2`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};
use rlx_core::weight_map::WeightMap;

use super::config::{FfnKind, VitConfig};
use super::preprocess::{PreprocessWeights, extract_preprocess_weights};

/// A checkpoint prepared for [`super::forward::build_vit_graph`].
pub struct LoadedVit {
    /// Canonical matmul-ready block params (name → row-major f32), keyed
    /// exactly as the graph's `Op::Param` nodes.
    pub params: HashMap<String, Vec<f32>>,
    /// Host-side patch/token assembly weights.
    pub preprocess: PreprocessWeights,
}

/// Load + prepare a checkpoint from a safetensors path.
pub fn load_vit(path: &Path, cfg: &VitConfig) -> Result<LoadedVit> {
    let wm = rlx_core::load_weight_map(path, &[])?;
    prepare_from_weightmap(wm, cfg)
}

/// Prepare an in-memory checkpoint (also the test entry point).
pub fn prepare_from_weightmap(wm: WeightMap, cfg: &VitConfig) -> Result<LoadedVit> {
    let mut wm = canonicalize(wm, cfg)?;
    let preprocess = extract_preprocess_weights(&mut wm, cfg)?;
    let params = prepare_block_params(&mut wm, cfg)?;
    Ok(LoadedVit { params, preprocess })
}

/// Rewrite HF `ViTModel` keys to timm-canonical names (concatenating q/k/v);
/// timm checkpoints pass through unchanged.
fn canonicalize(mut wm: WeightMap, cfg: &VitConfig) -> Result<WeightMap> {
    // Already timm-canonical?
    if wm.has("blocks.0.attn.qkv.weight") || wm.has("patch_embed.proj.weight") {
        return Ok(wm);
    }
    let hf_probe = "vit.encoder.layer.0.attention.attention.query.weight";
    if !wm.has(hf_probe) && !wm.has("vit.embeddings.cls_token") {
        bail!(
            "unrecognized ViT checkpoint format: no timm `patch_embed.proj.weight` \
             nor HF `{hf_probe}` / `vit.embeddings.cls_token` key present"
        );
    }

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    // Embeddings.
    t.insert("cls_token".into(), wm.take("vit.embeddings.cls_token")?);
    t.insert(
        "pos_embed".into(),
        wm.take("vit.embeddings.position_embeddings")?,
    );
    t.insert(
        "patch_embed.proj.weight".into(),
        wm.take("vit.embeddings.patch_embeddings.projection.weight")?,
    );
    t.insert(
        "patch_embed.proj.bias".into(),
        wm.take("vit.embeddings.patch_embeddings.projection.bias")?,
    );
    // Blocks.
    for i in 0..cfg.num_hidden_layers {
        let src = format!("vit.encoder.layer.{i}");
        let dst = format!("blocks.{i}");
        // qkv = concat(query, key, value) along the output (row) axis.
        let (qw, qs) = wm.take(&format!("{src}.attention.attention.query.weight"))?;
        let (kw, _) = wm.take(&format!("{src}.attention.attention.key.weight"))?;
        let (vw, _) = wm.take(&format!("{src}.attention.attention.value.weight"))?;
        let h = qs[1];
        let qkv_w: Vec<f32> = qw.into_iter().chain(kw).chain(vw).collect();
        t.insert(format!("{dst}.attn.qkv.weight"), (qkv_w, vec![3 * h, h]));
        let (qb, _) = wm.take(&format!("{src}.attention.attention.query.bias"))?;
        let (kb, _) = wm.take(&format!("{src}.attention.attention.key.bias"))?;
        let (vb, _) = wm.take(&format!("{src}.attention.attention.value.bias"))?;
        let qkv_b: Vec<f32> = qb.into_iter().chain(kb).chain(vb).collect();
        t.insert(format!("{dst}.attn.qkv.bias"), (qkv_b, vec![3 * h]));

        t.insert(
            format!("{dst}.attn.proj.weight"),
            wm.take(&format!("{src}.attention.output.dense.weight"))?,
        );
        t.insert(
            format!("{dst}.attn.proj.bias"),
            wm.take(&format!("{src}.attention.output.dense.bias"))?,
        );
        t.insert(
            format!("{dst}.mlp.fc1.weight"),
            wm.take(&format!("{src}.intermediate.dense.weight"))?,
        );
        t.insert(
            format!("{dst}.mlp.fc1.bias"),
            wm.take(&format!("{src}.intermediate.dense.bias"))?,
        );
        t.insert(
            format!("{dst}.mlp.fc2.weight"),
            wm.take(&format!("{src}.output.dense.weight"))?,
        );
        t.insert(
            format!("{dst}.mlp.fc2.bias"),
            wm.take(&format!("{src}.output.dense.bias"))?,
        );
        t.insert(
            format!("{dst}.norm1.weight"),
            wm.take(&format!("{src}.layernorm_before.weight"))?,
        );
        t.insert(
            format!("{dst}.norm1.bias"),
            wm.take(&format!("{src}.layernorm_before.bias"))?,
        );
        t.insert(
            format!("{dst}.norm2.weight"),
            wm.take(&format!("{src}.layernorm_after.weight"))?,
        );
        t.insert(
            format!("{dst}.norm2.bias"),
            wm.take(&format!("{src}.layernorm_after.bias"))?,
        );
    }
    t.insert("norm.weight".into(), wm.take("vit.layernorm.weight")?);
    t.insert("norm.bias".into(), wm.take("vit.layernorm.bias")?);
    Ok(WeightMap::from_tensors(t))
}

fn take_1d(wm: &mut WeightMap, key: &str) -> Result<Vec<f32>> {
    Ok(wm.take(key)?.0)
}

/// Transpose + split all block weights into the canonical param map.
fn prepare_block_params(wm: &mut WeightMap, cfg: &VitConfig) -> Result<HashMap<String, Vec<f32>>> {
    let h = cfg.hidden_size;
    let inner = cfg.ffn_inner();
    let mut p: HashMap<String, Vec<f32>> = HashMap::new();

    for li in 0..cfg.num_hidden_layers {
        let lp = format!("blocks.{li}");
        p.insert(
            format!("{lp}.norm1.weight"),
            take_1d(wm, &format!("{lp}.norm1.weight"))?,
        );
        p.insert(
            format!("{lp}.norm1.bias"),
            take_1d(wm, &format!("{lp}.norm1.bias"))?,
        );
        p.insert(
            format!("{lp}.norm2.weight"),
            take_1d(wm, &format!("{lp}.norm2.weight"))?,
        );
        p.insert(
            format!("{lp}.norm2.bias"),
            take_1d(wm, &format!("{lp}.norm2.bias"))?,
        );

        p.insert(
            format!("{lp}.attn.qkv.weight"),
            wm.take_transposed(&format!("{lp}.attn.qkv.weight"))?.0,
        );
        p.insert(
            format!("{lp}.attn.qkv.bias"),
            take_1d(wm, &format!("{lp}.attn.qkv.bias"))?,
        );
        p.insert(
            format!("{lp}.attn.proj.weight"),
            wm.take_transposed(&format!("{lp}.attn.proj.weight"))?.0,
        );
        p.insert(
            format!("{lp}.attn.proj.bias"),
            take_1d(wm, &format!("{lp}.attn.proj.bias"))?,
        );

        if cfg.layer_scale {
            p.insert(
                format!("{lp}.ls1.gamma"),
                take_1d(wm, &format!("{lp}.ls1.gamma"))?,
            );
            p.insert(
                format!("{lp}.ls2.gamma"),
                take_1d(wm, &format!("{lp}.ls2.gamma"))?,
            );
        }

        match cfg.ffn_kind {
            FfnKind::Gelu => {
                p.insert(
                    format!("{lp}.mlp.fc1.weight"),
                    wm.take_transposed(&format!("{lp}.mlp.fc1.weight"))?.0,
                );
                p.insert(
                    format!("{lp}.mlp.fc1.bias"),
                    take_1d(wm, &format!("{lp}.mlp.fc1.bias"))?,
                );
                p.insert(
                    format!("{lp}.mlp.fc2.weight"),
                    wm.take_transposed(&format!("{lp}.mlp.fc2.weight"))?.0,
                );
                p.insert(
                    format!("{lp}.mlp.fc2.bias"),
                    take_1d(wm, &format!("{lp}.mlp.fc2.bias"))?,
                );
            }
            FfnKind::PackedSwiGLU => {
                // fc1 [2*inner, h] → value (rows 0..inner) / gate (rows inner..) → transpose to [h, inner].
                let (fc1_w, fc1_shape) = wm.take(&format!("{lp}.mlp.fc1.weight"))?;
                if fc1_shape != vec![2 * inner, h] {
                    bail!(
                        "{lp}.mlp.fc1.weight expected [{}, {h}], got {fc1_shape:?}",
                        2 * inner
                    );
                }
                let (fc1_b, _) = wm.take(&format!("{lp}.mlp.fc1.bias"))?;
                let mut val_w = vec![0f32; h * inner];
                let mut gate_w = vec![0f32; h * inner];
                for o in 0..inner {
                    let vrow = o * h;
                    let grow = (o + inner) * h;
                    for c in 0..h {
                        val_w[c * inner + o] = fc1_w[vrow + c];
                        gate_w[c * inner + o] = fc1_w[grow + c];
                    }
                }
                p.insert(format!("{lp}.mlp.fc1_value.weight"), val_w);
                p.insert(format!("{lp}.mlp.fc1_gate.weight"), gate_w);
                p.insert(format!("{lp}.mlp.fc1_value.bias"), fc1_b[0..inner].to_vec());
                p.insert(
                    format!("{lp}.mlp.fc1_gate.bias"),
                    fc1_b[inner..2 * inner].to_vec(),
                );
                p.insert(
                    format!("{lp}.mlp.fc2.weight"),
                    wm.take_transposed(&format!("{lp}.mlp.fc2.weight"))?.0,
                );
                p.insert(
                    format!("{lp}.mlp.fc2.bias"),
                    take_1d(wm, &format!("{lp}.mlp.fc2.bias"))?,
                );
            }
        }
    }
    p.insert("norm.weight".into(), take_1d(wm, "norm.weight")?);
    p.insert("norm.bias".into(), take_1d(wm, "norm.bias")?);
    Ok(p)
}

/// Deterministic small pseudo-random values in `[-scale, scale]` (test data).
fn pseudo(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Build a synthetic timm-canonical checkpoint (raw PyTorch layouts) for tests.
pub fn synthetic_checkpoint(cfg: &VitConfig, seed: u32) -> WeightMap {
    let h = cfg.hidden_size;
    let inner = cfg.ffn_inner();
    let pd = cfg.patch_dim();
    let np = cfg.num_patches();
    let ps = cfg.patch_size;
    let pos_p = if cfg.no_embed_class { np } else { 1 + np };
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut sd = seed;
    let mut r = |n: usize, scale: f32| {
        sd = sd.wrapping_add(0x9E3779B9);
        pseudo(n, sd, scale)
    };

    t.insert(
        "patch_embed.proj.weight".into(),
        (r(h * pd, 0.05), vec![h, 3, ps, ps]),
    );
    t.insert("patch_embed.proj.bias".into(), (r(h, 0.02), vec![h]));
    t.insert("cls_token".into(), (r(h, 0.02), vec![1, 1, h]));
    if cfg.num_register_tokens > 0 {
        t.insert(
            "reg_token".into(),
            (
                r(cfg.num_register_tokens * h, 0.02),
                vec![1, cfg.num_register_tokens, h],
            ),
        );
    }
    t.insert("pos_embed".into(), (r(pos_p * h, 0.02), vec![1, pos_p, h]));

    for i in 0..cfg.num_hidden_layers {
        let lp = format!("blocks.{i}");
        t.insert(format!("{lp}.norm1.weight"), (vec![1.0; h], vec![h]));
        t.insert(format!("{lp}.norm1.bias"), (vec![0.0; h], vec![h]));
        t.insert(format!("{lp}.norm2.weight"), (vec![1.0; h], vec![h]));
        t.insert(format!("{lp}.norm2.bias"), (vec![0.0; h], vec![h]));
        t.insert(
            format!("{lp}.attn.qkv.weight"),
            (r(3 * h * h, 0.05), vec![3 * h, h]),
        );
        t.insert(format!("{lp}.attn.qkv.bias"), (r(3 * h, 0.02), vec![3 * h]));
        t.insert(
            format!("{lp}.attn.proj.weight"),
            (r(h * h, 0.05), vec![h, h]),
        );
        t.insert(format!("{lp}.attn.proj.bias"), (r(h, 0.02), vec![h]));
        if cfg.layer_scale {
            t.insert(format!("{lp}.ls1.gamma"), (vec![0.1; h], vec![h]));
            t.insert(format!("{lp}.ls2.gamma"), (vec![0.1; h], vec![h]));
        }
        match cfg.ffn_kind {
            FfnKind::Gelu => {
                t.insert(
                    format!("{lp}.mlp.fc1.weight"),
                    (r(inner * h, 0.05), vec![inner, h]),
                );
                t.insert(format!("{lp}.mlp.fc1.bias"), (r(inner, 0.02), vec![inner]));
                t.insert(
                    format!("{lp}.mlp.fc2.weight"),
                    (r(h * inner, 0.05), vec![h, inner]),
                );
                t.insert(format!("{lp}.mlp.fc2.bias"), (r(h, 0.02), vec![h]));
            }
            FfnKind::PackedSwiGLU => {
                t.insert(
                    format!("{lp}.mlp.fc1.weight"),
                    (r(2 * inner * h, 0.05), vec![2 * inner, h]),
                );
                t.insert(
                    format!("{lp}.mlp.fc1.bias"),
                    (r(2 * inner, 0.02), vec![2 * inner]),
                );
                t.insert(
                    format!("{lp}.mlp.fc2.weight"),
                    (r(h * inner, 0.05), vec![h, inner]),
                );
                t.insert(format!("{lp}.mlp.fc2.bias"), (r(h, 0.02), vec![h]));
            }
        }
    }
    t.insert("norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert("norm.bias".into(), (vec![0.0; h], vec![h]));
    WeightMap::from_tensors(t)
}
