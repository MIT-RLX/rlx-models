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

//! CLIP-L/14-224 vision tower — global semantic branch of the deep encoder.
//!
//! Eager host f32 port of HF's `CLIPVisionTransformer` (`VitModel` in
//! `deepencoder.py`): patch/class embedding → resized absolute position
//! embedding → `pre_layrnorm` → 24 bidirectional transformer layers
//! (LN → combined-QKV self-attention → residual → LN → `quick_gelu` MLP →
//! residual).
//!
//! DeepSeek-OCR / Unlimited-OCR feeds this tower [`crate::sam_tower::SamTower`]'s
//! `[1024, q, q]` neck output as the patch-embedding tokens instead of running
//! CLIP's own `Conv2d` patch embed (see [`Self::encode`] / [`Self::encode_batch`]);
//! the raw-pixel `Conv2d` path is kept as a fallback for standalone use / testing.

use crate::config::ClipTowerConfig;
use crate::host_math;
use crate::nn;
use crate::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;

/// HF `CLIPVisionConfig.layer_norm_eps` default (the published checkpoint
/// does not override it).
const LAYER_NORM_EPS: f32 = 1e-5;

struct ClipBlockWeights {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    qkv_w: Vec<f32>, // [3*hidden, hidden]
    qkv_b: Vec<f32>,
    out_w: Vec<f32>, // [hidden, hidden]
    out_b: Vec<f32>,
    fc1_w: Vec<f32>, // [intermediate, hidden]
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>, // [hidden, intermediate]
    fc2_b: Vec<f32>,
}

/// Eager CLIP-L/14-224 encoder: weights + config, no compiled graph.
pub struct ClipTower {
    pub config: ClipTowerConfig,
    class_embedding: Option<Vec<f32>>,
    /// `Conv2d` patch-embedding weight (`[hidden, 3, patch, patch]`, `bias=False`),
    /// only needed by the raw-pixel fallback path in [`Self::encode`].
    patch_embed_w: Option<Vec<f32>>,
    /// `[1 + pretrained_grid^2, hidden]`, row 0 is the CLS position.
    position_embedding: Option<Vec<f32>>,
    pre_ln_g: Option<Vec<f32>>,
    pre_ln_b: Option<Vec<f32>>,
    blocks: Vec<ClipBlockWeights>,
}

impl ClipTower {
    pub fn from_config(config: &ClipTowerConfig) -> Self {
        Self {
            config: config.clone(),
            class_embedding: None,
            patch_embed_w: None,
            position_embedding: None,
            pre_ln_g: None,
            pre_ln_b: None,
            blocks: Vec::new(),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.config.hidden_size / self.config.num_attention_heads
    }

    pub fn load(&mut self, store: &UnlimitedOcrWeightStore) -> Result<()> {
        let mut map = store.load_clip_tower()?;
        let hidden = self.config.hidden_size;

        let (cls, cls_shape) = map
            .take(UnlimitedOcrWeightPrefix::clip_class_embedding())
            .context("clip embeddings.class_embedding")?;
        ensure!(
            cls.len() == hidden,
            "clip class_embedding shape {cls_shape:?} != [{hidden}]"
        );
        self.class_embedding = Some(cls);

        // Optional: only the sam-feature fusion path (encode/encode_batch
        // with non-empty `sam_features`) is exercised by the deep encoder;
        // the raw-pixel Conv2d patch embed may be absent from slim checkpoints.
        let patch_key = UnlimitedOcrWeightPrefix::clip_patch_embedding_w();
        if map.has(patch_key) {
            let (w, _) = map
                .take(patch_key)
                .context("clip embeddings.patch_embedding.weight")?;
            self.patch_embed_w = Some(w);
        }

        let (pos, _) = map
            .take(UnlimitedOcrWeightPrefix::clip_position_embedding_w())
            .context("clip embeddings.position_embedding.weight")?;
        ensure!(
            pos.len().is_multiple_of(hidden) && pos.len() / hidden >= 1,
            "clip position_embedding shape mismatch (len {} not a multiple of hidden {hidden})",
            pos.len()
        );
        self.position_embedding = Some(pos);

        let (pre_g, _) = map
            .take(UnlimitedOcrWeightPrefix::clip_pre_layernorm_w())
            .context("clip pre_layrnorm.weight")?;
        let (pre_b, _) = map
            .take(UnlimitedOcrWeightPrefix::clip_pre_layernorm_b())
            .context("clip pre_layrnorm.bias")?;
        self.pre_ln_g = Some(pre_g);
        self.pre_ln_b = Some(pre_b);

        self.blocks = Vec::with_capacity(self.config.num_hidden_layers);
        for i in 0..self.config.num_hidden_layers {
            let take = |m: &mut WeightMap, suffix: &str| -> Result<Vec<f32>> {
                let key = UnlimitedOcrWeightPrefix::clip_block(i, suffix);
                Ok(m.take(&key).with_context(|| format!("clip {key}"))?.0)
            };
            self.blocks.push(ClipBlockWeights {
                ln1_g: take(&mut map, "layer_norm1.weight")?,
                ln1_b: take(&mut map, "layer_norm1.bias")?,
                ln2_g: take(&mut map, "layer_norm2.weight")?,
                ln2_b: take(&mut map, "layer_norm2.bias")?,
                qkv_w: take(&mut map, "self_attn.qkv_proj.weight")?,
                qkv_b: take(&mut map, "self_attn.qkv_proj.bias")?,
                out_w: take(&mut map, "self_attn.out_proj.weight")?,
                out_b: take(&mut map, "self_attn.out_proj.bias")?,
                fc1_w: take(&mut map, "mlp.fc1.weight")?,
                fc1_b: take(&mut map, "mlp.fc1.bias")?,
                fc2_w: take(&mut map, "mlp.fc2.weight")?,
                fc2_b: take(&mut map, "mlp.fc2.bias")?,
            });
        }

        Ok(())
    }

    /// Encode one preprocessed CLIP-branch view into `[(1+q*q)*hidden]`
    /// tokens (CLS + patches, row-major). When `sam_features` (`[1024, q, q]`
    /// NCHW, [`crate::sam_tower::SamTower::encode`]'s output) is non-empty,
    /// it is used as the patch tokens directly (CLIP's own `Conv2d` patch
    /// embed is skipped, matching `deepencoder.py`'s fusion). Otherwise
    /// `pixels` (`[3, side, side]` CHW, `side` inferred from its length)
    /// is embedded via CLIP's own patch `Conv2d`.
    pub fn encode(&self, pixels: &[f32], sam_features: &[f32]) -> Result<Vec<f32>> {
        if !sam_features.is_empty() {
            self.encode_from_sam_features(sam_features)
        } else {
            ensure!(
                !pixels.is_empty(),
                "clip encode: both pixels and sam_features are empty"
            );
            ensure!(
                pixels.len().is_multiple_of(3),
                "clip encode: pixel buffer len {} not a multiple of 3",
                pixels.len()
            );
            let side = ((pixels.len() / 3) as f64).sqrt().round() as usize;
            ensure!(
                3 * side * side == pixels.len(),
                "clip encode: pixel buffer is not a square CHW image"
            );
            self.encode_from_pixels(pixels, side)
        }
    }

    /// Batched [`Self::encode`]. `pixels` is `[batch, 3, side, side]` NCHW
    /// (ignored, aside from shape validation, when `sam_features` is
    /// non-empty); `sam_features` is `[batch, 1024, q, q]` NCHW. Returns
    /// `[batch, 1+q*q, hidden]` row-major.
    pub fn encode_batch(
        &self,
        pixels: &[f32],
        sam_features: &[f32],
        side: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        ensure!(batch > 0, "clip encode_batch: batch must be > 0");
        let mut out = Vec::new();
        if !sam_features.is_empty() {
            ensure!(
                sam_features.len().is_multiple_of(batch),
                "clip encode_batch: sam_features len {} not divisible by batch {batch}",
                sam_features.len()
            );
            let per_image = sam_features.len() / batch;
            for b in 0..batch {
                let feat = &sam_features[b * per_image..(b + 1) * per_image];
                out.extend(self.encode_from_sam_features(feat)?);
            }
        } else {
            let per_image = 3 * side * side;
            ensure!(
                pixels.len() == batch * per_image,
                "clip encode_batch: pixel buffer len {} != {batch}*{per_image}",
                pixels.len()
            );
            for b in 0..batch {
                let px = &pixels[b * per_image..(b + 1) * per_image];
                out.extend(self.encode_from_pixels(px, side)?);
            }
        }
        Ok(out)
    }

    /// Core transformer body shared by both patch-token sources: `patch_tokens`
    /// is `[q*q, hidden]` channel-last (already the "patch embedding" HF adds
    /// the CLS token and resized position embedding to).
    fn encode_tokens(&self, patch_tokens: &[f32], q: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let dh = self.head_dim();
        let hw = q * q;
        ensure!(
            patch_tokens.len() == hw * hidden,
            "clip encode_tokens: patch token shape mismatch"
        );
        let s = 1 + hw;

        let cls = self.class_embedding.as_ref().context("clip not loaded")?;
        let mut x = vec![0f32; s * hidden];
        x[0..hidden].copy_from_slice(cls);
        x[hidden..].copy_from_slice(patch_tokens);

        let pos = self.abs_pos(q)?;
        nn::add_inplace(&mut x, &pos);

        let pre_g = self.pre_ln_g.as_ref().context("clip not loaded")?;
        let pre_b = self.pre_ln_b.as_ref().context("clip not loaded")?;
        let mut x = nn::layer_norm(&x, s, hidden, pre_g, pre_b, LAYER_NORM_EPS);

        for blk in &self.blocks {
            let normed = nn::layer_norm(&x, s, hidden, &blk.ln1_g, &blk.ln1_b, LAYER_NORM_EPS);
            let attn_out = self_attention(&normed, s, heads, dh, blk)?;
            nn::add_inplace(&mut x, &attn_out);

            let normed2 = nn::layer_norm(&x, s, hidden, &blk.ln2_g, &blk.ln2_b, LAYER_NORM_EPS);
            let mut inter = nn::linear_wt(
                &normed2,
                s,
                hidden,
                &blk.fc1_w,
                self.config.intermediate_size,
                Some(&blk.fc1_b),
            )?;
            nn::quick_gelu(&mut inter);
            let ffn = nn::linear_wt(
                &inter,
                s,
                self.config.intermediate_size,
                &blk.fc2_w,
                hidden,
                Some(&blk.fc2_b),
            )?;
            nn::add_inplace(&mut x, &ffn);
        }

        Ok(x)
    }

    fn encode_from_sam_features(&self, sam_features: &[f32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        ensure!(
            sam_features.len().is_multiple_of(hidden),
            "clip: sam_features len {} not a multiple of hidden {hidden}",
            sam_features.len()
        );
        let hw = sam_features.len() / hidden;
        let q = (hw as f64).sqrt().round() as usize;
        ensure!(
            q * q == hw,
            "clip: sam_features spatial size {hw} is not a perfect square"
        );
        let patch_tokens = chw_to_hwc(sam_features, q, q, hidden);
        self.encode_tokens(&patch_tokens, q)
    }

    fn encode_from_pixels(&self, pixels: &[f32], side: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let patch = self.config.patch_size;
        ensure!(
            side.is_multiple_of(patch),
            "clip: side {side} not divisible by patch {patch}"
        );
        ensure!(
            pixels.len() == 3 * side * side,
            "clip: pixel buffer len mismatch"
        );
        let grid = side / patch;

        let hwc = chw_to_hwc(pixels, side, side, 3);
        let patch_w = self.patch_embed_w.as_ref().context(
            "clip not loaded (patch_embedding.weight) — required for the raw-pixel path",
        )?;
        let (tokens, oh, ow) = nn::conv2d_hwc(
            &hwc, side, side, 3, patch_w, hidden, patch, patch, patch, 0, None,
        )?;
        ensure!(
            oh == grid && ow == grid,
            "clip patch_embed grid mismatch: {oh}x{ow} != {grid}x{grid}"
        );
        self.encode_tokens(&tokens, grid)
    }

    /// HF `get_abs_pos`: keep the CLS position row as-is, bicubic-resize the
    /// (square) patch-position grid from its pretrained resolution to `q x q`.
    fn abs_pos(&self, q: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let pos = self
            .position_embedding
            .as_ref()
            .context("clip not loaded")?;
        let total = pos.len() / hidden;
        ensure!(total >= 1, "clip position_embedding too small");
        let patch_total = total - 1;
        let pretrained_grid = (patch_total as f64).sqrt().round() as usize;
        ensure!(
            pretrained_grid * pretrained_grid == patch_total,
            "clip position_embedding patch count {patch_total} is not a perfect square"
        );

        let cls_pos = &pos[0..hidden];
        if pretrained_grid == q {
            let mut out = vec![0f32; (1 + q * q) * hidden];
            out[0..hidden].copy_from_slice(cls_pos);
            out[hidden..].copy_from_slice(&pos[hidden..]);
            return Ok(out);
        }

        let patch_pos = &pos[hidden..];
        let resized =
            nn::bicubic_resize_hwc(patch_pos, pretrained_grid, pretrained_grid, hidden, q, q);
        let mut out = vec![0f32; (1 + q * q) * hidden];
        out[0..hidden].copy_from_slice(cls_pos);
        out[hidden..].copy_from_slice(&resized);
        Ok(out)
    }
}

fn chw_to_hwc(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * c];
    for ci in 0..c {
        for p in 0..h * w {
            out[p * c + ci] = x[ci * h * w + p];
        }
    }
    out
}

/// Standard (non-causal, no rel-pos) multi-head self-attention with a
/// combined `qkv_proj` weight, matching HF CLIP's `CLIPAttention`.
fn self_attention(
    x: &[f32],
    s: usize,
    heads: usize,
    head_dim: usize,
    blk: &ClipBlockWeights,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    let qkv = nn::linear_wt(x, s, hidden, &blk.qkv_w, 3 * hidden, Some(&blk.qkv_b))?;

    let mut q = vec![0f32; s * hidden];
    let mut k = vec![0f32; s * hidden];
    let mut v = vec![0f32; s * hidden];
    for i in 0..s {
        let row = &qkv[i * 3 * hidden..(i + 1) * 3 * hidden];
        q[i * hidden..(i + 1) * hidden].copy_from_slice(&row[0..hidden]);
        k[i * hidden..(i + 1) * hidden].copy_from_slice(&row[hidden..2 * hidden]);
        v[i * hidden..(i + 1) * hidden].copy_from_slice(&row[2 * hidden..3 * hidden]);
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let merged = host_math::mha_with_mask(&q, &k, &v, s, s, heads, head_dim, scale, None)?;
    nn::linear_wt(&merged, s, hidden, &blk.out_w, hidden, Some(&blk.out_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chw_to_hwc_matches_manual_transpose() {
        // [c=2, h=2, w=2] -> [hw=4, c=2].
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = chw_to_hwc(&x, 2, 2, 2);
        assert_eq!(out, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
    }

    #[test]
    fn config_defaults_match_checkpoint() {
        let cfg = ClipTowerConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.intermediate_size, 4096);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.image_size, 224);
    }

    #[test]
    fn head_dim_matches_hidden_over_heads() {
        let cfg = ClipTowerConfig::default();
        let tower = ClipTower::from_config(&cfg);
        assert_eq!(tower.head_dim(), 64);
    }

    #[test]
    fn abs_pos_identity_when_grid_matches_pretrained() {
        let cfg = ClipTowerConfig::default();
        let mut tower = ClipTower::from_config(&cfg);
        let pretrained_grid = cfg.image_size / cfg.patch_size; // 16
        let total = 1 + pretrained_grid * pretrained_grid;
        let pos: Vec<f32> = (0..total * cfg.hidden_size).map(|i| i as f32).collect();
        tower.position_embedding = Some(pos.clone());
        let out = tower.abs_pos(pretrained_grid).expect("abs_pos");
        assert_eq!(out, pos);
    }

    #[test]
    fn encode_rejects_empty_inputs() {
        let cfg = ClipTowerConfig::default();
        let tower = ClipTower::from_config(&cfg);
        assert!(tower.encode(&[], &[]).is_err());
    }
}
