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

//! V-JEPA2 predictor — masked token prediction head.

use super::config::Vjepa2Config;
use super::layers::{block_forward, gather_rows};
use super::weights::Vjepa2PredictorWeights;
use anyhow::{Result, ensure};
use rlx_core::host_kernels::{layer_norm, linear};

/// Context / target patch indices for one batch element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vjepa2Masks {
    pub context: Vec<usize>,
    pub target: Vec<usize>,
    /// Which learned mask token vector to use (`0..pred_num_mask_tokens`).
    pub mask_index: usize,
}

pub struct Vjepa2PredictorOutput {
    pub tokens: Vec<f32>,
    pub num_target: usize,
    pub hidden: usize,
}

/// Baked gather / RoPE layout for a compiled predictor graph.
#[derive(Debug, Clone)]
pub struct Vjepa2PredictorLayout {
    pub n_ctxt: usize,
    pub n_tgt: usize,
    pub n_combined: usize,
    /// Flat `[batch, n_ctxt]` patch indices into encoder sequence.
    pub ctxt_idx: Vec<i64>,
    /// Flat `[batch, n_combined]` row gather for sort-by-position.
    pub sort_idx: Vec<i64>,
    /// Flat `[batch, n_combined]` inverse gather before target slice.
    pub unsort_idx: Vec<i64>,
    /// Flat `[batch, n_tgt, pred_hidden]` mask token rows.
    pub mask_rows: Vec<f32>,
    /// `[n_combined, pred_head_dim/2]`
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
}

/// Precompute indices and RoPE tables for [`super::builder::build_vjepa2_predictor_graph_sized`].
pub fn prepare_predictor_layout(
    cfg: &Vjepa2Config,
    masks: &Vjepa2Masks,
    batch: usize,
) -> Result<Vjepa2PredictorLayout> {
    use super::rope::build_vjepa2_rope_tables;

    ensure!(!masks.context.is_empty(), "context mask must be non-empty");
    ensure!(!masks.target.is_empty(), "target mask must be non-empty");

    let pred = cfg.pred_hidden_size;
    let pred_dh = cfg.pred_head_dim();
    let (d_dim, h_dim, w_dim) = cfg.pred_rope_segment_dims();
    let grid_h = cfg.grid_spatial();
    let grid_w = cfg.grid_spatial();
    let enc_seq = cfg.num_patches();

    let n_ctxt = masks.context.len();
    let n_tgt = masks.target.len();
    let n_combined = n_ctxt + n_tgt;

    let mut position_ids: Vec<usize> = Vec::with_capacity(n_combined);
    position_ids.extend_from_slice(&masks.context);
    position_ids.extend_from_slice(&masks.target);

    let mut order: Vec<usize> = (0..n_combined).collect();
    order.sort_by_key(|&i| position_ids[i]);

    let mut sort_idx = vec![0i64; n_combined];
    let mut unsort_idx = vec![0i64; n_combined];
    for (new_i, &old_i) in order.iter().enumerate() {
        sort_idx[new_i] = old_i as i64;
        unsort_idx[old_i] = new_i as i64;
    }

    let sorted_pos: Vec<usize> = order.iter().map(|&i| position_ids[i]).collect();
    let (full_cos, full_sin) =
        build_vjepa2_rope_tables(enc_seq, pred_dh, d_dim, h_dim, w_dim, grid_h, grid_w);
    let half = pred_dh / 2;
    let mut rope_cos = vec![0f32; n_combined * half];
    let mut rope_sin = vec![0f32; n_combined * half];
    for (i, &p) in sorted_pos.iter().enumerate() {
        rope_cos[i * half..(i + 1) * half].copy_from_slice(&full_cos[p * half..(p + 1) * half]);
        rope_sin[i * half..(i + 1) * half].copy_from_slice(&full_sin[p * half..(p + 1) * half]);
    }

    let mut ctxt_idx = Vec::with_capacity(batch * n_ctxt);
    let mut sort_flat = Vec::with_capacity(batch * n_combined);
    let mut unsort_flat = Vec::with_capacity(batch * n_combined);
    for _ in 0..batch {
        ctxt_idx.extend(masks.context.iter().map(|&i| i as i64));
        sort_flat.extend_from_slice(&sort_idx);
        unsort_flat.extend_from_slice(&unsort_idx);
    }

    Ok(Vjepa2PredictorLayout {
        n_ctxt,
        n_tgt,
        n_combined,
        ctxt_idx,
        sort_idx: sort_flat,
        unsort_idx: unsort_flat,
        mask_rows: vec![0f32; batch * n_tgt * pred],
        rope_cos,
        rope_sin,
    })
}

/// Tile the selected mask token into `[batch, n_tgt, pred_hidden]` row-major.
pub fn predictor_mask_rows(
    weights: &super::weights::Vjepa2PredictorWeights,
    cfg: &Vjepa2Config,
    masks: &Vjepa2Masks,
    batch: usize,
) -> Vec<f32> {
    let pred = cfg.pred_hidden_size;
    let n_tgt = masks.target.len();
    let mask_idx = masks.mask_index % cfg.pred_num_mask_tokens;
    let mask_vec = &weights.mask_tokens[mask_idx * pred..(mask_idx + 1) * pred];
    let mut rows = Vec::with_capacity(batch * n_tgt * pred);
    for _ in 0..batch {
        for _ in 0..n_tgt {
            rows.extend_from_slice(mask_vec);
        }
    }
    rows
}

/// Run the predictor on encoder outputs `[batch, seq, enc_dim]` flat.
pub fn predict_native(
    encoder_tokens: &[f32],
    weights: &Vjepa2PredictorWeights,
    cfg: &Vjepa2Config,
    batch: usize,
    seq: usize,
    masks: &Vjepa2Masks,
) -> Result<Vjepa2PredictorOutput> {
    let enc = cfg.hidden_size;
    let pred = cfg.pred_hidden_size;
    let nh = cfg.pred_num_attention_heads;
    let head_dim = cfg.pred_head_dim();
    let (d_dim, h_dim, w_dim) = cfg.pred_rope_segment_dims();
    let grid_t = cfg.grid_temporal();
    let grid_h = cfg.grid_spatial();
    let grid_w = cfg.grid_spatial();
    let eps = cfg.layer_norm_eps as f32;

    ensure!(!masks.context.is_empty(), "context mask must be non-empty");
    ensure!(!masks.target.is_empty(), "target mask must be non-empty");

    let n_ctxt = masks.context.len();
    let n_tgt = masks.target.len();
    let n_combined = n_ctxt + n_tgt;

    let mut per_batch = Vec::with_capacity(batch * n_combined * pred);

    for bi in 0..batch {
        let enc_batch = &encoder_tokens[bi * seq * enc..(bi + 1) * seq * enc];
        let ctxt = gather_rows(enc_batch, &masks.context, seq, enc);
        let mut x = linear(
            &ctxt,
            n_ctxt,
            enc,
            &weights.embed_w_t,
            pred,
            &weights.embed_b,
        )?;

        let mask_idx = masks.mask_index % cfg.pred_num_mask_tokens;
        let mask_vec = &weights.mask_tokens[mask_idx * pred..(mask_idx + 1) * pred];
        let mut targets = vec![0f32; n_tgt * pred];
        for ti in 0..n_tgt {
            targets[ti * pred..(ti + 1) * pred].copy_from_slice(mask_vec);
        }

        x.extend_from_slice(&targets);

        let mut position_ids: Vec<usize> = Vec::with_capacity(n_combined);
        position_ids.extend_from_slice(&masks.context);
        position_ids.extend_from_slice(&masks.target);

        // Sort by patch index (argsort of position_ids).
        let mut order: Vec<usize> = (0..n_combined).collect();
        order.sort_by_key(|&i| position_ids[i]);
        let mut sorted_pos = vec![0usize; n_combined];
        let mut sorted_x = vec![0f32; n_combined * pred];
        for (new_i, &old_i) in order.iter().enumerate() {
            sorted_pos[new_i] = position_ids[old_i];
            sorted_x[new_i * pred..(new_i + 1) * pred]
                .copy_from_slice(&x[old_i * pred..(old_i + 1) * pred]);
        }
        x = sorted_x;
        position_ids = sorted_pos;

        for block in &weights.blocks {
            block_forward(
                &mut x,
                block,
                1,
                n_combined,
                pred,
                nh,
                head_dim,
                d_dim,
                h_dim,
                w_dim,
                grid_t,
                grid_h,
                grid_w,
                eps,
                Some(&position_ids),
            )?;
        }
        x = layer_norm(&x, &weights.norm_w, &weights.norm_b, pred, eps)?;

        // Unsort and take target slice.
        let mut unsorted = vec![0f32; n_combined * pred];
        for (new_i, &old_i) in order.iter().enumerate() {
            unsorted[old_i * pred..(old_i + 1) * pred]
                .copy_from_slice(&x[new_i * pred..(new_i + 1) * pred]);
        }
        let target_slice = &unsorted[n_ctxt * pred..];
        let projected = linear(
            target_slice,
            n_tgt,
            pred,
            &weights.proj_w_t,
            enc,
            &weights.proj_b,
        )?;
        per_batch.extend_from_slice(&projected);
    }

    Ok(Vjepa2PredictorOutput {
        tokens: per_batch,
        num_target: n_tgt,
        hidden: enc,
    })
}
