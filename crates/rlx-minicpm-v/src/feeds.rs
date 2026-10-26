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

//! What MiniCPM-V's vision graphs take for one crop: its patches padded to
//! the bucket, position ids, masks and the regrouping orders of its
//! mergers, all computed from the crop's grid. Plain host math, which the
//! exported package's program builds too ([`crate::vision`] builds the
//! graphs).

use anyhow::{Result, ensure};

use crate::config::{DownsampleMode, MiniCpmVConfig, Projector};

/// Crop patch counts a tower graph is compiled for are multiples of this.
pub const BUCKET: usize = 256;
pub(crate) const IN_PATCHES: &str = "patches";
pub(crate) const IN_POS: &str = "pos_ids";
pub(crate) const IN_MASK: &str = "key_mask";
pub(crate) const IN_WIN: &str = "win_idx";
pub(crate) const IN_MASK2: &str = "key_mask2";
pub(crate) const IN_MERGE: &str = "merge_idx";
pub(crate) const IN_RPOS: &str = "resampler_pos";
pub(crate) const IN_RMASK: &str = "resampler_mask";
pub(crate) const IN_FEATS: &str = "tower_features";

/// Bucket to run a crop of `n` patches in, given the buckets already
/// compiled: the smallest compiled one that holds it (padding rows are masked,
/// so a larger bucket gives the same features), else a new one.
pub fn pick_bucket(compiled: impl IntoIterator<Item = usize>, n: usize) -> usize {
    let need = bucket_for(n);
    compiled
        .into_iter()
        .filter(|&b| b >= need)
        .min()
        .unwrap_or(need)
}

/// Smallest bucket holding `n` patches.
pub fn bucket_for(n: usize) -> usize {
    n.div_ceil(BUCKET).max(1) * BUCKET
}

/// Position ids of transformers' `Idefics2VisionEmbeddings` (4.5x) for a
/// crop it reads as one row of `n` patches: `j / n · (1 − 1e-6)` in f32,
/// bucketed like [`nearest_position_ids`].
pub fn flat_position_ids(n: usize, side: usize) -> Vec<u32> {
    let step = 1.0f64 / side as f64;
    let m = ((1.0 - step) / step).ceil() as usize;
    let bounds: Vec<f32> = (0..m).map(|i| (step + i as f64 * step) as f32).collect();
    let shrink = (1.0f64 - 1e-6) as f32;
    (0..n)
        .map(|j| {
            let x = j as f32 / n as f32 * shrink;
            bounds.iter().take_while(|&&b| b <= x).count() as u32
        })
        .collect()
}

/// `get_vision_nearest_position_ids` for one `h × w` grid: each patch's
/// fractional coordinate `i / h` is bucketed onto the `side × side` learned
/// grid with `torch.bucketize(right=True)`.
///
/// The boundaries are `torch.arange(1/side, 1, 1/side)` in f32 (computed in
/// f64 and rounded, as torch does) and the coordinates are f32 divisions;
/// a coordinate that lands exactly on a boundary belongs to the upper bucket.
pub fn nearest_position_ids(h: usize, w: usize, side: usize) -> Vec<u32> {
    let step = 1.0f64 / side as f64;
    let n = ((1.0 - step) / step).ceil() as usize;
    let bounds: Vec<f32> = (0..n).map(|i| (step + i as f64 * step) as f32).collect();
    let bucket = |x: f32| bounds.iter().take_while(|&&b| b <= x).count() as u32;
    let hb: Vec<u32> = (0..h).map(|i| bucket(i as f32 / h as f32)).collect();
    let wb: Vec<u32> = (0..w).map(|j| bucket(j as f32 / w as f32)).collect();
    let mut out = Vec::with_capacity(h * w);
    for &bh in &hb {
        for &bw in &wb {
            out.push(bh * side as u32 + bw);
        }
    }
    out
}

/// Row order that groups an `h × w` grid into `k × k` windows: windows
/// row-major, each window's rows row-major (`get_vision_window_index` with
/// a grid divisible by `k`). Also the 2×2 concat order of the merger.
pub fn window_order(h: usize, w: usize, k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(h * w);
    for wy in 0..h / k {
        for wx in 0..w / k {
            for dy in 0..k {
                for dx in 0..k {
                    out.push(((wy * k + dy) * w + wx * k + dx) as u32);
                }
            }
        }
    }
    out
}

/// Host-computed feeds for one crop in a bucket of `s` patches.
#[derive(Debug, Clone)]
pub struct CropFeeds {
    pub patches: Vec<f32>,
    pub pos_ids: Vec<f32>,
    pub key_mask: Vec<f32>,
    pub win_idx: Vec<f32>,
    pub key_mask2: Vec<f32>,
    pub merge_idx: Vec<f32>,
    /// Resampler key positions `[s, text_hidden]` (zeros on padding rows).
    pub resampler_pos: Vec<f32>,
    /// Resampler key mask `[s]`: the crop's real patches.
    pub resampler_mask: Vec<f32>,
    /// LLM tokens this crop yields (the output's leading rows).
    pub n_tokens: usize,
}

/// Lay one crop out in a bucket: real rows first, padding after, with every
/// gather index of a padding row pointing at itself.
pub fn crop_feeds(
    cfg: &MiniCpmVConfig,
    grid_h: usize,
    grid_w: usize,
    patches: &[f32],
    s: usize,
) -> Result<CropFeeds> {
    crop_feeds_padded(cfg, grid_h, grid_w, patches, s, grid_h * grid_w)
}

/// [`crop_feeds`] where the tower sees `tower_rows ≥ h·w` rows unmasked —
/// the crop's patches, then zero pixels — with flat positions over all of
/// them, while the resampler still attends to the real patches only.
///
/// This is MiniCPM-Llama3-V 2.5's own computation: it pads every crop of a
/// prompt to the longest one and builds the tower's patch mask with
/// `patch_attn_mask[i, :n] = True` on a `[B, 1, max]` tensor, which slices
/// the size-1 axis and so marks every row of every crop valid. Only
/// configurations with [`VisionConfig::flat_positions`](crate::VisionConfig)
/// and a resampler take it.
pub fn crop_feeds_padded(
    cfg: &MiniCpmVConfig,
    grid_h: usize,
    grid_w: usize,
    patches: &[f32],
    s: usize,
    tower_rows: usize,
) -> Result<CropFeeds> {
    let v = &cfg.vision;
    let n = grid_h * grid_w;
    let dim = 3 * v.patch_size * v.patch_size;
    let f = cfg.downsample.merge_factor();
    let resampler = match cfg.projector {
        Projector::Resampler { queries } => Some(queries),
        Projector::Merger => None,
    };
    ensure!(
        resampler.is_some() || (grid_h.is_multiple_of(f) && grid_w.is_multiple_of(f)),
        "minicpm-v: patch grid {grid_h}x{grid_w} is not divisible by {f}"
    );
    ensure!(n <= s, "minicpm-v: crop has {n} patches, bucket {s}");
    ensure!(
        tower_rows >= n && tower_rows <= s,
        "minicpm-v: tower rows {tower_rows} for a crop of {n} in bucket {s}"
    );
    ensure!(
        tower_rows == n || (resampler.is_some() && v.flat_positions),
        "minicpm-v: padded tower rows only apply to the 2.5 layout"
    );
    ensure!(
        patches.len() == n * dim,
        "minicpm-v: crop patch buffer size"
    );
    let mut pf = vec![0f32; s * dim];
    pf[..n * dim].copy_from_slice(patches);
    let mut pos = vec![0f32; s];
    let ids = if v.flat_positions {
        flat_position_ids(tower_rows, v.num_patches_per_side)
    } else {
        nearest_position_ids(grid_h, grid_w, v.num_patches_per_side)
    };
    for (i, p) in ids.into_iter().enumerate() {
        pos[i] = p as f32;
    }
    let mask = |len: usize, real: usize| -> Vec<f32> {
        (0..len).map(|i| if i < real { 1.0 } else { 0.0 }).collect()
    };
    let padded = |order: Vec<u32>, len: usize| -> Vec<f32> {
        let real = order.len();
        order
            .into_iter()
            .map(|i| i as f32)
            .chain((real..len).map(|i| i as f32))
            .collect()
    };

    if let Some(queries) = resampler {
        let d = cfg.text_hidden_size;
        let mut rpos = vec![0f32; s * d];
        rpos[..n * d].copy_from_slice(&sincos_2d(d, grid_h, grid_w));
        return Ok(CropFeeds {
            patches: pf,
            pos_ids: pos,
            key_mask: mask(s, tower_rows),
            win_idx: Vec::new(),
            key_mask2: Vec::new(),
            merge_idx: Vec::new(),
            resampler_pos: rpos,
            resampler_mask: mask(s, n),
            n_tokens: queries,
        });
    }
    let (win_idx, key_mask2, merge_idx, n_tokens) = match cfg.downsample {
        DownsampleMode::X16 => {
            let k = v.window;
            let (mh, mw) = (grid_h / k, grid_w / k);
            let m = cfg.merge_kernel;
            (
                padded(window_order(grid_h, grid_w, k), s),
                mask(s / 4, n / 4),
                padded(window_order(mh, mw, m), s / 4),
                n / 16,
            )
        }
        DownsampleMode::X4 => {
            let m = cfg.merge_kernel;
            (
                Vec::new(),
                Vec::new(),
                padded(window_order(grid_h, grid_w, m), s),
                n / 4,
            )
        }
    };
    Ok(CropFeeds {
        patches: pf,
        pos_ids: pos,
        key_mask: mask(s, n),
        win_idx,
        key_mask2,
        merge_idx,
        resampler_pos: Vec::new(),
        resampler_mask: Vec::new(),
        n_tokens,
    })
}

/// 4.5's temporal position of a video frame, `[d]`:
/// `get_1d_sincos_pos_embed_from_temporal_size` in f32 —
/// `[sin(t·ω), cos(t·ω)]` with `ω_i = 10000^(-i / (d/2))`.
pub fn sincos_temporal(d: usize, t: usize) -> Vec<f32> {
    let half = d / 2;
    let omega: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / (d as f32 / 2.0)))
        .collect();
    let mut out: Vec<f32> = omega.iter().map(|o| (t as f32 * o).sin()).collect();
    out.extend(omega.iter().map(|o| (t as f32 * o).cos()));
    out
}

/// One frame of a packed group: its tower features (`[h·w, c]` leading rows),
/// patch grid and temporal id.
pub struct GroupFrame<'a> {
    pub features: &'a [f32],
    pub grid_h: usize,
    pub grid_w: usize,
    pub temporal_id: usize,
}

/// Host feeds of a resampler-only graph of `rows` rows for one group of
/// frames: their tower features stacked, the 2-D position of every patch plus
/// its frame's temporal position on the keys, and a mask over the real rows.
pub fn group_resampler_feeds(
    cfg: &MiniCpmVConfig,
    frames: &[GroupFrame<'_>],
    rows: usize,
) -> Result<Vec<(&'static str, Vec<f32>)>> {
    let c = cfg.vision.hidden_size;
    let d = cfg.text_hidden_size;
    let total: usize = frames.iter().map(|f| f.grid_h * f.grid_w).sum();
    ensure!(
        total <= rows,
        "minicpm-v: {total} rows for a {rows}-row resampler"
    );
    let mut feats = vec![0f32; rows * c];
    let mut pos = vec![0f32; rows * d];
    let mut mask = vec![0f32; rows];
    let mut at = 0usize;
    for f in frames {
        let n = f.grid_h * f.grid_w;
        ensure!(
            f.features.len() >= n * c,
            "minicpm-v: tower features too short"
        );
        feats[at * c..(at + n) * c].copy_from_slice(&f.features[..n * c]);
        let p2 = sincos_2d(d, f.grid_h, f.grid_w);
        let pt = sincos_temporal(d, f.temporal_id);
        for i in 0..n {
            let row = &mut pos[(at + i) * d..(at + i + 1) * d];
            for ((dst, a), b) in row.iter_mut().zip(&p2[i * d..(i + 1) * d]).zip(&pt) {
                *dst = a + b;
            }
            mask[at + i] = 1.0;
        }
        at += n;
    }
    Ok(vec![(IN_FEATS, feats), (IN_RPOS, pos), (IN_RMASK, mask)])
}

/// Host feeds of a tower-only graph for one crop in a bucket of `s`.
pub fn tower_feeds(
    cfg: &MiniCpmVConfig,
    crop: &crate::preprocess::Crop,
    s: usize,
) -> Result<Vec<(&'static str, Vec<f32>)>> {
    let f = crop_feeds(cfg, crop.grid_h, crop.grid_w, &crop.patches, s)?;
    Ok(vec![
        (IN_PATCHES, f.patches),
        (IN_POS, f.pos_ids),
        (IN_MASK, f.key_mask),
    ])
}

/// The resampler's key positions for an `h × w` grid, `[h·w, d]` row-major:
/// `get_2d_sincos_pos_embed` in f32. Per patch `(r, c)`, with
/// `ω_i = 10000^(-i / (d/4))`: `[sin(c·ω), cos(c·ω), sin(r·ω), cos(r·ω)]` —
/// the column comes first because the reference's `meshgrid` puts `w` first.
pub fn sincos_2d(d: usize, h: usize, w: usize) -> Vec<f32> {
    let q = d / 4;
    let omega: Vec<f32> = (0..q)
        .map(|i| {
            let e = i as f32 / (d as f32 / 4.0);
            1.0 / 10000f32.powf(e)
        })
        .collect();
    let mut out = Vec::with_capacity(h * w * d);
    for r in 0..h {
        for c in 0..w {
            for coord in [c as f32, r as f32] {
                out.extend(omega.iter().map(|o| (coord * o).sin()));
                out.extend(omega.iter().map(|o| (coord * o).cos()));
            }
        }
    }
    out
}

impl CropFeeds {
    /// `(name, data)` pairs in the order the compiled graph declares them.
    pub fn as_inputs(&self) -> Vec<(&'static str, &[f32])> {
        let mut v = vec![
            (IN_PATCHES, self.patches.as_slice()),
            (IN_POS, self.pos_ids.as_slice()),
            (IN_MASK, self.key_mask.as_slice()),
        ];
        if !self.win_idx.is_empty() {
            v.push((IN_WIN, self.win_idx.as_slice()));
            v.push((IN_MASK2, self.key_mask2.as_slice()));
        }
        if self.resampler_pos.is_empty() {
            v.push((IN_MERGE, self.merge_idx.as_slice()));
        } else {
            v.push((IN_RPOS, self.resampler_pos.as_slice()));
            v.push((IN_RMASK, self.resampler_mask.as_slice()));
        }
        v
    }
}
