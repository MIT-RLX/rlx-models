// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A picture through a package's `vision` graph
//! (`rlx_qwen35::export_vl::symbolic_vision`): preprocessed as the runner
//! does (`preprocess_rgb`), its merge order, patch RoPE rows and the
//! position table's bilinear weights to the grid computed here.

use rlx_program::{Tensor, run};
use serde_json::Value;

use crate::config::MmProjConfig;
use crate::encoder::VisionEncodeOutput;
use crate::preprocess;

pub struct Tower {
    cfg: MmProjConfig,
    /// Side of the square position table.
    side: usize,
    head_dim: usize,
}

/// Bilinear weights `[dst, side]` onto a `side`-long axis, as the runner
/// resamples the position table (`linspace`, then two taps).
fn bilinear(dst: usize, side: usize) -> Vec<f32> {
    let mut r = vec![0f32; dst * side];
    let s = side as f32;
    for t in 0..dst {
        let g = if dst == 1 {
            0.0
        } else {
            t as f32 * (s - 1.0) / (dst as f32 - 1.0)
        };
        let y0 = g.floor() as usize;
        let y1 = (y0 + 1).min(side - 1);
        let f = g - y0 as f32;
        r[t * side + y0] += 1.0 - f;
        r[t * side + y1] += f;
    }
    r
}

impl Tower {
    pub fn from_config(v: &Value) -> Result<Self, String> {
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("config vision has no {k}"));
        let three = |k: &str| -> Result<[f32; 3], String> {
            serde_json::from_value(v[k].clone()).map_err(|e| format!("config vision {k}: {e}"))
        };
        Ok(Tower {
            cfg: MmProjConfig {
                patch_size: n("patch")?,
                n_merge: n("merge")?,
                image_min_pixels: n("min_pixels")?,
                image_max_pixels: n("max_pixels")?,
                image_mean: three("mean")?,
                image_std: three("std")?,
            },
            side: n("side")?,
            head_dim: n("head_dim")?,
        })
    }

    /// `Qwen35VisionEncoder::encode_rgb`.
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize, hidden: usize) -> anyhow::Result<VisionEncodeOutput> {
        let (nchw, tw, th) = preprocess::preprocess_rgb(rgb, w, h, &self.cfg);
        let ps = self.cfg.patch_size;
        let m = self.cfg.n_merge;
        let (hp, wp) = (th / ps, tw / ps);
        let n_pos = hp * wp;
        let idx = preprocess::build_spatial_merge_gather_idx(hp, wp, m);
        let (cos, sin) = preprocess::vision_rope_feeds(&preprocess::build_vision_position_hw(tw, th, &self.cfg), self.head_dim);
        let outs = run(
            "vision",
            vec![
                Tensor::f32("image", vec![1, 3, th, tw], &nchw),
                Tensor::f32("ry", vec![hp, self.side], &bilinear(hp, self.side)),
                Tensor::f32("rx", vec![wp, self.side], &bilinear(wp, self.side)),
                Tensor::f32("spatial_merge_idx", vec![n_pos], &idx),
                Tensor::f32("attn_mask", vec![1, n_pos], &vec![1.0; n_pos]),
                Tensor::f32("vision_rope_cos", vec![n_pos, self.head_dim], &cos),
                Tensor::f32("vision_rope_sin", vec![n_pos, self.head_dim], &sin),
            ],
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let embeddings = outs
            .first()
            .ok_or_else(|| anyhow::anyhow!("the vision tower returned nothing"))?
            .to_f32();
        let (gx, gy) = (wp / m, hp / m);
        anyhow::ensure!(embeddings.len() == gx * gy * hidden, "the vision tower returned {} values", embeddings.len());
        Ok(VisionEncodeOutput {
            embeddings,
            grid_x: gx,
            grid_y: gy,
            n_tokens: gx * gy,
        })
    }
}
