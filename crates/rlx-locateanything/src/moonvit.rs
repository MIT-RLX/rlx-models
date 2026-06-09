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

//! MoonViT vision encoder — CPU reference + compiled RLX graph.

use crate::config::MoonVitConfig;
use crate::moonvit_flow::build_moonvit_built;
use crate::preprocess::PreprocessedImage;
use crate::rope2d::{apply_rope_2d, freqs_cis_for_grid};
use crate::weights::LocateAnythingWeightPrefix;
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

const ROPE_THETA: f64 = 10_000.0;

/// Per-(grid, device) compiled MoonViT session cache.
#[derive(Default)]
pub struct MoonVitCache {
    graphs: HashMap<VitCacheKey, CompiledGraph>,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct VitCacheKey {
    grid_h: usize,
    grid_w: usize,
    device: Device,
}

impl MoonVitCache {
    pub fn has_graph(&self, img: &PreprocessedImage, device: Device) -> bool {
        let key = VitCacheKey {
            grid_h: img.grid_h,
            grid_w: img.grid_w,
            device,
        };
        self.graphs.contains_key(&key)
    }

    pub fn encode(
        &mut self,
        cfg: &MoonVitConfig,
        weights: Option<&mut WeightMap>,
        img: &PreprocessedImage,
        device: Device,
    ) -> Result<Vec<f32>> {
        let key = VitCacheKey {
            grid_h: img.grid_h,
            grid_w: img.grid_w,
            device,
        };
        if let std::collections::hash_map::Entry::Vacant(e) = self.graphs.entry(key) {
            let wm = weights.ok_or_else(|| {
                anyhow::anyhow!("MoonViT weights required for first compile on this grid")
            })?;
            let built = build_moonvit_built(cfg, wm, 1, img.grid_h, img.grid_w, device)?;
            let params = built.model.params().clone();
            let mut compiled = compile_built(built.model, device)?;
            for (n, d) in &params {
                compiled.set_param(n, d);
            }
            e.insert(compiled);
        }
        let graph = self.graphs.get_mut(&key).expect("vit cache");
        let merged = graph
            .run(&[("patches", img.patches.as_slice())])
            .into_iter()
            .next()
            .context("moonvit merged")?;
        Ok(merged)
    }
}

#[derive(Clone)]
pub struct MoonVitWeights {
    pub patch_w: Vec<f32>,
    pub patch_b: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub pos_h: usize,
    pub pos_w: usize,
    pub layers: Vec<MoonVitLayerWeights>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub merge: [usize; 2],
}

#[derive(Clone)]
pub struct MoonVitLayerWeights {
    pub norm0_w: Vec<f32>,
    pub norm0_b: Vec<f32>,
    pub wqkv_w: Vec<f32>,
    pub wqkv_b: Vec<f32>,
    pub wo_w: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub mlp0_w: Vec<f32>,
    pub mlp0_b: Vec<f32>,
    pub mlp1_w: Vec<f32>,
    pub mlp1_b: Vec<f32>,
}

pub fn load_moonvit_weights(wm: &mut WeightMap, cfg: &MoonVitConfig) -> Result<MoonVitWeights> {
    let hidden = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let head_dim = hidden / heads;

    let (patch_w, patch_shape) = wm.take(LocateAnythingWeightPrefix::vision_patch_proj_w())?;
    let (patch_b, _) = wm.take(LocateAnythingWeightPrefix::vision_patch_proj_b())?;
    ensure!(
        patch_shape == [hidden, 3, cfg.patch_size, cfg.patch_size],
        "patch proj shape {:?}",
        patch_shape
    );

    let (pos_emb, pos_shape) = wm.take(LocateAnythingWeightPrefix::vision_pos_emb())?;
    let pos_h = cfg.init_pos_emb_height;
    let pos_w = cfg.init_pos_emb_width;
    ensure!(
        pos_shape == [pos_h, pos_w, hidden],
        "pos emb shape {:?}",
        pos_shape
    );

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let mut take = |s: &str| wm.take(&LocateAnythingWeightPrefix::vision_block(i, s));
        let (norm0_w, _) = take("norm0.weight")?;
        let (norm0_b, _) = take("norm0.bias")?;
        let (wqkv_w, wqkv_shape) = take("wqkv.weight")?;
        let (wqkv_b, _) = take("wqkv.bias")?;
        let (wo_w, wo_shape) = take("wo.weight")?;
        let (wo_b, _) = take("wo.bias")?;
        let (norm1_w, _) = take("norm1.weight")?;
        let (norm1_b, _) = take("norm1.bias")?;
        let (mlp0_w, mlp0_shape) = take("mlp.fc0.weight")?;
        let (mlp0_b, _) = take("mlp.fc0.bias")?;
        let (mlp1_w, mlp1_shape) = take("mlp.fc1.weight")?;
        let (mlp1_b, _) = take("mlp.fc1.bias")?;
        ensure!(wqkv_shape[0] == hidden * 3 && wo_shape == [hidden, hidden]);
        ensure!(
            mlp0_shape[0] == cfg.intermediate_size && mlp1_shape == [hidden, cfg.intermediate_size]
        );
        layers.push(MoonVitLayerWeights {
            norm0_w,
            norm0_b,
            wqkv_w,
            wqkv_b,
            wo_w,
            wo_b,
            norm1_w,
            norm1_b,
            mlp0_w,
            mlp0_b,
            mlp1_w,
            mlp1_b,
        });
    }

    let (final_ln_w, _) = wm.take(LocateAnythingWeightPrefix::vision_final_ln_w())?;
    let (final_ln_b, _) = wm.take(LocateAnythingWeightPrefix::vision_final_ln_b())?;

    Ok(MoonVitWeights {
        patch_w,
        patch_b,
        pos_emb,
        pos_h,
        pos_w,
        layers,
        final_ln_w,
        final_ln_b,
        hidden,
        heads,
        head_dim,
        mlp_dim: cfg.intermediate_size,
        merge: cfg.merge_kernel_size,
    })
}

/// CPU reference forward (parity / fallback).
pub fn encode_image(w: &MoonVitWeights, img: &PreprocessedImage) -> Result<Vec<f32>> {
    let seq = img.num_patches();
    let h = w.hidden;
    let ps = (img.patch_dim / 3).isqrt();
    ensure!(ps * ps * 3 == img.patch_dim, "patch_dim");

    let mut hidden = vec![0f32; seq * h];
    for p in 0..seq {
        let patch = &img.patches[p * img.patch_dim..(p + 1) * img.patch_dim];
        for out_c in 0..h {
            let mut acc = w.patch_b[out_c];
            for ic in 0..3 {
                for dy in 0..ps {
                    for dx in 0..ps {
                        let pw = w.patch_w[((out_c * 3 + ic) * ps + dy) * ps + dx];
                        let pv = patch[(ic * ps + dy) * ps + dx];
                        acc += pw * pv;
                    }
                }
            }
            hidden[p * h + out_c] = acc;
        }
    }

    let pos = interpolate_pos_emb(&w.pos_emb, w.pos_h, w.pos_w, img.grid_h, img.grid_w, h);
    for i in 0..seq * h {
        hidden[i] += pos[i];
    }

    let freqs = freqs_cis_for_grid(
        &MoonVitConfig {
            model_type: "moonvit".into(),
            hidden_size: h,
            intermediate_size: w.mlp_dim,
            num_attention_heads: w.heads,
            num_hidden_layers: w.layers.len(),
            patch_size: ps,
            merge_kernel_size: w.merge,
            init_pos_emb_height: w.pos_h,
            init_pos_emb_width: w.pos_w,
        },
        img.grid_h,
        img.grid_w,
        ROPE_THETA,
    );

    for layer in &w.layers {
        encoder_layer(
            &mut hidden,
            seq,
            h,
            w.heads,
            w.head_dim,
            w.mlp_dim,
            layer,
            &freqs,
        )?;
    }
    layer_norm(&mut hidden, seq, h, &w.final_ln_w, &w.final_ln_b);

    Ok(patch_merger(&hidden, img.grid_h, img.grid_w, h, w.merge))
}

pub fn interpolate_pos_emb(
    table: &[f32],
    th: usize,
    tw: usize,
    gh: usize,
    gw: usize,
    dim: usize,
) -> Vec<f32> {
    if gh == th && gw == tw {
        let mut out = vec![0f32; gh * gw * dim];
        for y in 0..gh {
            for x in 0..gw {
                let src = (y * tw + x) * dim;
                let dst = (y * gw + x) * dim;
                out[dst..dst + dim].copy_from_slice(&table[src..src + dim]);
            }
        }
        return out;
    }
    use image::imageops::FilterType;
    use image::{ImageBuffer, Luma};

    let mut out = vec![0f32; gh * gw * dim];
    for c in 0..dim {
        let plane: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::from_fn(tw as u32, th as u32, |x, y| {
                Luma([table[(y as usize * tw + x as usize) * dim + c]])
            });
        let resized = image::imageops::resize(&plane, gw as u32, gh as u32, FilterType::CatmullRom);
        for y in 0..gh {
            for x in 0..gw {
                out[(y * gw + x) * dim + c] = resized.get_pixel(x as u32, y as u32).0[0];
            }
        }
    }
    out
}

pub fn patch_merger(
    x: &[f32],
    grid_h: usize,
    grid_w: usize,
    dim: usize,
    merge: [usize; 2],
) -> Vec<f32> {
    let kh = merge[0];
    let kw = merge[1];
    let nh = grid_h / kh;
    let nw = grid_w / kw;
    let out_dim = dim * kh * kw;
    let mut out = vec![0f32; nh * nw * out_dim];
    for py in 0..nh {
        for px in 0..nw {
            for dy in 0..kh {
                for dx in 0..kw {
                    let sy = py * kh + dy;
                    let sx = px * kw + dx;
                    let src = (sy * grid_w + sx) * dim;
                    let dst_off = (py * nw + px) * out_dim + (dy * kw + dx) * dim;
                    out[dst_off..dst_off + dim].copy_from_slice(&x[src..src + dim]);
                }
            }
        }
    }
    out
}

fn layer_norm(x: &mut [f32], seq: usize, dim: usize, gamma: &[f32], beta: &[f32]) {
    for t in 0..seq {
        let base = t * dim;
        let mut mean = 0f32;
        for i in 0..dim {
            mean += x[base + i];
        }
        mean /= dim as f32;
        let mut var = 0f32;
        for i in 0..dim {
            let d = x[base + i] - mean;
            var += d * d;
        }
        var /= dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..dim {
            x[base + i] = (x[base + i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

fn linear(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: &[f32],
    in_dim: usize,
    out_dim: usize,
    w_row_major: bool,
) {
    for o in 0..out_dim {
        let mut acc = b.get(o).copied().unwrap_or(0.0);
        for i in 0..in_dim {
            let ww = if w_row_major {
                w[o * in_dim + i]
            } else {
                w[i * out_dim + o]
            };
            acc += ww * x[i];
        }
        y[o] = acc;
    }
}

fn gelu(x: f32) -> f32 {
    // PyTorch `F.gelu` (erf), matches RLX `Op::Gelu` / HF MoonViT MLP.
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
    0.5 * x * (1.0 + erf(x / SQRT_2))
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
        + 0.254_829_6)
        * t;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - y * (-x * x).exp())
}

fn encoder_layer(
    hidden: &mut [f32],
    seq: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    mlp_hidden: usize,
    w: &MoonVitLayerWeights,
    freqs: &[f32],
) -> Result<()> {
    let mut normed = vec![0f32; seq * dim];
    layer_norm_copy(&mut normed, hidden, seq, dim, &w.norm0_w, &w.norm0_b);

    let qkv_dim = dim * 3;
    let mut qkv = vec![0f32; seq * qkv_dim];
    for t in 0..seq {
        linear(
            &mut qkv[t * qkv_dim..(t + 1) * qkv_dim],
            &normed[t * dim..(t + 1) * dim],
            &w.wqkv_w,
            &w.wqkv_b,
            dim,
            qkv_dim,
            true,
        );
    }

    let mut q = vec![0f32; seq * heads * head_dim];
    let mut k = vec![0f32; seq * heads * head_dim];
    let mut v = vec![0f32; seq * heads * head_dim];
    for t in 0..seq {
        let base = t * qkv_dim;
        for h in 0..heads {
            let qh = t * heads * head_dim + h * head_dim;
            let off = h * head_dim;
            q[qh..qh + head_dim].copy_from_slice(&qkv[base + off..base + off + head_dim]);
            k[qh..qh + head_dim]
                .copy_from_slice(&qkv[base + dim + off..base + dim + off + head_dim]);
            v[qh..qh + head_dim]
                .copy_from_slice(&qkv[base + 2 * dim + off..base + 2 * dim + off + head_dim]);
        }
    }

    apply_rope_2d(&mut q, &mut k, freqs, seq, heads, head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; seq * dim];
    for ti in 0..seq {
        for h in 0..heads {
            let qh = ti * heads * head_dim + h * head_dim;
            let mut scores = vec![0f32; seq];
            for tj in 0..seq {
                let kh = tj * heads * head_dim + h * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q[qh + d] * k[kh + d];
                }
                scores[tj] = dot * scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut acc = 0f32;
                for (tj, &a) in scores.iter().enumerate() {
                    let vh = tj * heads * head_dim + h * head_dim;
                    acc += (a / sum) * v[vh + d];
                }
                attn_out[ti * dim + h * head_dim + d] = acc;
            }
        }
    }

    let mut proj = vec![0f32; seq * dim];
    for t in 0..seq {
        linear(
            &mut proj[t * dim..(t + 1) * dim],
            &attn_out[t * dim..(t + 1) * dim],
            &w.wo_w,
            &w.wo_b,
            dim,
            dim,
            true,
        );
    }

    for i in 0..seq * dim {
        hidden[i] += proj[i];
    }

    layer_norm_copy(&mut normed, hidden, seq, dim, &w.norm1_w, &w.norm1_b);
    let mut mlp_h = vec![0f32; seq * mlp_hidden];
    for t in 0..seq {
        linear(
            &mut mlp_h[t * mlp_hidden..(t + 1) * mlp_hidden],
            &normed[t * dim..(t + 1) * dim],
            &w.mlp0_w,
            &w.mlp0_b,
            dim,
            mlp_hidden,
            true,
        );
        for i in 0..mlp_hidden {
            mlp_h[t * mlp_hidden + i] = gelu(mlp_h[t * mlp_hidden + i]);
        }
        let mut delta = vec![0f32; dim];
        linear(
            &mut delta,
            &mlp_h[t * mlp_hidden..(t + 1) * mlp_hidden],
            &w.mlp1_w,
            &w.mlp1_b,
            mlp_hidden,
            dim,
            true,
        );
        for i in 0..dim {
            hidden[t * dim + i] += delta[i];
        }
    }
    Ok(())
}

fn layer_norm_copy(
    dst: &mut [f32],
    src: &[f32],
    seq: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
) {
    dst.copy_from_slice(src);
    layer_norm(dst, seq, dim, gamma, beta);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_finite() {
        assert!(gelu(0.0).is_finite());
    }
}
