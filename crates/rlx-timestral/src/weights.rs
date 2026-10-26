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

//! `model.safetensors` loading under the upstream parameter names.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2};
use rlx_core::weight_map::WeightMap;

use crate::config::TimestralConfig;

/// `output(silu(hidden(x))) + residual(x)`; used by the patch embedding and the head.
#[derive(Debug, Clone)]
pub struct ResidualBlockWeight {
    pub hidden: Array2<f32>,
    pub hidden_b: Option<Array1<f32>>,
    pub output: Array2<f32>,
    pub output_b: Option<Array1<f32>>,
    pub residual: Array2<f32>,
    pub residual_b: Option<Array1<f32>>,
}

/// One attention sublayer.
///
/// Upstream packs `wQKV` rows as `(qkv, head_dim, n_heads)`; [`AttentionWeight::qkv`]
/// is reordered at load time to the usual `(qkv, n_heads, head_dim)`, the order the
/// attention output and the gate already use.
#[derive(Debug, Clone)]
pub struct AttentionWeight {
    pub is_time: bool,
    /// `[3 * d_model, d_model]`, rows `(qkv, head, dim)`.
    pub qkv: Array2<f32>,
    pub gate: Option<Array2<f32>>,
    pub out: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerWeight {
    pub attn_norms: Vec<Array1<f32>>,
    pub attns: Vec<AttentionWeight>,
    pub mlp_norm: Array1<f32>,
    /// First half of the fused `w12` (the multiplier).
    pub w_gate: Array2<f32>,
    /// Second half of `w12` (goes through SiLU).
    pub w_up: Array2<f32>,
    pub w3: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct TimestralWeights {
    pub patch_proj: ResidualBlockWeight,
    pub forecast_query: Array1<f32>,
    pub layers: Vec<LayerWeight>,
    pub out_norm: Array1<f32>,
    pub head: ResidualBlockWeight,
}

impl TimestralWeights {
    /// Load `model.safetensors` (a file, or a directory holding it).
    pub fn load(path: &Path, cfg: &TimestralConfig) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        let map = WeightMap::from_file(
            file.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 weights path"))?,
        )
        .with_context(|| format!("load {}", file.display()))?;
        Self::from_map(map, cfg)
    }

    pub fn from_map(mut map: WeightMap, cfg: &TimestralConfig) -> Result<Self> {
        let d = cfg.d_model;
        let ps = cfg.patch_size;
        let h = cfg.mlp_hidden_dim;
        let patch_proj = ResidualBlockWeight {
            hidden: take_mat(
                &mut map,
                "representation.patch_proj.hidden_layer.weight",
                4 * d,
                2 * ps,
            )?,
            hidden_b: Some(take_vec(
                &mut map,
                "representation.patch_proj.hidden_layer.bias",
                4 * d,
            )?),
            output: take_mat(
                &mut map,
                "representation.patch_proj.output_layer.weight",
                d,
                4 * d,
            )?,
            output_b: Some(take_vec(
                &mut map,
                "representation.patch_proj.output_layer.bias",
                d,
            )?),
            residual: take_mat(
                &mut map,
                "representation.patch_proj.residual_layer.weight",
                d,
                2 * ps,
            )?,
            residual_b: Some(take_vec(
                &mut map,
                "representation.patch_proj.residual_layer.bias",
                d,
            )?),
        };
        let forecast_query = take_vec(&mut map, "representation.forecast_query", d)?;

        let mut layers = Vec::with_capacity(cfg.num_layers());
        for (i, kind) in cfg.block_schedule.iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            let mut attn_norms = Vec::new();
            let mut attns = Vec::new();
            for (j, &is_time) in kind.axes().iter().enumerate() {
                attn_norms.push(take_vec(&mut map, &format!("{p}.attn_norms.{j}.scale"), d)?);
                let qkv = take_mat(&mut map, &format!("{p}.attns.{j}.wQKV.weight"), 3 * d, d)?;
                let gate = if cfg.gated(is_time) {
                    Some(take_mat(
                        &mut map,
                        &format!("{p}.attns.{j}._gate.weight"),
                        d,
                        d,
                    )?)
                } else {
                    None
                };
                attns.push(AttentionWeight {
                    is_time,
                    qkv: heads_major_qkv(&qkv, cfg.num_heads, cfg.head_dim()),
                    gate,
                    out: take_mat(&mut map, &format!("{p}.attns.{j}.wO.weight"), d, d)?,
                });
            }
            let w12 = take_mat(&mut map, &format!("{p}.mlp.w12.weight"), 2 * h, d)?;
            layers.push(LayerWeight {
                attn_norms,
                attns,
                mlp_norm: take_vec(&mut map, &format!("{p}.mlp_norm.scale"), d)?,
                w_gate: w12.slice(ndarray::s![..h, ..]).to_owned(),
                w_up: w12.slice(ndarray::s![h.., ..]).to_owned(),
                w3: take_mat(&mut map, &format!("{p}.mlp.w3.weight"), d, h)?,
            });
        }
        let out_norm = take_vec(&mut map, "transformer.out_norm.scale", d)?;

        let hq = ps * cfg.num_quantiles();
        let head = ResidualBlockWeight {
            hidden: take_mat(&mut map, "head.projection.hidden_layer.weight", 4 * d, d)?,
            hidden_b: None,
            output: take_mat(&mut map, "head.projection.output_layer.weight", hq, 4 * d)?,
            output_b: None,
            residual: take_mat(&mut map, "head.projection.residual_layer.weight", hq, d)?,
            residual_b: None,
        };

        if !map.is_empty() {
            let mut left: Vec<&str> = map.keys().collect();
            left.sort_unstable();
            bail!("unexpected weights in checkpoint: {left:?}");
        }
        Ok(Self {
            patch_proj,
            forecast_query,
            layers,
            out_norm,
            head,
        })
    }

    /// Small seeded random weights for tests that need no download.
    pub fn synth(cfg: &TimestralConfig, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let d = cfg.d_model;
        let ps = cfg.patch_size;
        let h = cfg.mlp_hidden_dim;
        let hq = ps * cfg.num_quantiles();
        let res =
            |rng: &mut Rng, inp: usize, hid: usize, out: usize, bias: bool| ResidualBlockWeight {
                hidden: rng.mat(hid, inp, inp),
                hidden_b: bias.then(|| rng.vec(hid, 0.0)),
                output: rng.mat(out, hid, hid),
                output_b: bias.then(|| rng.vec(out, 0.0)),
                residual: rng.mat(out, inp, inp),
                residual_b: bias.then(|| rng.vec(out, 0.0)),
            };
        let patch_proj = res(&mut rng, 2 * ps, 4 * d, d, true);
        let forecast_query = rng.vec(d, 0.0);
        let layers = cfg
            .block_schedule
            .iter()
            .map(|kind| LayerWeight {
                attn_norms: kind.axes().iter().map(|_| rng.vec(d, 1.0)).collect(),
                attns: kind
                    .axes()
                    .iter()
                    .map(|&is_time| AttentionWeight {
                        is_time,
                        qkv: rng.mat(3 * d, d, d),
                        gate: cfg.gated(is_time).then(|| rng.mat(d, d, d)),
                        out: rng.mat(d, d, d),
                    })
                    .collect(),
                mlp_norm: rng.vec(d, 1.0),
                w_gate: rng.mat(h, d, d),
                w_up: rng.mat(h, d, d),
                w3: rng.mat(d, h, h),
            })
            .collect();
        let out_norm = rng.vec(d, 1.0);
        let head = res(&mut rng, d, 4 * d, hq, false);
        Self {
            patch_proj,
            forecast_query,
            layers,
            out_norm,
            head,
        }
    }
}

/// Reorder `wQKV` rows from upstream `(qkv, head_dim, n_heads)` to `(qkv, n_heads, head_dim)`.
fn heads_major_qkv(w: &Array2<f32>, nh: usize, hd: usize) -> Array2<f32> {
    let d = nh * hd;
    let mut out = Array2::<f32>::zeros(w.raw_dim());
    for part in 0..3 {
        for h in 0..nh {
            for k in 0..hd {
                out.row_mut(part * d + h * hd + k)
                    .assign(&w.row(part * d + k * nh + h));
            }
        }
    }
    out
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// `[rows, cols]` with a fan-in scaled range so activations stay O(1).
    fn mat(&mut self, rows: usize, cols: usize, fan_in: usize) -> Array2<f32> {
        let s = 1.0 / (fan_in as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| self.next() * s)
    }

    fn vec(&mut self, n: usize, center: f32) -> Array1<f32> {
        Array1::from_shape_fn(n, |_| center + 0.1 * self.next())
    }
}

fn take_mat(map: &mut WeightMap, key: &str, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let (data, shape) = map
        .take(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure!(
        shape == [rows, cols],
        "{key}: expected [{rows}, {cols}], got {shape:?}"
    );
    Ok(Array2::from_shape_vec((rows, cols), data)?)
}

fn take_vec(map: &mut WeightMap, key: &str, n: usize) -> Result<Array1<f32>> {
    let (data, shape) = map
        .take(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure!(shape == [n], "{key}: expected [{n}], got {shape:?}");
    Ok(Array1::from_vec(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_reorder_moves_head_to_the_outer_axis() {
        let (nh, hd) = (2, 3);
        let d = nh * hd;
        // Row index encodes its upstream (part, k, h) position.
        let w = Array2::from_shape_fn((3 * d, 1), |(r, _)| r as f32);
        let out = heads_major_qkv(&w, nh, hd);
        // New row (part=1, h=1, k=2) came from upstream row part*d + k*nh + h.
        assert_eq!(out[[d + hd + 2, 0]], (d + 2 * nh + 1) as f32);
    }
}
