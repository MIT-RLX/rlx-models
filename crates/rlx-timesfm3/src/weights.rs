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

//! Safetensors weight loading for TimesFM-3.

use crate::config::TimesFM3Config;
use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2};
use rlx_core::weight_map::WeightMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LinearWeight {
    pub w: Array2<f32>,
    pub b: Option<Array1<f32>>,
}

#[derive(Debug, Clone)]
pub struct RmsWeight {
    pub weight: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct MhaWeight {
    pub query: Array2<f32>,
    pub key: Array2<f32>,
    pub value: Array2<f32>,
    pub out: Array2<f32>,
    pub query_ln: Array1<f32>,
    pub key_ln: Array1<f32>,
    pub per_dim_scale: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct MixingLayerWeight {
    pub pre_seq_ln: Array1<f32>,
    pub post_seq_ln: Array1<f32>,
    pub seq_attn: MhaWeight,
    pub pre_var_ln: Option<Array1<f32>>,
    pub post_var_ln: Option<Array1<f32>>,
    pub var_attn: Option<MhaWeight>,
    pub pre_ff_ln: Array1<f32>,
    pub post_ff_ln: Array1<f32>,
    pub ff0: Array2<f32>,
    pub ff1: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ResidualBlockWeight {
    pub hidden: Array2<f32>,
    pub output: Array2<f32>,
    pub residual: Array2<f32>,
    pub pre_norm: Option<Array1<f32>>,
}

#[derive(Debug, Clone)]
pub struct TimesFM3Weights {
    pub resblock: ResidualBlockWeight,
    pub layers: Vec<MixingLayerWeight>,
    pub output_head: LinearWeight,
}

impl TimesFM3Weights {
    pub fn load(path: &Path, cfg: &TimesFM3Config) -> Result<Self> {
        let map = if path.is_dir() {
            WeightMap::from_safetensors_dir(path)?
        } else {
            WeightMap::from_file(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-UTF8 weights path"))?,
            )?
        };
        Self::from_map(map, cfg)
    }

    pub fn from_map(mut map: WeightMap, cfg: &TimesFM3Config) -> Result<Self> {
        let d = cfg.model_dims();
        let ff = cfg.transformer_config.transformer.hidden_dims;
        let hd = cfg.head_dim();

        let rb = &cfg.residual_block_config;
        let in_dim = cfg.resblock_input_dim();
        let resblock = ResidualBlockWeight {
            hidden: take_mat(
                &mut map,
                "pre_transformer_resblock.hidden_layer.weight",
                rb.hidden_dims,
                in_dim,
            )?,
            output: take_mat(
                &mut map,
                "pre_transformer_resblock.output_layer.weight",
                rb.output_dims,
                rb.hidden_dims,
            )?,
            residual: take_mat(
                &mut map,
                "pre_transformer_resblock.residual_layer.weight",
                rb.output_dims,
                in_dim,
            )?,
            pre_norm: if rb.prenorm == "rms" {
                Some(take_vec(
                    &mut map,
                    "pre_transformer_resblock.pre_norm.weight",
                    in_dim,
                )?)
            } else {
                None
            },
        };

        let mut layers = Vec::with_capacity(cfg.num_layers());
        for i in 0..cfg.num_layers() {
            let p = format!("transformer_stack.layers.{i}");
            let seq = MhaWeight {
                query: take_mat(&mut map, &format!("{p}.seq_attn.query_proj.weight"), d, d)?,
                key: take_mat(&mut map, &format!("{p}.seq_attn.key_proj.weight"), d, d)?,
                value: take_mat(&mut map, &format!("{p}.seq_attn.value_proj.weight"), d, d)?,
                out: take_mat(&mut map, &format!("{p}.seq_attn.out_proj.weight"), d, d)?,
                query_ln: take_vec(&mut map, &format!("{p}.seq_attn.query_ln.weight"), hd)?,
                key_ln: take_vec(&mut map, &format!("{p}.seq_attn.key_ln.weight"), hd)?,
                per_dim_scale: take_vec(
                    &mut map,
                    &format!("{p}.seq_attn.per_dim_scale.per_dim_scale"),
                    hd,
                )?,
            };
            let (pre_var_ln, post_var_ln, var_attn) = if cfg.use_variate_attention {
                let var = MhaWeight {
                    query: take_mat(&mut map, &format!("{p}.var_attn.query_proj.weight"), d, d)?,
                    key: take_mat(&mut map, &format!("{p}.var_attn.key_proj.weight"), d, d)?,
                    value: take_mat(&mut map, &format!("{p}.var_attn.value_proj.weight"), d, d)?,
                    out: take_mat(&mut map, &format!("{p}.var_attn.out_proj.weight"), d, d)?,
                    query_ln: take_vec(&mut map, &format!("{p}.var_attn.query_ln.weight"), hd)?,
                    key_ln: take_vec(&mut map, &format!("{p}.var_attn.key_ln.weight"), hd)?,
                    per_dim_scale: take_vec(
                        &mut map,
                        &format!("{p}.var_attn.per_dim_scale.per_dim_scale"),
                        hd,
                    )?,
                };
                (
                    Some(take_vec(
                        &mut map,
                        &format!("{p}.pre_var_attn_ln.weight"),
                        d,
                    )?),
                    Some(take_vec(
                        &mut map,
                        &format!("{p}.post_var_attn_ln.weight"),
                        d,
                    )?),
                    Some(var),
                )
            } else {
                (None, None, None)
            };
            layers.push(MixingLayerWeight {
                pre_seq_ln: take_vec(&mut map, &format!("{p}.pre_seq_attn_ln.weight"), d)?,
                post_seq_ln: take_vec(&mut map, &format!("{p}.post_seq_attn_ln.weight"), d)?,
                seq_attn: seq,
                pre_var_ln,
                post_var_ln,
                var_attn,
                pre_ff_ln: take_vec(&mut map, &format!("{p}.pre_ff_ln.weight"), d)?,
                post_ff_ln: take_vec(&mut map, &format!("{p}.post_ff_ln.weight"), d)?,
                ff0: take_mat(&mut map, &format!("{p}.ff0.weight"), ff, d)?,
                ff1: take_mat(&mut map, &format!("{p}.ff1.weight"), d, ff)?,
            });
        }

        let out_dim = cfg.output_patch_len * cfg.num_quantiles();
        let output_head = LinearWeight {
            w: take_mat(&mut map, "output_head.weight", out_dim, d)?,
            b: Some(take_vec(&mut map, "output_head.bias", out_dim)?),
        };

        Ok(Self {
            resblock,
            layers,
            output_head,
        })
    }

    /// Random tiny weights for unit tests.
    pub fn synth(cfg: &TimesFM3Config, seed: u64) -> Self {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> f32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.02
            }
            fn mat(&mut self, rows: usize, cols: usize) -> Array2<f32> {
                Array2::from_shape_fn((rows, cols), |_| self.next())
            }
            fn vec(&mut self, n: usize) -> Array1<f32> {
                Array1::from_shape_fn(n, |_| 1.0 + self.next())
            }
            fn mha(&mut self, d: usize, hd: usize) -> MhaWeight {
                MhaWeight {
                    query: self.mat(d, d),
                    key: self.mat(d, d),
                    value: self.mat(d, d),
                    out: self.mat(d, d),
                    query_ln: self.vec(hd),
                    key_ln: self.vec(hd),
                    per_dim_scale: Array1::zeros(hd),
                }
            }
        }

        let mut rng = Rng(seed);
        let d = cfg.model_dims();
        let ff = cfg.transformer_config.transformer.hidden_dims;
        let hd = cfg.head_dim();
        let in_dim = cfg.resblock_input_dim();
        let rb = &cfg.residual_block_config;

        let resblock = ResidualBlockWeight {
            hidden: rng.mat(rb.hidden_dims, in_dim),
            output: rng.mat(rb.output_dims, rb.hidden_dims),
            residual: rng.mat(rb.output_dims, in_dim),
            pre_norm: None,
        };

        let mut layers = Vec::new();
        for _ in 0..cfg.num_layers() {
            layers.push(MixingLayerWeight {
                pre_seq_ln: rng.vec(d),
                post_seq_ln: rng.vec(d),
                seq_attn: rng.mha(d, hd),
                pre_var_ln: Some(rng.vec(d)),
                post_var_ln: Some(rng.vec(d)),
                var_attn: Some(rng.mha(d, hd)),
                pre_ff_ln: rng.vec(d),
                post_ff_ln: rng.vec(d),
                ff0: rng.mat(ff, d),
                ff1: rng.mat(d, ff),
            });
        }

        let out_dim = cfg.output_patch_len * cfg.num_quantiles();
        TimesFM3Weights {
            resblock,
            layers,
            output_head: LinearWeight {
                w: rng.mat(out_dim, d),
                b: Some(rng.vec(out_dim)),
            },
        }
    }
}

fn take_mat(map: &mut WeightMap, key: &str, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let (data, shape) = map
        .take(key)
        .with_context(|| format!("missing weight {key}"))?;
    if shape.len() != 2 || shape[0] != rows || shape[1] != cols {
        bail!("{key}: expected [{rows}, {cols}], got {shape:?}");
    }
    Ok(Array2::from_shape_vec((rows, cols), data)?)
}

fn take_vec(map: &mut WeightMap, key: &str, n: usize) -> Result<Array1<f32>> {
    let (data, shape) = map
        .take(key)
        .with_context(|| format!("missing weight {key}"))?;
    if shape.len() != 1 || shape[0] != n {
        bail!("{key}: expected [{n}], got {shape:?}");
    }
    Ok(Array1::from_vec(data))
}
