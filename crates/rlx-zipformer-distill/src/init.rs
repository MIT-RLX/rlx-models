// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Start the student from the teacher: each student tensor is the teacher's
//! tensor of the same name (layer `l` of stack `s` ← the teacher's layer `l`
//! of stack `s`), cut to the student's shape — leading channels, leading
//! heads within each segment of a fused projection, both halves of a GLU.
//! With the teacher's own config this reproduces the teacher exactly, which
//! is how the graph is checked against `rlx-kroko-core`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::weight_map::WeightMap;

use crate::config::{POS_HEAD, StudentConfig};

/// A converted teacher directory (`config.json` + `model.safetensors`).
pub struct Teacher {
    /// Its shape.
    pub cfg: StudentConfig,
    /// Its tensors.
    pub weights: WeightMap,
    /// `vocab_size`, `context_size` from its config.
    pub vocab: usize,
    /// Predictor context.
    pub context: usize,
}

impl Teacher {
    /// Open a directory written by `scripts/sherpa_zipformer_to_kroko.py` (or
    /// audio.cpp's converter).
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let (_, weights) = rlx_core::weights::open_map(dir.join("model.safetensors"))
            .with_context(|| format!("open {}", dir.display()))?;
        let ints = |k: &str| -> Result<Vec<usize>> {
            cfg[k]
                .as_array()
                .ok_or_else(|| anyhow!("config.json: {k}"))?
                .iter()
                .map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("{k}")))
                .collect()
        };
        let dims = ints("encoder_dims")?;
        let ff = (0..dims.len())
            .map(|s| {
                let p = StudentConfig::layer_prefix(s, 0);
                let w = |i: usize| {
                    weights
                        .get(&format!("{p}feed_forward{i}.in_proj.bias"))
                        .map_or(0, |(d, _)| d.len())
                };
                [w(1), w(2), w(3)]
            })
            .collect();
        let tcfg = StudentConfig {
            dims,
            layers: ints("num_encoder_layers")?,
            downs: ints("downsampling_factors")?,
            heads: ints("num_heads")?,
            kernels: ints("cnn_module_kernels")?,
            ff,
            qdim: ints("query_head_dims")?[0],
            vdim: ints("value_head_dims")?[0],
            qat_bits: None,
        };
        let one = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("{k}"))
        };
        Ok(Self {
            cfg: tcfg,
            vocab: one("vocab_size")?,
            context: one("context_size")?,
            weights,
        })
    }

    fn get(&self, name: &str) -> Result<(&[f32], &[usize])> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("teacher has no {name}"))
    }

    /// The student's initial value for `name` of shape `dims`.
    pub fn init(&self, student: &StudentConfig, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let (data, tdims) = self.get(name)?;
        let n: usize = dims.iter().product();
        if tdims == dims {
            return Ok(data.to_vec());
        }
        ensure!(
            tdims.len() == dims.len(),
            "{name}: teacher rank {tdims:?} vs student {dims:?}"
        );
        // Which teacher index each student row (axis 0) takes.
        let st = name
            .split("encoders.")
            .nth(1)
            .and_then(|r| r.split('.').next())
            .and_then(|v| v.parse::<usize>().ok());
        let rows: Vec<usize> = match st {
            Some(st) => {
                let (ht, hs) = (self.cfg.heads[st], student.heads[st]);
                let (dt, ds) = (self.cfg.dims[st], student.dims[st]);
                let (q, v) = (student.qdim, student.vdim);
                let seg = |segs: &[(usize, usize)]| -> Vec<usize> {
                    segs.iter().flat_map(|&(off, len)| off..off + len).collect()
                };
                if name.contains("self_attn_weights.in_proj") {
                    let mut idx = Vec::new();
                    for part in 0..2 {
                        for hh in 0..hs {
                            idx.extend((0..q).map(|i| part * ht * q + hh * q + i));
                        }
                    }
                    for hh in 0..hs {
                        idx.extend((0..POS_HEAD).map(|i| 2 * ht * q + hh * POS_HEAD + i));
                    }
                    idx
                } else if name.contains("linear_pos") {
                    seg(&[(0, hs * POS_HEAD)])
                } else if name.contains("nonlin_attention.in_proj") {
                    let (nt, ns) = (3 * dt / 4, 3 * ds / 4);
                    seg(&[(0, ns), (nt, ns), (2 * nt, ns)])
                } else if name.contains("conv_module") && name.contains(".in_proj") {
                    seg(&[(0, ds), (dt, ds)])
                } else if name.contains("self_attn1.in_proj") || name.contains("self_attn2.in_proj")
                {
                    (0..hs)
                        .flat_map(|hh| (0..v).map(move |i| hh * v + i))
                        .collect()
                } else {
                    (0..dims[0]).collect()
                }
            }
            None => (0..dims[0]).collect(),
        };
        ensure!(
            rows.len() == dims[0],
            "{name}: {} rows vs {dims:?}",
            rows.len()
        );
        // Columns: leading entries per remaining axis (the input side of a
        // linear is a stack's channels or a hidden width, both leading).
        let t_inner: usize = tdims[1..].iter().product();
        let mut out = Vec::with_capacity(n);
        for &rr in &rows {
            let base = rr * t_inner;
            // Copy the leading block of the trailing axes.
            push_leading(
                &mut out,
                &data[base..base + t_inner],
                &tdims[1..],
                &dims[1..],
            );
        }
        ensure!(out.len() == n, "{name}: built {} of {n}", out.len());
        Ok(out)
    }
}

/// Append the leading `want` sub-block of a row-major block shaped `have`.
fn push_leading(out: &mut Vec<f32>, block: &[f32], have: &[usize], want: &[usize]) {
    if have.is_empty() {
        out.push(block[0]);
        return;
    }
    let inner: usize = have[1..].iter().product();
    for i in 0..want[0] {
        push_leading(
            out,
            &block[i * inner..(i + 1) * inner],
            &have[1..],
            &want[1..],
        );
    }
}
