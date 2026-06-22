// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! `SimpleMLPAdaLN` — per-step flow head used by Pocket TTS. Operates on a
//! single `[B, ldim]` latent at a time conditioned on the backbone output `c`
//! plus two time scalars `(s, t)`.
//!
//! Closely follows `pocket_tts/modules/mlp.py`.

use anyhow::Result;
use ndarray::{Array1, Array2};

use crate::config::FlowLmConfig;
use crate::ops::{layernorm, linear, rmsnorm, silu_inplace};
use crate::weights::WeightFile;

#[derive(Debug, Clone)]
struct TimeEmbedder {
    freqs: Array1<f32>,  // [half]
    mlp0_w: Array2<f32>, // [model, 2*half]
    mlp0_b: Array1<f32>,
    mlp2_w: Array2<f32>, // [model, model]
    mlp2_b: Array1<f32>,
    rms_alpha: Array1<f32>,
}

#[derive(Debug, Clone)]
struct ResBlock {
    in_ln_w: Array1<f32>,
    in_ln_b: Array1<f32>,
    mlp0_w: Array2<f32>,
    mlp0_b: Array1<f32>,
    mlp2_w: Array2<f32>,
    mlp2_b: Array1<f32>,
    adaln_w: Array2<f32>,
    adaln_b: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct FlowMlp {
    cond_embed_w: Array2<f32>,
    cond_embed_b: Array1<f32>,
    input_proj_w: Array2<f32>,
    input_proj_b: Array1<f32>,
    time_embed: Vec<TimeEmbedder>,
    res_blocks: Vec<ResBlock>,
    final_adaln_w: Array2<f32>,
    final_adaln_b: Array1<f32>,
    final_linear_w: Array2<f32>,
    final_linear_b: Array1<f32>,
    model_channels: usize,
    out_channels: usize,
    eps: f32,
}

impl FlowMlp {
    pub fn load(wf: &WeightFile, prefix: &str, cfg: &FlowLmConfig) -> Result<Self> {
        let cond_embed_w = wf.get_2d(&format!("{prefix}.cond_embed.weight"))?;
        let cond_embed_b = wf.get_1d(&format!("{prefix}.cond_embed.bias"))?;
        let input_proj_w = wf.get_2d(&format!("{prefix}.input_proj.weight"))?;
        let input_proj_b = wf.get_1d(&format!("{prefix}.input_proj.bias"))?;

        let mut time_embed = Vec::with_capacity(2);
        for i in 0..2 {
            time_embed.push(TimeEmbedder {
                freqs: wf.get_1d(&format!("{prefix}.time_embed.{i}.freqs"))?,
                mlp0_w: wf.get_2d(&format!("{prefix}.time_embed.{i}.mlp.0.weight"))?,
                mlp0_b: wf.get_1d(&format!("{prefix}.time_embed.{i}.mlp.0.bias"))?,
                mlp2_w: wf.get_2d(&format!("{prefix}.time_embed.{i}.mlp.2.weight"))?,
                mlp2_b: wf.get_1d(&format!("{prefix}.time_embed.{i}.mlp.2.bias"))?,
                rms_alpha: wf.get_1d(&format!("{prefix}.time_embed.{i}.mlp.3.alpha"))?,
            });
        }

        let mut res_blocks = Vec::with_capacity(cfg.flow_blocks);
        for i in 0..cfg.flow_blocks {
            let bp = format!("{prefix}.res_blocks.{i}");
            res_blocks.push(ResBlock {
                in_ln_w: wf.get_1d(&format!("{bp}.in_ln.weight"))?,
                in_ln_b: wf.get_1d(&format!("{bp}.in_ln.bias"))?,
                mlp0_w: wf.get_2d(&format!("{bp}.mlp.0.weight"))?,
                mlp0_b: wf.get_1d(&format!("{bp}.mlp.0.bias"))?,
                mlp2_w: wf.get_2d(&format!("{bp}.mlp.2.weight"))?,
                mlp2_b: wf.get_1d(&format!("{bp}.mlp.2.bias"))?,
                adaln_w: wf.get_2d(&format!("{bp}.adaLN_modulation.1.weight"))?,
                adaln_b: wf.get_1d(&format!("{bp}.adaLN_modulation.1.bias"))?,
            });
        }

        let final_adaln_w =
            wf.get_2d(&format!("{prefix}.final_layer.adaLN_modulation.1.weight"))?;
        let final_adaln_b = wf.get_1d(&format!("{prefix}.final_layer.adaLN_modulation.1.bias"))?;
        let final_linear_w = wf.get_2d(&format!("{prefix}.final_layer.linear.weight"))?;
        let final_linear_b = wf.get_1d(&format!("{prefix}.final_layer.linear.bias"))?;

        Ok(Self {
            cond_embed_w,
            cond_embed_b,
            input_proj_w,
            input_proj_b,
            time_embed,
            res_blocks,
            final_adaln_w,
            final_adaln_b,
            final_linear_w,
            final_linear_b,
            model_channels: cfg.flow_dim,
            out_channels: cfg.latent_dim,
            eps: 1e-6,
        })
    }

    /// One step of the flow net: returns `u_t(c, s, t, x)`.
    /// - `c`: `[1, cond_dim]` backbone conditioning for this step
    /// - `s`, `t`: scalar time conditions
    /// - `x`: `[1, ldim]` current latent (Euler state)
    ///
    /// Returns `[1, ldim]`.
    pub fn forward(&self, c: &Array2<f32>, s: f32, t: f32, x: &Array2<f32>) -> Array2<f32> {
        let n = x.shape()[0];
        debug_assert_eq!(c.shape()[0], n);

        // input_proj
        let mut h = linear(
            x.view(),
            self.input_proj_w.view(),
            Some(self.input_proj_b.view()),
        );

        // Combine time conditions: t_combined = (te_s + te_t) / 2
        let mut t_emb_s = self.time_embed_forward(0, s, n);
        let t_emb_t = self.time_embed_forward(1, t, n);
        for i in 0..n {
            for j in 0..self.model_channels {
                t_emb_s[[i, j]] = (t_emb_s[[i, j]] + t_emb_t[[i, j]]) * 0.5;
            }
        }

        // cond_embed
        let c_emb = linear(
            c.view(),
            self.cond_embed_w.view(),
            Some(self.cond_embed_b.view()),
        );
        // y = t_combined + c_emb
        let mut y = t_emb_s;
        for i in 0..n {
            for j in 0..self.model_channels {
                y[[i, j]] += c_emb[[i, j]];
            }
        }

        // SiLU(y) — adaLN_modulation Sequential starts with SiLU.
        let mut y_silu = y.clone();
        silu_inplace(y_silu.as_slice_mut().unwrap());

        // Res blocks.
        for block in &self.res_blocks {
            // shift, scale, gate = adaLN(y).chunk(3, -1)
            let mod_out = linear(
                y_silu.view(),
                block.adaln_w.view(),
                Some(block.adaln_b.view()),
            );
            let chunks = self.model_channels;
            // in_ln(x) → modulate(shift, scale) → mlp → add gate-weighted update.
            let normed = layernorm(
                h.view(),
                Some(block.in_ln_w.view()),
                Some(block.in_ln_b.view()),
                self.eps,
            );
            let mut modulated = normed.clone();
            for i in 0..n {
                for j in 0..chunks {
                    let shift = mod_out[[i, j]];
                    let scale = mod_out[[i, chunks + j]];
                    modulated[[i, j]] = modulated[[i, j]] * (1.0 + scale) + shift;
                }
            }
            let mut mid = linear(
                modulated.view(),
                block.mlp0_w.view(),
                Some(block.mlp0_b.view()),
            );
            silu_inplace(mid.as_slice_mut().unwrap());
            let update = linear(mid.view(), block.mlp2_w.view(), Some(block.mlp2_b.view()));
            for i in 0..n {
                for j in 0..chunks {
                    let gate = mod_out[[i, 2 * chunks + j]];
                    h[[i, j]] += gate * update[[i, j]];
                }
            }
        }

        // Final layer: modulate(norm_final(h), shift, scale) with adaLN(2 * model_channels),
        // norm_final has no affine params (elementwise_affine=False).
        let final_mod = linear(
            y_silu.view(),
            self.final_adaln_w.view(),
            Some(self.final_adaln_b.view()),
        );
        let chunks = self.model_channels;
        let mut normed = layernorm(h.view(), None, None, self.eps);
        for i in 0..n {
            for j in 0..chunks {
                let shift = final_mod[[i, j]];
                let scale = final_mod[[i, chunks + j]];
                normed[[i, j]] = normed[[i, j]] * (1.0 + scale) + shift;
            }
        }
        linear(
            normed.view(),
            self.final_linear_w.view(),
            Some(self.final_linear_b.view()),
        )
    }

    fn time_embed_forward(&self, idx: usize, scalar: f32, n: usize) -> Array2<f32> {
        let te = &self.time_embed[idx];
        let half = te.freqs.len();
        // args = scalar * freqs; embedding = [cos(args), sin(args)] over channels dim.
        let mut emb = Array2::<f32>::zeros((n, 2 * half));
        for i in 0..n {
            for k in 0..half {
                let arg = scalar * te.freqs[k];
                emb[[i, k]] = arg.cos();
                emb[[i, half + k]] = arg.sin();
            }
        }
        // mlp.0: linear → SiLU; mlp.2: linear; mlp.3: RMSNorm.
        let mut h = linear(emb.view(), te.mlp0_w.view(), Some(te.mlp0_b.view()));
        silu_inplace(h.as_slice_mut().unwrap());
        let h = linear(h.view(), te.mlp2_w.view(), Some(te.mlp2_b.view()));
        rmsnorm(h.view(), te.rms_alpha.view(), 1e-5)
    }

    pub fn latent_dim(&self) -> usize {
        self.out_channels
    }
}
