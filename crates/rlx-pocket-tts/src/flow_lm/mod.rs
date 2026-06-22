// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! FlowLM — the autoregressive backbone + flow head.
//!
//! Composition (per pocket_tts/models/flow_lm.py):
//! - `conditioner.embed`: `nn.Embedding(n_bins+1, d_model)` — text token LUT
//! - `input_linear`: `Linear(ldim, d_model, bias=False)` — latent → backbone dim
//! - `transformer`: 6-layer streaming transformer
//! - `out_norm`: `LayerNorm(d_model)`
//! - `out_eos`: `Linear(d_model, 1)` — EOS logit
//! - `flow_net`: `SimpleMLPAdaLN` — per-step Euler flow head
//! - Buffers: `emb_mean`, `emb_std` (latent z-score stats), `bos_emb` (learnable)
//! - Optional: `bos_before_voice` `Parameter[1,1,d_model]` (insert_bos_before_voice).
//! - Optional: `speaker_proj_weight` `[d_model, mimi.inner_dim or seanet.dim]`
//!   (only used for voice cloning from raw audio).

pub mod mlp;
pub mod transformer;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::config::PocketTtsConfig;
use crate::ops::linear;
use crate::weights::WeightFile;

pub use mlp::FlowMlp;
pub use transformer::{KvCache, StreamingTransformer};

pub struct FlowLm {
    pub cfg: PocketTtsConfig,

    // Conditioner (text token embedding).
    embed: Array2<f32>, // [n_bins + 1, d_model]

    input_linear_w: Array2<f32>, // [d_model, ldim]
    out_norm_w: Array1<f32>,
    out_norm_b: Array1<f32>,
    out_eos_w: Array2<f32>, // [1, d_model]
    out_eos_b: Array1<f32>, // [1]

    emb_mean: Array1<f32>, // [ldim]
    emb_std: Array1<f32>,  // [ldim]
    bos_emb: Array1<f32>,  // [ldim]

    bos_before_voice: Option<Array2<f32>>, // [1, d_model] if `insert_bos_before_voice`
    pub speaker_proj_w: Option<Array2<f32>>, // [d_model, in_dim]

    pub transformer: StreamingTransformer,
    pub flow_net: FlowMlp,
}

impl FlowLm {
    pub fn load(wf: &WeightFile, cfg: PocketTtsConfig) -> Result<Self> {
        let prefix = "flow_lm";
        let embed = wf
            .get_2d(&format!("{prefix}.conditioner.embed.weight"))
            .context("flow_lm.conditioner.embed.weight")?;
        let input_linear_w = wf.get_2d(&format!("{prefix}.input_linear.weight"))?;
        let out_norm_w = wf.get_1d(&format!("{prefix}.out_norm.weight"))?;
        let out_norm_b = wf.get_1d(&format!("{prefix}.out_norm.bias"))?;
        let out_eos_w = wf.get_2d(&format!("{prefix}.out_eos.weight"))?;
        let out_eos_b = wf.get_1d(&format!("{prefix}.out_eos.bias"))?;

        let emb_mean = wf.get_1d(&format!("{prefix}.emb_mean"))?;
        let emb_std = wf.get_1d(&format!("{prefix}.emb_std"))?;
        let bos_emb = wf.get_1d(&format!("{prefix}.bos_emb"))?;

        let bos_before_voice = wf
            .tensor(&format!("{prefix}.bos_before_voice"))
            .ok()
            .and_then(|_| {
                // Shape [1, 1, d_model] in PyTorch; collapse to [1, d_model].
                let raw = wf.get_dyn(&format!("{prefix}.bos_before_voice")).ok()?;
                let total = raw.shape().iter().product::<usize>();
                let d = cfg.flow_lm.transformer.d_model;
                if total != d {
                    return None;
                }
                Array2::from_shape_vec((1, d), raw.into_raw_vec_and_offset().0).ok()
            });

        let speaker_proj_w = wf.get_2d(&format!("{prefix}.speaker_proj_weight")).ok();

        let transformer = StreamingTransformer::load(
            wf,
            &format!("{prefix}.transformer"),
            &cfg.flow_lm.transformer,
            cfg.flow_lm.norm_eps,
        )?;
        let flow_net = FlowMlp::load(wf, &format!("{prefix}.flow_net"), &cfg.flow_lm)?;

        Ok(Self {
            cfg,
            embed,
            input_linear_w,
            out_norm_w,
            out_norm_b,
            out_eos_w,
            out_eos_b,
            emb_mean,
            emb_std,
            bos_emb,
            bos_before_voice,
            speaker_proj_w,
            transformer,
            flow_net,
        })
    }

    /// Look up token embeddings: `tokens: [T]` (u32 ids) → `[T, d_model]`.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Array2<f32> {
        let (_, d) = self.embed.dim();
        let mut out = Array2::<f32>::zeros((tokens.len(), d));
        for (i, &tok) in tokens.iter().enumerate() {
            let row = self.embed.row(tok as usize);
            for j in 0..d {
                out[[i, j]] = row[j];
            }
        }
        out
    }

    /// Project a latent through `input_linear` (no bias). `latent: [T, ldim]` →
    /// `[T, d_model]`. NaN values are replaced with `bos_emb`.
    pub fn project_latent(&self, latent: &Array2<f32>) -> Array2<f32> {
        let (t, ldim) = latent.dim();
        let mut clean = Array2::<f32>::zeros((t, ldim));
        for i in 0..t {
            for j in 0..ldim {
                let v = latent[[i, j]];
                clean[[i, j]] = if v.is_nan() { self.bos_emb[j] } else { v };
            }
        }
        linear(clean.view(), self.input_linear_w.view(), None)
    }

    pub fn out_norm(&self, x: Array2<f32>) -> Array2<f32> {
        crate::ops::layernorm(
            x.view(),
            Some(self.out_norm_w.view()),
            Some(self.out_norm_b.view()),
            self.cfg.flow_lm.norm_eps,
        )
    }

    /// EOS head — returns the scalar logit for each row of `x: [T, d_model]`.
    pub fn eos_logit(&self, x_last: &Array2<f32>) -> f32 {
        let out = linear(
            x_last.view(),
            self.out_eos_w.view(),
            Some(self.out_eos_b.view()),
        );
        out[[0, 0]]
    }

    pub fn bos_before_voice(&self) -> Option<&Array2<f32>> {
        self.bos_before_voice.as_ref()
    }

    pub fn d_model(&self) -> usize {
        self.cfg.flow_lm.transformer.d_model
    }

    pub fn ldim(&self) -> usize {
        self.cfg.flow_lm.latent_dim
    }

    pub fn denormalize_latent(&self, latent: &Array2<f32>) -> Array2<f32> {
        let (t, ldim) = latent.dim();
        let mut out = Array2::<f32>::zeros((t, ldim));
        for i in 0..t {
            for j in 0..ldim {
                out[[i, j]] = latent[[i, j]] * self.emb_std[j] + self.emb_mean[j];
            }
        }
        out
    }
}

/// Sample one latent via Euler integration of the flow net.
/// `c`: `[1, d_model]` — last-token backbone output (already projected through `out_norm`).
/// Returns `[1, ldim]`.
pub fn sample_latent(
    flow_net: &FlowMlp,
    c: &Array2<f32>,
    cfg: &PocketTtsConfig,
    rng: &mut StdRng,
) -> Array2<f32> {
    let ldim = flow_net.latent_dim();
    let std = cfg.temperature.sqrt();
    let normal = Normal::new(0.0f32, std).expect("Normal");
    let mut x = Array2::<f32>::zeros((c.shape()[0], ldim));
    for v in x.iter_mut() {
        *v = normal.sample(rng);
    }
    let n = cfg.lsd_decode_steps.max(1) as f32;
    for i in 0..cfg.lsd_decode_steps.max(1) {
        let s = i as f32 / n;
        let t = (i + 1) as f32 / n;
        let flow = flow_net.forward(c, s, t, &x);
        for i_ in 0..x.shape()[0] {
            for j in 0..ldim {
                x[[i_, j]] += flow[[i_, j]] / n;
            }
        }
    }
    x
}

/// Convenience: build a deterministic RNG.
pub fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
