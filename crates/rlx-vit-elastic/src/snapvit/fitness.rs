// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Label-free fitness (SnapViT Eq. 6): the PCA-compressed cosine similarity
//! between a pruned model's embeddings and the original model's, averaged over
//! a set of images. Runs entirely on forward passes of the maskable graph (any
//! RLX backend) — the derivative-free objective the xNES search maximizes.

use anyhow::Result;
use rlx_runtime::Device;

use crate::vit::config::VitConfig;
use crate::vit::preprocess::assemble_hidden;
use crate::vit::runner::VitRunner;
use crate::vit::weights::LoadedVit;

use super::local::CalibImage;

/// Optional PCA basis fit on the original embeddings.
struct Pca {
    mean: Vec<f32>,
    components: Vec<Vec<f32>>, // k × D
}

impl Pca {
    /// Project `x [D]` → `[k]` (mean-centered onto the components).
    fn project(&self, x: &[f32]) -> Vec<f32> {
        self.components
            .iter()
            .map(|c| {
                x.iter()
                    .zip(&self.mean)
                    .zip(c)
                    .map(|((&xi, &m), &ci)| (xi - m) * ci)
                    .sum()
            })
            .collect()
    }
}

/// Fit top-`k` principal components of `data [n, d]` via power iteration with
/// deflation (fit once on the original embeddings, reused for every candidate).
fn fit_pca(data: &[f32], n: usize, d: usize, k: usize) -> Pca {
    let mut mean = vec![0f32; d];
    for i in 0..n {
        for j in 0..d {
            mean[j] += data[i * d + j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let mut x = vec![0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = data[i * d + j] - mean[j];
        }
    }
    let mut components: Vec<Vec<f32>> = Vec::with_capacity(k);
    for c in 0..k {
        let mut v = vec![0f32; d];
        v[c % d] = 1.0;
        for _ in 0..50 {
            // u = Xᵀ(Xv) ∝ Cv
            let mut w = vec![0f32; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..d {
                    s += x[i * d + j] * v[j];
                }
                w[i] = s;
            }
            let mut u = vec![0f32; d];
            for i in 0..n {
                let wi = w[i];
                for j in 0..d {
                    u[j] += x[i * d + j] * wi;
                }
            }
            for prev in &components {
                let dot: f32 = u.iter().zip(prev).map(|(a, b)| a * b).sum();
                for j in 0..d {
                    u[j] -= dot * prev[j];
                }
            }
            let norm = u.iter().map(|a| a * a).sum::<f32>().sqrt() + 1e-12;
            for j in 0..d {
                u[j] /= norm;
            }
            v = u;
        }
        for i in 0..n {
            let dot: f32 = (0..d).map(|j| x[i * d + j] * v[j]).sum();
            for j in 0..d {
                x[i * d + j] -= dot * v[j];
            }
        }
        components.push(v);
    }
    Pca { mean, components }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x * x) as f64;
        nb += (y * y) as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

/// A reusable fitness evaluator: original embeddings + optional PCA + a masked
/// runner. `eval` sets the candidate masks and returns the mean PCA-cosine.
pub struct Fitness {
    runner: VitRunner,
    stacked_hidden: Vec<f32>,
    n_img: usize,
    dim: usize,
    orig: Vec<Vec<f32>>,
    pca: Option<Pca>,
}

impl Fitness {
    /// Build over `images` (whole-image embeddings). `pca_dim == 0` or
    /// `>= hidden_size` disables PCA (raw-cosine fitness).
    pub fn new(
        cfg: &VitConfig,
        loaded: LoadedVit,
        images: &[CalibImage],
        device: Device,
        pca_dim: usize,
    ) -> Result<Self> {
        let n_img = images.len();
        let img = cfg.img_size;
        let d = cfg.hidden_size;

        // One resized whole-image view per image → a [n_img, seq, H] batch.
        let mut nchw = Vec::with_capacity(n_img * 3 * img * img);
        for im in images {
            let v = crate::vit::preprocess::rgb_u8_to_imagenet_nchw(&im.rgb, im.h, im.w, img);
            nchw.extend_from_slice(&v);
        }
        let preprocess = loaded.preprocess.clone();
        let mut runner = VitRunner::from_loaded(cfg.clone(), loaded, device, n_img)?;
        let stacked_hidden = assemble_hidden(&preprocess, &nchw, n_img)?;

        runner.reset_masks();
        let orig = runner.embed_hidden(&stacked_hidden)?;

        let pca = if pca_dim > 0 && pca_dim < d && n_img > 1 {
            let k = pca_dim.min(d).min(n_img - 1).max(1);
            let flat: Vec<f32> = orig.iter().flatten().copied().collect();
            Some(fit_pca(&flat, n_img, d, k))
        } else {
            None
        };

        Ok(Self {
            runner,
            stacked_hidden,
            n_img,
            dim: d,
            orig,
            pca,
        })
    }

    /// Mean cosine similarity (in PCA space if enabled) of the masked model's
    /// embeddings vs the original — in `[-1, 1]`, higher is better.
    pub fn eval(&mut self, head_mask: Vec<f32>, ffn_mask: Vec<f32>) -> Result<f32> {
        self.runner.set_masks(head_mask, ffn_mask)?;
        let pruned = self.runner.embed_hidden(&self.stacked_hidden)?;
        let mut acc = 0.0f32;
        for i in 0..self.n_img {
            let (a, b) = match &self.pca {
                Some(p) => (p.project(&self.orig[i]), p.project(&pruned[i])),
                None => (self.orig[i].clone(), pruned[i].clone()),
            };
            acc += cosine(&a, &b);
        }
        Ok(acc / self.n_img.max(1) as f32)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}
