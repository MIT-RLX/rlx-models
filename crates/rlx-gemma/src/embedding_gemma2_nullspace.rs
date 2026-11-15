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

//! The null-space audio head for `embeddinggemma-2`
//! ([`BrunoHays/embeddinggemma-2-nullspace-audio`](https://huggingface.co/BrunoHays/embeddinggemma-2-nullspace-audio),
//! write-up: <https://bruno-hays.github.io/posts/maeb-null-space-audio/>).
//!
//! The trunk works in 512 dimensions and `embedding_projection` maps that to
//! 768, so every embedding, text or audio, lies in a 512-dimensional subspace.
//! The other 256 directions (`u_perp`, an orthonormal basis of the projection's
//! orthogonal complement) are never used. The head writes acoustic features
//! there:
//!
//! 1. Pool audio-tower layer 2 (mean ‖ std over time, `[2048]`) and layer 6
//!    (mean, `[1024]`), each L2-normalized.
//! 2. Centre each, project with its fitted PCA (128 columns), divide by the
//!    soft-whitening scale `+ 1e-5`, L2-normalize, weight both blocks by
//!    `√(128/256)`, concatenate and L2-normalize: `z`, `[256]`.
//! 3. `normalize(normalize(base) + γ · u_perp · z)`, `γ = 1`.
//!
//! Text embeddings are left alone, and since `u_perp ⟂` every embedding, a
//! text–audio dot product is unchanged before the final renormalization;
//! audio–audio similarity gains a `⟨z₁, z₂⟩` term. The head adds no trained
//! weights to the model, only the PCA statistics and `u_perp`.
//!
//! Caveats from the write-up: the configuration was chosen on test scores, it
//! costs points on semantic audio tasks (spoken topic, intent, QA), and the
//! audio embedding cannot be truncated (Matryoshka) because `u_perp` is not
//! aligned with any prefix of the coordinates.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;

use crate::embedding_gemma2::l2_normalize;
use crate::embedding_gemma2_model::{AudioEmbedding, EmbeddingGemma2};

/// Audio-tower layers the head reads (1-based, HF `hidden_states` indices).
pub const HEAD_TAPS: [usize; 2] = [2, 6];
/// Added to the whitening scale before dividing.
const SCALE_EPS: f32 = 1e-5;
/// Variance floor of the std pool.
const STD_EPS: f64 = 1e-6;

/// The head's fitted statistics.
pub struct NullSpaceAudioHead {
    /// `[embedding_dim, 256]`, row-major.
    u_perp: Vec<f32>,
    embedding_dim: usize,
    l2: Block,
    l6: Block,
    /// Injection strength (`1.0` as published; `0.0` returns the base embedding).
    pub gamma: f32,
}

/// One PCA block: `normalize(((x − mean) · proj) / (scale + ε))`.
struct Block {
    mean: Vec<f32>,
    /// `[input, out]`, row-major.
    proj: Vec<f32>,
    scale: Vec<f32>,
}

impl Block {
    fn input(&self) -> usize {
        self.mean.len()
    }

    fn out(&self) -> usize {
        self.scale.len()
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let (n, k) = (self.input(), self.out());
        let mut z = vec![0f64; k];
        for (i, xi) in x.iter().enumerate().take(n) {
            let c = f64::from(xi - self.mean[i]);
            for (zj, p) in z.iter_mut().zip(&self.proj[i * k..(i + 1) * k]) {
                *zj += c * f64::from(*p);
            }
        }
        let mut z: Vec<f32> = z
            .iter()
            .zip(&self.scale)
            .map(|(v, s)| (*v / f64::from(s + SCALE_EPS)) as f32)
            .collect();
        l2_normalize(&mut z);
        z
    }
}

impl NullSpaceAudioHead {
    /// Load `nullspace_audio_head.safetensors`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut m = WeightMap::from_file(path.to_str().context("non-UTF-8 path")?)
            .with_context(|| format!("load {}", path.display()))?;
        let mut take = |k: &str, want: &[usize]| -> Result<Vec<f32>> {
            let (v, s) = m
                .take(k)
                .with_context(|| format!("{}: no {k}", path.display()))?;
            if s != want {
                bail!("{}: {k} is {s:?}, expected {want:?}", path.display());
            }
            Ok(v)
        };
        let u_perp_shape = [768, 256];
        let head = Self {
            u_perp: take("u_perp", &u_perp_shape)?,
            embedding_dim: u_perp_shape[0],
            l2: Block {
                mean: take("l2_ms_mean", &[1, 2048])?,
                proj: take("l2_ms_proj", &[2048, 128])?,
                scale: take("l2_ms_scale", &[128])?,
            },
            l6: Block {
                mean: take("l6_m_mean", &[1, 1024])?,
                proj: take("l6_m_proj", &[1024, 128])?,
                scale: take("l6_m_scale", &[128])?,
            },
            gamma: 1.0,
        };
        Ok(head)
    }

    /// Width of the injected subspace (`256`).
    pub fn dim(&self) -> usize {
        self.u_perp.len() / self.embedding_dim
    }

    /// Mean ‖ std over the `seq` rows of layer-2 states `[seq · hidden]`, L2
    /// normalized: `[2 · hidden]`.
    pub fn pool_mean_std(states: &[f32], seq: usize) -> Vec<f32> {
        let h = states.len() / seq.max(1);
        let mut mu = vec![0f64; h];
        for row in states.chunks_exact(h) {
            for (m, v) in mu.iter_mut().zip(row) {
                *m += f64::from(*v);
            }
        }
        mu.iter_mut().for_each(|m| *m /= seq as f64);
        let mut var = vec![0f64; h];
        for row in states.chunks_exact(h) {
            for ((s, v), m) in var.iter_mut().zip(row).zip(&mu) {
                let d = f64::from(*v) - m;
                *s += d * d;
            }
        }
        let mut out: Vec<f32> = mu.iter().map(|m| *m as f32).collect();
        out.extend(var.iter().map(|s| (s / seq as f64 + STD_EPS).sqrt() as f32));
        l2_normalize(&mut out);
        out
    }

    /// Mean over the `seq` rows of `[seq · hidden]`, L2 normalized.
    pub fn pool_mean(states: &[f32], seq: usize) -> Vec<f32> {
        let h = states.len() / seq.max(1);
        let mut mu = vec![0f64; h];
        for row in states.chunks_exact(h) {
            for (m, v) in mu.iter_mut().zip(row) {
                *m += f64::from(*v);
            }
        }
        let mut out: Vec<f32> = mu.iter().map(|m| (*m / seq as f64) as f32).collect();
        l2_normalize(&mut out);
        out
    }

    /// The null-space coordinates `z` (`[256]`, unit norm) of the two pools.
    pub fn z(&self, l2_mean_std: &[f32], l6_mean: &[f32]) -> Result<Vec<f32>> {
        if l2_mean_std.len() != self.l2.input() || l6_mean.len() != self.l6.input() {
            bail!(
                "nullspace head: pools are {} and {} wide, expected {} and {}",
                l2_mean_std.len(),
                l6_mean.len(),
                self.l2.input(),
                self.l6.input()
            );
        }
        let w = (self.l2.out() as f32 / self.dim() as f32).sqrt();
        let mut z: Vec<f32> = self
            .l2
            .apply(l2_mean_std)
            .into_iter()
            .chain(self.l6.apply(l6_mean))
            .map(|v| w * v)
            .collect();
        l2_normalize(&mut z);
        Ok(z)
    }

    /// `normalize(normalize(base) + γ · u_perp · z)`.
    pub fn inject(&self, base: &[f32], z: &[f32]) -> Result<Vec<f32>> {
        let k = self.dim();
        if base.len() != self.embedding_dim || z.len() != k {
            bail!(
                "nullspace head: base {} / z {} wide, expected {} / {k}",
                base.len(),
                z.len(),
                self.embedding_dim
            );
        }
        let mut v = base.to_vec();
        l2_normalize(&mut v);
        if self.gamma != 0.0 {
            for (vi, row) in v.iter_mut().zip(self.u_perp.chunks_exact(k)) {
                *vi += self.gamma * row.iter().zip(z).map(|(u, zj)| u * zj).sum::<f32>();
            }
            l2_normalize(&mut v);
        }
        Ok(v)
    }

    /// The head applied to an [`AudioEmbedding`] computed with
    /// [`HEAD_TAPS`].
    pub fn apply(&self, a: &AudioEmbedding) -> Result<Vec<f32>> {
        if a.taps.len() != HEAD_TAPS.len() {
            bail!("nullspace head: needs the tower states after layers {HEAD_TAPS:?}");
        }
        let l2 = Self::pool_mean_std(&a.taps[0], a.audio_tokens);
        let l6 = Self::pool_mean(&a.taps[1], a.audio_tokens);
        self.inject(&a.embedding, &self.z(&l2, &l6)?)
    }

    /// Largest `|⟨u_perp column, W column⟩| / ‖W column‖` over the
    /// `[embedding_dim, hidden]` projection `w`: how far `u_perp` is from
    /// orthogonal to everything the model can output. Near zero for the
    /// checkpoint the head was fitted to.
    pub fn max_overlap(&self, w: &[f32], hidden: usize) -> f32 {
        let (d, k) = (self.embedding_dim, self.dim());
        let mut worst = 0f32;
        for c in 0..hidden {
            let norm = (0..d)
                .map(|i| w[i * hidden + c].powi(2))
                .sum::<f32>()
                .sqrt();
            if norm == 0.0 {
                continue;
            }
            for j in 0..k {
                let dot: f32 = (0..d)
                    .map(|i| self.u_perp[i * k + j] * w[i * hidden + c])
                    .sum();
                worst = worst.max((dot / norm).abs());
            }
        }
        worst
    }
}

impl EmbeddingGemma2 {
    /// [`EmbeddingGemma2::embed_audio`] with the null-space head on top:
    /// the head's audio embedding, `[embedding_dim]`.
    pub fn embed_audio_nullspace(
        &mut self,
        head: &NullSpaceAudioHead,
        prompt_ids: &[u32],
        samples: &[f32],
    ) -> Result<Vec<f32>> {
        let a = self.embed_audio(prompt_ids, samples, &HEAD_TAPS)?;
        head.apply(&a)
    }
}
