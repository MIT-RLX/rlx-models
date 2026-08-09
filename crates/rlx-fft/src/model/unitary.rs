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

//! Learnable 2×2 complex butterfly mixing matrices (Tier C).

use crate::model::butterfly::num_stages;
use crate::model::config::FftLearnConfig;
use crate::model::reference::{fft_real_batch, max_abs_error};
use crate::model::twiddle::{exact_twiddles, twiddle_index};
use crate::train::random_batch;
use anyhow::{Result, ensure};
use rand::prelude::*;

/// Per butterfly: 2×2 complex matrix as 8 floats `[m00r,m00i,m01r,m01i,m10r,m10i,m11r,m11i]`.
#[derive(Debug, Clone)]
pub struct UnitaryWeights {
    pub matrices: Vec<f32>,
}

impl UnitaryWeights {
    pub fn param_count(n_fft: usize) -> usize {
        num_stages(n_fft) * (n_fft / 2) * 8
    }

    pub fn exact_init(cfg: &FftLearnConfig) -> Self {
        let tw = exact_twiddles(cfg);
        let n = cfg.n_fft;
        let half = n / 2;
        let stages = cfg.num_stages();
        let mut matrices = vec![0f32; stages * half * 8];
        for s in 0..stages {
            for b in 0..half {
                let w_base = twiddle_index(s, b, half, 0);
                let w_re = tw[w_base];
                let w_im = tw[w_base + 1];
                let base = (s * half + b) * 8;
                matrices[base] = 1.0;
                matrices[base + 1] = 0.0;
                matrices[base + 2] = w_re;
                matrices[base + 3] = w_im;
                matrices[base + 4] = 1.0;
                matrices[base + 5] = 0.0;
                matrices[base + 6] = -w_re;
                matrices[base + 7] = -w_im;
            }
        }
        Self { matrices }
    }

    fn apply_mat(m: &[f32], a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> (f32, f32, f32, f32) {
        let (m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i) =
            (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
        let top_re = m00r * a_re - m00i * a_im + m01r * b_re - m01i * b_im;
        let top_im = m00r * a_im + m00i * a_re + m01r * b_im + m01i * b_re;
        let bot_re = m10r * a_re - m10i * a_im + m11r * b_re - m11i * b_im;
        let bot_im = m10r * a_im + m10i * a_re + m11r * b_im + m11i * b_re;
        (top_re, top_im, bot_re, bot_im)
    }

    pub fn forward_real_batch(
        &self,
        signal: &[f32],
        batch: usize,
        n_fft: usize,
    ) -> Result<Vec<f32>> {
        ensure!(signal.len() == batch * n_fft);
        let mut out = vec![0f32; batch * n_fft * 2];
        for b in 0..batch {
            let mut state = vec![0f32; n_fft * 2];
            for i in 0..n_fft {
                state[i * 2] = signal[b * n_fft + i];
            }
            let spec = self.forward_one(&state, n_fft)?;
            out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&spec);
        }
        Ok(out)
    }

    fn forward_one(&self, input: &[f32], n_fft: usize) -> Result<Vec<f32>> {
        use crate::model::butterfly::bit_reverse_permute;
        let half = n_fft / 2;
        let stages = num_stages(n_fft);
        let mut buf = input.to_vec();
        bit_reverse_permute(&mut buf, n_fft);
        for s in 0..stages {
            let stride = 1usize << s;
            let mut next = vec![0f32; n_fft * 2];
            for b_idx in 0..half {
                let group = b_idx / stride;
                let k = b_idx % stride;
                let i0 = (group * 2 * stride + k) * 2;
                let i1 = i0 + stride * 2;
                let m_base = (s * half + b_idx) * 8;
                let m = &self.matrices[m_base..m_base + 8];
                let (top_re, top_im, bot_re, bot_im) =
                    Self::apply_mat(m, buf[i0], buf[i0 + 1], buf[i1], buf[i1 + 1]);
                next[i0] = top_re;
                next[i0 + 1] = top_im;
                next[i1] = bot_re;
                next[i1 + 1] = bot_im;
            }
            buf = next;
        }
        Ok(buf)
    }
}

pub fn train_unitary_quick(
    cfg: &FftLearnConfig,
    steps: usize,
    lr: f32,
    seed: u64,
) -> Result<(UnitaryWeights, f32)> {
    let mut weights = UnitaryWeights::exact_init(cfg);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eps = 1e-4f32;
    let mut last_err = f32::MAX;
    for _ in 0..steps {
        let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
        let pred = weights.forward_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        let target = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        last_err = max_abs_error(&pred, &target);
        let n = cfg.n_fft;
        let half = n / 2;
        let stages = num_stages(n);
        for s in 0..stages {
            for b_idx in 0..half {
                let m_base = (s * half + b_idx) * 8;
                for k in 0..8 {
                    let orig = weights.matrices[m_base + k];
                    weights.matrices[m_base + k] = orig + eps;
                    let p_plus = weights.forward_real_batch(&signal, cfg.batch, cfg.n_fft)?;
                    let err_plus = max_abs_error(&p_plus, &target);
                    weights.matrices[m_base + k] = orig - eps;
                    let p_minus = weights.forward_real_batch(&signal, cfg.batch, cfg.n_fft)?;
                    let err_minus = max_abs_error(&p_minus, &target);
                    weights.matrices[m_base + k] = orig;
                    let grad = (err_plus - err_minus) / (2.0 * eps);
                    weights.matrices[m_base + k] -= lr * grad;
                }
            }
        }
    }
    Ok((weights, last_err))
}
