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

//! Domain-adaptive twiddle training (Tier C).

use crate::model::butterfly::butterfly_train_step;
use crate::model::config::FftLearnConfig;
use crate::model::reference::{fft_real_batch, max_abs_error};
use crate::model::twiddle::exact_twiddles;
use anyhow::Result;
use rand::prelude::*;
use std::f32::consts::TAU;

/// Colored / structured signals closer to audio-ish spectra than uniform random.
pub fn domain_batch(rng: &mut impl Rng, batch: usize, n_fft: usize) -> Vec<f32> {
    let mut out = vec![0f32; batch * n_fft];
    for b in 0..batch {
        let f0 = 0.02 + rng.gen_range(0.0..1.0) * 0.15;
        let f1 = 0.2 + rng.gen_range(0.0..1.0) * 0.35;
        let a0 = 0.5 + rng.gen_range(0.0..1.0) * 0.5;
        let a1 = 0.1 + rng.gen_range(0.0..1.0) * 0.4;
        for i in 0..n_fft {
            let t = i as f32;
            out[b * n_fft + i] = a0 * (TAU * f0 * t).sin()
                + a1 * (TAU * f1 * t).cos()
                + 0.05 * rng.gen_range(0.0..1.0);
        }
    }
    out
}

pub fn train_domain_twiddles(
    cfg: &FftLearnConfig,
    steps: usize,
    lr: f32,
    seed: u64,
) -> Result<(Vec<f32>, f32)> {
    let mut tw = exact_twiddles(cfg);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut last_err = 0f32;
    for _ in 0..steps {
        let signal = domain_batch(&mut rng, cfg.batch, cfg.n_fft);
        butterfly_train_step(&signal, &mut tw, cfg.batch, cfg.n_fft, lr)?;
        let pred = crate::model::butterfly::butterfly_forward_real_batch(
            &signal, &tw, cfg.batch, cfg.n_fft,
        )?;
        let target = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        last_err = max_abs_error(&pred, &target);
    }
    Ok((tw, last_err))
}
