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

//! Compiled inference session for learned FFT / IFFT.

use crate::model::butterfly::{
    build_butterfly_forward_graph, build_butterfly_inverse_graph, butterfly_forward_real_batch,
    butterfly_inverse_complex_batch,
};
use crate::model::config::{FftLearnConfig, TransformDir};
use crate::model::reference::{fft_real_batch, ifft_complex_batch, max_abs_error};
use crate::model::twiddle::exact_twiddles;
use crate::model::weights::WeightStore;
use anyhow::{Result, bail};
use rlx_runtime::{CompiledGraph, Device};

pub struct FftLearnRunner {
    cfg: FftLearnConfig,
    direction: TransformDir,
    twiddles: Vec<f32>,
    compiled: Option<(Device, CompiledGraph)>,
}

impl FftLearnRunner {
    pub fn new(cfg: FftLearnConfig) -> Result<Self> {
        Self::new_dir(cfg, TransformDir::Forward)
    }

    pub fn new_ifft(cfg: FftLearnConfig) -> Result<Self> {
        Self::new_dir(cfg, TransformDir::Inverse)
    }

    pub fn new_dir(cfg: FftLearnConfig, direction: TransformDir) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            twiddles: exact_twiddles(&cfg),
            cfg,
            direction,
            compiled: None,
        })
    }

    pub fn with_weights(cfg: FftLearnConfig, weights: &WeightStore) -> Result<Self> {
        Self::with_weights_dir(cfg, weights, TransformDir::Forward)
    }

    pub fn with_weights_ifft(cfg: FftLearnConfig, weights: &WeightStore) -> Result<Self> {
        Self::with_weights_dir(cfg, weights, TransformDir::Inverse)
    }

    pub fn with_weights_dir(
        cfg: FftLearnConfig,
        weights: &WeightStore,
        direction: TransformDir,
    ) -> Result<Self> {
        let mut this = Self::new_dir(cfg, direction)?;
        this.twiddles = weights.to_twiddles(this.cfg.n_fft)?;
        Ok(this)
    }

    pub fn load_compiled(&mut self, device: Device) -> Result<()> {
        let built = if self.direction.is_forward() {
            build_butterfly_forward_graph(&self.cfg)?
        } else {
            build_butterfly_inverse_graph(&self.cfg)?
        };
        let store = WeightStore::from_twiddles(&self.twiddles, self.cfg.n_fft);
        let mut compiled = crate::exec::compile::try_compile_graph(device, built.graph)?;
        store.apply_butterfly(&mut compiled, self.cfg.batch, self.cfg.n_fft);
        self.compiled = Some((device, compiled));
        Ok(())
    }

    pub fn forward_eager(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.direction.is_forward() {
            butterfly_forward_real_batch(input, &self.twiddles, self.cfg.batch, self.cfg.n_fft)
        } else {
            butterfly_inverse_complex_batch(input, &self.twiddles, self.cfg.batch, self.cfg.n_fft)
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.compiled.is_some() {
            self.forward_compiled(input)
        } else {
            self.forward_eager(input)
        }
    }

    fn forward_compiled(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let expected = if self.direction.is_forward() {
            self.cfg.batch * self.cfg.n_fft
        } else {
            self.cfg.batch * self.cfg.n_fft * 2
        };
        if input.len() != expected {
            bail!("input len {} != expected {}", input.len(), expected);
        }
        let Some((_, ref mut exec)) = self.compiled else {
            bail!("compiled session not loaded");
        };
        let input_name = if self.direction.is_forward() {
            "signal"
        } else {
            "spectrum"
        };
        let outputs = exec.run(&[(input_name, input)]);
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("butterfly graph produced no outputs"))
    }

    pub fn compare_reference(&self, input: &[f32]) -> Result<(f32, f32)> {
        let pred = self.forward_eager(input)?;
        let target = if self.direction.is_forward() {
            fft_real_batch(input, self.cfg.batch, self.cfg.n_fft)?
        } else {
            ifft_complex_batch(input, self.cfg.batch, self.cfg.n_fft)?
        };
        Ok((
            crate::model::reference::mse(&pred, &target),
            max_abs_error(&pred, &target),
        ))
    }

    pub fn config(&self) -> &FftLearnConfig {
        &self.cfg
    }

    pub fn direction(&self) -> TransformDir {
        self.direction
    }
}
