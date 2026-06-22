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

//! Frequency-domain adaptive filter (FDAF) with overlap-save NLMS.

use crate::dtd::DoubleTalkDetector;
use crate::residual::ResidualWeights;
use anyhow::{Result, ensure};
use rlx_fft::reference::{fft_real_batch, ifft_complex_batch};

#[derive(Debug, Clone)]
pub struct FdafConfig {
    pub n_fft: usize,
    pub frame_samples: usize,
    pub step_size: f32,
    pub adapt: bool,
    pub use_residual: bool,
}

impl Default for FdafConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            frame_samples: 160,
            step_size: 0.05,
            adapt: true,
            use_residual: true,
        }
    }
}

pub struct FdafNlms {
    cfg: FdafConfig,
    w_re: Vec<f32>,
    w_im: Vec<f32>,
    psd: Vec<f32>,
    far_block: Vec<f32>,
    dtd: DoubleTalkDetector,
    residual: Option<ResidualWeights>,
    far_power_smooth: f32,
    psd_smooth: f32,
}

impl FdafNlms {
    pub fn new(cfg: FdafConfig, residual: Option<ResidualWeights>) -> Result<Self> {
        let n = cfg.n_fft;
        let hop = cfg.frame_samples;
        ensure!(hop > 0 && hop < n, "frame_samples must be in (0, n_fft)");
        let use_residual = cfg.use_residual;
        Ok(Self {
            cfg,
            w_re: vec![0.0; n],
            w_im: vec![0.0; n],
            psd: vec![1.0; n],
            far_block: vec![0.0; n],
            dtd: DoubleTalkDetector::default(),
            residual: residual.filter(|r| use_residual && r.n_fft == n),
            far_power_smooth: 0.0,
            psd_smooth: 0.98,
        })
    }

    pub fn reset(&mut self) {
        self.w_re.fill(0.0);
        self.w_im.fill(0.0);
        self.psd.fill(1.0);
        self.far_block.fill(0.0);
        self.far_power_smooth = 0.0;
    }

    pub fn config(&self) -> &FdafConfig {
        &self.cfg
    }

    fn fft_block(&self, time: &[f32]) -> Result<Vec<f32>> {
        ensure!(time.len() == self.cfg.n_fft);
        fft_real_batch(time, 1, self.cfg.n_fft)
    }

    fn ifft_block(&self, spec: &[f32]) -> Result<Vec<f32>> {
        ifft_complex_batch(spec, 1, self.cfg.n_fft)
    }

    /// Process one hop of mic and aligned far-end samples.
    pub fn process_frame(&mut self, mic: &[f32], far: &[f32], out: &mut [f32]) -> Result<()> {
        let hop = self.cfg.frame_samples;
        let n = self.cfg.n_fft;
        ensure!(mic.len() >= hop && far.len() >= hop && out.len() >= hop);

        self.far_block.copy_within(hop..n, 0);
        for i in 0..hop {
            self.far_block[n - hop + i] = far[i];
        }

        let xf = self.fft_block(&self.far_block)?;

        for k in 0..n {
            let xr = xf[k * 2];
            let xi = xf[k * 2 + 1];
            let power = xr * xr + xi * xi;
            self.psd[k] = self.psd_smooth * self.psd[k] + (1.0 - self.psd_smooth) * power;
        }

        let mut y_spec = vec![0.0f32; n * 2];
        for k in 0..n {
            let xr = xf[k * 2];
            let xi = xf[k * 2 + 1];
            let wr = self.w_re[k];
            let wi = self.w_im[k];
            y_spec[k * 2] = wr * xr - wi * xi;
            y_spec[k * 2 + 1] = wr * xi + wi * xr;
        }

        let y_time = self.ifft_block(&y_spec)?;
        let scale = 1.0 / n as f32;

        let mut echo_est = vec![0.0f32; hop];
        for i in 0..hop {
            echo_est[i] = y_time[(n - hop + i) * 2] * scale;
        }

        let mut error = vec![0.0f32; hop];
        for i in 0..hop {
            error[i] = mic[i] - echo_est[i];
        }

        let mic_power = DoubleTalkDetector::frame_power(&mic[..hop]);
        let echo_power = DoubleTalkDetector::frame_power(&echo_est);
        self.far_power_smooth =
            0.9 * self.far_power_smooth + 0.1 * (self.psd.iter().sum::<f32>() / n as f32);
        let pause = self
            .dtd
            .pause_adaptation(mic_power, self.far_power_smooth, echo_power);

        let mut e_time_buf = vec![0.0f32; n];
        for i in 0..hop {
            e_time_buf[n - hop + i] = error[i];
        }
        let mut ef = self.fft_block(&e_time_buf)?;

        if let Some(res) = &self.residual {
            res.apply_spectrum(&mut ef);
            let e_time = self.ifft_block(&ef)?;
            for i in 0..hop {
                error[i] = e_time[(n - hop + i) * 2] * scale;
            }
        }

        out[..hop].copy_from_slice(&error[..hop]);

        if self.cfg.adapt && !pause {
            let mu = self.cfg.step_size;
            for k in 0..n {
                let xr = xf[k * 2];
                let xi = xf[k * 2 + 1];
                let er = ef[k * 2];
                let ei = ef[k * 2 + 1];
                let gr = (xr * er + xi * ei) / (self.psd[k] + 1e-10);
                let gi = (xr * ei - xi * er) / (self.psd[k] + 1e-10);
                self.w_re[k] += mu * gr;
                self.w_im[k] += mu * gi;
            }
        }

        Ok(())
    }

    pub fn process_buffer(&mut self, mic: &[f32], far: &[f32], out: &mut [f32]) -> Result<()> {
        ensure!(mic.len() == far.len() && mic.len() == out.len());
        let hop = self.cfg.frame_samples;
        let mut pos = 0;
        while pos < mic.len() {
            let end = (pos + hop).min(mic.len());
            let chunk = end - pos;
            let mut mp = vec![0.0f32; hop];
            let mut fp = vec![0.0f32; hop];
            let mut op = vec![0.0f32; hop];
            mp[..chunk].copy_from_slice(&mic[pos..end]);
            fp[..chunk].copy_from_slice(&far[pos..end]);
            self.process_frame(&mp, &fp, &mut op)?;
            out[pos..end].copy_from_slice(&op[..chunk]);
            pos += chunk;
        }
        Ok(())
    }
}
