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

//! Top-level AEC session: delay alignment + FDAF + optional residual.

use crate::audio::SAMPLE_RATE;
use crate::delay::{ReferenceRing, estimate_delay_samples};
use crate::fdaf::{FdafConfig, FdafNlms};
use crate::residual::{ResidualWeights, embedded_residual_weights};
use anyhow::{Result, ensure};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AecConfig {
    pub n_fft: usize,
    pub frame_samples: usize,
    pub step_size: f32,
    pub max_delay_ms: u32,
    pub residual: bool,
    pub delay_reestimate_frames: usize,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            frame_samples: 160,
            step_size: 0.05,
            max_delay_ms: 300,
            residual: true,
            delay_reestimate_frames: 50,
        }
    }
}

pub struct AecSession {
    cfg: AecConfig,
    delay_samples: usize,
    max_delay_samples: usize,
    reference_ring: ReferenceRing,
    fdaf: FdafNlms,
    frame_count: usize,
    pending_far: Vec<f32>,
}

impl AecSession {
    pub fn new(cfg: AecConfig) -> Result<Self> {
        let max_delay_samples = (cfg.max_delay_ms as usize * SAMPLE_RATE) / 1000;
        let residual = if cfg.residual {
            Some(embedded_residual_weights()?)
        } else {
            None
        };
        let fdaf_cfg = FdafConfig {
            n_fft: cfg.n_fft,
            frame_samples: cfg.frame_samples,
            step_size: cfg.step_size,
            adapt: true,
            use_residual: cfg.residual,
        };
        let fdaf = FdafNlms::new(fdaf_cfg, residual)?;
        Ok(Self {
            cfg,
            delay_samples: 0,
            max_delay_samples,
            reference_ring: ReferenceRing::new(max_delay_samples),
            fdaf,
            frame_count: 0,
            pending_far: Vec::new(),
        })
    }

    pub fn with_residual_weights(cfg: AecConfig, weights: ResidualWeights) -> Result<Self> {
        let max_delay_samples = (cfg.max_delay_ms as usize * SAMPLE_RATE) / 1000;
        let fdaf_cfg = FdafConfig {
            n_fft: cfg.n_fft,
            frame_samples: cfg.frame_samples,
            step_size: cfg.step_size,
            adapt: true,
            use_residual: cfg.residual,
        };
        let residual = if cfg.residual { Some(weights) } else { None };
        let fdaf = FdafNlms::new(fdaf_cfg, residual)?;
        Ok(Self {
            cfg,
            delay_samples: 0,
            max_delay_samples,
            reference_ring: ReferenceRing::new(max_delay_samples),
            fdaf,
            frame_count: 0,
            pending_far: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.delay_samples = 0;
        self.frame_count = 0;
        self.reference_ring.clear();
        self.pending_far.clear();
        self.fdaf.reset();
    }

    pub fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    /// Push far-end reference (e.g. TTS playback tap) without mic input.
    pub fn push_reference(&mut self, far_end: &[f32]) {
        self.reference_ring.push(far_end);
        self.pending_far.extend_from_slice(far_end);
    }

    /// Process mic + synchronous far-end chunk.
    pub fn process(&mut self, mic: &[f32], far_end: &[f32]) -> Result<Vec<f32>> {
        ensure!(mic.len() == far_end.len());
        self.push_reference(far_end);
        self.process_mic(mic)?
            .ok_or_else(|| anyhow::anyhow!("empty mic buffer"))
    }

    /// Process mic using buffered reference ring.
    pub fn process_mic(&mut self, mic: &[f32]) -> Result<Option<Vec<f32>>> {
        if mic.is_empty() {
            return Ok(None);
        }
        let hop = self.cfg.frame_samples;
        let mut out = vec![0.0f32; mic.len()];
        let mut pos = 0;
        while pos < mic.len() {
            let end = (pos + hop).min(mic.len());
            let chunk = end - pos;
            let mut aligned = vec![0.0f32; hop];
            self.reference_ring
                .read_delayed(self.delay_samples, hop, &mut aligned);
            if chunk < hop {
                aligned.fill(0.0);
                self.reference_ring
                    .read_delayed(self.delay_samples, hop, &mut aligned);
            }
            let mut frame_out = vec![0.0f32; hop];
            if chunk == hop {
                self.fdaf
                    .process_frame(&mic[pos..end], &aligned, &mut frame_out)?;
                out[pos..end].copy_from_slice(&frame_out);
            } else {
                let mut mp = vec![0.0f32; hop];
                mp[..chunk].copy_from_slice(&mic[pos..end]);
                self.fdaf.process_frame(&mp, &aligned, &mut frame_out)?;
                out[pos..end].copy_from_slice(&frame_out[..chunk]);
            }
            pos = end;
            self.frame_count += 1;
            if self.frame_count % self.cfg.delay_reestimate_frames == 1 {
                self.maybe_reestimate_delay(mic);
            }
        }
        Ok(Some(out))
    }

    fn maybe_reestimate_delay(&mut self, mic: &[f32]) {
        let n = self
            .pending_far
            .len()
            .min(mic.len())
            .min(self.cfg.n_fft * 4);
        if n < 256 {
            return;
        }
        let est = estimate_delay_samples(
            &self.pending_far[..n],
            &mic[..n],
            self.cfg.n_fft,
            self.max_delay_samples,
        );
        self.delay_samples = est;
    }

    /// Offline: estimate delay once then process full buffers.
    pub fn process_aligned_buffers(&mut self, mic: &[f32], far: &[f32]) -> Result<Vec<f32>> {
        ensure!(mic.len() == far.len());
        let n = mic.len().min(far.len()).min(self.cfg.n_fft * 8);
        if n >= 256 {
            self.delay_samples = estimate_delay_samples(
                &far[..n],
                &mic[..n],
                self.cfg.n_fft,
                self.max_delay_samples,
            );
        }
        let mut aligned_far = vec![0.0f32; far.len()];
        if self.delay_samples < far.len() {
            aligned_far[self.delay_samples..]
                .copy_from_slice(&far[..far.len() - self.delay_samples]);
        }
        let mut out = vec![0.0f32; mic.len()];
        self.fdaf.process_buffer(mic, &aligned_far, &mut out)?;
        Ok(out)
    }

    /// Process paired WAV files at 16 kHz.
    pub fn process_wav_files(
        mic_path: &Path,
        ref_path: &Path,
        cfg: &AecConfig,
    ) -> Result<(Vec<f32>, usize)> {
        let mic = crate::audio::load_wav_16k(mic_path)?;
        let far = crate::audio::load_wav_16k(ref_path)?;
        let len = mic.len().min(far.len());
        let mut session = AecSession::new(cfg.clone())?;
        let out = session.process_aligned_buffers(&mic[..len], &far[..len])?;
        Ok((out, session.delay_samples()))
    }
}
