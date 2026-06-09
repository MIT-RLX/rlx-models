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

//! Streaming Earshot detector (256-sample frames @ 16 kHz).

use super::fft::{self, Complex32};
use super::filters;
use super::predictor::Predictor;

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const N_MELS: usize = 40;
const N_FEATURES: usize = N_MELS;
const N_CONTEXT_FRAMES: usize = 3;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const POWER_FAC: f32 = 1.0 / (32768.0f32 * 32768.0);

pub struct Detector {
    predictor: Predictor,
    prev_signal: f32,
    sample_ring_buffer: [f32; 768],
    features: [f32; N_FEATURES * N_CONTEXT_FRAMES],
    buffer: [f32; 1026],
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector {
    pub fn new() -> Self {
        Self {
            predictor: Predictor::new(),
            prev_signal: 0.0,
            sample_ring_buffer: [0.0; 768],
            features: [0.0; N_FEATURES * N_CONTEXT_FRAMES],
            buffer: [0.0; 1026],
        }
    }

    pub fn reset(&mut self) {
        self.predictor.reset();
        self.prev_signal = 0.0;
        self.sample_ring_buffer.fill(0.0);
        self.features.fill(0.0);
    }

    pub fn predict_i16(&mut self, frame: &[i16]) -> f32 {
        if frame.len() != super::FRAME_SAMPLES {
            return -1.0;
        }
        self.sample_ring_buffer.copy_within(256.., 0);
        for (emph, sample) in self.sample_ring_buffer[512..].iter_mut().zip(frame.iter()) {
            let sample = *sample as f32;
            *emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
            self.prev_signal = sample;
        }
        self.predict_inner()
    }

    pub fn predict_f32(&mut self, frame: &[f32]) -> f32 {
        if frame.len() != super::FRAME_SAMPLES {
            return -1.0;
        }
        const SCALE: f32 = 32768.0;
        self.sample_ring_buffer.copy_within(256.., 0);
        for (emph, sample) in self.sample_ring_buffer[512..].iter_mut().zip(frame.iter()) {
            let sample = *sample * SCALE;
            *emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
            self.prev_signal = sample;
        }
        self.predict_inner()
    }

    fn predict_inner(&mut self) -> f32 {
        for i in 0..WINDOW_SIZE {
            self.buffer[i] = self.sample_ring_buffer[i] * filters::HANN_WINDOW[i];
        }
        self.buffer[WINDOW_SIZE..1024].fill(0.0);
        self.buffer[1024..1026].fill(0.0);

        fft::rfft_1024(&mut self.buffer);
        for i in 0..N_BINS {
            let j = i * 2;
            self.buffer[i] =
                Complex32::new(self.buffer[j], self.buffer[j + 1]).norm_sqr() * POWER_FAC;
        }

        self.features.copy_within(N_FEATURES.., 0);
        let cur = &mut self.features[N_FEATURES * (N_CONTEXT_FRAMES - 1)..];
        for i in 0..N_MELS {
            let mut per_band_value = 0.0;
            let (start, coeffs) = filters::MEL_COEFFS[i];
            for (offs, coeff) in coeffs.iter().enumerate() {
                per_band_value += self.buffer[start + offs] * *coeff;
            }
            cur[i] = (per_band_value + 1e-20).ln();
        }
        self.predictor.normalize(cur);
        self.predictor.predict(&self.features, &mut self.buffer)
    }
}
