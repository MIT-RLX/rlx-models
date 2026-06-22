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

//! Delay estimation and far-end ring buffer.

/// Ring buffer for asynchronous far-end reference playback.
#[derive(Debug, Clone)]
pub struct ReferenceRing {
    buf: Vec<f32>,
    cap: usize,
    write_pos: usize,
    len: usize,
}

impl ReferenceRing {
    pub fn new(max_delay_samples: usize) -> Self {
        let cap = max_delay_samples.max(160) + 4096;
        Self {
            buf: vec![0.0; cap],
            cap,
            write_pos: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.buf[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.cap;
            self.len = self.len.saturating_add(1).min(self.cap);
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.write_pos = 0;
    }

    /// Read `out_len` samples ending `delay_samples` before the write head.
    pub fn read_delayed(&self, delay_samples: usize, out_len: usize, out: &mut [f32]) {
        let n = out.len().min(out_len);
        for i in 0..n {
            let age = delay_samples + (out_len - n) + i;
            out[i] = self.sample_at_age(age);
        }
    }

    fn sample_at_age(&self, age: usize) -> f32 {
        if self.len == 0 || age >= self.len {
            return 0.0;
        }
        let idx = (self.write_pos + self.cap - 1 - age) % self.cap;
        self.buf[idx]
    }
}

/// Estimate acoustic delay (samples): far leads mic by `lag` samples.
pub fn estimate_delay_samples(far: &[f32], mic: &[f32], _n_fft: usize, max_delay: usize) -> usize {
    let n = far.len().min(mic.len());
    if n < 64 {
        return 0;
    }
    let search = max_delay.min(n - 1);
    let mut best_lag = 0usize;
    let mut best = -1e30f32;
    for lag in 0..=search {
        let mut sum = 0.0f32;
        for i in lag..n {
            sum += far[i - lag] * mic[i];
        }
        if sum > best {
            best = sum;
            best_lag = lag;
        }
    }
    best_lag
}
