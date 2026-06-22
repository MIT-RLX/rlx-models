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

//! Synthetic echo helpers for tests and benches.

/// `mic = clean + alpha * far[i - delay]` (zero-padded).
pub fn simple_delayed_echo(clean: &[f32], far: &[f32], delay: usize, alpha: f32) -> Vec<f32> {
    let n = clean.len().max(far.len() + delay);
    let mut mic = vec![0.0f32; n];
    mic[..clean.len()].copy_from_slice(clean);
    for i in 0..far.len() {
        let t = i + delay;
        if t < n {
            mic[t] += alpha * far[i];
        }
    }
    mic.truncate(clean.len());
    mic
}

pub fn apply_echo(clean: &[f32], far: &[f32], delay: usize, alpha: f32) -> Vec<f32> {
    simple_delayed_echo(clean, far, delay, alpha)
}
