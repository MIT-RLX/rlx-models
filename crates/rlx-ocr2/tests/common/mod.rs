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

//! Shared helpers for the rlx-ocr2 integration tests.
#![allow(dead_code)] // each test binary uses a different subset

use rlx_runtime::Device;
use std::path::Path;

/// Read a little-endian `f32` blob.
pub fn read_f32(p: impl AsRef<Path>) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine similarity of two equal-length vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-20)
}

/// Max absolute elementwise difference.
pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0., f32::max)
}

/// Devices to sweep, from `OCR2_DEVICES` (comma-separated; default `cpu`).
pub fn devices() -> Vec<(String, Device)> {
    std::env::var("OCR2_DEVICES")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|name| {
            let d = rlx_cli::parse_standard_device("rlx-ocr2", name).expect("device");
            (name.to_string(), d)
        })
        .collect()
}
