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

//! Shared helpers for the env-gated parity tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_runtime::Device;

/// `(model dir, reference dir, device)` when `RLX_FISH_DIR` and
/// `RLX_FISH_REF/<file>` exist, else `None` (the test skips).
pub fn setup(file: &str) -> Option<(PathBuf, PathBuf, Device)> {
    setup_with("RLX_FISH_DIR", file)
}

pub fn setup_with(dir_var: &str, file: &str) -> Option<(PathBuf, PathBuf, Device)> {
    let dir = rlx_ir::env::var(dir_var).map(PathBuf::from)?;
    let reference = rlx_ir::env::var("RLX_FISH_REF").map(PathBuf::from)?;
    if !reference.join(file).is_file() {
        eprintln!("skip: {} missing", reference.join(file).display());
        return None;
    }
    let device = rlx_ir::env::var("RLX_FISH_DEVICE")
        .map(|d| rlx_cli_device(&d))
        .unwrap_or(Device::Cpu);
    Some((dir, reference, device))
}

fn rlx_cli_device(name: &str) -> Device {
    rlx_cli::parse_device(name).expect("RLX_FISH_DEVICE")
}

/// A reference safetensors file.
pub struct Fixture(SafetensorsCheckpoint);

impl Fixture {
    pub fn open(path: &Path) -> Self {
        Self(SafetensorsCheckpoint::open_file(path).expect("open fixture"))
    }

    pub fn f32(&self, key: &str) -> (Vec<f32>, Vec<usize>) {
        self.0.load_tensor_f32(key).expect(key)
    }

    pub fn i64(&self, key: &str) -> (Vec<i64>, Vec<usize>) {
        let (bytes, dtype, shape) = self.0.tensor_raw(key).expect(key);
        assert_eq!(dtype, safetensors::Dtype::I64, "{key}");
        let v = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        (v, shape)
    }
}

/// Max-abs and relative-L2 comparison; panics past `tol` (max-abs).
pub fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut max = 0f32;
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        max = max.max((a - b).abs());
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    eprintln!("{what}: max_abs {max:.3e} rel_l2 {rel:.3e}");
    assert!(max <= tol, "{what}: max_abs {max} > {tol} (rel_l2 {rel})");
}
