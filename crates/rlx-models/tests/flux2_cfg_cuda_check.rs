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

//! FLUX.2 CFG combine HIR on CUDA (`cuda` feature).

#[cfg(feature = "cuda")]
use rlx_models::flux2::{cfg_combine, compile_flux2_cfg_combine};
#[cfg(feature = "cuda")]
use rlx_runtime::Device;

#[cfg(feature = "cuda")]
#[test]
fn cfg_combine_runs_on_cuda() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("skip: CUDA not available");
        return;
    }
    let pos = vec![1.0f32, 2.0, 3.0, 4.0];
    let neg = vec![0.5f32, 1.0, 1.5, 2.0];
    let scale = 2.5f32;

    let native = cfg_combine(&pos, &neg, scale);
    let mut compiled = compile_flux2_cfg_combine(1, 2, 2, Device::Cuda, None).unwrap();
    let out = compiled
        .run(&[("pos", pos.as_slice()), ("neg", neg.as_slice())])
        .remove(0);
    assert_eq!(out.len(), native.len());
    let max_diff = native
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "CUDA CFG max_abs_diff={max_diff}");
}
