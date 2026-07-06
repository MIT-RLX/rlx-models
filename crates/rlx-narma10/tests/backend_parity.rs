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

//! NARMA-10 host reference vs compiled RLX graph on each backend.

use rlx_narma10::{generate, generate_on_device, max_abs_diff};
use rlx_runtime::{Device, is_available};

const N: usize = 256;
const SEED: u64 = 0x4e_41_52_4d_41; // "NARMA" in ASCII
const TOL: f64 = 1e-4;

fn assert_backend_matches_cpu_reference(device: Device) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip NARMA-10 on {device:?}: RLX backend not available in this build");
        return;
    }
    let host = generate(N, SEED);
    let got = generate_on_device(device, N, SEED)
        .unwrap_or_else(|e| panic!("generate_on_device({device:?}): {e}"));
    let err = max_abs_diff(&host.targets, &got.targets);
    assert!(
        err < TOL,
        "NARMA-10 mismatch on {device:?}: max |Δ| = {err:e} (tol {TOL:e})"
    );
}

#[test]
fn narma10_on_cpu() {
    assert_backend_matches_cpu_reference(Device::Cpu);
}

#[test]
fn narma10_on_metal() {
    assert_backend_matches_cpu_reference(Device::Metal);
}

#[test]
fn narma10_on_mlx() {
    assert_backend_matches_cpu_reference(Device::Mlx);
}

#[test]
fn narma10_on_cuda() {
    assert_backend_matches_cpu_reference(Device::Cuda);
}

#[test]
fn narma10_on_rocm() {
    assert_backend_matches_cpu_reference(Device::Rocm);
}

#[test]
fn narma10_on_wgpu() {
    assert_backend_matches_cpu_reference(Device::Gpu);
}

#[test]
fn narma10_on_vulkan() {
    assert_backend_matches_cpu_reference(Device::Vulkan);
}
