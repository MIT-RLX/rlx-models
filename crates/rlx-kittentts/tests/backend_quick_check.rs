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

//! Backend quick-check: session builds on each RLX device when compiled in.

#![cfg(feature = "onnx")]

mod support;

use rlx_kittentts::{Device, is_available, parse_device};

fn run_load_on(device: Device) {
    if !is_available(device) {
        eprintln!("skip rlx-kittentts load_on {device:?}: RLX backend not available in this build");
        return;
    }
    let Some(dir) = support::model_dir() else {
        eprintln!(
            "skip rlx-kittentts load_on {device:?}: set KITTENTTS_MODEL_DIR with model files"
        );
        return;
    };
    support::load_model_on(&dir, device)
        .unwrap_or_else(|e| panic!("load_on {device:?} failed: {e}"));
}

#[test]
fn load_on_cpu() {
    run_load_on(Device::Cpu);
}

#[test]
fn load_on_metal() {
    run_load_on(Device::Metal);
}

#[test]
fn load_on_mlx() {
    run_load_on(Device::Mlx);
}

#[test]
fn load_on_cuda() {
    run_load_on(Device::Cuda);
}

#[test]
fn load_on_rocm() {
    run_load_on(Device::Rocm);
}

#[test]
fn parse_devices() {
    assert_eq!(parse_device("cpu").unwrap(), Device::Cpu);
    assert_eq!(parse_device("metal").unwrap(), Device::Metal);
}
