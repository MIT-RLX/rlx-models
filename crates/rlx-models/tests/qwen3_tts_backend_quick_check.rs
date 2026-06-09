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

//! Talker prefill+decode on each backend (real 0.6B weights).
//!
//!   RLX_QWEN3_TTS_DIR=... cargo test -p rlx-models --test qwen3_tts_backend_quick_check --features all-backends
//!   just features=all-backends test-qwen3-tts-backends

use rlx_qwen3_tts::Qwen3TtsRunnerBuilder;
use rlx_runtime::Device;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_QWEN3_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
}

fn run_talker_bench(device: Device) {
    let Some(dir) = model_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let runner = Qwen3TtsRunnerBuilder::default()
        .model_dir(&dir)
        .device(device)
        .build()
        .expect("runner");
    let report = runner.bench_talker_synthetic(64, 8).expect("bench");
    assert!(report.talker_prefill_ms > 0.0);
    assert!(report.talker_decode_ms > 0.0);
}

macro_rules! backend_test {
    ($name:ident, $dev:expr, $feat:meta) => {
        #[cfg($feat)]
        #[test]
        fn $name() {
            if !rlx_runtime::is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            run_talker_bench($dev);
        }
    };
}

#[test]
fn qwen3_tts_talker_cpu() {
    run_talker_bench(Device::Cpu);
}

backend_test!(
    qwen3_tts_talker_metal,
    Device::Metal,
    all(target_os = "macos", feature = "metal")
);
backend_test!(
    qwen3_tts_talker_mlx,
    Device::Mlx,
    all(target_os = "macos", feature = "mlx")
);
backend_test!(qwen3_tts_talker_cuda, Device::Cuda, feature = "cuda");
backend_test!(qwen3_tts_talker_rocm, Device::Rocm, feature = "rocm");
backend_test!(qwen3_tts_talker_wgpu, Device::Gpu, feature = "gpu");
backend_test!(qwen3_tts_talker_vulkan, Device::Vulkan, feature = "vulkan");
