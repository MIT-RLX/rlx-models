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

// Env-gated: real GGUF heterogeneous batch=2 prefill + decode benchmarks.
//
//   QWEN35_GGUF_PATH=/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//     cargo test -p rlx-models --test qwen35_batch_gguf_bench --features "metal,mlx" --release -- --nocapture
//
// Reports steady-state prefill (batch=1 vs heterogeneous batch=2) and decode
// throughput (per-stream batch=1 vs aggregate batch=2, plus per-row token
// limits). Does not assert absolute SLAs — hardware-dependent numbers for
// Tier C / batch tracking.

#[path = "qwen35_gguf_support.rs"]
mod support;

use rlx_runtime::Device;
use support::{assert_finite_positive, bench_heterogeneous_batch, gguf_path};

macro_rules! het_batch_bench_test {
    ($name:ident, $device:expr) => {
        #[test]
        fn $name() {
            let path = match gguf_path() {
                Some(p) => p,
                None => {
                    eprintln!(
                        "skip qwen35_batch_gguf_bench::{}: set QWEN35_GGUF_PATH",
                        stringify!($name)
                    );
                    return;
                }
            };
            let report = bench_heterogeneous_batch(&path, $device);
            report.log();
            assert_finite_positive(&report);
        }
    };
}

het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_cpu, Device::Cpu);

#[cfg(all(target_os = "macos", feature = "metal"))]
het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_metal, Device::Metal);

#[cfg(all(target_os = "macos", feature = "mlx"))]
het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_mlx, Device::Mlx);

#[cfg(feature = "cuda")]
het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_cuda, Device::Cuda);

#[cfg(feature = "rocm")]
het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_rocm, Device::Rocm);

#[cfg(feature = "gpu")]
het_batch_bench_test!(qwen35_real_gguf_hetero_batch_bench_vulkan, Device::Vulkan);
