//! Per-backend synthetic checks for the Kyutai TTS RLX temporal graph.
//!
//! ```bash
//! cargo test -p rlx-kyutai-tts --test rlx_backends_all --features all-backends
//! ```

mod backend_common;

use backend_common::{BACKENDS, temporal_decode_step0_on_device, temporal_decode_step1_on_device};
use rlx_runtime::Device;

macro_rules! per_backend_tests {
    ($suite:ident, $fn:ident, $suffix:literal) => {
        mod $suite {
            use super::*;

            #[test]
            fn cpu() {
                $fn(Device::Cpu, concat!("CPU", $suffix));
            }
            #[test]
            fn metal() {
                $fn(Device::Metal, concat!("Metal", $suffix));
            }
            #[test]
            fn mlx() {
                $fn(Device::Mlx, concat!("MLX", $suffix));
            }
            #[test]
            fn cuda() {
                $fn(Device::Cuda, concat!("CUDA", $suffix));
            }
            #[test]
            fn rocm() {
                $fn(Device::Rocm, concat!("ROCm", $suffix));
            }
            #[test]
            fn wgpu_gpu() {
                $fn(Device::Gpu, concat!("wgpu/Gpu", $suffix));
            }
            #[test]
            fn vulkan() {
                $fn(Device::Vulkan, concat!("Vulkan", $suffix));
            }
            #[test]
            fn coreml_ane() {
                $fn(Device::Ane, concat!("CoreML/ANE", $suffix));
            }
        }
    };
}

per_backend_tests!(step0, temporal_decode_step0_on_device, " step0");
per_backend_tests!(step1, temporal_decode_step1_on_device, " step1");

#[test]
fn temporal_decode_all_available_backends_summary() {
    let mut tested = 0usize;
    for &(dev, label) in BACKENDS {
        if dev == Device::Cpu || is_available(dev) {
            temporal_decode_step0_on_device(dev, label);
            temporal_decode_step1_on_device(dev, label);
            tested += 1;
        } else {
            eprintln!("{label}: skipped in summary (not available)");
        }
    }
    eprintln!("kyutai temporal: exercised {tested} backend(s) (step0 + step1 each)");
}

use rlx_runtime::is_available;
