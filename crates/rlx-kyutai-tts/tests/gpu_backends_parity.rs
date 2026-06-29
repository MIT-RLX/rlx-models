//! RLX GPU backend codes parity vs RLX CPU (short prompt, real weights).
//!
//! Synthetic graph checks live in `rlx_backends_all`; this file compares full
//! generation on each accelerator against RLX CPU on the same graph path.
//!
//! ```bash
//! RLX_KYUTAI_TTS_DIR=… cargo test -p rlx-kyutai-tts --test gpu_backends_parity --features all-backends --release
//!
//! # Every available GPU backend (slow):
//! RLX_KYUTAI_GPU_PARITY=1 cargo test -p rlx-kyutai-tts --test gpu_backends_parity --features all-backends --release -- --nocapture
//! ```

mod backend_common;
mod parity_common;

use anyhow::Result;
use parity_common::{assert_frames_match, model_dir, rlx_codes, short_gen_cfg};
use rlx_runtime::{Device, is_available};

fn gpu_parity_all() -> bool {
    std::env::var("RLX_KYUTAI_GPU_PARITY").ok().as_deref() == Some("1")
}

fn rlx_matches_cpu(device: Device, label: &str) -> Result<()> {
    if device != Device::Cpu && !gpu_parity_all() && device != Device::Metal {
        eprintln!("skip {label}: set RLX_KYUTAI_GPU_PARITY=1 for full matrix");
        return Ok(());
    }
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip {label}: not available");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = short_gen_cfg();
    let reference = rlx_codes(&dir, Device::Cpu, &cfg, "Hello.")?;
    if device == Device::Cpu {
        eprintln!("{label}: {} frames (reference)", reference.len());
        return Ok(());
    }
    let actual = rlx_codes(&dir, device, &cfg, "Hello.")?;
    assert_frames_match(label, &reference, &actual)
}

macro_rules! gpu_backend_test {
    ($name:ident, $dev:expr, $label:literal) => {
        #[test]
        fn $name() -> Result<()> {
            rlx_matches_cpu($dev, $label)
        }
    };
}

gpu_backend_test!(rlx_cpu_reference, Device::Cpu, "RLX CPU");
gpu_backend_test!(rlx_metal_matches_cpu, Device::Metal, "RLX Metal vs CPU");
gpu_backend_test!(rlx_mlx_matches_cpu, Device::Mlx, "RLX MLX vs CPU");
gpu_backend_test!(rlx_cuda_matches_cpu, Device::Cuda, "RLX CUDA vs CPU");
gpu_backend_test!(rlx_rocm_matches_cpu, Device::Rocm, "RLX ROCm vs CPU");
gpu_backend_test!(rlx_wgpu_matches_cpu, Device::Gpu, "RLX wgpu vs CPU");
gpu_backend_test!(rlx_vulkan_matches_cpu, Device::Vulkan, "RLX Vulkan vs CPU");

#[test]
fn all_gpu_backends_match_rlx_cpu_summary() -> Result<()> {
    if !gpu_parity_all() {
        eprintln!("skip summary: set RLX_KYUTAI_GPU_PARITY=1");
        return Ok(());
    }
    let mut tested = 0usize;
    for &(dev, label) in backend_common::BACKENDS {
        if dev == Device::Ane {
            continue;
        }
        rlx_matches_cpu(dev, label)?;
        tested += 1;
    }
    eprintln!("gpu parity: {tested} RLX backend(s) matched CPU");
    Ok(())
}
