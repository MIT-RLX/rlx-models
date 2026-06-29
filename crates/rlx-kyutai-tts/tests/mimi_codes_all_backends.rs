//! Mimi code-frame parity: Moshi Python ref, eager CPU, and every RLX backend.
//!
//! ```bash
//! # Moshi reference export (once):
//! scripts/export_kyutai_mimi_codes.py
//!
//! # Eager vs Python (always when weights present):
//! RLX_KYUTAI_TTS_DIR=… cargo test -p rlx-kyutai-tts --test mimi_codes_all_backends --release eager_matches_moshi -- --nocapture
//!
//! # Cross-backend vs eager CPU (opt-in — real weights, slow):
//! RLX_KYUTAI_CODES_PARITY=1 cargo test -p rlx-kyutai-tts --test mimi_codes_all_backends --features all-backends --release -- --nocapture
//! ```

mod backend_common;
mod parity_common;

use anyhow::Result;
use parity_common::{
    PARITY_PROMPT, assert_frames_match, codes_parity_enabled, eager_codes, load_moshi_reference,
    mimi_codes_ref_path, model_dir, parity_gen_cfg, rlx_codes,
};
use rlx_runtime::{Device, is_available};

#[test]
fn eager_matches_moshi_export() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = mimi_codes_ref_path();
    if !path.is_file() {
        eprintln!(
            "skip: missing Moshi reference at {} (run scripts/export_kyutai_mimi_codes.py)",
            path.display()
        );
        return Ok(());
    }
    let reference = load_moshi_reference(&path)?;
    let cfg = parity_gen_cfg();
    let rust = eager_codes(&dir, &cfg, PARITY_PROMPT)?;
    eprintln!(
        "py: {} frames end={:?} delay={} | eager: {} frames",
        reference.trimmed.len(),
        reference.end,
        reference.delay,
        rust.len(),
    );
    assert_frames_match("eager vs moshi", &reference.trimmed, &rust)
}

fn backend_matches_eager(device: Device, label: &str) -> Result<()> {
    if !codes_parity_enabled() {
        eprintln!("skip {label}: set RLX_KYUTAI_CODES_PARITY=1");
        return Ok(());
    }
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip {label}: not available");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = parity_gen_cfg();
    let reference = eager_codes(&dir, &cfg, PARITY_PROMPT)?;
    let actual = rlx_codes(&dir, device, &cfg, PARITY_PROMPT)?;
    assert_frames_match(&format!("{label} vs eager"), &reference, &actual)
}

macro_rules! backend_codes_test {
    ($name:ident, $dev:expr, $label:literal) => {
        #[test]
        fn $name() -> Result<()> {
            backend_matches_eager($dev, $label)
        }
    };
}

backend_codes_test!(rlx_cpu_matches_eager, Device::Cpu, "RLX CPU");
backend_codes_test!(rlx_metal_matches_eager, Device::Metal, "RLX Metal");
backend_codes_test!(rlx_mlx_matches_eager, Device::Mlx, "RLX MLX");
backend_codes_test!(rlx_cuda_matches_eager, Device::Cuda, "RLX CUDA");
backend_codes_test!(rlx_rocm_matches_eager, Device::Rocm, "RLX ROCm");
backend_codes_test!(rlx_wgpu_matches_eager, Device::Gpu, "RLX wgpu");
backend_codes_test!(rlx_vulkan_matches_eager, Device::Vulkan, "RLX Vulkan");

#[test]
fn all_rlx_backends_match_eager_summary() -> Result<()> {
    if !codes_parity_enabled() {
        eprintln!("skip summary: set RLX_KYUTAI_CODES_PARITY=1");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = parity_gen_cfg();
    let reference = eager_codes(&dir, &cfg, PARITY_PROMPT)?;
    let mut tested = 0usize;
    for &(dev, label) in backend_common::BACKENDS {
        if dev == Device::Ane {
            eprintln!("{label}: skipped (ANE not on Kyutai session path)");
            continue;
        }
        if dev != Device::Cpu && !is_available(dev) {
            eprintln!("{label}: skipped (not available)");
            continue;
        }
        let actual = rlx_codes(&dir, dev, &cfg, PARITY_PROMPT)?;
        assert_frames_match(&format!("{label} vs eager (summary)"), &reference, &actual)?;
        tested += 1;
    }
    eprintln!("codes parity: {tested} RLX backend(s) matched eager CPU");
    Ok(())
}
