//! Greedy SNAC code parity: CPU synthesis reference vs every RLX LM backend.
//!
//! Default compares **dynamic decode** (same as CPU [`BackboneLoadOptions::synthesis`])
//! on each accelerator. Production bucket decode is opt-in:
//!
//! ```bash
//! just fetch-orpheus fetch-orpheus-snac export-orpheus-snac
//! export ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors
//!
//! # Kernel parity (Metal default on macOS):
//! cargo test -p rlx-orpheus --test backends_codes_parity --features all-backends --release
//!
//! # Production for_tts bucket decode (known divergences on short prompts):
//! ORPHEUS_FOR_TTS_PARITY=1 cargo test -p rlx-orpheus --test backends_codes_parity --features all-backends --release
//!
//! # All available accelerators:
//! ORPHEUS_CODES_PARITY=1 cargo test -p rlx-orpheus --test backends_codes_parity --features all-backends --release -- --nocapture
//! ```

mod backend_common;
mod support;

use rlx_orpheus::lm_kv_decode_supported;
use rlx_runtime::{Device, is_available};
use support::{bench_max_tokens, bench_text, bench_voice, greedy_codes_on_device, require_weights};

fn codes_parity_all() -> bool {
    std::env::var("ORPHEUS_CODES_PARITY").ok().as_deref() == Some("1")
}

fn assert_codes_match(label: &str, reference: &[i32], actual: &[i32]) {
    let n = reference.len().min(actual.len());
    for (i, (r, a)) in reference.iter().zip(actual.iter()).take(n).enumerate() {
        assert_eq!(
            r, a,
            "{label}: code mismatch at index {i} (ref={r} got={a})"
        );
    }
    assert_eq!(
        reference.len(),
        actual.len(),
        "{label}: code count ref {} vs got {}",
        reference.len(),
        actual.len()
    );
    eprintln!(
        "{label}: all {} greedy codes match CPU reference",
        reference.len()
    );
}

fn parity_on_device(device: Device, label: &str) -> anyhow::Result<()> {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip {label}: not available");
        return Ok(());
    }
    if device != Device::Cpu && !lm_kv_decode_supported(device) {
        eprintln!("skip {label}: LM KV decode not enabled (set ORPHEUS_MLX_KV=1 for MLX)");
        return Ok(());
    }
    let Some((gguf, _snac)) = require_weights() else {
        eprintln!("skip {label}: missing weights — run `just fetch-orpheus`");
        return Ok(());
    };
    let text = bench_text();
    let voice = bench_voice();
    let max_tokens = bench_max_tokens();
    let reference = greedy_codes_on_device(&gguf, Device::Cpu, &text, &voice, max_tokens)?;
    if device == Device::Cpu {
        assert!(
            reference.len() >= 7,
            "CPU reference produced too few codes: {}",
            reference.len()
        );
        eprintln!("{label}: {} codes (reference)", reference.len());
        return Ok(());
    }
    let actual = greedy_codes_on_device(&gguf, device, &text, &voice, max_tokens)?;
    eprintln!(
        "{label}: ref={} got={} head_ref={:?} head_got={:?}",
        reference.len(),
        actual.len(),
        &reference[..reference.len().min(8)],
        &actual[..actual.len().min(8)]
    );
    assert_codes_match(label, &reference, &actual);
    Ok(())
}

macro_rules! backend_codes_parity_test {
    ($name:ident, $dev:expr, $label:literal) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            let quick = $dev == Device::Metal
                || ($dev == Device::Mlx
                    && std::env::var("ORPHEUS_MLX_KV").ok().as_deref() == Some("1"));
            if $dev != Device::Cpu && !codes_parity_all() && !quick {
                eprintln!(
                    "skip {}: set ORPHEUS_CODES_PARITY=1 for full matrix",
                    $label
                );
                return Ok(());
            }
            parity_on_device($dev, $label)
        }
    };
}

backend_codes_parity_test!(greedy_codes_cpu_reference, Device::Cpu, "CPU");
backend_codes_parity_test!(greedy_codes_metal_matches_cpu, Device::Metal, "Metal");
backend_codes_parity_test!(greedy_codes_mlx_matches_cpu, Device::Mlx, "MLX");
backend_codes_parity_test!(greedy_codes_cuda_matches_cpu, Device::Cuda, "CUDA");
backend_codes_parity_test!(greedy_codes_rocm_matches_cpu, Device::Rocm, "ROCm");
backend_codes_parity_test!(greedy_codes_wgpu_matches_cpu, Device::Gpu, "wgpu");
backend_codes_parity_test!(greedy_codes_vulkan_matches_cpu, Device::Vulkan, "Vulkan");

#[test]
fn all_backends_greedy_codes_match_cpu_summary() -> anyhow::Result<()> {
    if !codes_parity_all() {
        eprintln!("skip summary: set ORPHEUS_CODES_PARITY=1");
        return Ok(());
    }
    let mut tested = 0usize;
    for &(dev, label) in backend_common::BACKENDS {
        if dev != Device::Cpu {
            if !is_available(dev) || !lm_kv_decode_supported(dev) {
                eprintln!("{label}: skipped");
                continue;
            }
        }
        parity_on_device(dev, label)?;
        tested += 1;
    }
    eprintln!("codes parity: {tested} backend(s) matched CPU reference");
    Ok(())
}
