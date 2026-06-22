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

//! Greedy JFK transcript parity: paired reference vs CPU and each GPU backend.
//!
//! ```sh
//! just test-whisper-all-backends
//! ```

#![cfg(any(
    feature = "metal",
    feature = "mlx",
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]

use anyhow::Result;
use rlx_models::whisper::{
    WhisperRunner, assert_transcript_matches_reference, ensure_jfk_fixture, load_wav_mono_f32,
};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn tiny_dir() -> PathBuf {
    std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny")
        })
}

fn gpu_backends() -> Vec<Device> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        out.push(Device::Cuda);
    }
    #[cfg(feature = "metal")]
    if is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(feature = "mlx")]
    if is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    #[cfg(feature = "rocm")]
    if is_available(Device::Rocm) {
        out.push(Device::Rocm);
    }
    #[cfg(feature = "gpu")]
    if is_available(Device::Gpu) {
        out.push(Device::Gpu);
    }
    #[cfg(feature = "vulkan")]
    if is_available(Device::Vulkan) {
        out.push(Device::Vulkan);
    }
    out
}

fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

#[test]
fn whisper_jfk_greedy_matches_reference_on_all_backends() -> Result<()> {
    let backends = gpu_backends();
    if backends.is_empty() {
        eprintln!("skip: no GPU backend available");
        return Ok(());
    }

    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("OPENBLAS_NUM_THREADS", "1");

    let dir = tiny_dir();
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        eprintln!("skip: need weights (just fetch-whisper)");
        return Ok(());
    }
    let (wav, reference) = match ensure_jfk_fixture() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip: {e}");
            return Ok(());
        }
    };
    let pcm = load_wav_mono_f32(&wav)?;

    let mut cpu = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let cpu_text = cpu.transcribe_greedy(&pcm)?;
    assert_transcript_matches_reference(&cpu_text, &reference);

    for dev in backends {
        let mut runner = WhisperRunner::builder()
            .weights(&weights)
            .config_path(dir.join("config.json"))
            .tokenizer_path(dir.join("tokenizer.json"))
            .device(dev)
            .language("en")
            .build()?;
        let dev_text = runner.transcribe_greedy(&pcm)?;
        eprintln!(
            "{} enc={:?} dec={:?}: {dev_text:?}",
            device_label(dev),
            runner.encoder_device(),
            runner.decode_device(),
        );
        assert_transcript_matches_reference(&dev_text, &reference);
        assert_transcript_matches_reference(&dev_text, &cpu_text);
    }
    Ok(())
}
