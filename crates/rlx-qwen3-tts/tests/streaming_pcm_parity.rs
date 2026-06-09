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

//! Streaming PCM checks: batched lossless chunking; progressive matches one-shot decode.
//!
//! Run:
//!   cargo test -p rlx-qwen3-tts --test streaming_pcm_parity --features all-backends --release
//!   RLX_QWEN3_TTS_DIR=... just features=all-backends test-qwen3-tts-streaming

use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
use rlx_runtime::Device;

const TEXT: &str = "The capital of France is Paris.";
const MIN_SAMPLES: usize = 12_000;
const MIN_PEAK: f32 = 0.05;

fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf)
        .unwrap_or(manifest)
}

fn model_dir() -> Option<std::path::PathBuf> {
    let candidates = [
        std::env::var("RLX_QWEN3_TTS_DIR")
            .ok()
            .map(std::path::PathBuf::from),
        Some(workspace_root().join(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").exists())
}

fn ref_wav() -> Option<std::path::PathBuf> {
    let p = workspace_root().join("assets/jfk/jfk_voice_clone.wav");
    p.exists().then_some(p)
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0f32, f32::max)
}

fn collect_stream_pcm(
    tts: &mut VoiceClone,
    reference: &rlx_qwen3_tts::SpeakerReference,
    config: StreamConfig,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let mut pcm = Vec::new();
    let stats = tts.generate_stream(reference, TEXT, config, |evt| {
        if let StreamEvent::Pcm(chunk) = evt {
            pcm.extend_from_slice(&chunk.samples);
        }
        StreamControl::Continue
    })?;
    Ok((pcm, stats.samples_emitted))
}

fn check_streaming_on_device(device: Device) -> anyhow::Result<()> {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: no Qwen3-TTS weights");
        return Ok(());
    };
    let Some(ref_wav) = ref_wav() else {
        eprintln!("skip: assets/jfk/jfk_voice_clone.wav");
        return Ok(());
    };

    let mut tts = VoiceClone::open_with_max_frames(&model_dir, device, 128)?;
    let reference = tts.extract_reference(&ref_wav)?;

    for config in [
        StreamConfig::batched().with_chunk_samples(1_200),
        StreamConfig::progressive(4).with_chunk_samples(1_200),
        StreamConfig::progressive(16).with_chunk_samples(480),
    ] {
        let label = format!("{config:?}");
        let (streamed, samples_emitted) = collect_stream_pcm(&mut tts, &reference, config)?;
        assert_eq!(
            streamed.len(),
            samples_emitted,
            "chunk emission must be lossless on {:?} ({label})",
            device
        );
        assert!(
            streamed.len() > MIN_SAMPLES,
            "stream too short on {:?} ({label})",
            device
        );
        assert!(
            peak(&streamed) > MIN_PEAK,
            "stream too quiet on {:?} ({label})",
            device
        );
    }

    // Progressive PCM parity vs one-shot decode of the same codec frames is enforced
    // inside `VoiceClone::run_progressive_parallel` (see `ensure_progressive_pcm_matches`).

    Ok(())
}

macro_rules! backend_test {
    ($name:ident, $dev:expr, $feat:meta) => {
        #[cfg($feat)]
        #[test]
        fn $name() -> anyhow::Result<()> {
            if !rlx_runtime::is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return Ok(());
            }
            check_streaming_on_device($dev)
        }
    };
}

#[test]
fn streaming_pcm_cpu() -> anyhow::Result<()> {
    check_streaming_on_device(Device::Cpu)
}

backend_test!(
    streaming_pcm_metal,
    Device::Metal,
    all(target_os = "macos", feature = "metal")
);
backend_test!(
    streaming_pcm_mlx,
    Device::Mlx,
    all(target_os = "macos", feature = "mlx")
);
backend_test!(streaming_pcm_cuda, Device::Cuda, feature = "cuda");
backend_test!(streaming_pcm_rocm, Device::Rocm, feature = "rocm");
backend_test!(streaming_pcm_wgpu, Device::Gpu, feature = "gpu");
backend_test!(streaming_pcm_vulkan, Device::Vulkan, feature = "vulkan");
