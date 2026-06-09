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

//! Backend quick-check: each RLX device runs both VAD algorithms on JFK when available.

use rlx_runtime::{Device, is_available};
use rlx_vad::{SegmentParams, bench_device_label, resolve_device, streaming_execution_device};

#[cfg(feature = "silero")]
use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};
#[cfg(feature = "earshot")]
use rlx_vad::speech_segments_earshot;

fn jfk_pcm() -> Option<Vec<f32>> {
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/jfk/jfk_rust_speech.wav");
    if !wav.is_file() {
        return None;
    }
    let (sr, pcm) = rlx_vad::load_wav_mono_f32(&wav).expect("wav");
    Some(if sr == rlx_vad::SAMPLE_RATE_16K {
        pcm
    } else {
        rlx_vad::resample_linear(&pcm, sr, rlx_vad::SAMPLE_RATE_16K)
    })
}

fn run_on_device(device: Device) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip rlx-vad on {device:?}: RLX backend not available in this build");
        return;
    }
    resolve_device(bench_device_label(device)).unwrap_or_else(|e| {
        panic!("resolve {device:?}: {e}");
    });
    let exec = streaming_execution_device(device);
    assert_eq!(exec, Device::Cpu);

    let pcm = match jfk_pcm() {
        Some(p) => p,
        None => {
            eprintln!("skip rlx-vad on {device:?}: assets/jfk missing");
            return;
        }
    };

    #[cfg(feature = "earshot")]
    {
        let segs = speech_segments_earshot(&pcm, &SegmentParams::earshot());
        assert!(
            !segs.is_empty(),
            "earshot on {device:?} produced no segments"
        );
    }

    #[cfg(feature = "silero")]
    {
        let mut session = SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
        let segs = rlx_vad::speech_segments_silero(&mut session, &pcm, &SegmentParams::silero())
            .unwrap_or_else(|e| panic!("silero on {device:?}: {e}"));
        assert!(
            !segs.is_empty(),
            "silero on {device:?} produced no segments"
        );
    }
}

#[test]
fn vad_on_cpu() {
    run_on_device(Device::Cpu);
}

#[test]
fn vad_on_metal() {
    run_on_device(Device::Metal);
}

#[test]
fn vad_on_mlx() {
    run_on_device(Device::Mlx);
}

#[test]
fn vad_on_cuda() {
    run_on_device(Device::Cuda);
}

#[test]
fn vad_on_rocm() {
    run_on_device(Device::Rocm);
}

#[test]
fn vad_on_wgpu() {
    run_on_device(Device::Gpu);
}

#[test]
fn vad_on_vulkan() {
    run_on_device(Device::Vulkan);
}

#[test]
fn prob_parity_across_device_slots() {
    let pcm = match jfk_pcm() {
        Some(p) => p,
        None => return,
    };

    #[cfg(feature = "silero")]
    {
        let mut ref_session =
            SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
        let hop = ref_session.frame_samples();
        let mut ref_probs = Vec::new();
        for chunk in pcm.chunks(hop).take(20) {
            ref_probs.push(ref_session.predict_frame_padded(chunk).unwrap());
        }

        for device in [
            Device::Cpu,
            Device::Metal,
            Device::Mlx,
            Device::Cuda,
            Device::Rocm,
            Device::Gpu,
            Device::Vulkan,
        ] {
            if device != Device::Cpu && !is_available(device) {
                continue;
            }
            resolve_device(bench_device_label(device)).unwrap();
            let mut session =
                SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
            for (chunk, &expected) in pcm.chunks(hop).take(20).zip(ref_probs.iter()) {
                let got = session.predict_frame_padded(chunk).unwrap();
                assert!(
                    (got - expected).abs() < 1e-6,
                    "silero prob mismatch on {device:?}: got {got}, expected {expected}"
                );
            }
        }
    }
}
