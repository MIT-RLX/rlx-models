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

//! JFK VAD bench: VAD algorithms × RLX devices, noise sweep, latency + quality.
//!
//! ```sh
//! cargo run -p rlx-vad --example jfk_bench --release
//! cargo run -p rlx-vad --example jfk_bench --release --features all-backends -- --devices all
//! ```

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use rlx_vad::audio::{SAMPLE_RATE_16K, load_wav_mono_f32, resample_linear};
use rlx_vad::{
    SegmentParams, SpeechSegment, bench_device_label, parse_device_list, resolve_device,
    streaming_execution_device,
};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "earshot")]
use rlx_vad::earshot::{Detector as EarshotDetector, FRAME_SAMPLES as EAR_FRAME};
#[cfg(feature = "silero")]
use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};

const SILENCE_PAD_S: f32 = 1.5;
const WARMUP_FRAMES: usize = 8;

fn mix_white_noise_at_snr(pcm: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    if !snr_db.is_finite() {
        return pcm.to_vec();
    }
    let sig_pwr = pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len().max(1) as f32;
    let noise_pwr = sig_pwr / 10f32.powf(snr_db / 10.0);
    let amp = noise_pwr.sqrt();
    let mut state = seed;
    pcm.iter()
        .map(|&s| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f32 / u32::MAX as f32;
            let n = (u * 2.0 - 1.0) * amp;
            (s + n).clamp(-1.0, 1.0)
        })
        .collect()
}

fn build_labeled_pcm(speech: &[f32]) -> (Vec<f32>, usize, usize) {
    let pad = (SILENCE_PAD_S * SAMPLE_RATE_16K as f32) as usize;
    let mut pcm = vec![0.0f32; pad];
    pcm.extend_from_slice(speech);
    pcm.resize(pad + speech.len() + pad, 0.0);
    (pcm, pad, pad + speech.len())
}

struct FrameStats {
    speech_recall: f32,
    silence_specificity: f32,
    frame_accuracy: f32,
    mean_speech_prob: f32,
    seg_iou: f32,
    mean_us: f64,
    p99_us: f64,
    rtf: f64,
}

fn speech_region_iou(segs: &[SpeechSegment], speech_start: usize, speech_end: usize) -> f32 {
    let mut covered = 0usize;
    for seg in segs {
        let s = seg.start.max(speech_start);
        let e = seg.end.min(speech_end);
        if e > s {
            covered += e - s;
        }
    }
    let expected = speech_end.saturating_sub(speech_start).max(1);
    (covered as f32 / expected as f32).min(1.0)
}

#[cfg(feature = "earshot")]
fn score_earshot(
    pcm: &[f32],
    speech_start: usize,
    speech_end: usize,
    params: &SegmentParams,
    _device: Device,
) -> FrameStats {
    let exec = streaming_execution_device(_device);
    assert_eq!(exec, Device::Cpu);

    let mut det = EarshotDetector::default();
    let threshold = params.threshold;
    let neg = params.neg_threshold();
    let mut lat_us = Vec::new();
    let mut speech_hits = 0usize;
    let mut speech_total = 0usize;
    let mut silence_hits = 0usize;
    let mut silence_total = 0usize;
    let mut speech_prob_sum = 0.0f32;
    let mut speech_prob_n = 0usize;

    for (fi, chunk) in pcm.chunks(EAR_FRAME).enumerate() {
        let mut frame = [0.0f32; EAR_FRAME];
        frame[..chunk.len()].copy_from_slice(chunk);
        let p = if fi < WARMUP_FRAMES {
            det.predict_f32(&frame)
        } else {
            let t0 = Instant::now();
            let prob = det.predict_f32(&frame);
            lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
            prob
        };

        let sample_off = fi * EAR_FRAME;
        let in_speech = sample_off + EAR_FRAME / 2 >= speech_start && sample_off < speech_end;
        if in_speech {
            speech_total += 1;
            speech_prob_sum += p;
            speech_prob_n += 1;
            if p >= threshold {
                speech_hits += 1;
            }
        } else {
            silence_total += 1;
            if p < neg {
                silence_hits += 1;
            }
        }
    }

    let segs = rlx_vad::speech_segments_earshot(pcm, params);
    finish_stats(
        lat_us,
        pcm.len(),
        speech_hits,
        speech_total,
        silence_hits,
        silence_total,
        speech_prob_sum,
        speech_prob_n,
        speech_region_iou(&segs, speech_start, speech_end),
    )
}

#[cfg(feature = "silero")]
fn score_silero(
    session: &mut SileroSession,
    pcm: &[f32],
    speech_start: usize,
    speech_end: usize,
    params: &SegmentParams,
    _device: Device,
) -> Result<FrameStats> {
    let exec = streaming_execution_device(_device);
    assert_eq!(exec, Device::Cpu);

    let hop = session.frame_samples();
    session.reset();
    let threshold = params.threshold;
    let neg = params.neg_threshold();
    let mut lat_us = Vec::new();
    let mut speech_hits = 0usize;
    let mut speech_total = 0usize;
    let mut silence_hits = 0usize;
    let mut silence_total = 0usize;
    let mut speech_prob_sum = 0.0f32;
    let mut speech_prob_n = 0usize;

    for (fi, chunk) in pcm.chunks(hop).enumerate() {
        let p = if fi < WARMUP_FRAMES {
            session.predict_frame_padded(chunk)?
        } else {
            let t0 = Instant::now();
            let prob = session.predict_frame_padded(chunk)?;
            lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
            prob
        };

        let sample_off = fi * hop;
        let in_speech = sample_off + hop / 2 >= speech_start && sample_off < speech_end;
        if in_speech {
            speech_total += 1;
            speech_prob_sum += p;
            speech_prob_n += 1;
            if p >= threshold {
                speech_hits += 1;
            }
        } else {
            silence_total += 1;
            if p < neg {
                silence_hits += 1;
            }
        }
    }

    let segs = rlx_vad::speech_segments_silero(session, pcm, params)?;
    Ok(finish_stats(
        lat_us,
        pcm.len(),
        speech_hits,
        speech_total,
        silence_hits,
        silence_total,
        speech_prob_sum,
        speech_prob_n,
        speech_region_iou(&segs, speech_start, speech_end),
    ))
}

fn finish_stats(
    mut lat_us: Vec<f64>,
    pcm_len: usize,
    speech_hits: usize,
    speech_total: usize,
    silence_hits: usize,
    silence_total: usize,
    speech_prob_sum: f32,
    speech_prob_n: usize,
    seg_iou: f32,
) -> FrameStats {
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_us = lat_us.iter().sum::<f64>() / lat_us.len().max(1) as f64;
    let p99_us = if lat_us.is_empty() {
        0.0
    } else {
        lat_us[lat_us.len() * 99 / 100.min(lat_us.len().saturating_sub(1))]
    };
    let wall_s = lat_us.iter().sum::<f64>() / 1e6;
    let audio_s = pcm_len as f64 / SAMPLE_RATE_16K as f64;
    FrameStats {
        speech_recall: speech_hits as f32 / speech_total.max(1) as f32,
        silence_specificity: silence_hits as f32 / silence_total.max(1) as f32,
        frame_accuracy: (speech_hits + silence_hits) as f32
            / (speech_total + silence_total).max(1) as f32,
        mean_speech_prob: speech_prob_sum / speech_prob_n.max(1) as f32,
        seg_iou,
        mean_us,
        p99_us,
        rtf: if audio_s > 0.0 { wall_s / audio_s } else { 0.0 },
    }
}

fn print_row(clip: &str, vad: &str, device: &str, snr: &str, s: &FrameStats) {
    println!(
        "{:<18} {:<7} {:<6} {:>6}  acc={:.1}% rec={:.1}% spec={:.1}% iou={:.2} p={:.3}  \
         {:.0}us p99={:.0}us rtf={:.5}",
        clip,
        vad,
        device,
        snr,
        s.frame_accuracy * 100.0,
        s.speech_recall * 100.0,
        s.silence_specificity * 100.0,
        s.seg_iou,
        s.mean_speech_prob,
        s.mean_us,
        s.p99_us,
        s.rtf,
    );
}

fn load_speech_16k(path: &Path) -> Result<Vec<f32>> {
    let (sr, pcm) = load_wav_mono_f32(path).with_context(|| format!("load {}", path.display()))?;
    Ok(if sr == SAMPLE_RATE_16K {
        pcm
    } else {
        resample_linear(&pcm, sr, SAMPLE_RATE_16K)
    })
}

fn parse_args() -> Result<Vec<Device>> {
    let mut devices = vec![Device::Cpu];
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--devices" {
            devices = parse_device_list(args.get(i + 1).context("--devices needs value")?)?;
            i += 2;
            continue;
        }
        if args[i] == "--device" {
            devices = vec![resolve_device(
                args.get(i + 1).context("--device needs value")?,
            )?];
            i += 2;
            continue;
        }
        if args[i].starts_with('-') {
            bail!("unknown flag {} (supported: --devices, --device)", args[i]);
        }
        i += 1;
    }
    Ok(devices)
}

#[cfg(feature = "silero")]
fn silero_session() -> SileroSession {
    SileroSession::new(SileroWeights::embedded(), SileroConfig::default())
}

fn bench_file(
    wav: &Path,
    devices: &[Device],
    #[cfg(feature = "silero")] silero: &mut SileroSession,
) -> Result<()> {
    let speech = load_speech_16k(wav)?;
    let (clean, speech_start, speech_end) = build_labeled_pcm(&speech);
    eprintln!(
        "\n=== {} ({:.1}s speech + {:.1}s silence pads) ===",
        wav.file_name().unwrap().to_string_lossy(),
        speech.len() as f64 / SAMPLE_RATE_16K as f64,
        SILENCE_PAD_S
    );

    let snr_levels: &[(&str, f32)] = &[
        ("clean", f32::INFINITY),
        ("20dB", 20.0),
        ("10dB", 10.0),
        ("5dB", 5.0),
        ("0dB", 0.0),
        ("-5dB", -5.0),
    ];

    println!(
        "{:<18} {:<7} {:<6} {:>6}  acc/rec/spec + seg IoU | latency (streaming on CPU BLAS)",
        "clip", "vad", "device", "SNR"
    );
    println!("{}", "-".repeat(108));

    let clip = wav.file_name().unwrap().to_string_lossy();
    for device in devices {
        let dev_label = bench_device_label(*device);
        resolve_device(dev_label)?;

        for (snr_label, snr) in snr_levels {
            let pcm = mix_white_noise_at_snr(&clean, *snr, 0xC0FFEE);
            #[cfg(feature = "earshot")]
            {
                let params = SegmentParams::earshot();
                let es = score_earshot(&pcm, speech_start, speech_end, &params, *device);
                print_row(&clip, "earshot", dev_label, snr_label, &es);
            }
            #[cfg(feature = "silero")]
            {
                let params = SegmentParams::silero();
                let sl = score_silero(silero, &pcm, speech_start, speech_end, &params, *device)?;
                let row_clip = if cfg!(feature = "earshot") {
                    ""
                } else {
                    clip.as_ref()
                };
                print_row(row_clip, "silero", dev_label, snr_label, &sl);
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    if rlx_vad::enabled_backends().is_empty() {
        bail!("no VAD backends enabled (use `--features earshot` and/or `silero`)");
    }

    let devices = parse_args()?;
    eprintln!(
        "VAD: {} | RLX devices: {}",
        rlx_vad::enabled_backends().join(", "),
        devices
            .iter()
            .map(|d| bench_device_label(*d))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("Frame inference: CPU BLAS (validated per --device slot)");

    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/jfk");
    let mut files = vec![
        assets.join("jfk_rust_speech.wav"),
        assets.join("jfk_voice_clone.wav"),
    ];
    if let Ok(one) = env::var("RLX_VAD_JFK_WAV") {
        files = vec![one.into()];
    }
    files.retain(|p| p.is_file());
    if files.is_empty() {
        bail!("no JFK wav under assets/jfk (or RLX_VAD_JFK_WAV)");
    }

    #[cfg(feature = "silero")]
    let mut silero_session = silero_session();

    for wav in &files {
        bench_file(
            wav,
            &devices,
            #[cfg(feature = "silero")]
            &mut silero_session,
        )?;
    }

    Ok(())
}
