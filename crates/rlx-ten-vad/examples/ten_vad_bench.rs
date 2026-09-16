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

//! Streaming vs batched throughput per backend, and the fixture dumper.
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example ten_vad_bench -- --devices all
//! cargo run -p rlx-ten-vad --release --example ten_vad_bench -- --dump-pcm out.pcm
//! ```

use anyhow::Result;
use rlx_ten_vad::device::{device_label, parse_device_list};
use rlx_ten_vad::frontend::{Frontend, pre_emphasis};
use rlx_ten_vad::model::{Shape, TenVadModel};
use rlx_ten_vad::session::{TenVad, TenVadBatch, TenVadConfig};
use rlx_ten_vad::{CONTEXT_FRAMES, FEATURE_LEN, TenVadWeights};
use rlx_ten_vad::{HOP_SIZE, SAMPLE_RATE, synth};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut devices = "cpu".to_string();
    let mut dump: Option<String> = None;
    let mut wav: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--devices" => {
                devices = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--dump-pcm" => {
                dump = args.get(i + 1).cloned();
                i += 2;
            }
            "--wav" => {
                wav = args.get(i + 1).cloned();
                i += 2;
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

    if let Some(path) = dump {
        let pcm = synth::speech_like_clip();
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(&path, &bytes)?;
        println!(
            "wrote {} samples ({} bytes) to {path}",
            pcm.len(),
            bytes.len()
        );
        return Ok(());
    }

    let pcm = match wav {
        Some(path) => {
            let (p, rate) = rlx_core::asr_bench::read_wav_mono(std::path::Path::new(&path))?;
            let p = if rate as usize == SAMPLE_RATE {
                p
            } else {
                rlx_core::resample_linear(&p, rate, SAMPLE_RATE as u32)
            };
            rlx_ten_vad::to_int16_scale(&p)
        }
        None => synth::speech_like_clip_scaled(),
    };
    let audio_s = pcm.len() as f64 / SAMPLE_RATE as f64;
    let frames = pcm.len() / HOP_SIZE;
    println!("clip: {audio_s:.2}s, {frames} frames\n");
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "device", "stream s", "stream RTF", "batch s", "batch RTF"
    );

    // Chunk-size sweep. Chunking trades `k · 16 ms` of latency for throughput,
    // and matters most on GPU where one 16 ms frame is pure launch latency.
    // Network only — the DSP frontend is per-frame and identical across devices.
    let feats = {
        let mut fe = Frontend::new(TenVadWeights::embedded().core());
        let mut emph = vec![0.0f32; HOP_SIZE];
        let mut prev = 0.0f32;
        let mut v = Vec::with_capacity(frames * CONTEXT_FRAMES * FEATURE_LEN);
        for raw in pcm.chunks_exact(HOP_SIZE) {
            pre_emphasis(raw, &mut prev, &mut emph);
            fe.push(raw, &emph);
            v.extend_from_slice(fe.context());
        }
        v
    };
    let stride = CONTEXT_FRAMES * FEATURE_LEN;
    const CHUNKS: [usize; 5] = [1, 4, 8, 16, 32];
    println!("\nnetwork-only RTF by frames/dispatch (latency = k · 16 ms)");
    print!("{:<8}", "device");
    for k in CHUNKS {
        print!("{:>10}", format!("k={k}"));
    }
    println!();
    for dev in parse_device_list(&devices)? {
        print!("{:<8}", device_label(dev));
        for k in CHUNKS {
            let mut m = TenVadModel::new(dev, Shape::Chunk(k), TenVadWeights::embedded())?;
            let usable = (frames / k) * k;
            let _ = m.run_batch(&feats[..k * stride])?; // warm the compile out
            let t = std::time::Instant::now();
            for c in feats[..usable * stride].chunks_exact(k * stride) {
                let _ = m.run_batch(c)?;
            }
            print!("{:>10.0}", audio_s / t.elapsed().as_secs_f64());
        }
        println!();
    }
    println!();

    let mut reference: Option<Vec<f32>> = None;
    for dev in parse_device_list(&devices)? {
        let cfg = TenVadConfig {
            device: dev,
            ..Default::default()
        };
        let mut vad = TenVad::new(cfg)?;
        let t0 = std::time::Instant::now();
        let streamed: Vec<f32> = pcm
            .chunks_exact(HOP_SIZE)
            .map(|c| vad.process_scaled(c).map(|f| f.probability))
            .collect::<Result<_>>()?;
        let stream_s = t0.elapsed().as_secs_f64();

        let mut batch = TenVadBatch::new(dev)?;
        // Warm the compile out of the timing.
        let _ = batch.probabilities(&pcm[..HOP_SIZE * frames.min(8)])?;
        let t1 = std::time::Instant::now();
        let batched = batch.probabilities(&pcm)?;
        let batch_s = t1.elapsed().as_secs_f64();

        println!(
            "{:<8} {stream_s:>10.3} {:>10.1} {batch_s:>10.3} {:>10.1}",
            device_label(dev),
            audio_s / stream_s,
            audio_s / batch_s
        );
        let drift = streamed
            .iter()
            .zip(&batched)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("         streaming vs batched on this device: max|Δ| {drift:.2e}");
        match &reference {
            None => reference = Some(streamed),
            Some(r) => {
                let d = r
                    .iter()
                    .zip(&streamed)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("         vs first device:                  max|Δ| {d:.2e}");
            }
        }
    }
    Ok(())
}
