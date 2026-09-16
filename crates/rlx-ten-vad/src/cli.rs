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

//! `rlx-ten-vad` command line.

use anyhow::{Result, bail};
use rlx_cli::req;
use std::path::{Path, PathBuf};

use crate::device::{available_device_labels, resolve_device};
use crate::segments::{SegmentParams, speech_segments};
use crate::session::{TenVad, TenVadBatch, TenVadConfig};
use crate::{DEFAULT_THRESHOLD, HOP_SIZE, SAMPLE_RATE};

const HELP: &str = "\
rlx-ten-vad — TEN-VAD voice activity detection on RLX

  rlx-ten-vad --wav PATH [flags]

  --wav PATH          16 kHz mono WAV (other rates are resampled)
  --device NAME       cpu|metal|mlx|cuda|rocm|gpu|vulkan  (default cpu)
  --threshold F       voice decision threshold          (default 0.5)
  --hop N             API hop in samples, >= 32         (default 256)
  --batch             score the clip with the batched graph (one dispatch
                      per 30 s window) instead of frame by frame
  --frames            print one `time probability flag pitch` line per frame
  --segments          print speech regions (default output)
  --seconds           print segments in seconds rather than samples
  --devices LIST      run every listed device and compare (cpu,metal|all)
";

/// Load a WAV as int16-scaled mono at 16 kHz.
fn load(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(path)?;
    let pcm = if rate as usize == SAMPLE_RATE {
        pcm
    } else {
        rlx_core::resample_linear(&pcm, rate, SAMPLE_RATE as u32)
    };
    Ok(crate::to_int16_scale(&pcm))
}

pub fn run(args: &[String]) -> Result<()> {
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut devices: Option<String> = None;
    let mut threshold = DEFAULT_THRESHOLD;
    let mut hop = HOP_SIZE;
    let mut batch = false;
    let mut show_frames = false;
    let mut seconds = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--devices" => devices = Some(req(args, &mut i)?),
            "--threshold" => {
                threshold = req(args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--threshold: expected f32"))?;
            }
            "--hop" => {
                hop = req(args, &mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--hop: expected usize"))?;
            }
            "--batch" => {
                batch = true;
                i += 1;
            }
            "--frames" => {
                show_frames = true;
                i += 1;
            }
            "--segments" => i += 1,
            "--seconds" => {
                seconds = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "{HELP}\nBackends in this build: {}",
                    available_device_labels().join(", ")
                );
                return Ok(());
            }
            other => bail!("unknown flag {other} (try --help)"),
        }
    }

    let Some(wav) = wav else {
        eprintln!("{HELP}");
        bail!("--wav is required");
    };
    let pcm = load(&wav)?;

    if let Some(list) = devices {
        return compare_devices(&list, &pcm, threshold);
    }
    if batch && hop != HOP_SIZE {
        bail!("--hop only applies to the streaming path; drop --batch or drop --hop");
    }

    let dev = resolve_device(&device)?;
    let probs = if batch {
        TenVadBatch::new(dev)?.probabilities(&pcm)?
    } else {
        let cfg = TenVadConfig {
            hop_size: hop,
            threshold,
            device: dev,
            ..Default::default()
        };
        let mut vad = TenVad::with_weights(cfg, crate::TenVadWeights::embedded())?;
        let mut out = Vec::new();
        for chunk in pcm.chunks_exact(hop) {
            let f = vad.process_scaled(chunk)?;
            if show_frames {
                println!(
                    "{:.3} {:.6} {} {:.1}",
                    out.len() as f64 * hop as f64 / SAMPLE_RATE as f64,
                    f.probability,
                    u8::from(f.voice),
                    f.pitch_hz
                );
            }
            out.push(f.probability);
        }
        out
    };

    if show_frames && batch {
        for (i, p) in probs.iter().enumerate() {
            println!(
                "{:.3} {p:.6} {}",
                i as f64 * HOP_SIZE as f64 / SAMPLE_RATE as f64,
                u8::from(*p > threshold)
            );
        }
    }
    if show_frames {
        return Ok(());
    }

    let params = SegmentParams {
        threshold,
        ..Default::default()
    };
    for seg in speech_segments(&probs, pcm.len(), &params) {
        if seconds {
            println!("{:.3} {:.3}", seg.start_seconds(), seg.end_seconds());
        } else {
            println!("{} {}", seg.start, seg.end);
        }
    }
    Ok(())
}

/// Score the same clip on several backends and report the largest divergence.
fn compare_devices(list: &str, pcm: &[f32], threshold: f32) -> Result<()> {
    let devices = crate::device::parse_device_list(list)?;
    let mut reference: Option<Vec<f32>> = None;
    for dev in devices {
        let label = crate::device::device_label(dev);
        let mut batch = TenVadBatch::new(dev)?;
        // Compile once outside the timed region, or the first device pays for
        // codegen and the RTF column means nothing.
        let _ = batch.probabilities(&pcm[..pcm.len().min(HOP_SIZE * 8)])?;
        let started = std::time::Instant::now();
        let probs = batch.probabilities(pcm)?;
        let elapsed = started.elapsed().as_secs_f64();
        let audio = pcm.len() as f64 / SAMPLE_RATE as f64;
        let voiced = probs.iter().filter(|&&p| p > threshold).count();
        match &reference {
            None => {
                println!(
                    "{label:8} {elapsed:8.3}s  RTF {:8.1}x  voiced {voiced}/{}",
                    audio / elapsed,
                    probs.len()
                );
                reference = Some(probs);
            }
            Some(r) => {
                let max = r
                    .iter()
                    .zip(&probs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "{label:8} {elapsed:8.3}s  RTF {:8.1}x  voiced {voiced}/{}  max|Δ| vs cpu {max:.2e}",
                    audio / elapsed,
                    probs.len()
                );
            }
        }
    }
    Ok(())
}
