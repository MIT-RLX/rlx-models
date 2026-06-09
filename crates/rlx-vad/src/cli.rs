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

use crate::audio::{SAMPLE_RATE_16K, load_wav_mono_f32, resample_linear};
use crate::device::resolve_device;
use crate::segments::SegmentParams;
use anyhow::{Result, bail};
use rlx_cli::req;
use std::path::PathBuf;

#[cfg(feature = "silero")]
use crate::SampleRate;
#[cfg(feature = "earshot")]
use crate::segments::speech_segments_earshot;
#[cfg(feature = "silero")]
use crate::segments::speech_segments_silero;
#[cfg(feature = "silero")]
use crate::silero::{SileroConfig, SileroSession, SileroWeights};

fn backends_help() -> String {
    crate::enabled_backends().join("|")
}

pub fn run(args: &[String]) -> Result<()> {
    if crate::enabled_backends().is_empty() {
        bail!("rlx-vad built without VAD backends (enable `earshot` and/or `silero` features)");
    }

    let mut backend = crate::default_backend().to_string();
    let mut wav: Option<PathBuf> = None;
    #[cfg(feature = "silero")]
    let mut weights: Option<PathBuf> = None;
    let mut threshold: Option<f32> = None;
    let mut device = "cpu".to_string();
    let mut return_seconds = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => backend = req(args, &mut i)?,
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--weights" => {
                #[cfg(feature = "silero")]
                {
                    weights = Some(req(args, &mut i)?.into());
                }
                #[cfg(not(feature = "silero"))]
                bail!("--weights requires the `silero` feature");
            }
            "--threshold" => {
                threshold = Some(
                    req(args, &mut i)?
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--threshold: f32"))?,
                );
            }
            "--device" => device = req(args, &mut i)?,
            "--seconds" => {
                return_seconds = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-vad — voice activity detection on RLX\n\
                     VAD backends enabled: {}\n\
                     Flags: --backend {} [--weights PATH] --wav PATH\n\
                       (Silero: embedded safetensors; --weights overrides)\n\
                       [--threshold override] [--device cpu|metal|…] [--seconds]",
                    crate::enabled_backends().join(", "),
                    backends_help(),
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let wav = wav.ok_or_else(|| anyhow::anyhow!("--wav PATH required"))?;
    let _dev = resolve_device(&device)?;

    let (sr, mut pcm) = load_wav_mono_f32(&wav)?;
    if sr != SAMPLE_RATE_16K {
        pcm = resample_linear(&pcm, sr, SAMPLE_RATE_16K);
    }

    let mut params = SegmentParams::for_algorithm(&backend);
    if let Some(t) = threshold {
        params.threshold = t;
    }

    let segs = match backend.as_str() {
        "earshot" => {
            #[cfg(not(feature = "earshot"))]
            bail!("backend `earshot` not enabled (rebuild with `--features earshot`)");
            #[cfg(feature = "earshot")]
            speech_segments_earshot(&pcm, &params)
        }
        "silero" => {
            #[cfg(not(feature = "silero"))]
            bail!("backend `silero` not enabled (rebuild with `--features silero`)");
            #[cfg(feature = "silero")]
            {
                let w = match weights {
                    Some(path) => SileroWeights::load(&path)?,
                    None => SileroWeights::embedded(),
                };
                let mut session = SileroSession::new(
                    w,
                    SileroConfig {
                        sample_rate: SampleRate::Hz16000,
                    },
                );
                speech_segments_silero(&mut session, &pcm, &params)?
            }
        }
        other => bail!(
            "unknown backend {other} (enabled: {})",
            crate::enabled_backends().join(", ")
        ),
    };

    for seg in segs {
        if return_seconds {
            println!(
                "{:.3} {:.3}",
                seg.start as f64 / SAMPLE_RATE_16K as f64,
                seg.end as f64 / SAMPLE_RATE_16K as f64
            );
        } else {
            println!("{} {}", seg.start, seg.end);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rlx_cli::parse_standard_device;

    #[test]
    fn parse_device_cpu() {
        parse_standard_device("rlx-vad", "cpu").unwrap();
    }
}
