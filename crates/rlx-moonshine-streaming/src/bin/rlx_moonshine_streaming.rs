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

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_moonshine_streaming::{MoonshineStreaming, SAMPLE_RATE};

const HELP: &str = "\
rlx-moonshine-streaming — Moonshine v2 streaming ASR (tiny / small / medium)

USAGE:
  rlx-moonshine-streaming --weights <DIR> --wav <FILE> [options]

OPTIONS:
  --weights <DIR>       moonshine-ai/moonshine-streaming-{tiny,small,medium} directory
  --wav <FILE>          PCM16 / float WAV (any rate; resampled to 16 kHz)
  --device <NAME>       cpu | metal | mlx | cuda | rocm | gpu | vulkan (default cpu)
  --max-new-tokens <N>  decode budget (default 6.5 per second of audio + 8)
  -h, --help            show this help
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut max_new = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(&args, &mut i)?.into()),
            "--wav" => wav = Some(req(&args, &mut i)?.into()),
            "--device" => device = req(&args, &mut i)?,
            "--max-new-tokens" => max_new = Some(req(&args, &mut i)?.parse()?),
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other} (try --help)"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let wav = wav.ok_or_else(|| anyhow!("--wav is required"))?;
    let device = parse_standard_device("moonshine-streaming", &device)?;

    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(&wav)?;
    let pcm = if rate as usize == SAMPLE_RATE {
        pcm
    } else {
        rlx_core::audio::resample_linear(&pcm, rate, SAMPLE_RATE as u32)
    };

    let t0 = std::time::Instant::now();
    let mut model = MoonshineStreaming::open(&weights, device)?;
    eprintln!("[rlx-moonshine-streaming] loaded in {:.2?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    let text = model.transcribe(&pcm, max_new)?;
    let secs = pcm.len() as f64 / SAMPLE_RATE as f64;
    let dt = t1.elapsed().as_secs_f64();
    println!("{text}");
    eprintln!(
        "[rlx-moonshine-streaming] {secs:.2}s audio in {dt:.2}s ({:.1}× real time) on {device:?}",
        secs / dt.max(1e-9)
    );
    Ok(())
}
