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

//! `rlx-zipvoice` — ZipVoice voice-cloning TTS CLI.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_runtime::parse_device;
use rlx_zipvoice::{DEFAULT_LOCAL_DIR, ZipVoice, zipvoice_opts};

const HELP: &str = "\
rlx-zipvoice — ZipVoice flow-matching voice-cloning TTS (Apache-2.0)

USAGE: rlx-zipvoice --prompt-wav ref.wav --prompt-text \"...\" --text \"...\"

OPTIONS:
    --data <DIR>          Model dir (default: weights/tts/zipvoice-distill)
    --prompt-wav <WAV>    Reference voice (resampled to 24 kHz mono)
    --prompt-text <T>     Transcript of the reference audio
    --text <T>            Text to speak in the cloned voice
    --steps <N>           Flow-matching steps (default: 4)
    --speed <F>           Speaking rate (default: 1.0)
    --device <DEV>        cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>          Output WAV (default: zipvoice.wav)
    -h, --help            Show help
";

fn read_wav_24k(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let m = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / m).collect()
        }
    };
    let mono: Vec<f32> = if spec.channels > 1 {
        raw.chunks(spec.channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
    } else {
        raw
    };
    if spec.sample_rate == 24000 {
        return Ok(mono);
    }
    let (from, to) = (spec.sample_rate as u64, 24000u64);
    let n = (mono.len() as u64 * to / from).max(1) as usize;
    Ok((0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = mono[idx.min(mono.len() - 1)];
            let b = mono[(idx + 1).min(mono.len() - 1)];
            a + (b - a) * f
        })
        .collect())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut data = PathBuf::from(DEFAULT_LOCAL_DIR);
    let (mut prompt_wav, mut prompt_text, mut text) = (None, None, None);
    let mut opts = zipvoice_opts();
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("zipvoice.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = PathBuf::from(next()?),
            "--prompt-wav" => prompt_wav = Some(PathBuf::from(next()?)),
            "--prompt-text" => prompt_text = Some(next()?),
            "--text" => text = Some(next()?),
            "--steps" => opts.num_step = next()?.parse().context("--steps")?,
            "--speed" => opts.speed = next()?.parse().context("--speed")?,
            "--device" => device_str = next()?,
            "--out" => out = PathBuf::from(next()?),
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }

    let prompt_wav = prompt_wav.context("--prompt-wav is required")?;
    let prompt_text = prompt_text.context("--prompt-text is required")?;
    let text = text.context("--text is required")?;

    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;
    let tts = ZipVoice::load_on(&data, device).with_context(|| format!("load from {}", data.display()))?;
    let pw = read_wav_24k(&prompt_wav)?;
    let audio = tts.synthesize(&text, &pw, &prompt_text, &opts)?;

    tts.write_wav(&audio, &out)?;
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {} [steps={}, ep={}]",
        audio.len(),
        tts.sample_rate(),
        out.display(),
        opts.num_step,
        tts.ort_ep()
    );
    Ok(())
}
