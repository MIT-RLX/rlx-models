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

//! `rlx-f5tts` — F5-TTS voice-cloning CLI (native RLX by default).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_f5tts::{
    F5Native, InferOpts, SAMPLE_RATE, config::DEFAULT_LOCAL_DIR, parse_device, write_wav,
};

const HELP: &str = "\
rlx-f5tts — F5-TTS flow-matching voice-cloning TTS (CC-BY-NC weights)
            native RLX path (no ONNX Runtime)

USAGE: rlx-f5tts --ref-wav ref.wav --ref-text \"...\" --text \"...\"

OPTIONS:
    --data <DIR>     Model dir (default: weights/tts/f5tts)
    --ref-wav <WAV>  Reference voice (resampled to 24 kHz mono)
    --ref-text <T>   Transcript of the reference audio
    --text <T>       Text to speak in the cloned voice
    --nfe <N>        Denoising steps (default: 32; lower = faster, rougher)
    --speed <F>      Speaking rate (default: 1.0)
    --device <DEV>   cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>     Output WAV (default: f5tts.wav)
    -h, --help       Show help
";

fn read_wav_24k(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = r.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let m = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / m)
                .collect()
        }
    };
    let mono: Vec<f32> = if spec.channels > 1 {
        raw.chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
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
    let (mut ref_wav, mut ref_text, mut text) = (None, None, None);
    let mut opts = InferOpts::default();
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("f5tts.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = PathBuf::from(next()?),
            "--ref-wav" => ref_wav = Some(PathBuf::from(next()?)),
            "--ref-text" => ref_text = Some(next()?),
            "--text" => text = Some(next()?),
            "--nfe" => opts.nfe = next()?.parse().context("--nfe")?,
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

    let ref_wav = ref_wav.context("--ref-wav is required")?;
    let ref_text = ref_text.context("--ref-text is required")?;
    let text = text.context("--text is required")?;

    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;
    let t0 = Instant::now();
    let tts = F5Native::load_on(&data, device)
        .with_context(|| format!("load from {}", data.display()))?;
    let exec = tts.execution_device();
    let dit = tts.dit_device();
    let rw = read_wav_24k(&ref_wav)?;
    let audio = tts.synthesize(&text, &rw, &ref_text, &opts)?;
    let synth_ms = t0.elapsed().as_secs_f64() * 1000.0;

    write_wav(&audio, SAMPLE_RATE, &out)?;
    let secs = audio.len() as f32 / SAMPLE_RATE as f32;
    let rtf = if secs > 0.0 {
        (synth_ms / 1000.0) / secs as f64
    } else {
        0.0
    };
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {} [native/{device:?} pre/dec={exec:?} dit={dit:?}, nfe={}] synth={synth_ms:.0}ms ({rtf:.2}× RT)",
        audio.len(),
        SAMPLE_RATE,
        out.display(),
        opts.nfe,
    );
    Ok(())
}
