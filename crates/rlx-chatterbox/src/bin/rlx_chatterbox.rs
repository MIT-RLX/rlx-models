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

//! `rlx-chatterbox` — ChatterBox zero-shot voice-cloning TTS CLI (native RLX).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_chatterbox::{DEFAULT_LOCAL_DIR, NativeChatterBox, SynthOpts, parse_device};

const HELP: &str = "\
rlx-chatterbox — ChatterBox zero-shot voice cloning (Resemble AI, MIT, 24 kHz)
                native RLX path (no ONNX Runtime)

USAGE: rlx-chatterbox --ref-wav ref.wav --text \"...\"

OPTIONS:
    --data <DIR>        Model dir (default: weights/tts/chatterbox)
    --ref-wav <WAV>     Reference voice to clone (else default_voice.wav)
    --text <T>          Text to speak
    --exaggeration <F>  Emotion/intensity (default: 0.5)
    --temperature <F>   Sampling temperature (default: 0.8)
    --seed <N>          Sampling seed
    --max-frames <N>    Max speech tokens (default: 1000)
    --greedy            Argmax sampling (deterministic)
    --device <DEV>      cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>        Output WAV (default: chatterbox.wav)
    -h, --help          Show help
";

fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
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
    Ok((mono, spec.sample_rate))
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
    let (mut ref_wav, mut text): (Option<PathBuf>, Option<String>) = (None, None);
    let mut opts = SynthOpts::default();
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("chatterbox.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = PathBuf::from(next()?),
            "--ref-wav" => ref_wav = Some(PathBuf::from(next()?)),
            "--text" => text = Some(next()?),
            "--exaggeration" => opts.exaggeration = next()?.parse().context("--exaggeration")?,
            "--temperature" => opts.temperature = next()?.parse().context("--temperature")?,
            "--seed" => opts.seed = next()?.parse().context("--seed")?,
            "--max-frames" => opts.max_frames = next()?.parse().context("--max-frames")?,
            "--greedy" => opts.greedy = true,
            "--device" => device_str = next()?,
            "--out" => out = PathBuf::from(next()?),
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }

    let text = text.context("--text is required")?;
    let ref_path = ref_wav.unwrap_or_else(|| data.join("default_voice.wav"));
    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;

    let tts = NativeChatterBox::load_on(&data, device)
        .with_context(|| format!("load from {}", data.display()))?;
    let (reference, ref_sr) = read_wav(&ref_path)?;
    let t0 = Instant::now();
    let audio = tts.synthesize(&text, &reference, ref_sr, &opts)?;
    let synth = t0.elapsed();

    tts.write_wav(&audio, &out)?;
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    let rtf = if synth.as_secs_f32() > 0.0 {
        secs / synth.as_secs_f32()
    } else {
        0.0
    };
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {} [native/{:?}] synth={synth:?} ({rtf:.2}× RT)",
        audio.len(),
        tts.sample_rate(),
        out.display(),
        device,
    );
    Ok(())
}
