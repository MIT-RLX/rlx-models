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

//! `rlx-moss-nano` — MOSS-TTS-Nano CLI (48 kHz stereo, builtin-voice cloning).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_moss_nano::{DEFAULT_LOCAL_DIR, MossNano, SynthOpts, parse_device};

const HELP: &str = "\
rlx-moss-nano — MOSS-TTS-Nano hierarchical AR TTS (OpenMOSS, Apache-2.0, 48 kHz)

USAGE: rlx-moss-nano --text \"...\" [--voice Trump]

OPTIONS:
    --data <DIR>     Model dir (default: weights/tts/moss-nano)
    --text <T>       Text to speak
    --voice <NAME>   Builtin voice (default: Trump). Use --list-voices to see all
    --seed <N>       Sampling seed (default: 0)
    --max-frames <N> Cap generated frames
    --device <DEV>   cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>     Output WAV (default: moss_nano.wav)
    --list-voices    Print builtin voices and exit
    -h, --help       Show help
";

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
    let mut text: Option<String> = None;
    let mut voice = "Trump".to_string();
    let mut opts = SynthOpts::default();
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("moss_nano.wav");
    let mut list_voices = false;

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = PathBuf::from(next()?),
            "--text" => text = Some(next()?),
            "--voice" => voice = next()?,
            "--seed" => opts.seed = next()?.parse().context("--seed")?,
            "--max-frames" => opts.max_frames = Some(next()?.parse().context("--max-frames")?),
            "--device" => device_str = next()?,
            "--out" => out = PathBuf::from(next()?),
            "--list-voices" => list_voices = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }

    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;
    let tts = MossNano::load_on(&data, device).with_context(|| format!("load from {}", data.display()))?;

    if list_voices {
        println!("builtin voices: {}", tts.voice_names().join(", "));
        return Ok(());
    }
    let text = text.context("--text is required")?;
    let audio = tts.synthesize(&text, &voice, &opts)?;

    tts.write_wav(&audio, &out)?;
    let frames = audio.len() / tts.channels().max(1) as usize;
    let secs = frames as f32 / tts.sample_rate() as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz, {}ch) to {} [voice={voice}, ep={}]",
        audio.len(),
        tts.sample_rate(),
        tts.channels(),
        out.display(),
        tts.ort_ep()
    );
    Ok(())
}
