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

//! `rlx-piper` — Piper VITS text-to-speech CLI.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_piper::{Piper, config::DEFAULT_LOCAL_DIR};
use rlx_runtime::parse_device;

const HELP: &str = "\
rlx-piper — Piper VITS text-to-speech

USAGE: rlx-piper --text \"<text>\"

OPTIONS:
    --data <DIR>      Voice dir with <voice>.onnx + .onnx.json (default: weights/tts/piper)
    --text <TEXT>     Text to synthesize
    --length <F>      Length scale (>1 slower; default: config)
    --device <DEV>    cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>      Output WAV (default: piper.wav)
    -h, --help        Show help
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
    let mut length: Option<f32> = None;
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("piper.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = PathBuf::from(next()?),
            "--text" => text = Some(next()?),
            "--length" => length = Some(next()?.parse().context("--length")?),
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
    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;
    let tts = Piper::load_on(&data, device).with_context(|| format!("load from {}", data.display()))?;

    #[cfg(feature = "espeak")]
    let audio = tts.synthesize(&text, length)?;
    #[cfg(not(feature = "espeak"))]
    let audio: Vec<f32> = {
        let _ = (&text, length);
        anyhow::bail!("rlx-piper needs the `espeak` feature");
    };

    tts.write_wav(&audio, &out)?;
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {} [ep={}]",
        audio.len(),
        tts.sample_rate(),
        out.display(),
        tts.ort_ep()
    );
    Ok(())
}
