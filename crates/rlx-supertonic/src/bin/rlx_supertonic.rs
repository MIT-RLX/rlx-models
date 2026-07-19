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

//! `rlx-supertonic` — Supertonic-3 flow-matching TTS CLI.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_runtime::parse_device;
use rlx_supertonic::{InferOpts, Supertonic, Voice, config::DEFAULT_LOCAL_DIR, list_voices};

const HELP: &str = "\
rlx-supertonic — Supertonic-3 flow-matching TTS

USAGE: rlx-supertonic [OPTIONS] --text \"<text>\"

OPTIONS:
    --data <DIR>      Model dir (default: $RLX_SUPERTONIC_DIR or weights/tts/supertonic-3)
    --text <TEXT>     Text to synthesize
    --lang <LANG>     Language code (default: en)
    --voice <NAME>    Voice name (default: F1)
    --steps <N>       Flow-matching denoising steps (default: 8)
    --speed <F>       Speaking rate (default: 1.05, higher=faster)
    --seed <N>        RNG seed for the noisy latent (default: 0)
    --device <DEV>    cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>      Output WAV (default: supertonic.wav)
    --list-voices     List voices and exit
    --download        Download the bundle (needs hf-download) and exit
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
    let mut data: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut lang = "en".to_string();
    let mut voice = "F1".to_string();
    let mut opts = InferOpts::default();
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("supertonic.wav");
    let mut list = false;
    let mut download = false;

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next(&mut a, "--data")?)),
            "--text" => text = Some(next(&mut a, "--text")?),
            "--lang" => lang = next(&mut a, "--lang")?,
            "--voice" => voice = next(&mut a, "--voice")?,
            "--steps" => opts.total_step = next(&mut a, "--steps")?.parse().context("--steps")?,
            "--speed" => opts.speed = next(&mut a, "--speed")?.parse().context("--speed")?,
            "--seed" => opts.seed = next(&mut a, "--seed")?.parse().context("--seed")?,
            "--device" => device_str = next(&mut a, "--device")?,
            "--out" => out = PathBuf::from(next(&mut a, "--out")?),
            "--list-voices" => list = true,
            "--download" => download = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }

    if download {
        let dir = rlx_supertonic::download::fetch_default()?;
        println!("Downloaded Supertonic-3 to {}", dir.display());
        return Ok(());
    }

    let dir = data
        .or_else(|| std::env::var_os("RLX_SUPERTONIC_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LOCAL_DIR));

    if list {
        for v in list_voices(&dir.join("voice_styles"))? {
            println!("  {v}");
        }
        return Ok(());
    }

    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;
    let tts = Supertonic::load_on(&dir, device)
        .with_context(|| format!("load from {}", dir.display()))?;
    let voice_obj = Voice::load(&dir.join("voice_styles").join(format!("{voice}.json")))
        .with_context(|| format!("load voice '{voice}'"))?;

    let text = text.context("provide --text \"<text>\"")?;
    let audio = tts.synthesize(&text, &lang, &voice_obj, &opts)?;
    tts.write_wav(&audio, &out)?;
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {} [voice={voice}, lang={lang}, steps={}, ep={}]",
        audio.len(),
        tts.sample_rate(),
        out.display(),
        opts.total_step,
        tts.ort_ep()
    );
    Ok(())
}

fn next(a: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    a.next()
        .with_context(|| format!("missing value for {flag}"))
}
