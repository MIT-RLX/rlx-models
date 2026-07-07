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

//! `rlx-maya1` — Maya1 expressive voice-design TTS CLI.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_maya1::{Maya1, parse_device};

const HELP: &str = "\
rlx-maya1 — Maya1 expressive voice-design TTS (Llama-3B + SNAC, Apache-2.0, 24 kHz)

USAGE: rlx-maya1 --gguf maya1.Q4_K_M.gguf --description \"...\" --text \"...\"

OPTIONS:
    --gguf <FILE>        Maya1 GGUF (default: weights/tts/maya1/maya1.Q4_K_M.gguf)
    --snac <FILE>        SNAC decoder weights (else rlx-orpheus default/env)
    --description <T>    Natural-language voice design (age/gender/accent/emotion)
    --text <T>           Text to speak (inline tags like <laugh>, <whisper> ok)
    --seed <N>           Sampling seed
    --device <DEV>       cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>         Output WAV (default: maya1.wav)
    -h, --help           Show help
";

fn find_gguf(dir: &Path) -> Option<PathBuf> {
    let preferred = dir.join("maya1.Q4_K_M.gguf");
    if preferred.is_file() {
        return Some(preferred);
    }
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.extension().is_some_and(|x| x == "gguf")
    })
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
    let mut gguf: Option<PathBuf> = None;
    let mut snac: Option<PathBuf> = None;
    let mut description = "Realistic male voice in his 30s with an American accent. Normal pitch, warm timbre, conversational pacing.".to_string();
    let mut text: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("maya1.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--gguf" => gguf = Some(PathBuf::from(next()?)),
            "--snac" => snac = Some(PathBuf::from(next()?)),
            "--description" => description = next()?,
            "--text" => text = Some(next()?),
            "--seed" => seed = Some(next()?.parse().context("--seed")?),
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
    let gguf = gguf
        .or_else(|| find_gguf(Path::new(rlx_maya1::DEFAULT_LOCAL_DIR)))
        .context("no GGUF found; pass --gguf or place one in weights/tts/maya1/")?;
    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;

    let mut tts = match &snac {
        Some(s) => Maya1::load_with_snac(&gguf, s, device)?,
        None => Maya1::load_on(&gguf, device)?,
    };
    if let Some(s) = seed {
        tts.config_mut().seed = s;
    }
    let audio = tts.synthesize(&description, &text)?;

    tts.write_wav(&audio, &out)?;
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz) to {}",
        audio.len(),
        tts.sample_rate(),
        out.display()
    );
    Ok(())
}
