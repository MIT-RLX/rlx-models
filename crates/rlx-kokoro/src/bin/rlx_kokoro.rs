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

//! `rlx-kokoro` — Kokoro-82M text-to-speech CLI.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_kokoro::config::SAMPLE_RATE;
use rlx_runtime::parse_device;

const HELP: &str = "\
rlx-kokoro — Kokoro-82M (StyleTTS2 + ISTFTNet) text-to-speech

USAGE:
    rlx-kokoro [OPTIONS] --text \"<text>\"

OPTIONS:
    --data <DIR>       Model directory (default: $RLX_KOKORO_DIR or .cache/kokoro-82m)
    --model <FILE>     ONNX variant filename (ort path only; default: model.onnx)
    --text <TEXT>      Text to synthesize (English; espeak-ng G2P)
    --ipa <PHONEMES>   Synthesize directly from an IPA/phoneme string
    --voice <NAME>     Voice name (default: af_heart)
    --speed <FLOAT>    Speaking rate (default: 1.0)
    --device <DEV>     cpu | metal | mlx | cuda | rocm | gpu (default: cpu)
    --out <FILE>       Output WAV path (default: kokoro.wav)
    --list-voices      List available voices and exit
    --download         Download the default bundle (needs `hf-download`) and exit
    -h, --help         Show this help

Native (default): needs `onnx/rlx-split/` from `scripts/split_kokoro.py`.
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
    let mut model_file = "model.onnx".to_string();
    let mut text: Option<String> = None;
    let mut ipa: Option<String> = None;
    let mut voice = "af_heart".to_string();
    let mut speed = 1.0f32;
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("kokoro.wav");
    let mut list_voices = false;
    let mut download = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next(&mut args, "--data")?)),
            "--model" => model_file = next(&mut args, "--model")?,
            "--text" => text = Some(next(&mut args, "--text")?),
            "--ipa" => ipa = Some(next(&mut args, "--ipa")?),
            "--voice" => voice = next(&mut args, "--voice")?,
            "--speed" => {
                speed = next(&mut args, "--speed")?
                    .parse()
                    .context("parse --speed")?
            }
            "--device" => device_str = next(&mut args, "--device")?,
            "--out" => out = PathBuf::from(next(&mut args, "--out")?),
            "--list-voices" => list_voices = true,
            "--download" => download = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }

    if download {
        #[cfg(feature = "hf-download")]
        {
            let dir = rlx_kokoro::fetch_default()?;
            println!("Downloaded Kokoro bundle to {}", dir.display());
            return Ok(());
        }
        #[cfg(not(feature = "hf-download"))]
        anyhow::bail!("--download requires building with `--features hf-download`");
    }

    let data_dir = data
        .or_else(|| std::env::var_os("RLX_KOKORO_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(rlx_kokoro::DEFAULT_LOCAL_DIR));

    let device =
        parse_device(&device_str).with_context(|| format!("invalid --device '{device_str}'"))?;

    #[cfg(feature = "native")]
    {
        use rlx_kokoro::{NativeKokoro, write_wav};

        if model_file != "model.onnx" {
            eprintln!("[kokoro] --model ignored on native path (uses onnx/rlx-split/)");
        }
        let tts = NativeKokoro::load(&data_dir, device)
            .with_context(|| format!("load native Kokoro from {}", data_dir.display()))?;

        if list_voices {
            println!("Available voices ({}):", tts.voice_names().len());
            for v in tts.voice_names() {
                println!("  {v}");
            }
            return Ok(());
        }

        let audio = synthesize(&tts, ipa.as_deref(), text.as_deref(), &voice, speed)?;
        write_wav(&audio, &out)?;
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        println!(
            "Wrote {} samples ({secs:.2}s @ {SAMPLE_RATE} Hz) to {} [voice={voice}, backend=native, device={device:?}]",
            audio.len(),
            out.display(),
        );
        return Ok(());
    }

    #[cfg(all(feature = "onnx", not(feature = "native")))]
    {
        use rlx_kokoro::Kokoro;

        let tts = Kokoro::load_on(&data_dir, &model_file, device)
            .with_context(|| format!("load Kokoro from {}", data_dir.display()))?;

        if list_voices {
            println!("Available voices ({}):", tts.voice_names().len());
            for v in tts.voice_names() {
                println!("  {v}");
            }
            return Ok(());
        }

        let audio = synthesize(&tts, ipa.as_deref(), text.as_deref(), &voice, speed)?;
        tts.write_wav(&audio, &out)?;
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        println!(
            "Wrote {} samples ({secs:.2}s @ {SAMPLE_RATE} Hz) to {} [voice={voice}, ep={}]",
            audio.len(),
            out.display(),
            tts.ort_ep()
        );
        return Ok(());
    }

    #[cfg(not(any(feature = "native", feature = "onnx")))]
    compile_error!("rlx-kokoro CLI requires `native` or `onnx` feature");

    #[allow(unreachable_code)]
    Ok(())
}

#[cfg(feature = "native")]
fn synthesize(
    tts: &rlx_kokoro::NativeKokoro,
    ipa: Option<&str>,
    text: Option<&str>,
    voice: &str,
    speed: f32,
) -> Result<Vec<f32>> {
    if let Some(ipa) = ipa {
        return tts.infer_phonemes(ipa, voice, speed);
    }
    if let Some(text) = text {
        #[cfg(feature = "espeak")]
        return tts.generate_from_text(text, voice, speed);
        #[cfg(not(feature = "espeak"))]
        {
            let _ = text;
            anyhow::bail!("--text requires the `espeak` feature; use --ipa instead");
        }
    }
    anyhow::bail!("provide --text \"<text>\" or --ipa \"<phonemes>\"\n\n{HELP}")
}

#[cfg(all(feature = "onnx", not(feature = "native")))]
fn synthesize(
    tts: &rlx_kokoro::Kokoro,
    ipa: Option<&str>,
    text: Option<&str>,
    voice: &str,
    speed: f32,
) -> Result<Vec<f32>> {
    if let Some(ipa) = ipa {
        return tts.infer_phonemes(ipa, voice, speed);
    }
    if let Some(text) = text {
        #[cfg(feature = "espeak")]
        return tts.generate_from_text(text, voice, speed);
        #[cfg(not(feature = "espeak"))]
        {
            let _ = text;
            anyhow::bail!("--text requires the `espeak` feature; use --ipa instead");
        }
    }
    anyhow::bail!("provide --text \"<text>\" or --ipa \"<phonemes>\"\n\n{HELP}")
}

fn next(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("missing value for {flag}"))
}
