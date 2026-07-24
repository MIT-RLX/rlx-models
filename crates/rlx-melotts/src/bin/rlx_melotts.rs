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

//! `rlx-melotts` — MeloTTS (VITS2) multilingual TTS CLI.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_melotts::{
    DEFAULT_LOCAL_DIR, InferOpts, MeloTts, is_bundle_dir, normalize_audio, parse_device,
    peak_amplitude, resolve_bundle_dir, write_wav,
};

const HELP: &str = "\
rlx-melotts — MeloTTS (VITS2) multilingual TTS (MIT, native rlx backends)

USAGE: rlx-melotts --text \"...\" [--data weights/tts/melotts]

OPTIONS:
    --data <DIR>        MeloTTS/TinyTTS bundle dir (default: auto-resolve)
    --text <T>          Text to speak
    --speed <F>         Speaking rate (default: 1.0; maps to 1/length_scale)
    --noise <F>         Flow noise scale (default: model config)
    --seed <N>          Sampling seed
    --device <DEV>      cpu | metal | mlx | cuda | gpu | vulkan (default: cpu)
    --out <FILE>        Output WAV (default: melotts.wav)
    -h, --help          Show help
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
    let (mut speed, mut noise, mut seed): (Option<f32>, Option<f32>, Option<u64>) =
        (None, None, None);
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("melotts.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next()?)),
            "--text" => text = Some(next()?),
            "--speed" => speed = Some(next()?.parse().context("--speed")?),
            "--noise" => noise = Some(next()?.parse().context("--noise")?),
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
    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;

    let data = match data {
        Some(p) if is_bundle_dir(&p) || p.is_file() => p,
        Some(p) => anyhow::bail!(
            "not a MeloTTS/TinyTTS bundle at {} (need config.json + onnx/); default is {DEFAULT_LOCAL_DIR}",
            p.display()
        ),
        None => resolve_bundle_dir()?,
    };

    let tts =
        MeloTts::load(&data).with_context(|| format!("load MeloTTS from {}", data.display()))?;
    let mut opts = InferOpts::from_config(tts.config());
    if let Some(s) = speed {
        anyhow::ensure!(s > 0.0, "--speed must be > 0");
        opts.length_scale = 1.0 / s;
    }
    if let Some(n) = noise {
        opts.noise_scale = n;
    }
    if let Some(s) = seed {
        opts.seed = s;
    }
    let wav = tts.synthesize_on(&text, device, &opts)?;
    let peak = peak_amplitude(&wav.samples);
    let normalized = normalize_audio(&wav.samples);
    write_wav(&out, &normalized, wav.sample_rate)
        .with_context(|| format!("write {}", out.display()))?;

    let secs = wav.samples.len() as f32 / wav.sample_rate as f32;
    println!(
        "Wrote {} samples ({secs:.2}s @ {} Hz, peak={peak:.3}) to {} [device={device:?}]",
        wav.samples.len(),
        wav.sample_rate,
        out.display()
    );
    Ok(())
}
