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

//! `rlx-openvoice` — OpenVoice v2 zero-shot voice-cloning TTS CLI.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use rlx_openvoice::{
    DEFAULT_MELO_DIR, DEFAULT_OPENVOICE_DIR, DEFAULT_TAU, OpenVoice, parse_device,
};

const HELP: &str = "\
rlx-openvoice — OpenVoice v2 zero-shot voice cloning (MeloTTS base + tone-color, MIT)

USAGE: rlx-openvoice --ref-wav ref.wav --text \"...\"

OPTIONS:
    --melo-dir <DIR>   MeloTTS bundle (default: weights/tiny-tts-rlx)
    --data <DIR>       OpenVoice ONNX dir (default: weights/tts/openvoice)
    --ref-wav <WAV>    Reference voice to clone
    --text <T>         Text to speak in the cloned voice
    --tau <F>          Flow temperature (default: 0.3)
    --device <DEV>     cpu | metal | mlx | cuda | gpu (default: cpu)
    --out <FILE>       Output WAV (default: openvoice.wav)
    -h, --help         Show help
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
    let mut melo_dir = PathBuf::from(DEFAULT_MELO_DIR);
    let mut data = PathBuf::from(DEFAULT_OPENVOICE_DIR);
    let (mut ref_wav, mut text): (Option<PathBuf>, Option<String>) = (None, None);
    let mut tau = DEFAULT_TAU;
    let mut device_str = "cpu".to_string();
    let mut out = PathBuf::from("openvoice.wav");

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        let mut next = || a.next().with_context(|| format!("missing value for {arg}"));
        match arg.as_str() {
            "--melo-dir" => melo_dir = PathBuf::from(next()?),
            "--data" => data = PathBuf::from(next()?),
            "--ref-wav" => ref_wav = Some(PathBuf::from(next()?)),
            "--text" => text = Some(next()?),
            "--tau" => tau = next()?.parse().context("--tau")?,
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
    let text = text.context("--text is required")?;
    let device = parse_device(&device_str).with_context(|| format!("device '{device_str}'"))?;

    let tts = OpenVoice::load_on(&melo_dir, &data, device)?;
    let (reference, ref_sr) = read_wav(&ref_wav)?;
    let audio = tts.synthesize(&text, &reference, ref_sr, tau)?;

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
