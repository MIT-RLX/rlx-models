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

//! Does a cloned voice actually sound like the speaker it was cloned from?
//!
//! Parity suites answer "does this port match the reference implementation"
//! and Whisper answers "are the words right". Neither answers the question a
//! voice cloner exists to answer. This one does: embed the reference clip and
//! each generated clip with WeSpeaker ResNet34-LM and report the cosine.
//!
//! On the VoxCeleb convention for x-vector systems, **> 0.7 is "same
//! speaker"**; 0.5–0.7 is a recognizable but imperfect clone, and below ~0.4
//! the identity has not transferred. Compare against the *reference clip*, not
//! against another synthesis — two bad clones can agree with each other.
//!
//! ```text
//! cargo run -p rlx-wespeaker --release --features native \
//!   --example clone_fidelity -- reference.wav cloned_a.wav cloned_b.wav
//! ```
//!
//! Any sample rate and channel count is accepted; input is downmixed to mono
//! and resampled to the 16 kHz the model expects.
//!
//! `--ort` runs the ONNX Runtime reference instead of the native graph
//! (`--features onnx`). Prefer it for measurement: the native graph bakes a
//! fixed 148-frame window, so it only sees the first ~1.5 s of each clip and
//! its speaker estimates are correspondingly noisy. The two agree on the
//! direction of every comparison, not on absolute values.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_core::audio::resample_linear;
use rlx_runtime::Device;
use rlx_wespeaker::{WeSpeaker, cosine};

/// Embedding backends this tool can measure with.
enum Embedder {
    Native(Box<WeSpeaker>),
    /// ONNX Runtime reference (`--features onnx`). Reference only — it exists
    /// so a fidelity number can be trusted independently of the native graph.
    #[cfg(feature = "onnx")]
    Ort(Box<rlx_wespeaker::OrtWeSpeaker>),
}

impl Embedder {
    fn embed(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Native(s) => s.embed_pcm(pcm),
            #[cfg(feature = "onnx")]
            Self::Ort(s) => s.embed_pcm(pcm),
        }
    }
}

/// WeSpeaker's fbank front end is defined at 16 kHz.
const MODEL_RATE: u32 = 16_000;

fn parse_device(s: &str) -> Device {
    match s.to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

/// Read a WAV as mono f32 at `MODEL_RATE`, whatever it started as.
fn load_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    if raw.is_empty() {
        bail!("{} contains no samples", path.display());
    }
    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch <= 1 {
        raw
    } else {
        raw.chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok(resample_linear(&mono, spec.sample_rate, MODEL_RATE))
}

fn verdict(cos: f32) -> &'static str {
    if cos >= 0.7 {
        "same speaker"
    } else if cos >= 0.5 {
        "recognizable, imperfect"
    } else if cos >= 0.4 {
        "weak"
    } else {
        "identity did not transfer"
    }
}

fn main() -> Result<()> {
    let mut model_dir = PathBuf::from("weights/wespeaker-voxceleb-resnet34-LM");
    let mut device = Device::Cpu;
    let mut use_ort = false;
    let mut wavs: Vec<PathBuf> = Vec::new();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(args.get(i + 1).context("--model-dir needs a path")?);
                i += 2;
            }
            "--ort" => {
                use_ort = true;
                i += 1;
            }
            "--device" => {
                device = parse_device(args.get(i + 1).context("--device needs a value")?);
                i += 2;
            }
            other => {
                wavs.push(PathBuf::from(other));
                i += 1;
            }
        }
    }
    if wavs.len() < 2 {
        bail!(
            "usage: clone_fidelity [--model-dir DIR] [--device D] <reference.wav> <cloned.wav>..."
        );
    }

    let mut spk = if use_ort {
        #[cfg(feature = "onnx")]
        {
            let onnx = model_dir.join("onnx/wespeaker_ref.onnx");
            Embedder::Ort(Box::new(
                rlx_wespeaker::OrtWeSpeaker::open(&onnx)
                    .with_context(|| format!("load WeSpeaker ONNX reference {}", onnx.display()))?,
            ))
        }
        #[cfg(not(feature = "onnx"))]
        bail!("--ort needs `--features onnx`");
    } else {
        Embedder::Native(Box::new(
            WeSpeaker::open_on(&model_dir, device)
                .with_context(|| format!("load WeSpeaker from {}", model_dir.display()))?,
        ))
    };

    let reference = &wavs[0];
    let ref_emb = spk.embed(&load_mono_16k(reference)?)?;
    println!("reference: {}", reference.display());

    let mut worst = f32::INFINITY;
    for wav in &wavs[1..] {
        let emb = spk.embed(&load_mono_16k(wav)?)?;
        let cos = cosine(&ref_emb, &emb);
        worst = worst.min(cos);
        println!("  {:>8.4}  {}  ({})", cos, wav.display(), verdict(cos));
    }
    println!("\nworst {worst:.4} — {}", verdict(worst));
    Ok(())
}
