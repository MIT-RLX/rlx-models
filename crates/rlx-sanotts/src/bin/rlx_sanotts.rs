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

//! `rlx-sanotts` — synthesize speech with a sanoTTS voice package.
//!
//! ```text
//! rlx-sanotts say "Hello from a two megabyte voice." --voice-dir ./amy-en-1p46m -o hello.wav
//! rlx-sanotts ids "Hello there" --voice-dir ./amy-en-1p46m
//! rlx-sanotts info --voice-dir ./amy-en-1p46m
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rlx_sanotts::{Backend, Synthesizer};

#[derive(Parser)]
#[command(
    name = "rlx-sanotts",
    about = "sanoTTS tiny neural text-to-speech on RLX"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Synthesize text to a WAV file.
    Say(SayArgs),
    /// Print the Piper phoneme ids a text would produce.
    Ids(IdsArgs),
    /// Print what a voice package contains.
    Info(VoiceArgs),
}

#[derive(Parser)]
struct VoiceArgs {
    /// Voice package directory (contains manifest.json).
    #[arg(long)]
    voice_dir: PathBuf,
}

#[derive(Parser)]
struct IdsArgs {
    /// Text to phonemize (or, with --phonemes, an espeak IPA string).
    text: String,
    /// Treat the argument as an espeak IPA string rather than text.
    #[arg(long)]
    phonemes: bool,
    #[command(flatten)]
    voice: VoiceArgs,
}

#[derive(Parser)]
struct SayArgs {
    /// Text to synthesize.
    text: String,
    #[command(flatten)]
    voice: VoiceArgs,
    /// Output WAV path.
    #[arg(short, long, default_value = "out.wav")]
    out: PathBuf,
    /// Speaking-rate scale (>0, larger = slower). Defaults to the voice's own.
    #[arg(long)]
    length_scale: Option<f32>,
    /// Where the frame-rate stages run: host, cpu, metal, mlx, cuda, rocm, gpu, vulkan, ane.
    #[arg(long, default_value = "host")]
    device: String,
    /// Synthesize from a comma-separated phoneme id list instead of text.
    #[arg(long, conflicts_with = "phonemes")]
    ids: Option<String>,
    /// Synthesize from an espeak IPA string instead of running G2P, e.g. the
    /// output of `espeak-ng -v en-us -q --ipa`. Bypasses the bundled
    /// phonemizer, whose en-US vowels differ slightly from the C library's.
    #[arg(long)]
    phonemes: Option<String>,
}

fn backend_from_str(s: &str) -> Result<Backend> {
    let s = s.to_ascii_lowercase();
    if s == "host" {
        return Ok(Backend::Host);
    }
    #[cfg(feature = "rlx-graph")]
    {
        use rlx_runtime::Device;
        let device = match s.as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "mlx" => Device::Mlx,
            "cuda" => Device::Cuda,
            "rocm" => Device::Rocm,
            "gpu" | "wgpu" => Device::Gpu,
            "vulkan" => Device::Vulkan,
            "ane" | "coreml" => Device::Ane,
            other => anyhow::bail!("unknown device {other:?}"),
        };
        if !rlx_runtime::is_available(device) {
            anyhow::bail!("device {device:?} is not available in this build");
        }
        Ok(Backend::Graph(device))
    }
    #[cfg(not(feature = "rlx-graph"))]
    anyhow::bail!("rlx-sanotts was built without the `rlx-graph` feature; only --device host works")
}

fn parse_ids(s: &str) -> Result<Vec<i64>> {
    s.split([',', ' '])
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .parse::<i64>()
                .context("phoneme ids must be integers")
        })
        .collect()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Info(a) => {
            let synth = Synthesizer::load(&a.voice_dir)?;
            let m = &synth.pack().manifest;
            println!("package        {}", m.package_name);
            println!("voice          {} ({})", m.voice, m.language);
            println!("sample rate    {} Hz", m.sample_rate);
            println!("hop length     {}", m.hop_length);
            println!("parameters     {}", m.total_parameters);
            println!("length scale   {}", synth.default_length_scale());
            match synth.phoneme_table() {
                Some(t) => println!(
                    "frontend       espeak {} ({} phonemes)",
                    t.espeak_voice,
                    t.id_map.len()
                ),
                None => println!("frontend       (no phoneme config in this package)"),
            }
            let mut comps: Vec<_> = m.components.iter().collect();
            comps.sort_by_key(|(k, _)| k.as_str());
            for (name, c) in comps {
                println!("component      {name}: {} tensors", c.tensors.len());
            }
        }
        Command::Ids(a) => {
            let synth = Synthesizer::load(&a.voice.voice_dir)?;
            let ids = match a.phonemes {
                true => synth.ids_from_phonemes(&a.text)?,
                false => synth.phoneme_ids(&a.text)?,
            };
            println!(
                "{}",
                ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
            );
        }
        Command::Say(a) => {
            let mut synth = Synthesizer::load(&a.voice.voice_dir)?;
            synth.set_backend(backend_from_str(&a.device)?);
            let scale = a
                .length_scale
                .unwrap_or_else(|| synth.default_length_scale());
            let started = Instant::now();
            let wav = match (a.ids.as_deref(), a.phonemes.as_deref()) {
                (Some(list), _) => synth.synthesize_ids(&parse_ids(list)?, scale)?,
                (_, Some(ipa)) => synth.synthesize_phonemes(ipa, scale)?,
                _ => synth.synthesize_with(&a.text, scale)?,
            };
            let elapsed = started.elapsed().as_secs_f32();
            wav.write(&a.out)?;
            let secs = wav.duration_secs();
            println!(
                "rlx-sanotts: wrote {} ({secs:.2}s at {} Hz) in {elapsed:.3}s — {:.1}x realtime",
                a.out.display(),
                wav.sample_rate,
                secs / elapsed.max(f32::EPSILON)
            );
        }
    }
    Ok(())
}
