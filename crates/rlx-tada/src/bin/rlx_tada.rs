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

//! `rlx-tada` — build a voice prompt from reference audio, then speak with it.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rlx_runtime::Device;
use rlx_tada::config::SAMPLE_RATE;
use rlx_tada::head::{CfgSchedule, SolveOptions, TimeSchedule};
use rlx_tada::model::TadaModel;
use rlx_tada::prompt::VoicePrompt;
use rlx_tada::prompt_builder::{PromptBuilder, PromptOptions};
use rlx_tada::synth::SynthOptions;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Parser)]
#[command(name = "rlx-tada", about = "HumeAI TADA voice cloning on rlx backends")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode reference audio + its transcript into a reusable voice prompt.
    Prompt {
        #[arg(long)]
        weights: PathBuf,
        /// Reference audio (any sample rate; mono is taken, stereo is mixed).
        /// Repeat with a matching `--text` to condition on several clips of the
        /// same speaker; they are cleaned individually and then concatenated.
        #[arg(long, required = true)]
        wav: Vec<PathBuf>,
        /// What the reference audio says. Repeat once per `--wav`, in order.
        #[arg(long, required = true)]
        text: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        /// Per-language aligner suffix from `HumeAI/tada-codec`
        /// (`de`, `es`, `fr`, `it`, `ja`, `pl`, `pt`, `ar`, `ch`).
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Encoder bottleneck noise. Upstream uses 0.5; 0 is deterministic.
        #[arg(long, default_value_t = 0.5)]
        latent_noise: f32,
        /// Skip reference-audio cleanup and validation: encode the clip
        /// exactly as supplied. Use when reproducing a known-good prompt
        /// byte-for-byte, not for real recordings.
        #[arg(long)]
        raw_reference: bool,
        #[arg(long, default_value_t = 0x7ada_0000_0000_0002)]
        seed: u64,
    },
    /// Synthesize speech from a voice prompt.
    Speak {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        prompt: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Euler steps in the flow-matching solve.
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// Classifier-free guidance on the acoustic field. 1.0 disables
        /// guidance entirely and halves the work per step.
        #[arg(long, default_value_t = 1.6)]
        cfg: f32,
        /// Guidance on the duration field.
        #[arg(long, default_value_t = 1.0)]
        duration_cfg: f32,
        #[arg(long, default_value_t = 0.9)]
        noise_temperature: f32,
        #[arg(long, default_value_t = 0x7ada_0000_0000_0001)]
        seed: u64,
        /// Keep the leading silence the model asked for.
        #[arg(long)]
        keep_leading_silence: bool,
        /// Keep everything the model emitted after an over-long internal
        /// silence, instead of treating it as a missed stop condition.
        #[arg(long)]
        keep_runaway: bool,
    },
    /// Print what a saved voice prompt contains.
    Info {
        #[arg(long)]
        prompt: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Prompt {
            weights,
            wav,
            text,
            out,
            language,
            device,
            latent_noise,
            raw_reference,
            seed,
        } => {
            let device = parse_device(&device)?;
            if wav.len() != text.len() {
                anyhow::bail!(
                    "got {} --wav and {} --text; each reference clip needs its own transcript",
                    wav.len(),
                    text.len()
                );
            }
            let clips = wav
                .iter()
                .map(|w| read_wav(w))
                .collect::<Result<Vec<_>>>()?;
            for ((pcm, rate), w) in clips.iter().zip(&wav) {
                eprintln!(
                    "[tada] reference: {:.2}s at {rate} Hz  {}",
                    pcm.len() as f32 / *rate as f32,
                    w.display()
                );
            }
            let builder = PromptBuilder::open(&weights, device, language.as_deref())?;
            let opts = PromptOptions {
                latent_noise_std: latent_noise,
                seed,
                preprocess: !raw_reference,
                limits: if raw_reference {
                    None
                } else {
                    PromptOptions::default().limits
                },
                ..PromptOptions::default()
            };
            let refs: Vec<(&[f32], usize, &str)> = clips
                .iter()
                .zip(&text)
                .map(|((pcm, rate), t)| (pcm.as_slice(), *rate as usize, t.as_str()))
                .collect();
            let prompt = builder.build_multi(&refs, &opts)?;
            eprintln!(
                "[tada] aligned {} tokens over {} frames (alignment {:.2})",
                prompt.num_tokens(),
                prompt.audio_frames(),
                prompt.alignment_score
            );
            prompt.save(&out)?;
            eprintln!("[tada] wrote {}", out.display());
        }
        Cmd::Speak {
            weights,
            prompt,
            text,
            out,
            device,
            steps,
            cfg,
            duration_cfg,
            noise_temperature,
            seed,
            keep_leading_silence,
            keep_runaway,
        } => {
            let device = parse_device(&device)?;
            let prompt = VoicePrompt::load(&prompt)?;
            let solve = SolveOptions {
                num_steps: steps,
                acoustic_cfg_scale: cfg,
                duration_cfg_scale: duration_cfg,
                cfg_schedule: CfgSchedule::Cosine,
                time_schedule: TimeSchedule::LogSnr,
                noise_temperature,
            };
            let t_load = std::time::Instant::now();
            let mut model = TadaModel::open(&weights, device, &solve)?;
            eprintln!(
                "[prof] load {:?} rss {} MB",
                t_load.elapsed(),
                rlx_tada::rss_mb()
            );
            let opts = SynthOptions {
                solve,
                seed,
                trim_leading_silence: !keep_leading_silence,
                runaway_trim: if keep_runaway {
                    None
                } else {
                    SynthOptions::default().runaway_trim
                },
            };
            let started = std::time::Instant::now();
            let pcm = model.synthesize(&prompt, &text, &opts)?;
            eprintln!("[prof] peak rss {} MB", rlx_tada::rss_mb());
            let seconds = pcm.len() as f32 / SAMPLE_RATE as f32;
            let elapsed = started.elapsed().as_secs_f32();
            eprintln!(
                "[tada] {seconds:.2}s of audio in {elapsed:.2}s (RTF {:.2}×)",
                seconds / elapsed.max(1e-6)
            );
            write_wav(&out, &pcm)?;
            eprintln!("[tada] wrote {}", out.display());
        }
        Cmd::Info { prompt } => {
            let p = VoicePrompt::load(&prompt)?;
            println!("text:      {}", p.text);
            println!("tokens:    {}", p.num_tokens());
            println!("frames:    {}", p.audio_frames());
            println!(
                "audio:     {:.2}s at {} Hz",
                p.audio_samples as f32 / p.sample_rate as f32,
                p.sample_rate
            );
            println!("latents:   {:?}", p.token_values.dim());
            if p.alignment_score.is_nan() {
                println!("alignment: (not recorded — prompt predates scoring)");
            } else {
                println!(
                    "alignment: {:.2}{}",
                    p.alignment_score,
                    if p.alignment_score < -1.5 {
                        "  <- transcript may not match the audio"
                    } else {
                        ""
                    }
                );
            }
            println!(
                "positions: {:?}{}",
                &p.token_positions[..p.token_positions.len().min(16)],
                if p.token_positions.len() > 16 {
                    " …"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

fn parse_device(s: &str) -> Result<Device> {
    Device::from_str(s).map_err(|e| anyhow::anyhow!("unknown device `{s}`: {e}"))
}

/// Read a WAV as mono f32, returning `(samples, sample_rate)`.
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
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
        bail!("{} has no samples", path.display());
    }
    let ch = spec.channels as usize;
    let mono = if ch <= 1 {
        raw
    } else {
        raw.chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn write_wav(path: &Path, pcm: &[f32]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &v in pcm {
        w.write_sample((v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
