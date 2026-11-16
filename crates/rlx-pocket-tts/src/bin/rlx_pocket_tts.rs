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

//! `rlx-pocket-tts` — Kyutai Pocket TTS on RLX.
//!
//! ```text
//! rlx-pocket-tts say "Hello world." --root weights/tts/pocket-tts-kyutai --voice alba -o hello.wav
//! rlx-pocket-tts say "Bonjour." --root … --language french --voice estelle --device metal
//! rlx-pocket-tts export-voice clip.wav --root <gated checkout> -o me.safetensors
//! rlx-pocket-tts languages
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rlx_pocket_tts::{ModelSpec, PocketTts};

#[derive(Parser)]
#[command(name = "rlx-pocket-tts", about = "Kyutai Pocket TTS on RLX")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Synthesize text to a WAV file.
    Say(SayArgs),
    /// Synthesize many texts: a TSV of `voice<TAB>text<TAB>out.wav` lines,
    /// one model load for all (files already there are skipped, so a run
    /// can be resumed).
    Batch(BatchArgs),
    /// Encode a voice prompt (WAV) into a reusable voice state.
    ExportVoice(ExportArgs),
    /// List the bundled model configs.
    Languages,
}

#[derive(Parser)]
struct ModelArgs {
    /// Model root: a checkout of kyutai/pocket-tts(-without-voice-cloning), a
    /// `languages/<name>` directory, or a flat directory with one model file.
    /// With the `hf-download` feature and no root, the files are fetched.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Bundled config name (see `languages`).
    #[arg(long, default_value = "english")]
    language: String,
    /// Device: auto (the best GPU built in and present: Metal, CUDA, ROCm,
    /// wgpu; else the CPU), cpu, metal, mlx, cuda, rocm, wgpu, vulkan, …
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[derive(Parser)]
struct SayArgs {
    /// Text to synthesize.
    text: String,
    #[command(flatten)]
    model: ModelArgs,
    /// Voice: a name from the model's embeddings/, a voice .safetensors, or
    /// (with voice-cloning weights) a WAV file.
    #[arg(long, default_value = "alba")]
    voice: String,
    /// Output WAV path.
    #[arg(short, long, default_value = "out.wav")]
    out: PathBuf,
    /// Sampling temperature (the model's default when unset).
    #[arg(long)]
    temperature: Option<f32>,
    /// EOS threshold (the model's default when unset).
    #[arg(long)]
    eos_threshold: Option<f32>,
    /// Sampler-head steps (LSD / flow-matching heads).
    #[arg(long, default_value_t = 1)]
    sampler_steps: usize,
    /// Truncate the sampler noise to ±this.
    #[arg(long)]
    noise_clamp: Option<f32>,
    /// Frames generated after EOS (derived per chunk when unset).
    #[arg(long)]
    frames_after_eos: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Parser)]
struct BatchArgs {
    /// The jobs: `voice<TAB>text<TAB>out.wav` per line.
    jobs: PathBuf,
    #[command(flatten)]
    model: ModelArgs,
}

#[derive(Parser)]
struct ExportArgs {
    /// Voice prompt WAV (any rate, mixed to mono).
    wav: PathBuf,
    #[command(flatten)]
    model: ModelArgs,
    /// Keep only the first 30 s of the prompt.
    #[arg(long)]
    truncate: bool,
    /// Output voice-state path.
    #[arg(short, long, default_value = "voice.safetensors")]
    out: PathBuf,
}

/// The best GPU this build has and this machine runs, else the CPU.
fn best_device() -> rlx_runtime::Device {
    use rlx_runtime::Device;
    [Device::Metal, Device::Cuda, Device::Rocm, Device::Gpu]
        .into_iter()
        .find(|d| rlx_runtime::is_available(*d))
        .unwrap_or(Device::Cpu)
}

fn open(model: &ModelArgs, voices: &[&str]) -> Result<PocketTts> {
    let device = match model.device.as_str() {
        "auto" => best_device(),
        name => rlx_runtime::parse_device(name).map_err(|e| anyhow::anyhow!("{e}"))?,
    };
    let root = match &model.root {
        Some(r) => r.clone(),
        None => fetch(&model.language, voices)?,
    };
    let t0 = Instant::now();
    let tts = PocketTts::open(&root, &model.language, device)
        .with_context(|| format!("open {} from {}", model.language, root.display()))?;
    eprintln!(
        "loaded {} ({}) on {device:?} in {:.2}s",
        model.language,
        tts.files().weights.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(tts)
}

#[cfg(feature = "hf-download")]
fn fetch(language: &str, voices: &[&str]) -> Result<PathBuf> {
    eprintln!(
        "fetching {language} from {}…",
        rlx_pocket_tts::download::KYUTAI_REPO
    );
    rlx_pocket_tts::download::fetch_language(
        rlx_pocket_tts::download::KYUTAI_REPO,
        language,
        voices,
    )
}

#[cfg(not(feature = "hf-download"))]
fn fetch(_language: &str, _voices: &[&str]) -> Result<PathBuf> {
    bail!("pass --root (or build with the `hf-download` feature to fetch the model)")
}

fn say(a: SayArgs) -> Result<()> {
    let is_name = !a.voice.contains(['/', '\\', '.']);
    let names: Vec<&str> = if is_name {
        vec![a.voice.as_str()]
    } else {
        vec![]
    };
    let mut tts = open(&a.model, &names)?;
    if let Some(t) = a.temperature {
        tts.options.temperature = t;
    }
    if let Some(e) = a.eos_threshold {
        tts.options.eos_threshold = e;
    }
    tts.options.sampler_steps = a.sampler_steps;
    tts.options.noise_clamp = a.noise_clamp;
    if a.frames_after_eos.is_some() {
        tts.options.frames_after_eos = a.frames_after_eos;
    }
    tts.options.seed = a.seed;

    let voice = if a.voice.to_ascii_lowercase().ends_with(".wav") {
        tts.voice_from_wav(&a.voice, false)?
    } else {
        tts.load_voice(&a.voice)?
    };
    let t0 = Instant::now();
    let audio = tts.generate(&a.text, &voice)?;
    let dt = t0.elapsed().as_secs_f32();
    audio.write_wav(&a.out)?;
    eprintln!(
        "wrote {} — {:.2}s of audio in {dt:.2}s ({:.2}× real time)",
        a.out.display(),
        audio.duration_secs(),
        audio.duration_secs() / dt.max(1e-6)
    );
    Ok(())
}

fn export_voice(a: ExportArgs) -> Result<()> {
    let mut tts = open(&a.model, &[])?;
    let t0 = Instant::now();
    let state = tts.voice_from_wav(&a.wav, a.truncate)?;
    tts.save_voice(&state, &a.out)?;
    eprintln!(
        "wrote {} ({} positions) in {:.2}s",
        a.out.display(),
        state.offset,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn batch(a: BatchArgs) -> Result<()> {
    let jobs = std::fs::read_to_string(&a.jobs).with_context(|| a.jobs.display().to_string())?;
    let jobs: Vec<(&str, &str, PathBuf)> = jobs
        .lines()
        .filter_map(|l| {
            let mut f = l.splitn(3, '\t');
            Some((f.next()?, f.next()?, PathBuf::from(f.next()?)))
        })
        .filter(|(_, _, out)| !out.exists())
        .collect();
    eprintln!("{} to make", jobs.len());
    let mut tts = open(&a.model, &[])?;
    let mut voices = std::collections::HashMap::new();
    let t0 = Instant::now();
    let (mut made, mut audio) = (0usize, 0f32);
    for (voice, text, out) in &jobs {
        if !voices.contains_key(*voice) {
            voices.insert(voice.to_string(), tts.load_voice(voice)?);
        }
        let result = tts.generate(text, &voices[*voice]);
        match result {
            Ok(wav) => {
                if let Some(dir) = out.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                wav.write_wav(out)?;
                made += 1;
                audio += wav.duration_secs();
            }
            Err(e) => eprintln!("skipped {text:?}: {e:#}"),
        }
        if made % 50 == 0 && made > 0 {
            eprintln!(
                "{made} / {} ({:.0}s of audio in {:.0}s)",
                jobs.len(),
                audio,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    eprintln!(
        "made {made} ({:.0}s of audio) in {:.0}s",
        audio,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Say(a) => say(a),
        Command::Batch(a) => batch(a),
        Command::ExportVoice(a) => export_voice(a),
        Command::Languages => {
            for l in ModelSpec::builtin_languages() {
                let s = ModelSpec::builtin(l)?;
                let t = &s.flow_lm.transformer;
                println!(
                    "{l:28} {:2} layers  d={:4}  head={:?}",
                    t.num_layers, t.d_model, s.flow_lm.flow.kind
                );
            }
            if ModelSpec::builtin_languages().is_empty() {
                bail!("no bundled configs");
            }
            Ok(())
        }
    }
}
