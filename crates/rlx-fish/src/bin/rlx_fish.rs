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

//! `rlx-fish` — Fish Audio S2-Pro / Drama 3 text-to-speech and the fish codec.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::{Args, Parser, Subcommand};
use rlx_fish::audio::{read_wav_44k, write_wav};
use rlx_fish::codec::{FRAME_SAMPLES, FishCodec, SAMPLE_RATE};
use rlx_fish::sampling::SampleParams;

#[derive(Parser)]
#[command(
    name = "rlx-fish",
    about = "Fish Audio S2-Pro / Drama 3 TTS on rlx backends"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct Common {
    /// Checkpoint directory (`config.json`, `model*.safetensors`,
    /// `tokenizer.json`, `codec.pth`).
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[derive(Args)]
struct Speak {
    #[command(flatten)]
    common: Common,
    /// Text to speak. `<|speaker:N|>` tags switch speakers.
    #[arg(long)]
    text: String,
    #[arg(long)]
    out: PathBuf,
    /// Reference audio to clone (repeat with a matching `--ref-text`).
    #[arg(long)]
    ref_wav: Vec<PathBuf>,
    /// Transcript of each `--ref-wav`, in order.
    #[arg(long)]
    ref_text: Vec<String>,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.7)]
    top_p: f32,
    #[arg(long, default_value_t = 30)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Frames per text chunk (21.5 frames = 1 s of audio).
    #[arg(long, default_value_t = rlx_fish::s2::DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: usize,
    /// UTF-8 bytes per chunk when the text has speaker tags.
    #[arg(long, default_value_t = rlx_fish::s2::DEFAULT_CHUNK_LENGTH)]
    chunk_length: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Synthesize with S2-Pro.
    S2(Speak),
    /// Synthesize with Drama 3.
    Drama3 {
        #[command(flatten)]
        speak: Speak,
        /// Bytes of dequantized experts to keep cached.
        #[arg(long, default_value_t = 4 << 30)]
        expert_cache: usize,
        /// Generate the whole text in one pass instead of chunking it.
        #[arg(long)]
        single_chunk: bool,
    },
    /// Encode a WAV with the codec and decode it again.
    Codec {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        wav: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn references(
    codec: &mut FishCodec,
    wavs: &[PathBuf],
    texts: &[String],
) -> Result<Vec<rlx_fish::s2::Reference>> {
    ensure!(
        wavs.len() == texts.len(),
        "give one --ref-text per --ref-wav"
    );
    wavs.iter()
        .zip(texts)
        .map(|(w, t)| {
            let pcm = read_wav_44k(w)?;
            Ok(rlx_fish::s2::Reference {
                text: t.clone(),
                codes: codec.encode(&pcm)?,
            })
        })
        .collect()
}

fn speak_s2(a: Speak) -> Result<()> {
    let device = rlx_cli::parse_device(&a.common.device)?;
    let dir = &a.common.model_dir;
    let mut codec = FishCodec::open(dir, device)?;
    let refs = references(&mut codec, &a.ref_wav, &a.ref_text)?;
    codec.release();

    let t0 = Instant::now();
    let mut model = rlx_fish::S2Model::open(dir, device, rlx_fish::S2Options::default())?;
    let tok = rlx_fish::prompt::load_tokenizer(dir)?;
    let opts = rlx_fish::s2::SynthOptions {
        params: SampleParams {
            temperature: a.temperature,
            top_p: a.top_p,
            top_k: a.top_k,
        },
        seed: a.seed,
        max_new_tokens: a.max_new_tokens,
        chunk_length: a.chunk_length,
    };
    let codes = rlx_fish::s2::synthesize_codes(&mut model, &tok, &a.text, &refs, &opts)?;
    drop(model);
    let frames = codes[0].len();
    let t_lm = t0.elapsed();
    ensure!(frames > 0, "the model produced no audio");

    let t1 = Instant::now();
    let pcm = codec.decode(&codes)?;
    write_wav(&a.out, &pcm)?;
    let secs = pcm.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "{frames} frames, {secs:.2} s audio; LM {:.1} s, codec {:.1} s → {}",
        t_lm.as_secs_f32(),
        t1.elapsed().as_secs_f32(),
        a.out.display()
    );
    Ok(())
}

fn speak_drama3(a: Speak, expert_cache: usize, single_chunk: bool) -> Result<()> {
    let device = rlx_cli::parse_device(&a.common.device)?;
    let dir = &a.common.model_dir;
    let mut codec = FishCodec::open(dir, device)?;
    let refs: Vec<rlx_fish::drama3_text::Turn> = references(&mut codec, &a.ref_wav, &a.ref_text)?
        .into_iter()
        .map(|r| rlx_fish::drama3_text::Turn {
            text: r.text,
            codes: r.codes,
        })
        .collect();
    codec.release();

    let t0 = Instant::now();
    let opts = rlx_fish::DramaOptions {
        expert_cache_bytes: expert_cache,
        ..Default::default()
    };
    let mut model = rlx_fish::DramaModel::open(dir, device, opts)?;
    let tok = rlx_fish::prompt::load_tokenizer(dir)?;
    let max_new = if a.max_new_tokens == rlx_fish::s2::DEFAULT_MAX_NEW_TOKENS {
        rlx_fish::drama3::DEFAULT_MAX_NEW_TOKENS
    } else {
        a.max_new_tokens
    };
    let opts = rlx_fish::drama3::SynthOptions {
        params: SampleParams {
            temperature: a.temperature,
            top_p: a.top_p,
            top_k: a.top_k,
        },
        seed: a.seed,
        max_new_tokens: max_new,
        single_chunk,
    };
    let codes = rlx_fish::drama3::synthesize_codes(&mut model, &tok, &a.text, &refs, &opts)?;
    let (hits, misses) = model.experts().cache_stats();
    drop(model);
    let frames = codes[0].len();
    let t_lm = t0.elapsed();
    ensure!(frames > 0, "the model produced no audio");
    let t1 = Instant::now();
    let pcm = codec.decode(&codes)?;
    write_wav(&a.out, &pcm)?;
    eprintln!(
        "{frames} frames, {:.2} s audio; LM {:.1} s (experts {hits} hit / {misses} loaded), codec {:.1} s → {}",
        pcm.len() as f32 / SAMPLE_RATE as f32,
        t_lm.as_secs_f32(),
        t1.elapsed().as_secs_f32(),
        a.out.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::S2(a) => speak_s2(a),
        Cmd::Drama3 {
            speak,
            expert_cache,
            single_chunk,
        } => speak_drama3(speak, expert_cache, single_chunk),
        Cmd::Codec { common, wav, out } => {
            let device = rlx_cli::parse_device(&common.device)?;
            let mut codec = FishCodec::open(&common.model_dir, device)?;
            let pcm = read_wav_44k(&wav)?;
            let codes = codec.encode(&pcm)?;
            let mut y = codec.decode(&codes)?;
            y.truncate(pcm.len().max(FRAME_SAMPLES));
            write_wav(&out, &y)?;
            eprintln!("{} frames → {}", codes[0].len(), out.display());
            Ok(())
        }
    }
}
