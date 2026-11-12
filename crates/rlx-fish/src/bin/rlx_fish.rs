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

//! `rlx-fish` — Fish Audio S2-Pro / Drama 3 and Audio8 text-to-speech, and the
//! fish codec.

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
    about = "Fish Audio S2-Pro / Drama 3 and Audio8 TTS on rlx backends"
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

/// Unset sampling options come from the checkpoint's `generation_config.json`.
#[derive(Args)]
struct Audio8Speak {
    #[command(flatten)]
    common: Common,
    /// Text to speak.
    #[arg(long)]
    text: String,
    #[arg(long)]
    out: PathBuf,
    /// Reference audio to clone (needs `--ref-text`).
    #[arg(long, requires = "ref_text")]
    ref_wav: Option<PathBuf>,
    /// Exact transcript of `--ref-wav`.
    #[arg(long, requires = "ref_wav")]
    ref_text: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Frames to generate at most (21.5 frames = 1 s of audio); per sentence
    /// with `--batch`.
    #[arg(long)]
    max_new_tokens: Option<usize>,
    /// Generate the text sentence by sentence, this many at once (one decode
    /// step serves them all). Without `--ref-wav`, the first sentence is
    /// generated alone and its voice carries the rest. 1 = the whole text in
    /// one pass, as the reference implementation does.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Play through the default output device while generating (`play`
    /// feature). The codec decodes on its own thread as frames arrive.
    #[arg(long)]
    play: bool,
    /// The codec's device when playing (default: `--device`). Streamed, the
    /// codec runs at ~2.6x real time on Metal; on a busy CPU it falls behind.
    #[arg(long)]
    codec_device: Option<String>,
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
    /// Synthesize with Audio8-TTS-Realistic-Air.
    Audio8(Audio8Speak),
    /// Write the codec as `codec_decoder.safetensors` + `codec_encoder.safetensors`
    /// (for the browser, which fetches what it needs of them in parts).
    ExportWeb {
        #[arg(long)]
        model_dir: PathBuf,
        /// Where to write them (default: the model directory).
        #[arg(long)]
        out: Option<PathBuf>,
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

fn speak_audio8(a: Audio8Speak) -> Result<()> {
    let device = rlx_cli::parse_device(&a.common.device)?;
    let dir = &a.common.model_dir;
    let mut codec = FishCodec::open(dir, device)?;
    let reference = match (&a.ref_wav, &a.ref_text) {
        (Some(wav), Some(text)) => Some(rlx_fish::s2::Reference {
            text: text.clone(),
            codes: codec.encode(&read_wav_44k(wav)?)?,
        }),
        _ => None,
    };
    codec.release();

    let t0 = Instant::now();
    let mut model = rlx_fish::S2Model::open(dir, device, rlx_fish::S2Options::default())?;
    let tok = rlx_fish::prompt::load_tokenizer(dir)?;
    let defaults = rlx_fish::audio8::SynthOptions::from_dir(dir)?;
    let opts = rlx_fish::audio8::SynthOptions {
        params: SampleParams {
            temperature: a.temperature.unwrap_or(defaults.params.temperature),
            top_p: a.top_p.unwrap_or(defaults.params.top_p),
            top_k: a.top_k.unwrap_or(defaults.params.top_k),
        },
        seed: a.seed,
        max_new_tokens: a.max_new_tokens.unwrap_or(defaults.max_new_tokens),
    };
    ensure!(
        a.batch == 1 || !a.play,
        "--play streams one pass; drop --batch to play"
    );
    if a.play {
        return play_audio8(a, model, &tok, reference, opts, t0);
    }
    let out = if a.batch > 1 {
        rlx_fish::audio8::synthesize_batch(
            &mut model,
            &tok,
            &a.text,
            reference.as_ref(),
            &opts,
            a.batch,
        )?
    } else {
        rlx_fish::audio8::synthesize_codes(&mut model, &tok, &a.text, reference.as_ref(), &opts)?
    };
    drop(model);
    let frames = out.codes[0].len();
    let t_lm = t0.elapsed();
    ensure!(frames > 0, "the model produced no audio");
    if !out.finished {
        eprintln!("warning: stopped at the {frames}-frame limit; the speech may be cut off");
    }

    let t1 = Instant::now();
    let pcm = codec.decode(&out.codes)?;
    write_wav(&a.out, &pcm)?;
    eprintln!(
        "{frames} frames, {:.2} s audio; LM {:.1} s, codec {:.1} s → {}",
        pcm.len() as f32 / SAMPLE_RATE as f32,
        t_lm.as_secs_f32(),
        t1.elapsed().as_secs_f32(),
        a.out.display()
    );
    Ok(())
}

/// Generate on this thread, decode and play on another as frames arrive.
#[cfg(feature = "play")]
fn play_audio8(
    a: Audio8Speak,
    mut model: rlx_fish::S2Model,
    tok: &tokenizers::Tokenizer,
    reference: Option<rlx_fish::s2::Reference>,
    opts: rlx_fish::audio8::SynthOptions,
    t0: Instant,
) -> Result<()> {
    use std::sync::mpsc;
    let codec_device =
        rlx_cli::parse_device(a.codec_device.as_deref().unwrap_or(&a.common.device))?;
    let dir = a.common.model_dir.clone();
    let nc = model.cfg.num_codebooks;
    let (tx, rx) = mpsc::channel::<Vec<u32>>();
    let t_gen = std::sync::Arc::new(std::sync::Mutex::new(None::<Instant>));
    let t_gen_player = t_gen.clone();
    // Expected length from the text (~2.1 frames a character, 6 for CJK),
    // refined as frames arrive: playback waits until the rest should be
    // ready before what is queued runs out.
    let mut expect = a
        .text
        .chars()
        .map(|c| {
            if ('\u{2e80}'..='\u{9fff}').contains(&c) || ('\u{ac00}'..='\u{d7af}').contains(&c) {
                6.0
            } else {
                2.1
            }
        })
        .sum::<f64>()
        .max(32.0);
    let max_frames = opts.max_new_tokens.min(model.cfg.max_seq_len);
    let player = std::thread::spawn(move || -> Result<(Vec<f32>, usize, f32)> {
        let mut codec = FishCodec::open(&dir, codec_device)?;
        // Every graph the utterance can need, built while the LM loads.
        codec.prepare(max_frames)?;
        let mut out = rlx_fish::play::StreamPlayer::open(SAMPLE_RATE)?;
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut pcm: Vec<f32> = Vec::new();
        let mut decoded = 0usize;
        let mut first_sound = None;
        let mut open = true;
        while open || decoded < frames.len() {
            // Wait for a frame, then take every one that is ready.
            match rx.recv() {
                Ok(f) => frames.push(f),
                Err(_) => open = false,
            }
            while let Ok(f) = rx.try_recv() {
                frames.push(f);
            }
            // Small pieces first (sound soon), larger ones after (less
            // re-decoding at window edges).
            let every = 16;
            if open && frames.len() - decoded < every {
                continue;
            }
            if frames.is_empty() {
                break;
            }
            let codes: Vec<Vec<u32>> = (0..nc)
                .map(|i| frames.iter().map(|f| f[1 + i]).collect())
                .collect();
            let new = codec.decode_from(&codes, pcm.len())?;
            out.push(&new);
            pcm.extend_from_slice(&new);
            decoded = frames.len();
            // Start with a cushion of one piece and its decode, and only once
            // the frames still to come should arrive before the queue runs dry
            // (when generation is slower than real time, that is later).
            let n = frames.len() as f64;
            if n > expect * 0.9 {
                expect = n * 1.15;
            }
            let began = t_gen_player.lock().unwrap().unwrap_or(t0);
            let per_frame = began.elapsed().as_secs_f64() / n.max(1.0);
            let remaining = (expect - n).max(0.0) * per_frame;
            let can_start = out.buffered() >= 1.0 && (!open || remaining <= out.buffered() * 0.9);
            if first_sound.is_none() && can_start {
                out.start();
                let since = t_gen_player.lock().unwrap().unwrap_or(t0);
                first_sound = Some(since.elapsed().as_secs_f32());
            }
        }
        out.finish();
        let underruns = *out.underruns.lock().unwrap();
        Ok((pcm, underruns, first_sound.unwrap_or(f32::NAN)))
    });
    let t_lm = Instant::now();
    *t_gen.lock().unwrap() = Some(t_lm);
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f32());
    let synth = rlx_fish::audio8::synthesize_codes_with(
        &mut model,
        tok,
        &a.text,
        reference.as_ref(),
        &opts,
        |row| {
            let _ = tx.send(row.to_vec());
        },
    )?;
    let lm_secs = t_lm.elapsed().as_secs_f32();
    drop(tx);
    let (pcm, underruns, first) = player
        .join()
        .map_err(|_| anyhow::anyhow!("player thread panicked"))??;
    let frames = synth.codes[0].len();
    write_wav(&a.out, &pcm)?;
    eprintln!(
        "{frames} frames, {:.2} s audio; generation {lm_secs:.1} s ({:.0} ms/frame, real time is 46 ms), \
         first sound {first:.1} s after generation began, {underruns} underruns → {}",
        pcm.len() as f32 / SAMPLE_RATE as f32,
        lm_secs * 1e3 / frames.max(1) as f32,
        a.out.display()
    );
    Ok(())
}

#[cfg(not(feature = "play"))]
fn play_audio8(
    _: Audio8Speak,
    _: rlx_fish::S2Model,
    _: &tokenizers::Tokenizer,
    _: Option<rlx_fish::s2::Reference>,
    _: rlx_fish::audio8::SynthOptions,
    _: Instant,
) -> Result<()> {
    anyhow::bail!("--play needs the `play` feature (cargo build --features play)")
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
        Cmd::Audio8(a) => speak_audio8(a),
        Cmd::ExportWeb { model_dir, out } => {
            let out = out.unwrap_or_else(|| model_dir.clone());
            std::fs::create_dir_all(&out)?;
            let codec = FishCodec::open(&model_dir, rlx_runtime::Device::Cpu)?;
            let [dec, enc] = codec.export_safetensors(&out)?;
            eprintln!(
                "codec_decoder.safetensors {:.0} MB, codec_encoder.safetensors {:.0} MB → {}",
                dec as f64 / 1e6,
                enc as f64 / 1e6,
                out.display()
            );
            Ok(())
        }
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
