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

//! All-Qwen voice chat: **Qwen3-ASR → Qwen3-0.6B → Qwen3-TTS**.
//!
//! A whisper-free duplex loop — every stage is a Qwen model:
//!   mic PCM ─► Qwen3-ASR (audio encoder + tied-head decoder) ─► text
//!           ─► Qwen3-0.6B chat (skip-thinking, greedy)        ─► reply
//!           ─► Qwen3-TTS voice clone (streamed)               ─► speaker PCM
//!
//! Like `bidirectional_voice_chat` but with Qwen3-ASR in place of Whisper, so
//! the whole pipeline shares one model family. Input is a 16 kHz mono WAV fed in
//! fixed mic chunks; a lightweight energy gate detects end-of-utterance, then
//! each utterance is piped ASR → LLM → TTS and written to `--out-dir`.
//!
//! Quick run:
//! ```sh
//! just fetch-qwen3 && just fetch-qwen3-asr && just fetch-qwen3-tts-base
//! just qwen-voice-chat-demo
//! ```
//!
//! Manual run:
//! ```sh
//! cargo run --release -p rlx-qwen3-tts --features apple-silicon \
//!   --example qwen_voice_chat -- --fast \
//!   --asr-dir .cache/qwen3-asr/Qwen3-ASR-0.6B \
//!   --qwen3-weights weights/Qwen3-0.6B \
//!   --tts-model-dir .cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base \
//!   --ref-wav assets/jfk/jfk_voice_clone.wav \
//!   --input-wav crates/rlx-qwen3-tts/examples/audio/voice_chat_question.wav \
//!   --out-dir /tmp/qwen_voice_chat
//! ```
//!
//! Override default cache paths with `RLX_QWEN3_ASR_DIR`, `RLX_QWEN3_WEIGHTS` /
//! `RLX_QWEN3_DIR`, and `RLX_QWEN3_TTS_DIR`.
//!
//! **`--fast`** (minimum input→output latency): preload + warm every model,
//! Qwen3-ASR + TTS on **Metal (MPS)**, Qwen3 LM on **MLX** (safetensors Metal AR
//! decode is currently incorrect — repeats a token), F16 LM head, empty thinking
//! prefill, `--max-tokens 24`, first-sentence-only TTS, progressive streamed PCM.

#![allow(dead_code)]

use anyhow::{Context, Result, bail, ensure};
use rlx_cli::{
    ChatMessage, ChatTemplate, WeightsResolveCli, parse_standard_device, resolve_weights_cli,
};
use rlx_qwen3::{Precision, Qwen3Runner, SampleOpts};
use rlx_qwen3_asr::AsrRunner;
use rlx_qwen3_asr::audio::{SAMPLE_RATE as ASR_RATE, load_wav_mono_f32};
use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
use rlx_runtime::{Device, is_available};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

const TTS_RATE: u32 = 24_000;
// Same shape as Qwen3 `tokenizer_config.json` chat_template (ChatML).
const QWEN3_CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
const ASSISTANT_HEAD: &str = "<|im_start|>assistant\n";
const ASSISTANT_THINK_PREFILL: &str = "<think>\n\n</think>\n";
/// Spoken "thinking" filler, pre-synthesized per voice and played instantly at
/// turn start to mask the LM+TTS latency (the user hears an acknowledgment while
/// the model is still composing the real reply).
const FILLER_TEXT: &str = "Hmm, let me think.";

struct Args {
    asr_dir: PathBuf,
    qwen3_weights: PathBuf,
    tts_model_dir: PathBuf,
    /// Reference voices to clone (repeat `--ref-wav` for several). The reply is
    /// spoken in `--voice`'s voice; `--cycle-voices` rotates per turn.
    ref_wavs: Vec<PathBuf>,
    /// Selected voice: an index (0-based) or a filename stem. Default: first.
    voice_select: Option<String>,
    /// Rotate through `ref_wavs` one per turn.
    cycle_voices: bool,
    input_wav: PathBuf,
    out_dir: PathBuf,
    /// Device for Qwen3-ASR + TTS (Metal recommended on Apple Silicon).
    device: Device,
    /// Device for Qwen3-ASR specifically (defaults to `device`).
    asr_device: Device,
    /// Qwen3-0.6B LM device — defaults to the fastest correct GPU (`mlx`).
    qwen3_device: Device,
    mic_chunk_ms: u32,
    silence_ms: u32,
    asr_max_tokens: usize,
    max_tokens: usize,
    max_seq: usize,
    tts_max_frames: usize,
    system_prompt: String,
    skip_thinking: bool,
    first_sentence_tts: bool,
    streaming_tts: bool,
    preload: bool,
    gpu_perf: bool,
    stream_lm_tokens: bool,
    /// Prior (user,assistant) exchanges kept in the LM prompt. 0 = stateless
    /// turns — keeps prompt length constant so Qwen3Runner reuses one prefill
    /// bucket (growing history recompiles a new bucket every turn → ~10× slower).
    lm_history: usize,
    /// Capture from the live microphone instead of `--input-wav` (needs the
    /// `mic` cargo feature). Replies play through the default speaker.
    mic: bool,
    /// Play each reply through the default speaker (WAV mode; always on in mic
    /// mode). Needs the `mic` cargo feature.
    play: bool,
    /// Stop after this many turns (mic mode). 0 = run until Ctrl-C.
    max_turns: usize,
    /// Pad each utterance up to a multiple of this many seconds before ASR, so
    /// the shape-keyed encoder/prefill caches are reused across utterances of
    /// different length (real speech varies every turn → otherwise recompiles).
    /// 0 = off.
    asr_bucket_s: f32,
    /// Run the whole Qwen3-TTS pipeline (talker + code predictor + codec conv)
    /// on Metal instead of the default CPU-eager path. NOTE: measured ~25%
    /// slower warm and much slower cold here — the 0.6B talker + 12 Hz codec is
    /// too small to beat Metal dispatch/compile overhead. Opt-in for GPU use.
    tts_gpu: bool,
    /// Mic energy gate sensitivity (RMS). Lower = more sensitive (catches softer
    /// speech, fewer dropped pieces); too low may trigger on background noise.
    vad_threshold: f32,
    /// Overlap LM decode with TTS: run the LM on a worker thread and synthesize
    /// each sentence as soon as it closes, while the LM keeps writing the rest.
    overlap: bool,
    /// Pre-roll lookback (ms) retained before the mic gate trips, so the first
    /// word(s) spoken before detection aren't clipped.
    preroll_ms: u32,
    /// Play a pre-synthesized "Hmm, let me think." filler at turn start to mask
    /// LM+TTS latency.
    filler: bool,
    /// Stop the LM after this many spoken sentences (clean boundary, bounded
    /// latency) — small models ramble to the token cap and end mid-word. 0 = off.
    max_sentences: usize,
}

fn parse_args() -> Result<Args> {
    let asr_dir = std::env::var("RLX_QWEN3_ASR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-asr/Qwen3-ASR-0.6B"));
    let qwen3_weights = std::env::var("RLX_QWEN3_WEIGHTS")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("RLX_QWEN3_DIR").map(PathBuf::from))
        .unwrap_or_else(|_| PathBuf::from("weights/Qwen3-0.6B"));
    let tts_model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"));

    let mut a = Args {
        asr_dir,
        qwen3_weights,
        tts_model_dir,
        ref_wavs: Vec::new(),
        voice_select: None,
        cycle_voices: false,
        input_wav: PathBuf::from("crates/rlx-qwen3-tts/examples/audio/voice_chat_question.wav"),
        out_dir: PathBuf::from("/tmp/qwen_voice_chat"),
        device: pick_device("auto")?,
        asr_device: pick_device("auto")?,
        qwen3_device: pick_lm_device(),
        mic_chunk_ms: 150,
        silence_ms: 500,
        asr_max_tokens: 96,
        max_tokens: 24,
        max_seq: 256,
        tts_max_frames: 96,
        system_prompt: "You are a helpful, friendly voice assistant. Answer \
            naturally and concisely, in one or two short spoken sentences."
            .to_string(),
        skip_thinking: true,
        first_sentence_tts: true,
        streaming_tts: true,
        preload: true,
        gpu_perf: false,
        stream_lm_tokens: true,
        lm_history: 2,
        mic: false,
        play: false,
        max_turns: 0,
        asr_bucket_s: 0.0,
        tts_gpu: false,
        vad_threshold: 0.005,
        overlap: true,
        preroll_ms: 1000,
        filler: true,
        max_sentences: 0,
    };
    let mut asr_device_set = false;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let val = |i: usize| -> Result<String> {
        raw.get(i + 1)
            .cloned()
            .with_context(|| format!("missing value for {}", raw[i]))
    };
    let mut i = 0;
    while i < raw.len() {
        let mut step = 1; // boolean flags advance by 1; value flags set step = 2
        match raw[i].as_str() {
            "--asr-dir" => {
                a.asr_dir = PathBuf::from(val(i)?);
                step = 2;
            }
            "--qwen3-weights" => {
                a.qwen3_weights = PathBuf::from(val(i)?);
                step = 2;
            }
            "--tts-model-dir" => {
                a.tts_model_dir = PathBuf::from(val(i)?);
                step = 2;
            }
            "--ref-wav" => {
                a.ref_wavs.push(PathBuf::from(val(i)?));
                step = 2;
            }
            "--voice" => {
                a.voice_select = Some(val(i)?);
                step = 2;
            }
            "--cycle-voices" => a.cycle_voices = true,
            "--no-overlap" => a.overlap = false,
            "--preroll-ms" => {
                a.preroll_ms = val(i)?.parse().context("--preroll-ms")?;
                step = 2;
            }
            "--no-filler" => a.filler = false,
            "--max-sentences" => {
                a.max_sentences = val(i)?.parse().context("--max-sentences")?;
                step = 2;
            }
            "--input-wav" => {
                a.input_wav = PathBuf::from(val(i)?);
                step = 2;
            }
            "--out-dir" => {
                a.out_dir = PathBuf::from(val(i)?);
                step = 2;
            }
            "--device" => {
                a.device = pick_device(&val(i)?)?;
                step = 2;
            }
            "--asr-device" => {
                a.asr_device = parse_standard_device("qwen_voice_chat", &val(i)?)?;
                asr_device_set = true;
                step = 2;
            }
            "--qwen3-device" => {
                a.qwen3_device = parse_standard_device("qwen_voice_chat", &val(i)?)?;
                step = 2;
            }
            "--mic-chunk-ms" => {
                a.mic_chunk_ms = val(i)?.parse().context("--mic-chunk-ms")?;
                step = 2;
            }
            "--silence-ms" => {
                a.silence_ms = val(i)?.parse().context("--silence-ms")?;
                step = 2;
            }
            "--asr-max-tokens" => {
                a.asr_max_tokens = val(i)?.parse().context("--asr-max-tokens")?;
                step = 2;
            }
            "--max-tokens" => {
                a.max_tokens = val(i)?.parse().context("--max-tokens")?;
                step = 2;
            }
            "--max-seq" => {
                a.max_seq = val(i)?.parse().context("--max-seq")?;
                step = 2;
            }
            "--tts-max-frames" => {
                a.tts_max_frames = val(i)?.parse().context("--tts-max-frames")?;
                step = 2;
            }
            "--system-prompt" => {
                a.system_prompt = val(i)?;
                step = 2;
            }
            "--history" => {
                a.lm_history = val(i)?.parse().context("--history")?;
                step = 2;
            }
            "--mic" => {
                a.mic = true;
                a.play = true;
            }
            "--play" => a.play = true,
            "--max-turns" => {
                a.max_turns = val(i)?.parse().context("--max-turns")?;
                step = 2;
            }
            "--asr-bucket-s" => {
                a.asr_bucket_s = val(i)?.parse().context("--asr-bucket-s")?;
                step = 2;
            }
            "--tts-gpu" => a.tts_gpu = true,
            "--vad-threshold" => {
                a.vad_threshold = val(i)?.parse().context("--vad-threshold")?;
                step = 2;
            }
            "--allow-thinking" => a.skip_thinking = false,
            "--full-reply-tts" => a.first_sentence_tts = false,
            "--no-streaming-tts" => a.streaming_tts = false,
            "--no-preload" => a.preload = false,
            "--gpu-perf" => a.gpu_perf = true,
            "--no-stream-lm-tokens" => a.stream_lm_tokens = false,
            "--fast" => {
                a.max_tokens = 64; // room for a few sentences (EOS still stops early)
                a.max_seq = 192;
                a.tts_max_frames = 96;
                a.skip_thinking = true;
                a.first_sentence_tts = true;
                a.streaming_tts = true;
                a.preload = true;
                a.gpu_perf = true;
                a.mic_chunk_ms = 90; // finer mic granularity → snappier turn detect
                a.silence_ms = 350; // end-of-turn pause (not so short it cuts mid-sentence)
                a.lm_history = 0; // stateless turns → constant prompt → reused prefill bucket
                a.asr_bucket_s = 2.0; // bucket utterance length → reused ASR graphs
                a.first_sentence_tts = false; // speak the whole reply, chopped into sentences
                a.vad_threshold = 0.004; // more sensitive mic gate
                a.max_sentences = 3; // bound rambly small-model replies (clean ending)
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
        i += step;
    }
    if !asr_device_set {
        a.asr_device = a.device;
    }
    if a.ref_wavs.is_empty() {
        a.ref_wavs
            .push(PathBuf::from("assets/jfk/jfk_voice_clone.wav"));
    }
    Ok(a)
}

fn print_help() {
    eprintln!(
        "Usage: qwen_voice_chat \\
  [--asr-dir DIR] [--qwen3-weights PATH] [--tts-model-dir DIR] \\
  [--ref-wav WAV] [--input-wav WAV] [--out-dir DIR] \\
  [--device auto|metal|mlx|cpu] [--asr-device …] [--qwen3-device mlx|cpu|cuda] \\
  [--mic-chunk-ms N] [--silence-ms N] [--asr-max-tokens N] [--max-tokens N] \\
  [--max-seq N] [--tts-max-frames N] [--system-prompt TEXT] \\
  [--fast] [--gpu-perf] [--no-streaming-tts] [--no-preload] \\
  [--allow-thinking] [--full-reply-tts] [--no-stream-lm-tokens] \\
  [--history N] [--mic] [--play] [--max-turns N] [--asr-bucket-s N] \\
  [--ref-wav WAV]... [--voice IDX|NAME] [--cycle-voices] [--vad-threshold F]

  --mic            live microphone → speaker loop (build with --features mic)
                   replies stream to the speaker chunk-by-chunk as synthesized
  --play           play each reply through the speaker (WAV mode)
  --max-turns      stop after N turns in mic mode (0 = until Ctrl-C)
  --asr-bucket-s   pad each utterance to a multiple of N s so ASR graphs are
                   reused across varying-length speech (0 = off; --fast: 2)
  --tts-gpu        run the whole TTS pipeline on Metal (slower here)
  --ref-wav        cloned reference voice; repeat for several voices
  --voice          pick a voice by index or filename stem
  --cycle-voices   rotate through the voices, one per turn
  --vad-threshold  mic gate RMS sensitivity (lower = catches softer speech;
                   default 0.005, --fast 0.004)
  --system-prompt  set the assistant persona (default: helpful voice assistant)
  --no-overlap     synthesize after the full reply instead of overlapping the
                   LM decode with TTS (default: overlap → first audio sooner)
  --preroll-ms     audio kept before the mic gate trips so the first word(s)
                   aren't clipped (default 1000)
  --no-filler      don't play the \"Hmm, let me think.\" thinking filler at
                   turn start (default: play it to mask LM+TTS latency)
  --max-sentences  stop the LM after N spoken sentences (0 = off; --fast: 3) —
                   bounds rambly small-model replies to a clean ending"
    );
}

// ── device selection ────────────────────────────────────────────────────────

fn pick_device(name: &str) -> Result<Device> {
    let d = if name == "auto" {
        if is_available(Device::Metal) {
            Device::Metal
        } else if is_available(Device::Cuda) {
            Device::Cuda
        } else {
            Device::Cpu
        }
    } else {
        parse_standard_device("qwen_voice_chat", name)?
    };
    Ok(d)
}

/// Fastest *correct* LM decode backend. Qwen3 safetensors AR decode on Metal is
/// known-bad (repeats a token); MLX matches CPU and is faster on Apple Silicon.
fn pick_lm_device() -> Device {
    if is_available(Device::Mlx) {
        Device::Mlx
    } else if is_available(Device::Cuda) {
        Device::Cuda
    } else {
        Device::Cpu
    }
}

fn lm_uses_gpu(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

fn apply_gpu_perf_env(args: &Args) {
    // Push the full TTS pipeline onto Metal (talker + code predictor + codec
    // conv). Independent of --gpu-perf so it can be toggled on its own.
    if args.tts_gpu && args.device == Device::Metal {
        unsafe {
            std::env::set_var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE", "1"); // talker
            std::env::set_var("RLX_QWEN3_TTS_CP_METAL", "1"); // code predictor
            std::env::set_var("RLX_QWEN3_TTS_SPEECH_CONV_GPU", "1"); // codec decoder conv
        }
    }
    if !args.gpu_perf {
        return;
    }
    unsafe {
        if std::env::var("VECLIB_MAXIMUM_THREADS").is_err() {
            std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        }
        std::env::set_var("RLX_QWEN3_TTS_PRECOMPILE_BUCKETS", "1");
        std::env::set_var("RLX_QWEN3_TTS_WAV_NORMALIZE", "1");
        if args.device == Device::Metal {
            std::env::set_var("RLX_QWEN3_TTS_GPU_KV", "1");
        }
    }
}

// ── text helpers ─────────────────────────────────────────────────────────────

/// Qwen3-ASR prefixes a `language English` language-ID tag — drop it so the chat
/// model sees only the spoken words.
fn clean_asr_text(raw: &str) -> String {
    let t = raw.trim();
    if t.to_lowercase().starts_with("language ") {
        let rest = t["language".len()..].trim_start();
        // Skip the language-name token (e.g. "English"), keep the transcript.
        if let Some(idx) = rest.find(char::is_whitespace) {
            return rest[idx..].trim_start().to_string();
        }
        return String::new();
    }
    t.to_string()
}

/// Precompute the (prefix, suffix) LM-prompt token ids that wrap the user text in
/// the ChatML template, so ASR transcript ids can splice straight in. `None`
/// unless the `fused-tokens` feature is on and history is off (the splice assumes
/// a fixed template). Matches the string the text path renders + tokenizes.
fn build_fused_template(
    tok: &Tokenizer,
    system: &str,
    skip_thinking: bool,
    lm_history: usize,
) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
    if !cfg!(feature = "fused-tokens") || lm_history != 0 {
        return Ok(None);
    }
    let prefix = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n");
    let mut suffix = String::from("<|im_end|>\n<|im_start|>assistant\n");
    if skip_thinking {
        suffix.push_str(ASSISTANT_THINK_PREFILL);
    }
    Ok(Some((
        encode_prompt(tok, &prefix)?,
        encode_prompt(tok, &suffix)?,
    )))
}

/// Split ASR output ids into (clean transcript text, transcript token ids),
/// dropping the leading `language <lang>` tag on a **token boundary** so the body
/// ids pass straight through to the LM with no re-tokenization. Falls back to
/// re-encoding the clean text only if no token-aligned boundary is found.
fn split_after_lang_tag(asr_ids: &[u32], tok: &Tokenizer) -> (String, Vec<u32>) {
    let decode = |ids: &[u32]| tok.decode(ids, true).unwrap_or_default();
    let full = decode(asr_ids);
    let clean = clean_asr_text(&full);
    if clean == full.trim() {
        return (clean, asr_ids.to_vec()); // no tag present
    }
    for k in 0..asr_ids.len() {
        if decode(&asr_ids[k..]).trim() == clean {
            return (clean, asr_ids[k..].to_vec()); // token-aligned strip
        }
    }
    let ids = tok
        .encode(clean.as_str(), false)
        .map(|e| e.get_ids().to_vec())
        .unwrap_or_default();
    (clean, ids)
}

fn strip_thinking(text: &str) -> String {
    const END: &str = "</think>";
    if let Some(pos) = text.find(END) {
        return text[pos + END.len()..].trim().to_string();
    }
    text.trim().to_string()
}

fn finalize_lm_prompt(mut prompt: String, skip_thinking: bool) -> String {
    if skip_thinking && prompt.ends_with(ASSISTANT_HEAD) {
        prompt.push_str(ASSISTANT_THINK_PREFILL);
    }
    prompt
}

fn first_sentence(text: &str) -> String {
    let t = text.trim();
    for (i, ch) in t.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let rest = t[i + ch.len_utf8()..].trim_start();
            if rest.is_empty() || rest.starts_with(|c: char| c.is_uppercase()) {
                return t[..i + ch.len_utf8()].trim().to_string();
            }
        }
    }
    t.to_string()
}

fn spoken_text(raw: &str, first_sentence_only: bool) -> String {
    let clean = strip_thinking(raw);
    if first_sentence_only {
        first_sentence(&clean)
    } else {
        clean
    }
}

/// True if `s` ends with terminal sentence punctuation (i.e. it's a closed
/// sentence, not a still-growing fragment).
fn ends_sentence(s: &str) -> bool {
    s.trim_end().ends_with(['.', '!', '?'])
}

/// Worth speaking? Skips bare list markers / punctuation like `"1."` so TTS
/// doesn't read "one." before the actual sentence.
fn is_speakable(s: &str) -> bool {
    s.chars().any(|c| c.is_alphabetic())
}

/// Split a reply into sentences (on `.?!` at clause boundaries) so TTS can
/// synthesize + stream them one at a time — audio starts on sentence 1 while the
/// rest are still being synthesized.
fn split_sentences(text: &str) -> Vec<String> {
    let t = strip_thinking(text);
    let t = t.trim();
    if t.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, ch) in t.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let rest = t[i + ch.len_utf8()..].trim_start();
            if rest.is_empty() || rest.starts_with(|c: char| c.is_uppercase()) {
                let s = t[start..i + ch.len_utf8()].trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
                start = i + ch.len_utf8();
            }
        }
    }
    let tail = t[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    if out.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// Resolve `--voice` (index or name) to a voice slot; defaults to the first.
fn resolve_voice_index(
    voices: &[(String, rlx_qwen3_tts::SpeakerReference)],
    sel: Option<&str>,
) -> usize {
    match sel {
        None => 0,
        Some(s) => {
            if let Ok(i) = s.parse::<usize>() {
                return i.min(voices.len().saturating_sub(1));
            }
            voices
                .iter()
                .position(|(n, _)| n.eq_ignore_ascii_case(s))
                .unwrap_or(0)
        }
    }
}

/// TTS needs a closed phrase; without a terminal `.` the codec yields sparse PCM.
fn prepare_tts_text(text: &str) -> String {
    let t = strip_thinking(text);
    let t = t.trim();
    if t.is_empty() {
        return String::new();
    }
    if t.ends_with(['.', '!', '?']) {
        t.to_string()
    } else {
        format!("{t}.")
    }
}

fn encode_prompt(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

fn decode_response(tokenizer: &Tokenizer, ids: &[u32], eos_id: Option<u32>) -> Result<String> {
    let end = eos_id
        .and_then(|eos| ids.iter().position(|&t| t == eos))
        .unwrap_or(ids.len());
    let text = tokenizer
        .decode(&ids[..end], true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    Ok(strip_thinking(&text))
}

fn take_response_ids(generated: &[u32], think_end: Option<u32>, eos_id: Option<u32>) -> Vec<u32> {
    let after_think = think_end
        .and_then(|te| {
            generated
                .iter()
                .position(|&t| t == te)
                .map(|pos| &generated[pos + 1..])
        })
        .unwrap_or(generated);
    match eos_id {
        Some(eos) => after_think
            .iter()
            .copied()
            .take_while(|&t| t != eos)
            .collect(),
        None => after_think.to_vec(),
    }
}

fn resolve_lm_tokenizer(weights: &Path) -> Result<PathBuf> {
    let sibling = weights.with_extension("tokenizer.json");
    if sibling.is_file() {
        return Ok(sibling);
    }
    weights
        .parent()
        .map(|d| d.join("tokenizer.json"))
        .filter(|p| p.is_file())
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json not found next to {}", weights.display()))
}

// ── energy-gate utterance segmentation (whisper-free) ────────────────────────

fn chunk_rms(chunk: &[f32]) -> f32 {
    if chunk.is_empty() {
        return 0.0;
    }
    (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt()
}

/// Strip dead air from a synthesized sentence: drop leading/trailing silence and
/// collapse any internal silent gap longer than ~180 ms down to a ~70 ms pause.
/// The Qwen3-TTS talker often over-runs into silence frames after a short
/// sentence; concatenating those stacks up into long gaps (measured ~1.8 s).
fn clean_tts_pcm(pcm: &[f32], sr: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let fl = (sr as usize / 100).max(1); // 10 ms frames
    let thr = 0.012f32; // speech RMS ≈ 0.1; silence ≈ 0.004
    let n = pcm.len() / fl;
    let voiced: Vec<bool> = (0..n)
        .map(|k| {
            let f = &pcm[k * fl..(k + 1) * fl];
            (f.iter().map(|x| x * x).sum::<f32>() / fl as f32).sqrt() >= thr
        })
        .collect();
    let (Some(first), Some(last)) = (
        voiced.iter().position(|&v| v),
        voiced.iter().rposition(|&v| v),
    ) else {
        return pcm.to_vec(); // all silence — leave it to the caller
    };
    let max_gap = 18usize; // collapse silent runs > 180 ms
    let keep = 7usize; // ...down to 70 ms
    let mut out = Vec::with_capacity((last - first + 1) * fl);
    let mut k = first;
    while k <= last {
        if voiced[k] {
            out.extend_from_slice(&pcm[k * fl..(k + 1) * fl]);
            k += 1;
        } else {
            let g0 = k;
            while k <= last && !voiced[k] {
                k += 1;
            }
            let glen = k - g0;
            let take = if glen > max_gap { keep } else { glen };
            out.extend_from_slice(&pcm[g0 * fl..(g0 + take) * fl]);
        }
    }
    out
}

/// Trim leading/trailing near-silence — less audio → faster ASR.
fn trim_edges(pcm: &[f32], threshold: f32, pad: usize) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let mut start = 0usize;
    let mut end = pcm.len();
    while start < end && pcm[start].abs() < threshold {
        start += 1;
    }
    while end > start && pcm[end - 1].abs() < threshold {
        end -= 1;
    }
    if start >= end {
        return pcm.to_vec();
    }
    let lo = start.saturating_sub(pad);
    let hi = (end + pad).min(pcm.len());
    pcm[lo..hi].to_vec()
}

/// Zero-pad `pcm` up to the next multiple of `bucket` samples. Real speech is a
/// different length every utterance; bucketing collapses those to a handful of
/// fixed shapes so the (shape-keyed) ASR encoder + prefill compile caches hit on
/// turn 2+ instead of recompiling every turn. `bucket == 0` disables it.
fn bucket_pad_audio(pcm: &[f32], bucket: usize) -> Vec<f32> {
    if bucket == 0 || pcm.is_empty() {
        return pcm.to_vec();
    }
    let target = pcm.len().div_ceil(bucket) * bucket;
    let mut out = pcm.to_vec();
    out.resize(target, 0.0);
    out
}

/// Emit an utterance after `silence_ms` of quiet following speech. A rolling
/// **pre-roll** of the last `preroll` samples is always retained, so when speech
/// is detected the utterance is seeded with the audio from *before* the gate
/// tripped — the first word(s) are never clipped by detection lag.
struct UtteranceGate {
    buf: Vec<f32>,
    /// Rolling lookback kept while idle; prepended to `buf` on speech onset.
    preroll: Vec<f32>,
    preroll_max: usize,
    speech_seen: bool,
    silence_samples: usize,
    silence_needed: usize,
    energy_threshold: f32,
}

impl UtteranceGate {
    fn new(silence_ms: u32, threshold: f32, preroll_samples: usize) -> Self {
        Self {
            buf: Vec::new(),
            preroll: Vec::new(),
            preroll_max: preroll_samples,
            speech_seen: false,
            silence_samples: 0,
            silence_needed: (ASR_RATE * silence_ms as usize / 1000).max(1),
            energy_threshold: threshold,
        }
    }

    fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        let speech = chunk_rms(chunk) > self.energy_threshold;
        if !self.speech_seen {
            if speech {
                // Onset: seed the utterance with the pre-roll lookback so the
                // first word (spoken before the gate tripped) is included.
                self.speech_seen = true;
                self.silence_samples = 0;
                self.buf.clear();
                self.buf.append(&mut self.preroll);
                self.buf.extend_from_slice(chunk);
            } else {
                // Idle: keep a rolling pre-roll window.
                self.preroll.extend_from_slice(chunk);
                if self.preroll.len() > self.preroll_max {
                    let drop = self.preroll.len() - self.preroll_max;
                    self.preroll.drain(..drop);
                }
            }
            return None;
        }
        self.buf.extend_from_slice(chunk);
        if speech {
            self.silence_samples = 0;
            return None;
        }
        self.silence_samples += chunk.len();
        if self.silence_samples >= self.silence_needed {
            let utterance = std::mem::take(&mut self.buf);
            self.speech_seen = false;
            self.silence_samples = 0;
            return Some(utterance);
        }
        None
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        if self.speech_seen && self.buf.len() > (ASR_RATE / 5) {
            self.speech_seen = false;
            self.silence_samples = 0;
            Some(std::mem::take(&mut self.buf))
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.preroll.clear();
        self.speech_seen = false;
        self.silence_samples = 0;
    }
}

// ── session ──────────────────────────────────────────────────────────────────

struct Session {
    asr: AsrRunner,
    asr_system: String,
    /// `Option` so it can be moved to a worker thread during overlapped decode.
    lm: Option<Qwen3Runner>,
    lm_tokenizer: Tokenizer,
    lm_template: ChatTemplate,
    eos_id: Option<u32>,
    tts: VoiceClone,
    /// Cloned reference voices (name, x-vector). `voice_idx` selects the active one.
    voices: Vec<(String, rlx_qwen3_tts::SpeakerReference)>,
    voice_idx: usize,
    cycle_voices: bool,
    /// Pre-synthesized "thinking" filler PCM per voice (parallel to `voices`).
    fillers: Vec<Vec<f32>>,
    use_filler: bool,
    max_sentences: usize,
    chat: Vec<ChatMessage>,
    system_prompt: String,
    max_tokens: usize,
    skip_thinking: bool,
    first_sentence_tts: bool,
    streaming_tts: bool,
    stream_lm_tokens: bool,
    lm_history: usize,
    asr_bucket_samples: usize,
    overlap: bool,
    turbo_tts: bool,
    warmed: bool,
    /// `fused-tokens` feature: precomputed (prefix, suffix) LM-prompt token ids
    /// for stateless turns, so ASR transcript ids splice straight in without a
    /// detokenize→retokenize round-trip. `None` = use the text path.
    fused_template: Option<(Vec<u32>, Vec<u32>)>,
}

struct TurnReport {
    user_text: String,
    assistant_text: String,
    tts_text: String,
    asr_secs: f64,
    lm_secs: f64,
    tts_secs: f64,
    tts_ttfa_secs: f64,
    e2e_first_audio_secs: f64,
    reply_pcm: Vec<f32>,
}

fn open_asr(dir: &Path, device: Device, max_tokens: usize) -> Result<AsrRunner> {
    ensure!(
        dir.join("vocab.json").is_file(),
        "missing Qwen3-ASR weights under {} — run `just fetch-qwen3-asr`",
        dir.display()
    );
    AsrRunner::builder()
        .weights(dir)
        .device(device)
        .max_new_tokens(max_tokens)
        .build()
        .with_context(|| format!("open Qwen3-ASR from {}", dir.display()))
}

/// Prefer a sibling `<dir>-gguf/` directory when present — Q4_K_M GGUF with
/// bucketed F32 decode is the fastest path for short 0.6B voice replies (the
/// F32 safetensors decode recompiles per call and is ~10× slower here).
fn resolve_qwen3_weights_path(weights_in: &Path) -> PathBuf {
    if weights_in.is_file() {
        return weights_in.to_path_buf();
    }
    if std::env::var("RLX_QWEN3_PREFER_SAFETENSORS").is_ok() {
        return weights_in.to_path_buf();
    }
    let sibling = weights_in
        .parent()
        .zip(weights_in.file_name())
        .map(|(p, n)| p.join(format!("{}-gguf", n.to_string_lossy())));
    match sibling {
        Some(dir) if dir.is_dir() => dir,
        _ => weights_in.to_path_buf(),
    }
}

fn open_lm(
    weights_in: &Path,
    device: Device,
    max_seq: usize,
    gpu_perf: bool,
) -> Result<(Qwen3Runner, Tokenizer, ChatTemplate, Option<u32>, PathBuf)> {
    let weights_root = resolve_qwen3_weights_path(weights_in);
    let resolve = WeightsResolveCli {
        prefer_gguf: Some("Q4_K_M".into()),
        ..WeightsResolveCli::default()
    };
    let weights = resolve_weights_cli(&weights_root, &resolve)?;
    let is_gguf = weights
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    let tokenizer_path = resolve_lm_tokenizer(&weights)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let eos_id = tokenizer.token_to_id("<|im_end|>");
    let template = ChatTemplate::from_source(QWEN3_CHATML)
        .context("compile Qwen3 chat template")?
        .with_tokens(None, Some("<|im_end|>".into()));

    // Greedy (fast, stops at EOS → concise) plus a repetition penalty to break the
    // 0.6B's greedy loops ("…you're here. …you're here." repeated). Full sampling
    // was tried but it rambles past EOS and runs much slower; greedy+penalty keeps
    // replies short while discouraging echoing.
    let sampler = SampleOpts::greedy().with_repetition_penalty(1.3);
    let mut builder = Qwen3Runner::builder()
        .weights(weights)
        .device(device)
        .max_seq(max_seq)
        .stream(false)
        .sample(sampler);
    if gpu_perf && lm_uses_gpu(device) {
        builder = builder.precision(Precision::F16LmHead);
    }
    // Short voice replies: bucketed F32 decode beats packed prefill-per-token on
    // 0.6B GGUF (same choice as bidirectional_voice_chat --turbo).
    if is_gguf {
        builder = builder.packed_weights(false);
    }
    let mut lm = builder.build().context("open Qwen3-0.6B LM")?;
    if device == Device::Metal {
        lm.disable_decode_compile_cache();
    }
    Ok((lm, tokenizer, template, eos_id, weights_root))
}

impl Session {
    fn open(args: &Args) -> Result<Self> {
        let t0 = Instant::now();
        let asr = open_asr(&args.asr_dir, args.asr_device, args.asr_max_tokens)?;
        let (lm, lm_tokenizer, lm_template, eos_id, lm_path) = open_lm(
            &args.qwen3_weights,
            args.qwen3_device,
            args.max_seq,
            args.gpu_perf,
        )?;
        if lm_path != args.qwen3_weights {
            println!("  LM weights: {} (fast path)", lm_path.display());
        }
        let tts = VoiceClone::open_with_max_frames(
            &args.tts_model_dir,
            args.device,
            args.tts_max_frames,
        )?;
        // Extract a cloned-voice reference per `--ref-wav`.
        let mut voices = Vec::new();
        for w in &args.ref_wavs {
            let name = w
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "voice".into());
            voices.push((name, tts.extract_reference(w)?));
        }
        ensure!(!voices.is_empty(), "no reference voices");
        let voice_idx = resolve_voice_index(&voices, args.voice_select.as_deref());
        if voices.len() > 1 {
            println!(
                "  voices: [{}] (start: {}, cycle={})",
                voices
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                voices[voice_idx].0,
                args.cycle_voices
            );
        }

        let fused_template = build_fused_template(
            &lm_tokenizer,
            &args.system_prompt,
            args.skip_thinking,
            args.lm_history,
        )?;
        if fused_template.is_some() {
            println!("  fused-tokens: ASR→LM token splice (no detokenize/retokenize)");
        }

        let mut session = Self {
            asr,
            asr_system: String::new(),
            lm: Some(lm),
            lm_tokenizer,
            lm_template,
            eos_id,
            tts,
            voices,
            voice_idx,
            cycle_voices: args.cycle_voices,
            fillers: Vec::new(),
            use_filler: args.filler,
            max_sentences: args.max_sentences,
            chat: vec![ChatMessage {
                role: "system".into(),
                content: args.system_prompt.clone(),
            }],
            system_prompt: args.system_prompt.clone(),
            max_tokens: args.max_tokens,
            skip_thinking: args.skip_thinking,
            first_sentence_tts: args.first_sentence_tts,
            streaming_tts: args.streaming_tts,
            stream_lm_tokens: args.stream_lm_tokens,
            lm_history: args.lm_history,
            asr_bucket_samples: (args.asr_bucket_s * ASR_RATE as f32).round() as usize,
            overlap: args.overlap,
            turbo_tts: args.gpu_perf,
            warmed: false,
            fused_template,
        };
        println!(
            "opened Qwen3-ASR ({:?}) + Qwen3-0.6B ({:?}) + Qwen3-TTS ({:?}) in {:.2}s",
            args.asr_device,
            args.qwen3_device,
            args.device,
            t0.elapsed().as_secs_f64()
        );
        if args.preload {
            let warm = session.preload()?;
            session.warmed = true;
            println!("preloaded (warm compile) in {warm:.2}s");
        }
        Ok(session)
    }

    fn preload(&mut self) -> Result<f64> {
        let t = Instant::now();
        // Warm ASR: short silent clip compiles the encoder + decode graphs.
        let silence = vec![0.0f32; ASR_RATE / 2];
        let _ = self.asr.transcribe_pcm(&silence, &self.asr_system)?;
        // Warm LM: one short decode compiles the prefill + decode buckets.
        let chat = vec![
            ChatMessage {
                role: "system".into(),
                content: self.system_prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What is the capital of France?".into(),
            },
        ];
        let prompt = finalize_lm_prompt(
            self.lm_template
                .render(&chat, true)
                .context("warmup chat prompt")?,
            self.skip_thinking,
        );
        let prompt_ids = encode_prompt(&self.lm_tokenizer, &prompt)?;
        let eos_id = self.eos_id;
        let _ = self
            .lm
            .as_mut()
            .context("LM not loaded")?
            .generate_stoppable(&prompt_ids, self.max_tokens, |tok| eos_id != Some(tok))?;

        // Warm TTS + pre-synthesize the "thinking" filler per voice. This compiles
        // the talker/codec graphs (so turn 1 isn't a cold-Metal stall) AND caches
        // the filler PCM so it can play with zero latency at turn start, masking
        // the LM+TTS thinking time.
        let anchor = Instant::now();
        let mut sink = |_: &[f32]| {};
        let cfg = stream_config_for_reply(FILLER_TEXT, self.turbo_tts, self.warmed);
        if self.use_filler {
            let orig = self.voice_idx;
            for vi in 0..self.voices.len() {
                self.voice_idx = vi;
                let mut ttfa = Some(0.0); // quiet during warmup
                let (pcm, _) =
                    self.run_tts(FILLER_TEXT, cfg.clone(), anchor, &mut ttfa, &mut sink)?;
                self.fillers.push(pcm);
            }
            self.voice_idx = orig;
        } else {
            let mut ttfa = Some(0.0);
            let _ = self.run_tts(FILLER_TEXT, cfg, anchor, &mut ttfa, &mut sink)?;
        }
        Ok(t.elapsed().as_secs_f64())
    }

    /// ASR → LM → TTS for one utterance. Returns timings + reply PCM.
    fn handle_turn(
        &mut self,
        user_pcm: &[f32],
        turn_idx: usize,
        out_dir: &Path,
        pcm_sink: &mut dyn FnMut(&[f32]),
    ) -> Result<TurnReport> {
        ensure!(
            user_pcm.len() >= (ASR_RATE / 4),
            "utterance too short for ASR ({} samples)",
            user_pcm.len()
        );
        println!("\n── turn {turn_idx} ──");
        let turn_anchor = Instant::now();

        // 1) Qwen3-ASR — mic PCM → text. Gentle edge-trim (low threshold + 40 ms
        // pad) so soft onsets/endings aren't clipped.
        let audio = trim_edges(user_pcm, 0.003, ASR_RATE / 25);
        let audio = if audio.len() >= (ASR_RATE / 8) {
            audio
        } else {
            user_pcm.to_vec()
        };
        // Bucket the length so the encoder + ASR-prefill graphs are reused across
        // varying-length utterances (otherwise every live turn recompiles).
        let audio = bucket_pad_audio(&audio, self.asr_bucket_samples);
        let t_asr = Instant::now();
        // 1→2) Build the LM prompt. Fused path: splice ASR transcript token ids
        // straight in (shared Qwen3 vocab) — no detokenize→retokenize. Text path:
        // detokenize, render the chat template, re-tokenize.
        let (user_text, prompt_ids) = if let Some((prefix, suffix)) = self.fused_template.clone() {
            let asr_ids = self.asr.transcribe_pcm_ids(&audio, &self.asr_system)?;
            let (text, transcript_ids) = split_after_lang_tag(&asr_ids, &self.lm_tokenizer);
            let mut prompt_ids = prefix;
            prompt_ids.extend_from_slice(&transcript_ids);
            prompt_ids.extend_from_slice(&suffix);
            (text, prompt_ids)
        } else {
            let raw = self.asr.transcribe_pcm(&audio, &self.asr_system)?;
            let user_text = clean_asr_text(&raw);
            ensure!(!user_text.is_empty(), "Qwen3-ASR returned empty transcript");
            self.chat.push(ChatMessage {
                role: "user".into(),
                content: user_text.clone(),
            });
            let chat_for_lm = truncate_chat_history(&self.chat, self.lm_history);
            let prompt = finalize_lm_prompt(
                self.lm_template
                    .render(&chat_for_lm, true)
                    .context("render chat prompt")?,
                self.skip_thinking,
            );
            (user_text, encode_prompt(&self.lm_tokenizer, &prompt)?)
        };
        let asr_secs = t_asr.elapsed().as_secs_f64();
        ensure!(!user_text.is_empty(), "Qwen3-ASR returned empty transcript");
        println!("  🎙  mic → text ({asr_secs:.2}s): {user_text:?}");

        let eos_id = self.eos_id;
        let show_tokens = self.stream_lm_tokens;
        let think_end = self.lm_tokenizer.token_to_id("</think>");
        let max_tokens = self.max_tokens;
        let first_only = self.first_sentence_tts;

        // Pick the voice for this turn.
        if self.cycle_voices && self.voices.len() > 1 {
            self.voice_idx = (self.voice_idx + 1) % self.voices.len();
        }
        if self.voices.len() > 1 {
            println!("  🎚  voice: {}", self.voices[self.voice_idx].0);
        }

        // Play the pre-synthesized "thinking" filler right now (zero latency) so the
        // user hears an acknowledgment while the LM+TTS compose the real reply. Not
        // part of the saved reply WAV; only reaches the speaker (no-op in WAV mode).
        if self.use_filler {
            if let Some(f) = self.fillers.get(self.voice_idx) {
                println!("  💭 (filler: {FILLER_TEXT:?})");
                pcm_sink(f);
            }
        }

        let t_lm = Instant::now();
        let mut pcm: Vec<f32> = Vec::new();
        let mut ttfa: Option<f64> = None;
        let mut first_stats: Option<rlx_qwen3_tts::StreamStats> = None;
        let mut sent_no = 0usize;

        let (assistant_text, lm_secs) = if self.overlap {
            // 2+3) Overlap: the LM decodes on a worker thread (MLX) while THIS
            // thread synthesizes (CPU) each sentence the moment it closes — so the
            // first audio lands while the LM is still writing the rest.
            let tokenizer = self.lm_tokenizer.clone();
            let (tx, rx) = std::sync::mpsc::channel::<u32>();
            let mut lm = self.lm.take().context("LM busy")?;
            let prompt = prompt_ids.clone();
            // Set true the instant the LM finishes — lets the synth trace show that
            // sentence N is synthesized while the LM is *still decoding* the rest.
            let lm_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let lm_done_w = lm_done.clone();
            // Set by the main thread when it detects a repeated sentence — stops the
            // LM early instead of looping ("…you're here. …you're here." forever).
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_w = stop.clone();
            if show_tokens {
                print!("  💬 LM: ");
                let _ = std::io::stdout().flush();
            }
            let worker = std::thread::spawn(move || -> (Qwen3Runner, Vec<u32>, f64) {
                let mut out_ids = Vec::new();
                let _ = lm.generate_stoppable(&prompt, max_tokens, |tok| {
                    if eos_id == Some(tok) || stop_w.load(std::sync::atomic::Ordering::Relaxed) {
                        return false;
                    }
                    out_ids.push(tok);
                    tx.send(tok).is_ok()
                });
                lm_done_w.store(true, std::sync::atomic::Ordering::Relaxed);
                (lm, out_ids, t_lm.elapsed().as_secs_f64())
            });

            let mut answer_ids: Vec<u32> = Vec::new();
            let mut spoken = 0usize;
            let mut seen: Vec<String> = Vec::new();
            let mut loop_err: Option<anyhow::Error> = None;
            for tok in rx.iter() {
                answer_ids.push(tok);
                if show_tokens {
                    if let Ok(piece) = tokenizer.decode(&[tok], false) {
                        print!("{piece}");
                        let _ = std::io::stdout().flush();
                    }
                }
                if first_only {
                    continue; // first-sentence mode: synthesize once, after decode
                }
                let closes = tokenizer
                    .decode(&[tok], false)
                    .map(|p| p.contains(['.', '!', '?']))
                    .unwrap_or(false);
                if !closes {
                    continue;
                }
                let text = decode_response(&tokenizer, &answer_ids, eos_id).unwrap_or_default();
                let sents = split_sentences(&text);
                // Synthesize every sentence that has *closed* (ends in . ! ?) the
                // moment it closes — don't wait for the next sentence or EOS. The
                // streaming player plays sentence N while we synthesize N+1.
                let complete = sents.iter().filter(|s| ends_sentence(s)).count();
                while spoken < complete {
                    let s = sents[spoken].clone();
                    spoken += 1;
                    if !is_speakable(&s) {
                        continue; // skip bare list markers like "1."
                    }
                    // Loop guard: if a sentence repeats one we already spoke, the
                    // model is stuck — stop the LM and don't synthesize the echo.
                    let norm = s.to_lowercase();
                    if seen.contains(&norm) {
                        println!("  ⛔ repeated sentence — stopping (loop)");
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                    seen.push(norm);
                    sent_no += 1;
                    let lm_state = if lm_done.load(std::sync::atomic::Ordering::Relaxed) {
                        "LM done"
                    } else {
                        "LM still decoding"
                    };
                    println!(
                        "  🔊 sentence {sent_no} @ {:.2}s [{lm_state}]: {s:?}",
                        t_lm.elapsed().as_secs_f64()
                    );
                    match self.synth_sentence(&s, turn_anchor, &mut ttfa, pcm_sink) {
                        Ok((spcm, st)) => {
                            pcm.extend_from_slice(&spcm);
                            first_stats.get_or_insert(st);
                        }
                        Err(e) => {
                            loop_err = Some(e);
                            break;
                        }
                    }
                    // Bound rambly replies: stop the LM at a clean sentence boundary.
                    if self.max_sentences > 0 && sent_no >= self.max_sentences {
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                }
                if loop_err.is_some() || stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            if show_tokens {
                println!();
            }
            let (lm, generated, lm_secs) = worker
                .join()
                .map_err(|_| anyhow::anyhow!("LM worker thread panicked"))?;
            self.lm = Some(lm); // always restore before bailing
            if let Some(e) = loop_err {
                return Err(e);
            }
            let response_ids = take_response_ids(&generated, think_end, eos_id);
            let assistant_text = decode_response(&self.lm_tokenizer, &response_ids, eos_id)?;
            ensure!(!assistant_text.is_empty(), "Qwen3 returned empty reply");
            // Synthesize the remaining (final) sentence(s) the loop didn't cover.
            let speak = spoken_text(&assistant_text, first_only);
            let all = split_sentences(&speak);
            let start = if first_only { 0 } else { spoken.min(all.len()) };
            for s in all.iter().skip(start) {
                if self.max_sentences > 0 && sent_no >= self.max_sentences {
                    break;
                }
                if !is_speakable(s) {
                    continue;
                }
                let norm = s.to_lowercase();
                if seen.contains(&norm) {
                    break; // don't speak a repeated sentence
                }
                seen.push(norm);
                sent_no += 1;
                if all.len() > 1 {
                    println!("  🔊 sentence {sent_no}: {s:?}");
                }
                let (spcm, st) = self.synth_sentence(s, turn_anchor, &mut ttfa, pcm_sink)?;
                pcm.extend_from_slice(&spcm);
                first_stats.get_or_insert(st);
            }
            (assistant_text, lm_secs)
        } else {
            // Sequential: full LM decode, then sentence-by-sentence TTS.
            let tokenizer = self.lm_tokenizer.clone();
            if show_tokens {
                print!("  💬 LM: ");
                let _ = std::io::stdout().flush();
            }
            let generated = self
                .lm
                .as_mut()
                .context("LM not loaded")?
                .generate_stoppable(&prompt_ids, max_tokens, |tok| {
                    if eos_id == Some(tok) {
                        return false;
                    }
                    if show_tokens {
                        if let Ok(piece) = tokenizer.decode(&[tok], false) {
                            print!("{piece}");
                            let _ = std::io::stdout().flush();
                        }
                    }
                    true
                })?;
            if show_tokens {
                println!();
            }
            let lm_secs = t_lm.elapsed().as_secs_f64();
            let response_ids = take_response_ids(&generated, think_end, eos_id);
            let assistant_text = decode_response(&self.lm_tokenizer, &response_ids, eos_id)?;
            ensure!(!assistant_text.is_empty(), "Qwen3 returned empty reply");
            let speak = spoken_text(&assistant_text, first_only);
            let sents = split_sentences(&speak);
            for s in sents.iter() {
                if !is_speakable(s) {
                    continue;
                }
                sent_no += 1;
                if sents.len() > 1 {
                    println!("  🔊 sentence {sent_no}: {s:?}");
                }
                let (spcm, st) = self.synth_sentence(s, turn_anchor, &mut ttfa, pcm_sink)?;
                pcm.extend_from_slice(&spcm);
                first_stats.get_or_insert(st);
            }
            (assistant_text, lm_secs)
        };

        self.chat.push(ChatMessage {
            role: "assistant".into(),
            content: assistant_text.clone(),
        });
        let tts_text = spoken_text(&assistant_text, first_only);
        println!("  🤖 reply (LM {lm_secs:.2}s): {assistant_text:?}");

        let stats = first_stats.context("TTS produced no audio")?;
        let turn_total = turn_anchor.elapsed().as_secs_f64();
        let tts_secs = (turn_total - asr_secs - lm_secs).max(0.0);
        println!(
            "  🔊 reply → audio (first audio {:.2}s from turn start, {} sentence(s), {} samples)",
            ttfa.unwrap_or(stats.time_to_first_audio_secs),
            sent_no,
            pcm.len()
        );

        let out_wav = out_dir.join(format!("turn_{turn_idx:02}_reply.wav"));
        rlx_qwen3_tts::runner::write_wav_mono(&out_wav, &pcm, TTS_RATE)?;
        println!("  wrote {}", out_wav.display());

        let tts_ttfa = ttfa.unwrap_or(stats.time_to_first_audio_secs);
        Ok(TurnReport {
            user_text,
            assistant_text,
            tts_text,
            asr_secs,
            lm_secs,
            tts_secs,
            tts_ttfa_secs: tts_ttfa,
            e2e_first_audio_secs: turn_anchor.elapsed().as_secs_f64(),
            reply_pcm: pcm,
        })
    }

    /// Synthesize one sentence in the active voice with a low-latency stream config.
    fn synth_sentence(
        &mut self,
        s: &str,
        turn_anchor: Instant,
        ttfa: &mut Option<f64>,
        sink: &mut dyn FnMut(&[f32]),
    ) -> Result<(Vec<f32>, rlx_qwen3_tts::StreamStats)> {
        let cfg = stream_config_for_reply(s, self.turbo_tts, self.warmed);
        self.run_tts(s, cfg, turn_anchor, ttfa, sink)
    }

    fn run_tts(
        &mut self,
        text: &str,
        stream_cfg: StreamConfig,
        turn_anchor: Instant,
        ttfa: &mut Option<f64>,
        sink: &mut dyn FnMut(&[f32]),
    ) -> Result<(Vec<f32>, rlx_qwen3_tts::StreamStats)> {
        let text = prepare_tts_text(text);
        ensure!(!text.is_empty(), "TTS text empty after prepare");
        let thr = 0.012f32; // skip silent chunks (talker trailing-silence frames)
        if !self.streaming_tts {
            // Batched: synthesize fully, strip dead air, play.
            let raw = self.tts.generate(&self.voices[self.voice_idx].1, &text)?;
            let pcm = with_tail_pause(clean_tts_pcm(&raw, TTS_RATE));
            if ttfa.is_none() {
                *ttfa = Some(turn_anchor.elapsed().as_secs_f64());
                println!(
                    "  ▶ first speaker audio {:.2}s from turn start",
                    ttfa.unwrap()
                );
            }
            sink(&pcm);
            let stats = rlx_qwen3_tts::StreamStats {
                frames_emitted: 0,
                chunks_emitted: 1,
                samples_emitted: pcm.len(),
                audio_secs: pcm.len() as f64 / TTS_RATE as f64,
                wall_secs: 0.0,
                time_to_first_audio_secs: 0.0,
                stopped_early: false,
            };
            return Ok((pcm, stats));
        }
        // Streaming: play chunks as they synthesize (first chunk ≈0.7 s even cold,
        // since the codec horizons are pre-warmed), but DROP silent chunks so the
        // talker's leading/trailing silence frames don't become audible gaps.
        let mut pcm = Vec::new();
        let stats =
            self.tts
                .generate_stream(&self.voices[self.voice_idx].1, &text, stream_cfg, |evt| {
                    if let StreamEvent::Pcm(chunk) = evt {
                        if chunk_rms(&chunk.samples) >= thr {
                            if ttfa.is_none() {
                                *ttfa = Some(turn_anchor.elapsed().as_secs_f64());
                                println!(
                                    "  ▶ first speaker audio {:.2}s from turn start",
                                    ttfa.unwrap()
                                );
                            }
                            sink(&chunk.samples);
                            pcm.extend_from_slice(&chunk.samples);
                        }
                    }
                    StreamControl::Continue
                })?;
        // Small inter-sentence pause for natural cadence (silent chunks were dropped).
        if !pcm.is_empty() {
            let pause = vec![0.0f32; TTS_RATE as usize * 70 / 1000];
            sink(&pause);
            pcm.extend_from_slice(&pause);
        }
        Ok((pcm, stats))
    }
}

/// Append a short trailing pause so concatenated sentences don't run together.
fn with_tail_pause(mut pcm: Vec<f32>) -> Vec<f32> {
    if !pcm.is_empty() {
        pcm.extend(std::iter::repeat_n(0.0f32, TTS_RATE as usize * 70 / 1000));
    }
    pcm
}

/// Keep `system` + the last `keep_pairs` (user,assistant) exchanges + the
/// current trailing (unanswered) user message. `keep_pairs == 0` yields a
/// stateless `[system, current_user]` prompt of constant length.
fn truncate_chat_history(chat: &[ChatMessage], keep_pairs: usize) -> Vec<ChatMessage> {
    let system = chat.first().filter(|m| m.role == "system");
    let rest: Vec<_> = chat
        .iter()
        .skip(system.map(|_| 1).unwrap_or(0))
        .cloned()
        .collect();
    // last (keep_pairs * 2) history msgs + the 1 current user msg
    let keep_msgs = keep_pairs * 2 + 1;
    let start = rest.len().saturating_sub(keep_msgs);
    let mut out = Vec::new();
    if let Some(s) = system {
        out.push(s.clone());
    }
    out.extend_from_slice(&rest[start..]);
    out
}

/// Low-latency stream profile: small progressive buckets for short replies.
fn stream_config_for_reply(text: &str, turbo: bool, warmed: bool) -> StreamConfig {
    if text.len() < 80 {
        return StreamConfig::progressive(4).with_chunk_samples(1_200);
    }
    if turbo && warmed {
        return StreamConfig::realtime_second();
    }
    StreamConfig::progressive(8).with_chunk_samples(2_400)
}

// ── input audio ──────────────────────────────────────────────────────────────

/// Load a mono 16 kHz f32 stream from `path`, resampling if needed.
fn load_input_pcm(path: &Path) -> Result<Vec<f32>> {
    if let Ok(pcm) = load_wav_mono_f32(path) {
        return Ok(pcm);
    }
    let (pcm, rate) = read_wav_f32_with_rate(path)?;
    Ok(resample_linear(&pcm, rate, ASR_RATE as u32))
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn read_wav_f32_with_rate(path: &Path) -> Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len() >= 44, "wav too small");
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bps = u16::from_le_bytes([bytes[34], bytes[35]]);
    ensure!(channels == 1, "expected mono WAV");
    ensure!(bps == 16 || bps == 32, "expected 16- or 32-bit PCM");
    let data_off = find_data_chunk(&bytes)?;
    let pcm_bytes = &bytes[data_off..];
    let pcm = if bps == 16 {
        pcm_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect()
    } else {
        pcm_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    Ok((pcm, rate))
}

fn find_data_chunk(bytes: &[u8]) -> Result<usize> {
    let mut i = 12usize;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            return Ok(i + 8);
        }
        i += 8 + size + (size % 2);
    }
    bail!("wav missing data chunk")
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = parse_args()?;
    apply_gpu_perf_env(&args);
    std::fs::create_dir_all(&args.out_dir)?;

    println!("┌─ Qwen voice chat (ASR → LM → TTS, all Qwen3) ─────────────────");
    println!("│ Qwen3-ASR: {}", args.asr_dir.display());
    println!("│ Qwen3 LM:  {}", args.qwen3_weights.display());
    println!("│ Qwen3-TTS: {}", args.tts_model_dir.display());
    println!(
        "│ voices:    {}{}",
        args.ref_wavs
            .iter()
            .map(|w| w.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        if args.cycle_voices {
            " (cycle per turn)"
        } else {
            ""
        }
    );
    if args.mic {
        println!("│ input:     🎙  live microphone");
    } else {
        println!("│ input:     {}", args.input_wav.display());
    }
    println!("│ out dir:   {}", args.out_dir.display());
    println!(
        "│ backends:  asr={:?}  lm={:?}  tts={:?}",
        args.asr_device, args.qwen3_device, args.device
    );
    println!(
        "│ latency:   preload={} gpu_perf={} streaming_tts={} max_tokens={}",
        args.preload, args.gpu_perf, args.streaming_tts, args.max_tokens
    );
    println!(
        "│ tts:       {}",
        if args.tts_gpu {
            "full pipeline on Metal (--tts-gpu)"
        } else {
            "CPU-eager talker/CP/codec (default, faster)"
        }
    );
    println!("└───────────────────────────────────────────────────────────────");

    if args.qwen3_device == Device::Metal {
        eprintln!(
            "warning: Qwen3 Metal LM decode is known-bad on safetensors — prefer `--qwen3-device mlx`"
        );
    }

    if args.mic {
        // Grab the audio devices BEFORE the (long) model preload so the CoreAudio
        // session stays active throughout — opening the mic after a 30 s+ preload
        // can hit a stale-config error.
        return run_mic(&args);
    }

    let mut session = Session::open(&args)?;
    let input_pcm = load_input_pcm(&args.input_wav)?;
    println!(
        "\ninput audio: {:.2}s @ {} Hz",
        input_pcm.len() as f64 / ASR_RATE as f64,
        ASR_RATE
    );

    let chunk = (ASR_RATE * args.mic_chunk_ms as usize / 1000).max(160);
    println!(
        "── streamed mic ingress ({} ms chunks, {} ms silence gate) ──",
        args.mic_chunk_ms, args.silence_ms
    );
    let mut gate = UtteranceGate::new(
        args.silence_ms,
        args.vad_threshold,
        ASR_RATE * args.preroll_ms as usize / 1000,
    );
    let mut reports: Vec<TurnReport> = Vec::new();
    let mut turn_idx = 0usize;
    let mut noop_sink = |_: &[f32]| {};
    for c in input_pcm.chunks(chunk) {
        if let Some(utterance) = gate.push(c) {
            turn_idx += 1;
            let report =
                session.handle_turn(&utterance, turn_idx, &args.out_dir, &mut noop_sink)?;
            maybe_play_reply(&report.reply_pcm, &args);
            reports.push(report);
        }
    }
    if let Some(utterance) = gate.flush() {
        turn_idx += 1;
        let report = session.handle_turn(&utterance, turn_idx, &args.out_dir, &mut noop_sink)?;
        maybe_play_reply(&report.reply_pcm, &args);
        reports.push(report);
    }
    ensure!(
        !reports.is_empty(),
        "no speech utterances detected — try a longer --input-wav or lower --silence-ms"
    );

    let mut full = Vec::new();
    for r in &reports {
        full.extend_from_slice(&r.reply_pcm);
    }
    if !full.is_empty() {
        let combined = args.out_dir.join("session_reply.wav");
        rlx_qwen3_tts::runner::write_wav_mono(&combined, &full, TTS_RATE)?;
        println!("\n✓ combined speaker output → {}", combined.display());
    }

    println!("\n── session summary ({} turn(s)) ──", reports.len());
    let (mut asr_t, mut lm_t, mut tts_t) = (0.0, 0.0, 0.0);
    for (i, r) in reports.iter().enumerate() {
        println!("  {}  user={:?}", i + 1, r.user_text);
        println!("       reply={:?}", r.assistant_text);
        println!(
            "       ASR {:>5.2}s  LM {:>5.2}s  TTS {:>5.2}s  TTFA {:>5.2}s  e2e {:>5.2}s",
            r.asr_secs, r.lm_secs, r.tts_secs, r.tts_ttfa_secs, r.e2e_first_audio_secs
        );
        asr_t += r.asr_secs;
        lm_t += r.lm_secs;
        tts_t += r.tts_secs;
    }
    println!(
        "  total: ASR {asr_t:.2}s  LM {lm_t:.2}s  TTS {tts_t:.2}s  over {} turn(s)",
        reports.len()
    );
    Ok(())
}

// ── live microphone + speaker (opt-in `mic` cargo feature) ───────────────────

#[cfg(feature = "mic")]
fn maybe_play_reply(pcm: &[f32], args: &Args) {
    if args.play {
        if let Err(e) = mic::play_pcm(pcm, TTS_RATE) {
            eprintln!("playback failed: {e}");
        }
    }
}

#[cfg(not(feature = "mic"))]
fn maybe_play_reply(_pcm: &[f32], args: &Args) {
    if args.play {
        eprintln!("--play needs the `mic` cargo feature (rebuild with `--features mic`); skipping");
    }
}

/// Acquire the mic + speaker FIRST (so the CoreAudio session stays live across the
/// long model preload), then open the models and run the live loop.
#[cfg(feature = "mic")]
fn run_mic(args: &Args) -> Result<()> {
    let cap = mic::MicCapture::start()?;
    // One persistent output stream for the whole session; the TTS pipeline pushes
    // PCM into it chunk-by-chunk as audio is synthesized (true streaming playback).
    let player = mic::StreamPlayer::start(TTS_RATE)?;
    println!(
        "🎙  mic + speaker acquired ({} Hz in); loading models…",
        cap.in_rate
    );
    let mut session = Session::open(args)?;
    run_mic_session(&mut session, args, &cap, &player)
}

#[cfg(not(feature = "mic"))]
fn run_mic(_args: &Args) -> Result<()> {
    bail!("--mic requires the `mic` cargo feature: rebuild with `--features apple-silicon,mic`")
}

/// Live half-duplex loop: capture mic → end-of-utterance gate → ASR → LM → TTS →
/// speaker. While a turn is processing/playing, captured audio is discarded so
/// the assistant never transcribes its own voice (no AEC needed).
#[cfg(feature = "mic")]
fn run_mic_session(
    session: &mut Session,
    args: &Args,
    cap: &mic::MicCapture,
    player: &mic::StreamPlayer,
) -> Result<()> {
    use std::time::Duration;
    println!(
        "\n🎙  listening on default mic ({} Hz → {} Hz). Speak, then pause ~{} ms to send. Ctrl-C to quit.",
        cap.in_rate, ASR_RATE, args.silence_ms
    );
    let chunk_ms = args.mic_chunk_ms.max(20) as u64;
    let mut gate = UtteranceGate::new(
        args.silence_ms,
        args.vad_threshold,
        ASR_RATE * args.preroll_ms as usize / 1000,
    );
    cap.clear(); // discard audio buffered during model preload
    let mut turn_idx = 0usize;
    loop {
        std::thread::sleep(Duration::from_millis(chunk_ms));
        let raw = cap.drain();
        if raw.is_empty() {
            continue;
        }
        let chunk16 = resample_linear(&raw, cap.in_rate, ASR_RATE as u32);
        if let Some(utterance) = gate.push(&chunk16) {
            turn_idx += 1;
            player.reset();
            let mut sink = |pcm: &[f32]| player.push(pcm);
            let res = session.handle_turn(&utterance, turn_idx, &args.out_dir, &mut sink);
            if let Err(e) = res {
                eprintln!("  turn {turn_idx} skipped: {e}");
            }
            // Wait for the streamed reply to finish playing (no tail cut), then
            // drop audio captured during compute + playback (half-duplex).
            player.wait_drained();
            cap.clear();
            gate.reset();
            if args.max_turns != 0 && turn_idx >= args.max_turns {
                println!("\nreached --max-turns {}", args.max_turns);
                break;
            }
            println!("\n🎙  listening…");
        }
    }
    Ok(())
}

#[cfg(feature = "mic")]
mod mic {
    use anyhow::{Context, Result, bail};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{FromSample, Sample, SampleFormat, SizedSample};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Pick a usable config from a device's supported ranges (preferring f32 at
    /// 48 k / 16 k). cpal's `default_*_config()` intermittently returns an opaque
    /// CoreAudio error on macOS, so callers retry it then fall back to this.
    fn pick_from_ranges<I: Iterator<Item = cpal::SupportedStreamConfigRange>>(
        ranges: I,
    ) -> Option<cpal::SupportedStreamConfig> {
        let mut best: Option<cpal::SupportedStreamConfig> = None;
        for r in ranges {
            let (min, max) = (r.min_sample_rate().0, r.max_sample_rate().0);
            let want = if (min..=max).contains(&48_000) {
                48_000
            } else if (min..=max).contains(&16_000) {
                16_000
            } else {
                max
            };
            let cfg = r.with_sample_rate(cpal::SampleRate(want));
            if cfg.sample_format() == SampleFormat::F32 {
                return Some(cfg); // prefer f32 — no sample conversion
            }
            best.get_or_insert(cfg);
        }
        best
    }

    fn pick_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
        for _ in 0..3 {
            if let Ok(c) = device.default_input_config() {
                return Ok(c);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let ranges = device.supported_input_configs().context(
            "no input configs — grant mic permission in System Settings → Privacy → Microphone",
        )?;
        pick_from_ranges(ranges).context("no usable input config (mic permission?)")
    }

    fn pick_output_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
        for _ in 0..3 {
            if let Ok(c) = device.default_output_config() {
                return Ok(c);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let ranges = device
            .supported_output_configs()
            .context("no output configs")?;
        pick_from_ranges(ranges).context("no usable output config")
    }

    /// Background mic capture into a shared mono f32 buffer at the device rate.
    pub struct MicCapture {
        _stream: cpal::Stream,
        buf: Arc<Mutex<Vec<f32>>>,
        pub in_rate: u32,
    }

    impl MicCapture {
        pub fn start() -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .context("no default input device (grant mic permission?)")?;
            let supported = pick_input_config(&device)?;
            let in_rate = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let cfg: cpal::StreamConfig = supported.config();
            let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
            let stream = match supported.sample_format() {
                SampleFormat::F32 => build_input::<f32>(&device, &cfg, channels, buf.clone())?,
                SampleFormat::I16 => build_input::<i16>(&device, &cfg, channels, buf.clone())?,
                SampleFormat::U16 => build_input::<u16>(&device, &cfg, channels, buf.clone())?,
                other => bail!("unsupported mic sample format {other:?}"),
            };
            stream.play().context("start mic stream")?;
            Ok(Self {
                _stream: stream,
                buf,
                in_rate,
            })
        }

        /// Take everything captured since the last call (device rate, mono).
        pub fn drain(&self) -> Vec<f32> {
            std::mem::take(&mut *lock(&self.buf))
        }

        pub fn clear(&self) {
            lock(&self.buf).clear();
        }
    }

    fn build_input<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        buf: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let stream = device.build_input_stream(
            cfg,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mut b = lock(&buf);
                for frame in data.chunks(channels.max(1)) {
                    let mut acc = 0.0f32;
                    for &s in frame {
                        acc += f32::from_sample(s);
                    }
                    b.push(acc / channels.max(1) as f32);
                }
            },
            |e| eprintln!("mic stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }

    /// Play mono f32 PCM (at `rate`) through the default speaker, blocking until done.
    pub fn play_pcm(pcm: &[f32], rate: u32) -> Result<()> {
        if pcm.is_empty() {
            return Ok(());
        }
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default output device")?;
        let supported = pick_output_config(&device)?;
        let out_rate = supported.sample_rate().0;
        let channels = supported.channels() as usize;
        let cfg: cpal::StreamConfig = supported.config();
        let data = super::resample_linear(pcm, rate, out_rate);
        let total = data.len();
        let pos = Arc::new(Mutex::new(0usize));
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build_output::<f32>(&device, &cfg, channels, data, pos.clone())?,
            SampleFormat::I16 => build_output::<i16>(&device, &cfg, channels, data, pos.clone())?,
            SampleFormat::U16 => build_output::<u16>(&device, &cfg, channels, data, pos.clone())?,
            other => bail!("unsupported speaker sample format {other:?}"),
        };
        stream.play().context("start speaker stream")?;
        while *lock(&pos) < total {
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(150)); // let the device drain
        Ok(())
    }

    fn build_output<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        data: Vec<f32>,
        pos: Arc<Mutex<usize>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample + FromSample<f32>,
    {
        let stream = device.build_output_stream(
            cfg,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut i = lock(&pos);
                for frame in out.chunks_mut(channels.max(1)) {
                    let s = data.get(*i).copied().unwrap_or(0.0);
                    let v = T::from_sample(s);
                    for c in frame.iter_mut() {
                        *c = v;
                    }
                    if *i < data.len() {
                        *i += 1;
                    }
                }
            },
            |e| eprintln!("speaker stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }

    /// Persistent streaming speaker: one output stream for the whole session that
    /// continuously drains a shared sample queue. The TTS pipeline `push`es PCM
    /// chunks as they are synthesized, so audio starts playing at the first chunk
    /// (≈TTFA) and the stream is never dropped mid-utterance (no tail cut).
    pub struct StreamPlayer {
        _stream: cpal::Stream,
        queue: Arc<Mutex<VecDeque<f32>>>,
        src_rate: u32,
        out_rate: u32,
    }

    impl StreamPlayer {
        pub fn start(src_rate: u32) -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .context("no default output device")?;
            let supported = pick_output_config(&device)?;
            let out_rate = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let cfg: cpal::StreamConfig = supported.config();
            let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
            let stream = match supported.sample_format() {
                SampleFormat::F32 => {
                    build_stream_out::<f32>(&device, &cfg, channels, queue.clone())?
                }
                SampleFormat::I16 => {
                    build_stream_out::<i16>(&device, &cfg, channels, queue.clone())?
                }
                SampleFormat::U16 => {
                    build_stream_out::<u16>(&device, &cfg, channels, queue.clone())?
                }
                other => bail!("unsupported speaker sample format {other:?}"),
            };
            stream.play().context("start speaker stream")?;
            Ok(Self {
                _stream: stream,
                queue,
                src_rate,
                out_rate,
            })
        }

        /// Enqueue a freshly-synthesized PCM chunk (at `src_rate`) for playback.
        pub fn push(&self, pcm_src: &[f32]) {
            if pcm_src.is_empty() {
                return;
            }
            let resampled = super::resample_linear(pcm_src, self.src_rate, self.out_rate);
            let mut q = lock(&self.queue);
            q.extend(resampled);
        }

        /// Drop any queued audio (call at the start of a turn).
        pub fn reset(&self) {
            lock(&self.queue).clear();
        }

        /// Block until the queued reply has finished playing.
        pub fn wait_drained(&self) {
            loop {
                if lock(&self.queue).is_empty() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(Duration::from_millis(180)); // let the device buffer drain
        }
    }

    fn build_stream_out<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        queue: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample + FromSample<f32>,
    {
        let stream = device.build_output_stream(
            cfg,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut q = lock(&queue);
                for frame in out.chunks_mut(channels.max(1)) {
                    let s = q.pop_front().unwrap_or(0.0); // underrun → silence
                    let v = T::from_sample(s);
                    for c in frame.iter_mut() {
                        *c = v;
                    }
                }
            },
            |e| eprintln!("speaker stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }
}
