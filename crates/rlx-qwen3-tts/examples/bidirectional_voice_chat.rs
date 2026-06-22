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

//! Bidirectional voice chat: streamed mic audio → Whisper ASR → Qwen3-0.6B →
//! Qwen3-TTS voice clone → streamed speaker PCM.
//!
//! Simulates a live duplex session by feeding the input WAV in fixed-size mic
//! chunks, detecting end-of-utterance with a lightweight energy gate, then
//! piping each utterance through ASR → LLM → TTS. Default `--turbo` uses batched
//! `VoiceClone::generate` (same quality path as `jfk_voice_clone`); pass
//! `--streaming-tts` for progressive partial-decode during AR.
//!
//! Quick run (bundled question WAV ships in `examples/audio/`):
//! ```sh
//! just fetch-qwen3 && just fetch-whisper-base && just fetch-qwen3-tts-base
//! just voice-chat-demo
//! ```
//!
//! Manual run:
//! ```sh
//! cargo run --release -p rlx-qwen3-tts --features apple-silicon \
//!   --example bidirectional_voice_chat -- --turbo \
//!   --ref-wav assets/jfk/jfk_voice_clone.wav \
//!   --input-wav crates/rlx-qwen3-tts/examples/audio/voice_chat_question.wav \
//!   --out-dir /tmp/voice_chat_roundtrip
//! ```
//!
//! Set `RLX_QWEN3_WEIGHTS` / `RLX_QWEN3_DIR`, `RLX_QWEN3_TTS_DIR`, and
//! `RLX_WHISPER_DIR` to override default cache paths.
//!
//! **Turbo / `--fast`** (minimum input→output latency):
//! - Preload: warm Whisper + Qwen3 + TTS before the first mic chunk
//! - LM streaming (`--overlap`, on in `--turbo`): token stream + TTS on first closed sentence
//! - Streaming ASR (`--streaming-asr`, on in `--turbo`): partial Whisper while mic chunks
//!   arrive; turn starts with prefetched transcript (0 s ASR at flush)
//! - Mic turns flush on shorter silence (`--silence-ms 300`) without waiting for EOF
//! - Apple Silicon: Whisper + TTS on **Metal (MPS)**; Qwen3 LM on **MLX** (Metal LM
//!   autoregressive decode is incorrect on safetensors until fixed upstream).
//!   RLX routes through GPU/MPS, not Core ML ANE directly.
//! - `--turbo` / `--gpu-perf`: F16 Whisper, F16 LM-head, bucket precompile,
//!   MLX safetensors LM (fastest for short replies), warm-session
//!   `realtime_second()` TTS streaming. Set `RLX_QWEN3_PREFER_GGUF=1` to force
//!   Q4_K_M from `weights/Qwen3-0.6B-gguf/` (`just fetch-qwen3-gguf`).
//! - Empty thinking prefill, `--max-tokens 16`, first-sentence TTS only
//! - Speaker PCM via batched `VoiceClone::generate` (intelligible speech; opt-in
//!   `--streaming-tts` for progressive partial-decode if you accept quality risk)
//!
//! **Streaming pipeline** (`--stream-pipeline`, default on):
//! - LM tokens stream to stdout; TTS PCM streams before the turn fully finishes
//! - Whisper round-trip optional (`--whisper-validate`, off in `--turbo`)

#![allow(dead_code)]

use anyhow::{Context, Result, bail, ensure};
use rlx_aec::{AecConfig, AecSession};
use rlx_cli::{
    ChatMessage, ChatTemplate, WeightsResolveCli, parse_standard_device, resolve_weights_cli,
};
use rlx_qwen3::{Precision, Qwen3Runner, SampleOpts};
use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
use rlx_runtime::{Device, is_available};
use rlx_whisper::WhisperRunner;
use rlx_whisper::vad::VadConfig;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, load_wav_mono_f32, pcm_segments_by_vad_config};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tokenizers::Tokenizer;

const TTS_RATE: u32 = 24_000;
// Same shape as Qwen3 `tokenizer_config.json` chat_template (ChatML).
const QWEN3_CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
/// Skip Qwen3 thinking tokens — prefilled empty block, answer starts immediately.
const ASSISTANT_HEAD: &str = "<|im_start|>assistant\n";
const ASSISTANT_THINK_PREFILL: &str = "<think>\n\n</think>\n";

struct Args {
    qwen3_weights: PathBuf,
    tts_model_dir: PathBuf,
    whisper_dir: PathBuf,
    ref_wav: PathBuf,
    input_wav: PathBuf,
    out_dir: PathBuf,
    /// Device for Whisper + TTS (Metal recommended).
    device: Device,
    /// Qwen3-0.6B LM device — defaults to fastest working GPU (`mlx` on Apple Silicon).
    qwen3_device: Device,
    mic_chunk_ms: u32,
    silence_ms: u32,
    max_tokens: usize,
    max_seq: usize,
    tts_max_frames: usize,
    system_prompt: String,
    /// Skip mic streaming; transcribe whole input once (batch duplex).
    batch: bool,
    /// Low-latency TTS progressive(4) + 8k emit chunks.
    low_latency_tts: bool,
    /// Prefill empty thinking block so Qwen3 answers without a long think phase.
    skip_thinking: bool,
    /// Keep only the first sentence for TTS (spoken precision).
    first_sentence_tts: bool,
    /// Overlap LM decode with TTS synthesis for lower time-to-first-audio.
    stream_pipeline: bool,
    /// Print LM tokens as they are sampled.
    stream_lm_tokens: bool,
    /// Re-transcribe TTS output with Whisper and check it covers the spoken reply.
    whisper_validate: bool,
    /// Warm all models before the first utterance (amortizes compile on turn 1).
    preload: bool,
    /// Run LM on a side thread and pipe the first closed sentence to TTS early.
    overlap_lm_tts: bool,
    /// Turbo TTS profile (shorter max_frames cap unless overridden).
    turbo_tts: bool,
    /// Progressive partial-decode streaming (lower TTFA, can sound non-speech on short replies).
    streaming_tts: bool,
    /// Enable Metal/MPS/MLX env tuning (precompile buckets, GPU KV, F16 paths).
    gpu_perf: bool,
    /// Prefer whisper-tiny.en when present (faster ASR).
    whisper_tiny: bool,
    /// Start TTS after this many LM words (turbo: 3) without waiting for `.?!`.
    early_tts_words: usize,
    /// When true, pipe partial LM text to TTS as soon as `early_tts_words` is met.
    aggressive_early_tts: bool,
    /// Mic end-of-turn gate: `rms` today; `earshot` / `silero` reserved for rlx-vad.
    vad_gate: VadGate,
    /// Prefetch ASR during mic ingress (66% silence) + partial transcripts while speaking.
    streaming_asr: bool,
    /// Experimental native Metal talker decode (parity not guaranteed).
    metal_tts_native: bool,
    /// Run mic chunks through rlx-aec; TTS playback feeds far-end reference.
    aec: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadGate {
    Rms,
}

struct TurnReport {
    user_text: String,
    assistant_text: String,
    asr_secs: f64,
    lm_secs: f64,
    tts_secs: f64,
    tts_ttfa_secs: f64,
    /// Wall from end of ASR to first speaker PCM chunk.
    turn_first_audio_secs: f64,
    /// Wall from start of user audio (incl. ASR) to first speaker PCM chunk.
    e2e_first_audio_secs: f64,
    out_pcm_samples: usize,
    tts_text: String,
    reply_pcm: Vec<f32>,
    whisper_heard: Option<String>,
    whisper_ok: bool,
}

struct LmThreadOutput {
    lm: Qwen3Runner,
    generated: Vec<u32>,
    lm_secs: f64,
}

struct Session {
    whisper: WhisperRunner,
    /// Dedicated ASR for streaming prefetch / partial (keeps main whisper free).
    whisper_stream: Option<WhisperRunner>,
    lm: Option<Qwen3Runner>,
    lm_tokenizer: Tokenizer,
    lm_template: ChatTemplate,
    eos_id: Option<u32>,
    tts: VoiceClone,
    voice_ref: rlx_qwen3_tts::SpeakerReference,
    chat: Vec<ChatMessage>,
    system_prompt: String,
    max_tokens: usize,
    skip_thinking: bool,
    first_sentence_tts: bool,
    stream_pipeline: bool,
    stream_lm_tokens: bool,
    whisper_validate: bool,
    overlap_lm_tts: bool,
    turbo_tts: bool,
    streaming_tts: bool,
    early_tts_words: usize,
    aggressive_early_tts: bool,
    streaming_asr: bool,
    /// TTS/Whisper buckets compiled during preload.
    warmed: bool,
    aec: Option<AecSession>,
}

/// Lightweight streaming gate: emit an utterance after `silence_ms` of quiet
/// audio following speech (simulates end-of-turn on a live mic).
struct StreamUtteranceGate {
    buf: Vec<f32>,
    speech_seen: bool,
    silence_samples: usize,
    silence_needed: usize,
    energy_threshold: f32,
    prefetched_asr: Option<String>,
    prefetch_done: bool,
}

struct PartialAsrTracker {
    last_run: Instant,
    min_interval: std::time::Duration,
    last_text: String,
}

impl StreamUtteranceGate {
    fn new(silence_ms: u32) -> Self {
        Self {
            buf: Vec::new(),
            speech_seen: false,
            silence_samples: 0,
            silence_needed: (WHISPER_RATE as u32 * silence_ms / 1000).max(1) as usize,
            energy_threshold: 0.008,
            prefetched_asr: None,
            prefetch_done: false,
        }
    }

    fn speech_active(&self) -> bool {
        self.speech_seen
    }

    fn active_buffer(&self) -> &[f32] {
        &self.buf
    }

    fn should_prefetch_asr(&self) -> bool {
        self.speech_seen
            && !self.prefetch_done
            && self.silence_samples >= (self.silence_needed * 2 / 3).max(1)
    }

    fn set_prefetch(&mut self, text: String) {
        self.prefetch_done = true;
        if !text.is_empty() {
            self.prefetched_asr = Some(text);
        }
    }

    fn take_prefetch(&mut self) -> Option<String> {
        self.prefetched_asr.take()
    }

    fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        let speech = chunk_rms(chunk) > self.energy_threshold;
        self.buf.extend_from_slice(chunk);
        if speech {
            self.speech_seen = true;
            self.silence_samples = 0;
            return None;
        }
        if !self.speech_seen {
            trim_leading_silence(&mut self.buf, self.silence_needed * 2);
            return None;
        }
        self.silence_samples += chunk.len();
        if self.silence_samples >= self.silence_needed {
            let utterance = std::mem::take(&mut self.buf);
            self.speech_seen = false;
            self.silence_samples = 0;
            self.prefetch_done = false;
            return Some(utterance);
        }
        None
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        if self.speech_seen && self.buf.len() > WHISPER_RATE / 5 {
            let utterance = std::mem::take(&mut self.buf);
            self.speech_seen = false;
            self.silence_samples = 0;
            self.prefetch_done = false;
            Some(utterance)
        } else {
            None
        }
    }
}

impl PartialAsrTracker {
    fn new(interval_ms: u32) -> Self {
        Self {
            last_run: Instant::now() - std::time::Duration::from_secs(60),
            min_interval: std::time::Duration::from_millis(interval_ms.max(100) as u64),
            last_text: String::new(),
        }
    }

    fn maybe_update(&mut self, whisper: &mut WhisperRunner, pcm: &[f32]) -> Option<String> {
        if pcm.len() < WHISPER_RATE / 4 {
            return None;
        }
        if self.last_run.elapsed() < self.min_interval {
            return None;
        }
        self.last_run = Instant::now();
        let text = transcribe_user_audio(whisper, pcm).ok()?;
        if !text.is_empty() {
            if text != self.last_text {
                println!("  partial ASR: {text:?}");
            }
            self.last_text = text.clone();
            return Some(text);
        }
        None
    }

    fn latest(&self) -> Option<String> {
        if self.last_text.is_empty() {
            None
        } else {
            Some(self.last_text.clone())
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut qwen3_weights = std::env::var("RLX_QWEN3_WEIGHTS")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("RLX_QWEN3_DIR").map(PathBuf::from))
        .unwrap_or_else(|_| PathBuf::from("weights/Qwen3-0.6B"));
    let mut tts_model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"));
    let mut whisper_dir = std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/whisper-base.en"));
    let mut ref_wav = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    let mut input_wav = ref_wav.clone();
    let mut out_dir = PathBuf::from("/tmp/voice_chat");
    let mut device = pick_device("auto")?;
    let mut qwen3_device: Option<Device> = None;
    let mut mic_chunk_ms = 200u32;
    let mut silence_ms = 600u32;
    let mut max_tokens = 16usize;
    let mut max_seq = 256usize;
    let mut tts_max_frames = 64usize;
    let mut system_prompt = "Reply in one short spoken sentence.".to_string();
    let mut batch = false;
    let mut low_latency_tts = true;
    let mut skip_thinking = true;
    let mut first_sentence_tts = true;
    let mut stream_pipeline = true;
    let mut stream_lm_tokens = true;
    let mut whisper_validate = true;
    let mut preload = true;
    let mut overlap_lm_tts = false;
    let mut turbo_tts = false;
    let mut streaming_tts = false;
    let mut gpu_perf = false;
    let mut whisper_tiny = false;
    let mut early_tts_words = 5usize;
    let mut aggressive_early_tts = false;
    let mut vad_gate = VadGate::Rms;
    let mut streaming_asr = false;
    let mut metal_tts_native = false;
    let mut aec = false;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--qwen3-weights" => {
                qwen3_weights = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--tts-model-dir" => {
                tts_model_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--whisper-dir" => {
                whisper_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--ref-wav" => {
                ref_wav = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--input-wav" => {
                input_wav = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--device" => {
                device = pick_device(&raw[i + 1])?;
                i += 2;
            }
            "--qwen3-device" => {
                qwen3_device = Some(parse_standard_device(
                    "bidirectional_voice_chat",
                    &raw[i + 1],
                )?);
                i += 2;
            }
            "--mic-chunk-ms" => {
                mic_chunk_ms = raw[i + 1].parse().context("--mic-chunk-ms")?;
                i += 2;
            }
            "--silence-ms" => {
                silence_ms = raw[i + 1].parse().context("--silence-ms")?;
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = raw[i + 1].parse().context("--max-tokens")?;
                i += 2;
            }
            "--max-seq" => {
                max_seq = raw[i + 1].parse().context("--max-seq")?;
                i += 2;
            }
            "--tts-max-frames" => {
                tts_max_frames = raw[i + 1].parse().context("--tts-max-frames")?;
                i += 2;
            }
            "--system-prompt" => {
                system_prompt = raw[i + 1].clone();
                i += 2;
            }
            "--batch" => {
                batch = true;
                i += 1;
            }
            "--low-latency-tts" => {
                low_latency_tts = true;
                i += 1;
            }
            "--no-low-latency-tts" => {
                low_latency_tts = false;
                i += 1;
            }
            "--skip-thinking" => {
                skip_thinking = true;
                i += 1;
            }
            "--allow-thinking" => {
                skip_thinking = false;
                i += 1;
            }
            "--first-sentence-tts" => {
                first_sentence_tts = true;
                i += 1;
            }
            "--full-reply-tts" => {
                first_sentence_tts = false;
                i += 1;
            }
            "--stream-pipeline" => {
                stream_pipeline = true;
                i += 1;
            }
            "--sequential" => {
                stream_pipeline = false;
                i += 1;
            }
            "--stream-lm-tokens" => {
                stream_lm_tokens = true;
                i += 1;
            }
            "--no-stream-lm-tokens" => {
                stream_lm_tokens = false;
                i += 1;
            }
            "--no-whisper-validate" => {
                whisper_validate = false;
                i += 1;
            }
            "--whisper-validate" => {
                whisper_validate = true;
                i += 1;
            }
            "--preload" => {
                preload = true;
                i += 1;
            }
            "--no-preload" => {
                preload = false;
                i += 1;
            }
            "--overlap" => {
                overlap_lm_tts = true;
                i += 1;
            }
            "--no-overlap" => {
                overlap_lm_tts = false;
                i += 1;
            }
            "--gpu-perf" => {
                gpu_perf = true;
                i += 1;
            }
            "--no-gpu-perf" => {
                gpu_perf = false;
                i += 1;
            }
            "--whisper-tiny" => {
                whisper_tiny = true;
                i += 1;
            }
            "--fast" => {
                max_tokens = 16;
                max_seq = 128;
                tts_max_frames = 64;
                low_latency_tts = true;
                skip_thinking = true;
                first_sentence_tts = true;
                stream_pipeline = true;
                stream_lm_tokens = true;
                preload = true;
                gpu_perf = true;
                i += 1;
            }
            "--streaming-tts" => {
                streaming_tts = true;
                i += 1;
            }
            "--no-streaming-tts" => {
                streaming_tts = false;
                i += 1;
            }
            "--turbo" => {
                max_tokens = 12;
                max_seq = 64;
                tts_max_frames = 128;
                low_latency_tts = true;
                skip_thinking = true;
                first_sentence_tts = true;
                stream_pipeline = true;
                stream_lm_tokens = false;
                preload = true;
                turbo_tts = true;
                streaming_tts = true;
                gpu_perf = true;
                whisper_tiny = true;
                overlap_lm_tts = true;
                early_tts_words = 5;
                aggressive_early_tts = false;
                streaming_asr = true;
                mic_chunk_ms = 100;
                silence_ms = 300;
                whisper_validate = false;
                i += 1;
            }
            "--streaming-asr" => {
                streaming_asr = true;
                i += 1;
            }
            "--no-streaming-asr" => {
                streaming_asr = false;
                i += 1;
            }
            "--metal-tts-native" => {
                metal_tts_native = true;
                i += 1;
            }
            "--aec" => {
                aec = true;
                i += 1;
            }
            "--early-tts-words" => {
                early_tts_words = raw[i + 1].parse().context("--early-tts-words")?;
                i += 2;
            }
            "--aggressive-early-tts" => {
                aggressive_early_tts = true;
                i += 1;
            }
            "--no-aggressive-early-tts" => {
                aggressive_early_tts = false;
                i += 1;
            }
            "--vad" => {
                vad_gate = match raw[i + 1].as_str() {
                    "rms" => VadGate::Rms,
                    other => bail!(
                        "--vad: expected rms (earshot/silero via rlx-vad coming soon), got {other}"
                    ),
                };
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
    }

    Ok(Args {
        qwen3_weights,
        tts_model_dir,
        whisper_dir,
        ref_wav,
        input_wav,
        out_dir,
        device,
        qwen3_device: qwen3_device.unwrap_or_else(pick_lm_device),
        mic_chunk_ms,
        silence_ms,
        max_tokens,
        max_seq,
        tts_max_frames,
        system_prompt,
        batch,
        low_latency_tts,
        skip_thinking,
        first_sentence_tts,
        stream_pipeline,
        stream_lm_tokens,
        whisper_validate,
        preload,
        overlap_lm_tts,
        turbo_tts,
        streaming_tts,
        gpu_perf,
        whisper_tiny,
        early_tts_words,
        aggressive_early_tts,
        vad_gate,
        streaming_asr,
        metal_tts_native,
        aec,
    })
}

fn print_help() {
    eprintln!(
        "Usage: bidirectional_voice_chat \\
  [--qwen3-weights PATH] [--tts-model-dir DIR] [--whisper-dir DIR] \\
  [--ref-wav WAV] [--input-wav WAV] [--out-dir DIR] [--device auto|metal|cpu] \\
  [--qwen3-device mlx|metal|cpu|cuda] [--mic-chunk-ms N] [--silence-ms N] \\
  [--max-tokens N] [--max-seq N] [--tts-max-frames N] [--system-prompt TEXT] \\
  [--batch] [--fast|--turbo] [--gpu-perf|--no-gpu-perf] \\
  [--whisper-tiny] [--stream-pipeline|--sequential] \\
  [--preload|--no-preload] [--overlap|--no-overlap] \\
  [--streaming-tts|--no-streaming-tts] \\
  [--low-latency-tts|--no-low-latency-tts] \\
  [--skip-thinking|--allow-thinking] [--first-sentence-tts|--full-reply-tts] \\
  [--stream-lm-tokens|--no-stream-lm-tokens] \\
  [--whisper-validate|--no-whisper-validate] \\
  [--aec]"
    );
}

fn push_tts_reference(aec: &mut AecSession, pcm_24k: &[f32]) {
    let pcm_16k = resample_linear(pcm_24k, TTS_RATE, WHISPER_RATE as u32);
    aec.push_reference(&pcm_16k);
}

fn process_mic_aec(aec: Option<&mut AecSession>, mic: &[f32]) -> Vec<f32> {
    if let Some(aec) = aec {
        if let Ok(Some(out)) = aec.process_mic(mic) {
            return out;
        }
    }
    mic.to_vec()
}

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
        parse_standard_device("bidirectional_voice_chat", name)?
    };
    Ok(d)
}

/// Fast LM backend. Metal autoregressive decode on safetensors Qwen3 is currently
/// incorrect (repeats token 83940); MLX matches CPU and is faster on Apple Silicon.
fn whisper_tiny_dir() -> PathBuf {
    PathBuf::from(".cache/whisper-tiny")
}

fn resolve_whisper_dir(requested: &Path, prefer_tiny: bool) -> PathBuf {
    let tiny = whisper_tiny_dir();
    if prefer_tiny && tiny.join("model.safetensors").is_file() {
        return tiny;
    }
    requested.to_path_buf()
}

/// Pick the fastest Qwen3 weights for voice-chat decode on this device.
///
/// MLX + safetensors uses bucketed decode and beats Q4 GGUF packed prefill on
/// short 16-token replies. GGUF is used when safetensors is absent or when
/// `RLX_QWEN3_PREFER_GGUF=1` is set (bucketed F32 decode, not packed prefill).
fn resolve_qwen3_weights_path(weights_in: &Path, device: Device, gpu_perf: bool) -> PathBuf {
    if weights_in.is_file() {
        return weights_in.to_path_buf();
    }
    let prefer_gguf = std::env::var("RLX_QWEN3_PREFER_GGUF")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));
    let has_st = weights_in.join("model.safetensors").is_file();
    let gguf_sibling = weights_in
        .parent()
        .and_then(|p| {
            weights_in
                .file_name()
                .map(|n| p.join(format!("{}-gguf", n.to_string_lossy())))
        })
        .filter(|d| d.is_dir());
    if prefer_gguf {
        if let Some(dir) = gguf_sibling {
            return dir;
        }
    }
    // Default: MLX safetensors for 0.6B voice chat (fastest correct path today).
    if has_st && device == Device::Mlx {
        return weights_in.to_path_buf();
    }
    if gpu_perf {
        if let Some(dir) = gguf_sibling {
            return dir;
        }
    }
    weights_in.to_path_buf()
}

/// Tune RLX env for Metal/MPS/MLX before opening models (call once at startup).
fn apply_gpu_perf_env(args: &Args) {
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
        if args.metal_tts_native {
            std::env::set_var("RLX_QWEN3_TTS_METAL_DECODE_NATIVE", "1");
        }
    }
}

fn lm_uses_gpu(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

fn pick_lm_device() -> Device {
    if is_available(Device::Mlx) {
        Device::Mlx
    } else if is_available(Device::Cuda) {
        Device::Cuda
    } else if is_available(Device::Metal) {
        eprintln!(
            "warning: Qwen3 Metal LM decode is known-bad on safetensors — \
             use `--qwen3-device mlx` on Apple Silicon or `--qwen3-device cpu` for correctness"
        );
        Device::Metal
    } else {
        Device::Cpu
    }
}

fn warn_if_metal_lm(device: Device) {
    if device == Device::Metal {
        eprintln!(
            "warning: `--qwen3-device metal` may produce garbage LM output on safetensors weights; \
             prefer `mlx` (Apple Silicon) or `cpu`"
        );
    }
}

fn chunk_rms(chunk: &[f32]) -> f32 {
    if chunk.is_empty() {
        return 0.0;
    }
    (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt()
}

fn trim_leading_silence(buf: &mut Vec<f32>, max_keep: usize) {
    if buf.len() > max_keep {
        let drain = buf.len() - max_keep;
        buf.drain(..drain);
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

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))
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

/// Qwen3-0.6B may emit a thinking block — keep only user-facing text.
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

fn trim_transcript(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Trim leading/trailing silence before ASR — less audio → faster Whisper.
fn trim_utterance_edges(pcm: &[f32], threshold: f32, pad_samples: usize) -> Vec<f32> {
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
    let lo = start.saturating_sub(pad_samples);
    let hi = (end + pad_samples).min(pcm.len());
    pcm[lo..hi].to_vec()
}

fn transcribe_user_audio(whisper: &mut WhisperRunner, pcm: &[f32]) -> Result<String> {
    let trimmed = trim_utterance_edges(pcm, 0.006, WHISPER_RATE / 50);
    let audio = if trimmed.len() >= WHISPER_RATE / 8 {
        trimmed
    } else {
        pcm.to_vec()
    };
    Ok(trim_transcript(&whisper.transcribe_greedy(&audio)?))
}

/// Use prefetched streaming ASR when available; otherwise run Whisper on the utterance.
fn resolve_user_text(
    whisper: &mut WhisperRunner,
    pcm: &[f32],
    prefetched: Option<String>,
) -> Result<(String, f64, bool)> {
    if let Some(text) = prefetched.filter(|t| !t.is_empty()) {
        println!("  mic → text (stream prefetch 0.00s): {text:?}");
        return Ok((text, 0.0, true));
    }
    let t_asr = Instant::now();
    let text = transcribe_user_audio(whisper, pcm)?;
    let asr_secs = t_asr.elapsed().as_secs_f64();
    Ok((text, asr_secs, false))
}

/// TTS needs a closed phrase; without `.` the codec often yields sparse / inaudible PCM.
fn prepare_tts_text(text: &str) -> String {
    let stripped = strip_thinking(text);
    let t = stripped.trim();
    if t.is_empty() {
        return String::new();
    }
    if t.ends_with(['.', '!', '?']) {
        t.to_string()
    } else {
        format!("{t}.")
    }
}

/// Text safe to hand to TTS mid-LM decode.
fn tts_trigger_text(partial: &str, aggressive: bool) -> Option<String> {
    let cleaned = strip_thinking(partial).trim().to_string();
    if cleaned.is_empty() {
        return None;
    }
    let words = cleaned.split_whitespace().count();
    if aggressive {
        if words < 3 {
            return None;
        }
        let out = prepare_tts_text(&cleaned);
        return if out.is_empty() { None } else { Some(out) };
    }
    speakable_for_tts(&cleaned).map(|s| prepare_tts_text(&s))
}

/// Drop leading near-silence so playback starts on speech.
fn trim_pcm_leading_silence(pcm: &[f32], threshold: f32, pad_samples: usize) -> Vec<f32> {
    let start = pcm.iter().position(|&s| s.abs() > threshold).unwrap_or(0);
    let lo = start.saturating_sub(pad_samples);
    pcm[lo..].to_vec()
}

/// First sentence only — avoids TTS reading lists or follow-up paragraphs aloud.
fn first_sentence(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }
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

/// Natural duplex: frequent small speaker chunks without `progressive(4)` wall-time tax.
fn stream_config_natural(low_latency: bool) -> StreamConfig {
    if low_latency {
        StreamConfig::progressive(12).with_chunk_samples(1_200)
    } else {
        StreamConfig::progressive(8).with_chunk_samples(4_800)
    }
}

fn stream_config_for_reply(
    text: &str,
    low_latency: bool,
    turbo: bool,
    warmed: bool,
) -> StreamConfig {
    if turbo {
        // Short voice replies: progressive(4) beats realtime_second TTFA on <1 s PCM.
        if text.len() < 80 {
            return StreamConfig::progressive(4).with_chunk_samples(1_200);
        }
        if warmed {
            return StreamConfig::realtime_second();
        }
        return StreamConfig::progressive(4).with_chunk_samples(1_200);
    }
    if low_latency {
        if text.len() < 80 {
            // Short replies: moderate bucket + ~50 ms PCM chunks for duplex feel.
            return StreamConfig::progressive(16).with_chunk_samples(1_200);
        }
        return stream_config_natural(true);
    }
    if text.len() < 80 {
        StreamConfig::progressive(16).with_chunk_samples(8_000)
    } else {
        StreamConfig::progressive(8).with_chunk_samples(8_000)
    }
}

/// First closed sentence — safe point to start TTS without waiting for EOS.
fn speakable_for_tts(text: &str) -> Option<String> {
    let cleaned = strip_thinking(text);
    let t = cleaned.trim();
    if t.is_empty() {
        return None;
    }
    for (i, ch) in t.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let rest = t[i + ch.len_utf8()..].trim_start();
            if rest.is_empty() {
                return Some(t[..i + ch.len_utf8()].trim().to_string());
            }
        }
    }
    None
}

/// Avoid O(n²) full detokenize on every LM step; check on punctuation or length.
fn maybe_early_speakable(
    tokenizer: &Tokenizer,
    answer_ids: &[u32],
    eos_id: Option<u32>,
    last_tok: u32,
    min_words: usize,
) -> Option<String> {
    let punct = tokenizer
        .decode(&[last_tok], false)
        .ok()
        .is_some_and(|piece| piece.contains(['.', '!', '?']));
    if !punct && answer_ids.len() < min_words.saturating_mul(2) {
        return None;
    }
    let partial = decode_response(tokenizer, answer_ids, eos_id).ok()?;
    early_speakable(&partial, min_words)
}

/// Returns speakable text once a sentence closes or enough words have arrived.
fn early_speakable(text: &str, min_words: usize) -> Option<String> {
    let cleaned = strip_thinking(text);
    let t = cleaned.trim();
    if t.is_empty() {
        return None;
    }
    for (i, ch) in t.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let rest = t[i + ch.len_utf8()..].trim_start();
            if rest.is_empty() {
                return Some(t[..i + ch.len_utf8()].trim().to_string());
            }
        }
    }
    if t.split_whitespace().count() >= min_words {
        return Some(t.to_string());
    }
    None
}

/// Split on `.?!` boundaries for multi-sentence streaming playback.
fn split_sentences(text: &str) -> Vec<String> {
    let stripped = strip_thinking(text);
    let t = stripped.trim();
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
                while start < t.len() && t.as_bytes()[start] == b' ' {
                    start += 1;
                }
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

struct TtsRun {
    stats: rlx_qwen3_tts::StreamStats,
    pcm: Vec<f32>,
}

const TTS_CHUNK_SAMPLES: usize = 1_200;

fn emit_pcm_chunks(pcm: &[f32], turn_anchor: Instant, first_audio: &Arc<Mutex<Option<f64>>>) {
    for (idx, chunk) in pcm.chunks(TTS_CHUNK_SAMPLES).enumerate() {
        if first_audio.lock().ok().and_then(|g| *g).is_none() {
            if let Ok(mut g) = first_audio.lock() {
                *g = Some(turn_anchor.elapsed().as_secs_f64());
            }
            if let Some(ttfa) = *first_audio.lock().unwrap_or_else(|e| e.into_inner()) {
                println!("  ▶ first speaker audio {ttfa:.2}s from turn start");
            }
        }
        println!("  ♪ {} samples @ {}", chunk.len(), idx * TTS_CHUNK_SAMPLES);
        let _ = std::io::stdout().flush();
    }
}

/// Batched synthesis — same path as `jfk_voice_clone` / `VoiceClone::generate_to_wav`.
fn run_tts_batched(
    tts: &mut VoiceClone,
    voice_ref: &rlx_qwen3_tts::SpeakerReference,
    text: &str,
    turn_anchor: Instant,
    first_audio: &Arc<Mutex<Option<f64>>>,
    aec: Option<&mut AecSession>,
) -> Result<TtsRun> {
    let text = prepare_tts_text(text);
    ensure!(!text.is_empty(), "TTS text empty after prepare");
    let t = Instant::now();
    let pcm = tts.generate(voice_ref, &text)?;
    if let Some(aec) = aec {
        push_tts_reference(aec, &pcm);
    }
    emit_pcm_chunks(&pcm, turn_anchor, first_audio);
    let wall = t.elapsed().as_secs_f64();
    let audio_secs = pcm.len() as f64 / TTS_RATE as f64;
    let stats = rlx_qwen3_tts::StreamStats {
        frames_emitted: 0,
        chunks_emitted: pcm.len().div_ceil(TTS_CHUNK_SAMPLES).max(1),
        samples_emitted: pcm.len(),
        audio_secs,
        wall_secs: wall,
        time_to_first_audio_secs: wall,
        stopped_early: false,
    };
    Ok(TtsRun { stats, pcm })
}

/// Progressive partial-decode streaming — opt-in only (`--streaming-tts`).
fn run_tts_streaming(
    tts: &mut VoiceClone,
    voice_ref: &rlx_qwen3_tts::SpeakerReference,
    text: &str,
    stream_cfg: StreamConfig,
    turn_anchor: Instant,
    first_audio: &Arc<Mutex<Option<f64>>>,
    aec: Option<&mut AecSession>,
) -> Result<TtsRun> {
    let text = prepare_tts_text(text);
    ensure!(!text.is_empty(), "TTS text empty after prepare");
    let mut pcm = Vec::new();
    let stats = tts.generate_stream(voice_ref, &text, stream_cfg, |evt| {
        match evt {
            StreamEvent::Pcm(chunk) => {
                if first_audio.lock().ok().and_then(|g| *g).is_none() {
                    if let Ok(mut g) = first_audio.lock() {
                        *g = Some(turn_anchor.elapsed().as_secs_f64());
                    }
                    if let Some(ttfa) = *first_audio.lock().unwrap_or_else(|e| e.into_inner()) {
                        println!("  ▶ first speaker audio {ttfa:.2}s from turn start");
                    }
                }
                println!(
                    "  ♪ {} samples @ {}",
                    chunk.samples.len(),
                    chunk.sample_offset
                );
                let _ = std::io::stdout().flush();
                pcm.extend_from_slice(&chunk.samples);
            }
            StreamEvent::FrameProduced { frame_index, .. } => {
                print!("\r  synth frame {frame_index}   ");
                let _ = std::io::stdout().flush();
            }
        }
        StreamControl::Continue
    })?;
    if let Some(aec) = aec {
        push_tts_reference(aec, &pcm);
    }
    Ok(TtsRun { stats, pcm })
}

fn run_tts(
    tts: &mut VoiceClone,
    voice_ref: &rlx_qwen3_tts::SpeakerReference,
    text: &str,
    stream_cfg: StreamConfig,
    streaming_tts: bool,
    turn_anchor: Instant,
    first_audio: &Arc<Mutex<Option<f64>>>,
    aec: Option<&mut AecSession>,
) -> Result<TtsRun> {
    if streaming_tts {
        run_tts_streaming(
            tts,
            voice_ref,
            text,
            stream_cfg,
            turn_anchor,
            first_audio,
            aec,
        )
    } else {
        run_tts_batched(tts, voice_ref, text, turn_anchor, first_audio, aec)
    }
}

fn truncate_chat_history(chat: &[ChatMessage], keep_turns: usize) -> Vec<ChatMessage> {
    let system = chat.first().filter(|m| m.role == "system");
    let rest: Vec<_> = chat
        .iter()
        .skip(system.map(|_| 1).unwrap_or(0))
        .cloned()
        .collect();
    let keep_msgs = keep_turns * 2;
    let start = rest.len().saturating_sub(keep_msgs);
    let mut out = Vec::new();
    if let Some(s) = system {
        out.push(s.clone());
    }
    out.extend_from_slice(&rest[start..]);
    out
}

fn open_whisper(dir: &Path, device: Device, f16: bool) -> Result<WhisperRunner> {
    ensure!(
        dir.join("model.safetensors").is_file(),
        "missing Whisper weights under {} — run `just fetch-whisper-base` or `just fetch-whisper-tiny`",
        dir.display()
    );
    let mut b = WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(device)
        .language("en");
    if f16 && matches!(device, Device::Metal | Device::Mlx) {
        b = b.use_f16_compute(true);
    }
    b.build()
}

fn open_lm(
    weights_in: &Path,
    device: Device,
    max_seq: usize,
    max_tokens: usize,
    gpu_perf: bool,
) -> Result<(Qwen3Runner, Tokenizer, ChatTemplate, Option<u32>, PathBuf)> {
    let weights_root = resolve_qwen3_weights_path(weights_in, device, gpu_perf);
    let resolve = WeightsResolveCli {
        prefer_gguf: if gpu_perf {
            Some("Q4_K_M".into())
        } else {
            None
        },
        ..WeightsResolveCli::default()
    };
    let weights = resolve_weights_cli(&weights_root, &resolve)?;
    let is_gguf = weights
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    let tokenizer_path = resolve_lm_tokenizer(&weights)?;
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    // HF publishes the ChatML end marker as `<|im_end|>` in tokenizer_config.
    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id(""));
    let template = ChatTemplate::from_source(QWEN3_CHATML)
        .context("compile Qwen3 chat template")?
        .with_tokens(None, Some("<|im_end|>".into()));

    let mut builder = Qwen3Runner::builder()
        .weights(weights)
        .device(device)
        .max_seq(max_seq)
        .stream(false)
        .sample(SampleOpts::greedy());
    if gpu_perf && lm_uses_gpu(device) {
        builder = builder.precision(Precision::F16LmHead);
    }
    // Short voice replies: bucketed F32 decode beats packed prefill-per-token on 0.6B GGUF.
    if is_gguf && max_seq <= 256 && max_tokens <= 64 {
        builder = builder.packed_weights(false);
    }
    let mut lm = builder.build().context("open Qwen3-0.6B")?;
    // Bucketed Metal decode can drop the latest K row; eager path is still incorrect
    // on safetensors today — prefer MLX via `pick_lm_device()`.
    if device == Device::Metal {
        lm.disable_decode_compile_cache();
    }
    Ok((lm, tokenizer, template, eos_id, weights_root))
}

impl Session {
    fn open(args: &Args) -> Result<Self> {
        let t0 = Instant::now();
        let whisper_dir = resolve_whisper_dir(&args.whisper_dir, args.whisper_tiny);
        if whisper_dir != args.whisper_dir {
            println!(
                "using faster ASR weights: {} (whisper-tiny)",
                whisper_dir.display()
            );
        }
        let whisper_f16 = args.gpu_perf && matches!(args.device, Device::Metal | Device::Mlx);
        let whisper = open_whisper(&whisper_dir, args.device, whisper_f16)?;
        let whisper_stream = if args.streaming_asr {
            let ws = open_whisper(&whisper_dir, args.device, whisper_f16)?;
            println!("  streaming ASR: dedicated whisper prefetch runner");
            Some(ws)
        } else {
            None
        };
        let (lm, lm_tokenizer, lm_template, eos_id, qwen3_resolved) = open_lm(
            &args.qwen3_weights,
            args.qwen3_device,
            args.max_seq,
            args.max_tokens,
            args.gpu_perf,
        )?;
        let tts = VoiceClone::open_with_max_frames(
            &args.tts_model_dir,
            args.device,
            args.tts_max_frames,
        )?;
        let voice_ref = tts.extract_reference(&args.ref_wav)?;
        let aec = if args.aec {
            let s = AecSession::new(AecConfig::default()).context("aec session")?;
            println!("  AEC: enabled (TTS playback → far-end reference)");
            Some(s)
        } else {
            None
        };
        let mut session = Self {
            whisper,
            whisper_stream,
            lm: Some(lm),
            lm_tokenizer,
            lm_template,
            eos_id,
            tts,
            voice_ref,
            chat: vec![ChatMessage {
                role: "system".into(),
                content: args.system_prompt.clone(),
            }],
            system_prompt: args.system_prompt.clone(),
            max_tokens: args.max_tokens,
            skip_thinking: args.skip_thinking,
            first_sentence_tts: args.first_sentence_tts,
            stream_pipeline: args.stream_pipeline,
            stream_lm_tokens: args.stream_lm_tokens,
            whisper_validate: args.whisper_validate,
            overlap_lm_tts: args.overlap_lm_tts,
            turbo_tts: args.turbo_tts,
            streaming_tts: args.streaming_tts,
            early_tts_words: args.early_tts_words,
            aggressive_early_tts: args.aggressive_early_tts,
            streaming_asr: args.streaming_asr,
            warmed: false,
            aec,
        };
        println!(
            "opened whisper + Qwen3 ({:?}, {}) + TTS ({:?}) in {:.2}s",
            args.qwen3_device,
            qwen3_resolved.display(),
            args.device,
            t0.elapsed().as_secs_f64()
        );
        if args.preload {
            let warm = session.preload(args.gpu_perf)?;
            session.warmed = true;
            println!("preloaded (warm compile) in {warm:.2}s");
        }
        Ok(session)
    }

    fn preload(&mut self, _gpu_perf: bool) -> Result<f64> {
        let t = Instant::now();
        let silence = vec![0.0f32; WHISPER_RATE / 4];
        let _ = self.whisper.transcribe_greedy(&silence)?;
        if let Some(ws) = self.whisper_stream.as_mut() {
            let _ = ws.transcribe_greedy(&silence)?;
        }

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
        let max_tokens = self.max_tokens;
        let lm = self.lm.as_mut().context("LM not loaded")?;
        let _ = lm.generate_stoppable(&prompt_ids, max_tokens, |tok| eos_id != Some(tok))?;

        Ok(t.elapsed().as_secs_f64())
    }

    fn finish_turn(
        &mut self,
        turn_idx: usize,
        out_dir: &Path,
        user_text: String,
        assistant_text: String,
        tts_text: String,
        pcm: &[f32],
        asr_secs: f64,
        lm_secs: f64,
        tts_secs: f64,
        tts_ttfa_secs: f64,
        turn_first_audio_secs: f64,
    ) -> Result<TurnReport> {
        let out_wav = out_dir.join(format!("turn_{turn_idx:02}_reply.wav"));
        rlx_qwen3_tts::runner::write_wav_mono(&out_wav, pcm, TTS_RATE)?;
        println!("  wrote {}", out_wav.display());

        let (whisper_heard, whisper_ok) = if self.whisper_validate {
            let (heard, ok, whisper_secs) =
                validate_reply_with_whisper(&mut self.whisper, &tts_text, pcm)?;
            println!("  Whisper round-trip ({whisper_secs:.2}s): {heard:?}");
            if ok {
                println!(
                    "  ✓ reply audio covers TTS text (word recall ≥ {:.0}%)",
                    WHISPER_MIN_RECALL * 100.0
                );
            } else {
                bail!(
                    "Whisper round-trip mismatch on turn {turn_idx}: TTS {:?}, heard {:?}",
                    tts_text,
                    heard
                );
            }
            (Some(heard), ok)
        } else {
            (None, true)
        };

        Ok(TurnReport {
            user_text,
            assistant_text,
            asr_secs,
            lm_secs,
            tts_secs,
            tts_ttfa_secs,
            turn_first_audio_secs,
            e2e_first_audio_secs: asr_secs + turn_first_audio_secs,
            out_pcm_samples: pcm.len(),
            tts_text,
            reply_pcm: pcm.to_vec(),
            whisper_heard,
            whisper_ok,
        })
    }

    fn handle_turn(
        &mut self,
        user_pcm: &[f32],
        turn_idx: usize,
        out_dir: &Path,
        low_latency_tts: bool,
        prefetched_asr: Option<String>,
    ) -> Result<TurnReport> {
        if self.stream_pipeline && self.overlap_lm_tts {
            // LM tokens on a side thread; TTS starts on first closed sentence.
            self.handle_turn_streaming(user_pcm, turn_idx, out_dir, low_latency_tts, prefetched_asr)
        } else if self.stream_pipeline {
            self.handle_turn_pipelined(user_pcm, turn_idx, out_dir, low_latency_tts, prefetched_asr)
        } else {
            self.handle_turn_sequential(
                user_pcm,
                turn_idx,
                out_dir,
                low_latency_tts,
                prefetched_asr,
            )
        }
    }

    fn handle_turn_sequential(
        &mut self,
        user_pcm: &[f32],
        turn_idx: usize,
        out_dir: &Path,
        low_latency_tts: bool,
        prefetched_asr: Option<String>,
    ) -> Result<TurnReport> {
        ensure!(
            user_pcm.len() >= WHISPER_RATE / 4,
            "utterance too short for ASR ({} samples)",
            user_pcm.len()
        );

        println!("\n── turn {turn_idx} ──");
        let (user_text, asr_secs, prefetched) =
            resolve_user_text(&mut self.whisper, user_pcm, prefetched_asr)?;
        if !prefetched {
            println!("  mic → text ({asr_secs:.2}s): {user_text:?}");
        }
        ensure!(!user_text.is_empty(), "Whisper returned empty transcript");

        self.chat.push(ChatMessage {
            role: "user".into(),
            content: user_text.clone(),
        });
        let chat_for_lm = truncate_chat_history(&self.chat, 2);
        let prompt = finalize_lm_prompt(
            self.lm_template
                .render(&chat_for_lm, true)
                .context("render chat prompt")?,
            self.skip_thinking,
        );
        let prompt_ids = encode_prompt(&self.lm_tokenizer, &prompt)?;

        let turn_anchor = Instant::now();
        let first_audio = Arc::new(Mutex::new(None::<f64>));

        let t_lm = Instant::now();
        let eos_id = self.eos_id;
        let generated = self
            .lm
            .as_mut()
            .context("LM in flight on another thread")?
            .generate_stoppable(&prompt_ids, self.max_tokens, |tok| eos_id != Some(tok))?;
        let think_end = self.lm_tokenizer.token_to_id("</think>");
        let response_ids = take_response_ids(&generated, think_end, self.eos_id);
        let assistant_text = decode_response(&self.lm_tokenizer, &response_ids, self.eos_id)?;
        let lm_secs = t_lm.elapsed().as_secs_f64();
        ensure!(!assistant_text.is_empty(), "Qwen3 returned empty reply");
        let tts_text = spoken_text(&assistant_text, self.first_sentence_tts);
        ensure!(!tts_text.is_empty(), "spoken reply empty after trimming");
        log_lm_reply(&assistant_text, &tts_text, lm_secs);
        self.chat.push(ChatMessage {
            role: "assistant".into(),
            content: assistant_text.clone(),
        });

        let stream_cfg =
            stream_config_for_reply(&tts_text, low_latency_tts, self.turbo_tts, self.warmed);
        let t_tts = Instant::now();
        let tts_run = run_tts(
            &mut self.tts,
            &self.voice_ref,
            &tts_text,
            stream_cfg,
            self.streaming_tts,
            turn_anchor,
            &first_audio,
            self.aec.as_mut(),
        )?;
        let tts_secs = t_tts.elapsed().as_secs_f64();
        println!(
            "  reply → audio ({tts_secs:.2}s, synth-TTFA {:.2}s, RTF {:.2}×, {} samples)",
            tts_run.stats.time_to_first_audio_secs,
            tts_run.stats.realtime_factor(),
            tts_run.pcm.len()
        );

        self.finish_turn(
            turn_idx,
            out_dir,
            user_text,
            assistant_text,
            tts_text,
            &tts_run.pcm,
            asr_secs,
            lm_secs,
            tts_secs,
            tts_run.stats.time_to_first_audio_secs,
            first_audio
                .lock()
                .ok()
                .and_then(|g| *g)
                .unwrap_or(tts_run.stats.time_to_first_audio_secs),
        )
    }

    fn handle_turn_streaming(
        &mut self,
        user_pcm: &[f32],
        turn_idx: usize,
        out_dir: &Path,
        low_latency_tts: bool,
        prefetched_asr: Option<String>,
    ) -> Result<TurnReport> {
        ensure!(
            user_pcm.len() >= WHISPER_RATE / 4,
            "utterance too short for ASR ({} samples)",
            user_pcm.len()
        );

        println!("\n── turn {turn_idx} (streaming) ──");
        let (user_text, asr_secs, prefetched) =
            resolve_user_text(&mut self.whisper, user_pcm, prefetched_asr)?;
        if !prefetched {
            println!("  mic → text ({asr_secs:.2}s): {user_text:?}");
        }
        ensure!(!user_text.is_empty(), "Whisper returned empty transcript");

        let turn_anchor = Instant::now();
        let first_audio = Arc::new(Mutex::new(None::<f64>));

        self.chat.push(ChatMessage {
            role: "user".into(),
            content: user_text.clone(),
        });
        let chat_for_lm = truncate_chat_history(&self.chat, 2);
        let prompt = finalize_lm_prompt(
            self.lm_template
                .render(&chat_for_lm, true)
                .context("render chat prompt")?,
            self.skip_thinking,
        );
        let prompt_ids = encode_prompt(&self.lm_tokenizer, &prompt)?;

        let eos_id = self.eos_id;
        let think_end = self.lm_tokenizer.token_to_id("</think>");
        let show_tokens = self.stream_lm_tokens;
        let max_tokens = self.max_tokens;
        let tokenizer = self.lm_tokenizer.clone();

        let mut lm = self
            .lm
            .take()
            .context("LM already running on another thread")?;
        let lm_thread = thread::spawn(move || -> Result<LmThreadOutput> {
            if show_tokens {
                print!("  LM tokens: ");
            }
            let t_lm = Instant::now();
            let generated = lm.generate_stoppable(&prompt_ids, max_tokens, |tok| {
                if show_tokens {
                    if let Ok(piece) = tokenizer.decode(&[tok], false) {
                        print!("{piece}");
                        let _ = std::io::stdout().flush();
                    }
                }
                eos_id != Some(tok)
            })?;
            if show_tokens {
                println!();
            }
            Ok(LmThreadOutput {
                lm,
                generated,
                lm_secs: t_lm.elapsed().as_secs_f64(),
            })
        });

        let lm_out = lm_thread
            .join()
            .map_err(|_| anyhow::anyhow!("LM thread panicked"))??;
        self.lm = Some(lm_out.lm);

        let response_ids = take_response_ids(&lm_out.generated, think_end, eos_id);
        let assistant_text = decode_response(&self.lm_tokenizer, &response_ids, self.eos_id)?;
        let lm_secs = lm_out.lm_secs;
        ensure!(!assistant_text.is_empty(), "Qwen3 returned empty reply");
        self.chat.push(ChatMessage {
            role: "assistant".into(),
            content: assistant_text.clone(),
        });

        let sentences: Vec<String> = if self.first_sentence_tts {
            vec![spoken_text(&assistant_text, true)]
        } else {
            split_sentences(&assistant_text)
        };
        ensure!(!sentences.is_empty(), "no spoken sentences");
        let tts_text = sentences.join(" ");
        log_lm_reply(&assistant_text, &tts_text, lm_secs);

        let mut pcm_all = Vec::new();
        let mut last_stats = None;
        let t_tts = Instant::now();
        for (i, sent) in sentences.iter().enumerate() {
            if sentences.len() > 1 {
                println!("  🔊 sentence {}/{}: {sent:?}", i + 1, sentences.len());
            } else {
                println!("  🔊 synthesizing…");
            }
            let stream_cfg =
                stream_config_for_reply(sent, low_latency_tts, self.turbo_tts, self.warmed);
            let run = run_tts(
                &mut self.tts,
                &self.voice_ref,
                sent,
                stream_cfg,
                self.streaming_tts,
                turn_anchor,
                &first_audio,
                self.aec.as_mut(),
            )?;
            pcm_all.extend(run.pcm);
            last_stats = Some(run.stats);
        }

        let tts_secs = t_tts.elapsed().as_secs_f64();
        let stats = last_stats.context("TTS produced no audio")?;
        println!(
            "  reply → audio ({tts_secs:.2}s, synth-TTFA {:.2}s, RTF {:.2}×, {} samples)",
            stats.time_to_first_audio_secs,
            stats.realtime_factor(),
            pcm_all.len()
        );

        self.finish_turn(
            turn_idx,
            out_dir,
            user_text,
            assistant_text,
            tts_text,
            &pcm_all,
            asr_secs,
            lm_secs,
            tts_secs,
            stats.time_to_first_audio_secs,
            first_audio
                .lock()
                .ok()
                .and_then(|g| *g)
                .unwrap_or(stats.time_to_first_audio_secs),
        )
    }

    fn handle_turn_pipelined(
        &mut self,
        user_pcm: &[f32],
        turn_idx: usize,
        out_dir: &Path,
        low_latency_tts: bool,
        prefetched_asr: Option<String>,
    ) -> Result<TurnReport> {
        ensure!(
            user_pcm.len() >= WHISPER_RATE / 4,
            "utterance too short for ASR ({} samples)",
            user_pcm.len()
        );

        println!("\n── turn {turn_idx} (streaming) ──");
        let (user_text, asr_secs, prefetched) =
            resolve_user_text(&mut self.whisper, user_pcm, prefetched_asr)?;
        if !prefetched {
            println!("  mic → text ({asr_secs:.2}s): {user_text:?}");
        }
        ensure!(!user_text.is_empty(), "Whisper returned empty transcript");

        let turn_anchor = Instant::now();
        let first_audio = Arc::new(Mutex::new(None::<f64>));

        self.chat.push(ChatMessage {
            role: "user".into(),
            content: user_text.clone(),
        });
        let chat_for_lm = truncate_chat_history(&self.chat, 2);
        let prompt = finalize_lm_prompt(
            self.lm_template
                .render(&chat_for_lm, true)
                .context("render chat prompt")?,
            self.skip_thinking,
        );
        let prompt_ids = encode_prompt(&self.lm_tokenizer, &prompt)?;

        let eos_id = self.eos_id;
        let think_end = self.lm_tokenizer.token_to_id("</think>");
        let mut answer_ids: Vec<u32> = Vec::new();
        let mut speakable_at: Option<f64> = None;

        let show_tokens = self.stream_lm_tokens;
        if show_tokens {
            print!("  LM tokens: ");
        }
        let t_lm = Instant::now();
        let generated = self
            .lm
            .as_mut()
            .context("LM in flight on another thread")?
            .generate_stoppable(&prompt_ids, self.max_tokens, |tok| {
                if eos_id == Some(tok) {
                    return false;
                }
                answer_ids.push(tok);
                if show_tokens {
                    if let Ok(piece) = self.lm_tokenizer.decode(&[tok], false) {
                        print!("{piece}");
                        let _ = std::io::stdout().flush();
                    }
                }
                if speakable_at.is_none()
                    && maybe_early_speakable(&self.lm_tokenizer, &answer_ids, eos_id, tok, 7)
                        .is_some()
                {
                    speakable_at = Some(turn_anchor.elapsed().as_secs_f64());
                    println!(
                        "\n  ⚡ speakable text ready ({:.2}s from turn start)",
                        speakable_at.unwrap()
                    );
                }
                true
            })?;
        if show_tokens {
            println!();
        }

        let response_ids = take_response_ids(&generated, think_end, eos_id);
        let assistant_text = decode_response(&self.lm_tokenizer, &response_ids, self.eos_id)?;
        let lm_secs = t_lm.elapsed().as_secs_f64();
        ensure!(!assistant_text.is_empty(), "Qwen3 returned empty reply");
        self.chat.push(ChatMessage {
            role: "assistant".into(),
            content: assistant_text.clone(),
        });

        let sentences: Vec<String> = if self.first_sentence_tts {
            vec![spoken_text(&assistant_text, true)]
        } else {
            split_sentences(&assistant_text)
        };
        ensure!(!sentences.is_empty(), "no spoken sentences");
        let tts_text = sentences.join(" ");
        log_lm_reply(&assistant_text, &tts_text, lm_secs);

        let mut pcm_all = Vec::new();
        let mut last_stats = None;
        let t_tts = Instant::now();
        for (i, sent) in sentences.iter().enumerate() {
            let stream_cfg =
                stream_config_for_reply(sent, low_latency_tts, self.turbo_tts, self.warmed);
            if sentences.len() > 1 {
                println!("  🔊 sentence {}/{}: {sent:?}", i + 1, sentences.len());
            } else {
                println!("  🔊 synthesizing…");
            }
            let run = run_tts(
                &mut self.tts,
                &self.voice_ref,
                sent,
                stream_cfg,
                self.streaming_tts,
                turn_anchor,
                &first_audio,
                self.aec.as_mut(),
            )?;
            pcm_all.extend(run.pcm);
            last_stats = Some(run.stats);
        }
        let tts_secs = t_tts.elapsed().as_secs_f64();
        let stats = last_stats.context("TTS produced no audio")?;
        println!(
            "  reply → audio ({tts_secs:.2}s, synth-TTFA {:.2}s, RTF {:.2}×, {} samples)",
            stats.time_to_first_audio_secs,
            stats.realtime_factor(),
            pcm_all.len()
        );

        self.finish_turn(
            turn_idx,
            out_dir,
            user_text,
            assistant_text,
            tts_text,
            &pcm_all,
            asr_secs,
            lm_secs,
            tts_secs,
            stats.time_to_first_audio_secs,
            first_audio
                .lock()
                .ok()
                .and_then(|g| *g)
                .unwrap_or(stats.time_to_first_audio_secs),
        )
    }
}

const WHISPER_MIN_RECALL: f32 = 0.5;
const WHISPER_TARGET_PEAK: f32 = 0.95;

fn log_lm_reply(full: &str, spoken: &str, lm_secs: f64) {
    println!("  LM reply ({lm_secs:.2}s, full): {full:?}");
    if spoken == full {
        println!("  TTS speaks: (same as LM reply)");
    } else {
        println!("  TTS speaks: {spoken:?}");
    }
}

fn pcm_peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn scale_pcm_to_peak(pcm: &[f32], target: f32) -> Vec<f32> {
    let peak = pcm_peak(pcm);
    if peak < 1e-4 {
        return pcm.to_vec();
    }
    let gain = target / peak;
    pcm.iter().map(|v| v * gain).collect()
}

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn transcript_covers_reference(reference: &str, transcript: &str, min_ratio: f32) -> bool {
    let reference_words: Vec<_> = normalize_words(reference)
        .into_iter()
        .filter(|w| w.len() >= 3)
        .collect();
    if reference_words.is_empty() {
        let lower = transcript.to_lowercase();
        return reference
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .all(|w| lower.contains(w));
    }
    let heard = normalize_words(transcript);
    let hits = reference_words
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / reference_words.len() as f32 >= min_ratio
}

fn transcribe_reply_pcm(whisper: &mut WhisperRunner, pcm_24k: &[f32]) -> Result<String> {
    let scaled = scale_pcm_to_peak(pcm_24k, WHISPER_TARGET_PEAK);
    let pcm_16k = resample_linear(&scaled, TTS_RATE, WHISPER_RATE as u32);
    ensure!(
        pcm_16k.len() >= WHISPER_RATE / 2,
        "reply audio too short for Whisper round-trip ({} samples @ 16 kHz)",
        pcm_16k.len()
    );
    Ok(trim_transcript(&whisper.transcribe_greedy(&pcm_16k)?))
}

fn validate_reply_with_whisper(
    whisper: &mut WhisperRunner,
    reference: &str,
    pcm_24k: &[f32],
) -> Result<(String, bool, f64)> {
    let t = Instant::now();
    let heard = transcribe_reply_pcm(whisper, pcm_24k)?;
    let ok = transcript_covers_reference(reference, &heard, WHISPER_MIN_RECALL);
    Ok((heard, ok, t.elapsed().as_secs_f64()))
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

fn load_input_pcm(path: &Path) -> Result<Vec<f32>> {
    if let Ok(pcm) = load_wav_mono_f32(path) {
        return Ok(pcm);
    }
    let (pcm, rate) = read_wav_f32_with_rate(path)?;
    Ok(resample_linear(&pcm, rate, WHISPER_RATE as u32))
}

fn utterances_from_batch_vad(pcm: &[f32]) -> Vec<Vec<f32>> {
    let cfg = VadConfig::default();
    pcm_segments_by_vad_config(&cfg, pcm)
        .into_iter()
        .map(|seg| pcm[seg.start..seg.end].to_vec())
        .filter(|u| u.len() >= WHISPER_RATE / 4)
        .collect()
}

fn run_streaming_mic_turns(
    session: &mut Session,
    input_pcm: &[f32],
    chunk_samples: usize,
    silence_ms: u32,
    out_dir: &Path,
    low_latency_tts: bool,
) -> Result<Vec<TurnReport>> {
    let mut gate = StreamUtteranceGate::new(silence_ms);
    let mut partial = PartialAsrTracker::new(350);
    let mut reports = Vec::new();
    let mut turn_idx = 0usize;
    let mut pos = 0usize;
    let mut chunk_idx = 0usize;
    for chunk in input_pcm.chunks(chunk_samples.max(1)) {
        let mic = process_mic_aec(session.aec.as_mut(), chunk);
        chunk_idx += 1;
        let rms = chunk_rms(&mic);
        let speech = rms > 0.008;
        println!(
            "  mic chunk {chunk_idx:>3}: {pos:>6}–{:<6} samples  rms={rms:.4}  {}",
            pos + mic.len(),
            if speech { "speech" } else { "quiet" }
        );
        pos += mic.len();

        if let Some(utterance) = gate.push(&mic) {
            turn_idx += 1;
            println!(
                "  └─ utterance flushed: {:.2}s (end-of-turn)",
                utterance.len() as f64 / WHISPER_RATE as f64
            );
            let prefetch = gate.take_prefetch().or_else(|| partial.latest());
            reports.push(session.handle_turn(
                &utterance,
                turn_idx,
                out_dir,
                low_latency_tts,
                prefetch,
            )?);
        } else if session.streaming_asr {
            if let Some(ws) = session.whisper_stream.as_mut() {
                if gate.speech_active() {
                    partial.maybe_update(ws, gate.active_buffer());
                }
                if gate.should_prefetch_asr() {
                    let t0 = Instant::now();
                    let buf = gate.active_buffer().to_vec();
                    if let Ok(text) = transcribe_user_audio(ws, &buf) {
                        println!(
                            "  ASR prefetch ({:.2}s, {:.2}s audio): {text:?}",
                            t0.elapsed().as_secs_f64(),
                            buf.len() as f64 / WHISPER_RATE as f64
                        );
                        gate.set_prefetch(text);
                    }
                }
            }
        }
    }
    if let Some(utterance) = gate.flush() {
        turn_idx += 1;
        println!(
            "  └─ utterance flushed: {:.2}s (end of input stream)",
            utterance.len() as f64 / WHISPER_RATE as f64
        );
        let prefetch = gate.take_prefetch().or_else(|| partial.latest());
        reports.push(session.handle_turn(
            &utterance,
            turn_idx,
            out_dir,
            low_latency_tts,
            prefetch,
        )?);
    }
    Ok(reports)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    apply_gpu_perf_env(&args);
    warn_if_metal_lm(args.qwen3_device);
    std::fs::create_dir_all(&args.out_dir)?;

    let qwen3_resolved =
        resolve_qwen3_weights_path(&args.qwen3_weights, args.qwen3_device, args.gpu_perf);
    println!("┌─ Bidirectional voice chat ────────────────────────────────────");
    println!("│ Qwen3:    {}", qwen3_resolved.display());
    if qwen3_resolved != args.qwen3_weights {
        println!("│           (from {})", args.qwen3_weights.display());
    }
    println!("│ TTS:      {}", args.tts_model_dir.display());
    println!("│ Whisper:  {}", args.whisper_dir.display());
    println!("│ ref WAV:  {}", args.ref_wav.display());
    println!("│ input:    {}", args.input_wav.display());
    println!("│ out dir:  {}", args.out_dir.display());
    println!(
        "│ mode:     {}",
        if args.batch {
            "batch (VAD segments)"
        } else {
            "streamed mic chunks"
        }
    );
    println!(
        "│ backends: whisper={:?}  qwen3={:?}  tts={:?}",
        args.device, args.qwen3_device, args.device
    );
    println!("│ qwen3:    skip_thinking={}", args.skip_thinking);
    println!(
        "│ tts:      max_frames={} batched={} streaming_tts={} turbo={}",
        args.tts_max_frames, !args.streaming_tts, args.streaming_tts, args.turbo_tts
    );
    println!(
        "│ latency:  preload={} gpu_perf={} overlap={} early_tts={}w validate={}",
        args.preload,
        args.gpu_perf,
        args.overlap_lm_tts,
        if args.aggressive_early_tts {
            args.early_tts_words
        } else {
            0
        },
        args.whisper_validate
    );
    println!("│ vad gate: {:?}", args.vad_gate);
    println!(
        "│ stream:   asr_prefetch={} metal_tts_native={}",
        args.streaming_asr, args.metal_tts_native
    );
    println!("└───────────────────────────────────────────────────────────────");

    let mut session = Session::open(&args)?;

    let input_pcm = load_input_pcm(&args.input_wav)?;
    println!(
        "\ninput audio: {:.2}s @ {} Hz",
        input_pcm.len() as f64 / WHISPER_RATE as f64,
        WHISPER_RATE
    );

    let reports = if args.batch {
        println!("\n── batch VAD segmentation ──");
        let segs = utterances_from_batch_vad(&input_pcm);
        for (i, u) in segs.iter().enumerate() {
            println!(
                "  segment {}: {:.2}s ({} samples)",
                i + 1,
                u.len() as f64 / WHISPER_RATE as f64,
                u.len()
            );
        }
        ensure!(
            !segs.is_empty(),
            "no speech utterances detected — try --batch or a longer --input-wav"
        );
        println!("detected {} utterance(s)", segs.len());
        let mut out = Vec::new();
        for (idx, utterance) in segs.into_iter().enumerate() {
            out.push(session.handle_turn(
                &utterance,
                idx + 1,
                &args.out_dir,
                args.low_latency_tts,
                None,
            )?);
        }
        out
    } else {
        let chunk = (WHISPER_RATE as u32 * args.mic_chunk_ms / 1000).max(160) as usize;
        println!(
            "\n── streamed mic ingress ({} ms chunks, {} ms silence gate) ──",
            args.mic_chunk_ms, args.silence_ms
        );
        let reps = run_streaming_mic_turns(
            &mut session,
            &input_pcm,
            chunk,
            args.silence_ms,
            &args.out_dir,
            args.low_latency_tts,
        )?;
        ensure!(
            !reps.is_empty(),
            "no speech utterances detected — try --batch or a longer --input-wav"
        );
        println!("completed {} turn(s)", reps.len());
        reps
    };

    let mut full_reply_pcm = Vec::new();
    for report in &reports {
        full_reply_pcm.extend_from_slice(&report.reply_pcm);
    }
    if !full_reply_pcm.is_empty() {
        let combined = args.out_dir.join("session_reply.wav");
        rlx_qwen3_tts::runner::write_wav_mono(&combined, &full_reply_pcm, TTS_RATE)?;
        println!("\n✓ combined speaker output → {}", combined.display());
    }

    println!("\n── session summary ──");
    println!(
        "{:<6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "turn", "ASR", "LM", "TTS", "audio", "samples"
    );
    for (i, r) in reports.iter().enumerate() {
        println!("  {}  user={:?}", i + 1, r.user_text);
        println!("       LM full={:?}", r.assistant_text);
        println!("       TTS spoke={:?}", r.tts_text);
        if let Some(heard) = &r.whisper_heard {
            println!("       Whisper heard={:?}  match={}", heard, r.whisper_ok);
        }
        println!(
            "       ASR {:>5.2}s  LM {:>5.2}s  TTS {:>5.2}s  reply-audio {:>5.2}s  e2e {:>5.2}s  {} samples",
            r.asr_secs,
            r.lm_secs,
            r.tts_secs,
            r.turn_first_audio_secs,
            r.e2e_first_audio_secs,
            r.out_pcm_samples
        );
    }

    if args.whisper_validate {
        let all_ok = reports.iter().all(|r| r.whisper_ok);
        ensure!(
            all_ok,
            "one or more turns failed Whisper round-trip validation"
        );
        println!("\n✓ all turns passed Whisper round-trip validation");
    }

    Ok(())
}

/// Minimal mono PCM WAV reader (any sample rate).
fn read_wav_f32_with_rate(path: &Path) -> Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 44 {
        bail!("wav too small");
    }
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

fn read_wav_f32_any_rate(path: &Path) -> Result<Vec<f32>> {
    Ok(read_wav_f32_with_rate(path)?.0)
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
