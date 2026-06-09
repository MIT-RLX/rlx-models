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

//! Qwen3-TTS → Whisper-base.en ASR round-trip (intelligibility check).
//!
//! Two layers:
//! 1. **Golden codec → speech decode → Whisper** — isolates speech tokenizer / GPU conv quality
//!    using committed HF greedy frames for `"Hi."` (no talker/CP drift).
//! 2. **Full CustomVoice e2e → Whisper** — end-to-end synthesis via the parity-validated path
//!    (`synthesize_custom_voice_greedy`, CPU pipeline on Metal) then ASR.
//!
//! Whisper-base.en is the reference ASR model (`.cache/whisper-base.en` or `RLX_WHISPER_DIR`).
//! Greedy decode, 16 kHz after linear resample from 24 kHz TTS output.
//!
//! ```sh
//! export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
//! just fetch-whisper-base   # if needed
//! cargo test -p rlx-models --test qwen3_tts_whisper_roundtrip --features metal --release -- --nocapture
//! ```

use anyhow::Result;
use rlx_models::qwen3_tts::Qwen3TtsSession;
use rlx_models::qwen3_tts::{GenerationConfig, write_wav_mono};
use rlx_models::whisper::WhisperRunner;
use rlx_qwen3_tts::speech_tokenizer::decode_codec_frames;
use rlx_qwen3_tts::tokens::SAMPLE_RATE_HZ;
use rlx_runtime::{Device, is_available};
use rlx_whisper::SAMPLE_RATE as WHISPER_RATE;
use std::path::{Path, PathBuf};

const MIN_PEAK: f32 = 1e-4;
const TARGET_PEAK: f32 = 0.95;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn repo_cache() -> PathBuf {
    repo_root().join(".cache")
}

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var("RLX_QWEN3_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            let d = repo_cache().join("qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice");
            d.join("model.safetensors").is_file().then_some(d)
        })
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    for name in ["whisper-base.en", "whisper-small.en", "whisper-tiny.en"] {
        let p = repo_cache().join(name);
        if p.join("model.safetensors").is_file() && p.join("tokenizer.json").is_file() {
            return Some(p);
        }
    }
    None
}

fn synth_device() -> Device {
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) {
        Device::Mlx
    } else if is_available(Device::Cuda) {
        Device::Cuda
    } else if is_available(Device::Rocm) {
        Device::Rocm
    } else {
        Device::Cpu
    }
}

fn load_golden_hi_codec() -> Vec<Vec<u32>> {
    let path = repo_root().join("crates/rlx-models/tests/fixtures/qwen3_tts_hi_greedy_codec.txt");
    let text = std::fs::read_to_string(&path).expect("golden fixture");
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let n: usize = lines.next().expect("count").trim().parse().expect("n");
    let mut frames = Vec::with_capacity(n);
    for line in lines.take(n) {
        let vals: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.parse().expect("token"))
            .collect();
        assert_eq!(vals.len(), 16);
        frames.push(vals);
    }
    assert_eq!(frames.len(), 22);
    frames
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

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Enough reference words (len ≥ 3) appear in the Whisper transcript.
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

fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn scale_to_peak(pcm: &[f32], target: f32) -> Vec<f32> {
    let p = peak(pcm);
    if p < MIN_PEAK {
        return pcm.to_vec();
    }
    let g = target / p;
    pcm.iter().map(|v| v * g).collect()
}

fn whisper_runner(dir: &Path) -> Result<WhisperRunner> {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
}

fn transcribe_tts_pcm(pcm_24k: &[f32], whisper_dir: &Path) -> Result<String> {
    let scaled = scale_to_peak(pcm_24k, TARGET_PEAK);
    let pcm_16k = resample_linear(&scaled, SAMPLE_RATE_HZ, WHISPER_RATE as u32);
    anyhow::ensure!(
        pcm_16k.len() >= WHISPER_RATE / 2,
        "resampled audio too short for Whisper"
    );
    let mut whisper = whisper_runner(whisper_dir)?;
    whisper.transcribe_greedy(&pcm_16k)
}

/// Golden HF codec frames → native speech decode → Whisper-base.en.
#[test]
fn golden_hi_speech_decode_whisper_base() -> Result<()> {
    let Some(model_dir) = qwen3_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR or .cache/qwen3-tts weights");
        return Ok(());
    };
    if !model_dir
        .join("speech_tokenizer/model.safetensors")
        .is_file()
    {
        eprintln!("skip: incomplete Qwen3-TTS bundle (speech_tokenizer)");
        return Ok(());
    }
    let Some(whisper_dir) = whisper_dir() else {
        eprintln!("skip: run `just fetch-whisper-base`");
        return Ok(());
    };

    let device = synth_device();
    // GPU conv matmul path still diverges on full decode (see golden_codec_decode_cpu_matches_metal).
    // SAFETY: test-only — force CPU conv so golden codec → PCM matches HF before Whisper ASR.
    unsafe {
        std::env::set_var("RLX_QWEN3_TTS_SPEECH_CONV_CPU", "1");
    }
    let frames = load_golden_hi_codec();
    eprintln!(
        "[golden-decode] device={device:?} speech_conv={}",
        rlx_qwen3_tts::speech_tokenizer::speech_conv_backend_label(device)
    );

    let pcm = decode_codec_frames(&model_dir, &frames, device)?;
    assert_eq!(pcm.len(), 22 * 1920, "22 frames × 1920 samples");
    let p = peak(&pcm);
    assert!(p > 0.3, "golden decode peak too small: {p}");

    let out_wav = std::env::temp_dir().join("qwen3-whisper-golden-hi.wav");
    write_wav_mono(&out_wav, &pcm, SAMPLE_RATE_HZ)?;
    eprintln!("[golden-decode] wav: {}", out_wav.display());

    let transcript = transcribe_tts_pcm(&pcm, &whisper_dir)?;
    eprintln!("[golden-decode] reference: Hi.");
    eprintln!("[golden-decode] whisper:   {transcript}");

    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("hi") || lower.contains("high") || lower.contains("hey"),
        "Whisper did not recognize golden Hi decode (got: {transcript})"
    );
    Ok(())
}

struct E2eCase {
    text: &'static str,
    max_frames: usize,
    min_recall: f32,
    min_samples: usize,
    must_contain: &'static [&'static str],
}

/// Full CustomVoice greedy synthesis (parity path) → Whisper-base.en.
#[test]
fn qwen3_tts_whisper_base_roundtrip() -> Result<()> {
    let Some(model_dir) = qwen3_dir() else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR or .cache/qwen3-tts weights");
        return Ok(());
    };
    if !model_dir
        .join("speech_tokenizer/model.safetensors")
        .is_file()
    {
        eprintln!("skip: incomplete Qwen3-TTS bundle (speech_tokenizer)");
        return Ok(());
    }
    let Some(whisper_dir) = whisper_dir() else {
        eprintln!("skip: run `just fetch-whisper-base`");
        return Ok(());
    };

    let device = synth_device();
    let mut session = Qwen3TtsSession::open(&model_dir, device)?;
    let mut gen_cfg = GenerationConfig::greedy_for_model_dir(&model_dir)?;

    eprintln!(
        "[whisper-roundtrip] qwen3={} whisper={} device={device:?} gpu_pipeline={}",
        model_dir.display(),
        whisper_dir.display(),
        rlx_qwen3_tts::gpu_pipeline::gpu_session_enabled(device),
    );

    let cases = [
        E2eCase {
            text: "Hi.",
            max_frames: 32,
            min_recall: 0.0,
            min_samples: 8_000,
            must_contain: &["hi"],
        },
        E2eCase {
            text: "Hello world.",
            max_frames: 32,
            min_recall: 0.5,
            min_samples: 8_000,
            must_contain: &["hello", "world"],
        },
        E2eCase {
            text: "The quick brown fox jumps over the lazy dog.",
            max_frames: 96,
            min_recall: 0.4,
            min_samples: 20_000,
            must_contain: &["quick", "brown", "fox"],
        },
    ];

    for (i, case) in cases.iter().enumerate() {
        gen_cfg.max_new_tokens = case.max_frames;
        let result =
            session.synthesize_custom_voice(case.text, "vivian", "english", &gen_cfg, false)?;
        let pcm = &result.pcm;
        let p = peak(pcm);
        eprintln!(
            "[case {i}] prompt={:?} frames={} pcm={} peak={p:.4} duration={:.2}s",
            case.text,
            result.codec_frames.len(),
            pcm.len(),
            pcm.len() as f64 / SAMPLE_RATE_HZ as f64,
        );
        assert!(
            pcm.len() >= case.min_samples,
            "case {i}: audio too short ({} < {})",
            pcm.len(),
            case.min_samples
        );
        assert!(p >= MIN_PEAK, "case {i}: inaudible peak={p}");

        let out_wav = std::env::temp_dir().join(format!("qwen3-whisper-val-{i}.wav"));
        write_wav_mono(&out_wav, pcm, result.sample_rate)?;
        eprintln!("[case {i}] wav: {}", out_wav.display());

        let transcript = transcribe_tts_pcm(pcm, &whisper_dir)?;
        let recall = if case.min_recall > 0.0 {
            let reference_words: Vec<_> = normalize_words(case.text)
                .into_iter()
                .filter(|w| w.len() >= 3)
                .collect();
            let heard = normalize_words(&transcript);
            let hits = reference_words
                .iter()
                .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
                .count();
            if reference_words.is_empty() {
                0.0
            } else {
                hits as f32 / reference_words.len() as f32
            }
        } else {
            1.0
        };
        eprintln!("[case {i}] reference: {}", case.text);
        eprintln!("[case {i}] whisper:   {transcript}");
        eprintln!("[case {i}] word recall: {:.0}%", recall * 100.0);

        if case.min_recall > 0.0 {
            assert!(
                transcript_covers_reference(case.text, &transcript, case.min_recall),
                "case {i}: Whisper missed too much of the prompt (recall {:.0}% < {:.0}%)\nref: {}\ngot: {}",
                recall * 100.0,
                case.min_recall * 100.0,
                case.text,
                transcript,
            );
        }
        let lower = transcript.to_lowercase();
        for needle in case.must_contain {
            assert!(
                lower.contains(needle),
                "case {i}: expected '{needle}' in transcript, got: {transcript}"
            );
        }
    }

    Ok(())
}
