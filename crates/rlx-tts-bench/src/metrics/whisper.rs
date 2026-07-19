//! Whisper greedy ASR + word coverage.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rlx_runtime::Device;
use rlx_whisper::WhisperRunner;
use serde::{Deserialize, Serialize};

use crate::phrases::{FOX_WORDS, content_words};
use crate::wav::{peak_normalize, resample_linear};

pub struct WhisperState {
    pub runner: WhisperRunner,
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperMetrics {
    pub transcript: String,
    pub fox_hits: usize,
    pub fox_total: usize,
    pub content_hits: usize,
    pub content_total: usize,
    pub coverage: f64,
}

pub fn try_load_whisper() -> Option<WhisperState> {
    let dir = whisper_dir()?;
    let runner = WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()?;
    Some(WhisperState { runner, dir })
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    let cache = PathBuf::from(".cache");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = cache.join(name);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    None
}

fn whisper_ready(dir: &Path) -> bool {
    dir.join("model.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
        && dir.join("config.json").is_file()
}

pub fn whisper_coverage(
    state: &mut WhisperState,
    pcm: &[f32],
    sample_rate: u32,
    expected_text: &str,
) -> Result<WhisperMetrics> {
    let norm = peak_normalize(pcm, 0.95);
    let pcm16 = resample_linear(&norm, sample_rate, 16_000);
    let transcript = state.runner.transcribe_greedy(&pcm16)?;
    let lower = transcript.to_lowercase();
    let fox_hits = FOX_WORDS.iter().filter(|w| lower.contains(*w)).count();
    let words = content_words(expected_text);
    let content_total = words.len();
    let content_hits = words.iter().filter(|w| lower.contains(w.as_str())).count();
    let coverage = if content_total == 0 {
        0.0
    } else {
        content_hits as f64 / content_total as f64
    };
    Ok(WhisperMetrics {
        transcript,
        fox_hits,
        fox_total: FOX_WORDS.len(),
        content_hits,
        content_total,
        coverage,
    })
}
