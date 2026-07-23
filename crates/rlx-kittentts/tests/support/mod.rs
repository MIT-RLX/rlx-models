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

//! Shared helpers for integration tests.

#![allow(dead_code)]

use rlx_kittentts::{
    MIN_AUDIBLE_PEAK, SAMPLE_RATE, assets, ipa_style_index, peak_amplitude,
};

// Aliased (not `pub use`, which test binaries strip as unused) so integration
// tests can reference `support::LONG_IPA`.
pub const LONG_IPA: &str = rlx_kittentts::phrase_fixtures::LONG_IPA;

pub fn model_dir() -> Option<std::path::PathBuf> {
    assets::default_model_dir().ok()
}

pub fn assert_audible(audio: &[f32], min_samples: usize) {
    assert!(
        audio.len() >= min_samples,
        "expected at least {min_samples} samples, got {}",
        audio.len()
    );
    let peak = peak_amplitude(audio);
    assert!(
        peak >= MIN_AUDIBLE_PEAK,
        "expected audible waveform (peak >= {MIN_AUDIBLE_PEAK}), got peak={peak:.2e}"
    );
}

pub fn style_for(ipa: &str) -> usize {
    ipa_style_index(ipa)
}

/// Native smokes: production infer + compile headroom (wide-seq single pass).
#[cfg(feature = "native")]
pub fn setup_native_smoke_env() {
    unsafe {
        std::env::set_var("KITTEN_RLX_INFER", "production");
        std::env::remove_var("KITTEN_RLX_FULL_GRAPH");
        std::env::remove_var("KITTEN_RLX_COMPILE_EXACT");
        std::env::remove_var("KITTEN_RLX_PARITY");
        std::env::set_var("KITTEN_RLX_PREFER_METAL", "0");
        std::env::set_var("KITTENTTS_TAIL_TRIM", "0");
        std::env::set_var("KITTEN_RLX_SKIP_PREWARM", "1");
        let aot = std::env::temp_dir().join(format!("kitten_smoke_aot_{}", std::process::id()));
        std::env::set_var("KITTEN_RLX_AOT_CACHE", &aot);
    }
}

/// Slice `candidate` at the lag that best aligns with `reference` (for ASR after parity metrics).
pub fn align_candidate_for_asr(reference: &[f32], candidate: &[f32], max_lag: usize) -> Vec<f32> {
    let (lag, _) = max_abs_best_lag(reference, candidate, max_lag);
    let n = reference.len().min(candidate.len().saturating_sub(lag));
    if n == 0 {
        return candidate.to_vec();
    }
    candidate[lag..lag + n].to_vec()
}

/// Min peak error over sample lag (handles small vocoder phase offsets).
pub fn max_abs_best_lag(reference: &[f32], candidate: &[f32], max_lag: usize) -> (usize, f32) {
    let n = reference.len().min(candidate.len());
    if n == 0 {
        return (0, 0.0);
    }
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut best_lag = 0usize;
    let mut best = f32::MAX;
    for lag in 0..=max_lag {
        let m = n - lag;
        let mut peak = 0.0f32;
        for i in 0..m {
            peak = peak.max((reference[i] - candidate[i + lag]).abs());
        }
        if peak < best {
            best = peak;
            best_lag = lag;
        }
    }
    (best_lag, best)
}

pub fn duration_secs(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE as f32
}

/// Linear resample (adequate for Whisper round-trip checks).
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = samples.len() as u64 * to_hz as u64 / from_hz as u64;
    let out_len = out_len.max(1) as usize;
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
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

/// True when enough reference words appear in the Whisper transcript.
pub fn transcript_covers_reference(reference: &str, transcript: &str, min_ratio: f32) -> bool {
    let reference_words = normalize_words(reference);
    if reference_words.is_empty() {
        return false;
    }
    let heard = normalize_words(transcript);
    let hits = reference_words
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / reference_words.len() as f32 >= min_ratio
}

pub fn normalize_words_for_test(text: &str) -> Vec<String> {
    normalize_words(text)
}

pub fn whisper_asr_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("RLX_WHISPER_DIR") {
        return whisper_dir_if_ready(std::path::PathBuf::from(dir));
    }
    let cache = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        if let Some(dir) = whisper_dir_if_ready(cache.join(name)) {
            return Some(dir);
        }
    }
    None
}

fn whisper_dir_if_ready(dir: std::path::PathBuf) -> Option<std::path::PathBuf> {
    if dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file() {
        Some(dir)
    } else {
        None
    }
}

/// Legacy name used by early round-trip tests.
pub fn whisper_tiny_dir() -> Option<std::path::PathBuf> {
    whisper_asr_dir()
}
