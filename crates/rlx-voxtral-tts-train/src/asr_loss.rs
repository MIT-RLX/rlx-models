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

//! ASR auxiliary loss — mel proxy + optional Whisper CER.

use rlx_whisper::WhisperRunner;
use rlx_whisper::audio::SAMPLE_RATE;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct AsrLoss {
    whisper: Mutex<Option<WhisperRunner>>,
}

impl AsrLoss {
    pub fn from_env() -> Self {
        Self {
            whisper: Mutex::new(open_whisper_from_env()),
        }
    }

    pub fn loss(&self, recon: &[f32], target: &[f32], transcript: Option<&str>) -> f32 {
        let mel = whisper_mel_mse(recon, target);
        let Some(text) = transcript.filter(|t| !t.trim().is_empty()) else {
            return mel;
        };
        let mut guard = self.whisper.lock().expect("asr whisper lock");
        let Some(runner) = guard.as_mut() else {
            return mel;
        };
        let pcm16 = resample16k(recon);
        let cer = match runner.transcribe_greedy(&pcm16) {
            Ok(hyp) => normalized_cer(text, &hyp),
            Err(_) => 1.0,
        };
        0.5 * mel + 0.5 * cer
    }
}

fn open_whisper_from_env() -> Option<WhisperRunner> {
    if std::env::var("USE_WHISPER_ASR")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        // fall through
    } else {
        return None;
    }
    let path = std::env::var("WHISPER_MODEL_DIR")
        .or_else(|_| std::env::var("WHISPER_WEIGHTS"))
        .ok()
        .map(PathBuf::from)?;
    WhisperRunner::builder()
        .weights(path.join("model.safetensors"))
        .config_path(path.join("config.json"))
        .device(rlx_runtime::Device::Cpu)
        .build()
        .ok()
}

/// Mean squared error between coarse log-energy mel bands of recon vs target PCM.
pub fn whisper_mel_mse(recon: &[f32], target: &[f32]) -> f32 {
    let recon_b = mel_bands(&resample16k(recon));
    let target_b = mel_bands(&resample16k(target));
    let n = recon_b.len().min(target_b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0f32;
    for i in 0..n {
        let d = recon_b[i] - target_b[i];
        sum += d * d;
    }
    sum / n as f32
}

pub fn normalized_cer(reference: &str, hypothesis: &str) -> f32 {
    let ref_t = normalize_words(reference);
    let hyp_t = normalize_words(hypothesis);
    if ref_t.is_empty() {
        return 0.0;
    }
    let dist = levenshtein(&ref_t, &hyp_t);
    (dist as f32 / ref_t.len() as f32).clamp(0.0, 1.0)
}

fn normalize_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, wa) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, wb) in b.iter().enumerate() {
            let cost = if wa == wb { 0 } else { 1 };
            cur[j + 1] = (cur[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn resample16k(pcm: &[f32]) -> Vec<f32> {
    if pcm.is_empty() {
        return vec![0.0; SAMPLE_RATE / 10];
    }
    let src_rate = 24_000usize;
    let out_len = pcm.len() * SAMPLE_RATE / src_rate.max(1);
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len.max(1) {
        let src = i * src_rate / SAMPLE_RATE.max(1);
        out.push(pcm[src.min(pcm.len() - 1)]);
    }
    out
}

fn mel_bands(pcm: &[f32]) -> Vec<f32> {
    const BANDS: usize = 80;
    let n = pcm.len().max(1);
    let band = (n / BANDS.max(1)).max(1);
    let mut out = Vec::with_capacity(BANDS);
    for b in 0..BANDS {
        let start = b * band;
        let end = (start + band).min(n);
        if start >= end {
            out.push(0.0);
            continue;
        }
        let mut e = 0f32;
        for i in start..end {
            e += pcm[i] * pcm[i];
        }
        e /= (end - start) as f32;
        out.push((e + 1e-8).ln());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cer_zero_on_exact_match() {
        assert!(normalized_cer("hello world", "Hello world.") < 1e-6);
    }

    #[test]
    fn cer_positive_on_mismatch() {
        assert!(normalized_cer("hello world", "goodbye") > 0.2);
    }

    #[test]
    fn mel_mse_zero_for_identical_pcm() {
        let pcm: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.001).sin()).collect();
        assert!(whisper_mel_mse(&pcm, &pcm).abs() < 1e-6);
    }
}
