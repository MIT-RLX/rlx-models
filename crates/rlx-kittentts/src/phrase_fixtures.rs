// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! IPA phrase fixtures for native production / Whisper round-trip tests and audio export.

/// Long IPA sentence (~5 s ONNX).
pub const LONG_IPA: &str =
    "ðɪs ɪz ə lɔŋɡɚ sɛntəns fɔɹ tɛstɪŋ ðə kɪtən tɛkst tə spitʃ sɪstəm ɪn ɹʌst";

/// Shared phrase fixture for native production Whisper checks.
#[derive(Debug, Clone, Copy)]
pub struct PhraseCase {
    pub label: &'static str,
    pub ipa: &'static str,
    pub voice: Option<&'static str>,
    pub asr_reference: &'static str,
    pub min_ratio: f32,
    pub max_peak: f32,
    pub max_lag: usize,
    pub ort_candidates: usize,
    pub min_samples: usize,
    pub strict_asr: bool,
    pub strict_waveform: bool,
    pub asr_retries: usize,
}

/// Short IPA phrases (CI strict).
pub const SHORT_PHRASES: &[PhraseCase] = &[
    PhraseCase {
        label: "hello",
        ipa: "həˈloʊ",
        voice: None,
        asr_reference: "hello",
        min_ratio: 0.5,
        max_peak: 0.15,
        max_lag: 512,
        ort_candidates: 64,
        min_samples: 4_000,
        strict_asr: true,
        strict_waveform: true,
        asr_retries: 2,
    },
    PhraseCase {
        label: "good morning",
        ipa: "ɡʊd mɔɹnɪŋ",
        voice: None,
        asr_reference: "good morning",
        min_ratio: 0.5,
        max_peak: 0.15,
        max_lag: 512,
        ort_candidates: 64,
        min_samples: 6_000,
        strict_asr: true,
        strict_waveform: true,
        asr_retries: 1,
    },
];

/// Long IPA phrases (CI strict).
pub const LONG_PHRASES: &[PhraseCase] = &[
    PhraseCase {
        label: "kitten rust",
        ipa: LONG_IPA,
        voice: Some("Jasper"),
        asr_reference: "kitten text to speech system rust",
        min_ratio: 0.45,
        max_peak: 0.15,
        max_lag: 4096,
        ort_candidates: 8,
        min_samples: 20_000,
        strict_asr: true,
        strict_waveform: true,
        asr_retries: 0,
    },
    PhraseCase {
        label: "quick fox",
        ipa: "ðə kwɪk braʊn fɑks dʒʌmps oʊvɚ ðə leɪzi dɔɡ",
        voice: None,
        asr_reference: "quick brown fox jumps lazy dog",
        min_ratio: 0.4,
        max_peak: 0.15,
        max_lag: 4096,
        ort_candidates: 16,
        min_samples: 15_000,
        strict_asr: true,
        strict_waveform: true,
        asr_retries: 0,
    },
];

/// Borderline phrases (extended matrix; log-only gates).
pub const SHORT_PHRASES_EXTENDED: &[PhraseCase] = &[
    PhraseCase {
        label: "hi",
        ipa: "haɪ",
        voice: None,
        asr_reference: "hi",
        min_ratio: 0.5,
        max_peak: 0.15,
        max_lag: 256,
        ort_candidates: 64,
        min_samples: 4_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 2,
    },
    PhraseCase {
        label: "thanks",
        ipa: "θæŋks",
        voice: None,
        asr_reference: "thanks",
        min_ratio: 0.5,
        max_peak: 0.15,
        max_lag: 512,
        ort_candidates: 64,
        min_samples: 4_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 2,
    },
    PhraseCase {
        label: "thank you",
        ipa: "θæŋk ju",
        voice: None,
        asr_reference: "thank",
        min_ratio: 0.5,
        max_peak: 0.15,
        max_lag: 512,
        ort_candidates: 64,
        min_samples: 4_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 2,
    },
];

pub const LONG_PHRASES_EXTENDED: &[PhraseCase] = &[
    PhraseCase {
        label: "weather",
        ipa: "ðə wɛðɚ tədeɪ ɪz sʌni ænd wɔɹm",
        voice: None,
        asr_reference: "weather today sunny warm",
        min_ratio: 0.4,
        max_peak: 0.15,
        max_lag: 4096,
        ort_candidates: 16,
        min_samples: 15_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 0,
    },
    PhraseCase {
        label: "numbers",
        ipa: "wʌn tu θri fɔɹ faɪv sɪks sɛvən eɪt naɪn tɛn",
        voice: None,
        asr_reference: "one two three four five six seven eight nine ten",
        min_ratio: 0.35,
        max_peak: 0.15,
        max_lag: 4096,
        ort_candidates: 16,
        min_samples: 20_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 0,
    },
    PhraseCase {
        label: "welcome demo",
        ipa: "wɛlkəm tə ðə kɪtən tɛkst tə spitʃ demos",
        voice: None,
        asr_reference: "welcome kitten text speech",
        min_ratio: 0.4,
        max_peak: 0.15,
        max_lag: 4096,
        ort_candidates: 16,
        min_samples: 15_000,
        strict_asr: false,
        strict_waveform: false,
        asr_retries: 0,
    },
];

/// All phrases for manual WAV export (strict + extended).
pub fn all_export_phrases() -> Vec<&'static PhraseCase> {
    let mut out: Vec<&PhraseCase> = Vec::new();
    out.extend(SHORT_PHRASES.iter());
    out.extend(LONG_PHRASES.iter());
    out.extend(SHORT_PHRASES_EXTENDED.iter());
    out.extend(LONG_PHRASES_EXTENDED.iter());
    out
}
