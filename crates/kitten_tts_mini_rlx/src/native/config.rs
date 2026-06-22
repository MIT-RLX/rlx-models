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

//! Fixed architecture constants for Kitten TTS mini 0.8 (native RLX graph).

/// GGUF architecture tag written by `scripts/onnx_decompose_to_gguf.py`.
pub const KITTEN_GGUF_ARCH: &str = "kitten-tts-mini";

/// Allowed GGUF arch strings for `load_weight_map` validation.
pub const KITTEN_GGUF_ARCHES: &[&str] = &[KITTEN_GGUF_ARCH, "onnx-decompose"];

/// Semantic module boundaries in the native graph (weight key / ONNX node prefixes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    /// `kmodel.bert.*` — Albert-style text encoder on phoneme ids.
    Bert,
    /// `/text_encoder/*`, `/text_encoder_1/*` — style-conditioned encoders + LSTM stacks.
    TextEncoder,
    /// `kmodel.decoder.*`, `/decoder/encode/*`, `/decoder/decode.*` — mel decoder.
    MelDecoder,
    /// `kmodel.predictor.*`, `/N.*`, `/F0.*` — duration / F0 predictors.
    Predictor,
    /// `/duration_proj/*`, `/Expand_1`, `/Where_1` — duration epilogue + feedback.
    Duration,
    /// `/decoder/generator/*` — HiFi-GAN vocoder (AdaIN resblocks, source filter).
    Vocoder,
}

impl ModuleKind {
    pub const ALL: [ModuleKind; 6] = [
        ModuleKind::Bert,
        ModuleKind::TextEncoder,
        ModuleKind::MelDecoder,
        ModuleKind::Predictor,
        ModuleKind::Duration,
        ModuleKind::Vocoder,
    ];

    pub fn doc(self) -> &'static str {
        match self {
            ModuleKind::Bert => "Phoneme embedding + Albert layers (128-d hidden)",
            ModuleKind::TextEncoder => "Style/speed fusion, quantized LSTM banks",
            ModuleKind::MelDecoder => "ASR + decode conv stacks → mel frames",
            ModuleKind::Predictor => "F0 + duration conv/LSTM heads",
            ModuleKind::Duration => "Duration projection, clip, fixed-point carry",
            ModuleKind::Vocoder => "Neural vocoder: upsample mel → 24 kHz waveform",
        }
    }

    pub fn weight_prefixes(self) -> &'static [&'static str] {
        match self {
            ModuleKind::Bert => &["kmodel.bert."],
            ModuleKind::TextEncoder => &["kmodel.text_encoder.", "/text_encoder/"],
            ModuleKind::MelDecoder => &["kmodel.decoder.", "/decoder/encode/", "/decoder/decode"],
            ModuleKind::Predictor => &["kmodel.predictor.", "/N.", "/F0."],
            ModuleKind::Duration => &["kmodel.predictor.duration", "/duration_proj/"],
            ModuleKind::Vocoder => &["/decoder/generator/", "kmodel.decoder.generator."],
        }
    }
}

/// Fixed hyper-parameters for [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8).
#[derive(Debug, Clone, Copy)]
pub struct KittenTtsConfig {
    pub vocab_size: usize,
    pub style_dim: usize,
    pub bert_hidden: usize,
    pub mel_div: usize,
    pub harmonics: usize,
    pub sample_rate: u32,
}

impl Default for KittenTtsConfig {
    fn default() -> Self {
        Self::mini_v0_8()
    }
}

impl KittenTtsConfig {
    pub fn mini_v0_8() -> Self {
        Self {
            vocab_size: 178,
            style_dim: 256,
            bert_hidden: 128,
            mel_div: 300,
            harmonics: 9,
            sample_rate: 24_000,
        }
    }

    pub fn frame_cap(&self, max_waveform_samples: usize) -> usize {
        max_waveform_samples.div_ceil(self.mel_div).max(1)
    }
}
