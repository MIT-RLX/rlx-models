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

//! Model configuration. Each family ships with faithful built-in defaults
//! (Paraformer-large / SenseVoiceSmall / FSMN-VAD / CT-Transformer / CAM++);
//! a FunASR `config.yaml` can override scalar fields through a small
//! section-aware YAML scalar reader (no external YAML dependency).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::frontend::FrontendConfig;

/// Assign a parsed YAML scalar to a config field only when present.
macro_rules! set {
    ($opt:expr => $dst:expr) => {
        if let Some(v) = $opt {
            $dst = v;
        }
    };
}

/// Which FunASR model a checkpoint directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Non-autoregressive Paraformer ASR.
    Paraformer,
    /// SenseVoiceSmall multilingual CTC ASR.
    SenseVoice,
    /// FSMN voice-activity detection.
    FsmnVad,
    /// CT-Transformer punctuation restoration.
    CtTransformer,
    /// CAM++ speaker embedding.
    CamPlus,
}

/// SAN-M encoder hyper-parameters, shared by Paraformer / SenseVoice / punc.
#[derive(Debug, Clone)]
pub struct SanmEncoderConfig {
    /// Input feature dimension (e.g. 560 = 80 mels × LFR-7).
    pub input_size: usize,
    /// Model / attention dimension `d`.
    pub output_size: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feed-forward hidden width.
    pub linear_units: usize,
    /// Number of main encoder blocks (`encoders0` + `encoders`).
    pub num_blocks: usize,
    /// Extra "temporal processing" blocks after the main stack (SenseVoice).
    pub tp_blocks: usize,
    /// FSMN depthwise-conv kernel size.
    pub kernel_size: usize,
    /// FSMN left-context shift (0 = symmetric padding).
    pub sanm_shfit: usize,
    /// LayerNorm epsilon.
    pub ln_eps: f32,
}

impl SanmEncoderConfig {
    /// Per-head dimension (`output_size / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.output_size / self.n_heads
    }
}

/// CIF predictor (`CifPredictorV2`).
#[derive(Debug, Clone)]
pub struct CifConfig {
    /// Input (encoder) dimension.
    pub idim: usize,
    /// Left context of the alpha conv.
    pub l_order: usize,
    /// Right context of the alpha conv.
    pub r_order: usize,
    /// Firing threshold (1.0).
    pub threshold: f32,
    /// Tail-handling threshold that forces a final partial fire.
    pub tail_threshold: f32,
    /// Alpha smoothing factor.
    pub smooth_factor: f32,
    /// Alpha noise floor (subtracted before ReLU).
    pub noise_threshold: f32,
}

/// SAN-M decoder (`ParaformerSANMDecoder`).
#[derive(Debug, Clone)]
pub struct SanmDecoderConfig {
    /// Model dimension `d`.
    pub dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feed-forward hidden width.
    pub linear_units: usize,
    /// Total decoder blocks.
    pub num_blocks: usize,
    /// Number of leading blocks that include cross-attention.
    pub att_layer_num: usize,
    /// FSMN self-attention kernel size.
    pub self_kernel: usize,
    /// FSMN self-attention left-context shift.
    pub self_sanm_shfit: usize,
    /// LayerNorm epsilon.
    pub ln_eps: f32,
}

impl SanmDecoderConfig {
    /// Per-head dimension (`dim / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

/// Paraformer-large (`paraformer-zh`) configuration.
#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    /// Acoustic frontend (fbank + LFR + CMVN).
    pub frontend: FrontendConfig,
    /// SAN-M encoder.
    pub encoder: SanmEncoderConfig,
    /// CIF predictor head.
    pub predictor: CifConfig,
    /// SAN-M decoder.
    pub decoder: SanmDecoderConfig,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// Blank token id (removed from output).
    pub blank_id: usize,
    /// Start-of-sequence id.
    pub sos: usize,
    /// End-of-sequence id.
    pub eos: usize,
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        let d = 512;
        Self {
            frontend: FrontendConfig::default(),
            encoder: SanmEncoderConfig {
                input_size: 560,
                output_size: d,
                n_heads: 4,
                linear_units: 2048,
                num_blocks: 50,
                tp_blocks: 0,
                kernel_size: 11,
                sanm_shfit: 0,
                ln_eps: 1e-12,
            },
            predictor: CifConfig {
                idim: d,
                l_order: 1,
                r_order: 1,
                threshold: 1.0,
                tail_threshold: 0.45,
                smooth_factor: 1.0,
                noise_threshold: 0.0,
            },
            decoder: SanmDecoderConfig {
                dim: d,
                n_heads: 4,
                linear_units: 2048,
                num_blocks: 16,
                att_layer_num: 16,
                self_kernel: 11,
                self_sanm_shfit: 0,
                ln_eps: 1e-12,
            },
            vocab_size: 4752,
            blank_id: 0,
            sos: 1,
            eos: 2,
        }
    }
}

impl ParaformerConfig {
    /// Override the defaults from a FunASR `config.yaml` string.
    pub fn from_yaml(text: &str) -> Self {
        let y = Yaml::parse(text);
        let mut c = Self::default();
        set!(y.us("frontend_conf", "n_mels") => c.frontend.n_mels);
        set!(y.us("frontend_conf", "lfr_m") => c.frontend.lfr_m);
        set!(y.us("frontend_conf", "lfr_n") => c.frontend.lfr_n);
        if let Some(v) = y.us("encoder_conf", "output_size") {
            c.encoder.output_size = v;
            c.predictor.idim = v;
            c.decoder.dim = v;
        }
        set!(y.us("encoder_conf", "attention_heads") => c.encoder.n_heads);
        set!(y.us("encoder_conf", "linear_units") => c.encoder.linear_units);
        set!(y.us("encoder_conf", "num_blocks") => c.encoder.num_blocks);
        set!(y.us("encoder_conf", "kernel_size") => c.encoder.kernel_size);
        set!(y.us("predictor_conf", "l_order") => c.predictor.l_order);
        set!(y.us("predictor_conf", "r_order") => c.predictor.r_order);
        set!(y.f("predictor_conf", "threshold") => c.predictor.threshold);
        set!(y.f("predictor_conf", "tail_threshold") => c.predictor.tail_threshold);
        set!(y.us("decoder_conf", "attention_heads") => c.decoder.n_heads);
        set!(y.us("decoder_conf", "linear_units") => c.decoder.linear_units);
        if let Some(v) = y.us("decoder_conf", "num_blocks") {
            c.decoder.num_blocks = v;
            c.decoder.att_layer_num = v;
        }
        set!(y.us("decoder_conf", "att_layer_num") => c.decoder.att_layer_num);
        set!(y.us("decoder_conf", "kernel_size") => c.decoder.self_kernel);
        // input feature dim follows the frontend.
        c.encoder.input_size = c.frontend.n_mels * c.frontend.lfr_m;
        c
    }

    /// Load from `<dir>/config.yaml`, falling back to defaults if absent.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.yaml");
        if p.is_file() {
            let t = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            Ok(Self::from_yaml(&t))
        } else {
            Ok(Self::default())
        }
    }
}

/// SenseVoiceSmall configuration.
#[derive(Debug, Clone)]
pub struct SenseVoiceConfig {
    /// Acoustic frontend (fbank + LFR).
    pub frontend: FrontendConfig,
    /// SAN-M encoder (main + temporal-processing blocks).
    pub encoder: SanmEncoderConfig,
    /// CTC vocabulary size.
    pub vocab_size: usize,
    /// Blank token id (removed from CTC output).
    pub blank_id: usize,
    /// Embedding rows used as the [LID, EVENT, EMO, TEXTNORM] prompt prefix.
    pub embed_vocab: usize,
}

impl Default for SenseVoiceConfig {
    fn default() -> Self {
        Self {
            frontend: FrontendConfig::default(),
            encoder: SanmEncoderConfig {
                input_size: 560,
                output_size: 512,
                n_heads: 4,
                linear_units: 2048,
                num_blocks: 50,
                tp_blocks: 20,
                kernel_size: 11,
                sanm_shfit: 0,
                ln_eps: 1e-12,
            },
            vocab_size: 25055,
            blank_id: 0,
            embed_vocab: 16,
        }
    }
}

impl SenseVoiceConfig {
    /// `lid_dict` — language → embedding row index.
    pub fn lid(lang: &str) -> usize {
        match lang {
            "zh" => 3,
            "en" => 4,
            "yue" => 7,
            "ja" => 11,
            "ko" => 12,
            "nospeech" => 13,
            _ => 0, // "auto"
        }
    }
    /// `textnorm_dict` — with/without inverse-text-normalization.
    pub fn textnorm(use_itn: bool) -> usize {
        if use_itn { 14 } else { 15 }
    }

    /// Override the defaults from a FunASR `config.yaml` string.
    pub fn from_yaml(text: &str) -> Self {
        let y = Yaml::parse(text);
        let mut c = Self::default();
        set!(y.us("encoder_conf", "output_size") => c.encoder.output_size);
        set!(y.us("encoder_conf", "attention_heads") => c.encoder.n_heads);
        set!(y.us("encoder_conf", "linear_units") => c.encoder.linear_units);
        set!(y.us("encoder_conf", "num_blocks") => c.encoder.num_blocks);
        set!(y.us("encoder_conf", "tp_blocks") => c.encoder.tp_blocks);
        c
    }

    /// Load from `<dir>/config.yaml`, falling back to defaults if absent.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.yaml");
        if p.is_file() {
            let t = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            Ok(Self::from_yaml(&t))
        } else {
            Ok(Self::default())
        }
    }
}

/// FSMN-VAD encoder + decision configuration.
#[derive(Debug, Clone)]
pub struct FsmnVadConfig {
    /// Acoustic frontend (fbank + LFR-5).
    pub frontend: FrontendConfig,
    /// Encoder input dimension (80 mels × LFR-5 = 400).
    pub input_dim: usize,
    /// First affine projection width.
    pub input_affine_dim: usize,
    /// Number of DFSMN blocks.
    pub fsmn_layers: usize,
    /// Hidden width between projections.
    pub linear_dim: usize,
    /// DFSMN memory-block projection width.
    pub proj_dim: usize,
    /// DFSMN left-context order.
    pub lorder: usize,
    /// DFSMN right-context (future) order.
    pub rorder: usize,
    /// DFSMN left dilation/stride.
    pub lstride: usize,
    /// DFSMN right dilation/stride.
    pub rstride: usize,
    /// Output affine projection width.
    pub output_affine_dim: usize,
    /// Posterior output dimension (pdf count).
    pub output_dim: usize,
    /// Silence (ms) that ends a speech segment.
    pub max_end_silence_ms: f32,
    /// Maximum single-segment length (ms) before forced split.
    pub max_single_segment_ms: f32,
    /// Speech-vs-noise decision threshold.
    pub speech_noise_thres: f32,
    /// Index of the silence posterior in the output.
    pub sil_pdf_id: usize,
}

impl Default for FsmnVadConfig {
    fn default() -> Self {
        let mut fe = FrontendConfig {
            lfr_m: 5,
            lfr_n: 1,
            ..FrontendConfig::default()
        };
        fe.n_mels = 80;
        Self {
            frontend: fe,
            input_dim: 400,
            input_affine_dim: 140,
            fsmn_layers: 4,
            linear_dim: 250,
            proj_dim: 128,
            lorder: 20,
            rorder: 0,
            lstride: 1,
            rstride: 1,
            output_affine_dim: 140,
            output_dim: 248,
            max_end_silence_ms: 800.0,
            max_single_segment_ms: 60_000.0,
            speech_noise_thres: 0.6,
            sil_pdf_id: 0,
        }
    }
}

impl FsmnVadConfig {
    /// Load from `<dir>/config.yaml`, falling back to defaults if absent.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.yaml");
        if p.is_file() {
            let t = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            let y = Yaml::parse(&t);
            let mut c = Self::default();
            set!(y.us("encoder_conf", "input_dim") => c.input_dim);
            set!(y.us("encoder_conf", "fsmn_layers") => c.fsmn_layers);
            set!(y.us("encoder_conf", "proj_dim") => c.proj_dim);
            set!(y.us("encoder_conf", "lorder") => c.lorder);
            set!(y.us("encoder_conf", "output_dim") => c.output_dim);
            Ok(c)
        } else {
            Ok(Self::default())
        }
    }
}

/// CT-Transformer punctuation configuration.
#[derive(Debug, Clone)]
pub struct CtTransformerConfig {
    /// Input token vocabulary size.
    pub vocab_size: usize,
    /// Token embedding dimension.
    pub embed_unit: usize,
    /// SAN-M encoder over the token embeddings.
    pub encoder: SanmEncoderConfig,
    /// Punctuation labels (index = class id; `_` = none).
    pub punc_list: Vec<String>,
    /// Punctuation id marking a sentence end.
    pub sentence_end_id: usize,
}

impl Default for CtTransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 272727,
            embed_unit: 256,
            encoder: SanmEncoderConfig {
                input_size: 256,
                output_size: 256,
                n_heads: 8,
                linear_units: 1024,
                num_blocks: 4,
                tp_blocks: 0,
                kernel_size: 11,
                sanm_shfit: 0,
                ln_eps: 1e-12,
            },
            punc_list: ["<unk>", "_", "，", "。", "？", "、"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            sentence_end_id: 3,
        }
    }
}

impl CtTransformerConfig {
    /// Load from `<dir>/config.yaml`, falling back to defaults if absent.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.yaml");
        if !p.is_file() {
            return Ok(Self::default());
        }
        let t = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let y = Yaml::parse(&t);
        let mut c = Self::default();
        if let Some(d) = y.us("encoder_conf", "output_size") {
            c.encoder.output_size = d;
            c.encoder.input_size = d; // embedding feeds the encoder directly
            c.embed_unit = d;
        }
        set!(y.us("encoder_conf", "attention_heads") => c.encoder.n_heads);
        set!(y.us("encoder_conf", "linear_units") => c.encoder.linear_units);
        set!(y.us("encoder_conf", "num_blocks") => c.encoder.num_blocks);
        set!(y.us("encoder_conf", "kernel_size") => c.encoder.kernel_size);
        Ok(c)
    }
}

/// CAM++ speaker-embedding configuration.
#[derive(Debug, Clone)]
pub struct CamPlusConfig {
    /// Input fbank dimension.
    pub feat_dim: usize,
    /// Output speaker-embedding dimension (192).
    pub embedding_size: usize,
    /// Dense-block channel growth per layer.
    pub growth_rate: usize,
    /// Bottleneck size multiplier (`bn_channels = bn_size * growth_rate`).
    pub bn_size: usize,
    /// Channels after the initial TDNN.
    pub init_channels: usize,
    /// (num_layers, kernel_size, dilation) for each dense block.
    pub blocks: Vec<(usize, usize, usize)>,
    /// BatchNorm epsilon.
    pub bn_eps: f32,
}

impl Default for CamPlusConfig {
    fn default() -> Self {
        Self {
            feat_dim: 80,
            embedding_size: 192,
            growth_rate: 32,
            bn_size: 4,
            init_channels: 128,
            blocks: vec![(12, 3, 1), (24, 3, 2), (16, 3, 2)],
            bn_eps: 1e-5,
        }
    }
}

/// Tiny section-aware YAML scalar reader: top-level (column-0) keys ending in
/// `:` open a section; two-space-indented `key: value` lines are its leaves.
struct Yaml {
    sections: HashMap<String, HashMap<String, String>>,
}

impl Yaml {
    fn parse(text: &str) -> Self {
        let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut cur = String::new();
        for line in text.lines() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let trimmed = line.trim_end();
            if indent == 0 {
                if let Some(k) = trimmed.strip_suffix(':') {
                    cur = k.trim().to_string();
                    sections.entry(cur.clone()).or_default();
                } else if let Some((k, _)) = trimmed.split_once(':') {
                    cur = k.trim().to_string();
                    sections.entry(cur.clone()).or_default();
                }
            } else if let Some((k, v)) = trimmed.split_once(':') {
                let v = v.trim();
                if !v.is_empty() {
                    sections
                        .entry(cur.clone())
                        .or_default()
                        .insert(k.trim().to_string(), v.to_string());
                }
            }
        }
        Self { sections }
    }

    fn raw(&self, section: &str, key: &str) -> Option<&str> {
        self.sections.get(section)?.get(key).map(|s| s.as_str())
    }
    fn us(&self, section: &str, key: &str) -> Option<usize> {
        self.raw(section, key)?.parse().ok()
    }
    fn f(&self, section: &str, key: &str) -> Option<f32> {
        self.raw(section, key)?.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paraformer_yaml_override() {
        let yaml = "encoder_conf:\n  output_size: 256\n  num_blocks: 30\ndecoder_conf:\n  num_blocks: 8\npredictor_conf:\n  tail_threshold: 0.5\n";
        let c = ParaformerConfig::from_yaml(yaml);
        assert_eq!(c.encoder.output_size, 256);
        assert_eq!(c.encoder.num_blocks, 30);
        assert_eq!(c.decoder.num_blocks, 8);
        assert_eq!(c.decoder.dim, 256);
        assert!((c.predictor.tail_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sensevoice_defaults() {
        let c = SenseVoiceConfig::default();
        assert_eq!(c.encoder.num_blocks, 50);
        assert_eq!(c.encoder.tp_blocks, 20);
        assert_eq!(SenseVoiceConfig::lid("zh"), 3);
        assert_eq!(SenseVoiceConfig::textnorm(false), 15);
    }
}
