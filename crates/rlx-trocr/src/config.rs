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

//! `VisionEncoderDecoderModel` configuration as TrOCR checkpoints ship it:
//! a ViT (`model_type: vit`, base / large) or DeiT (`deit`, small) encoder
//! and a `trocr` decoder, plus `preprocessor_config.json`.

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::image_preprocess::Filter;
use serde_json::Value;
use std::path::Path;

/// Prefix tokens before the patch tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    /// `[CLS]` + patches.
    Vit,
    /// `[CLS]` + distillation token + patches.
    Deit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub kind: EncoderKind,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub eps: f32,
    pub qkv_bias: bool,
}

impl EncoderConfig {
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn prefix_tokens(&self) -> usize {
        match self.kind {
            EncoderKind::Vit => 1,
            EncoderKind::Deit => 2,
        }
    }

    /// Encoder sequence length (prefix + patches).
    pub fn seq(&self) -> usize {
        self.prefix_tokens() + self.grid() * self.grid()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderAct {
    Relu,
    Gelu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub act: DecoderAct,
    pub vocab: usize,
    pub max_positions: usize,
    /// Token embeddings are multiplied by `√d_model`.
    pub scale_embedding: bool,
    pub layernorm_embedding: bool,
    /// Width of the encoder states the cross-attention reads.
    pub cross_hidden: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Preprocess {
    pub size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub filter: Filter,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrOcrConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub decoder_start: u32,
    pub eos: u32,
    pub pad: u32,
    pub preprocess: Preprocess,
}

fn uint(v: &Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| anyhow!("config: `{k}` missing"))
}

fn flag(v: &Value, k: &str, default: bool) -> bool {
    v.get(k).and_then(Value::as_bool).unwrap_or(default)
}

impl TrOcrConfig {
    /// Read `config.json` and `preprocessor_config.json` from a model
    /// directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Value> {
            let p = dir.join(name);
            let s =
                std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))
        };
        let cfg = read("config.json")?;
        let pre = read("preprocessor_config.json").ok();
        Self::from_values(&cfg, pre.as_ref())
    }

    pub fn from_values(cfg: &Value, pre: Option<&Value>) -> Result<Self> {
        let e = cfg
            .get("encoder")
            .ok_or_else(|| anyhow!("config has no encoder"))?;
        let d = cfg
            .get("decoder")
            .ok_or_else(|| anyhow!("config has no decoder"))?;
        let kind = match e.get("model_type").and_then(Value::as_str) {
            Some("vit") => EncoderKind::Vit,
            Some("deit") => EncoderKind::Deit,
            other => bail!("unsupported TrOCR encoder {other:?} (vit / deit)"),
        };
        ensure!(
            d.get("model_type").and_then(Value::as_str) == Some("trocr"),
            "decoder is not a `trocr` decoder"
        );
        let act = |v: &Value, k: &str, default: &str| -> Result<String> {
            Ok(v.get(k)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string())
        };
        ensure!(
            act(e, "hidden_act", "gelu")? == "gelu",
            "encoder activation must be gelu"
        );
        let encoder = EncoderConfig {
            kind,
            hidden: uint(e, "hidden_size")?,
            layers: uint(e, "num_hidden_layers")?,
            heads: uint(e, "num_attention_heads")?,
            intermediate: uint(e, "intermediate_size")?,
            image_size: uint(e, "image_size")?,
            patch_size: uint(e, "patch_size")?,
            channels: e.get("num_channels").and_then(Value::as_u64).unwrap_or(3) as usize,
            eps: e
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-12) as f32,
            qkv_bias: flag(e, "qkv_bias", true),
        };
        ensure!(
            flag(d, "use_learned_position_embeddings", true),
            "sinusoidal-position TrOCR decoders are not supported"
        );
        let d_model = uint(d, "d_model")?;
        let decoder = DecoderConfig {
            d_model,
            layers: uint(d, "decoder_layers")?,
            heads: uint(d, "decoder_attention_heads")?,
            ffn: uint(d, "decoder_ffn_dim")?,
            act: match act(d, "activation_function", "gelu")?.as_str() {
                "relu" => DecoderAct::Relu,
                "gelu" => DecoderAct::Gelu,
                other => bail!("unsupported decoder activation {other:?}"),
            },
            vocab: uint(d, "vocab_size")?,
            max_positions: uint(d, "max_position_embeddings")?,
            scale_embedding: flag(d, "scale_embedding", false),
            layernorm_embedding: flag(d, "layernorm_embedding", true),
            cross_hidden: d
                .get("cross_attention_hidden_size")
                .and_then(Value::as_u64)
                .map_or(encoder.hidden, |n| n as usize),
        };
        ensure!(
            decoder.cross_hidden == encoder.hidden,
            "encoder width {} differs from the decoder's cross-attention width {} \
             (an enc_to_dec_proj checkpoint) — not supported",
            encoder.hidden,
            decoder.cross_hidden
        );
        let tok = |k: &str, default: u32| -> u32 {
            cfg.get(k)
                .or_else(|| d.get(k))
                .and_then(Value::as_u64)
                .map_or(default, |n| n as u32)
        };
        let preprocess = match pre {
            Some(p) => Preprocess {
                size: p
                    .get("size")
                    .and_then(|s| s.as_u64().or_else(|| s.get("height")?.as_u64()))
                    .map_or(encoder.image_size, |n| n as usize),
                mean: triple(p, "image_mean")?.unwrap_or([0.5; 3]),
                std: triple(p, "image_std")?.unwrap_or([0.5; 3]),
                filter: match p.get("resample").and_then(Value::as_u64) {
                    Some(2) => Filter::Bilinear,
                    Some(3) | None => Filter::Bicubic,
                    Some(1) => Filter::Lanczos,
                    Some(r) => bail!("unsupported resample {r}"),
                },
            },
            None => Preprocess {
                size: encoder.image_size,
                mean: [0.5; 3],
                std: [0.5; 3],
                filter: Filter::Bicubic,
            },
        };
        ensure!(
            preprocess.size == encoder.image_size,
            "processor size {} differs from the encoder's {}",
            preprocess.size,
            encoder.image_size
        );
        Ok(Self {
            encoder,
            decoder,
            decoder_start: tok("decoder_start_token_id", 2),
            eos: tok("eos_token_id", 2),
            pad: tok("pad_token_id", 1),
            preprocess,
        })
    }
}

fn triple(p: &Value, k: &str) -> Result<Option<[f32; 3]>> {
    let Some(v) = p.get(k) else { return Ok(None) };
    let a = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| anyhow!("{k}: expected 3 numbers"))?;
    let f = |i: usize| a[i].as_f64().unwrap_or(0.0) as f32;
    Ok(Some([f(0), f(1), f(2)]))
}
