// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Checkpoint key prefixes for HF Moonshine safetensors.

use rlx_core::weight_map::WeightMap;

/// Detected weight-key prefixes (`model.encoder` / `model.decoder` vs bare).
#[derive(Debug, Clone)]
pub struct MoonshineWeightPrefix {
    pub encoder: String,
    pub decoder: String,
    /// Top-level LM head key if present (`proj_out.weight` or prefixed).
    pub proj_out: Option<String>,
}

impl MoonshineWeightPrefix {
    pub fn detect(weights: &WeightMap) -> Self {
        Self::detect_with(|k| weights.has(k))
    }

    pub fn detect_with(has: impl Fn(&str) -> bool) -> Self {
        let (encoder, decoder) = if has("model.encoder.conv1.weight") {
            ("model.encoder".into(), "model.decoder".into())
        } else if has("encoder.conv1.weight") {
            ("encoder".into(), "decoder".into())
        } else {
            ("model.encoder".into(), "model.decoder".into())
        };
        let proj_out = if has("proj_out.weight") {
            Some("proj_out.weight".into())
        } else if has("model.proj_out.weight") {
            Some("model.proj_out.weight".into())
        } else {
            None
        };
        Self {
            encoder,
            decoder,
            proj_out,
        }
    }

    pub fn enc_layer(&self, i: usize, suffix: &str) -> String {
        format!("{}.layers.{i}.{suffix}", self.encoder)
    }

    pub fn dec_layer(&self, i: usize, suffix: &str) -> String {
        format!("{}.layers.{i}.{suffix}", self.decoder)
    }

    pub fn enc_conv1_w(&self) -> String {
        format!("{}.conv1.weight", self.encoder)
    }
    pub fn enc_conv2_w(&self) -> String {
        format!("{}.conv2.weight", self.encoder)
    }
    pub fn enc_conv2_b(&self) -> String {
        format!("{}.conv2.bias", self.encoder)
    }
    pub fn enc_conv3_w(&self) -> String {
        format!("{}.conv3.weight", self.encoder)
    }
    pub fn enc_conv3_b(&self) -> String {
        format!("{}.conv3.bias", self.encoder)
    }
    pub fn enc_groupnorm_w(&self) -> String {
        format!("{}.groupnorm.weight", self.encoder)
    }
    pub fn enc_groupnorm_b(&self) -> String {
        format!("{}.groupnorm.bias", self.encoder)
    }
    pub fn enc_ln_w(&self) -> String {
        format!("{}.layer_norm.weight", self.encoder)
    }
    pub fn dec_embed_tokens(&self) -> String {
        format!("{}.embed_tokens.weight", self.decoder)
    }
    pub fn dec_norm_w(&self) -> String {
        format!("{}.norm.weight", self.decoder)
    }

    /// Resolve `o_proj` vs `out_proj` for a layer attention block.
    pub fn attn_o_proj(has: &impl Fn(&str) -> bool, pfx: &str) -> String {
        let o = format!("{pfx}.o_proj.weight");
        if has(&o) {
            o
        } else {
            format!("{pfx}.out_proj.weight")
        }
    }
}
