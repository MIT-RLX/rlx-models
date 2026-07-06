// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 configuration — HF `config.json` and GGUF `llama.*` metadata.

use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Llama32RopeType {
    #[default]
    Default,
    #[serde(rename = "llama3")]
    Llama3,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32RopeScaling {
    pub factor: f32,
    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f32,
    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: Llama32RopeType,
}

fn default_low_freq_factor() -> f32 {
    1.0
}
fn default_high_freq_factor() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama32Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    /// Explicit head dim (Llama 3.x); when absent, derived from hidden/heads.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<Llama32RopeScaling>,
    /// RoPE pairing flavor. GGUF Llama weights are permuted by the HF→GGUF
    /// converter for llama.cpp's interleaved (`NORM`) RoPE, so GGUF-backed
    /// inference must rotate with [`rlx_ir::RopeStyle::GptJ`]; HF-safetensors
    /// checkpoints use [`rlx_ir::RopeStyle::NeoX`] (default). Not present in
    /// HF `config.json`, so skipped during deserialization.
    #[serde(skip)]
    pub rope_style: rlx_ir::RopeStyle,
    /// GGUF `general.architecture` tag when loaded from GGUF (`llama`, `phi3`, …).
    #[serde(skip)]
    pub gguf_arch: Option<String>,
    /// Rotary dimension when it differs from [`head_dim`] (Phi-3 partial RoPE).
    #[serde(skip)]
    pub rope_dim: Option<usize>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}

impl Llama32Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn from_gguf(raw: &GgufFile) -> anyhow::Result<Self> {
        llama32_cfg_from_gguf(raw)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// Leading per-head dims that receive RoPE (equals [`head_dim`] for Llama;
    /// may be smaller for Phi-3 partial RoPE).
    pub fn n_rot(&self) -> usize {
        self.rope_dim
            .filter(|&r| r > 0 && r <= self.head_dim())
            .unwrap_or_else(|| self.head_dim())
    }

    pub fn uses_partial_rope(&self) -> bool {
        self.n_rot() < self.head_dim()
    }

    pub fn is_phi_arch(&self) -> bool {
        matches!(self.gguf_arch.as_deref(), Some("phi3") | Some("phi4"))
    }

    #[cfg(test)]
    pub(crate) fn tiny_test() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
        }
    }
}

pub fn llama32_cfg_from_gguf(raw: &GgufFile) -> anyhow::Result<Llama32Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("llama");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("llama.")?;
            if arch_prefix == "llama" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> anyhow::Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("llama.embedding_length")? as usize;
    let num_attention_heads = get_u32("llama.attention.head_count")? as usize;
    let head_dim_key = get_u32("llama.attention.key_length")
        .ok()
        .map(|v| v as usize);
    let rope_dim = get_u32("llama.rope.dimension_count")
        .ok()
        .map(|v| v as usize);
    let head_dim = head_dim_key.or(rope_dim);

    let rope_scaling = match get_meta("llama.rope.scaling.type").and_then(MetaValue::as_str) {
        Some("none") | None => {
            // Llama 3.x often bakes scaling into rope_freqs.weight; HF fields may be absent.
            None
        }
        Some("linear") | Some("yarn") | Some("longrope") => {
            let factor = get_f32("llama.rope.scaling.factor")
                .or_else(|| get_f32("llama.rope.scale_linear"))
                .unwrap_or(1.0);
            let original = get_u32("llama.rope.scaling.original_context_length")
                .map(|v| v as usize)
                .unwrap_or(8192);
            Some(Llama32RopeScaling {
                factor,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: original,
                rope_type: Llama32RopeType::Llama3,
            })
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported llama.rope.scaling.type: {other:?}"
            ));
        }
    };

    Ok(Llama32Config {
        vocab_size: infer_vocab_size_from_gguf(raw),
        hidden_size,
        intermediate_size: get_u32("llama.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("llama.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("llama.attention.head_count_kv")? as usize,
        max_position_embeddings: get_u32("llama.context_length").unwrap_or(8192) as usize,
        rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64,
        rope_theta: get_f32("llama.rope.freq_base").unwrap_or(500_000.0) as f64,
        hidden_act: "silu".into(),
        tie_word_embeddings: get_bool("llama.tie_word_embeddings").unwrap_or_else(|| {
            // Llama-2 / TinyLlama GGUF often omits the flag; untied checkpoints
            // carry a separate `output.weight` tensor.
            !raw.tensors.contains_key("output.weight")
        }),
        attention_bias: false,
        head_dim,
        rope_scaling,
        // Phi-3/4 GGUF uses HF NeoX rotate-half; plain Llama GGUF is GPT-J.
        rope_style: if matches!(arch_prefix, "phi3" | "phi4") {
            rlx_ir::RopeStyle::NeoX
        } else {
            rlx_ir::RopeStyle::GptJ
        },
        gguf_arch: Some(arch_prefix.to_string()),
        rope_dim: rope_dim.filter(|r| head_dim_key.is_some() && *r <= head_dim_key.unwrap()),
    })
}

/// Resolve vocab size from GGUF metadata / tensors. Llama-3 GGUF carries
/// `llama.vocab_size`; older llama-tagged files (TinyLlama, SmolLM2, …) often
/// only expose `tokenizer.ggml.tokens` or an embed row count.
fn infer_vocab_size_from_gguf(raw: &GgufFile) -> usize {
    if let Some(v) = raw
        .metadata
        .get("llama.vocab_size")
        .and_then(MetaValue::as_u32)
    {
        return v as usize;
    }
    if let Some(MetaValue::Array(tokens)) = raw.metadata.get("tokenizer.ggml.tokens") {
        if !tokens.is_empty() {
            return tokens.len();
        }
    }
    for name in ["token_embd.weight", "model.embed_tokens.weight"] {
        if let Some(t) = raw.tensors.get(name) {
            if !t.shape.is_empty() {
                return t.shape[0];
            }
        }
    }
    128_256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama32_1b_like() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let cfg: Llama32Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_group_size(), 4);
        assert!(cfg.rope_scaling.is_some());
    }

    #[test]
    fn gguf_vocab_inferred_from_tokenizer_tokens() {
        use rlx_gguf::GgmlType;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_llama32_vocab_{}_{}_{}.gguf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&9u64.to_le_bytes()); // metadata keys

        let write_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        let write_string_array = |buf: &mut Vec<u8>, k: &str, items: &[String]| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
        };

        write_str(&mut buf, "general.architecture", "llama");
        write_u32(&mut buf, "llama.embedding_length", 2048);
        write_u32(&mut buf, "llama.feed_forward_length", 5632);
        write_u32(&mut buf, "llama.block_count", 22);
        write_u32(&mut buf, "llama.attention.head_count", 32);
        write_u32(&mut buf, "llama.attention.head_count_kv", 4);
        write_u32(&mut buf, "llama.context_length", 2048);
        write_u32(&mut buf, "llama.rope.freq_base", 10_000);
        let vocab = 128u32;
        let tokens: Vec<String> = (0..vocab).map(|i| format!("t{i}")).collect();
        write_string_array(&mut buf, "tokenizer.ggml.tokens", &tokens);

        let embed_bytes = vocab as u64 * 2048 * 4;
        for (name, rows, cols, offset) in [
            ("token_embd.weight", vocab as u64, 2048u64, 0u64),
            ("output.weight", 2048u64, vocab as u64, embed_bytes),
        ] {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&rows.to_le_bytes());
            buf.extend_from_slice(&cols.to_le_bytes());
            buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        let n_floats = (vocab as usize * 2048) * 2;
        for _ in 0..n_floats {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let raw = rlx_gguf::GgufFile::from_path(&path).expect("parse tinyllama-like gguf");
        let cfg = llama32_cfg_from_gguf(&raw).expect("llama32 config");
        assert_eq!(cfg.vocab_size, vocab as usize);
        assert!(!cfg.tie_word_embeddings);
        std::fs::remove_file(path).ok();
    }
}
