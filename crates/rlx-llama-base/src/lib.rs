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

//! Shared "Llama-shaped" architecture config for the M4 family group.
//!
//! All the families listed in `PLAN.md` M4 share the same skeleton —
//! GQA + RoPE + RMSNorm + SwiGLU FFN, differing mainly in RoPE base/
//! scaling, sliding-window setting, and exact tensor-name conventions:
//!
//! | family       | llama.cpp arch tag      | RoPE scaling     | notes                                  |
//! |--------------|-------------------------|------------------|----------------------------------------|
//! | Mistral 3+   | `mistral3` / `mistral4` | Default          | Sliding-window (5K for 3.5)            |
//! | Phi 3 / 4    | `phi3`                  | Default + YaRN   | (`phi4` reuses `phi3` upstream)        |
//! | Bonsai       | (`llama`)               | Default          | Ships as llama-tagged GGUF             |
//! | OmniCoder    | (`qwen3`)               | Default          | Ships as qwen3-tagged GGUF             |
//! | Granite      | `granite`               | Default          | IBM Llama-shaped                       |
//! | Command-R    | `command-r` / `cohere2` | Default          | Cohere Llama-shaped                    |
//!
//! [`LlamaBaseConfig`] captures the union of fields the rlx-models M4
//! stub crates need; `from_gguf_path` reads them straight from a GGUF
//! header (no full weight load).
//!
//! Existing `rlx-llama32` / `rlx-qwen3` / `rlx-gemma` keep their own
//! per-family configs for now — this crate is the **new** shared base
//! the M4 stubs grow against. A future cleanup may migrate the existing
//! crates to use it, but that's an invasive refactor explicitly out of
//! scope for the M4 stub work.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// PLAN.md milestone this crate is the foundation for.
pub const PLAN_MILESTONE: &str = "M4";

/// Shared Llama-shaped arch config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaBaseConfig {
    /// llama.cpp `general.architecture` tag this config was sourced from.
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Per-expert FFN dim for MoE; total FFN dim for dense.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit `head_dim` when GGUF carries it; otherwise infer as
    /// `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    /// Sliding-window size in tokens; `None` = full attention. Mistral 3.5
    /// uses 5120; Mistral 7B uses 4096; most Llama-shaped models have `None`.
    pub sliding_window: Option<usize>,
    /// Maximum context window the checkpoint was trained / declared for.
    pub max_position_embeddings: usize,
}

/// RoPE scaling variants llama.cpp recognises across the M4 family set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RopeScaling {
    /// `linear` — multiply position indices by `1/factor`.
    Linear { factor: f32 },
    /// `dynamic` — NTK-aware scaling that re-bases per inference length.
    Dynamic { factor: f32 },
    /// `llama3` — Llama-3 / 3.1 piecewise frequency rescaling.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
    /// `yarn` — YaRN extension (Phi-4 long-context). `attention_factor` is
    /// the multiplier applied to attention scores after rescaling.
    YaRN {
        factor: f32,
        original_max_position_embeddings: usize,
        attention_factor: Option<f32>,
        beta_fast: Option<f32>,
        beta_slow: Option<f32>,
    },
}

impl LlamaBaseConfig {
    /// Convenience: effective head dim, inferred when not explicit.
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// GQA group size (`num_heads / num_kv_heads`). 1 = MHA.
    pub fn gqa_groups(&self) -> usize {
        self.num_attention_heads
            .checked_div(self.num_key_value_heads)
            .unwrap_or(1)
    }

    /// Read directly from a GGUF file on disk.
    pub fn from_gguf_path(path: &Path) -> Result<Self> {
        let raw = GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?;
        Self::from_gguf(&raw)
    }

    /// Read from an already-parsed GGUF.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow::anyhow!("GGUF missing general.architecture"))?
            .to_string();

        let req_u32 = |suffix: &str| -> Result<u32> {
            let key = format!("{arch}.{suffix}");
            raw.metadata
                .get(&key)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow::anyhow!("GGUF missing required key `{key}`"))
        };
        let opt_u32 = |suffix: &str| -> Option<u32> {
            let key = format!("{arch}.{suffix}");
            raw.metadata.get(&key).and_then(MetaValue::as_u32)
        };
        let req_u64 = |suffix: &str| -> Result<u64> {
            let key = format!("{arch}.{suffix}");
            raw.metadata
                .get(&key)
                .and_then(MetaValue::as_u64)
                .ok_or_else(|| anyhow::anyhow!("GGUF missing required key `{key}`"))
        };
        let opt_f32 = |suffix: &str| -> Option<f32> {
            let key = format!("{arch}.{suffix}");
            raw.metadata.get(&key).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                MetaValue::F64(x) => Some(*x as f32),
                _ => None,
            })
        };

        let context_length = req_u64("context_length")? as usize;
        let embedding_length = req_u32("embedding_length")? as usize;
        let block_count = req_u32("block_count")? as usize;
        let ffn_length = req_u32("feed_forward_length")? as usize;
        let head_count = req_u32("attention.head_count")? as usize;
        let head_count_kv = opt_u32("attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(head_count);
        let head_dim = opt_u32("attention.key_length").map(|v| v as usize);
        let rms_norm_eps = opt_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f64;
        let rope_theta = opt_f32("rope.freq_base").unwrap_or(10_000.0) as f64;
        let sliding_window = opt_u32("attention.sliding_window")
            .filter(|v| *v > 0)
            .map(|v| v as usize);

        // Vocab from tokenizer.ggml.tokens length when present, else
        // `<arch>.vocab_size` if some converters carry it.
        let vocab_size = raw
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetaValue::Array(arr) => Some(arr.len()),
                _ => None,
            })
            .or_else(|| opt_u32("vocab_size").map(|v| v as usize))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GGUF missing vocab_size — neither tokenizer.ggml.tokens \
                     nor {arch}.vocab_size present"
                )
            })?;

        let rope_scaling = parse_rope_scaling(raw, &arch)?;

        Ok(LlamaBaseConfig {
            arch,
            vocab_size,
            hidden_size: embedding_length,
            intermediate_size: ffn_length,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: head_count_kv,
            head_dim,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
            sliding_window,
            max_position_embeddings: context_length,
        })
    }
}

fn parse_rope_scaling(raw: &GgufFile, arch: &str) -> Result<Option<RopeScaling>> {
    let kind = raw
        .metadata
        .get(&format!("{arch}.rope.scaling.type"))
        .and_then(MetaValue::as_str)
        .map(str::to_string);
    let factor = raw
        .metadata
        .get(&format!("{arch}.rope.scaling.factor"))
        .and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            MetaValue::F64(x) => Some(*x as f32),
            _ => None,
        });
    let orig_ctx = raw
        .metadata
        .get(&format!("{arch}.rope.scaling.original_context_length"))
        .and_then(MetaValue::as_u32)
        .map(|v| v as usize);
    let Some(kind) = kind else {
        return Ok(None);
    };
    let factor = factor.unwrap_or(1.0);
    match kind.as_str() {
        "linear" => Ok(Some(RopeScaling::Linear { factor })),
        "dynamic" => Ok(Some(RopeScaling::Dynamic { factor })),
        "llama3" => {
            let orig = orig_ctx.ok_or_else(|| {
                anyhow::anyhow!(
                    "{arch}.rope.scaling.type=llama3 requires \
                     {arch}.rope.scaling.original_context_length"
                )
            })?;
            // Defaults from Meta's Llama-3 release (low=1.0, high=4.0).
            Ok(Some(RopeScaling::Llama3 {
                factor,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: orig,
            }))
        }
        "yarn" => Ok(Some(RopeScaling::YaRN {
            factor,
            original_max_position_embeddings: orig_ctx.unwrap_or(0),
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
        })),
        other => bail!("unsupported {arch}.rope.scaling.type: {other:?}"),
    }
}

/// Per-family preset hints. Returned as `&'static` so M4 stub crates can
/// reference them without owning the data. Not all fields are
/// authoritative — they're starting points that a `LlamaBaseConfig`
/// derived from real weights should override.
pub fn family_preset(arch_tag: &str) -> Option<FamilyPreset> {
    match arch_tag {
        "mistral3" => Some(FamilyPreset {
            name: "Mistral 3",
            sliding_window_default: Some(5120),
            rope_theta_default: 1_000_000.0,
        }),
        "mistral4" => Some(FamilyPreset {
            name: "Mistral 4",
            sliding_window_default: None,
            rope_theta_default: 1_000_000.0,
        }),
        "phi3" | "phi4" => Some(FamilyPreset {
            name: "Phi 3 / Phi 4",
            sliding_window_default: None,
            rope_theta_default: 10_000.0,
        }),
        "granite" => Some(FamilyPreset {
            name: "Granite",
            sliding_window_default: None,
            rope_theta_default: 10_000.0,
        }),
        "command-r" | "cohere2" => Some(FamilyPreset {
            name: "Command-R / Cohere",
            sliding_window_default: None,
            rope_theta_default: 10_000.0,
        }),
        _ => None,
    }
}

/// Preset hint returned by [`family_preset`].
#[derive(Debug, Clone, Copy)]
pub struct FamilyPreset {
    pub name: &'static str,
    pub sliding_window_default: Option<usize>,
    pub rope_theta_default: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Builds a minimal GGUF with the keys [`LlamaBaseConfig::from_gguf`]
    /// reads, then verifies it parses with the expected values.
    fn write_test_gguf(arch: &str, extras: &[(&str, MetaValueOwned)]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rlx_llama_base_{}_{}_{}.gguf",
            arch,
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        let total_kv = 1 + extras.len();
        buf.extend_from_slice(&(total_kv as u64).to_le_bytes());

        write_string_kv(&mut buf, "general.architecture", arch);
        for (k, v) in extras {
            match v {
                MetaValueOwned::Str(s) => write_string_kv(&mut buf, k, s),
                MetaValueOwned::U32(n) => write_u32_kv(&mut buf, k, *n),
                MetaValueOwned::U64(n) => write_u64_kv(&mut buf, k, *n),
                MetaValueOwned::F32(n) => write_f32_kv(&mut buf, k, *n),
                MetaValueOwned::StringArray(items) => write_string_array_kv(&mut buf, k, items),
            }
        }
        // dummy f32 tensor
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();
        path
    }

    enum MetaValueOwned {
        Str(String),
        U32(u32),
        U64(u64),
        F32(f32),
        StringArray(Vec<String>),
    }

    fn write_string_kv(buf: &mut Vec<u8>, k: &str, v: &str) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    }
    fn write_u32_kv(buf: &mut Vec<u8>, k: &str, v: u32) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64_kv(buf: &mut Vec<u8>, k: &str, v: u64) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_f32_kv(buf: &mut Vec<u8>, k: &str, v: f32) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_string_array_kv(buf: &mut Vec<u8>, k: &str, items: &[String]) {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // String element type
        buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for s in items {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
    }

    #[test]
    fn parses_dense_mistral3_gguf() {
        let path = write_test_gguf(
            "mistral3",
            &[
                ("mistral3.context_length", MetaValueOwned::U64(32768)),
                ("mistral3.embedding_length", MetaValueOwned::U32(4096)),
                ("mistral3.block_count", MetaValueOwned::U32(32)),
                ("mistral3.feed_forward_length", MetaValueOwned::U32(14336)),
                ("mistral3.attention.head_count", MetaValueOwned::U32(32)),
                ("mistral3.attention.head_count_kv", MetaValueOwned::U32(8)),
                (
                    "mistral3.attention.layer_norm_rms_epsilon",
                    MetaValueOwned::F32(1e-5),
                ),
                ("mistral3.rope.freq_base", MetaValueOwned::F32(1_000_000.0)),
                (
                    "mistral3.attention.sliding_window",
                    MetaValueOwned::U32(5120),
                ),
                (
                    "tokenizer.ggml.tokens",
                    MetaValueOwned::StringArray(
                        (0..32).map(|i| format!("t{i}")).collect::<Vec<_>>(),
                    ),
                ),
            ],
        );
        let cfg = LlamaBaseConfig::from_gguf_path(&path).expect("parse mistral3");
        assert_eq!(cfg.arch, "mistral3");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.gqa_groups(), 4);
        assert_eq!(cfg.effective_head_dim(), 128);
        assert_eq!(cfg.sliding_window, Some(5120));
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert_eq!(cfg.vocab_size, 32);
        assert!(cfg.rope_scaling.is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parses_llama3_rope_scaling() {
        let path = write_test_gguf(
            "llama",
            &[
                ("llama.context_length", MetaValueOwned::U64(131072)),
                ("llama.embedding_length", MetaValueOwned::U32(4096)),
                ("llama.block_count", MetaValueOwned::U32(32)),
                ("llama.feed_forward_length", MetaValueOwned::U32(14336)),
                ("llama.attention.head_count", MetaValueOwned::U32(32)),
                ("llama.attention.head_count_kv", MetaValueOwned::U32(8)),
                (
                    "llama.rope.scaling.type",
                    MetaValueOwned::Str("llama3".into()),
                ),
                ("llama.rope.scaling.factor", MetaValueOwned::F32(8.0)),
                (
                    "llama.rope.scaling.original_context_length",
                    MetaValueOwned::U32(8192),
                ),
                (
                    "tokenizer.ggml.tokens",
                    MetaValueOwned::StringArray(
                        (0..16).map(|i| format!("t{i}")).collect::<Vec<_>>(),
                    ),
                ),
            ],
        );
        let cfg = LlamaBaseConfig::from_gguf_path(&path).expect("parse llama3 rope");
        match cfg.rope_scaling {
            Some(RopeScaling::Llama3 {
                factor,
                original_max_position_embeddings,
                ..
            }) => {
                assert!((factor - 8.0).abs() < 1e-6);
                assert_eq!(original_max_position_embeddings, 8192);
            }
            other => panic!("expected Llama3 scaling, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn family_preset_known_archs() {
        assert!(family_preset("mistral3").is_some());
        assert!(family_preset("phi3").is_some());
        assert!(family_preset("phi4").is_some());
        assert!(family_preset("granite").is_some());
        assert!(family_preset("command-r").is_some());
        assert!(family_preset("cohere2").is_some());
        assert!(family_preset("totally-fake").is_none());
    }

    #[test]
    fn rope_scaling_json_round_trip() {
        let v = RopeScaling::YaRN {
            factor: 4.0,
            original_max_position_embeddings: 8192,
            attention_factor: Some(1.0),
            beta_fast: None,
            beta_slow: None,
        };
        let j = serde_json::to_string(&v).unwrap();
        let back: RopeScaling = serde_json::from_str(&j).unwrap();
        assert_eq!(v, back);
    }
}
