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

//! Read HuggingFace-shaped config fields from GGUF metadata (`{arch}.*` keys).

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};

use crate::config::{BertConfig, NomicBertConfig};
use crate::gguf_support::gguf_architecture_str;

fn arch_prefix(raw: &GgufFile) -> &str {
    gguf_architecture_str(raw).unwrap_or("bert")
}

fn get_meta<'a>(raw: &'a GgufFile, _arch: &str, key: &str) -> Option<&'a MetaValue> {
    raw.metadata.get(key)
}

fn meta_u32(raw: &GgufFile, arch: &str, key: &str) -> Result<u32> {
    get_meta(raw, arch, key)
        .and_then(MetaValue::as_u32)
        .with_context(|| format!("missing or invalid GGUF metadata key: {key}"))
}

fn meta_u32_or(raw: &GgufFile, arch: &str, key: &str, default: u32) -> u32 {
    get_meta(raw, arch, key)
        .and_then(MetaValue::as_u32)
        .unwrap_or(default)
}

fn meta_f64(raw: &GgufFile, arch: &str, key: &str, default: f64) -> f64 {
    get_meta(raw, arch, key).map_or(default, |v| match v {
        MetaValue::F32(x) => *x as f64,
        MetaValue::F64(x) => *x,
        _ => default,
    })
}

/// GGUF `general.architecture` values for embedding models (validate in `rlx-embed`, not the loader).
pub const EMBED_GGUF_ARCHES: &[&str] = &["bert", "modern-bert", "nomic-bert", "nomic-bert-moe"];

/// GGUF architectures for FLUX denoiser checkpoints (validate in `rlx-flux2`, not the loader).
pub const FLUX_GGUF_ARCHES: &[&str] = &["flux"];

/// DINOv2 ViT (e.g. dinov2.cpp / community converters); F32 drain via `PrefixStripGgufResolver`.
pub const DINOV2_GGUF_ARCHES: &[&str] = &["dinov2"];

/// SAM v1 ViT-H and MobileSAM GGUF (`sam`, `mobile-sam`).
pub const SAM_GGUF_ARCHES: &[&str] = &["sam", "mobile-sam"];

/// SAM 2 Hiera checkpoints converted to GGUF (`sam2` tag).
pub const SAM2_GGUF_ARCHES: &[&str] = &["sam2"];

/// SAM 3 (e.g. rob-laz/sam3-gguf, sam3.cpp).
pub const SAM3_GGUF_ARCHES: &[&str] = &["sam3"];

/// V-JEPA 2 (no common Hub GGUF yet; validate when present).
pub const VJEPA2_GGUF_ARCHES: &[&str] = &["vjepa2", "vjepa"];

/// Wav2Vec2-BERT / classic Wav2Vec2 GGUF (`w2v-bert` converters; ASR repos often use `wav2vec2`).
pub const W2V_BERT_GGUF_ARCHES: &[&str] = &["w2v-bert", "wav2vec2", "wav2vec"];

pub fn is_flux_gguf_arch(arch: &str) -> bool {
    FLUX_GGUF_ARCHES.contains(&arch)
}

pub fn is_embed_gguf_arch(arch: &str) -> bool {
    EMBED_GGUF_ARCHES.contains(&arch)
}

pub fn is_dinov2_gguf_arch(arch: &str) -> bool {
    DINOV2_GGUF_ARCHES.contains(&arch)
}

pub fn is_sam_gguf_arch(arch: &str) -> bool {
    SAM_GGUF_ARCHES.contains(&arch)
}

pub fn is_sam2_gguf_arch(arch: &str) -> bool {
    SAM2_GGUF_ARCHES.contains(&arch)
}

pub fn is_sam3_gguf_arch(arch: &str) -> bool {
    SAM3_GGUF_ARCHES.contains(&arch)
}

pub fn is_vjepa2_gguf_arch(arch: &str) -> bool {
    VJEPA2_GGUF_ARCHES.contains(&arch)
}

pub fn is_w2v_bert_gguf_arch(arch: &str) -> bool {
    W2V_BERT_GGUF_ARCHES.contains(&arch)
}

/// BERT vs NomicBERT discriminator from GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedGgufKind {
    Bert,
    NomicBert,
}

/// Suggested runner / crate for a GGUF architecture tag (for CLI and errors).
pub fn gguf_runner_hint(arch: &str) -> &'static str {
    if let Some(hint) = crate::model_registry::hint_for_gguf_arch(arch) {
        return hint;
    }
    "unknown — register via `rlx_core::model_registry::register_gguf_model`"
}

/// Estimated RAM if every tensor is dequantized to F32 vs kept packed on disk.
#[derive(Debug, Clone, Copy)]
pub struct GgufMemoryFootprint {
    pub f32_bytes: u64,
    pub packed_file_bytes: u64,
}

pub fn gguf_memory_footprint(raw: &GgufFile) -> GgufMemoryFootprint {
    let mut f32_bytes = 0u64;
    let mut packed_file_bytes = 0u64;
    for t in raw.tensors.values() {
        let n = t.n_elements() as u64;
        f32_bytes += n * 4;
        // Canonical on-disk block size for the dtype (correct even on a
        // header-only read where the data segment isn't loaded — e.g. the
        // 1-bit Q1_0 Bonsai-27B, which is ~1.125 bpw, not F32-sized).
        packed_file_bytes += rlx_gguf::bytes_for_public(t.dtype, t.n_elements())
            .map(|b| b as u64)
            .unwrap_or(n * 4);
    }
    GgufMemoryFootprint {
        f32_bytes,
        packed_file_bytes,
    }
}

/// Read `{arch}.{suffix}` as u32 when present.
pub fn gguf_meta_u32(raw: &GgufFile, arch: &str, suffix: &str) -> Option<u32> {
    let key = format!("{arch}.{suffix}");
    raw.metadata.get(&key).and_then(MetaValue::as_u32)
}

pub fn embed_gguf_kind(raw: &GgufFile) -> Result<EmbedGgufKind> {
    let arch = arch_prefix(raw);
    if !is_embed_gguf_arch(arch) {
        bail!(
            "GGUF architecture {arch:?} is not supported for embeddings; \
             expected one of: {}",
            EMBED_GGUF_ARCHES.join(", ")
        );
    }
    if matches!(arch, "nomic-bert" | "nomic-bert-moe") {
        Ok(EmbedGgufKind::NomicBert)
    } else {
        Ok(EmbedGgufKind::Bert)
    }
}

impl BertConfig {
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = arch_prefix(raw);
        if !matches!(arch, "bert" | "modern-bert") {
            bail!("BertConfig::from_gguf expected bert/modern-bert, got {arch}");
        }
        let hidden_size = meta_u32(raw, arch, &format!("{arch}.embedding_length"))? as usize;
        let num_attention_heads =
            meta_u32(raw, arch, &format!("{arch}.attention.head_count"))? as usize;
        Ok(Self {
            vocab_size: meta_u32(raw, arch, &format!("{arch}.vocab_size"))? as usize,
            hidden_size,
            num_hidden_layers: meta_u32(raw, arch, &format!("{arch}.block_count"))? as usize,
            num_attention_heads,
            intermediate_size: meta_u32(raw, arch, &format!("{arch}.feed_forward_length"))?
                as usize,
            max_position_embeddings: meta_u32_or(raw, arch, &format!("{arch}.context_length"), 512)
                as usize,
            type_vocab_size: meta_u32_or(raw, arch, "tokenizer.ggml.token_type_count", 2) as usize,
            layer_norm_eps: meta_f64(
                raw,
                arch,
                &format!("{arch}.attention.layer_norm_epsilon"),
                1e-12,
            ),
            hidden_act: "gelu".into(),
        })
    }
}

impl NomicBertConfig {
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = arch_prefix(raw);
        if !matches!(arch, "nomic-bert" | "nomic-bert-moe") {
            bail!("NomicBertConfig::from_gguf expected nomic-bert, got {arch}");
        }
        let hidden_size = meta_u32(raw, arch, &format!("{arch}.embedding_length"))? as usize;
        let num_attention_heads =
            meta_u32(raw, arch, &format!("{arch}.attention.head_count"))? as usize;
        let head_dim = meta_u32_or(
            raw,
            arch,
            &format!("{arch}.attention.key_length"),
            (hidden_size / num_attention_heads.max(1)) as u32,
        ) as usize;
        Ok(Self {
            vocab_size: meta_u32(raw, arch, &format!("{arch}.vocab_size"))? as usize,
            hidden_size,
            num_hidden_layers: meta_u32(raw, arch, &format!("{arch}.block_count"))? as usize,
            num_attention_heads,
            intermediate_size: meta_u32(raw, arch, &format!("{arch}.feed_forward_length"))?
                as usize,
            max_position_embeddings: meta_u32_or(raw, arch, &format!("{arch}.context_length"), 8192)
                as usize,
            type_vocab_size: meta_u32_or(raw, arch, "tokenizer.ggml.token_type_count", 2) as usize,
            layer_norm_eps: meta_f64(
                raw,
                arch,
                &format!("{arch}.attention.layer_norm_epsilon"),
                1e-12,
            ),
            head_dim,
            rotary_emb_base: meta_f64(raw, arch, &format!("{arch}.rope.freq_base"), 1000.0),
        })
    }
}
