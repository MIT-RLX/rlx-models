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

//! Model registry with metadata for all supported text and image embedding models.

use super::pooling::Pooling;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Supported text embedding models.
///
/// Each variant maps to a specific HuggingFace model repository with
/// pre-trained safetensors weights. Use [`EmbeddingModel::get_info`] to
/// access metadata (dimension, pooling strategy, max sequence length).
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    // ── MiniLM / MPNet ──────────────────────────────────────────────────────
    /// sentence-transformers/all-MiniLM-L6-v2 — 384-dim, 6 layers, mean pooling.
    #[default]
    AllMiniLML6V2,
    /// sentence-transformers/all-MiniLM-L12-v2 — 384-dim, 12 layers, mean pooling.
    AllMiniLML12V2,
    /// sentence-transformers/all-mpnet-base-v2 — 768-dim, 12 layers, mean pooling.
    /// Note: mpnet architecture — uses different attention key naming.
    AllMpnetBaseV2,

    // ── BGE ─────────────────────────────────────────────────────────────────
    /// BAAI/bge-small-en-v1.5 — 384-dim, 12 layers, CLS pooling.
    BGESmallENV15,
    /// BAAI/bge-base-en-v1.5 — 768-dim, 12 layers, CLS pooling.
    BGEBaseENV15,
    /// BAAI/bge-large-en-v1.5 — 1024-dim, 24 layers, CLS pooling.
    BGELargeENV15,
    /// BAAI/bge-small-zh-v1.5 — 512-dim, 12 layers, CLS pooling.
    BGESmallZHV15,
    /// BAAI/bge-large-zh-v1.5 — 1024-dim, 24 layers, CLS pooling.
    BGELargeZHV15,

    // ── Multilingual E5 ─────────────────────────────────────────────────────
    /// intfloat/multilingual-e5-small — 384-dim, 12 layers, mean pooling.
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base — 768-dim, 12 layers, mean pooling.
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large — 1024-dim, 24 layers, mean pooling.
    MultilingualE5Large,

    // ── Paraphrase ──────────────────────────────────────────────────────────
    /// sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — 384-dim, 12 layers, mean pooling.
    ParaphraseMLMiniLML12V2,
    /// sentence-transformers/paraphrase-multilingual-mpnet-base-v2 — 768-dim, 12 layers, mean pooling.
    ParaphraseMLMpnetBaseV2,

    // ── Snowflake Arctic ────────────────────────────────────────────────────
    /// snowflake/snowflake-arctic-embed-xs — 384-dim, CLS pooling.
    SnowflakeArcticEmbedXS,
    /// snowflake/snowflake-arctic-embed-s — 384-dim, CLS pooling.
    SnowflakeArcticEmbedS,
    /// Snowflake/snowflake-arctic-embed-m — 768-dim, CLS pooling.
    SnowflakeArcticEmbedM,
    /// snowflake/snowflake-arctic-embed-l — 1024-dim, CLS pooling.
    SnowflakeArcticEmbedL,

    // ── MxBai ───────────────────────────────────────────────────────────────
    /// mixedbread-ai/mxbai-embed-large-v1 — 1024-dim, CLS pooling.
    MxbaiEmbedLargeV1,

    // ── Nomic ───────────────────────────────────────────────────────────────
    /// nomic-ai/nomic-embed-text-v1.5 — 768-dim, 12 layers, mean pooling, RoPE, SwiGLU.
    NomicEmbedTextV15,
}

/// Supported image embedding models.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImageEmbeddingModel {
    /// nomic-ai/nomic-embed-vision-v1.5 — 768-dim, 12 layers, CLS pooling, 224px.
    #[default]
    NomicEmbedVisionV15,
}

/// Metadata for an image embedding model.
#[derive(Debug, Clone)]
pub struct ImageModelInfo {
    pub model: ImageEmbeddingModel,
    pub dim: usize,
    pub description: &'static str,
    pub hf_repo: &'static str,
    pub model_file: &'static str,
    pub img_size: usize,
}

static IMAGE_MODEL_MAP: OnceLock<HashMap<ImageEmbeddingModel, ImageModelInfo>> = OnceLock::new();

fn init_image_models_map() -> HashMap<ImageEmbeddingModel, ImageModelInfo> {
    vec![ImageModelInfo {
        model: ImageEmbeddingModel::NomicEmbedVisionV15,
        dim: 768,
        description: "Nomic embed vision v1.5, 12 layers, 224px",
        hf_repo: "nomic-ai/nomic-embed-vision-v1.5",
        model_file: "model.safetensors",
        img_size: 224,
    }]
    .into_iter()
    .map(|info| (info.model.clone(), info))
    .collect()
}

impl ImageEmbeddingModel {
    pub fn get_info(&self) -> Option<&'static ImageModelInfo> {
        IMAGE_MODEL_MAP.get_or_init(init_image_models_map).get(self)
    }
}

/// Model architecture type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    /// Standard BERT architecture (absolute position embeddings, GELU FFN).
    Bert,
    /// NomicBERT architecture (RoPE, fused QKV, SwiGLU FFN).
    NomicBert,
}

/// Metadata for an embedding model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model: EmbeddingModel,
    /// Output embedding dimension.
    pub dim: usize,
    /// Human-readable description.
    pub description: &'static str,
    /// HuggingFace repository ID.
    pub hf_repo: &'static str,
    /// Weight file name within the repository.
    pub model_file: &'static str,
    /// Pooling strategy for this model.
    pub pooling: Pooling,
    /// Maximum input sequence length.
    pub max_length: usize,
    /// Model architecture.
    pub arch: ModelArch,
}

static MODEL_MAP: OnceLock<HashMap<EmbeddingModel, ModelInfo>> = OnceLock::new();

fn init_models_map() -> HashMap<EmbeddingModel, ModelInfo> {
    vec![
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            dim: 384,
            description: "MiniLM-L6-v2, 6 layers, fast and lightweight",
            hf_repo: "sentence-transformers/all-MiniLM-L6-v2",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 256,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML12V2,
            dim: 384,
            description: "MiniLM-L12-v2, 12 layers, higher quality",
            hf_repo: "sentence-transformers/all-MiniLM-L12-v2",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 256,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15,
            dim: 384,
            description: "BGE small English v1.5, compact and fast",
            hf_repo: "BAAI/bge-small-en-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15,
            dim: 768,
            description: "BGE base English v1.5, balanced quality and speed",
            hf_repo: "BAAI/bge-base-en-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15,
            dim: 1024,
            description: "BGE large English v1.5, highest quality",
            hf_repo: "BAAI/bge-large-en-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallZHV15,
            dim: 512,
            description: "BGE small Chinese v1.5, CLS pooling",
            hf_repo: "BAAI/bge-small-zh-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2,
            dim: 384,
            description: "Paraphrase multilingual MiniLM L12 v2, mean pooling",
            hf_repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 128,
            arch: ModelArch::Bert,
        },
        // mpnet
        ModelInfo {
            model: EmbeddingModel::AllMpnetBaseV2,
            dim: 768,
            description: "mpnet-base-v2, strong general-purpose embeddings",
            hf_repo: "sentence-transformers/all-mpnet-base-v2",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 384,
            arch: ModelArch::Bert,
        },
        // BGE Chinese large
        ModelInfo {
            model: EmbeddingModel::BGELargeZHV15,
            dim: 1024,
            description: "BGE large Chinese v1.5",
            hf_repo: "BAAI/bge-large-zh-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        // Multilingual E5
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Small,
            dim: 384,
            description: "Multilingual E5 small, 100+ languages",
            hf_repo: "intfloat/multilingual-e5-small",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Base,
            dim: 768,
            description: "Multilingual E5 base, 100+ languages",
            hf_repo: "intfloat/multilingual-e5-base",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Large,
            dim: 1024,
            description: "Multilingual E5 large, 100+ languages",
            hf_repo: "intfloat/multilingual-e5-large",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        // Paraphrase mpnet
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMpnetBaseV2,
            dim: 768,
            description: "Paraphrase multilingual mpnet base v2",
            hf_repo: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 384,
            arch: ModelArch::Bert,
        },
        // Snowflake Arctic
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedXS,
            dim: 384,
            description: "Snowflake Arctic Embed XS",
            hf_repo: "snowflake/snowflake-arctic-embed-xs",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedS,
            dim: 384,
            description: "Snowflake Arctic Embed S",
            hf_repo: "snowflake/snowflake-arctic-embed-s",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedM,
            dim: 768,
            description: "Snowflake Arctic Embed M",
            hf_repo: "Snowflake/snowflake-arctic-embed-m",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedL,
            dim: 1024,
            description: "Snowflake Arctic Embed L",
            hf_repo: "snowflake/snowflake-arctic-embed-l",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        // MxBai
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1,
            dim: 1024,
            description: "MxBai embed large v1",
            hf_repo: "mixedbread-ai/mxbai-embed-large-v1",
            model_file: "model.safetensors",
            pooling: Pooling::Cls,
            max_length: 512,
            arch: ModelArch::Bert,
        },
        // Nomic
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15,
            dim: 768,
            description: "Nomic embed text v1.5, RoPE, SwiGLU, 8192 context",
            hf_repo: "nomic-ai/nomic-embed-text-v1.5",
            model_file: "model.safetensors",
            pooling: Pooling::Mean,
            max_length: 8192,
            arch: ModelArch::NomicBert,
        },
    ]
    .into_iter()
    .map(|info| (info.model.clone(), info))
    .collect()
}

/// Get the global model registry.
pub fn models_map() -> &'static HashMap<EmbeddingModel, ModelInfo> {
    MODEL_MAP.get_or_init(init_models_map)
}

impl EmbeddingModel {
    /// Look up metadata for this model.
    pub fn get_info(&self) -> Option<&'static ModelInfo> {
        models_map().get(self)
    }

    /// List all supported models.
    pub fn list_supported() -> Vec<&'static ModelInfo> {
        models_map().values().collect()
    }
}
