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

//! Inkling multimodal MoE — [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling).
//!
//! RLX path today: HF config, Unsloth GGUF metadata sniff, chat helpers, and an
//! eager CPU text forward / greedy generate (synth + HF tiny fixture). Quant
//! GGUF dequant and compiled graphs are follow-ups — inference stays on RLX,
//! not llama.cpp.

pub mod chat;
pub mod cli;
pub mod config;
pub mod eager;
pub mod fixture;
pub mod gguf_layout;
pub mod gguf_probe;
pub mod probe;
pub mod runner;
pub mod shapes;
pub mod synth;
pub mod weights;

pub use config::{
    ARCHITECTURE, AnnouncedParams, AttnLayerType, FAMILY, GGUF_ARCH, HF_GGUF_REPO,
    HF_GGUF_REPO_SMALL, HF_MODEL_ID, HF_MODEL_ID_SMALL, InklingAudioConfig, InklingConfig,
    InklingTextConfig, InklingVariant, InklingVisionConfig, MODEL_TYPE, MlpLayerType,
};
pub use eager::{TextWeights, forward_logits, greedy_next};
pub use gguf_layout::{GGUF_ARCHES, gguf_to_eager_key};
pub use gguf_probe::{DEFAULT_QUANT, GgufProbeReport};
pub use probe::{ProbeReport, compare_shapes, read_local_header, validate_model_dir};
pub use runner::InklingRunner;
pub use shapes::expected_hf_shapes;
pub use synth::{synthetic_text_weights, tiny_cfg, tiny_mm_cfg};
