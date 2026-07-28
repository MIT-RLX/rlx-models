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

//! Microsoft **Fara1.5** computer-use agent for RLX.
//!
//! Fara1.5-4B / 9B are supervised fine-tunes of Qwen3.5 multimodal
//! ([`microsoft/Fara1.5-4B`](https://huggingface.co/microsoft/Fara1.5-4B),
//! [`microsoft/Fara1.5-9B`](https://huggingface.co/microsoft/Fara1.5-9B)).
//! Inference reuses [`rlx_qwen35::Qwen35Runner`]; this crate adds size
//! presets, the Fara system prompt, `<tool_call>` parsing, and a CLI.

pub mod cli;
pub mod config;
pub mod download;
pub mod prompt;
pub mod runner;
pub mod tools;

pub use config::{
    EOS_TOKEN_ID, FAMILY, FaraSize, HF_MODEL_ID_4B, HF_MODEL_ID_9B, IMAGE_TOKEN_ID, TRAIN_SCREEN_H,
    TRAIN_SCREEN_W, VIDEO_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID, default_cache_root,
    default_model_dir, fara_qwen35_config, fara_vision_config, is_model_dir,
};
pub use download::{download_fara, read_snapshot_pointer, resolve_or_download};
pub use prompt::{fara_system_prompt, format_fara_multimodal_prompt};
pub use runner::{FaraRunner, FaraRunnerBuilder, FaraStep};
pub use tools::{
    ToolCall, extract_tool_call_bodies, format_tool_response, parse_tool_call_body,
    parse_tool_calls, text_before_tool_calls,
};
