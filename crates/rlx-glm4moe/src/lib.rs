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

//! GLM-4.5 / GLM-4.6 (`glm4_moe`) for RLX: standard pre-norm decoder with
//! partial-RoPE GQA attention (optional qk-norm) + DeepSeek-style fine-grained
//! MoE (reusing [`rlx_deepseek`]'s router + experts).

pub mod attention;
pub mod config;
pub mod flow;

pub use attention::{GlmAttnDims, emit_glm_attention};
pub use config::Glm4MoeConfig;
pub use flow::build_glm_text_flow;
