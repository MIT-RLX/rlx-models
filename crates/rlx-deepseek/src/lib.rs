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

//! DeepSeek-V3 / V3.1 (MLA + fine-grained MoE), incl. Kimi-K2 — RLX runner.
//!
//! WIP: `config` parsing is implemented; the MLA attention + fine-grained MoE
//! graph, runner, and CLI are not landed yet.

pub mod config;
pub mod flow;
pub mod mla;
pub mod moe;

pub use config::DeepseekV3Config;
pub use flow::build_deepseek_text_flow;
pub use mla::{MlaDims, emit_mla_attention};
pub use moe::{DeepseekMoeDims, emit_deepseek_moe};
