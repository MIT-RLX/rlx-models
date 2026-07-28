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

//! Jamba (Mamba-1 + attention + MoE hybrid) for RLX.
//!
//! Most layers are Mamba-1 SSM mixers; every `attn_layer_period` layer is
//! attention, and every `expert_layer_period` layer is MoE. This crate builds
//! on `rlx-ssm`'s `MambaScanStage` (`Op::SelectiveScan`) for the SSM core,
//! `rlx-deepseek` for MoE, and the standard attention scaffold.

pub mod attention;
pub mod flow;
pub mod mamba;

pub use attention::{JambaAttnDims, emit_jamba_attention};
pub use flow::{JambaFlowDims, build_jamba_text_flow};
pub use mamba::{MambaDims, emit_mamba1_block};
