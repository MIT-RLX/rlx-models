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

//! Compiled e2e RLX IR scaffold for Laguna packed MoE.
//!
//! **Status:** not implemented. Production generate uses
//! [`crate::packed_forward`] (KV-cached host / Metal `DequantMatMul`).
//!
//! Reference patterns when implementing:
//! - Dense packed prefill: `rlx-qwen3` `PackedForward` +
//!   `rlx_core::flow_bridge::packed_gguf_*`
//! - MoE FFN IR: `rlx-qwen35` `build_moe_ffn` (TopK + shared expert)
//! - Hard parts for Laguna: dual RoPE (YaRN vs SWA), softplus attn gate,
//!   sigmoid TopK over 256 experts, 3-D expert packs `[E,n,k]`

use anyhow::{Result, bail};

/// Human-readable build status for CLI / docs.
pub fn build_status() -> &'static str {
    "scaffolded — packed KV generate only; no e2e IR graph yet"
}

/// Placeholder for a future packed prefill graph compile.
pub fn build_prefill_graph() -> Result<()> {
    bail!(
        "rlx-laguna: compiled e2e IR not implemented ({})",
        build_status()
    )
}
