// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Nanbeige checkpoint as an `.rlxp` package rlx runs alone:
//! rlx-llama32's full-sequence graph with the layers unrolled `num_loops`
//! times over shared weights, run over the whole context each token
//! ([`rlx_llama32::export::export_forward_with`]). A Nanbeige GGUF is a
//! `llama` GGUF with `llama.num_loops`, which rlx-llama32's exporter takes.

use std::path::Path;

use anyhow::Result;
pub use rlx_core::export::{ExportOptions, ExportReport};

/// Export the Nanbeige checkpoint in `dir` (`config.json`, safetensors,
/// `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    if dir.extension().is_some_and(|e| e == "gguf") {
        return rlx_llama32::export::export_rlxp(dir, out, opts);
    }
    crate::config::validate_hf_config(dir)?;
    let cfg = crate::config::llama_config_from_hf(dir)?;
    rlx_llama32::export::export_forward_with(dir, cfg, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// for `model_type = nanbeige`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("nanbeige", "text-generation", export_rlxp).claims(&["nanbeige"]),
    );
}
