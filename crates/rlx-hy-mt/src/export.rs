// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export an HY-MT (Hunyuan dense) checkpoint as an `.rlxp` package rlx
//! runs alone, through rlx-qwen3's decoder exporter: the config mapped as
//! the runner maps it ([`crate::config::qwen3_config_from_hf`]), the step
//! symbolic in its cache and batch, a prompt function, the tokenizer and
//! chat template. A `hunyuan-dense` GGUF goes to rlx-qwen3's GGUF exporter,
//! which reads Hunyuan's metadata as the runner does.

use std::path::Path;

use anyhow::Result;
pub use rlx_core::export::{ExportOptions, ExportReport};

/// Export the HY-MT checkpoint `path` (a Hugging Face directory or a
/// `hunyuan-dense` GGUF with a `tokenizer.json` beside it) as `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    if path.extension().is_some_and(|e| e == "gguf") {
        crate::config::validate_weights_kind(path)?;
        return rlx_qwen3::export::export_gguf_with(path, out, opts, None, &mut |_| Ok(()));
    }
    let cfg = crate::config::qwen3_config_from_hf(path)?;
    rlx_qwen3::export::export_with_config(path, cfg, "hunyuan_v1_dense", out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `hy-mt` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("hy-mt", "translation", export_rlxp)
            .claims(&["hunyuan_v1_dense", "hunyuan-dense"]),
    );
}
