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

//! Export a Hugging Face Qwen3 checkpoint — or Llama (3.x's scaled RoPE
//! included), Mistral or Qwen2, which share its graphs — as an `.rlxp`
//! package that rlx runs with no rlx-models code, through the one exporter
//! ([`rlx_core::export::decoder`]): the decode step, a `generate` entry
//! (`chat`, `generate_text`, `generate`), the RoPE tables, the tokenizer,
//! the chat template, the model card, and its weights referenced from the
//! checkpoint rather than copied.

use std::path::Path;

use anyhow::{Result, bail};
use rlx_core::export::decoder::{Built, Decoder};
use rlx_core::export::rope::Rope;
use rlx_core::weight_loader::WeightLoader;
use serde_json::{Value, json};

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::builder::{build_qwen3_decode_graph_sized_ext, build_qwen3_graph_sized_last_logits};
use crate::config::Qwen3Config;
use crate::generator::compute_rope_slice;

/// The `rope_scaling` types the package's tables are computed for.
const ROPE_TYPES: [&str; 7] = [
    "default", "llama3", "linear", "yarn", "longrope", "su", "phi3",
];

/// Qwen3 and the Llama family (Llama, Mistral, Qwen2) through rlx-qwen3's
/// graphs.
struct Qwen3Decoder {
    cfg: Qwen3Config,
    family: String,
    /// Scaled frequencies (`rope_scaling`), when the config scales them;
    /// else the step's rows are the runner's own (`compute_rope_slice`).
    scaled: Option<Rope>,
    /// The sliding window, where every layer has one (Mistral's): the
    /// package's context stops at it, so attention needs no window.
    window: Option<usize>,
    /// A GGUF's `tokenizer.json` from its metadata (a crate that builds it).
    tokenizer: Option<GgufTokenizer>,
    /// The graphs a prompt with media runs ([`Decoder::media`]), for a model
    /// a crate of its own adds them to (a speaker's row in a TTS prompt).
    media: Option<MediaBuilder>,
}

/// Builds a GGUF's `tokenizer.json` from its metadata.
pub type GgufTokenizer = fn(&Path) -> Result<Vec<u8>>;

/// [`Decoder::media`] for a model built on this crate's graphs: from a
/// fresh copy of the checkpoint each call, the RoPE rows the package runs,
/// each graph's params to `take`.
pub type MediaBuilder = Box<
    dyn Fn(
        &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        usize,
        &mut dyn FnMut(std::collections::HashMap<String, Vec<f32>>) -> Result<()>,
    ) -> Result<Option<rlx_core::export::decoder::MediaParts>>,
>;

impl Decoder for Qwen3Decoder {
    fn family(&self) -> String {
        self.family.clone()
    }
    fn exported_by(&self) -> &str {
        "rlx-qwen3"
    }
    fn max_positions(&self) -> usize {
        match self.window {
            Some(w) => self.cfg.max_position_embeddings.min(w),
            None => self.cfg.max_position_embeddings,
        }
    }
    fn layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn rope_inputs(&self) -> Vec<(String, String, usize)> {
        vec![("rope_cos".into(), "rope_sin".into(), self.cfg.head_dim / 2)]
    }
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        match &self.scaled {
            Some(r) => vec![r.row(pos)],
            None => vec![compute_rope_slice(&self.cfg, pos)],
        }
    }
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built> {
        build_qwen3_decode_graph_sized_ext(&self.cfg, weights, batch, upper, true)
    }
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built> {
        let (g, mut params) =
            build_qwen3_graph_sized_last_logits(&self.cfg, weights, batch, seq, true)?;
        // The prompt graph's tables are params: scaled, they are the
        // step's rows, which the exporter then shares.
        if let Some(r) = &self.scaled {
            let rows = params.get("rope.cos").map_or(0, Vec::len) / r.half().max(1);
            let (cos, sin) = r.table(rows);
            params.insert("rope.cos".into(), cos);
            params.insert("rope.sin".into(), sin);
        }
        Ok((g, params))
    }
    fn check(&self, weights: &mut dyn WeightLoader, past: usize, tokens: usize) -> Result<Built> {
        let o = crate::flow::Qwen3DecodeOpts {
            batch: 1,
            past_seq: past,
            use_custom_mask: true,
            ..Default::default()
        };
        rlx_core::flow_util::graph_from_built(crate::flow::build_qwen3_check_built(
            &self.cfg, weights, &o, tokens,
        )?)
    }
    fn prompt_rope_params(&self) -> Vec<(String, String)> {
        vec![("rope.cos".into(), "rope.sin".into())]
    }
    fn media(
        &self,
        weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        rows: usize,
        take: &mut dyn FnMut(std::collections::HashMap<String, Vec<f32>>) -> Result<()>,
    ) -> Result<Option<rlx_core::export::decoder::MediaParts>> {
        match &self.media {
            Some(build) => build(weights, rows, take),
            None => Ok(None),
        }
    }
    fn gguf_tokenizer(&self, gguf: &Path) -> Result<Vec<u8>> {
        match self.tokenizer {
            Some(build) => build(gguf),
            None => bail!(
                "{}: no tokenizer.json beside it, and no builder from its metadata",
                gguf.display()
            ),
        }
    }
    fn key_rope(&self) -> Option<Value> {
        // A batch rotating keys back recomputes them from these tables.
        Some(
            json!({"cos": "rope_cos.table", "sin": "rope_sin.table", "style": "neox",
                    "head_dim": self.cfg.head_dim, "keys": "past_k_"}),
        )
    }
}

/// Export the checkpoint in `dir` (`config.json`, `model.safetensors`,
/// `tokenizer.json`, `tokenizer_config.json`, optional
/// `generation_config.json` and `README.md`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    if dir.extension().is_some_and(|e| e == "gguf") {
        return export_gguf_with(dir, out, opts, None, &mut |_| Ok(()));
    }
    export_rlxp_with(dir, out, opts, &mut |_| Ok(()))
}

/// Export the Qwen2 / Qwen3 GGUF `gguf` as the package `out`: its config
/// from its metadata, its tensors read from it; its tokenizer a
/// `tokenizer.json` beside it, else `tokenizer`'s from its metadata.
pub fn export_gguf_with(
    gguf: &Path,
    out: &Path,
    opts: &ExportOptions,
    tokenizer: Option<GgufTokenizer>,
    extend: &mut dyn FnMut(&mut rlx_core::export::Package) -> Result<()>,
) -> Result<ExportReport> {
    export_gguf_media(gguf, out, opts, tokenizer, None, extend)
}

/// [`export_gguf_with`] with the graphs a prompt with media runs: `media`
/// builds them ([`MediaBuilder`]).
pub fn export_gguf_media(
    gguf: &Path,
    out: &Path,
    opts: &ExportOptions,
    tokenizer: Option<GgufTokenizer>,
    media: Option<MediaBuilder>,
    extend: &mut dyn FnMut(&mut rlx_core::export::Package) -> Result<()>,
) -> Result<ExportReport> {
    let cfg = crate::qwen3_config_from_gguf_path(gguf)?;
    if cfg.num_experts > 0 {
        bail!(
            "{}: mixture-of-experts GGUFs are not exported",
            gguf.display()
        );
    }
    // Qwen2 has no QK-norm; Qwen3 does.
    let family = if cfg.qk_norm { "qwen3" } else { "qwen2" }.to_string();
    let model = Qwen3Decoder {
        cfg,
        family,
        scaled: None,
        window: None,
        tokenizer,
        media,
    };
    rlx_core::export::decoder::export_decoder_with(&model, gguf, out, opts, extend)
}

/// Export the checkpoint in `dir` built as `cfg` says, not as its
/// `config.json` reads: a family whose config another crate maps onto
/// Qwen3's (Hunyuan dense), under the model block's `family`. No RoPE
/// scaling and no sliding window.
pub fn export_with_config(
    dir: &Path,
    cfg: Qwen3Config,
    family: &str,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    if cfg.num_experts > 0 {
        bail!("{family}: mixture-of-experts checkpoints are not exported");
    }
    let model = Qwen3Decoder {
        cfg,
        family: family.to_string(),
        scaled: None,
        window: None,
        tokenizer: None,
        media: None,
    };
    rlx_core::export::decoder::export_decoder_with(&model, dir, out, opts, &mut |_| Ok(()))
}

/// [`export_rlxp`] for a model this language model is part of (a TTS whose
/// LM speaks codes): `extend` adds the rest to the package
/// ([`rlx_core::export::decoder::export_decoder_with`]).
pub fn export_rlxp_with(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    extend: &mut dyn FnMut(&mut rlx_core::export::Package) -> Result<()>,
) -> Result<ExportReport> {
    let cfg = Qwen3Config::from_file(&dir.join("config.json"))?;
    let mut raw_cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    rlx_core::config::legacy_rope_keys(&mut raw_cfg);
    // Llama, Mistral and Qwen2 share the graph (no QK-norm).
    let family = raw_cfg
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("qwen3")
        .to_string();
    if !matches!(family.as_str(), "qwen3" | "llama" | "mistral" | "qwen2") {
        bail!("model_type {family:?}: not one this exporter builds");
    }
    if cfg.num_experts > 0 {
        bail!("{family}: mixture-of-experts checkpoints are not exported");
    }
    let scaling = raw_cfg.get("rope_scaling").filter(|v| !v.is_null());
    if let Some(s) = scaling {
        let kind = s
            .get("rope_type")
            .or_else(|| s.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        if !ROPE_TYPES.contains(&kind) {
            bail!("rope_scaling {s}: a type this exporter does not compute");
        }
    }
    let rope = Rope::from_config(&raw_cfg, cfg.head_dim, cfg.rope_theta);
    // Mistral windows every layer whenever it names a window; Qwen only
    // with `use_sliding_window`, from `max_window_layers` on.
    let window = match family.as_str() {
        "mistral" => cfg.sliding_window,
        _ if cfg.use_sliding_window && cfg.max_window_layers == 0 => cfg.sliding_window,
        _ if cfg.use_sliding_window && cfg.max_window_layers < cfg.num_hidden_layers => {
            bail!("{family}: sliding window on some layers only is not exported")
        }
        _ => None,
    };
    let model = Qwen3Decoder {
        cfg,
        family,
        scaled: rope.is_scaled().then_some(rope),
        window,
        tokenizer: None,
        media: None,
    };
    rlx_core::export::decoder::export_decoder_with(&model, dir, out, opts, extend)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `qwen3` and `llama32` / `mistral` runners' names, for the
/// model types it builds.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("qwen3", "text-generation", export_rlxp).claims(&["qwen3", "qwen2"]),
    );
    register_exporter(Exporter::new("llama32", "text-generation", export_rlxp).claims(&["llama"]));
    register_exporter(
        Exporter::new("mistral", "text-generation", export_rlxp).claims(&["mistral"]),
    );
}
