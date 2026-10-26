// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Llama GGUF as an `.rlxp` package that rlx runs with no
//! rlx-models code, through the one exporter
//! ([`rlx_core::export::decoder`]). A GGUF's Q/K rows are in llama.cpp's
//! interleaved order, so its graphs are this crate's (GPT-J RoPE, read from
//! the GGUF's own `rope_freqs`), not rlx-qwen3's; a Hugging Face checkpoint
//! still goes to [`rlx_qwen3::export`].

use std::path::Path;

use anyhow::{Result, bail};
use rlx_core::export::decoder::{Built, Decoder, Forward, export_forward};
use rlx_core::weight_loader::WeightLoader;
use serde_json::{Value, json};

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::builder::{build_llama32_graph_sized_last_logits, build_llama32_graph_sized_packed};
use crate::config::{DenseArch, Llama32Config, llama32_cfg_from_gguf};
use crate::flow::Llama32Flow;
use crate::rope::{build_rope_tables, resolve_inv_freq, rope_slice};

/// A Llama GGUF through rlx-llama32's graphs.
struct GgufLlamaDecoder {
    cfg: Llama32Config,
    /// The step's frequencies: the config's, scaled by the GGUF's
    /// `rope_freqs` when it has them — as the prompt graph's tables are.
    inv_freq: Vec<f64>,
}

impl Decoder for GgufLlamaDecoder {
    fn family(&self) -> String {
        "llama".into()
    }
    fn exported_by(&self) -> &str {
        "rlx-llama32"
    }
    fn max_positions(&self) -> usize {
        self.cfg.max_position_embeddings
    }
    fn layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn rope_inputs(&self) -> Vec<(String, String, usize)> {
        vec![("rope_cos".into(), "rope_sin".into(), self.inv_freq.len())]
    }
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![rope_slice(&self.inv_freq, pos)]
    }
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built> {
        rlx_core::flow_util::graph_from_built(
            Llama32Flow::for_decode(&self.cfg, batch, upper)
                .custom_mask()
                .rope_inputs()
                .build(weights)?,
        )
    }
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built> {
        build_llama32_graph_sized_last_logits(&self.cfg, weights, batch, seq, true)
    }
    fn prompt_rope_params(&self) -> Vec<(String, String)> {
        vec![("rope.cos".into(), "rope.sin".into())]
    }
    fn key_rope(&self) -> Option<Value> {
        Some(
            json!({"cos": "rope_cos.table", "sin": "rope_sin.table", "style": "gptj",
                    "head_dim": self.cfg.head_dim(), "keys": "past_k_"}),
        )
    }
    fn gguf_tokenizer(&self, gguf: &Path) -> Result<Vec<u8>> {
        rlx_qwen35::gguf_tokenizer_json(gguf)
    }
}

/// A loader that never hands out packed bytes, so the packed builder binds
/// every matmul weight as f32: the GGUF's tensors as rlx dequantizes them,
/// which the package then references from the file.
struct Dequantized<'a>(&'a mut dyn WeightLoader);

impl WeightLoader for Dequantized<'_> {
    fn format_id(&self) -> &'static str {
        self.0.format_id()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take(key)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take_transposed(key)
    }
    fn gguf_file(&self) -> Option<&rlx_gguf::GgufFile> {
        self.0.gguf_file()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.0.stored_dtype(key)
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.0.remaining_keys()
    }
    fn arch_hint(&self) -> Option<&str> {
        self.0.arch_hint()
    }
}

/// A Llama-shaped checkpoint whose block is not Llama's own (OLMo-2, Nemotron,
/// GLM, Granite's multipliers, Exaone, Phi-3's partial RoPE, Muse
/// Glimmer; Nanbeige's looped layers): the runner's full-sequence graph,
/// every arch's deltas in it, run over the whole context each token.
struct ArchForward {
    cfg: Llama32Config,
    inv_freq: Vec<f64>,
}

impl Forward for ArchForward {
    fn family(&self) -> String {
        self.cfg.gguf_arch.clone().unwrap_or_else(|| "llama".into())
    }
    fn exported_by(&self) -> &str {
        "rlx-llama32"
    }
    fn max_positions(&self) -> usize {
        self.cfg.max_position_embeddings
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn forward(&self, weights: &mut dyn WeightLoader, seq: usize) -> Result<Built> {
        let mut packed = std::collections::HashMap::new();
        let mut embed_host = None;
        let built = build_llama32_graph_sized_packed(
            &self.cfg,
            &mut Dequantized(weights),
            1,
            seq,
            true,
            false,
            false,
            &mut packed,
            &mut embed_host,
        )?;
        if !packed.is_empty() || embed_host.is_some() {
            bail!("{}: the graph kept packed weights", self.family());
        }
        Ok(built)
    }
    fn tables(&self, rows: usize) -> Vec<(String, Vec<f32>, usize)> {
        let (cos, sin) = build_rope_tables(&self.inv_freq, rows);
        let half = self.inv_freq.len();
        vec![
            ("rope.cos".into(), cos, half),
            ("rope.sin".into(), sin, half),
        ]
    }
}

/// The GGUF architectures this crate's runner builds (Cohere's is coded but
/// not routed: the runner refuses it, so the exporter does too).
const GGUF_ARCHES: &[&str] = &[
    "llama",
    "mistral3",
    "mistral4",
    "granite",
    "exaone",
    "olmo",
    "olmo2",
    "nemotron",
    "glm4",
    "chatglm",
    "phi3",
    "phi4",
    "muse-glimmer",
];

/// Export the Llama GGUF `gguf` (its tokenizer from its metadata, or a
/// `tokenizer.json` beside it) as the package `out`.
pub fn export_gguf(gguf: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_gguf_with(gguf, out, opts, &mut |_| Ok(()))
}

/// [`export_gguf`] for a model this language model is part of (a TTS whose
/// LM speaks codes): `extend` adds the rest to the package
/// ([`rlx_core::export::decoder::export_decoder_with`]).
pub fn export_gguf_with(
    gguf: &Path,
    out: &Path,
    opts: &ExportOptions,
    extend: &mut dyn FnMut(&mut rlx_core::export::Package) -> Result<()>,
) -> Result<ExportReport> {
    let raw = rlx_gguf::GgufFile::from_path(gguf)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", gguf.display()))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(rlx_gguf::MetaValue::as_str)
        .unwrap_or("");
    if !GGUF_ARCHES.contains(&arch) {
        bail!(
            "{}: GGUF architecture {arch:?}, not one rlx-llama32 builds",
            gguf.display()
        );
    }
    let arch = arch.to_string();
    let cfg = llama32_cfg_from_gguf(&raw)?;
    drop(raw);
    let mut loader = rlx_core::export::Checkpoint::open(gguf)?.loader()?;
    let factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
    drop(loader);
    let inv_freq = resolve_inv_freq(&cfg, factors.as_deref());
    // The step is Llama's block alone; any other arch runs its full graph.
    let plain = arch == "llama"
        && cfg.dense_arch() == DenseArch::Llama
        && !cfg.uses_partial_rope()
        && !cfg.has_granite_scalars()
        && cfg.kv_layers() == cfg.num_hidden_layers
        && cfg.sliding_window.is_none();
    if !plain {
        return export_forward(&ArchForward { cfg, inv_freq }, gguf, out, opts);
    }
    let model = GgufLlamaDecoder { cfg, inv_freq };
    rlx_core::export::decoder::export_decoder_with(&model, gguf, out, opts, extend)
}

/// Export the checkpoint at `path` (a GGUF or a Hugging Face directory),
/// built as `cfg` says, through the runner's full-sequence graph
/// ([`build_llama32_graph_sized_packed`], which applies every arch delta and
/// unrolls looped layers): a `generate` entry with no step.
pub fn export_forward_with(
    path: &Path,
    cfg: Llama32Config,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let mut loader = rlx_core::export::Checkpoint::open(path)?.loader()?;
    let factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
    drop(loader);
    let inv_freq = resolve_inv_freq(&cfg, factors.as_deref());
    export_forward(&ArchForward { cfg, inv_freq }, path, out, opts)
}

/// A Llama checkpoint: a GGUF through [`export_gguf`]; a Hugging Face
/// Llama through rlx-qwen3's exporter, and any other `model_type` this
/// runner takes (Granite, OLMo, Nemotron, …) through the config-driven one
/// ([`rlx_core::export::standard`]).
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    if path.extension().is_some_and(|e| e == "gguf") {
        return export_gguf(path, out, opts);
    }
    let ck = rlx_core::export::Checkpoint::open(path)?;
    match ck.model_type() {
        Some("llama" | "llama2" | "llama3") | None => {
            rlx_qwen3::export::export_rlxp(path, out, opts)
        }
        Some(_) if rlx_core::export::standard::supports(&ck) => {
            rlx_core::export::standard::export_rlxp(path, out, opts)
        }
        Some(kind) => bail!("{kind}: not a model type rlx-llama32's exporter builds"),
    }
}

/// Register the `llama32` exporter ([`rlx_core::export::register_exporter`])
/// for Llama GGUFs and checkpoints, over rlx-qwen3's (which takes
/// checkpoints only).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("llama32", "text-generation", export_rlxp).claims(&["llama"]));
}
