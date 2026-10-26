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

//! The one `.rlxp` exporter: any model, any graph, as a package rlx runs
//! with no rlx-models code.
//!
//! A family says what its model *is* — graphs built from its checkpoint —
//! and this module does the rest the same way for every family:
//!
//! - [`Checkpoint`] reads a Hugging Face directory or a GGUF: its config,
//!   tokenizer, chat template, stop ids, sampling defaults and card.
//! - [`Package`] writes the package: functions, entries, held tensors,
//!   sidecars, and the params found in the checkpoint referenced rather
//!   than copied ([`Referencer`]).
//! - [`decoder::Decoder`] is a decoder-only language model: its prompt
//!   graph and decode step at a size. [`decoder::export_decoder`] makes them
//!   symbolic, puts the cache inside the step, adds the prompt, batch and
//!   speculative functions and writes a `generate` entry.
//! - [`graph::export_graphs`] is any graph: tensors in, tensors out, as a
//!   `pipeline` entry.
//! - [`register_exporter`] / [`export`] pick a family's exporter for a
//!   checkpoint the way a runner is picked for it ([`crate::lm_registry`]),
//!   and [`check::check_generate`] proves a package decodes what the
//!   family's [`LmRunner`](rlx_runtime::lm::LmRunner) does.

pub mod audio;
pub mod check;
pub mod decoder;
pub mod graph;
mod package;
pub mod program;
mod registry;
pub mod rope;
mod source;
pub mod spm;
pub mod standard;
pub mod vision;

pub use crate::package_export::{
    Referencer, add_huggingface_card, matmul_weights_as, package_dir, param_dims, relative_path,
    tidy,
};
pub use package::Package;
pub use registry::{Exporter, ExporterFn, detect, export, exporter, exporters, register_exporter};
pub use source::{Checkpoint, Format};

/// How to export, for every family: each reads the fields that apply to
/// it, and an empty list means the family's own default.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// The model's own version, for the model block.
    pub version: String,
    /// Copy every weight into the package instead of referencing the
    /// checkpoint.
    pub embed_weights: bool,
    /// The dtype of the matmul weights (`F16` / `BF16` halve the bytes a
    /// decode reads; the checkpoint's are converted at bind). `F32` keeps
    /// them as the checkpoint widens to; unset, as the exporter says the
    /// checkpoint stores them (`Package::weights_as_stored`), else f32.
    pub weight_dtype: Option<rlx_ir::DType>,
    /// Store one graph symbolic in its sizes, run at each, rather than one
    /// per size (the fallback when it is not symbolic).
    pub symbolic: bool,
    /// Which of the checkpoint's variants to export, for a family that
    /// ships several (Jina v5's task adapters). `None`: the family's.
    pub variant: Option<String>,

    // ── Decoders (`generate`, `transcribe`) ────────────────────────────
    /// Cache capacities to build the decode step at; the largest is the
    /// package's context length. Empty: the family's.
    pub uppers: Vec<usize>,
    /// With `symbolic`: put the decode state in the graph
    /// (`rlx_ir::stateful`), so the step takes the token and its position
    /// and its caches feed straight back.
    pub stateful: bool,
    /// With `stateful`: prompt lengths for a `prefill` function, which runs
    /// a whole prompt (padded to the shortest that holds it) at once and
    /// seeds the step's caches. Those past the context are dropped; empty
    /// leaves prompts to the step, one token at a time.
    pub prefill: Vec<usize>,
    /// The longest context the package runs: its RoPE tables cover it, and
    /// a symbolic step grows past `uppers` to it (`max_context`). `None`:
    /// the largest of `uppers`. Each 1024 positions are 0.5 MB of tables.
    pub max_context: Option<usize>,
    /// With `symbolic`: the step symbolic in its batch too (`batch`), so
    /// one package decodes any number of sequences together
    /// (`generate_batch`).
    pub batch: bool,
    /// With `stateful`: draft tokens a speculative decoder checks at once.
    /// Adds a `check` function of `speculate + 1` tokens (the last token,
    /// then the draft) whose logits say which of the draft to keep; 0
    /// leaves it out.
    pub speculate: usize,
    /// With `prefill`: batch sizes of further prompt functions, each
    /// running that many prompts at once. Empty: a batch's prompts prefill
    /// one after another.
    pub prefill_batches: Vec<usize>,

    // ── Encoders (`embed`, `score`) ────────────────────────────────────
    /// Sequence lengths to build an encoder at, capped by the model's
    /// position table. Empty: the family's.
    pub seqs: Vec<usize>,
    /// Texts an encoder runs at once.
    pub texts: usize,
    /// `"mean"` or `"cls"`; `None`: the family's.
    pub pooling: Option<String>,

    // ── Pipelines ──────────────────────────────────────────────────────
    /// Write a shared front end (Whisper's log-mel, SigLIP's resize) to
    /// this package instead of the model's, and reference it from there:
    /// one front end for every model that shares it.
    pub frontend_package: Option<std::path::PathBuf>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            version: "1.0.0".into(),
            embed_weights: false,
            weight_dtype: None,
            symbolic: true,
            variant: None,
            uppers: Vec::new(),
            stateful: true,
            prefill: vec![16, 64, 256],
            max_context: None,
            batch: true,
            // A 4-token check runs Metal's few-row GEMV (two passes past 4
            // rows).
            speculate: 3,
            prefill_batches: vec![2, 4, 8],
            seqs: Vec::new(),
            texts: 1,
            pooling: None,
            frontend_package: None,
        }
    }
}

impl ExportOptions {
    /// `uppers`, or `default` when empty.
    pub fn uppers_or(&self, default: &[usize]) -> Vec<usize> {
        if self.uppers.is_empty() {
            default.to_vec()
        } else {
            self.uppers.clone()
        }
    }

    /// `seqs`, or `default` when empty.
    pub fn seqs_or(&self, default: &[usize]) -> Vec<usize> {
        if self.seqs.is_empty() {
            default.to_vec()
        } else {
            self.seqs.clone()
        }
    }
}

/// What an export wrote, for every family.
#[derive(Debug, Default)]
pub struct ExportReport {
    /// The package's functions, in order.
    pub functions: Vec<String>,
    /// Params read from the checkpoint at load time (directly or
    /// transposed).
    pub referenced: usize,
    /// Params read from the checkpoint as stored (quantized blocks).
    pub packed: usize,
    /// Params the package holds (derived ones, or all with
    /// `embed_weights`).
    pub held: usize,
    /// Matmul weights declared `weight_dtype` (in every function).
    pub narrowed: usize,
    /// Why a graph is stored one per size, if it is.
    pub symbolic_failure: Option<String>,
    /// Why the decode state stays with the host, if it does.
    pub stateful_failure: Option<String>,
    /// Why there is no prefill function, if a stateful step has none.
    pub prefill_failure: Option<String>,
    /// Why the step is not symbolic in its batch, if it is not.
    pub batch_failure: Option<String>,
    /// Why there is no check function, if speculation was asked for.
    pub check_failure: Option<String>,
}

impl ExportReport {
    /// Each option the export could not give, and why.
    pub fn failures(&self) -> Vec<(&'static str, &str)> {
        [
            ("symbolic", &self.symbolic_failure),
            ("stateful", &self.stateful_failure),
            ("prefill", &self.prefill_failure),
            ("batch", &self.batch_failure),
            ("check", &self.check_failure),
        ]
        .into_iter()
        .filter_map(|(k, v)| v.as_deref().map(|v| (k, v)))
        .collect()
    }
}

impl std::fmt::Display for ExportReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "functions: {}\nparams: {} referenced, {} packed, {} held",
            self.functions.join(", "),
            self.referenced,
            self.packed,
            self.held
        )?;
        if self.narrowed > 0 {
            write!(f, ", {} matmul weights narrowed", self.narrowed)?;
        }
        for (what, why) in self.failures() {
            write!(f, "\nno {what}: {why}")?;
        }
        Ok(())
    }
}
