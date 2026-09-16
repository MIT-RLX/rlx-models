//! The model interface the lens is written against.
//!
//! Swapping models is the point: the lens does not know what a Qwen or a Llama
//! is, only that a model can hand it a *graph* whose input is a residual stream
//! and whose output is a residual stream. Everything else — the estimator, the
//! Jacobian accumulation, the readout — is model-independent.
//!
//! Implement [`LensModel`] and the whole crate works. `models::qwen35` is one
//! implementation and is feature-gated, so `rlx-jlens` itself depends on no
//! model crate; adding a second model adds a feature, not a dependency on the
//! core.
//!
//! # What a new model has to provide
//!
//! [`LensModel::block`] returns a graph for a single residual block: residual
//! in, residual out, with the incoming residual as a *named graph input*. That
//! is enough to fit a per-block Jacobian, because a named input is addressable
//! across autodiff preparation without any further cooperation from the model.
//!
//! [`LensModel::stack`] is the full-depth version — one graph, `outputs[0]` the
//! residual at the target layer and `outputs[1..]` the source-layer residuals —
//! and is what a lens over a whole model needs. It has a default
//! implementation that returns [`LensError::Unsupported`], so a model can
//! usefully implement `block` alone and add `stack` later.

use std::collections::HashMap;

use rlx_ir::Graph;

use crate::vjp::TappedGraph;

/// Parameter bindings for a graph a [`LensModel`] produced.
pub type Params = HashMap<String, Vec<f32>>;

#[derive(Debug, thiserror::Error)]
pub enum LensError {
    #[error("{model} does not support {what}")]
    Unsupported { model: String, what: String },
    #[error("layer {layer} out of range for a {n_layers}-layer model")]
    LayerOutOfRange { layer: usize, n_layers: usize },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, LensError>;

/// The whole residual stack, tapped for a lens.
///
/// Unlike [`BlockGraph`], the input is a *prompt* — the graph starts at the
/// embedding — so the lens feeds token ids rather than a residual.
pub struct StackGraph {
    /// `outputs[0]` is the residual at the target layer; `outputs[1..]` are the
    /// source-layer residuals, in the order `source_layers` requested.
    pub tapped: TappedGraph,
    pub params: Params,
    /// Name of the graph input carrying token ids, shaped `[batch, seq]`.
    pub token_input: String,
    /// Auxiliary graph inputs bound on every run, by name.
    ///
    /// Not every trunk is a function of its token sequence alone. A Qwen-VL
    /// prefill takes mRoPE `cos`/`sin` tables as *inputs* rather than params,
    /// because they depend on where the image sits in the prompt; other models
    /// want masks or position ids the same way. These are fed alongside
    /// [`Self::token_input`] on both halves of the split.
    pub extra_feeds: Vec<(String, Vec<f32>)>,
    /// Layer index per tap, parallel to `tapped.taps()`.
    pub layers: Vec<usize>,
    pub batch: usize,
    pub seq: usize,
}

/// The model's unembedding, as a graph from residual rows to logits.
pub struct UnembedGraph {
    /// `outputs[0]` is `[rows, vocab]`.
    pub graph: Graph,
    pub params: Params,
    /// Name of the graph input taking `[rows, d_model]` residuals.
    pub residual_input: String,
    pub rows: usize,
    pub vocab: usize,
}

/// A single residual block, as a graph from residual to residual.
pub struct BlockGraph {
    /// `outputs[0]` is the outgoing residual, shaped `[batch, seq, d_model]`.
    pub graph: Graph,
    /// Parameters to bind before running.
    pub params: Params,
    /// Name of the graph input carrying the incoming residual.
    pub residual_input: String,
    pub batch: usize,
    pub seq: usize,
}

/// What the lens needs from a model.
pub trait LensModel {
    /// Name, for diagnostics.
    fn name(&self) -> &str;

    /// Number of residual blocks.
    fn n_layers(&self) -> usize;

    /// Residual-stream width.
    fn d_model(&self) -> usize;

    /// One residual block as a graph: residual in → residual out.
    ///
    /// The incoming residual must be a named `Op::Input` so it can be named as
    /// a gradient target; the block's output must be `outputs[0]`.
    fn block(&self, layer: usize, batch: usize, seq: usize) -> Result<BlockGraph>;

    /// The full residual stack, tapped at `source_layers`.
    ///
    /// This is the lens proper: one forward over a *prompt*, with the residual
    /// stream tapped at each source layer, so a single VJP yields
    /// `∂h_target/∂h_l` for every `l` at once.
    ///
    /// Returns [`LensError::Unsupported`] by default — a model can usefully
    /// implement [`Self::block`] alone and add this later.
    fn stack(
        &self,
        source_layers: &[usize],
        target_layer: usize,
        batch: usize,
        seq: usize,
    ) -> Result<StackGraph> {
        let _ = (source_layers, target_layer, batch, seq);
        Err(LensError::Unsupported {
            model: self.name().to_string(),
            what: "whole-stack tapping (LensModel::stack)".to_string(),
        })
    }

    /// The model's own unembedding: residual → logits.
    ///
    /// Final norm plus LM head, as a graph over `[rows, d_model]`. The lens
    /// decodes a *transported* residual with this — using the model's own head
    /// is the whole point, since the transported vector is meant to live in the
    /// final-layer basis.
    ///
    /// Returns [`LensError::Unsupported`] by default.
    fn unembed(&self, rows: usize) -> Result<UnembedGraph> {
        let _ = rows;
        Err(LensError::Unsupported {
            model: self.name().to_string(),
            what: "unembedding (LensModel::unembed)".to_string(),
        })
    }

    /// Bounds-check a layer index.
    fn check_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.n_layers() {
            return Err(LensError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers(),
            });
        }
        Ok(())
    }
}
