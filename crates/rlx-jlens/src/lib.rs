//! **Jacobian lens** — read out what an internal activation is disposed to make
//! the model say.
//!
//! The lens linearly transports a residual-stream vector at any layer and
//! position into a later basis, then decodes it with the model's own
//! unembedding:
//!
//! ```text
//!   lens_l(h) = unembed( J_l · h ),   J_l = E[ ∂h_final / ∂h_l ]
//! ```
//!
//! `J_l` is one `[d_model, d_model]` matrix per layer, estimated as the average
//! input–output Jacobian over a corpus. It is prompt-independent: fit once,
//! apply anywhere. Unlike a logit lens — which decodes a mid-layer residual as
//! if the remaining layers were the identity — the transport accounts for what
//! the rest of the stack would have done to it.
//!
//! A native RLX port of the reference implementation accompanying *Verbalizable
//! Representations Form a Global Workspace in Language Models*.
//!
//! # Swapping models
//!
//! Nothing here knows what a Qwen is. A model implements [`model::LensModel`]
//! — hand back a graph whose input is a residual stream and whose output is a
//! residual stream — and the estimator, the fitting loop and the readout work
//! unchanged. Implementations live in [`models`], each behind its own feature,
//! so the core crate depends on no model crate.
//!
//! # Layout
//!
//! * [`estimator`] — the estimator's index arithmetic: which positions count,
//!   how a one-hot cotangent is laid out, how a VJP becomes rows of `J`.
//! * [`vjp`] — turning a model's forward graph into a lens VJP graph.
//! * [`model`] — the interface a model implements.
//! * [`fit`] — driving the VJP over a corpus to fit `J`.
//! * [`models`] — concrete [`model::LensModel`] implementations.
//!
//! # Why this works on RLX graphs
//!
//! `rlx_autodiff::grad_with_loss_wrt` exposes `d_output` as a real graph input
//! shaped like `outputs[0]`, so a VJP can be seeded with an arbitrary cotangent
//! rather than a scalar `1.0`; and `Wrt::Output` addresses an *intermediate*
//! activation across the renumbering that autodiff preparation performs. Those
//! two together are the whole primitive — no forward hooks, no retained tape.
//!
//! # Status
//!
//! Per-block Jacobians are implemented and verified against finite differences.
//! Whole-stack tapping ([`model::LensModel::stack`]) is not yet implemented for
//! any model; see the crate README for the open items, including a known
//! gradient discrepancy in Qwen3.5/3.6 gated-delta-net blocks.

pub mod corpus;
pub mod estimator;
pub mod fit;
pub mod heatmap;
pub mod lens;
pub mod model;
pub mod models;
pub mod readout;
pub mod taps;
pub mod vjp;

pub use corpus::{CorpusFit, FitProgress, paragraphs, to_prompts};
pub use estimator::{
    ResidualShape, SKIP_FIRST_N_POSITIONS, fill_onehot_cotangent, scaled_frobenius_norm,
    valid_positions, write_rows,
};
pub use fit::{
    BlockLens, FitConfig, Jacobian, LensTiming, StackLens, gdn_backward_is_fused_on,
    select_gdn_backward_for,
};
pub use lens::JacobianLens;
pub use model::{BlockGraph, LensError, LensModel, Params, StackGraph, UnembedGraph};
pub use readout::{Readout, TopToken, rank_of, top_tokens};
pub use taps::{
    chain_shape, layer_exit_taps, residual_chain, residual_stream, residual_stream_from,
};
pub use vjp::{Tap, TappedGraph};
