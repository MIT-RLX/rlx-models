//! # rlx-tinystories
//!
//! Train a small **nanoGPT / GPT-2-style** decoder-only transformer **from
//! scratch** on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
//! dataset — a showcase of the RLX training flow: the model graph is written in
//! the [`rlx!`](rlx_tensor::rlx) DSL, gradients come from `rlx-tensor`'s
//! autodiff, and the loop is driven by the `Func::train_step_*` helpers with
//! `AdamW` + a warmup-cosine schedule, on Apple GPU (Metal) or CPU.
//!
//! The whole training objective is one line of DSL:
//! `mean(cross_entropy(logits, targets))`.
//!
//! ## Binaries
//! - `rlx-tinystories-train` — download/point-at TinyStories and train.
//! - `rlx-tinystories-generate` — sample stories from a checkpoint.
//!
//! ## Library entry points
//! - [`model::build`] / [`model::init`] — build + initialize the GPT graph.
//! - [`data::Corpus`] / [`data::Batcher`] — mmap corpus + one-hot minibatches.
//! - [`sample::generate`] — autoregressive text generation.
//! - [`checkpoint`] — save/load trained weights.

pub mod bpe;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod model;
pub mod optim;
pub mod precision;
pub mod progress;
pub mod rng;
pub mod sample;
pub mod tokenizer;

pub use config::GptConfig;
pub use optim::HybridOptimizer;
pub use progress::Progress;
pub use rng::Rng;
pub use sample::{GenOptions, generate};
