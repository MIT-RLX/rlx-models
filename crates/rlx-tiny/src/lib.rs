//! # rlx-tiny
//!
//! Train a small decoder-only transformer **from scratch** on the
//! [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
//! where **every weight matrix is synthesized from a tiny codebook** instead of
//! stored dense — "functions not data". It is a near-clone of the sibling
//! [`rlx-tinystories`](https://docs.rs) dense GPT (same dataset, tokenizer, and
//! train loop) so the two can be A/B compared on identical data.
//!
//! What makes it "tiny": each `[k,n]` weight is a `k·n → NE·ED`-number codebook
//! weight-synthesis (`Op::SynthMatMul`, residual multi-stage VQ + optional LoRA),
//! the FFN activation is a learnable KAN spline (`Op::SplineActivation`), and the
//! forward is expressed as a **single [`rlx!`](rlx_tensor::rlx) block**. The
//! codebooks can be random-init'd or **product-quantized from a trained dense
//! model** (`--init-from`), and a dense teacher can **distill** into the run
//! (`--distill`). See [`model`] for the architecture in detail.
//!
//! ## Binaries
//! - `rlx-tiny-train` — download/point-at TinyStories and train.
//! - `rlx-tiny-generate` — sample stories from a checkpoint.
//!
//! ## Library entry points
//! - [`model::build`] / [`model::init`] — build + initialize the codebook GPT graph.
//! - [`Trainer`] / [`TrainOpts`] — the training loop (build+init → `Trainer::run`).
//! - [`data::Corpus`] / [`data::Batcher`] — mmap corpus + gather-fed minibatches.
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
pub mod quantize;
pub mod rng;
pub mod sample;
pub mod tokenizer;
pub mod train;

pub use config::GptConfig;
pub use optim::HybridOptimizer;
pub use progress::Progress;
pub use rng::Rng;
pub use sample::{GenOptions, generate};
pub use train::{TrainOpts, Trainer};
