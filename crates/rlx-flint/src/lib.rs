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

//! # rlx-flint — Recursive Flint for RLX
//!
//! [Flint](https://huggingface.co/RecursiveSI/flint) is Recursive's
//! mixture-of-experts coding model: 39.8 B parameters, ~2.5 B active per
//! token. HF `model_type = flint`, `FlintForCausalLM`; BF16 safetensors. The
//! only published implementation is the SGLang plugin in the model repo
//! (`sglang/flint_sglang-*.whl`, `model.py`), which this crate follows.
//!
//! | Piece | Where | Notes |
//! |---|---|---|
//! | [`attention`] | every layer | 32 q / 4 kv heads × 128, per-head QK RMSNorm, **learned sink logit per head** |
//! | sliding / full | 35 / 14 layers | sliding sees the current token + 512 before it; full layers are causal |
//! | RoPE | sliding layers only | NeoX, θ = 10000, whole head; full layers use **no** positional encoding |
//! | dense MLP | layers 0, 1 | SwiGLU, 6144 wide |
//! | [`moe`] | layers 2–48 | **sigmoid** router, top-6 of 256, renormalized; + an **ungated** shared expert (1024) |
//! | [`runner`] | the full checkpoint | per-layer graphs; routed experts read and computed on the host ([`experts`]) |
//! | [`chat`], [`tools`] | prompts | the chat template, tool declarations and calls, reply parsing |
//!
//! ## Running it
//!
//! ```no_run
//! use rlx_flint::{FlintRunner, RunnerOptions, chat};
//! use rlx_runtime::Device;
//!
//! let dir = "weights/lm/flint";
//! let tok = chat::FlintTokenizer::open(dir)?;
//! let prompt = chat::render(&[chat::Message::new(chat::Role::User, "Hi")], true);
//! let ids = tok.encode(&prompt)?;
//! let mut run = FlintRunner::open(dir, RunnerOptions::new(Device::Cpu))?;
//! let out = run.generate_greedy(&ids, 64, |_| true)?;
//! println!("{}", tok.decode(&out)?);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Or `cargo run -p rlx-flint --release --features metal -- --weights
//! weights/lm/flint --device metal --prompt "…"`.
//!
//! [`flow::build_flint_text_flow`] builds the whole model as one resident
//! graph instead — every expert a parameter — which is what the paged runner
//! is checked against on synthetic weights.

pub mod attention;
pub mod chat;
pub mod checkpoint;
pub mod cli;
pub mod common;
pub mod config;
pub mod experts;
pub mod flow;
pub mod mask;
pub mod moe;
pub mod prefetch;
pub mod quant;
pub mod runner;
pub mod synth;
pub mod tools;

pub use checkpoint::{Checkpoint, CheckpointF32, CheckpointRef};
pub use config::{ARCHITECTURE, FlintConfig, LayerSpec, MODEL_TYPE};
pub use flow::build_flint_text_flow;
pub use runner::{FlintRunner, RunnerOptions, Timings};

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "RecursiveSI/flint";

/// Register [`FlintRunner`] with the generic LM path
/// (`rlx_core::lm_registry`), under the runner name `flint`.
pub fn register() {
    use rlx_core::lm_registry::{LmRunnerRegistration, register_lm_runner};
    register_lm_runner(
        LmRunnerRegistration::new("flint", |path, opts| {
            let mut ro = RunnerOptions::new(opts.device_or_auto());
            if let Some(n) = opts.max_seq {
                ro.max_context = n;
            }
            Ok(Box::new(FlintRunner::open(path, ro)?))
        })
        .mmproj_note("Flint is text-only"),
    );
}
