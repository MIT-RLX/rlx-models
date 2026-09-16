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

//! DFlash speculative-decoding drafter (`general.architecture = "dflash"`).
//!
//! Ships alongside Muse-Glimmer-30B as `dflash-kquant.gguf` (1.63 GB, 2.6B
//! params, 5 blocks). It is an **Eagle-style** head: no embedding, no LM head of
//! its own. It reads the TARGET model's residual streams at
//! `dflash.target_layers` (`[2, 14, 26, 38, 50]` for Muse-Glimmer's 52 layers),
//! fuses them through `fc`, and proposes `dflash.block_size` (16) tokens per
//! step, which the target then verifies in one batched forward.
//!
//! The target-side hook already exists upstream: llama.cpp's
//! `muse-glimmer.cpp` does `res->t_layer_inp[il] = inpL; // expose per-layer
//! residual for speculative drafts (see LLM_KV_TARGET_LAYERS)`.
//!
//! ## DFlash2
//!
//! A DFlash2 checkpoint announces itself with `dflash.selector_top_k > 0` and
//! adds two modules that recover the coherence a block drafter loses by
//! predicting every position independently:
//!
//! - [`conv::emit_dyn_conv`] — grouped dynamic depthwise convolution, so a
//!   position can see its predecessor inside the block.
//! - [`selector::emit_selector_lattice`] — low-rank bilinear scores over
//!   adjacent candidate pairs, walked by [`selector::walk_lattice`].
//!
//! ## Status
//!
//! - [`config::DflashConfig`] / [`config::Dflash2Config`] — GGUF
//!   hyper-parameters, both generations. **Done.**
//! - [`conv`] and [`selector`] — the DFlash2 modules, each validated against
//!   an independently written CPU oracle. **Done.**
//! - [`builder`] — the three passes upstream actually runs:
//!   [`builder::build_encoder_graph`] (`fc` + `enc.output_norm`, nothing else),
//!   [`builder::build_kv_inject_graph`] (fused features → per-layer K/V), and
//!   [`builder::build_decoder_graph`] (the `[anchor, MASK, …]` noise block
//!   under **non-causal** attention over `[cache ‖ block]`). The DFlash2
//!   modules are wired into the decoder. **Done.**
//! - [`runner`] — the propose/verify loop. [`runner::DflashDrafter`] owns the
//!   graphs and the KV cache; [`runner::DflashTarget`] is the (small) surface a
//!   target model implements; [`runner::DflashLoop`] drives it, entered through
//!   `seed` / `generate`, verifying with `speculative_accept_sparse` when the
//!   draft was sampled and prefix agreement when it was greedy. **Done**, and
//!   exercised end-to-end against a mock target — including that greedy
//!   speculation reproduces plain decoding token for token.
//! - [`runner::CallbackTarget`] — plug an existing decode loop in with three
//!   closures. This crate stays free of any model dependency on purpose; the
//!   model-side wiring belongs in the model's crate or a dev-dependency test,
//!   the way `rlx-eagle3` does it.
//! - Validated against the released `z-lab/Qwen3.8-27B-DFlash2` GGUF (5 layers,
//!   hidden 5120, block 8, conv group 16 / kernel 2, selector rank 256 /
//!   top-k 16, taps at layers `[6, 20, 34, 48, 62]`): config parses, all three
//!   graphs build, 46 packed K-quant tensors load, and the loop runs rounds
//!   with consistent cache bookkeeping. The selector ships 128 candidate ids +
//!   1792 pair scores per block instead of 1,986,560 logits — a 1029x cut in
//!   what crosses back to the host.
//! - [`config::DflashConfig::from_hf_json`] + [`config::hf_to_dflash_name`] —
//!   the released *small* drafters (`*/qwen3-8b-dflash-*`) ship safetensors
//!   with transformers naming, not GGUF, so both entry points exist.
//! - Target side: `rlx-qwen3` now exports residual taps
//!   (`Qwen3Flow::with_tap_layers`, backed by `rlx-flow`'s
//!   `Qwen3DecodeLayerStage::layer_with_tap`), which is what an Eagle-style
//!   drafter fuses. `examples/dflash_measure.rs` wires the two into a real
//!   acceptance measurement.
//! - Reduced draft vocabularies (`d2t`) are rejected at build time rather
//!   than mistranslated.
//! - **Sliding-window attention past the window is unsupported**, and
//!   [`runner::DflashDrafter::new`] refuses rather than diverging: bucketed
//!   padding breaks `key index == absolute position`, which is where rlx
//!   derives a positional mask's origin. The released checkpoints set a 2048
//!   window, so that is the current context ceiling for the drafter.
//!
//! Not to be confused with `rlx-laguna`'s `dflash` module: poolside ships a
//! *block-diffusion* draft checkpoint under the same product name, which is a
//! different algorithm.

pub mod builder;
pub mod config;
pub mod conv;
pub mod runner;
pub mod selector;
pub mod speculate;

pub use builder::{
    DecoderOutputs, build_decoder_graph, build_encoder_graph, build_kv_inject_graph, rope_tables,
};
pub use config::{Dflash2Config, DflashConfig, hf_to_dflash_name};
pub use conv::{ConvSide, emit_dyn_conv};
pub use runner::{
    CallbackTarget, DflashDrafter, DflashLoop, DflashTarget, DrafterOptions, ReplayLoader,
    TargetStep,
};
pub use selector::{SelectorLattice, emit_selector_lattice, walk_lattice};
