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
//! ## Status
//!
//! - [`config::DflashConfig`] — GGUF hyper-parameters. **Done.**
//! - [`builder::build_dflash_graph`] — encoder (`fc` + `enc.output_norm`) and
//!   the decoder blocks, transcribed from llama.cpp `src/models/dflash.cpp`.
//!   **Done.**
//! - Target-side taps + the propose/verify loop — see [`speculate`].
//!
//! Not to be confused with `rlx-laguna`'s `dflash` module: poolside ships a
//! *block-diffusion* draft checkpoint under the same product name, which is a
//! different algorithm.

pub mod builder;
pub mod config;
pub mod speculate;

pub use builder::build_dflash_graph;
pub use config::DflashConfig;
