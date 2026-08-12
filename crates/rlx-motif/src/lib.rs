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

//! # rlx-motif — Motif-3 (Motif Technologies) for RLX
//!
//! [Motif-3](https://huggingface.co/Motif-Technologies/Motif-3) is a `Motif`
//! (`MotifForCausalLM`) decoder: 53 layers, ~314 B total parameters, 4096 hidden,
//! 220 160 vocab, 262 144 context. Three things make it unlike anything else in
//! this repo, and all three are load-bearing:
//!
//! * **GDLA** — Grouped Differential Latent Attention. MLA-style low-rank Q
//!   (1024) and KV (512 + one shared 64-wide RoPE head) feeding 80 heads of 192,
//!   grouped 5-to-a-bundle: 4 signal heads and 1 *noise* head whose output is
//!   subtracted with an input-dependent λ, then an element-wise sigmoid output
//!   gate driven by the Q latent. 3 layers in 4 are 128-key sliding-window with
//!   their own RoPE base; the rest are global with YaRN and an `mscale²` softmax
//!   scale. See [`gdla`].
//! * **MHC** — Manifold-constrained Hyper-Connections. There is no single
//!   residual stream: the hidden state is 4 parallel streams, and each sublayer
//!   reduces them through a learned gate, then re-expands and mixes them through
//!   a **doubly stochastic** 4×4 matrix produced by 20 Sinkhorn iterations. See
//!   [`mhc`].
//! * **PolyNorm** — the FFN activation is a trainable polynomial
//!   `w₀·n(x³) + w₁·n(x²) + w₂·n(x) + b`, with *per-expert* coefficients across
//!   the 384-expert MoE. See [`polynorm`].
//!
//! The MoE itself is conventional-ish: sigmoid router, top-8 of 384 with a
//! load-balancing selection bias, normalized weights × `route_scale`, one shared
//! expert — so it reuses rlx-llada2's `group_limited_gate` kernel with a single
//! group.
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_motif::{MotifConfig, build_motif_text_flow, prepare_checkpoint};
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let dir = std::path::Path::new("/weights/Motif-3");
//! let cfg = MotifConfig::from_file(dir.join("config.json"))?;
//! let mut wm = WeightMap::from_safetensors_dir(dir)?;
//! rlx_motif::drop_mtp_layers(&mut wm);
//! prepare_checkpoint(&cfg, &mut wm)?;
//!
//! let seq = 8;
//! let built = build_motif_text_flow(&cfg, &mut wm, seq, true)?;
//! let mut compiled = compile_built(built, Device::Cpu)?;
//!
//! let (cos, sin) = cfg.rope_tables(seq);
//! let (swa_cos, swa_sin) = cfg.swa_rope_tables(seq);
//! let ids: Vec<f32> = vec![1.0; seq];
//! let logits = compiled.run(&[
//!     ("input_ids", ids.as_slice()),
//!     ("rope_cos", cos.as_slice()),
//!     ("rope_sin", sin.as_slice()),
//!     ("swa_rope_cos", swa_cos.as_slice()),
//!     ("swa_rope_sin", swa_sin.as_slice()),
//! ]);
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```
//!
//! ## Status
//!
//! The architecture is complete and checked against `modeling_motif.py`,
//! `configuration_motif.py` and every tensor name and shape in
//! `model.safetensors.index.json`. Tests cover PolyNorm, the Sinkhorn gate, GDLA
//! and the MoE against host references, plus the full graph end to end on
//! synthetic weights — all green on every backend: CPU, Metal, MLX, CUDA, ROCm,
//! Vulkan, wgpu and CoreML
//! (`RLX_TEST_DEVICE=<dev> cargo test -p rlx-motif --features <dev>`). The one
//! caveat is wgpu on a *Vulkan* adapter, which needs `RLX_ARENA_NO_REUSE=1` —
//! `rlx-wgpu`'s slot-reuse corruption, not this model; see the crate README.
//!
//! A real-weight run has **not** been done: the checkpoint is 629 GB of bf16
//! across 155 shards and [`rlx_core::weight_map::WeightMap`] is f32, so the
//! expert banks alone would be ~2.5 TB resident. That needs the paged/packed
//! expert path (as in `rlx_kimi_k3::moe`), which is not wired here.
//!
//! `model.mtp_layers.0` (`num_nextn_predict_layers = 1`) is a speculative-decode
//! head that `modeling_motif.py` itself never instantiates; [`drop_mtp_layers`]
//! discards it and inference ignores it, exactly as the reference does.

pub mod config;
pub mod flow;
pub mod gdla;
pub mod mhc;
pub mod moe;
pub mod polynorm;
pub mod weights;

pub use config::{LayerAttn, MotifConfig};
pub use flow::{EMBED_KEY, build_motif_text_flow};
pub use gdla::{ROPE_COS, ROPE_SIN, SWA_ROPE_COS, SWA_ROPE_SIN};
pub use weights::{drop_mtp_layers, prepare_checkpoint};

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "Motif-Technologies/Motif-3";
/// `config.json` `model_type` this crate claims.
pub const MODEL_TYPE: &str = "Motif";
