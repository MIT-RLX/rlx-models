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

//! Laguna MoE — [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) /
//! [Laguna-XS-2.1](https://huggingface.co/poolside/Laguna-XS-2.1).
//!
//! Production path: Unsloth / Poolside GGUF **packed mmap** ([`packed`]) +
//! KV-cached greedy generate ([`packed_forward`]), optionally accelerating
//! packed matmuls with [`device_matmul::DeviceMatmul`] (`--device metal|mlx`).
//! Chat via [`chat`]; OpenAI via `LagunaEngine` (feature `serve`) — prefer
//! the central `rlx-openai` host. Quant→F32 expand is off by default
//! ([`memory`]; `RLX_LAGUNA_ALLOW_F32_EXPAND=1` / `--allow-f32-expand` to opt in).
//! Synth / fixture graphs use [`LagunaRunner`]. Compiled e2e IR and DFlash are
//! scaffolded only ([`builder`], [`dflash`]).

pub mod builder;
pub mod chat;
pub mod cli;
pub mod config;
pub mod device_matmul;
pub mod dflash;
pub mod eager;
pub mod gguf_layout;
#[cfg(feature = "hf-probe")]
pub mod gguf_probe;
pub mod memory;
pub mod mlx_affine;
pub mod mlx_load;
pub mod packed;
pub mod packed_forward;
pub mod runner;
#[cfg(feature = "serve")]
pub mod serve;
pub mod synth;
pub mod weights;

pub use builder::{build_prefill_graph, build_status};
pub use chat::{EOS_TOKEN, LagunaChat};
pub use config::{
    ARCHITECTURE, AttnGating, AttnLayerType, FAMILY, GGUF_ARCH, HF_GGUF_REPO, HF_GGUF_REPO_XS,
    HF_MODEL_ID, HF_MODEL_ID_XS, LagunaConfig, LagunaVariant, MODEL_TYPE, MlpLayerType,
    RopeLayerParams,
};
pub use device_matmul::{DeviceMatmul, parse_device};
pub use dflash::{DFlashConfig, propose_and_verify};
pub use eager::{TextWeights, forward_logits, greedy_next};
pub use gguf_layout::{GGUF_ARCHES, gguf_to_eager_key};
#[cfg(feature = "hf-probe")]
pub use gguf_probe::{DEFAULT_QUANT, GgufProbeReport};
pub use memory::{
    ALLOW_F32_EXPAND, GgufRamEstimate, PACKED_ONLY_POLICY, allow_f32_expand, estimate_ram,
    open_gguf_header_only, refuse_f32_expand,
};
pub use packed::{
    LagunaPackedFfn, LagunaPackedLayer, LagunaPackedWeights, MatWeight, PackedParams,
};
pub use packed_forward::{LayerKvCache, PackedKvCache};
pub use runner::{LagunaPackedRunner, LagunaRunner, LagunaRunnerBuilder};
#[cfg(feature = "serve")]
pub use serve::LagunaEngine;
pub use synth::{synthetic_text_weights, tiny_cfg};
pub use weights::{expected_hf_keys, hf_to_eager_key};
