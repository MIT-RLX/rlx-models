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

//! Nemotron-3 Nano Omni runner (PLAN.md M7) — text + vision + audio.
//!
//! NVIDIA's Nemotron-3 Nano Omni is a 30 B-parameter A3B-routed MoE
//! that accepts text, images, and audio. Arch tags vary across
//! converters; we accept the union below.
//!
//! ## Status (per modality)
//!
//! - **Vision tower** ([`NemotronOmniVisionRunner`]) — implemented.
//!   SigLIP-variant encoder via
//!   [`rlx_flow::blocks::siglip_layer_fused_with_prefix`] +
//!   LLaVA-style projector reading `vision_tower.…` / `mm_projector.…`
//!   weight names.
//! - **Audio encoder** ([`NemotronOmniAudioEncoder`]) — adapter over
//!   [`rlx_whisper::WhisperRunner`]; implements
//!   [`rlx_vlm_base::AudioEncoder`].
//! - **LM text path** — Nemotron-H is a hybrid Mamba + attention
//!   stack. The `MambaScanStage` / IR ops landed in `rlx-ssm` (M5)
//!   with CPU + Metal kernel paths; `rlx-nemotron`'s `build()` is the
//!   final wiring step (still upstream-deferred until the
//!   `Mamba2Block` ergonomic wrapper lands in `rlx-ssm`). Once
//!   `rlx-nemotron::NemotronRunner` ships, this crate composes:
//!   vision → projector → image embed slot · audio → projector →
//!   audio embed slot · text → token embeddings → interleave → LM.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod audio;
pub mod config;
pub mod vision;

pub use audio::{AudioEncoderBox, NemotronOmniAudioEncoder, SyntheticAudioEncoder};
pub use config::NemotronOmniVisionConfig;
pub use vision::{
    NemotronOmniIdentityProjector, NemotronOmniImagePreprocessor, NemotronOmniPreprocessWeights,
    NemotronOmniVisionBuilt, NemotronOmniVisionRunner, NemotronOmniVisionRunnerBuilder,
    build_nemotron_omni_vision,
};

pub const PLAN_MILESTONE: &str = "M7";
pub const FAMILY: &str = "Nemotron-3 Nano Omni";

const ACCEPTED_ARCHES: &[&str] = &[
    "nemotron-omni",
    "nemotron_omni",
    "nemotron3-omni",
    "nemotron_h_omni",
];

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-nemotron-omni: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-nemotron-omni: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
            eprintln!(
                "[rlx-nemotron-omni] {path}: arch `{}` accepted. Use the library API \
                 (NemotronOmniVisionRunner::builder() / NemotronOmniAudioEncoder::new(...)) \
                 for multimodal inference.",
                cfg.arch
            );
            return Ok(());
        }
    }
    bail!(
        "rlx-nemotron-omni: usage: --weights <lm.gguf>; for inference use the \
         library API. LM text path requires rlx-nemotron + Mamba2 wrapper (M5)."
    )
}
