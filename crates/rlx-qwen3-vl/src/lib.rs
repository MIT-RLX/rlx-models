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

//! Qwen3-VL runner (PLAN.md M7).
//!
//! Qwen3-VL ships as `general.architecture = qwen3vl` /
//! `qwen3vlmoe` / `qwen3-vl` / `qwen3_vl` in its GGUF converters
//! (the catalog row is `qwen3-vl-30b` = `Qwen3-VL-30B-A3B-Instruct`,
//! a 30 B-parameter MoE with A3B active-experts routing on the LM
//! side and a SigLIP-variant vision tower).
//!
//! ## Status
//!
//! - **Vision tower** ([`Qwen3VlVisionRunner`]) — implemented.
//!   SigLIP-variant ViT (pre-LN, separate Q/K/V, GELU FFN, no
//!   LayerScale, no CLS) via [`rlx_flow::blocks::siglip_layer_fused`]
//!   + multimodal projector (LayerNorm → 2× linear with GELU). Reads
//!     the mmproj GGUF + an HF `config.json` for hyperparameters.
//! - **Image preprocessing** ([`Qwen3VlImagePreprocessor`]) — bicubic
//!   resize + SigLIP normalization (mean/std = 0.5) + host-side
//!   patch embedding.
//! - **LM integration** — [`Qwen3VlRunner`] composes [`rlx_qwen3::Qwen3Runner`]
//!   with [`Qwen3VlVisionRunner`], implements [`rlx_cli::LmRunner`] (including
//!   `generate_multimodal`), and is wired through `auto_runner_with_mmproj`.
//! - **MoE block** (`rlx_flow::blocks::moe::MoeFfnStage`) — already
//!   landed via PLAN.md M5.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod config;
pub mod flow;
pub mod multimodal;
pub mod preprocess;
pub mod runner;
pub mod vl_runner;

pub use config::Qwen3VlVisionConfig;
pub use flow::{Qwen3VlVisionBuilt, build_qwen3_vl_vision, build_qwen3_vl_vision_with_packed};
pub use multimodal::{
    IMAGE_PAD, MEDIA_MARKER, MultimodalPrefill, MultimodalPrompt, VISION_END, VISION_START,
    VisionEncodeOutput, normalize_media_prompt,
};
pub use preprocess::{
    Qwen3VlImagePreprocessor, Qwen3VlPreprocessWeights, assemble_hidden,
    extract_preprocess_weights, image_to_patch_tensor,
};
pub use runner::{Qwen3VlIdentityProjector, Qwen3VlVisionRunner, Qwen3VlVisionRunnerBuilder};
pub use vl_runner::{Qwen3VlRunner, Qwen3VlRunnerBuilder};

pub const PLAN_MILESTONE: &str = "M7";
pub const FAMILY: &str = "Qwen3-VL";

const ACCEPTED_ARCHES: &[&str] = &["qwen3vl", "qwen3vlmoe", "qwen3_vl", "qwen3-vl"];

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-qwen3-vl: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-qwen3-vl: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
            eprintln!(
                "[rlx-qwen3-vl] {path}: arch `{}` accepted. Use the library API \
                 (Qwen3VlVisionRunner::builder()) for image inference.",
                cfg.arch
            );
            return Ok(());
        }
    }
    bail!(
        "rlx-qwen3-vl: usage: --weights <lm.gguf>; for image inference use the \
         library API (Qwen3VlVisionRunner::builder().mmproj(...).hf_config(...))"
    )
}
