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

//! LFM2.5-VL runner (PLAN.md M7).
//!
//! LiquidAI's LFM2.5-VL-1.6B catalog row ships as
//! `general.architecture = lfm2-vl` / `lfm25-vl` / `lfm2_5_vl` in its
//! GGUF converters. LFM2-VL uses a SigLIP2 vision tower from Google
//! (separate-Q/K/V pre-LN ViT) plus a LLaVA-style 2-layer GELU
//! projector into the LFM2.5 LM hidden dim.
//!
//! ## Status
//!
//! - **Vision tower** ([`LfmVlVisionRunner`]) — implemented. SigLIP2
//!   encoder via [`rlx_flow::blocks::siglip_layer_fused_with_prefix`]
//!   plus LLaVA-style projector (linear → GELU → linear). Weight
//!   names follow the HF LFM2-VL layout
//!   (`vision_tower.vision_model.encoder.layers.{i}.…`,
//!   `multi_modal_projector.linear_{1,2}.…`).
//! - **Image preprocessing** ([`LfmVlImagePreprocessor`]) — bicubic
//!   resize + SigLIP normalization (mean/std = 0.5) + host-side patch
//!   embedding.
//! - **Text path** — uses [`rlx_lfm`] (LFM2.5 LM). The runner produces
//!   `[num_patches, lm_hidden]` embeddings the caller interleaves
//!   with text-token embeds at the `<image>` placeholder positions.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod config;
pub mod flow;
pub mod preprocess;
pub mod runner;

pub use config::LfmVlVisionConfig;
pub use flow::{LfmVlVisionBuilt, build_lfm_vl_vision, build_lfm_vl_vision_with_packed};
pub use preprocess::{
    LfmVlImagePreprocessor, LfmVlPreprocessWeights, assemble_hidden, extract_preprocess_weights,
    image_to_patch_tensor,
};
pub use runner::{LfmVlIdentityProjector, LfmVlVisionRunner, LfmVlVisionRunnerBuilder};

pub const PLAN_MILESTONE: &str = "M7";
pub const FAMILY: &str = "LFM2.5-VL";

const ACCEPTED_ARCHES: &[&str] = &["lfm2-vl", "lfm25-vl", "lfm2_5_vl", "lfm-vl"];

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-lfm-vl: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-lfm-vl: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
            eprintln!(
                "[rlx-lfm-vl] {path}: arch `{}` accepted. Use the library API \
                 (LfmVlVisionRunner::builder()) for image inference.",
                cfg.arch
            );
            return Ok(());
        }
    }
    bail!(
        "rlx-lfm-vl: usage: --weights <lm.gguf>; for image inference use the \
         library API (LfmVlVisionRunner::builder().mmproj(...).hf_config(...))"
    )
}
