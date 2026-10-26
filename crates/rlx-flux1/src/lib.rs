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

//! FLUX.1 `[dev]` / `[schnell]` (Black Forest Labs) text-to-image for RLX.
//!
//! The full pipeline, native: T5-XXL + CLIP-L text encoders, the 12B MMDiT
//! rectified-flow denoiser (BFL single-file checkpoint), the 16-channel VAE
//! decoder, and LoRA adapters (diffusers/PEFT, BFL or kohya keys) merged
//! exactly into the weights on load.
//!
//! ```no_run
//! use rlx_flux1::{Flux1Options, Flux1Paths, Flux1Pipeline, Flux1Sample};
//! # fn main() -> anyhow::Result<()> {
//! let paths = Flux1Paths::discover("weights/image/flux1-dev".as_ref())?;
//! let mut pipe = Flux1Pipeline::open(paths, Flux1Options::default())?;
//! let img = pipe.generate("a red fox in the snow", &Flux1Sample::default())?;
//! img.save("fox.png".as_ref())?;
//! # Ok(()) }
//! ```
//!
//! Memory: weights stay at checkpoint width on device (bf16 → 24 GB for the
//! denoiser), and every large network is built as a chain of graph chunks so
//! host memory never holds more than one chunk at f32 (see [`transformer`]).

pub mod cli;
pub mod clip;
pub mod config;
pub mod embed;
pub mod lora;
pub mod pipeline;
pub mod precision;
pub mod rope;
pub mod sampler;
pub mod t5;
pub mod transformer;
pub mod vae;
pub mod web;
pub mod weights;

pub use config::{ClipTextConfig, Flux1Config, T5Config};
pub use lora::LoraSet;
pub use pipeline::{
    Flux1Image, Flux1Options, Flux1Paths, Flux1Pipeline, Flux1Sample, LatentGuide, PromptEmbeds,
};
pub use transformer::{DitGeometry, DitPart, Flux1Dit, plan_chunks};
pub use weights::{DitSource, TensorFile};
