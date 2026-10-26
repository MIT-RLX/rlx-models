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

//! FLUX 3 Image Klein (Black Forest Labs) for RLX: text-to-image and
//! (multi-)reference editing.
//!
//! Native end to end: the Qwen3-VL-4B text tower as text encoder
//! ([`text`]), the 5.5B per-mode DiT ([`dit`]), the FLUX.2 autoencoder
//! ([`vae`]), and the prompt upsampler `flux-3-image-klein-pup`
//! (Qwen3.5-9B, [`pup`]). Both checkpoints run: the undistilled **base**
//! (50 steps, CFG) and the distilled **klein** flow map (16 steps, no CFG),
//! told apart by the checkpoint's tensors.
//!
//! ```no_run
//! use rlx_flux3::{Flux3Options, Flux3Paths, Flux3Pipeline, Flux3Sample};
//! # fn main() -> anyhow::Result<()> {
//! let paths = Flux3Paths::discover("weights/image/flux3-klein-base".as_ref())?;
//! let mut pipe = Flux3Pipeline::open(paths, Flux3Options::default())?;
//! let caption = "A red fox curled up asleep in fresh snow, soft overcast light.";
//! let img = pipe.generate(caption, &[], &Flux3Sample::default())?;
//! img.save("fox.png".as_ref())?;
//! # Ok(()) }
//! ```
//!
//! The model expects full scene captions; [`pup::Upsampler`] writes one
//! from a short request, as the reference does by default.

pub mod bbox;
pub mod cli;
pub mod config;
pub mod dit;
pub mod embed;
pub mod pipeline;
pub mod pup;
pub mod rope;
pub mod sampler;
pub mod text;
pub mod vae;
pub mod weights;

pub use config::{Flux3Config, Mode, PackageConfig, SamplerDefaults, TextEncoderConfig};
pub use pipeline::{
    Flux3Image, Flux3Options, Flux3Paths, Flux3Pipeline, Flux3Sample, ReferenceTokens,
};
pub use pup::{Upsampler, UpsamplerPaths};
pub use text::{TextEncoder, TextTokens};
