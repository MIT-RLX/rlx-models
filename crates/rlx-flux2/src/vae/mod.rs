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

pub mod config;
pub mod encoder;
pub mod flow;
pub mod forward;
pub mod hir_builder;
pub mod layers;
pub mod weights;

pub use config::Flux2VaeConfig;
pub use encoder::flux2_vae_encode;
pub use flow::{
    Flux2VaeDecoderFlow, Flux2VaeEncoderFlow, build_flux2_vae_decoder_built,
    build_flux2_vae_encoder_built,
};
pub use forward::{flux2_decode_packed_latents, flux2_rgb_to_u8, flux2_vae_decode};
pub use hir_builder::{
    Flux2VaeGraph, build_flux2_vae_encoder_hir, build_flux2_vae_hir, compile_flux2_vae_encoder_hir,
    compile_flux2_vae_hir,
};
pub use weights::{
    Flux2VaeWeights, extract_flux2_vae_weights, load_flux2_vae_weights, resolve_vae_dir,
    synthetic_vae_weights,
};
