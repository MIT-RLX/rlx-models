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

//! FLUX.2 VAE decoder HIR on Metal (macOS + `metal` feature).

#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_models::flux2::{Flux2VaeConfig, compile_flux2_vae_hir, synthetic_vae_weights};
#[cfg(all(target_os = "macos", feature = "metal"))]
use rlx_runtime::Device;

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn vae_tiny_runs_on_metal() {
    let cfg = Flux2VaeConfig::tiny();
    let w = synthetic_vae_weights(&cfg);
    let batch = 1usize;
    let h = 4usize;
    let w_px = 4usize;
    let latents = vec![0.1f32; batch * cfg.latent_channels * h * w_px];

    let (mut compiled, _) =
        compile_flux2_vae_hir(&cfg, &w, batch, h, w_px, Device::Metal, None).unwrap();
    let out = compiled.run(&[("latents", latents.as_slice())]).remove(0);
    let up = 2usize.pow(cfg.block_out_channels.len().saturating_sub(1) as u32);
    assert_eq!(out.len(), batch * cfg.out_channels * h * up * w_px * up);
}
