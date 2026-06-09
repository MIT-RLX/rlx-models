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

//! FLUX.2 full denoiser HIR on Vulkan ([`Device::Vulkan`], `vulkan` feature).
//!
//! Uses wgpu's Vulkan backend (MoltenVK on macOS when available).

#[cfg(feature = "vulkan")]
use rlx_models::flux2::{Flux2Config, compile_flux2_forward, host_temb, synthetic_weights};
#[cfg(feature = "vulkan")]
use rlx_models::{extract_flux2_weights, prepare_weight_map};
#[cfg(feature = "vulkan")]
use rlx_runtime::Device;

#[cfg(feature = "vulkan")]
#[test]
fn denoiser_tiny_runs_on_vulkan() {
    if !rlx_runtime::is_available(Device::Vulkan) {
        eprintln!("skip: Vulkan (wgpu) not available");
        return;
    }
    let cfg = Flux2Config::tiny();
    let w = extract_flux2_weights(prepare_weight_map(synthetic_weights(&cfg)), &cfg).unwrap();
    let b = 1usize;
    let img_seq = 4usize;
    let txt_seq = 3usize;
    let img_ids = vec![0.0f32; img_seq * 4];
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let hidden = vec![0.1f32; b * img_seq * cfg.in_channels];
    let encoder = vec![0.2f32; b * txt_seq * cfg.joint_attention_dim];
    let timestep = vec![0.5f32];
    let guidance = vec![3.5f32];

    let (mut compiled, _) = compile_flux2_forward(
        &cfg,
        &w,
        b,
        img_seq,
        txt_seq,
        &img_ids,
        &txt_ids,
        Device::Vulkan,
        None,
        None,
        None,
    )
    .unwrap();
    let temb = host_temb(&w, &cfg, &timestep, Some(&guidance)).unwrap();
    let out = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("encoder", encoder.as_slice()),
            ("temb", temb.as_slice()),
        ])
        .remove(0);
    assert_eq!(out.len(), b * img_seq * cfg.proj_out_dim());
}
