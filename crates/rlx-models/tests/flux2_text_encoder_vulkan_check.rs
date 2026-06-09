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

//! FLUX.2 text encoder HIR on Vulkan ([`Device::Vulkan`], `vulkan` feature).

#[cfg(feature = "vulkan")]
use rlx_models::flux2::TINY_TEXT_ENCODER_LAYERS;
#[cfg(feature = "vulkan")]
use rlx_models::flux2::{
    compile_flux2_text_encoder_hir, synthetic_text_encoder_weights, tiny_text_encoder_config,
};
#[cfg(feature = "vulkan")]
use rlx_runtime::Device;

#[cfg(feature = "vulkan")]
#[test]
fn text_encoder_tiny_runs_on_vulkan() {
    if !rlx_runtime::is_available(Device::Vulkan) {
        eprintln!("skip: Vulkan (wgpu) not available");
        return;
    }
    let cfg = tiny_text_encoder_config();
    let w = synthetic_text_encoder_weights(&cfg);
    let batch = 1usize;
    let seq = 4usize;
    let ids_f32: Vec<f32> = (0..seq as u32).map(|x| x as f32).collect();

    let (mut compiled, _) = compile_flux2_text_encoder_hir(
        &cfg,
        &w,
        batch,
        seq,
        TINY_TEXT_ENCODER_LAYERS,
        Device::Vulkan,
    )
    .unwrap();
    let out = compiled.run(&[("input_ids", ids_f32.as_slice())]).remove(0);
    let joint = cfg.hidden_size * TINY_TEXT_ENCODER_LAYERS.len();
    assert_eq!(out.len(), batch * seq * joint);
}
