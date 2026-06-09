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

//! Multi-step Flow-Match Euler on compiled CPU denoiser (no full runner).

mod compile_support;

use rlx_models::flux2::{
    Flux2Config, compile_flux2_forward, flow_match_euler_step, flow_match_sigmas, host_temb,
    init_latent_noise, prepare_latent_ids, synthetic_weights,
};
use rlx_models::{extract_flux2_weights, prepare_weight_map};
use rlx_runtime::Device;

#[test]
fn two_step_compiled_cpu() {
    let cfg = Flux2Config::tiny();
    let w = extract_flux2_weights(prepare_weight_map(synthetic_weights(&cfg)), &cfg).unwrap();
    let b = 1usize;
    let latent_h = 2usize;
    let latent_w = 2usize;
    let img_seq = latent_h * latent_w;
    let txt_seq = 3usize;
    let img_ids = prepare_latent_ids(b, latent_h, latent_w);
    let txt_ids = vec![0.0f32; txt_seq * 4];
    let encoder = vec![0.2f32; b * txt_seq * cfg.joint_attention_dim];
    let guidance = vec![3.5f32; b];

    let (mut compiled, _) = compile_flux2_forward(
        &cfg,
        &w,
        b,
        img_seq,
        txt_seq,
        &img_ids,
        &txt_ids,
        Device::Cpu,
        None,
        None,
        None,
    )
    .unwrap();

    let steps = 2usize;
    let mut latents = init_latent_noise(b, img_seq, cfg.in_channels, 99);
    let sigmas = flow_match_sigmas(steps);
    for i in 0..steps {
        let timestep = vec![sigmas[i]; b];
        let temb = host_temb(&w, &cfg, &timestep, Some(&guidance)).unwrap();
        let noise = compiled
            .run(&[
                ("hidden", latents.as_slice()),
                ("encoder", encoder.as_slice()),
                ("temb", temb.as_slice()),
            ])
            .remove(0);
        flow_match_euler_step(&mut latents, &noise, sigmas[i], sigmas[i + 1]);
    }
    assert_eq!(latents.len(), b * img_seq * cfg.in_channels);
}
