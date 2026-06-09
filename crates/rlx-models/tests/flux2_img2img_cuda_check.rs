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

//! FLUX.2 img2img / edit conditioning + CUDA denoiser concat forward (`cuda` feature).

#[cfg(feature = "cuda")]
use rlx_models::flux2::{
    Flux2Config, compile_flux2_forward, concat_latent_ids, concat_packed_latents, host_temb,
    init_latent_noise, prepare_latent_ids, prepare_latent_ids_with_t,
    prepare_reference_conditioning, slice_gen_noise, synthetic_weights,
};
use rlx_models::flux2::{
    Flux2VaeConfig, flow_match_init_timestep, flux2_latent_geometry, flux2_vae_encode,
    prepare_img2img_latents, synthetic_vae_weights,
};
#[cfg(feature = "cuda")]
use rlx_models::{extract_flux2_weights, prepare_weight_map};
#[cfg(feature = "cuda")]
use rlx_runtime::Device;

#[test]
fn vae_encode_native_runs() {
    let vae_cfg = Flux2VaeConfig::tiny();
    let vae = synthetic_vae_weights(&vae_cfg);
    let batch = 1usize;
    let pixel_h = 32usize;
    let pixel_w = 32usize;
    let rgb: Vec<f32> = (0..batch * 3 * pixel_h * pixel_w)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let encoded = flux2_vae_encode(&vae, &vae_cfg, &rgb, batch, pixel_h, pixel_w).unwrap();
    let stride = vae_cfg.encode_spatial_stride();
    let enc_h = pixel_h / stride;
    let enc_w = pixel_w / stride;
    assert_eq!(
        encoded.len(),
        batch * vae_cfg.latent_channels * enc_h * enc_w
    );
}

#[test]
fn img2img_blend_produces_packed() {
    let vae_cfg = Flux2VaeConfig::tiny();
    let vae = synthetic_vae_weights(&vae_cfg);
    let batch = 1usize;
    let pixel_h = 32usize;
    let pixel_w = 32usize;
    let (latent_h, latent_w, eff_h, eff_w) = flux2_latent_geometry(pixel_h, pixel_w);
    let gen_seq = latent_h * latent_w;
    let channels = vae_cfg.bn_channels();
    let rgb: Vec<f32> = (0..batch * 3 * pixel_h * pixel_w)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let noise = vec![0.5f32; batch * gen_seq * channels];
    let steps = 20usize;
    let strength = 0.75f32;
    let blended = prepare_img2img_latents(
        &vae, &vae_cfg, &rgb, batch, pixel_h, pixel_w, latent_h, latent_w, eff_h, eff_w, &noise,
        strength, steps,
    )
    .unwrap();
    assert_eq!(blended.len(), noise.len());
    assert_eq!(flow_match_init_timestep(strength, steps), 15);
    assert_ne!(blended, noise);
}

#[cfg(feature = "cuda")]
#[test]
fn edit_denoiser_concat_forward_on_cuda() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("skip: CUDA not available");
        return;
    }

    let vae_cfg = Flux2VaeConfig::tiny();
    let vae = synthetic_vae_weights(&vae_cfg);
    let batch = 1usize;
    let pixel_h = 32usize;
    let pixel_w = 32usize;
    let (latent_h, latent_w, eff_h, eff_w) = flux2_latent_geometry(pixel_h, pixel_w);
    let gen_seq = latent_h * latent_w;
    let rgb: Vec<f32> = (0..batch * 3 * pixel_h * pixel_w)
        .map(|i| (i as f32 * 0.002).cos())
        .collect();
    let refs = [(&rgb[..], pixel_h, pixel_w)];
    let reference = prepare_reference_conditioning(
        &vae, &vae_cfg, &refs, batch, eff_h, eff_w, latent_h, latent_w,
    )
    .unwrap();
    assert_eq!(reference.seq, gen_seq);

    let cfg = Flux2Config::tiny();
    let w = extract_flux2_weights(prepare_weight_map(synthetic_weights(&cfg)), &cfg).unwrap();
    let txt_seq = 3usize;
    let gen_ids = prepare_latent_ids(batch, latent_h, latent_w);
    let img_ids = concat_latent_ids(&gen_ids, &reference.img_ids, batch);
    let total_seq = gen_seq + reference.seq;
    assert_eq!(img_ids.len(), batch * total_seq * 4);

    let (mut compiled, _) = compile_flux2_forward(
        &cfg,
        &w,
        batch,
        total_seq,
        txt_seq,
        &img_ids,
        &vec![0.0f32; txt_seq * 4],
        Device::Cuda,
        None,
        None,
        None,
    )
    .unwrap();

    let latents = init_latent_noise(batch, gen_seq, cfg.in_channels, 7);
    let hidden = concat_packed_latents(&latents, &reference.packed, batch, cfg.in_channels);
    let encoder = vec![0.2f32; batch * txt_seq * cfg.joint_attention_dim];
    let timestep = vec![0.5f32; batch];
    let guidance = vec![3.5f32; batch];
    let temb = host_temb(&w, &cfg, &timestep, Some(&guidance)).unwrap();

    let noise = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("encoder", encoder.as_slice()),
            ("temb", temb.as_slice()),
        ])
        .remove(0);
    assert_eq!(noise.len(), batch * total_seq * cfg.proj_out_dim());

    let gen_noise = slice_gen_noise(&noise, batch, cfg.proj_out_dim(), gen_seq);
    assert_eq!(gen_noise.len(), batch * gen_seq * cfg.proj_out_dim());

    let ref_ids = prepare_latent_ids_with_t(batch, latent_h, latent_w, 10);
    assert_eq!(&reference.img_ids, &ref_ids);
}
