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

//! FLUX.2 (Black Forest Labs) rectified-flow image denoiser transformer.
//!
//! Phase 1 ships the denoiser trunk: config parsing, BFL/NVFP4 weight
//! adaptation, native CPU forward, and compiled HIR on GPU backends.
//! **Backends** (via [`Flux2Runner`] / `rlx-flux2 --device …`):
//! `cpu` (native), `metal`/`mps`, `mlx`, `cuda`, `rocm`/`hip`, `gpu`/`wgpu`, `vulkan`.
//! Enable matching `rlx-flux2` / `rlx-models` features (`cuda`, `metal`, …) or `nvidia-gpu` for CUDA.
//! **Compile policy:** denoiser + VAE use compiled HIR on non-CPU backends; text encoder
//! uses compiled HIR on **Metal/MLX only** — CUDA/ROCm/wgpu use native CPU encode once, then
//! drop TE weights before denoiser compile. VAE / text-encoder / scheduler pipeline wiring follows.

pub mod adapt;
pub mod builder;
pub mod cfg;
pub mod cli;
pub mod compile_util;
pub mod conditioning;
pub mod config;
pub mod device;
pub mod diamond;
pub mod download;
pub mod flow;
pub mod forward;
pub mod hir_builder;
pub mod image;
pub mod latent_ops;
pub mod layers;
pub mod lora;
pub mod packed;
pub mod packed_gguf;
pub mod paths;
pub mod pipeline;
pub mod rope;
pub mod runner;
pub mod scheduler;
pub mod session;
pub mod text_encoder;
pub mod typed_linear;
pub mod vae;
pub mod weights;

pub use adapt::{adapt_bfl_weights, normalize_flux2_key, prepare_weight_map, split_stacked_qkv};
pub use builder::{
    Flux2GraphParams, build_flux2_minimal_graph, build_flux2_minimal_hir, compile_flux2_minimal,
};
pub use cfg::{
    Flux2CfgCombineGraph, build_flux2_cfg_combine_built, build_flux2_cfg_combine_hir, cfg_combine,
    compile_flux2_cfg_combine, emit_flux2_cfg_combine,
};
pub use compile_util::{
    aot_cache_from_dir, compile_hir_cached, flux2_cfg_aot_key, flux2_compile_profile,
    flux2_denoiser_aot_key, flux2_text_encoder_aot_key, flux2_vae_decoder_aot_key,
    flux2_vae_encoder_aot_key,
};
pub use conditioning::{
    Flux2ReferenceConditioning, encode_rgb_to_packed, pack_encoded_latents, prepare_generation_ids,
    prepare_img2img_latents, prepare_reference_conditioning,
};
pub use config::Flux2Config;
pub use device::{
    assert_flux2_device_available, flux2_device_feature, flux2_prefers_compiled_hir,
    flux2_prefers_compiled_te,
};
pub use diamond::{
    DiamondGuidanceParams, DiamondMethod, FLOW_MAP_LORA_HF_REPO, FLOW_MAP_LORA_HF_WEIGHT,
    FlowMapPrediction, FluxGlassReference, HybridLatentDecodeReward, apply_dps_guidance_step,
    apply_weighted_guidance_step, flow_map_predict, glass_posterior_sample,
    sample_rectified_flow_diamond,
};
pub use download::{Flux2Checkpoint, download_flux2_repo, resolve_flux2_checkpoint};
pub use flow::{
    Flux2CfgCombineFlow, Flux2Flow, Flux2ForwardBuilt, build_flux2_minimal_built,
    compile_flux2_forward_via_flow,
};
pub use forward::{Flux2ForwardInput, flux2_transformer_forward};
pub use hir_builder::host_temb_dual;
pub use hir_builder::{
    Flux2ForwardGraph, build_flux2_dual_section_hir, build_flux2_forward_graph,
    build_flux2_forward_hir, compile_flux2_forward, host_temb,
};
pub use image::load_rgb_planar;
pub use latent_ops::{
    concat_latent_ids, concat_packed_latents, denorm_patchified_latents, flux2_latent_geometry,
    pack_latents, prepare_latent_ids, prepare_latent_ids_with_t, slice_gen_noise,
    unpack_latents_with_ids, unpatchify_latents,
};
pub use lora::{
    apply_flux2_lora, load_and_apply_flux2_lora, load_and_apply_flux2_lora_dir, parse_lora_scale,
};
pub use packed::{
    Flux2GgufLinearPacked, Flux2PackedParams, Nvfp4LinearPacked, load_flux2_nvfp4_from_file,
    safetensors_has_nvfp4, synthetic_flux2_packed_tiny,
};
pub use packed_gguf::{gguf_has_packed_linears, load_flux2_from_gguf};
pub use paths::{
    find_component_dir, find_tokenizer_json, find_transformer_config, resolve_transformer_config,
};
pub use pipeline::{
    Flux2SampleOutput, Flux2SampleParams, generate_to_rgb, init_latent_noise,
    sample_rectified_flow, write_ppm,
};
pub use rlx_diamond::{BluenessReward, LatentReward, LinearMeasurementReward};
pub use runner::{Flux2Output, Flux2Runner, Flux2RunnerBuilder};
pub use scheduler::{flow_match_euler_step, flow_match_init_timestep, flow_match_sigmas};
pub use session::{Flux2Session, Flux2SessionCache, Flux2SessionKey};
pub use text_encoder::{
    DEFAULT_TEXT_ENCODER_LAYERS, Flux2PromptOutput, Flux2TextEncoderBuilt, Flux2TextEncoderFlow,
    Flux2TextEncoderGraph, Flux2TextEncoderWeights, TINY_TEXT_ENCODER_LAYERS,
    build_flux2_text_encoder_built, build_flux2_text_encoder_hir, compile_flux2_text_encoder_hir,
    encode_flux2_prompt, encode_prompt_embeds, encode_prompt_embeds_default_layers,
    encode_prompt_padded, extract_text_encoder_weights, load_text_encoder_weights,
    prepare_text_ids, resolve_text_encoder_dir, resolve_tokenizer_path,
    synthetic_text_encoder_weights, tiny_text_encoder_config,
};
pub use typed_linear::{TypedLinearStore, load_typed_linears_from_file};
pub use vae::{
    Flux2VaeConfig, Flux2VaeDecoderFlow, Flux2VaeEncoderFlow, Flux2VaeGraph, Flux2VaeWeights,
    build_flux2_vae_decoder_built, build_flux2_vae_encoder_built, build_flux2_vae_encoder_hir,
    build_flux2_vae_hir, compile_flux2_vae_encoder_hir, compile_flux2_vae_hir,
    extract_flux2_vae_weights, flux2_decode_packed_latents, flux2_rgb_to_u8, flux2_vae_decode,
    flux2_vae_encode, load_flux2_vae_weights, resolve_vae_dir, synthetic_vae_weights,
};
pub use weights::{
    ExtractFlux2Opts, Flux2Weights, extract_flux2_weights, extract_flux2_weights_with_opts,
    load_flux2_weight_map, load_flux2_weights,
};

fn insert_linear(
    t: &mut std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    name: &str,
    out_d: usize,
    in_d: usize,
) {
    t.insert(
        format!("{name}.weight"),
        (vec![0.0f32; out_d * in_d], vec![out_d, in_d]),
    );
    t.insert(format!("{name}.bias"), (vec![0.0f32; out_d], vec![out_d]));
}

/// Build a zero-initialized weight map for unit tests / quick runs.
pub fn synthetic_weights(cfg: &Flux2Config) -> rlx_core::weight_map::WeightMap {
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    {
        let d = cfg.inner_dim();
        let ff = cfg.ff_inner_dim();
        let in_ch = cfg.in_channels;
        let joint = cfg.joint_attention_dim;
        let ch = cfg.timestep_guidance_channels;
        let out = cfg.proj_out_dim();

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        insert_linear(&mut t, "x_embedder", d, in_ch);
        insert_linear(&mut t, "context_embedder", d, joint);
        insert_linear(
            &mut t,
            "time_guidance_embed.timestep_embedder.linear_1",
            d,
            ch,
        );
        insert_linear(
            &mut t,
            "time_guidance_embed.timestep_embedder.linear_2",
            d,
            d,
        );
        if cfg.guidance_embeds {
            insert_linear(
                &mut t,
                "time_guidance_embed.guidance_embedder.linear_1",
                d,
                ch,
            );
            insert_linear(
                &mut t,
                "time_guidance_embed.guidance_embedder.linear_2",
                d,
                d,
            );
        }
        insert_linear(&mut t, "double_stream_modulation_img.linear", 6 * d, d);
        insert_linear(&mut t, "double_stream_modulation_txt.linear", 6 * d, d);
        insert_linear(&mut t, "single_stream_modulation.linear", 3 * d, d);

        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}.attn");
            insert_linear(&mut t, &format!("{p}.to_q"), d, d);
            insert_linear(&mut t, &format!("{p}.to_k"), d, d);
            insert_linear(&mut t, &format!("{p}.to_v"), d, d);
            insert_linear(&mut t, &format!("{p}.add_q_proj"), d, d);
            insert_linear(&mut t, &format!("{p}.add_k_proj"), d, d);
            insert_linear(&mut t, &format!("{p}.add_v_proj"), d, d);
            insert_linear(&mut t, &format!("{p}.to_out.0"), d, d);
            insert_linear(&mut t, &format!("{p}.to_add_out"), d, d);
            let hd = cfg.attention_head_dim;
            for suffix in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                t.insert(format!("{p}.{suffix}.weight"), (vec![0.0f32; hd], vec![hd]));
            }
            insert_linear(
                &mut t,
                &format!("transformer_blocks.{i}.ff.linear_in"),
                2 * ff,
                d,
            );
            insert_linear(
                &mut t,
                &format!("transformer_blocks.{i}.ff.linear_out"),
                d,
                ff,
            );
            insert_linear(
                &mut t,
                &format!("transformer_blocks.{i}.ff_context.linear_in"),
                2 * ff,
                d,
            );
            insert_linear(
                &mut t,
                &format!("transformer_blocks.{i}.ff_context.linear_out"),
                d,
                ff,
            );
        }

        let mlp_h = ff;
        for i in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{i}.attn");
            insert_linear(
                &mut t,
                &format!("{p}.to_qkv_mlp_proj"),
                3 * d + 2 * mlp_h,
                d,
            );
            insert_linear(&mut t, &format!("{p}.to_out"), d, d + mlp_h);
            let hd = cfg.attention_head_dim;
            for suffix in ["norm_q", "norm_k"] {
                t.insert(format!("{p}.{suffix}.weight"), (vec![0.0f32; hd], vec![hd]));
            }
        }

        insert_linear(&mut t, "norm_out.linear", 2 * d, d);
        insert_linear(&mut t, "proj_out", out, d);
        WeightMap::from_tensors(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_native_forward_runs() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let b = 1usize;
        let img_seq = 4usize;
        let txt_seq = 3usize;
        let hidden = vec![0.1f32; b * img_seq * cfg.in_channels];
        let encoder = vec![0.2f32; b * txt_seq * cfg.joint_attention_dim];
        let timestep = vec![0.5f32; b];
        let guidance = vec![3.5f32; b];
        let img_ids = vec![0.0f32; img_seq * 4];
        let txt_ids = vec![0.0f32; txt_seq * 4];
        let out = flux2_transformer_forward(
            &w,
            &cfg,
            Flux2ForwardInput {
                hidden_states: &hidden,
                encoder_hidden_states: &encoder,
                timestep: &timestep,
                timestep_target: None,
                guidance: Some(&guidance),
                img_ids: &img_ids,
                txt_ids: &txt_ids,
                batch: b,
                img_seq,
                txt_seq,
            },
        )
        .unwrap();
        assert_eq!(out.len(), b * img_seq * cfg.proj_out_dim());
    }

    #[test]
    fn minimal_graph_builds() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let w = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let (g, _) = build_flux2_minimal_graph(&cfg, &w, 1, 4).unwrap();
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn flux2_klein_9b_config_dims() {
        let cfg = Flux2Config::flux2_klein_9b();
        assert_eq!(cfg.inner_dim(), 4096);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.num_single_layers, 24);
        assert_eq!(cfg.joint_attention_dim, 12288);
        assert!(!cfg.guidance_embeds);
    }

    #[test]
    fn flux2_dev_config_dims() {
        let cfg = Flux2Config::flux2_dev();
        assert_eq!(cfg.inner_dim(), 6144);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.num_single_layers, 48);
    }

    #[test]
    fn cfg_combine_native() {
        let pos = vec![1.0f32, 2.0, 3.0];
        let neg = vec![0.0f32, 1.0, 2.0];
        let out = super::cfg::cfg_combine(&pos, &neg, 2.0);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn vae_tiny_decode_runs() {
        use super::vae::{Flux2VaeConfig, flux2_vae_decode, synthetic_vae_weights};
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let batch = 1usize;
        let h = 4usize;
        let w_px = 4usize;
        let latents = vec![0.1f32; batch * cfg.latent_channels * h * w_px];
        let rgb = flux2_vae_decode(&w, &cfg, &latents, batch, h, w_px).unwrap();
        let up = 2usize.pow(cfg.block_out_channels.len().saturating_sub(1) as u32);
        assert_eq!(rgb.len(), batch * cfg.out_channels * h * up * w_px * up);
    }

    #[test]
    fn text_encoder_tiny_produces_joint_dim() {
        use super::text_encoder::{
            TINY_TEXT_ENCODER_LAYERS, encode_flux2_prompt, synthetic_text_encoder_weights,
            tiny_text_encoder_config,
        };
        let te_cfg = tiny_text_encoder_config();
        let te = synthetic_text_encoder_weights(&te_cfg);
        let batch = 1usize;
        let seq = 4usize;
        let ids: Vec<u32> = (0..seq as u32).collect();
        let (out, txt_ids) =
            encode_flux2_prompt(&te, &te_cfg, &ids, batch, seq, TINY_TEXT_ENCODER_LAYERS).unwrap();
        assert_eq!(
            out.joint_dim,
            te_cfg.hidden_size * TINY_TEXT_ENCODER_LAYERS.len()
        );
        assert_eq!(out.prompt_embeds.len(), batch * seq * out.joint_dim);
        assert_eq!(txt_ids.len(), batch * seq * 4);
    }

    #[test]
    fn img2img_latent_blend_matches_geometry() {
        use super::conditioning::prepare_img2img_latents;
        use super::latent_ops::flux2_latent_geometry;
        use super::scheduler::flow_match_init_timestep;
        use super::vae::{Flux2VaeConfig, synthetic_vae_weights};

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
            &vae, &vae_cfg, &rgb, batch, pixel_h, pixel_w, latent_h, latent_w, eff_h, eff_w,
            &noise, strength, steps,
        )
        .unwrap();
        assert_eq!(blended.len(), noise.len());
        assert_eq!(flow_match_init_timestep(strength, steps), 15);
        assert_ne!(blended, noise);
    }

    #[test]
    fn edit_reference_conditioning_concat() {
        use super::conditioning::prepare_reference_conditioning;
        use super::latent_ops::{flux2_latent_geometry, prepare_latent_ids_with_t};
        use super::vae::{Flux2VaeConfig, synthetic_vae_weights};

        let vae_cfg = Flux2VaeConfig::tiny();
        let vae = synthetic_vae_weights(&vae_cfg);
        let batch = 1usize;
        let pixel_h = 32usize;
        let pixel_w = 32usize;
        let (latent_h, latent_w, eff_h, eff_w) = flux2_latent_geometry(pixel_h, pixel_w);
        let gen_seq = latent_h * latent_w;
        let channels = vae_cfg.bn_channels();
        let rgb: Vec<f32> = (0..batch * 3 * pixel_h * pixel_w)
            .map(|i| (i as f32 * 0.002).cos())
            .collect();
        let refs = [(&rgb[..], pixel_h, pixel_w), (&rgb[..], pixel_h, pixel_w)];
        let cond = prepare_reference_conditioning(
            &vae, &vae_cfg, &refs, batch, eff_h, eff_w, latent_h, latent_w,
        )
        .unwrap();
        assert_eq!(cond.seq, 2 * gen_seq);
        assert_eq!(cond.packed.len(), batch * cond.seq * channels);
        assert_eq!(cond.img_ids.len(), batch * cond.seq * 4);
        let id0 = prepare_latent_ids_with_t(batch, latent_h, latent_w, 10);
        let id1 = prepare_latent_ids_with_t(batch, latent_h, latent_w, 20);
        assert_eq!(&cond.img_ids[..id0.len()], &id0);
        assert_eq!(&cond.img_ids[id0.len()..], &id1);
    }
}
