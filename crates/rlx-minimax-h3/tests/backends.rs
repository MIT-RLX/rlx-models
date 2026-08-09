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

//! Cross-backend agreement for the two compiled graphs.
//!
//! CPU is the reference. Each enabled accelerator runs the same graph on the
//! same inputs and has to land within tolerance — the DiT stacks 50 blocks, so a
//! backend that gets one op subtly wrong shows up as a large relative error long
//! before it shows up as a `NaN`.
//!
//! Backends compile in only when their feature is on, so this file is a no-op on
//! a default build.

#![allow(unused_imports)]

use rlx_minimax_h3::config::{H3TransformerConfig, H3VideoVaeConfig, Modality};
use rlx_minimax_h3::layout::{H3Geometry, build_packed_sequence, build_row_timesteps};
use rlx_minimax_h3::rope::RopeTables;
use rlx_minimax_h3::transformer::{H3DitInputs, H3DitLayout, compile_dit};
use rlx_minimax_h3::vae_video::compile_video_decoder;
use rlx_minimax_h3::weights::synthetic_dit_weights;
use rlx_runtime::Device;

/// Accelerators to compare against CPU, given the enabled features.
// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn accelerators() -> Vec<(&'static str, Device)> {
    // `mut` is used only when an accelerator feature is on.
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, Device)> = Vec::new();
    #[cfg(feature = "metal")]
    v.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    v.push(("mlx", Device::Mlx));
    #[cfg(feature = "gpu")]
    v.push(("wgpu", Device::Gpu));
    v
}

/// Largest relative difference, scaled by the reference's own magnitude so a
/// near-zero element does not dominate.
fn max_rel_diff(reference: &[f32], other: &[f32]) -> f32 {
    assert_eq!(reference.len(), other.len(), "output lengths differ");
    let scale = reference
        .iter()
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    reference
        .iter()
        .zip(other)
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs() / scale))
}

/// Cosine similarity — catches a backend that scales or permutes the output even
/// when the absolute error looks plausible.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

fn dit_cfg() -> H3TransformerConfig {
    H3TransformerConfig {
        num_attention_heads: 2,
        attention_head_dim: 16,
        hidden_size: 24,
        num_layers: 2,
        num_refiner_layers: 1,
        ffn_dim: 32,
        in_channels: 4,
        audio_in_channels: 6,
        patch_size: [1, 2, 2],
        text_dim: 8,
        freq_dim: 16,
        time_embed_hidden_dim: 24,
        time_embed_dim: 12,
        rope_freq_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        final_norm_eps: 1e-5,
    }
}

fn geometry() -> H3Geometry {
    H3Geometry {
        height: 64,
        width: 64,
        num_frames: 39,
        num_latent_frames: 3,
        latent_height: 4,
        latent_width: 4,
        num_audio_latents: 3,
    }
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 / 17.0) - 0.5).collect()
}

/// Run the DiT once on `device`, returning `(video, audio)`.
fn run_dit(device: Device) -> (Vec<f32>, Vec<f32>) {
    let cfg = dit_cfg();
    let g = geometry();
    let layout =
        build_packed_sequence(&[Modality::Text.tag(); 5], &g, cfg.patch_size, &[]).unwrap();
    let mut weights = synthetic_dit_weights(&cfg, 31);
    let mut dit = compile_dit(
        &cfg,
        &mut weights,
        device,
        layout.sequence_length(),
        layout.text_indices.len(),
        layout.video_indices.len(),
        layout.audio_indices.len(),
    )
    .unwrap_or_else(|e| panic!("compile DiT on {device:?}: {e}"));

    let rows = build_row_timesteps(&layout, 0.3, 0.4, 0.999, 1.0).unwrap();
    let dl = H3DitLayout::new(&layout, &rows, &cfg).unwrap();
    let tables = RopeTables::build(
        &layout.flat_position_ids(),
        cfg.rope_freq_dim,
        cfg.rope_theta,
    )
    .unwrap();

    let video = ramp(layout.video_indices.len() * cfg.video_patch_dim());
    let audio = ramp(layout.audio_indices.len() * cfg.audio_in_channels);
    let text = ramp(layout.text_indices.len() * cfg.text_dim);

    let out = dit
        .forward(&H3DitInputs {
            video_rows: &video,
            audio_rows: &audio,
            text_rows: &text,
            cos: &tables.cos,
            sin: &tables.sin,
            layout: &dl,
        })
        .unwrap_or_else(|e| panic!("DiT forward on {device:?}: {e}"));
    (out.video, out.audio)
}

#[test]
fn dit_agrees_across_backends() {
    let accel = accelerators();
    if accel.is_empty() {
        eprintln!("skipping: no accelerator features enabled");
        return;
    }
    let (ref_video, ref_audio) = run_dit(Device::Cpu);
    assert!(ref_video.iter().all(|v| v.is_finite()));

    for (name, device) in accel {
        let (video, audio) = run_dit(device);
        assert!(
            video.iter().all(|v| v.is_finite()),
            "{name}: video velocity has non-finite values"
        );
        assert!(
            audio.iter().all(|v| v.is_finite()),
            "{name}: audio velocity has non-finite values"
        );
        let (dv, da) = (
            max_rel_diff(&ref_video, &video),
            max_rel_diff(&ref_audio, &audio),
        );
        let (cv, ca) = (cosine(&ref_video, &video), cosine(&ref_audio, &audio));
        eprintln!("{name}: video rel {dv:.2e} cos {cv:.6} | audio rel {da:.2e} cos {ca:.6}");
        // Every backend lands at ~1e-7 here. A loose gate is worse than none:
        // partial RoPE was wrong on Metal and wgpu at 1.8e-4 and still passed a
        // `cos > 0.9999` check, so the bound has to be tight enough to notice.
        assert!(cv > 0.999_999, "{name}: video cosine {cv} against CPU");
        assert!(ca > 0.999_999, "{name}: audio cosine {ca} against CPU");
        assert!(
            dv < 1e-5,
            "{name}: video max relative diff {dv} against CPU"
        );
        assert!(
            da < 1e-5,
            "{name}: audio max relative diff {da} against CPU"
        );
    }
}

fn vae_cfg() -> H3VideoVaeConfig {
    serde_json::from_str(
        r#"{"in_channels":3,"out_channels":3,"latent_channels":8,
            "block_out_channels":[16,32],"layers_per_block":1,
            "spatial_downsample_factors":[2,1],
            "temporal_downsample_factors":[1,1],
            "norm_num_groups":4,"norm_eps":1e-06,"spatial_padding_mode":"reflect",
            "decoder_num_layers":2,"decoder_num_attention_heads":2,
            "decoder_attention_head_dim":16,"decoder_num_register_tokens":2,
            "decoder_ffn_mult":2,"decoder_rope_theta":100.0,
            "decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-05,
            "clip_length":17,"token_drop":3}"#,
    )
    .unwrap()
}

/// Deterministic synthetic weights for the scaled-down ViT decoder.
fn vae_weights(cfg: &H3VideoVaeConfig) -> rlx_core::weight_map::WeightMap {
    use std::collections::HashMap;
    let dim = cfg.decoder_hidden_size();
    let ffn = cfg.decoder_ffn_mult * dim;
    let patch = cfg.out_channels * 4 * 16 * 16;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut put = |k: &str, shape: Vec<usize>, seed: usize| {
        let n: usize = shape.iter().product();
        let fan = *shape.last().unwrap_or(&1) as f32;
        let data = (0..n)
            .map(|i| (((i * 37 + seed * 101) % 211) as f32 / 211.0 - 0.5) / fan.sqrt())
            .collect();
        t.insert(k.to_string(), (data, shape));
    };
    // The decode path starts at `post_quant_conv`, not `proj_in`.
    put(
        "post_quant_conv.weight",
        vec![cfg.latent_channels, cfg.latent_channels],
        0,
    );
    put("post_quant_conv.bias", vec![cfg.latent_channels], 9);
    put("decoder.proj_in.weight", vec![dim, cfg.latent_channels], 1);
    put("decoder.proj_in.bias", vec![dim], 2);
    put(
        "decoder.register_tokens",
        vec![1, cfg.decoder_num_register_tokens, dim],
        3,
    );
    put("decoder.norm_out.weight", vec![dim], 4);
    put("decoder.norm_out.bias", vec![dim], 5);
    put("decoder.proj_out.weight", vec![patch, dim], 6);
    put("decoder.proj_out.bias", vec![patch], 7);
    for b in 0..cfg.decoder_num_layers {
        let p = format!("decoder.transformer_blocks.{b}");
        put(&format!("{p}.norm1.weight"), vec![dim], 10 + b);
        put(&format!("{p}.norm2.weight"), vec![dim], 11 + b);
        put(&format!("{p}.scale1"), vec![dim], 12 + b);
        put(&format!("{p}.scale2"), vec![dim], 13 + b);
        for n in ["to_q", "to_k", "to_v"] {
            put(&format!("{p}.attn.{n}.weight"), vec![dim, dim], 14 + b);
            put(&format!("{p}.attn.{n}.bias"), vec![dim], 15 + b);
        }
        put(&format!("{p}.attn.to_out.0.weight"), vec![dim, dim], 16 + b);
        put(&format!("{p}.attn.to_out.0.bias"), vec![dim], 17 + b);
        put(
            &format!("{p}.ff.net.0.proj.weight"),
            vec![2 * ffn, dim],
            18 + b,
        );
        put(&format!("{p}.ff.net.0.proj.bias"), vec![2 * ffn], 19 + b);
        put(&format!("{p}.ff.net.2.weight"), vec![dim, ffn], 20 + b);
        put(&format!("{p}.ff.net.2.bias"), vec![dim], 21 + b);
    }
    rlx_core::weight_map::WeightMap::from_tensors(t)
}

fn run_vae_decoder(device: Device) -> Vec<f32> {
    let cfg = vae_cfg();
    let (frames, h, w) = (2usize, 2usize, 2usize);
    let mut weights = vae_weights(&cfg);
    let mut dec = compile_video_decoder(&cfg, &mut weights, device, frames, h, w)
        .unwrap_or_else(|e| panic!("compile video decoder on {device:?}: {e}"));
    let latents = ramp(cfg.latent_channels * frames * h * w);
    dec.decode_chunk(&latents)
        .unwrap_or_else(|e| panic!("decode on {device:?}: {e}"))
}

#[test]
fn video_decoder_agrees_across_backends() {
    let accel = accelerators();
    if accel.is_empty() {
        eprintln!("skipping: no accelerator features enabled");
        return;
    }
    let reference = run_vae_decoder(Device::Cpu);
    assert!(reference.iter().all(|v| v.is_finite()));

    for (name, device) in accel {
        let got = run_vae_decoder(device);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}: decoder produced non-finite pixels"
        );
        let d = max_rel_diff(&reference, &got);
        let c = cosine(&reference, &got);
        eprintln!("{name}: video decoder rel {d:.2e} cos {c:.6}");
        assert!(c > 0.999_999, "{name}: decoder cosine {c} against CPU");
        assert!(
            d < 1e-5,
            "{name}: decoder max relative diff {d} against CPU"
        );
    }
}

/// The tapped text stack, which uses a **full** rotation rather than a partial
/// one — included so a future change to its RoPE path cannot regress silently.
#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]
fn run_text_encoder(device: Device) -> Vec<f32> {
    use rlx_minimax_h3::config::H3TextEncoderConfig;
    use rlx_minimax_h3::qwen3vl::{compile_text_encoder, synthetic_weights};
    let cfg = H3TextEncoderConfig {
        hidden_size: 32,
        num_hidden_layers: 64,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 5e6,
        vocab_size: 64,
        mrope_section: [2, 1, 1],
        mrope_interleaved: true,
    };
    let mut w = synthetic_weights(&cfg, 41);
    let mut enc = compile_text_encoder(&cfg, &mut w, device, 5)
        .unwrap_or_else(|e| panic!("compile text encoder on {device:?}: {e}"));
    enc.encode_tokens(&[3, 11, 19, 27, 35])
        .unwrap_or_else(|e| panic!("encode on {device:?}: {e}"))
        .hidden
}

#[test]
#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]
fn text_encoder_agrees_across_backends() {
    let accel = accelerators();
    if accel.is_empty() {
        eprintln!("skipping: no accelerator features enabled");
    } else {
        let reference = run_text_encoder(Device::Cpu);
        assert!(reference.iter().all(|v| v.is_finite()));
        for (name, device) in accel {
            let got = run_text_encoder(device);
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{name}: text encoder produced non-finite conditioning"
            );
            let d = max_rel_diff(&reference, &got);
            let c = cosine(&reference, &got);
            eprintln!("{name}: text encoder rel {d:.2e} cos {c:.6}");
            assert!(c > 0.999_99, "{name}: text encoder cosine {c} against CPU");
            assert!(
                d < 1e-4,
                "{name}: text encoder max relative diff {d} against CPU"
            );
        }
    }
}
