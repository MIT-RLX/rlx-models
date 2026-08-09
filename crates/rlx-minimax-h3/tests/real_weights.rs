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

//! Checks against a real MiniMax-H3 checkpoint.
//!
//! Point `RLX_MINIMAX_H3` at a checkpoint root to enable these; they skip
//! otherwise. Only the two VAEs and the component configs need to be present —
//! the 28 GB DiT partitions and the ~60 GB text encoder are not required.
//!
//! ```bash
//! RLX_MINIMAX_H3=/path/to/MiniMax-H3 cargo test -p rlx-minimax-h3 \
//!     --release --test real_weights -- --nocapture
//! ```

use rlx_minimax_h3::config::{H3AudioVaeConfig, H3Config, H3TransformerConfig, H3VideoVaeConfig};
use rlx_minimax_h3::vae_audio::{H3AudioDecoder, Signal, denormalize_latents};
use std::path::PathBuf;

fn root() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("RLX_MINIMAX_H3").ok()?);
    p.is_dir().then_some(p)
}

macro_rules! checkpoint {
    () => {
        match root() {
            Some(r) => r,
            None => {
                eprintln!("skipping: set RLX_MINIMAX_H3 to a checkpoint root");
                return;
            }
        }
    };
}

#[test]
fn component_configs_parse_and_agree() {
    let root = checkpoint!();
    let cfg = H3Config::from_root(&root).expect("parse component configs");

    assert_eq!(cfg.transformer.hidden_size, 5376);
    assert_eq!(cfg.transformer.num_layers, 50);
    assert_eq!(cfg.transformer.inner_dim(), 7168);
    assert_eq!(cfg.transformer.video_patch_dim(), 96);
    assert_eq!(cfg.transformer.rope_rotary_dim(), 96);
    assert_eq!(cfg.vae.spatial_compression(), 16);
    assert_eq!(cfg.vae.temporal_compression(), 4);
    assert_eq!(cfg.audio_vae.sampling_rate, 32_000);
    assert_eq!(cfg.audio_vae.encoder_hop(), 800);
    assert_eq!(cfg.audio_vae.decoder_hop(), 800);
    assert_eq!(cfg.text_encoder.hidden_size, 5120);
    assert_eq!(cfg.text_encoder.num_hidden_layers, 64);

    // The Ref2VA partition ships the same architecture.
    let r = cfg
        .transformer_ref
        .as_ref()
        .expect("transformer_ref config");
    assert_eq!(r.hidden_size, cfg.transformer.hidden_size);
    assert_eq!(r.num_layers, cfg.transformer.num_layers);

    // The two schedules differ only in their shift.
    assert_eq!(cfg.scheduler.shift, 12.0);
    assert_eq!(cfg.audio_scheduler.shift, 3.0);

    // Cross-component agreement is what `validate` enforces.
    cfg.validate().expect("component configs must agree");
}

#[test]
fn dit_index_covers_every_expected_parameter() {
    let root = checkpoint!();
    let index = root.join("transformer/diffusion_pytorch_model.safetensors.index.json");
    if !index.is_file() {
        eprintln!("skipping: {} not present", index.display());
        return;
    }
    let raw = std::fs::read_to_string(&index).expect("read index");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
    let map = json["weight_map"].as_object().expect("weight_map");

    let cfg = H3TransformerConfig::from_dir(&root.join("transformer")).expect("transformer config");
    let expected = rlx_minimax_h3::weights::dit_parameter_keys(&cfg);

    let have: std::collections::HashSet<&str> = map.keys().map(String::as_str).collect();
    let missing: Vec<&String> = expected
        .iter()
        .filter(|k| !have.contains(k.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "{} expected DiT key(s) absent from the checkpoint, first few: {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );
    assert_eq!(
        expected.len(),
        map.len(),
        "the checkpoint holds {} tensors, the port expects {}",
        map.len(),
        expected.len()
    );
}

#[test]
fn audio_vae_decodes_real_weights_to_audio_range() {
    let root = checkpoint!();
    let cfg = H3AudioVaeConfig::from_dir(&root.join("audio_vae")).expect("audio vae config");
    let weights = rlx_minimax_h3::weights::load_audio_vae(&root).expect("load audio vae");
    let decoder = H3AudioDecoder::load(&cfg, &weights).expect("build audio decoder");

    // A short latent: 4 frames -> 4 * 800 = 3200 samples at 32 kHz.
    let frames = 4usize;
    let mut latents = Signal::new(cfg.latent_channels, frames);
    for c in 0..cfg.latent_channels {
        for t in 0..frames {
            latents.data[c * frames + t] = ((c * 7 + t * 3) % 11) as f32 / 11.0 - 0.5;
        }
    }
    denormalize_latents(&mut latents, &cfg).expect("denormalize");

    let wave = decoder.decode_mono(&latents).expect("decode");
    assert_eq!(
        wave.len(),
        frames * cfg.decoder_hop(),
        "one latent frame must expand to {} samples",
        cfg.decoder_hop()
    );
    assert!(
        wave.iter().all(|v| v.is_finite()),
        "decoder produced non-finite audio"
    );
    // The decoder ends on tanh, so the waveform is bounded.
    assert!(
        wave.iter().all(|v| v.abs() <= 1.0),
        "decoder output left the [-1, 1] range"
    );
    assert!(
        wave.iter().any(|v| v.abs() > 1e-4),
        "decoder produced silence from a non-trivial latent"
    );
    let peak = wave.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    eprintln!("decoded {} samples, peak {peak:.4}", wave.len());
}

#[test]
fn video_vae_index_matches_the_configured_architecture() {
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let index = root.join("vae/diffusion_pytorch_model.safetensors.index.json");
    let raw = std::fs::read_to_string(&index).expect("read vae index");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse vae index");
    let map = json["weight_map"].as_object().expect("weight_map");

    // The decoder is a ViT of `decoder_num_layers` blocks.
    for b in 0..cfg.decoder_num_layers {
        for k in [
            "attn.to_q.weight",
            "attn.to_k.weight",
            "attn.to_v.weight",
            "attn.to_out.0.weight",
            "norm1.weight",
            "norm2.weight",
            "scale1",
            "scale2",
        ] {
            let key = format!("decoder.transformer_blocks.{b}.{k}");
            assert!(map.contains_key(&key), "missing {key}");
        }
    }
    assert!(map.contains_key("decoder.register_tokens"));

    // The encoder is a 3D CNN with `layers_per_block` resnets per stage.
    let stages = cfg.block_out_channels.len();
    for s in 0..stages - 1 {
        for r in 0..cfg.layers_per_block {
            let key = format!("encoder.down_blocks.{s}.resnets.{r}.conv1.weight");
            assert!(map.contains_key(&key), "missing {key}");
        }
    }
}

#[test]
fn video_vae_decoder_runs_on_real_weights() {
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let mut weights =
        rlx_minimax_h3::weights::load_video_vae_decoder(&root, &cfg).expect("load vae decoder");

    // One latent frame on a 2x2 latent canvas: 4 voxels + 4 register tokens +
    // the zero token = 9 tokens through all 36 blocks.
    let (frames, height, width) = (1usize, 2usize, 2usize);
    let mut dec = rlx_minimax_h3::vae_video::compile_video_decoder(
        &cfg,
        &mut weights,
        rlx_runtime::Device::Cpu,
        frames,
        height,
        width,
    )
    .expect("compile video decoder");
    assert_eq!(dec.num_tokens(), 4 + cfg.decoder_num_register_tokens + 1);

    let voxels = frames * height * width;
    let mut latents: Vec<f32> = (0..cfg.latent_channels * voxels)
        .map(|i| ((i % 13) as f32 / 13.0) - 0.5)
        .collect();
    rlx_minimax_h3::vae_video::denormalize_latents(&mut latents, &cfg, voxels).expect("denorm");

    let pixels = dec.decode_chunk(&latents).expect("decode chunk");
    let expect = cfg.out_channels * frames * 4 * height * 16 * width * 16;
    assert_eq!(pixels.len(), expect, "decoded pixel count");
    assert!(
        pixels.iter().all(|v| v.is_finite()),
        "video decoder produced non-finite pixels"
    );
    assert!(
        pixels.iter().any(|v| v.abs() > 1e-5),
        "video decoder produced a flat image"
    );
    let peak = pixels.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    eprintln!("decoded {} pixels, peak magnitude {peak:.4}", pixels.len());
}

#[test]
fn video_vae_encoder_runs_on_real_weights() {
    use rlx_minimax_h3::vae_video_encoder::{H3VideoEncoder, Volume, normalize_pixels};
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let want: std::collections::HashSet<String> =
        rlx_minimax_h3::vae_video_encoder::encoder_parameter_keys(&cfg)
            .into_iter()
            .collect();
    let weights =
        rlx_core::weight_map::WeightMap::from_safetensors_dir_selected(&root.join("vae"), &want)
            .expect("load vae encoder");
    let enc = H3VideoEncoder::load(&cfg, &weights).expect("build video encoder");

    // A single 32x32 image: one latent frame at 2x2 after the 16x compression.
    let (h, w) = (32usize, 32usize);
    let mut img = Volume::new(3, 1, h, w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                // A smooth gradient plus a checker, so the latent is not flat.
                let v = (x as f32 / w as f32) * 0.5
                    + (y as f32 / h as f32) * 0.3
                    + if (x / 4 + y / 4) % 2 == 0 { 0.1 } else { 0.0 }
                    + c as f32 * 0.05;
                img.data[(c * h + y) * w + x] = v.clamp(0.0, 1.0);
            }
        }
    }
    normalize_pixels(&mut img).expect("normalize");

    let posterior = enc.encode(&img).expect("encode");
    let mean = posterior.mode();
    assert_eq!(mean.channels, cfg.latent_channels, "latent channel count");
    assert_eq!(mean.frames, 1, "a single image stays one latent frame");
    assert_eq!(
        (mean.height, mean.width),
        (h / cfg.spatial_compression(), w / cfg.spatial_compression()),
        "16x spatial compression"
    );
    assert!(mean.is_finite(), "encoder produced non-finite latents");
    assert!(
        posterior.logvar.is_finite(),
        "encoder produced non-finite log-variance"
    );
    assert!(
        mean.data.iter().any(|v| v.abs() > 1e-3),
        "encoder collapsed a textured image to zero"
    );

    let peak = mean.data.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let avg = mean.data.iter().map(|v| v.abs()).sum::<f32>() / mean.data.len() as f32;
    eprintln!(
        "encoded {}x{} -> latent {}x{}x{}, |mean| avg {avg:.4} peak {peak:.4}",
        h, w, mean.channels, mean.height, mean.width
    );
}

#[test]
fn video_vae_round_trip_returns_to_pixel_range() {
    // Encode an image, normalize the latents the way the DiT sees them,
    // denormalize, and decode. The result must land back in display range —
    // this is what ties the encoder, the latent statistics and the decoder
    // together, and it fails loudly if any of the three disagree.
    use rlx_minimax_h3::vae_video::{denormalize_latents, normalize_latents, to_display_range};
    use rlx_minimax_h3::vae_video_encoder::{H3VideoEncoder, Volume, normalize_pixels};
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");

    let enc_keys: std::collections::HashSet<String> =
        rlx_minimax_h3::vae_video_encoder::encoder_parameter_keys(&cfg)
            .into_iter()
            .collect();
    let enc_w = rlx_core::weight_map::WeightMap::from_safetensors_dir_selected(
        &root.join("vae"),
        &enc_keys,
    )
    .expect("load encoder");
    let enc = H3VideoEncoder::load(&cfg, &enc_w).expect("encoder");

    let (h, w) = (32usize, 32usize);
    let mut img = Volume::new(3, 1, h, w);
    for i in 0..img.data.len() {
        img.data[i] = ((i % 29) as f32) / 29.0;
    }
    normalize_pixels(&mut img).expect("normalize pixels");
    let posterior = enc.encode(&img).expect("encode");

    let voxels = posterior.mean.voxels();
    let mut latents = posterior.mode().data.clone();
    // Round-trip through the DiT's normalization.
    normalize_latents(&mut latents, &cfg, voxels).expect("normalize latents");
    assert!(latents.iter().all(|v| v.is_finite()));
    denormalize_latents(&mut latents, &cfg, voxels).expect("denormalize latents");
    for (a, b) in latents.iter().zip(&posterior.mode().data) {
        assert!(
            (a - b).abs() < 1e-3,
            "latent normalization is not invertible"
        );
    }

    let mut dec_w =
        rlx_minimax_h3::weights::load_video_vae_decoder(&root, &cfg).expect("load decoder");
    let mut dec = rlx_minimax_h3::vae_video::compile_video_decoder(
        &cfg,
        &mut dec_w,
        rlx_runtime::Device::Cpu,
        posterior.mean.frames,
        posterior.mean.height,
        posterior.mean.width,
    )
    .expect("compile decoder");

    let mut pixels = dec.decode_chunk(&latents).expect("decode");
    assert_eq!(
        pixels.len(),
        cfg.out_channels * (h / 16 * 16) * (w / 16 * 16) * 4,
        "the decoder expands each latent voxel to 4x16x16"
    );
    assert!(pixels.iter().all(|v| v.is_finite()));
    to_display_range(&mut pixels, cfg.out_channels).expect("display range");
    assert!(
        pixels.iter().all(|v| (0.0..=1.0).contains(v)),
        "round trip left the display range"
    );
    let mean: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;
    eprintln!("round trip: {} pixels, mean {mean:.4}", pixels.len());
}

#[test]
fn audio_vae_encoder_runs_on_real_weights() {
    use rlx_minimax_h3::vae_audio::H3AudioEncoder;
    let root = checkpoint!();
    let cfg = H3AudioVaeConfig::from_dir(&root.join("audio_vae")).expect("audio vae config");
    let weights = rlx_minimax_h3::weights::load_audio_vae(&root).expect("load audio vae");
    let enc = H3AudioEncoder::load(&cfg, &weights).expect("build audio encoder");

    // Two hops of a 220 Hz tone at 32 kHz -> exactly 2 latent frames.
    let hop = cfg.encoder_hop();
    let samples = 2 * hop;
    let wave: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / cfg.sampling_rate as f32;
            0.4 * (std::f32::consts::TAU * 220.0 * t).sin()
        })
        .collect();

    let posterior = enc.encode_mono(&wave).expect("encode");
    assert_eq!(posterior.mean.channels, cfg.latent_channels);
    assert_eq!(
        posterior.mean.length,
        enc.latent_frames(samples),
        "{samples} samples must give {} latent frames",
        enc.latent_frames(samples)
    );
    assert_eq!(posterior.mean.length, 2);
    assert!(
        posterior.mean.is_finite(),
        "encoder produced non-finite latents"
    );
    assert!(
        posterior.logs.is_finite(),
        "encoder produced non-finite log-std"
    );
    assert!(
        posterior.mean.data.iter().any(|v| v.abs() > 1e-3),
        "encoder collapsed a tone to zero"
    );
    let peak = posterior
        .mean
        .data
        .iter()
        .fold(0.0f32, |a, v| a.max(v.abs()));
    eprintln!(
        "encoded {samples} samples -> {}x{} latent, peak |mean| {peak:.4}",
        posterior.mean.channels, posterior.mean.length
    );
}

#[test]
fn audio_vae_round_trip_preserves_the_latent_rate() {
    // Encode a tone, push the latents through the DiT's normalization and back,
    // then decode. The sample count must return to what went in — which is the
    // property that keeps audio and video in sync.
    use rlx_minimax_h3::vae_audio::{
        H3AudioDecoder, H3AudioEncoder, denormalize_latents, normalize_latents,
    };
    let root = checkpoint!();
    let cfg = H3AudioVaeConfig::from_dir(&root.join("audio_vae")).expect("audio vae config");
    let weights = rlx_minimax_h3::weights::load_audio_vae(&root).expect("load audio vae");
    let enc = H3AudioEncoder::load(&cfg, &weights).expect("encoder");
    let dec = H3AudioDecoder::load(&cfg, &weights).expect("decoder");

    let hop = cfg.encoder_hop();
    let samples = 3 * hop;
    let wave: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / cfg.sampling_rate as f32;
            0.3 * (std::f32::consts::TAU * 440.0 * t).sin()
                + 0.1 * (std::f32::consts::TAU * 1320.0 * t).sin()
        })
        .collect();

    let posterior = enc.encode_mono(&wave).expect("encode");
    let mut latents = posterior.mode().clone();
    assert_eq!(latents.length, 3);

    normalize_latents(&mut latents, &cfg).expect("normalize");
    assert!(latents.is_finite());
    denormalize_latents(&mut latents, &cfg).expect("denormalize");
    for (a, b) in latents.data.iter().zip(&posterior.mode().data) {
        assert!(
            (a - b).abs() < 1e-3,
            "latent normalization is not invertible"
        );
    }

    let out = dec.decode_mono(&latents).expect("decode");
    assert_eq!(
        out.len(),
        samples,
        "the round trip must return the sample count it was given"
    );
    assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    eprintln!(
        "audio round trip: {} samples, output RMS {rms:.4}",
        out.len()
    );
    assert!(rms > 1e-4, "the round trip produced silence");
}

#[test]
fn text_encoder_tap_keys_exist_in_the_real_index() {
    // The encoder itself is ~60 GB and was not fetched, but its index is small
    // and pins down that the tapped key names are right — and that tapping at
    // layer 50 really does skip the last 14 layers, the final norm and lm_head.
    let root = checkpoint!();
    let index = root.join("text_encoder/model.safetensors.index.json");
    if !index.is_file() {
        eprintln!("skipping: {} not present", index.display());
        return;
    }
    let raw = std::fs::read_to_string(&index).expect("read index");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
    let map = json["weight_map"].as_object().expect("weight_map");

    let cfg = rlx_minimax_h3::config::H3TextEncoderConfig::from_dir(&root.join("text_encoder"))
        .expect("text encoder config");
    cfg.validate().expect("config must reach the tap");

    let keys = rlx_minimax_h3::qwen3vl::parameter_keys(&cfg);
    let missing: Vec<&String> = keys.iter().filter(|k| !map.contains_key(*k)).collect();
    assert!(
        missing.is_empty(),
        "{} tapped key(s) absent from the checkpoint, first few: {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );

    // The saving: the tap reads far fewer tensors than the encoder ships.
    assert!(
        keys.len() < map.len(),
        "the tap should read fewer tensors ({}) than the checkpoint holds ({})",
        keys.len(),
        map.len()
    );
    assert!(
        !keys.iter().any(|k| k.contains("lm_head")),
        "lm_head must never be loaded"
    );
    assert!(
        map.contains_key("lm_head.weight"),
        "the checkpoint does ship lm_head; the tap just skips it"
    );
    eprintln!(
        "text encoder: tap reads {} of {} tensors ({} layers of {})",
        keys.len(),
        map.len(),
        rlx_minimax_h3::qwen3vl::layers_to_run(&cfg),
        cfg.num_hidden_layers
    );
}

#[test]
fn video_vae_decodes_a_full_clip_with_cross_faded_seams() {
    // The window scheme is the part most likely to be subtly wrong: windows
    // overlap, carry an implicit leading pad, and cross-fade. This runs the real
    // decoder over a multi-window clip and checks the frame count lands exactly
    // where the encoder's inverse says it should.
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");

    let latent_frames = 7usize; // -> 22 pixel frames, 1 window + trailing overlap
    let g = cfg.chunk_geometry(latent_frames);
    assert!(g.is_decodable());
    assert_eq!(g.window_frames, 7);
    assert_eq!(g.num_pixel_frames, 22);

    let (lh, lw) = (2usize, 2usize);
    let mut weights =
        rlx_minimax_h3::weights::load_video_vae_decoder(&root, &cfg).expect("load decoder");
    let mut dec = rlx_minimax_h3::vae_video::compile_video_decoder(
        &cfg,
        &mut weights,
        rlx_runtime::Device::Cpu,
        g.window_frames,
        lh,
        lw,
    )
    .expect("compile decoder for one window");

    let voxels = latent_frames * lh * lw;
    let latents: Vec<f32> = (0..cfg.latent_channels * voxels)
        .map(|i| ((i % 19) as f32 / 19.0) - 0.5)
        .collect();

    let pixels = dec
        .decode_clip(&latents, latent_frames)
        .expect("decode clip");
    let frame_px = cfg.out_channels * (lh * 16) * (lw * 16);
    assert_eq!(
        pixels.len(),
        g.num_pixel_frames * frame_px,
        "a {latent_frames}-frame clip must decode to {} pixel frames",
        g.num_pixel_frames
    );
    assert!(pixels.iter().all(|v| v.is_finite()));

    // No frame may be identically zero — an empty window would show up here.
    for f in 0..g.num_pixel_frames {
        let frame = &pixels[f * frame_px..(f + 1) * frame_px];
        assert!(
            frame.iter().any(|v| v.abs() > 1e-6),
            "pixel frame {f} came out blank"
        );
    }
    eprintln!(
        "decoded clip: {} latent -> {} pixel frames ({} windows, {} cross-faded)",
        latent_frames, g.num_pixel_frames, g.num_chunks, g.frame_overlap
    );
}

#[test]
fn full_clip_decode_matches_a_single_window_on_its_body() {
    // The first window's body is untouched by cross-fading, so a full-clip
    // decode has to agree with a direct window decode there. If the pre-pad cut
    // or the segment split were off, this would drift.
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let latent_frames = 7usize;
    let g = cfg.chunk_geometry(latent_frames);
    let (lh, lw) = (2usize, 2usize);

    let mut weights =
        rlx_minimax_h3::weights::load_video_vae_decoder(&root, &cfg).expect("load decoder");
    let mut dec = rlx_minimax_h3::vae_video::compile_video_decoder(
        &cfg,
        &mut weights,
        rlx_runtime::Device::Cpu,
        g.window_frames,
        lh,
        lw,
    )
    .expect("compile");

    let voxels = latent_frames * lh * lw;
    let latents: Vec<f32> = (0..cfg.latent_channels * voxels)
        .map(|i| ((i % 19) as f32 / 19.0) - 0.5)
        .collect();

    let clip = dec.decode_clip(&latents, latent_frames).expect("clip");
    let window = dec.decode_chunk(&latents).expect("single window");

    let frame_px = cfg.out_channels * (lh * 16) * (lw * 16);
    // The clip's first frames are the window's frames past the leading pad.
    let skip = g.frame_pre_padding;
    let compare = g.frames_per_window.min(g.num_pixel_frames);
    for f in 0..compare {
        let a = &clip[f * frame_px..(f + 1) * frame_px];
        let b = &window[(skip + f) * frame_px..(skip + f + 1) * frame_px];
        let d: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(
            d < 1e-4,
            "frame {f} differs by {d} between clip and window decode"
        );
    }
    eprintln!("first {compare} frames agree between clip and window decode");
}

#[test]
fn vae_parameter_sets_partition_the_checkpoint_exactly() {
    // A missed module is invisible in shapes: the decoder ran for a while
    // without `post_quant_conv` and produced perfectly finite, in-range, wrong
    // pixels. This asserts the encoder and decoder key sets together account for
    // *every* tensor the VAE ships — nothing skipped, nothing invented.
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let raw =
        std::fs::read_to_string(root.join("vae/diffusion_pytorch_model.safetensors.index.json"))
            .expect("read vae index");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse vae index");
    let shipped: std::collections::HashSet<String> = json["weight_map"]
        .as_object()
        .expect("weight_map")
        .keys()
        .cloned()
        .collect();

    let decoder: std::collections::HashSet<String> =
        rlx_minimax_h3::vae_video::decoder_parameter_keys(&cfg)
            .into_iter()
            .collect();
    let encoder: std::collections::HashSet<String> =
        rlx_minimax_h3::vae_video_encoder::encoder_parameter_keys(&cfg)
            .into_iter()
            .collect();

    let invented: Vec<&String> = decoder
        .union(&encoder)
        .filter(|k| !shipped.contains(*k))
        .collect();
    assert!(
        invented.is_empty(),
        "the port expects {} tensor(s) the checkpoint does not ship: {:?}",
        invented.len(),
        &invented[..invented.len().min(5)]
    );

    let covered: std::collections::HashSet<&String> = decoder.union(&encoder).collect();
    let skipped: Vec<&String> = shipped.iter().filter(|k| !covered.contains(*k)).collect();
    assert!(
        skipped.is_empty(),
        "the port never reads {} shipped tensor(s): {:?}",
        skipped.len(),
        &skipped[..skipped.len().min(8)]
    );

    // The two halves must not overlap.
    let both: Vec<&String> = decoder.intersection(&encoder).collect();
    assert!(both.is_empty(), "tensors claimed by both halves: {both:?}");
    eprintln!(
        "VAE tensors: {} decoder + {} encoder = {} shipped",
        decoder.len(),
        encoder.len(),
        shipped.len()
    );
}

#[test]
fn audio_vae_parameter_sets_partition_the_checkpoint_exactly() {
    let root = checkpoint!();
    let cfg = H3AudioVaeConfig::from_dir(&root.join("audio_vae")).expect("audio vae config");
    let path = root.join("audio_vae/diffusion_pytorch_model.safetensors");
    let bytes = std::fs::read(&path).expect("read audio vae");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse audio vae");
    let shipped: std::collections::HashSet<String> =
        st.names().into_iter().map(|s| s.to_string()).collect();

    let (enc, dec) = rlx_minimax_h3::vae_audio::audio_parameter_keys(&cfg);
    let enc: std::collections::HashSet<String> = enc.into_iter().collect();
    let dec: std::collections::HashSet<String> = dec.into_iter().collect();

    let invented: Vec<&String> = enc.union(&dec).filter(|k| !shipped.contains(*k)).collect();
    assert!(
        invented.is_empty(),
        "the port expects {} tensor(s) the checkpoint does not ship: {:?}",
        invented.len(),
        &invented[..invented.len().min(8)]
    );
    let covered: std::collections::HashSet<&String> = enc.union(&dec).collect();
    let skipped: Vec<&String> = shipped.iter().filter(|k| !covered.contains(*k)).collect();
    assert!(
        skipped.is_empty(),
        "the port never reads {} shipped tensor(s): {:?}",
        skipped.len(),
        &skipped[..skipped.len().min(8)]
    );
    let both: Vec<&String> = enc.intersection(&dec).collect();
    assert!(both.is_empty(), "tensors claimed by both halves: {both:?}");
    eprintln!(
        "audio VAE tensors: {} encoder + {} decoder = {} shipped",
        enc.len(),
        dec.len(),
        shipped.len()
    );
}

#[test]
fn video_vae_decodes_a_tiled_clip_on_real_weights() {
    // Tiling is on by default in the released pipeline, so this is the path a
    // normal decode takes. A 384x384 canvas at 128-pixel tiles is a 4x4 grid —
    // enough for interior tiles that blend on both axes.
    use rlx_minimax_h3::vae_video::{decode_clip_tiled, split_tiles};
    let root = checkpoint!();
    let cfg = H3VideoVaeConfig::from_dir(&root.join("vae")).expect("video vae config");
    let ratio = cfg.spatial_compression();

    let (tile_px, overlap_px) = (128usize, 32usize);
    let (latent_h, latent_w) = (24usize, 24usize); // 384x384 pixels
    let y = split_tiles(latent_h * ratio, tile_px, overlap_px, ratio).unwrap();
    let x = split_tiles(latent_w * ratio, tile_px, overlap_px, ratio).unwrap();
    assert_eq!((y.count(), x.count()), (4, 4), "expected a 4x4 tile grid");
    assert_eq!(y.covered(), latent_h * ratio);

    let latent_frames = 7usize;
    let g = cfg.chunk_geometry(latent_frames);
    let (tile_lh, tile_lw) = (tile_px / ratio, tile_px / ratio);

    let mut weights =
        rlx_minimax_h3::weights::load_video_vae_decoder(&root, &cfg).expect("load decoder");
    let mut dec = rlx_minimax_h3::vae_video::compile_video_decoder(
        &cfg,
        &mut weights,
        rlx_runtime::Device::Cpu,
        g.window_frames,
        tile_lh,
        tile_lw,
    )
    .expect("compile decoder for one tile");

    let latents: Vec<f32> = (0..cfg.latent_channels * latent_frames * latent_h * latent_w)
        .map(|i| ((i % 31) as f32 / 31.0) - 0.5)
        .collect();

    let out = decode_clip_tiled(
        &mut dec,
        &latents,
        latent_frames,
        latent_h,
        latent_w,
        tile_px,
        overlap_px,
    )
    .expect("tiled decode");

    assert_eq!(out.channels, cfg.out_channels);
    assert_eq!(out.frames, g.num_pixel_frames);
    assert_eq!(
        (out.height, out.width),
        (latent_h * ratio, latent_w * ratio),
        "the stitched canvas must match the requested resolution"
    );
    assert!(out.is_finite(), "tiled decode produced non-finite pixels");

    // No seam should be blank, and no tile should have gone missing: every
    // 16-pixel latent column must carry some signal.
    for x0 in (0..out.width).step_by(16) {
        let col: f32 = (0..out.height)
            .map(|y0| out.data[(y0 * out.width) + x0].abs())
            .sum();
        assert!(col > 1e-4, "column {x0} of the stitched canvas is blank");
    }
    eprintln!(
        "tiled decode: {}x{} from a {}x{} grid, {} frames",
        out.height,
        out.width,
        y.count(),
        x.count(),
        out.frames
    );
}
