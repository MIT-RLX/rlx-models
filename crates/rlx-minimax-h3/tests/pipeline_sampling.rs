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

//! The full sampling loop on a scaled-down DiT: two schedules, one packed
//! sequence, conditioning rows pinned across every step.

use rlx_minimax_h3::config::H3TransformerConfig;
use rlx_minimax_h3::layout::{H3Geometry, H3Reference, KeyframeAnchor};
use rlx_minimax_h3::pipeline::{H3Conditioning, H3Pipeline, H3Request, H3Task};
use rlx_minimax_h3::scheduler::H3Scheduler;
use rlx_minimax_h3::text_encoder::placeholder_conditioning;
use rlx_minimax_h3::transformer::compile_dit;
use rlx_minimax_h3::weights::synthetic_dit_weights;
use rlx_runtime::Device;

fn tiny_cfg() -> H3TransformerConfig {
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

fn build(
    task: H3Task,
    anchors: Vec<KeyframeAnchor>,
    refs: Vec<H3Reference>,
    steps: usize,
) -> (
    H3Pipeline,
    H3Request,
    rlx_minimax_h3::layout::PackedLayout,
    rlx_minimax_h3::text_encoder::H3TextConditioning,
) {
    let cfg = tiny_cfg();
    let cond = placeholder_conditioning(5, cfg.text_dim);
    let mut request = H3Request::t2va(geometry(), steps);
    request.task = task;
    request.keyframe_anchors = anchors;
    request.references = refs;

    let layout = request.build_layout(&cond, cfg.patch_size).expect("layout");
    let mut weights = synthetic_dit_weights(&cfg, 21);
    let dit = compile_dit(
        &cfg,
        &mut weights,
        Device::Cpu,
        layout.sequence_length(),
        layout.text_indices.len(),
        layout.video_indices.len(),
        layout.audio_indices.len(),
    )
    .expect("compile DiT");
    let pipeline = H3Pipeline::new(dit, H3Scheduler::video(), H3Scheduler::audio());
    (pipeline, request, layout, cond)
}

#[test]
fn t2va_sampling_runs_to_completion() {
    let (mut p, req, layout, cond) = build(H3Task::T2VA, vec![], vec![], 6);
    let cfg = tiny_cfg();
    let out = p
        .sample(&req, &layout, &cond, &H3Conditioning::default())
        .expect("sample");

    assert_eq!(
        out.video.len(),
        layout.video_indices.len() * cfg.video_patch_dim()
    );
    assert_eq!(
        out.audio.len(),
        layout.audio_indices.len() * cfg.audio_in_channels
    );
    assert!(
        out.video.iter().all(|v| v.is_finite()),
        "video latents diverged"
    );
    assert!(
        out.audio.iter().all(|v| v.is_finite()),
        "audio latents diverged"
    );
    assert_eq!(out.num_condition_video_rows, 0);
}

#[test]
fn sampling_is_reproducible_from_the_seed() {
    let (mut p, mut req, layout, cond) = build(H3Task::T2VA, vec![], vec![], 4);
    req.seed = 1234;
    let a = p
        .sample(&req, &layout, &cond, &H3Conditioning::default())
        .unwrap();
    let b = p
        .sample(&req, &layout, &cond, &H3Conditioning::default())
        .unwrap();
    assert_eq!(a.video, b.video);
    assert_eq!(a.audio, b.audio);

    req.seed = 5678;
    let c = p
        .sample(&req, &layout, &cond, &H3Conditioning::default())
        .unwrap();
    assert_ne!(a.video, c.video, "a different seed must change the sample");
}

#[test]
fn conditioning_rows_are_held_fixed_across_every_step() {
    // This is what keeps a keyframe from smearing: the DiT predicts a velocity
    // for the anchor rows too, and the loop has to write them back.
    let (mut p, req, layout, cond) = build(
        H3Task::FL2VA,
        vec![KeyframeAnchor::First, KeyframeAnchor::Last],
        vec![],
        5,
    );
    let cfg = tiny_cfg();
    let vpd = cfg.video_patch_dim();
    let cond_rows = layout.num_condition_video_rows;
    assert!(cond_rows > 0);

    let anchor_values: Vec<f32> = (0..cond_rows * vpd)
        .map(|i| ((i % 23) as f32 / 23.0) - 0.5)
        .collect();
    let anchors = H3Conditioning {
        video_rows: anchor_values.clone(),
        audio_rows: Vec::new(),
    };

    let out = p.sample(&req, &layout, &cond, &anchors).expect("sample");
    assert_eq!(
        &out.video[..cond_rows * vpd],
        anchor_values.as_slice(),
        "the conditioning rows drifted during sampling"
    );
    assert_eq!(out.num_condition_video_rows, cond_rows);
    // The generated rows did move.
    let generated = out.generated_video(vpd);
    assert!(generated.iter().any(|v| v.abs() > 1e-6));
    assert!(generated.iter().all(|v| v.is_finite()));
}

#[test]
fn ref2va_sampling_pins_both_reference_kinds() {
    let refs = vec![
        H3Reference::Image {
            latent_frames: 1,
            latent_height: 4,
            latent_width: 4,
        },
        H3Reference::Audio { audio_rows: 4 },
    ];
    let (mut p, req, layout, cond) = build(H3Task::Ref2VA, vec![], refs, 4);
    let cfg = tiny_cfg();
    let vpd = cfg.video_patch_dim();
    let aic = cfg.audio_in_channels;
    assert_eq!(layout.num_condition_video_rows, 4); // 1 frame x 2x2 patches
    assert_eq!(layout.num_condition_audio_rows, 4);

    let v: Vec<f32> = (0..layout.num_condition_video_rows * vpd)
        .map(|i| i as f32 * 0.01)
        .collect();
    let a: Vec<f32> = (0..layout.num_condition_audio_rows * aic)
        .map(|i| i as f32 * -0.01)
        .collect();
    let anchors = H3Conditioning {
        video_rows: v.clone(),
        audio_rows: a.clone(),
    };

    let out = p.sample(&req, &layout, &cond, &anchors).expect("sample");
    assert_eq!(
        &out.video[..v.len()],
        v.as_slice(),
        "video references drifted"
    );
    assert_eq!(
        &out.audio[..a.len()],
        a.as_slice(),
        "audio references drifted"
    );
    assert!(out.video.iter().all(|x| x.is_finite()));
    assert!(out.audio.iter().all(|x| x.is_finite()));
}

#[test]
fn mismatched_conditioning_is_rejected_before_sampling() {
    let (mut p, req, layout, cond) = build(H3Task::I2VA, vec![KeyframeAnchor::First], vec![], 4);
    // The right row count is `num_condition_video_rows * video_patch_dim`.
    let wrong = H3Conditioning {
        video_rows: vec![0.0; 3],
        audio_rows: Vec::new(),
    };
    let err = p
        .sample(&req, &layout, &cond, &wrong)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("conditioning holds"),
        "unexpected error: {err}"
    );
}

#[test]
fn text_conditioning_width_is_checked_against_the_transformer() {
    let (mut p, req, layout, _) = build(H3Task::T2VA, vec![], vec![], 4);
    // Right row count, wrong width.
    let bad = placeholder_conditioning(layout.text_indices.len(), 999);
    let err = p
        .sample(&req, &layout, &bad, &H3Conditioning::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("text_dim"), "unexpected error: {err}");
}

#[test]
fn the_two_schedules_advance_independently() {
    // Video runs shift 12 and audio shift 3, so at the same step index the two
    // modalities sit at different noise levels — that is the whole reason the
    // DiT takes a per-row timestep.
    let mut v = H3Scheduler::video();
    let mut a = H3Scheduler::audio();
    v.set_timesteps(16).unwrap();
    a.set_timesteps(16).unwrap();
    let differing = v
        .timesteps()
        .iter()
        .zip(a.timesteps())
        .filter(|(x, y)| (*x - *y).abs() > 1e-6)
        .count();
    assert!(
        differing > 10,
        "only {differing} of the steps differ between the two schedules"
    );
}
