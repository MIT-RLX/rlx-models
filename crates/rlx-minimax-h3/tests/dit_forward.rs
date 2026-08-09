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

//! End-to-end CPU checks of the compiled MiniMax-H3 DiT on a scaled-down
//! architecture with deterministic synthetic weights.
//!
//! The released checkpoint is ~28 GB per partition, so these run the real graph
//! construction — every stage, both output heads, the packed scatter and the
//! AdaLN table gather — at a size that fits in a test.

use rlx_minimax_h3::config::{H3TransformerConfig, MODALITY_NUM, Modality};
use rlx_minimax_h3::layout::{
    H3Geometry, KeyframeAnchor, PackedLayout, build_packed_sequence, build_row_timesteps,
};
use rlx_minimax_h3::rope::RopeTables;
use rlx_minimax_h3::transformer::{CompiledH3Dit, H3DitInputs, H3DitLayout, compile_dit};
use rlx_minimax_h3::weights::synthetic_dit_weights;
use rlx_runtime::Device;

/// A DiT small enough to compile in a test but structurally identical to the
/// released one: same block shape, same AdaLN table, same two heads.
fn tiny_cfg() -> H3TransformerConfig {
    H3TransformerConfig {
        num_attention_heads: 2,
        attention_head_dim: 16,
        // Deliberately != heads * head_dim, as in the released checkpoint
        // (5376 vs 7168).
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

/// A layout small enough to compile: 6 latent frames on a 4x4 latent canvas
/// with 3 audio latents.
fn tiny_layout(anchors: &[KeyframeAnchor]) -> (PackedLayout, H3Geometry) {
    let geometry = H3Geometry {
        height: 64,
        width: 64,
        num_frames: 39,
        num_latent_frames: 3,
        latent_height: 4,
        latent_width: 4,
        num_audio_latents: 3,
    };
    let text_tags = vec![Modality::Text.tag(); 5];
    let layout = build_packed_sequence(&text_tags, &geometry, [1, 2, 2], anchors).unwrap();
    (layout, geometry)
}

struct Harness {
    dit: CompiledH3Dit,
    layout: PackedLayout,
    cfg: H3TransformerConfig,
}

fn build(anchors: &[KeyframeAnchor]) -> Harness {
    let cfg = tiny_cfg();
    let (layout, _) = tiny_layout(anchors);
    let mut weights = synthetic_dit_weights(&cfg, 11);
    let dit = compile_dit(
        &cfg,
        &mut weights,
        Device::Cpu,
        layout.sequence_length(),
        layout.text_indices.len(),
        layout.video_indices.len(),
        layout.audio_indices.len(),
    )
    .expect("compile tiny DiT");
    Harness { dit, layout, cfg }
}

fn run(
    h: &mut Harness,
    video: &[f32],
    audio: &[f32],
    text: &[f32],
    t_video: f32,
    t_audio: f32,
) -> (Vec<f32>, Vec<f32>) {
    let rows = build_row_timesteps(&h.layout, t_video, t_audio, 0.999, 1.0).unwrap();
    let dl = H3DitLayout::new(&h.layout, &rows, &h.cfg).unwrap();
    let tables = RopeTables::build(
        &h.layout.flat_position_ids(),
        h.cfg.rope_freq_dim,
        h.cfg.rope_theta,
    )
    .unwrap();
    let out = h
        .dit
        .forward(&H3DitInputs {
            video_rows: video,
            audio_rows: audio,
            text_rows: text,
            cos: &tables.cos,
            sin: &tables.sin,
            layout: &dl,
        })
        .expect("DiT forward");
    (out.video, out.audio)
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 17) as f32 / 17.0 - 0.5) * scale)
        .collect()
}

#[test]
fn t2va_forward_produces_finite_velocities_of_the_right_shape() {
    let mut h = build(&[]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();

    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio = ramp(na * c.audio_in_channels, 1.0);
    let text = ramp(nt * c.text_dim, 1.0);
    let (v, a) = run(&mut h, &video, &audio, &text, 0.3, 0.4);

    assert_eq!(v.len(), nv * c.video_patch_dim());
    assert_eq!(a.len(), na * c.audio_in_channels);
    assert!(
        v.iter().all(|x| x.is_finite()),
        "video velocity has non-finite values"
    );
    assert!(
        a.iter().all(|x| x.is_finite()),
        "audio velocity has non-finite values"
    );
    // A trained-looking synthetic stack should not collapse to exactly zero.
    assert!(v.iter().any(|x| x.abs() > 1e-6));
    assert!(a.iter().any(|x| x.abs() > 1e-6));
}

#[test]
fn forward_is_deterministic() {
    let mut h = build(&[]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();
    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio = ramp(na * c.audio_in_channels, 1.0);
    let text = ramp(nt * c.text_dim, 1.0);

    let (v1, a1) = run(&mut h, &video, &audio, &text, 0.3, 0.4);
    let (v2, a2) = run(&mut h, &video, &audio, &text, 0.3, 0.4);
    assert_eq!(v1, v2);
    assert_eq!(a1, a2);
}

#[test]
fn timestep_changes_the_prediction() {
    // The AdaLN table is addressed by (timestep, modality), so moving the
    // timestep must move both heads. A stuck timestep path is the failure this
    // catches.
    let mut h = build(&[]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();
    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio = ramp(na * c.audio_in_channels, 1.0);
    let text = ramp(nt * c.text_dim, 1.0);

    let (v1, a1) = run(&mut h, &video, &audio, &text, 0.1, 0.2);
    let (v2, a2) = run(&mut h, &video, &audio, &text, 0.8, 0.9);
    let dv: f32 = v1.iter().zip(&v2).map(|(x, y)| (x - y).abs()).sum();
    let da: f32 = a1.iter().zip(&a2).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        dv > 1e-5,
        "video velocity ignored the timestep (delta {dv})"
    );
    assert!(
        da > 1e-5,
        "audio velocity ignored the timestep (delta {da})"
    );
}

#[test]
fn text_conditioning_reaches_both_heads() {
    // Full self-attention over one packed sequence means the text rows must
    // influence the video *and* the audio prediction. If the scatter
    // permutation were wrong, one of these would go dead.
    let mut h = build(&[]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();
    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio = ramp(na * c.audio_in_channels, 1.0);

    let text_a = ramp(nt * c.text_dim, 1.0);
    let text_b: Vec<f32> = text_a.iter().map(|x| -x).collect();
    let (v1, a1) = run(&mut h, &video, &audio, &text_a, 0.3, 0.4);
    let (v2, a2) = run(&mut h, &video, &audio, &text_b, 0.3, 0.4);

    let dv: f32 = v1.iter().zip(&v2).map(|(x, y)| (x - y).abs()).sum();
    let da: f32 = a1.iter().zip(&a2).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        dv > 1e-5,
        "the video head ignored the text stream (delta {dv})"
    );
    assert!(
        da > 1e-5,
        "the audio head ignored the text stream (delta {da})"
    );
}

#[test]
fn audio_rows_influence_the_video_head_and_the_reverse() {
    // Joint generation: the two modalities share one attention document, so
    // each must see the other.
    let mut h = build(&[]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();
    let text = ramp(nt * c.text_dim, 1.0);
    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio_a = ramp(na * c.audio_in_channels, 1.0);
    let audio_b: Vec<f32> = audio_a.iter().map(|x| x + 0.5).collect();

    let (v1, _) = run(&mut h, &video, &audio_a, &text, 0.3, 0.4);
    let (v2, _) = run(&mut h, &video, &audio_b, &text, 0.3, 0.4);
    let dv: f32 = v1.iter().zip(&v2).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        dv > 1e-5,
        "the video head ignored the audio rows (delta {dv})"
    );

    let video_b: Vec<f32> = video.iter().map(|x| x + 0.5).collect();
    let (_, a1) = run(&mut h, &video, &audio_a, &text, 0.3, 0.4);
    let (_, a2) = run(&mut h, &video_b, &audio_a, &text, 0.3, 0.4);
    let da: f32 = a1.iter().zip(&a2).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        da > 1e-5,
        "the audio head ignored the video rows (delta {da})"
    );
}

#[test]
fn fl2va_layout_with_condition_rows_runs() {
    let mut h = build(&[KeyframeAnchor::First, KeyframeAnchor::Last]);
    let c = h.cfg.clone();
    let nv = h.layout.video_indices.len();
    let na = h.layout.audio_indices.len();
    let nt = h.layout.text_indices.len();
    assert!(h.layout.num_condition_video_rows > 0);

    let video = ramp(nv * c.video_patch_dim(), 1.0);
    let audio = ramp(na * c.audio_in_channels, 1.0);
    let text = ramp(nt * c.text_dim, 1.0);
    let (v, a) = run(&mut h, &video, &audio, &text, 0.3, 0.4);
    assert_eq!(v.len(), nv * c.video_patch_dim());
    assert_eq!(a.len(), na * c.audio_in_channels);
    assert!(v.iter().all(|x| x.is_finite()));
    assert!(a.iter().all(|x| x.is_finite()));
}

#[test]
fn conditioning_rows_run_at_their_own_noise_level() {
    // With keyframe anchors present the sequence carries three distinct
    // timesteps, so the AdaLN table is genuinely addressed at more than one
    // row per modality.
    let (layout, _) = tiny_layout(&[KeyframeAnchor::First]);
    let rows = build_row_timesteps(&layout, 0.2, 0.35, 0.999, 1.0).unwrap();
    assert_eq!(rows.timesteps.len(), 3);
    let cfg = tiny_cfg();
    let dl = H3DitLayout::new(&layout, &rows, &cfg).unwrap();
    let limit = MAX_TIMESTEPS_TABLE_ROWS;
    assert!(dl.adaln_indices.iter().all(|&i| (i as usize) < limit));
    // Every modality must actually appear.
    let tags: std::collections::HashSet<u32> = layout.token_tags.iter().copied().collect();
    assert!(tags.contains(&Modality::Video.tag()));
    assert!(tags.contains(&Modality::Text.tag()));
    assert!(tags.contains(&Modality::Audio.tag()));
}

const MAX_TIMESTEPS_TABLE_ROWS: usize = rlx_minimax_h3::transformer::MAX_TIMESTEPS * MODALITY_NUM;

#[test]
fn rope_tables_match_the_packed_grid() {
    let (layout, _) = tiny_layout(&[]);
    let cfg = tiny_cfg();
    let t = RopeTables::build(
        &layout.flat_position_ids(),
        cfg.rope_freq_dim,
        cfg.rope_theta,
    )
    .unwrap();
    assert_eq!(t.seq_len, layout.sequence_length());
    assert_eq!(t.half, 3 * cfg.rope_freq_dim);
    assert_eq!(t.n_rot(), cfg.rope_rotary_dim());
    assert!(t.cos.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    assert!(t.sin.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}
