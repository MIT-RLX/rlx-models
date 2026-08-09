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

//! The MiniMax-H3 sampling loop.
//!
//! One request drives **two** rectified-flow schedules at once — `shift = 12`
//! for video, `shift = 3` for audio — over a single packed sequence. Each step:
//!
//! 1. Assign a timestep to every row. Generated video and audio rows step down
//!    their own schedules; conditioning rows stay pinned at their
//!    noise-augmentation level ([`crate::layout::KEYFRAME_NOISE_AUG`] for visual
//!    anchors, `1.0` for audio references). Text rows never reach an output head
//!    and inherit the video timestep.
//! 2. Reduce those to the DiT's `(timestep, timestep_indices)` pair — at most
//!    [`crate::transformer::MAX_TIMESTEPS`] distinct levels.
//! 3. Run the DiT once, getting a video and an audio velocity.
//! 4. Step each modality on its own scheduler, and **write the conditioning rows
//!    back** — the DiT predicts a velocity for every row including the anchors,
//!    and letting those rows drift is what turns a keyframe into a smear.
//!
//! The rotary tables and the packed layout are built once per request, not per
//! step: they depend only on the geometry.

use crate::config::{H3Config, H3TransformerConfig};
use crate::layout::{
    self, AUDIO_CHANNELS, H3Geometry, H3Reference, KEYFRAME_NOISE_AUG, KeyframeAnchor, PackedLayout,
};
use crate::rope::RopeTables;
use crate::scheduler::H3Scheduler;
use crate::text_encoder::H3TextConditioning;
use crate::transformer::{CompiledH3Dit, H3DitInputs, H3DitLayout};
use crate::weights::DitPartition;
use anyhow::{Result, bail, ensure};

/// Which generation path a request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Task {
    /// Text to video + audio.
    T2VA,
    /// One keyframe anchors an end of the clip.
    I2VA,
    /// First *and* last keyframes anchor both ends.
    FL2VA,
    /// Arbitrary image / video / audio references on a shared rotary clock.
    Ref2VA,
}

impl H3Task {
    /// Which DiT partition the task loads. `ref2va` has its own weights.
    #[must_use]
    pub fn partition(self) -> DitPartition {
        match self {
            Self::Ref2VA => DitPartition::Reference,
            _ => DitPartition::Base,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T2VA => "t2va",
            Self::I2VA => "i2va",
            Self::FL2VA => "fl2va",
            Self::Ref2VA => "ref2va",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "t2va" => Self::T2VA,
            "i2va" => Self::I2VA,
            "fl2va" => Self::FL2VA,
            "ref2va" => Self::Ref2VA,
            other => bail!("unknown task `{other}`; expected t2va, i2va, fl2va or ref2va"),
        })
    }
}

/// One generation request.
#[derive(Debug, Clone)]
pub struct H3Request {
    pub task: H3Task,
    pub geometry: H3Geometry,
    /// Which end each keyframe anchors, in packed order. Empty for `t2va`.
    pub keyframe_anchors: Vec<KeyframeAnchor>,
    /// Reference blocks, for `ref2va` only.
    pub references: Vec<H3Reference>,
    pub num_inference_steps: usize,
    /// Overrides the checkpoint's video `shift` when set.
    pub flow_shift: Option<f32>,
    /// Overrides the checkpoint's audio `shift` when set.
    pub audio_flow_shift: Option<f32>,
    pub seed: u64,
}

impl H3Request {
    /// A text-to-video+audio request on the released defaults.
    pub fn t2va(geometry: H3Geometry, num_inference_steps: usize) -> Self {
        Self {
            task: H3Task::T2VA,
            geometry,
            keyframe_anchors: Vec::new(),
            references: Vec::new(),
            num_inference_steps,
            flow_shift: None,
            audio_flow_shift: None,
            seed: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_inference_steps >= 2,
            "num_inference_steps must be at least 2, got {}",
            self.num_inference_steps
        );
        match self.task {
            H3Task::T2VA => ensure!(
                self.keyframe_anchors.is_empty() && self.references.is_empty(),
                "t2va takes no keyframes or references"
            ),
            H3Task::I2VA => ensure!(
                self.keyframe_anchors.len() == 1,
                "i2va takes exactly one keyframe, got {}",
                self.keyframe_anchors.len()
            ),
            H3Task::FL2VA => ensure!(
                self.keyframe_anchors.len() == 2,
                "fl2va takes exactly two keyframes, got {}",
                self.keyframe_anchors.len()
            ),
            H3Task::Ref2VA => ensure!(
                !self.references.is_empty(),
                "ref2va needs at least one reference"
            ),
        }
        Ok(())
    }

    /// Build the packed layout this request implies.
    pub fn build_layout(
        &self,
        conditioning: &H3TextConditioning,
        patch_size: [usize; 3],
    ) -> Result<PackedLayout> {
        self.validate()?;
        match self.task {
            H3Task::Ref2VA => layout::build_ref2va_packed_sequence(
                &conditioning.token_tags,
                &self.references,
                &self.geometry,
                patch_size,
            ),
            _ => layout::build_packed_sequence(
                &conditioning.token_tags,
                &self.geometry,
                patch_size,
                &self.keyframe_anchors,
            ),
        }
    }
}

/// Latents to seed the loop with, and the conditioning rows to hold fixed.
#[derive(Debug, Clone, Default)]
pub struct H3Conditioning {
    /// Patchified latent rows for the leading conditioning video rows, in
    /// `video_indices` order. Empty for `t2va`.
    pub video_rows: Vec<f32>,
    /// Latent rows for the leading audio reference rows.
    pub audio_rows: Vec<f32>,
}

/// What one sampling run produces, still in latent space.
#[derive(Debug, Clone)]
pub struct H3Latents {
    /// `[num_video_rows * video_patch_dim]`, in `video_indices` order —
    /// conditioning rows first, then the generated rows.
    pub video: Vec<f32>,
    /// `[num_audio_rows * audio_in_channels]`, in `audio_indices` order.
    pub audio: Vec<f32>,
    /// How many leading rows of each are conditioning rather than generated.
    pub num_condition_video_rows: usize,
    pub num_condition_audio_rows: usize,
}

impl H3Latents {
    /// The generated video rows, with the conditioning prefix dropped.
    #[must_use]
    pub fn generated_video(&self, video_patch_dim: usize) -> &[f32] {
        &self.video[self.num_condition_video_rows * video_patch_dim..]
    }

    /// The generated audio rows, with the reference prefix dropped.
    #[must_use]
    pub fn generated_audio(&self, audio_channels: usize) -> &[f32] {
        &self.audio[self.num_condition_audio_rows * audio_channels..]
    }
}

/// A compiled DiT plus the two schedules, ready to sample.
pub struct H3Pipeline {
    dit: CompiledH3Dit,
    cfg: H3TransformerConfig,
    video_scheduler: H3Scheduler,
    audio_scheduler: H3Scheduler,
}

impl H3Pipeline {
    /// Wrap an already-compiled DiT.
    pub fn new(
        dit: CompiledH3Dit,
        video_scheduler: H3Scheduler,
        audio_scheduler: H3Scheduler,
    ) -> Self {
        let cfg = dit.config().clone();
        Self {
            dit,
            cfg,
            video_scheduler,
            audio_scheduler,
        }
    }

    /// Build the two schedulers a checkpoint declares.
    #[must_use]
    pub fn schedulers_from(cfg: &H3Config) -> (H3Scheduler, H3Scheduler) {
        let v = H3Scheduler::new(cfg.scheduler.shift).unwrap_or_else(|_| H3Scheduler::video());
        let a =
            H3Scheduler::new(cfg.audio_scheduler.shift).unwrap_or_else(|_| H3Scheduler::audio());
        (v, a)
    }

    #[must_use]
    pub fn config(&self) -> &H3TransformerConfig {
        &self.cfg
    }

    /// Run the sampling loop.
    pub fn sample(
        &mut self,
        request: &H3Request,
        layout: &PackedLayout,
        conditioning: &H3TextConditioning,
        anchors: &H3Conditioning,
    ) -> Result<H3Latents> {
        request.validate()?;
        conditioning.check_against(self.cfg.text_dim)?;
        layout.validate()?;

        let vpd = self.cfg.video_patch_dim();
        let aic = self.cfg.audio_in_channels;
        let n_video = layout.video_indices.len();
        let n_audio = layout.audio_indices.len();
        let cond_v = layout.num_condition_video_rows;
        let cond_a = layout.num_condition_audio_rows;

        ensure!(
            anchors.video_rows.len() == cond_v * vpd,
            "conditioning holds {} video values for {cond_v} rows of {vpd}",
            anchors.video_rows.len()
        );
        ensure!(
            anchors.audio_rows.len() == cond_a * aic,
            "conditioning holds {} audio values for {cond_a} rows of {aic}",
            anchors.audio_rows.len()
        );

        if let Some(s) = request.flow_shift {
            self.video_scheduler.set_shift(s)?;
        }
        if let Some(s) = request.audio_flow_shift {
            self.audio_scheduler.set_shift(s)?;
        }
        self.video_scheduler
            .set_timesteps(request.num_inference_steps)?;
        self.audio_scheduler
            .set_timesteps(request.num_inference_steps)?;
        self.video_scheduler.reset();
        self.audio_scheduler.reset();

        let steps = self
            .video_scheduler
            .num_inference_steps()
            .min(self.audio_scheduler.num_inference_steps());
        ensure!(steps >= 1, "the schedules collapsed to zero evaluations");

        // Pure noise for the generated rows, the anchors written into the
        // conditioning prefix.
        let mut video = noise(n_video * vpd, request.seed);
        let mut audio = noise(n_audio * aic, request.seed ^ 0x5EED_A0D1);
        video[..cond_v * vpd].copy_from_slice(&anchors.video_rows);
        audio[..cond_a * aic].copy_from_slice(&anchors.audio_rows);

        // The rotary tables and the RoPE grid depend only on the geometry.
        let tables = RopeTables::build(
            &layout.flat_position_ids(),
            self.cfg.rope_freq_dim,
            self.cfg.rope_theta,
        )?;

        for step in 0..steps {
            let t_video = self.video_scheduler.timesteps()[step];
            let t_audio = self.audio_scheduler.timesteps()[step];
            // Visual anchors sit just short of clean, but never *ahead* of the
            // rows they condition.
            let t_cond_video = t_video.max(KEYFRAME_NOISE_AUG);
            let rows = layout::build_row_timesteps(layout, t_video, t_audio, t_cond_video, 1.0)?;
            let dl = H3DitLayout::new(layout, &rows, &self.cfg)?;

            let out = self.dit.forward(&H3DitInputs {
                video_rows: &video,
                audio_rows: &audio,
                text_rows: &conditioning.hidden,
                cos: &tables.cos,
                sin: &tables.sin,
                layout: &dl,
            })?;

            video = self.video_scheduler.step(&out.video, t_video, &video)?;
            audio = self.audio_scheduler.step(&out.audio, t_audio, &audio)?;

            // The DiT predicts a velocity for *every* row, conditioning rows
            // included. Restore them so the anchors do not drift.
            video[..cond_v * vpd].copy_from_slice(&anchors.video_rows);
            audio[..cond_a * aic].copy_from_slice(&anchors.audio_rows);
        }

        Ok(H3Latents {
            video,
            audio,
            num_condition_video_rows: cond_v,
            num_condition_audio_rows: cond_a,
        })
    }

    /// Reassemble generated audio rows into the two channel-major streams.
    ///
    /// H3 packs stereo as two contiguous blocks of `num_audio_latents` rows.
    pub fn split_audio_channels(
        rows: &[f32],
        audio_channels: usize,
        num_audio_latents: usize,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            rows.len() == AUDIO_CHANNELS * num_audio_latents * audio_channels,
            "audio rows hold {} values, expected {} channels × {num_audio_latents} latents × {audio_channels}",
            rows.len(),
            AUDIO_CHANNELS
        );
        // Rows are channel-major; each row carries `audio_channels` latent
        // features. Transpose to `[latent_channels][frames]` per audio channel.
        let mut out = Vec::with_capacity(AUDIO_CHANNELS);
        for ch in 0..AUDIO_CHANNELS {
            let mut plane = vec![0.0f32; audio_channels * num_audio_latents];
            for t in 0..num_audio_latents {
                let row = (ch * num_audio_latents + t) * audio_channels;
                for c in 0..audio_channels {
                    plane[c * num_audio_latents + t] = rows[row + c];
                }
            }
            out.push(plane);
        }
        Ok(out)
    }
}

/// Deterministic standard-normal noise from a counter-based generator.
///
/// A splitmix64 stream feeding a Box-Muller transform: reproducible from a seed
/// without a random-number dependency, which is what makes a run repeatable.
#[must_use]
pub fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Open interval, so the log below never sees zero.
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next();
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> H3Geometry {
        H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap()
    }

    #[test]
    fn task_partitions_and_names() {
        assert_eq!(H3Task::T2VA.partition(), DitPartition::Base);
        assert_eq!(H3Task::I2VA.partition(), DitPartition::Base);
        assert_eq!(H3Task::FL2VA.partition(), DitPartition::Base);
        assert_eq!(H3Task::Ref2VA.partition(), DitPartition::Reference);
        for t in [H3Task::T2VA, H3Task::I2VA, H3Task::FL2VA, H3Task::Ref2VA] {
            assert_eq!(H3Task::parse(t.as_str()).unwrap(), t);
        }
        assert!(H3Task::parse("t2v").is_err());
    }

    #[test]
    fn request_validation_matches_each_task() {
        let g = geom();
        let mut r = H3Request::t2va(g, 32);
        assert!(r.validate().is_ok());

        r.task = H3Task::I2VA;
        assert!(r.validate().is_err(), "i2va needs one keyframe");
        r.keyframe_anchors = vec![KeyframeAnchor::First];
        assert!(r.validate().is_ok());

        r.task = H3Task::FL2VA;
        assert!(r.validate().is_err(), "fl2va needs two keyframes");
        r.keyframe_anchors = vec![KeyframeAnchor::First, KeyframeAnchor::Last];
        assert!(r.validate().is_ok());

        r.task = H3Task::Ref2VA;
        assert!(r.validate().is_err(), "ref2va needs a reference");
        r.references = vec![H3Reference::Audio { audio_rows: 4 }];
        assert!(r.validate().is_ok());

        r.num_inference_steps = 1;
        assert!(
            r.validate().is_err(),
            "a schedule needs at least two points"
        );
    }

    #[test]
    fn layout_routes_ref2va_to_its_own_builder() {
        let g = geom();
        let cond = crate::text_encoder::placeholder_conditioning(4, 5120);
        let mut r = H3Request::t2va(g, 8);
        let l1 = r.build_layout(&cond, [1, 2, 2]).unwrap();
        assert_eq!(l1.num_condition_video_rows, 0);

        r.task = H3Task::Ref2VA;
        r.references = vec![H3Reference::Image {
            latent_frames: 1,
            latent_height: 16,
            latent_width: 16,
        }];
        let l2 = r.build_layout(&cond, [1, 2, 2]).unwrap();
        assert_eq!(l2.num_condition_video_rows, 64);
    }

    #[test]
    fn noise_is_deterministic_and_standard_normal_ish() {
        let a = noise(4096, 7);
        let b = noise(4096, 7);
        let c = noise(4096, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|v| v.is_finite()));

        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.06, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.12, "variance = {var}");
    }

    #[test]
    fn noise_handles_odd_lengths() {
        // Box-Muller emits pairs; an odd request must not overrun.
        assert_eq!(noise(1, 3).len(), 1);
        assert_eq!(noise(7, 3).len(), 7);
    }

    #[test]
    fn generated_slices_skip_the_conditioning_prefix() {
        let l = H3Latents {
            video: (0..12).map(|i| i as f32).collect(),
            audio: (0..8).map(|i| i as f32).collect(),
            num_condition_video_rows: 1,
            num_condition_audio_rows: 2,
        };
        assert_eq!(
            l.generated_video(4),
            &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(l.generated_audio(2), &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn audio_channel_split_is_channel_major() {
        // 2 audio channels x 3 latents, each row carrying 2 features.
        let rows: Vec<f32> = (0..2 * 3 * 2).map(|i| i as f32).collect();
        let planes = H3Pipeline::split_audio_channels(&rows, 2, 3).unwrap();
        assert_eq!(planes.len(), 2);
        // Channel 0 rows are 0..3 -> features [0,1],[2,3],[4,5]
        assert_eq!(planes[0], vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        // Channel 1 rows are 3..6 -> features [6,7],[8,9],[10,11]
        assert_eq!(planes[1], vec![6.0, 8.0, 10.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn audio_channel_split_rejects_a_ragged_buffer() {
        assert!(H3Pipeline::split_audio_channels(&[0.0; 5], 2, 3).is_err());
    }
}
