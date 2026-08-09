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

//! The packed sequence MiniMax-H3 attends over.
//!
//! H3 runs **one** stack of blocks over a single 1-D sequence holding the text
//! condition, the conditioning image / video rows, the audio rows and the target
//! video rows. Attention is full self-attention over that sequence — there is no
//! cross-attention and no per-modality block weights anywhere in the model.
//! Everything modality-specific lives in the two input patch projections, the
//! per-row AdaLN tag, and the two output heads.
//!
//! That makes this module the contract the DiT is addressed through. It builds:
//!
//! - `position_ids` — the `(t, h, w)` rotary coordinate of every row, in `f64`.
//! - `token_tags` — `0` video, `1` text, `2` audio.
//! - `video_indices` / `audio_indices` / `text_indices` — where each modality's
//!   rows sit in the packed sequence.
//! - per-row timestep indices, since one forward serves rows at several noise
//!   levels at once.
//!
//! Two layouts exist. `t2va` / `i2va` / `fl2va` use
//! [`build_packed_sequence`]: `[text | keyframe conditions | target audio |
//! target video]`. `ref2va` uses [`build_ref2va_packed_sequence`]:
//! `[text | reference blocks | target audio | target video]`, where each
//! reference block advances a shared rotary clock.
//!
//! The rotary grid is computed in `f64` throughout because the reference does,
//! and the angles it feeds are sensitive to the last few bits.

use crate::config::{MODALITY_NUM, Modality};
use anyhow::{Result, bail, ensure};

/// MiniMax-H3 generates at a fixed 24 fps.
pub const FPS: f64 = 24.0;
/// Shortest video the released checkpoint generates, in seconds.
pub const MIN_DURATION: f64 = 5.0;
/// Longest video the released checkpoint generates, in seconds.
pub const MAX_DURATION: f64 = 15.0;
/// Aspect-ratio range the released checkpoint was trained over.
pub const MIN_ASPECT_RATIO: f64 = 0.25;
/// Aspect-ratio range the released checkpoint was trained over.
pub const MAX_ASPECT_RATIO: f64 = 4.0;
/// The audio VAE hops 800 samples at 32 kHz.
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
/// The generated soundtrack is stereo, packed channel-major.
pub const AUDIO_CHANNELS: usize = 2;
/// Pixel frames the video VAE encodes per chunk (its `clip_length`).
pub const VAE_FRAMES_PER_CHUNK: usize = 17;
/// Latent frames a chunk keeps (the VAE's `tokens_chunk_size`).
pub const VAE_LATENTS_PER_CHUNK: usize = 5;
/// Default short edge of the generated canvas.
pub const CANVAS_SHORT_EDGE: usize = 768;
/// Default area cap of the generated canvas.
pub const CANVAS_MAX_PIXELS: usize = 768 * 1344;
/// The `t` a visual conditioning anchor is pinned at — just short of clean,
/// because the released model was trained with slightly noised anchors.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;
/// Seed the conditioning posterior is sampled under, fixed in the reference so
/// the same keyframe always encodes to the same anchor.
pub const KEYFRAME_ENCODE_SEED: u64 = 42;
/// Per-channel mean the video VAE's input is normalized by (ImageNet).
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Per-channel standard deviation the video VAE's input is normalized by.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// One latent frame spans `5/3 * frames_per_latent` rotary units.
const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping.
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
/// The spatial axes are normalized by sqrt(area) and scaled by this.
const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// Which end of the video a keyframe conditioning block is anchored to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeAnchor {
    /// Anchored at the first latent frame.
    First,
    /// Anchored at the last latent frame.
    Last,
}

/// One `ref2va` reference block, carrying its own latent geometry.
///
/// Modelling the geometry on the reference itself (rather than as parallel
/// lists, the way the Python blocks pass it) removes the possibility of the
/// reference list and the encoded-latent list falling out of step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum H3Reference {
    /// A still image: a single frame occupying one integer rotary slot.
    Image {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
    },
    /// A standalone soundtrack. `audio_rows` counts **all** channels.
    Audio { audio_rows: usize },
    /// A video, optionally with a soundtrack packed immediately before it.
    Video {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        /// `0` when the reference carries no audio.
        audio_rows: usize,
    },
}

/// The resolved geometry of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H3Geometry {
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub num_audio_latents: usize,
}

impl H3Geometry {
    /// Resolve a request into the geometry every later stage keys off.
    ///
    /// `num_frames` is snapped **up** to the next `17 * n + 5` the video VAE can
    /// encode; the resulting duration must land within
    /// [`MIN_DURATION`]..=[`MAX_DURATION`].
    pub fn resolve(
        height: usize,
        width: usize,
        num_frames: usize,
        spatial_compression: usize,
        patch_w: usize,
    ) -> Result<Self> {
        let multiple = spatial_compression * patch_w;
        ensure!(
            multiple > 0,
            "canvas multiple must be positive (spatial_compression {spatial_compression} × patch_w {patch_w})"
        );
        if !height.is_multiple_of(multiple) || !width.is_multiple_of(multiple) {
            bail!("`height` and `width` must be multiples of {multiple}, got {height}x{width}");
        }
        let num_frames = align_num_frames(num_frames)?;
        let duration = num_frames as f64 / FPS;
        if !(MIN_DURATION..=MAX_DURATION).contains(&duration) {
            bail!(
                "MiniMax-H3 generates between {MIN_DURATION} and {MAX_DURATION} seconds at {FPS} fps; \
                 `num_frames` rounded up to {num_frames} is {duration:.3} s"
            );
        }
        Ok(Self {
            height,
            width,
            num_frames,
            num_latent_frames: video_latent_num_frames(num_frames)?,
            latent_height: height / spatial_compression,
            latent_width: width / spatial_compression,
            num_audio_latents: audio_latent_num_frames(num_frames),
        })
    }

    /// Packed rows one latent frame contributes.
    #[must_use]
    pub fn rows_per_frame(&self, patch_h: usize, patch_w: usize) -> usize {
        (self.latent_height / patch_h) * (self.latent_width / patch_w)
    }
}

/// Resolve a display aspect ratio into a canvas.
///
/// The short edge starts at `short_edge`, the area is capped at `max_pixels`,
/// and both axes are then rounded to the nearest `canvas_multiple` — so the
/// final area can land slightly above the pre-rounding budget. Only the *ratio*
/// of the first two arguments matters.
///
/// Returns `(height, width)`.
pub fn resolve_canvas_size(
    aspect_width: f64,
    aspect_height: f64,
    canvas_multiple: usize,
    short_edge: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(
        aspect_width > 0.0 && aspect_height > 0.0,
        "the aspect ratio must be positive, got {aspect_width}:{aspect_height}"
    );
    ensure!(canvas_multiple > 0, "canvas_multiple must be positive");
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        bail!(
            "MiniMax-H3 supports aspect ratios from 1:{} to {MAX_ASPECT_RATIO}:1, got {aspect_width}:{aspect_height} ({ratio})",
            1.0 / MIN_ASPECT_RATIO
        );
    }

    let (mut width, mut height) = if ratio >= 1.0 {
        (short_edge as f64 * ratio, short_edge as f64)
    } else {
        (short_edge as f64, short_edge as f64 / ratio)
    };
    let area = width * height;
    if area > max_pixels as f64 {
        let scale = (max_pixels as f64 / area).sqrt();
        width *= scale;
        height *= scale;
    }
    let m = canvas_multiple as f64;
    let h = (round_half_even(height / m) * m) as usize;
    let w = (round_half_even(width / m) * m) as usize;
    Ok((h.max(canvas_multiple), w.max(canvas_multiple)))
}

/// Snap a frame count up to the next `17 * n + 5` the video VAE can encode.
pub fn align_num_frames(num_frames: usize) -> Result<usize> {
    ensure!(num_frames >= 1, "`num_frames` must be positive");
    let mut n = num_frames;
    while n % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK {
        n += 1;
    }
    Ok(n)
}

/// Latent frames the video VAE produces for an aligned frame count: `5 * n + 2`.
pub fn video_latent_num_frames(num_frames: usize) -> Result<usize> {
    if num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK {
        bail!(
            "`num_frames` must be of the form {VAE_FRAMES_PER_CHUNK} * n + {VAE_LATENTS_PER_CHUNK}, got {num_frames}"
        );
    }
    Ok((num_frames - VAE_LATENTS_PER_CHUNK) / VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2)
}

/// Audio latents covering a video of `num_frames` frames.
#[must_use]
pub fn audio_latent_num_frames(num_frames: usize) -> usize {
    (num_frames as f64 / FPS * AUDIO_LATENTS_PER_SECOND as f64).round() as usize
}

/// Pack video latents into transformer rows.
///
/// `latents` is `(channels, num_frames, height, width)` in channel-major
/// (C, T, H, W) order. Output rows are ordered frame-major then row-major, each
/// `channels * prod(patch_size)` wide, matching the reference's
/// `permute(0, 2, 4, 6, 1, 3, 5, 7)`.
pub fn patchify_video_latents(
    latents: &[f32],
    channels: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    patch_size: [usize; 3],
) -> Result<Vec<f32>> {
    let [pt, ph, pw] = patch_size;
    ensure!(pt > 0 && ph > 0 && pw > 0, "patch size must be non-zero");
    if !num_frames.is_multiple_of(pt) || !height.is_multiple_of(ph) || !width.is_multiple_of(pw) {
        bail!(
            "latents ({channels}, {num_frames}, {height}, {width}) are not divisible by the patch {patch_size:?}"
        );
    }
    ensure!(
        latents.len() == channels * num_frames * height * width,
        "latents len {} != C*T*H*W = {}",
        latents.len(),
        channels * num_frames * height * width
    );
    let (ft, fh, fw) = (num_frames / pt, height / ph, width / pw);
    let row_dim = channels * pt * ph * pw;
    let mut out = vec![0.0f32; ft * fh * fw * row_dim];

    for t in 0..ft {
        for hh in 0..fh {
            for ww in 0..fw {
                let row = ((t * fh + hh) * fw + ww) * row_dim;
                // Row layout is (channels, pt, ph, pw) — the trailing axes of
                // the reference permutation.
                for c in 0..channels {
                    for dt in 0..pt {
                        for dh in 0..ph {
                            for dw in 0..pw {
                                let src = ((c * num_frames + t * pt + dt) * height + hh * ph + dh)
                                    * width
                                    + ww * pw
                                    + dw;
                                let dst = ((c * pt + dt) * ph + dh) * pw + dw;
                                out[row + dst] = latents[src];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Inverse of [`patchify_video_latents`] — rows back to `(C, T, H, W)`.
pub fn unpatchify_video_latents(
    rows: &[f32],
    channels: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    patch_size: [usize; 3],
) -> Result<Vec<f32>> {
    let [pt, ph, pw] = patch_size;
    if !num_frames.is_multiple_of(pt) || !height.is_multiple_of(ph) || !width.is_multiple_of(pw) {
        bail!(
            "target ({num_frames}, {height}, {width}) is not divisible by the patch {patch_size:?}"
        );
    }
    let (ft, fh, fw) = (num_frames / pt, height / ph, width / pw);
    let row_dim = channels * pt * ph * pw;
    ensure!(
        rows.len() == ft * fh * fw * row_dim,
        "rows len {} != {} patches × {row_dim}",
        rows.len(),
        ft * fh * fw
    );
    let mut out = vec![0.0f32; channels * num_frames * height * width];
    for t in 0..ft {
        for hh in 0..fh {
            for ww in 0..fw {
                let row = ((t * fh + hh) * fw + ww) * row_dim;
                for c in 0..channels {
                    for dt in 0..pt {
                        for dh in 0..ph {
                            for dw in 0..pw {
                                let dst = ((c * num_frames + t * pt + dt) * height + hh * ph + dh)
                                    * width
                                    + ww * pw
                                    + dw;
                                let src = ((c * pt + dt) * ph + dh) * pw + dw;
                                out[dst] = rows[row + src];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// One aspect-normalized spatial rotary axis.
///
/// `dim / patch` coordinates centred on the unit interval and scaled by 32, with
/// the right endpoint excluded, so a square canvas spans `[0, 32)`.
///
/// The reference builds this with `np.linspace(..., endpoint=False)`, which is
/// `start + arange(num) * (stop - start) / num` — *not* what `torch.linspace`
/// computes — and the `f64` grid has to be reproduced exactly.
fn spatial_position_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let num = dim / patch;
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    if num == 0 {
        return Vec::new();
    }
    let step = ratio / num as f64;
    (0..num)
        .map(|i| (i as f64 * step + left) * ROPE_SPATIAL_SCALE)
        .collect()
}

/// The rotary time of every latent frame, starting at `origin`.
///
/// Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`, repeating.
fn temporal_position_grid(num_latent_frames: usize, origin: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(num_latent_frames);
    let mut acc = origin;
    for i in 0..num_latent_frames {
        out.push(acc);
        acc += frame_span(i);
    }
    out
}

/// Rotary units latent frame `i` spans.
#[must_use]
fn frame_span(i: usize) -> f64 {
    ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()]
}

/// The `(h, w)` rotary coordinates of one latent frame, plus the width axis.
///
/// Returns `(frame_grid, width_grid)` where `frame_grid` is `rows_per_frame`
/// pairs in row-major `(h, w)` order.
fn frame_position_grid(
    latent_height: usize,
    latent_width: usize,
    patch_h: usize,
    patch_w: usize,
) -> (Vec<[f64; 2]>, Vec<f64>) {
    let sqrt_area = ((latent_height * latent_width) as f64).sqrt();
    let height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area);
    let width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area);
    let mut grid = Vec::with_capacity(height_grid.len() * width_grid.len());
    for &h in &height_grid {
        for &w in &width_grid {
            grid.push([h, w]);
        }
    }
    (grid, width_grid)
}

/// Sum a series the way numpy does — pairwise, with an 8-accumulator inner loop.
///
/// The `"last"` keyframe anchor sums the per-frame rotary spans through
/// `np.sum`, while the `ref2va` soundtrack span sums the same series
/// sequentially. The two orders differ in the last ulp from 16 latent frames
/// onwards, and the reference keeps both — one per call site — so this port
/// does too.
fn pairwise_sum(a: &[f64]) -> f64 {
    const BLOCKSIZE: usize = 128;
    let n = a.len();
    if n < 8 {
        return a.iter().sum();
    }
    if n <= BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// Total rotary time `num_latent_frames` frames span, summed pairwise.
fn temporal_span_pairwise(num_latent_frames: usize) -> f64 {
    let spans: Vec<f64> = (0..num_latent_frames).map(frame_span).collect();
    pairwise_sum(&spans)
}

/// Total rotary time `num_latent_frames` frames span, summed sequentially.
fn temporal_span_sequential(num_latent_frames: usize) -> f64 {
    (0..num_latent_frames).map(frame_span).sum()
}

/// The packed layout of one request.
#[derive(Debug, Clone)]
pub struct PackedLayout {
    /// The `(t, h, w)` rotary coordinate of every row.
    pub position_ids: Vec<[f64; 3]>,
    /// The modality tag of every row: `0` video, `1` text, `2` audio.
    pub token_tags: Vec<u32>,
    /// Sequence positions of the video rows, conditioning rows first.
    pub video_indices: Vec<usize>,
    /// Sequence positions of the audio rows, reference rows first.
    pub audio_indices: Vec<usize>,
    /// Sequence positions of the text rows.
    pub text_indices: Vec<usize>,
    /// How many leading video rows are conditioning rather than generated.
    pub num_condition_video_rows: usize,
    /// How many leading audio rows are references rather than generated.
    pub num_condition_audio_rows: usize,
}

impl PackedLayout {
    #[must_use]
    pub fn sequence_length(&self) -> usize {
        self.position_ids.len()
    }

    /// Flatten `position_ids` to the `(seq_len * 3)` buffer the RoPE tables are
    /// built from.
    #[must_use]
    pub fn flat_position_ids(&self) -> Vec<f64> {
        self.position_ids.iter().flatten().copied().collect()
    }

    /// Check the invariants the DiT relies on.
    pub fn validate(&self) -> Result<()> {
        let n = self.sequence_length();
        ensure!(
            self.token_tags.len() == n,
            "token_tags len {} != sequence length {n}",
            self.token_tags.len()
        );
        let covered = self.video_indices.len() + self.audio_indices.len() + self.text_indices.len();
        ensure!(
            covered == n,
            "video ({}) + audio ({}) + text ({}) rows = {covered}, expected {n}",
            self.video_indices.len(),
            self.audio_indices.len(),
            self.text_indices.len()
        );
        let mut seen = vec![false; n];
        for &i in self
            .video_indices
            .iter()
            .chain(&self.audio_indices)
            .chain(&self.text_indices)
        {
            ensure!(i < n, "row index {i} out of range for sequence length {n}");
            ensure!(!seen[i], "row {i} is claimed by more than one modality");
            seen[i] = true;
        }
        for (i, &t) in self.token_tags.iter().enumerate() {
            ensure!(
                (t as usize) < MODALITY_NUM,
                "row {i} has modality tag {t}, expected < {MODALITY_NUM}"
            );
        }
        ensure!(
            self.num_condition_video_rows <= self.video_indices.len(),
            "num_condition_video_rows {} exceeds {} video rows",
            self.num_condition_video_rows,
            self.video_indices.len()
        );
        ensure!(
            self.num_condition_audio_rows <= self.audio_indices.len(),
            "num_condition_audio_rows {} exceeds {} audio rows",
            self.num_condition_audio_rows,
            self.audio_indices.len()
        );
        Ok(())
    }
}

/// Build the `[text | keyframe conditions | target audio | target video]`
/// layout used by `t2va`, `i2va` and `fl2va`.
///
/// `text_token_tags` is the modality tag of every text row: text is tagged `1`,
/// except the rows of a keyframe's vision block, which H3 tags `0` (video).
/// Pass an empty `keyframe_anchors` for `t2va`.
pub fn build_packed_sequence(
    text_token_tags: &[u32],
    geometry: &H3Geometry,
    patch_size: [usize; 3],
    keyframe_anchors: &[KeyframeAnchor],
) -> Result<PackedLayout> {
    let [_, patch_h, patch_w] = patch_size;
    let rows_per_frame = geometry.rows_per_frame(patch_h, patch_w);
    ensure!(
        rows_per_frame > 0,
        "the latent canvas is smaller than one patch"
    );

    let num_text_tokens = text_token_tags.len();
    let num_condition_rows = keyframe_anchors.len() * rows_per_frame;
    let num_audio_rows = geometry.num_audio_latents * AUDIO_CHANNELS;
    let num_video_rows = geometry.num_latent_frames * rows_per_frame;
    let sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows;

    let condition_start = num_text_tokens;
    let audio_start = condition_start + num_condition_rows;
    let video_start = audio_start + num_audio_rows;

    let mut position_ids = vec![[0.0f64; 3]; sequence_length];
    for (i, p) in position_ids.iter_mut().take(num_text_tokens).enumerate() {
        p[0] = i as f64;
    }

    let (frame_grid, width_grid) = frame_position_grid(
        geometry.latent_height,
        geometry.latent_width,
        patch_h,
        patch_w,
    );
    ensure!(
        frame_grid.len() == rows_per_frame,
        "frame grid has {} rows, expected {rows_per_frame}",
        frame_grid.len()
    );

    // 1. Keyframe conditioning blocks.
    for (index, anchor) in keyframe_anchors.iter().enumerate() {
        let anchor_time = match anchor {
            KeyframeAnchor::First => num_text_tokens as f64,
            KeyframeAnchor::Last => {
                num_text_tokens as f64 + temporal_span_pairwise(geometry.num_latent_frames)
                    - ROPE_FRAME_RESCALE
            }
        };
        let start = condition_start + index * rows_per_frame;
        for (k, g) in frame_grid.iter().enumerate() {
            position_ids[start + k] = [anchor_time, g[0], g[1]];
        }
    }

    // 2. Target audio: channel-major, sharing the video's rotary clock. Audio
    // rows carry no height coordinate and are pinned to the two extremes of the
    // width grid.
    fill_audio_positions(
        &mut position_ids,
        audio_start,
        geometry.num_audio_latents,
        num_text_tokens as f64,
        &width_grid,
    )?;

    // 3. Target video.
    let frame_time = temporal_position_grid(geometry.num_latent_frames, num_text_tokens as f64);
    for (f, &t) in frame_time.iter().enumerate() {
        let base = video_start + f * rows_per_frame;
        for (k, g) in frame_grid.iter().enumerate() {
            position_ids[base + k] = [t, g[0], g[1]];
        }
    }

    // 4. Row indices and modality tags.
    let video_indices: Vec<usize> = (condition_start..audio_start)
        .chain(video_start..sequence_length)
        .collect();
    let audio_indices: Vec<usize> = (audio_start..video_start).collect();
    let text_indices: Vec<usize> = (0..num_text_tokens).collect();

    let token_tags = assemble_tags(
        sequence_length,
        text_token_tags,
        &text_indices,
        &audio_indices,
        &video_indices,
    )?;

    let layout = PackedLayout {
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows: num_condition_rows,
        num_condition_audio_rows: 0,
    };
    layout.validate()?;
    Ok(layout)
}

/// Build the `[text | reference blocks | target audio | target video]` layout of
/// the `ref2va` task.
///
/// `rotary_time` is a clock shared by audio and video: it starts where the text
/// rows end and each reference block pushes it forward by the time that block
/// occupies.
pub fn build_ref2va_packed_sequence(
    text_token_tags: &[u32],
    references: &[H3Reference],
    geometry: &H3Geometry,
    patch_size: [usize; 3],
) -> Result<PackedLayout> {
    let [_, patch_h, patch_w] = patch_size;
    let num_text_tokens = text_token_tags.len();
    let target_rows_per_frame = geometry.rows_per_frame(patch_h, patch_w);
    ensure!(
        target_rows_per_frame > 0,
        "the latent canvas is smaller than one patch"
    );
    let num_target_video_rows = geometry.num_latent_frames * target_rows_per_frame;
    let num_target_audio_rows = geometry.num_audio_latents * AUDIO_CHANNELS;

    let rows_of = |frames: usize, h: usize, w: usize| frames * (h / patch_h) * (w / patch_w);

    let mut num_reference_video_rows = 0usize;
    let mut num_reference_audio_rows = 0usize;
    for r in references {
        match *r {
            H3Reference::Image {
                latent_frames,
                latent_height,
                latent_width,
            } => num_reference_video_rows += rows_of(latent_frames, latent_height, latent_width),
            H3Reference::Audio { audio_rows } => num_reference_audio_rows += audio_rows,
            H3Reference::Video {
                latent_frames,
                latent_height,
                latent_width,
                audio_rows,
            } => {
                num_reference_video_rows += rows_of(latent_frames, latent_height, latent_width);
                num_reference_audio_rows += audio_rows;
            }
        }
    }

    let sequence_length = num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows;

    let mut position_ids = vec![[0.0f64; 3]; sequence_length];
    for (i, p) in position_ids.iter_mut().take(num_text_tokens).enumerate() {
        p[0] = i as f64;
    }
    let (target_frame_grid, target_width_grid) = frame_position_grid(
        geometry.latent_height,
        geometry.latent_width,
        patch_h,
        patch_w,
    );

    let mut video_indices: Vec<usize> = Vec::new();
    let mut audio_indices: Vec<usize> = Vec::new();
    let mut cursor = num_text_tokens;
    let mut rotary_time = num_text_tokens as f64;

    for reference in references {
        match *reference {
            H3Reference::Image {
                latent_frames,
                latent_height,
                latent_width,
            } => {
                let n_rows = rows_of(latent_frames, latent_height, latent_width);
                let (frame_grid, _) =
                    frame_position_grid(latent_height, latent_width, patch_h, patch_w);
                ensure!(
                    !frame_grid.is_empty() && n_rows % frame_grid.len() == 0,
                    "image reference geometry {latent_frames}x{latent_height}x{latent_width} does not tile the patch"
                );
                for f in 0..latent_frames {
                    for (k, g) in frame_grid.iter().enumerate() {
                        position_ids[cursor + f * frame_grid.len() + k] = [rotary_time, g[0], g[1]];
                    }
                }
                video_indices.extend(cursor..cursor + n_rows);
                cursor += n_rows;
                // An image takes a single integer rotary slot, not a latent
                // frame's 5/3 units.
                rotary_time += 1.0;
            }
            H3Reference::Audio { audio_rows } => {
                ensure!(
                    audio_rows % AUDIO_CHANNELS == 0,
                    "audio reference has {audio_rows} rows, not a multiple of {AUDIO_CHANNELS} channels"
                );
                let latents = audio_rows / AUDIO_CHANNELS;
                fill_audio_positions(
                    &mut position_ids,
                    cursor,
                    latents,
                    rotary_time,
                    &target_width_grid,
                )?;
                audio_indices.extend(cursor..cursor + audio_rows);
                cursor += audio_rows;
                rotary_time += latents as f64;
            }
            H3Reference::Video {
                latent_frames,
                latent_height,
                latent_width,
                audio_rows,
            } => {
                ensure!(
                    audio_rows % AUDIO_CHANNELS == 0,
                    "video reference has {audio_rows} audio rows, not a multiple of {AUDIO_CHANNELS} channels"
                );
                let latents = audio_rows / AUDIO_CHANNELS;
                let n_video_rows = rows_of(latent_frames, latent_height, latent_width);
                let (frame_grid, width_grid) =
                    frame_position_grid(latent_height, latent_width, patch_h, patch_w);
                ensure!(!frame_grid.is_empty(), "video reference tiles to zero rows");

                // A video reference's soundtrack rows are packed immediately
                // before its video rows and share their origin, exactly as the
                // generated audio and video do.
                let audio_at = cursor;
                fill_audio_positions(
                    &mut position_ids,
                    audio_at,
                    latents,
                    rotary_time,
                    &width_grid,
                )?;
                audio_indices.extend(audio_at..audio_at + audio_rows);
                let video_at = audio_at + audio_rows;

                let frame_time = temporal_position_grid(latent_frames, rotary_time);
                for (f, &t) in frame_time.iter().enumerate() {
                    let base = video_at + f * frame_grid.len();
                    for (k, g) in frame_grid.iter().enumerate() {
                        position_ids[base + k] = [t, g[0], g[1]];
                    }
                }
                video_indices.extend(video_at..video_at + n_video_rows);
                cursor = video_at + n_video_rows;

                // Summed sequentially here — see `pairwise_sum`.
                let video_span = temporal_span_sequential(latent_frames);
                rotary_time += (latents as f64).max(video_span);
            }
        }
    }

    // The generated rows share the origin the reference blocks left behind.
    let audio_start = cursor;
    let video_start = audio_start + num_target_audio_rows;
    fill_audio_positions(
        &mut position_ids,
        audio_start,
        geometry.num_audio_latents,
        rotary_time,
        &target_width_grid,
    )?;
    let frame_time = temporal_position_grid(geometry.num_latent_frames, rotary_time);
    for (f, &t) in frame_time.iter().enumerate() {
        let base = video_start + f * target_rows_per_frame;
        for (k, g) in target_frame_grid.iter().enumerate() {
            position_ids[base + k] = [t, g[0], g[1]];
        }
    }

    video_indices.extend(video_start..sequence_length);
    audio_indices.extend(audio_start..video_start);
    let text_indices: Vec<usize> = (0..num_text_tokens).collect();

    let token_tags = assemble_tags(
        sequence_length,
        text_token_tags,
        &text_indices,
        &audio_indices,
        &video_indices,
    )?;

    let layout = PackedLayout {
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows: num_reference_video_rows,
        num_condition_audio_rows: num_reference_audio_rows,
    };
    layout.validate()?;
    Ok(layout)
}

/// Place one channel-major audio block.
///
/// Audio rows carry no height coordinate and are pinned to the two extremes of
/// the width grid of *their own* block — the target grid for a standalone audio
/// reference, the video's grid for a soundtrack.
fn fill_audio_positions(
    position_ids: &mut [[f64; 3]],
    start: usize,
    num_audio_latents: usize,
    rotary_time: f64,
    width_grid: &[f64],
) -> Result<()> {
    ensure!(
        !width_grid.is_empty(),
        "audio rows need a width grid to pin to"
    );
    let first = width_grid[0];
    let last = width_grid[width_grid.len() - 1];
    for ch in 0..AUDIO_CHANNELS {
        for i in 0..num_audio_latents {
            let row = start + ch * num_audio_latents + i;
            ensure!(
                row < position_ids.len(),
                "audio row {row} exceeds sequence length {}",
                position_ids.len()
            );
            position_ids[row] = [
                rotary_time + i as f64,
                0.0,
                if ch == 0 { first } else { last },
            ];
        }
    }
    Ok(())
}

fn assemble_tags(
    sequence_length: usize,
    text_token_tags: &[u32],
    text_indices: &[usize],
    audio_indices: &[usize],
    video_indices: &[usize],
) -> Result<Vec<u32>> {
    let mut tags = vec![Modality::Video.tag(); sequence_length];
    // Applied in the reference's order: text, then audio, then video — a text
    // row tagged `0` (a keyframe's vision block) keeps that tag because the
    // video pass only writes rows listed in `video_indices`.
    for (&i, &t) in text_indices.iter().zip(text_token_tags) {
        ensure!(i < sequence_length, "text row {i} out of range");
        tags[i] = t;
    }
    for &i in audio_indices {
        ensure!(i < sequence_length, "audio row {i} out of range");
        tags[i] = Modality::Audio.tag();
    }
    for &i in video_indices {
        ensure!(i < sequence_length, "video row {i} out of range");
        tags[i] = Modality::Video.tag();
    }
    Ok(tags)
}

/// The `(timestep, timestep_indices)` pair one DiT forward is driven by.
#[derive(Debug, Clone, PartialEq)]
pub struct RowTimesteps {
    /// The distinct timestep values present in the sequence, ascending.
    pub timesteps: Vec<f32>,
    /// For every row, its index into [`Self::timesteps`].
    pub indices: Vec<u32>,
}

/// Assign a timestep to every row and reduce it to the DiT's
/// `(timestep, timestep_indices)` pair.
///
/// One forward serves rows at several noise levels: the generated video and
/// audio rows step down their own schedules while the conditioning rows stay
/// pinned at their noise-augmentation level. Text rows never reach an output
/// head and inherit the video timestep.
pub fn build_row_timesteps(
    layout: &PackedLayout,
    video_timestep: f32,
    audio_timestep: f32,
    condition_video_timestep: f32,
    condition_audio_timestep: f32,
) -> Result<RowTimesteps> {
    let n = layout.sequence_length();
    let mut row = vec![video_timestep; n];
    for &i in layout
        .video_indices
        .iter()
        .take(layout.num_condition_video_rows)
    {
        row[i] = condition_video_timestep;
    }
    for &i in layout
        .audio_indices
        .iter()
        .skip(layout.num_condition_audio_rows)
    {
        row[i] = audio_timestep;
    }
    for &i in layout
        .audio_indices
        .iter()
        .take(layout.num_condition_audio_rows)
    {
        row[i] = condition_audio_timestep;
    }

    // `torch.unique(sorted=True, return_inverse=True)`.
    let mut distinct: Vec<f32> = row.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("timesteps are finite"));
    distinct.dedup();
    let indices = row
        .iter()
        .map(|t| {
            distinct
                .binary_search_by(|p| p.partial_cmp(t).expect("timesteps are finite"))
                .expect("every row timestep is in the distinct set") as u32
        })
        .collect();
    Ok(RowTimesteps {
        timesteps: distinct,
        indices,
    })
}

/// Round half to even, matching Python's `round`.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> H3Geometry {
        // 124 frames = 17*7 + 5, so 37 latent frames; 768x1344 / 16 = 48x84.
        H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap()
    }

    #[test]
    fn frame_alignment_and_latent_counts() {
        assert_eq!(align_num_frames(124).unwrap(), 124); // already 17*7+5
        assert_eq!(align_num_frames(120).unwrap(), 124);
        assert_eq!(align_num_frames(1).unwrap(), 5);
        assert_eq!(video_latent_num_frames(124).unwrap(), 37); // 5*7+2
        assert_eq!(video_latent_num_frames(5).unwrap(), 2);
        assert!(video_latent_num_frames(123).is_err());
        // 124 frames / 24 fps * 40 = 206.67 -> 207
        assert_eq!(audio_latent_num_frames(124), 207);
    }

    #[test]
    fn geometry_resolves_released_defaults() {
        let g = geom();
        assert_eq!((g.latent_height, g.latent_width), (48, 84));
        assert_eq!(g.num_latent_frames, 37);
        assert_eq!(g.num_audio_latents, 207);
        assert_eq!(g.rows_per_frame(2, 2), 24 * 42);
    }

    #[test]
    fn geometry_rejects_off_grid_canvas_and_duration() {
        assert!(H3Geometry::resolve(700, 1344, 124, 16, 2).is_err());
        // 60 frames -> 2.5 s, below the 5 s floor.
        assert!(H3Geometry::resolve(768, 1344, 60, 16, 2).is_err());
        // 400 frames -> above the 15 s ceiling.
        assert!(H3Geometry::resolve(768, 1344, 400, 16, 2).is_err());
    }

    #[test]
    fn canvas_resolution_is_16_9_by_default() {
        let (h, w) =
            resolve_canvas_size(16.0, 9.0, 32, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(w > h, "16:9 must be landscape, got {w}x{h}");
        assert!((w as f64 / h as f64 - 16.0 / 9.0).abs() < 0.05);
    }

    #[test]
    fn canvas_rejects_extreme_ratios() {
        assert!(resolve_canvas_size(10.0, 1.0, 32, 768, CANVAS_MAX_PIXELS).is_err());
        assert!(resolve_canvas_size(1.0, 10.0, 32, 768, CANVAS_MAX_PIXELS).is_err());
        assert!(resolve_canvas_size(0.0, 1.0, 32, 768, CANVAS_MAX_PIXELS).is_err());
    }

    #[test]
    fn patchify_round_trips() {
        let (c, t, h, w) = (3usize, 2usize, 4usize, 6usize);
        let src: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video_latents(&src, c, t, h, w, [1, 2, 2]).unwrap();
        assert_eq!(rows.len(), t * (h / 2) * (w / 2) * (c * 2 * 2));
        let back = unpatchify_video_latents(&rows, c, t, h, w, [1, 2, 2]).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn patchify_row_order_is_frame_major() {
        // One channel, 1 frame, 2x4 latents, patch 1x2x2 -> 2 rows of 4 values.
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect(); // rows [0..3],[4..7]
        let rows = patchify_video_latents(&src, 1, 1, 2, 4, [1, 2, 2]).unwrap();
        // First patch covers (h0..1, w0..1) = 0,1,4,5 laid out (ph, pw).
        assert_eq!(&rows[0..4], &[0.0, 1.0, 4.0, 5.0]);
        // Second patch covers (h0..1, w2..3) = 2,3,6,7.
        assert_eq!(&rows[4..8], &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn patchify_rejects_indivisible_shapes() {
        let src = vec![0.0f32; 3 * 3 * 4];
        assert!(patchify_video_latents(&src, 3, 1, 3, 4, [1, 2, 2]).is_err());
    }

    #[test]
    fn spatial_grid_spans_zero_to_32_on_square_canvas() {
        let g = spatial_position_grid(8, 2, 8.0); // square: ratio = 1
        assert_eq!(g.len(), 4);
        assert!((g[0] - 0.0).abs() < 1e-12);
        // endpoint excluded: last entry is 32 * (1 - 1/4)
        assert!((g[3] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn spatial_grid_is_aspect_centred() {
        // A wide canvas: the short axis is centred inside [0, 32).
        let sqrt_area = ((4 * 16) as f64).sqrt(); // 8
        let h = spatial_position_grid(4, 2, sqrt_area);
        let w = spatial_position_grid(16, 2, sqrt_area);
        // The height axis has ratio 0.5, so it starts at 32 * 0.25 = 8.
        assert!((h[0] - 8.0).abs() < 1e-12);
        // The width axis has ratio 2, so it starts at 32 * -0.5 = -16.
        assert!((w[0] + 16.0).abs() < 1e-12);
    }

    #[test]
    fn temporal_grid_spacing_follows_1_4_4_4_4() {
        let g = temporal_position_grid(6, 0.0);
        let r = ROPE_FRAME_RESCALE;
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - r).abs() < 1e-12); // + 5/3 * 1
        assert!((g[2] - (r + 4.0 * r)).abs() < 1e-12);
        assert!((g[3] - (r + 8.0 * r)).abs() < 1e-12);
        assert!((g[5] - (r + 16.0 * r)).abs() < 1e-12);
    }

    #[test]
    fn pairwise_sum_matches_sequential_on_small_input() {
        let v: Vec<f64> = (0..7).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&v), v.iter().sum::<f64>());
    }

    #[test]
    fn pairwise_and_sequential_spans_agree_closely() {
        // The two orders are kept apart deliberately; they must still agree to
        // within a few ulp.
        for n in [1usize, 5, 15, 16, 37, 100, 200] {
            let p = temporal_span_pairwise(n);
            let s = temporal_span_sequential(n);
            assert!(
                (p - s).abs() <= 1e-12 * p.abs().max(1.0),
                "n = {n}: pairwise {p} vs sequential {s}"
            );
        }
    }

    #[test]
    fn t2va_layout_shape_and_invariants() {
        let g = geom();
        let text_tags = vec![Modality::Text.tag(); 32];
        let l = build_packed_sequence(&text_tags, &g, [1, 2, 2], &[]).unwrap();
        let rows_per_frame = g.rows_per_frame(2, 2);
        assert_eq!(l.text_indices.len(), 32);
        assert_eq!(l.audio_indices.len(), g.num_audio_latents * AUDIO_CHANNELS);
        assert_eq!(l.video_indices.len(), g.num_latent_frames * rows_per_frame);
        assert_eq!(l.num_condition_video_rows, 0);
        assert_eq!(l.num_condition_audio_rows, 0);
        l.validate().unwrap();
        // Text rows sit on the time axis at their own index.
        for i in 0..32 {
            assert_eq!(l.position_ids[i][0], i as f64);
        }
        // Media rows continue the clock from where the text ends.
        let first_audio = l.audio_indices[0];
        assert_eq!(l.position_ids[first_audio][0], 32.0);
    }

    #[test]
    fn fl2va_layout_adds_condition_rows() {
        let g = geom();
        let text_tags = vec![Modality::Text.tag(); 8];
        let anchors = [KeyframeAnchor::First, KeyframeAnchor::Last];
        let l = build_packed_sequence(&text_tags, &g, [1, 2, 2], &anchors).unwrap();
        let rows_per_frame = g.rows_per_frame(2, 2);
        assert_eq!(l.num_condition_video_rows, 2 * rows_per_frame);
        assert_eq!(
            l.video_indices.len(),
            2 * rows_per_frame + g.num_latent_frames * rows_per_frame
        );
        l.validate().unwrap();
        // The "first" anchor sits at the text end; "last" sits later.
        let t_first = l.position_ids[l.video_indices[0]][0];
        let t_last = l.position_ids[l.video_indices[rows_per_frame]][0];
        assert_eq!(t_first, 8.0);
        assert!(t_last > t_first);
    }

    #[test]
    fn keyframe_vision_rows_keep_their_video_tag() {
        // A keyframe's vision block lives in the *text* stream but is tagged 0.
        let g = geom();
        let mut text_tags = vec![Modality::Text.tag(); 10];
        text_tags[3] = Modality::Video.tag();
        text_tags[4] = Modality::Video.tag();
        let l = build_packed_sequence(&text_tags, &g, [1, 2, 2], &[]).unwrap();
        assert_eq!(l.token_tags[3], Modality::Video.tag());
        assert_eq!(l.token_tags[4], Modality::Video.tag());
        assert_eq!(l.token_tags[0], Modality::Text.tag());
    }

    #[test]
    fn audio_rows_are_channel_major_and_pinned_to_width_extremes() {
        let g = geom();
        let l = build_packed_sequence(&[Modality::Text.tag(); 4], &g, [1, 2, 2], &[]).unwrap();
        let n = g.num_audio_latents;
        let a0 = l.audio_indices[0];
        // Channel 0 occupies the first n rows, channel 1 the next n.
        let w_first = l.position_ids[a0][2];
        let w_last = l.position_ids[a0 + n][2];
        assert_ne!(w_first, w_last, "the two channels pin to different widths");
        // Height is always zero for audio.
        for &i in &l.audio_indices {
            assert_eq!(l.position_ids[i][1], 0.0);
        }
        // Time advances by one unit per latent within a channel.
        assert_eq!(l.position_ids[a0 + 1][0] - l.position_ids[a0][0], 1.0);
        // ...and restarts for the second channel.
        assert_eq!(l.position_ids[a0 + n][0], l.position_ids[a0][0]);
    }

    #[test]
    fn ref2va_layout_orders_soundtrack_before_its_video() {
        let g = geom();
        let refs = [
            H3Reference::Image {
                latent_frames: 1,
                latent_height: 16,
                latent_width: 16,
            },
            H3Reference::Video {
                latent_frames: 2,
                latent_height: 16,
                latent_width: 16,
                audio_rows: 8,
            },
            H3Reference::Audio { audio_rows: 6 },
        ];
        let l =
            build_ref2va_packed_sequence(&[Modality::Text.tag(); 5], &refs, &g, [1, 2, 2]).unwrap();
        l.validate().unwrap();
        assert_eq!(l.num_condition_audio_rows, 8 + 6);
        // image 1*8*8 = 64, video 2*8*8 = 128
        assert_eq!(l.num_condition_video_rows, 64 + 128);
        // Reference rows come first in each index list.
        assert_eq!(l.video_indices[0], 5);
        // The video reference's audio rows precede its video rows.
        let video_ref_audio_start = 5 + 64;
        assert!(l.audio_indices.contains(&video_ref_audio_start));
        assert!(l.video_indices.contains(&(video_ref_audio_start + 8)));
    }

    #[test]
    fn ref2va_clock_advances_monotonically() {
        let g = geom();
        let refs = [
            H3Reference::Image {
                latent_frames: 1,
                latent_height: 16,
                latent_width: 16,
            },
            H3Reference::Audio { audio_rows: 4 },
        ];
        let l =
            build_ref2va_packed_sequence(&[Modality::Text.tag(); 3], &refs, &g, [1, 2, 2]).unwrap();
        // image sits at t = 3, audio starts at t = 4, target starts at 4 + 2.
        assert_eq!(l.position_ids[3][0], 3.0);
        assert_eq!(l.position_ids[3 + 64][0], 4.0);
        let target_audio = l.audio_indices[l.num_condition_audio_rows];
        assert_eq!(l.position_ids[target_audio][0], 6.0);
    }

    #[test]
    fn ref2va_rejects_odd_audio_row_counts() {
        let g = geom();
        let refs = [H3Reference::Audio { audio_rows: 5 }];
        assert!(build_ref2va_packed_sequence(&[1u32; 2], &refs, &g, [1, 2, 2]).is_err());
    }

    #[test]
    fn row_timesteps_reduce_to_distinct_levels() {
        let g = geom();
        let anchors = [KeyframeAnchor::First];
        let l = build_packed_sequence(&[Modality::Text.tag(); 4], &g, [1, 2, 2], &anchors).unwrap();
        let rt = build_row_timesteps(&l, 0.2, 0.3, 0.999, 1.0).unwrap();
        assert_eq!(rt.indices.len(), l.sequence_length());
        // video 0.2, audio 0.3, condition video 0.999 — three distinct levels
        // (no audio references here, so 1.0 never appears).
        assert_eq!(rt.timesteps, vec![0.2, 0.3, 0.999]);
        // Text rows inherit the video timestep.
        assert_eq!(rt.indices[0], 0);
        // Conditioning video rows are pinned.
        let cond = l.video_indices[0];
        assert_eq!(rt.timesteps[rt.indices[cond] as usize], 0.999);
        // Generated audio rows are on the audio schedule.
        let aud = l.audio_indices[0];
        assert_eq!(rt.timesteps[rt.indices[aud] as usize], 0.3);
        // Generated video rows are on the video schedule.
        let vid = *l.video_indices.last().unwrap();
        assert_eq!(rt.timesteps[rt.indices[vid] as usize], 0.2);
    }

    #[test]
    fn row_timesteps_collapse_when_levels_coincide() {
        let g = geom();
        let l = build_packed_sequence(&[Modality::Text.tag(); 2], &g, [1, 2, 2], &[]).unwrap();
        let rt = build_row_timesteps(&l, 0.5, 0.5, 0.5, 0.5).unwrap();
        assert_eq!(rt.timesteps, vec![0.5]);
        assert!(rt.indices.iter().all(|&i| i == 0));
    }

    #[test]
    fn adaln_index_stays_inside_the_table() {
        // The DiT addresses its modulation table with
        // `timestep_index * MODALITY_NUM + tag`.
        let g = geom();
        let l = build_packed_sequence(&[Modality::Text.tag(); 6], &g, [1, 2, 2], &[]).unwrap();
        let rt = build_row_timesteps(&l, 0.2, 0.3, 0.999, 1.0).unwrap();
        let table_rows = rt.timesteps.len() * MODALITY_NUM;
        for (row, (&ti, &tag)) in rt.indices.iter().zip(&l.token_tags).enumerate() {
            let idx = ti as usize * MODALITY_NUM + tag as usize;
            assert!(idx < table_rows, "row {row} indexes {idx} of {table_rows}");
        }
    }
}
