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

//! Video input: frame sampling (`MiniCPMV4_6VideoProcessor.sample_frames`)
//! and decoding through the `ffmpeg` / `ffprobe` command-line tools.
//!
//! Every sampled frame then goes through the image path on its own (resize,
//! optional slices) and becomes one `<image>…</image>` block of
//! `<|video_pad|>` tokens.
//!
//! Only `stack_frames = 1` (the HF default) is supported. With more, the HF
//! processor builds per-second composites of sub-frames for the prompt but
//! encodes the raw frames, and its own forward then rejects the mismatch, so
//! there is no reference to follow.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, anyhow, ensure};

use crate::runner::RgbImage;

/// What frame sampling needs to know about a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoMeta {
    pub width: usize,
    pub height: usize,
    pub total_frames: usize,
    pub fps: f64,
}

impl VideoMeta {
    /// Seconds, as HF's `VideoMetadata` derives it: frames / fps.
    pub fn duration(&self) -> f64 {
        self.total_frames as f64 / self.fps
    }
}

/// Stream size, frame count and average frame rate of the first video stream.
pub fn probe(path: &Path) -> Result<VideoMeta> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("minicpm-v: run ffprobe (video input needs ffmpeg installed)")?;
    ensure!(
        out.status.success(),
        "minicpm-v: ffprobe failed on {path:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = &v["streams"][0];
    let num = |k: &str| -> Result<usize> {
        s[k].as_u64()
            .map(|n| n as usize)
            .or_else(|| s[k].as_str().and_then(|t| t.parse().ok()))
            .ok_or_else(|| anyhow!("minicpm-v: ffprobe gave no {k} for {path:?}"))
    };
    let rate = s["avg_frame_rate"]
        .as_str()
        .ok_or_else(|| anyhow!("minicpm-v: ffprobe gave no frame rate for {path:?}"))?;
    let fps = match rate.split_once('/') {
        Some((n, d)) => n.parse::<f64>()? / d.parse::<f64>()?,
        None => rate.parse()?,
    };
    ensure!(
        fps.is_finite() && fps > 0.0,
        "minicpm-v: frame rate {rate} of {path:?}"
    );
    Ok(VideoMeta {
        width: num("width")?,
        height: num("height")?,
        total_frames: num("nb_frames")?,
        fps,
    })
}

/// Indices of the frames to encode: one per second, or — for a clip longer
/// than `max_num_frames` seconds — one per 0.1 s thinned to `max_num_frames`
/// with numpy's integer `linspace`.
pub fn sample_frames(meta: &VideoMeta, max_num_frames: usize) -> Vec<usize> {
    let duration = meta.duration();
    let last = meta.total_frames.saturating_sub(1);
    if duration > max_num_frames as f64 {
        // `[round(i * 0.1, 1) for i in range(int(duration / 0.1))]`, then
        // `int(ts * fps)`: the timestamp is the f64 nearest a multiple of 0.1.
        let n = (duration / 0.1) as usize;
        let main: Vec<usize> = (0..n)
            .map(|i| {
                let ts = round_to_tenth(i as f64 * 0.1);
                ((ts * meta.fps) as usize).min(last)
            })
            .collect();
        if main.len() > max_num_frames {
            return linspace_int(main.len() - 1, max_num_frames)
                .into_iter()
                .map(|i| main[i])
                .collect();
        }
        main
    } else {
        (0..duration.ceil() as usize)
            .map(|i| (i as f64 * meta.fps) as usize)
            .collect()
    }
}

/// Python's `round(x, 1)`: the f64 nearest `x` to one decimal (ties to
/// even on the exactly representable halves it actually meets).
fn round_to_tenth(x: f64) -> f64 {
    (x * 10.0).round_ties_even() / 10.0
}

/// `np.linspace(0, last, n, dtype=int)`: evenly spaced, truncated toward 0.
fn linspace_int(last: usize, n: usize) -> Vec<usize> {
    if n == 1 {
        return vec![0];
    }
    let step = last as f64 / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i == n - 1 {
                last
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect()
}

/// Decode `indices` (any order, repeats allowed) to RGB frames with
/// `ffmpeg`, in the order given.
pub fn decode_frames(path: &Path, meta: &VideoMeta, indices: &[usize]) -> Result<Vec<RgbImage>> {
    let mut unique: Vec<usize> = indices.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let select = unique
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select='{select}'"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .context("minicpm-v: run ffmpeg (video input needs ffmpeg installed)")?;
    ensure!(
        out.status.success(),
        "minicpm-v: ffmpeg failed on {path:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let frame_bytes = meta.width * meta.height * 3;
    ensure!(
        out.stdout.len() == frame_bytes * unique.len(),
        "minicpm-v: ffmpeg returned {} bytes for {} frames of {}x{}",
        out.stdout.len(),
        unique.len(),
        meta.width,
        meta.height
    );
    indices
        .iter()
        .map(|i| {
            let k = unique.binary_search(i).expect("from the same list");
            Ok(RgbImage {
                width: meta.width,
                height: meta.height,
                rgb: out.stdout[k * frame_bytes..(k + 1) * frame_bytes].to_vec(),
            })
        })
        .collect()
}

/// First line of `ffmpeg -version` (decoders differ by build in their
/// YUV → RGB conversion).
pub fn ffmpeg_version() -> Result<String> {
    let out = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("minicpm-v: run ffmpeg")?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string())
}

/// Settings of 4.5's packed video (`encode_video` in the model card).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackOpts {
    /// Frames per second to sample (`choose_fps`).
    pub fps: usize,
    /// Frames per group the resampler packs into one 64-row unit when the
    /// clip is long (`MAX_NUM_FRAMES`-bounded), or forced.
    pub force_packing: Option<usize>,
    /// `MAX_NUM_FRAMES` (groups after packing).
    pub max_frames: usize,
    /// `MAX_NUM_PACKING`.
    pub max_packing: usize,
}

impl Default for PackOpts {
    fn default() -> Self {
        Self {
            fps: 3,
            force_packing: None,
            max_frames: 180,
            max_packing: 3,
        }
    }
}

/// `encode_video` of the MiniCPM-V-4_5 model card: the frame indices, and
/// their temporal ids (time in 0.1 s steps) grouped `packing` at a time.
pub fn sample_packed(meta: &VideoMeta, o: &PackOpts) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n = meta.total_frames;
    let fps = meta.fps;
    let duration = n as f64 / fps;
    let (mut packing, choose) = if (o.fps as f64) * duration.trunc() <= o.max_frames as f64 {
        let per_sec = (o.fps as f64).min(fps.round_ties_even());
        (
            1,
            (per_sec * (o.max_frames as f64).min(duration)).round_ties_even(),
        )
    } else {
        let p = (duration * o.fps as f64 / o.max_frames as f64).ceil() as usize;
        if p <= o.max_packing {
            (p, (duration * o.fps as f64).round_ties_even())
        } else {
            (
                o.max_packing,
                ((o.max_frames * o.max_packing) as f64).round_ties_even(),
            )
        }
    };
    let choose = choose as usize;
    // `uniform_sample`: the middle of `choose` equal spans.
    let gap = n as f64 / choose as f64;
    let idx: Vec<usize> = (0..choose)
        .map(|i| (i as f64 * gap + gap / 2.0) as usize)
        .collect();
    if let Some(f) = o.force_packing {
        packing = f.min(o.max_packing);
    }
    // Nearest point of `np.arange(0, duration, 0.1)`, then `/ 0.1` truncated.
    let steps = (duration / 0.1).ceil() as usize;
    let ids: Vec<usize> = idx
        .iter()
        .map(|&i| {
            let ts = i as f64 / fps;
            let mut best = 0usize;
            let mut err = f64::INFINITY;
            for k in 0..steps {
                let e = (k as f64 * 0.1 - ts).abs();
                if e < err {
                    err = e;
                    best = k;
                }
            }
            ((best as f64 * 0.1) / 0.1) as usize
        })
        .collect();
    let groups = ids.chunks(packing.max(1)).map(<[usize]>::to_vec).collect();
    (idx, groups)
}

/// Probe, sample and decode a clip.
pub fn load_video(path: &Path, max_num_frames: usize) -> Result<Vec<RgbImage>> {
    ensure!(max_num_frames > 0, "minicpm-v: max_num_frames must be > 0");
    let meta = probe(path)?;
    let indices = sample_frames(&meta, max_num_frames);
    ensure!(!indices.is_empty(), "minicpm-v: {path:?} has no frames");
    decode_frames(path, &meta, &indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(total_frames: usize, fps: f64) -> VideoMeta {
        VideoMeta {
            width: 2,
            height: 2,
            total_frames,
            fps,
        }
    }

    #[test]
    fn long_clip_matches_hf_sampling() {
        // football.mp4 (2239 frames at 60 fps), max_num_frames 16: the
        // indices the HF processor picked.
        assert_eq!(
            sample_frames(&meta(2239, 60.0), 16),
            [
                0, 144, 294, 444, 594, 744, 888, 1038, 1188, 1338, 1488, 1632, 1782, 1932, 2082,
                2232
            ]
        );
        assert_eq!(
            sample_frames(&meta(2239, 60.0), 8),
            [0, 318, 636, 954, 1272, 1590, 1908, 2232]
        );
    }

    #[test]
    fn short_clip_takes_one_frame_per_second() {
        assert_eq!(sample_frames(&meta(75, 30.0), 128), [0, 30, 60]);
    }

    #[test]
    fn numpy_integer_linspace() {
        assert_eq!(linspace_int(9, 4), [0, 3, 6, 9]);
        assert_eq!(linspace_int(372, 16)[1], 24);
    }
}
