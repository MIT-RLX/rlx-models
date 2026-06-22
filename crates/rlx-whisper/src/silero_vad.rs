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

//! Silero VAD integration via [`rlx-vad`] (optional feature).

use crate::audio::{SAMPLE_RATE, SpeechSegment};

#[cfg(feature = "silero-vad")]
pub fn silero_segments(pcm: &[f32]) -> anyhow::Result<Vec<SpeechSegment>> {
    use rlx_vad::segments::{SegmentParams, speech_segments_silero};
    use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};
    let mut session = SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
    let params = SegmentParams::silero();
    let segs = speech_segments_silero(&mut session, pcm, &params)?;
    Ok(segs
        .into_iter()
        .map(|s| SpeechSegment {
            start: s.start,
            end: s.end,
        })
        .collect())
}

#[cfg(not(feature = "silero-vad"))]
pub fn silero_segments(_pcm: &[f32]) -> anyhow::Result<Vec<SpeechSegment>> {
    anyhow::bail!("rebuild with --features silero-vad for Silero VAD")
}

pub fn segment_start_sec(seg: &SpeechSegment) -> f32 {
    seg.start as f32 / SAMPLE_RATE as f32
}

pub fn segment_end_sec(seg: &SpeechSegment) -> f32 {
    seg.end as f32 / SAMPLE_RATE as f32
}
