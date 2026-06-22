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

//! Wav2Vec2 CTC forced alignment bridge (WhisperX-style).

use crate::transcript::{TranscriptSegment, WordAlignMode, WordTiming};
use anyhow::Result;

#[cfg(feature = "word-w2v")]
pub fn align_segment_wav2vec2(
    session: &mut rlx_wav2vec2_asr::AlignSession,
    pcm: &[f32],
    seg: &mut TranscriptSegment,
    language: &str,
) -> Result<()> {
    let slice_start = (seg.start * crate::audio::SAMPLE_RATE as f32) as usize;
    let slice_end = (seg.end * crate::audio::SAMPLE_RATE as f32).min(pcm.len() as f32) as usize;
    if slice_start >= slice_end {
        return Ok(());
    }
    let words = session.align_text(&pcm[slice_start..slice_end], &seg.text, language)?;
    seg.words = words
        .into_iter()
        .map(|w| WordTiming {
            word: w.text,
            start: seg.start + w.start,
            end: seg.start + w.end,
            prob: w.score,
        })
        .collect();
    Ok(())
}

#[cfg(not(feature = "word-w2v"))]
pub fn align_segment_wav2vec2(
    _session: &mut (),
    _pcm: &[f32],
    _seg: &mut TranscriptSegment,
    _language: &str,
) -> Result<()> {
    anyhow::bail!("rebuild with --features word-w2v for Wav2Vec2 alignment")
}

pub fn apply_forced_align(
    mode: WordAlignMode,
    #[cfg(feature = "word-w2v")] session: Option<&mut rlx_wav2vec2_asr::AlignSession>,
    #[cfg(not(feature = "word-w2v"))] _session: Option<&mut ()>,
    pcm: &[f32],
    segments: &mut [TranscriptSegment],
    language: &str,
) -> Result<()> {
    if mode != WordAlignMode::Wav2Vec2 {
        return Ok(());
    }
    #[cfg(feature = "word-w2v")]
    {
        let session = session.ok_or_else(|| anyhow::anyhow!("Wav2Vec2 align session required"))?;
        for seg in segments.iter_mut() {
            align_segment_wav2vec2(session, pcm, seg, language)?;
        }
    }
    #[cfg(not(feature = "word-w2v"))]
    {
        let _ = (pcm, segments, language);
        anyhow::bail!("word-w2v feature not enabled");
    }
    Ok(())
}
