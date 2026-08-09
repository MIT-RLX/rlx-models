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

//! Assign speaker labels from [`rlx_diarize`] turns to transcript segments.

use crate::transcript::WhisperTranscript;
use anyhow::Result;

pub fn assign_speakers(
    diar: &mut rlx_diarize::DiarizeSession,
    pcm: &[f32],
    transcript: &mut WhisperTranscript,
) -> Result<()> {
    let turns = diar.diarize(pcm)?;
    for seg in &mut transcript.segments {
        seg.speaker = best_speaker(&turns, seg.start, seg.end).map(|id| format!("SPEAKER_{id:02}"));
    }
    Ok(())
}

fn best_speaker(turns: &[rlx_diarize::SpeakerTurn], start: f32, end: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for t in turns {
        let overlap = (end.min(t.end) - start.max(t.start)).max(0.0);
        if overlap > 0.0 && best.map(|(_, o)| overlap > o).unwrap_or(true) {
            best = Some((t.speaker_id, overlap));
        }
    }
    best.map(|(id, _)| id)
}
