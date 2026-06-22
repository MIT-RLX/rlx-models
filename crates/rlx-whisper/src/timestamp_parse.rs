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

//! Parse Whisper timestamp tokens into segment boundaries.

use crate::transcript::{TranscriptSegment, WhisperTranscript};
use anyhow::Result;

/// Whisper quantizes time to 20 ms steps (`0.02` s per token index).
pub const TIMESTAMP_QUANTUM_SEC: f32 = 0.02;

/// Tokens per second of audio (inverse of quantum).
pub const TOKENS_PER_SECOND: f32 = 1.0 / TIMESTAMP_QUANTUM_SEC;

/// Resolve the first timestamp token id from the HF tokenizer (`<|0.00|>`).
pub fn timestamp_begin(tokenizer: &tokenizers::Tokenizer) -> Option<u32> {
    tokenizer.token_to_id("<|0.00|>")
}

pub fn is_timestamp_token(id: u32, ts_begin: u32) -> bool {
    id >= ts_begin
}

pub fn token_to_seconds(id: u32, ts_begin: u32) -> f32 {
    (id - ts_begin) as f32 * TIMESTAMP_QUANTUM_SEC
}

/// Strip special prompt tokens (SOT, language, task, notimestamps) from the front.
pub fn content_tokens(tokens: &[u32], prompt_len: usize) -> &[u32] {
    if tokens.len() > prompt_len {
        &tokens[prompt_len..]
    } else {
        &[]
    }
}

/// Build segments from a generated token stream (after prompt).
pub fn segments_from_tokens(
    tokenizer: &tokenizers::Tokenizer,
    tokens: &[u32],
    time_offset_sec: f32,
    eot_id: u32,
) -> Result<Vec<TranscriptSegment>> {
    let ts_begin = timestamp_begin(tokenizer).unwrap_or(u32::MAX);
    let mut segments = Vec::new();
    let mut current_start: Option<f32> = None;
    let mut current_end: Option<f32> = None;
    let mut current_ids: Vec<u32> = Vec::new();
    let mut seg_id = 0u32;

    let mut flush = |start: f32, end: f32, ids: &mut Vec<u32>, out: &mut Vec<TranscriptSegment>| {
        if ids.is_empty() {
            return;
        }
        let text = tokenizer
            .decode(ids, true)
            .unwrap_or_default()
            .trim()
            .to_string();
        if !text.is_empty() {
            out.push(TranscriptSegment {
                id: seg_id,
                start: time_offset_sec + start,
                end: time_offset_sec + end.max(start),
                text,
                words: Vec::new(),
                speaker: None,
            });
            seg_id += 1;
        }
        ids.clear();
    };

    for &id in tokens {
        if id == eot_id {
            break;
        }
        if is_timestamp_token(id, ts_begin) {
            let t = token_to_seconds(id, ts_begin);
            if current_start.is_none() {
                current_start = Some(t);
            } else {
                current_end = Some(t);
                if let (Some(s), Some(e)) = (current_start, current_end) {
                    flush(s, e, &mut current_ids, &mut segments);
                }
                current_start = Some(t);
                current_end = None;
            }
        } else {
            current_ids.push(id);
        }
    }

    if !current_ids.is_empty() {
        let s = current_start.unwrap_or(0.0);
        let e = current_end.unwrap_or(s);
        flush(s, e, &mut current_ids, &mut segments);
    }

    Ok(segments)
}

/// Fill missing segment bounds when the model emits no timestamp tokens.
pub fn normalize_segment_times(segments: &mut [TranscriptSegment], duration_sec: f32) {
    if segments.is_empty() {
        return;
    }
    if segments.len() == 1 {
        let seg = &mut segments[0];
        if seg.end <= seg.start {
            seg.start = 0.0;
            seg.end = duration_sec;
        }
        return;
    }
    for seg in segments.iter_mut() {
        if seg.end <= seg.start {
            seg.end = seg.start + TIMESTAMP_QUANTUM_SEC;
        }
    }
}

pub fn build_transcript(
    tokenizer: &tokenizers::Tokenizer,
    tokens: &[u32],
    prompt_len: usize,
    time_offset_sec: f32,
    duration_sec: f32,
    language: Option<&str>,
    eot_id: u32,
) -> Result<WhisperTranscript> {
    let content = content_tokens(tokens, prompt_len);
    let mut segments = segments_from_tokens(tokenizer, content, time_offset_sec, eot_id)?;
    normalize_segment_times(&mut segments, duration_sec);
    Ok(WhisperTranscript {
        language: language.map(str::to_string),
        duration: duration_sec,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_to_seconds_quantum() {
        assert!((token_to_seconds(50364, 50364) - 0.0).abs() < 1e-6);
        assert!((token_to_seconds(50365, 50364) - 0.02).abs() < 1e-6);
        assert!((token_to_seconds(50399, 50364) - 0.70).abs() < 1e-6);
    }
}
