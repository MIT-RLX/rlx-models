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

//! Segment timestamp unit tests (no Python).

use rlx_whisper::subtitles::{to_srt, to_vtt};
use rlx_whisper::timestamp_parse::{TIMESTAMP_QUANTUM_SEC, token_to_seconds};
use rlx_whisper::transcript::{TranscriptSegment, WhisperTranscript};

#[test]
fn timestamp_quantum_is_20ms() {
    assert!((TIMESTAMP_QUANTUM_SEC - 0.02).abs() < 1e-6);
    assert!((token_to_seconds(50365, 50364) - 0.02).abs() < 1e-6);
}

#[test]
fn srt_vtt_render_nonempty() {
    let t = WhisperTranscript {
        language: Some("en".into()),
        duration: 5.0,
        segments: vec![TranscriptSegment {
            id: 0,
            start: 0.0,
            end: 2.5,
            text: "Hello world".into(),
            words: vec![],
            speaker: None,
        }],
    };
    assert!(to_srt(&t).contains("Hello world"));
    assert!(to_vtt(&t).starts_with("WEBVTT"));
}
