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

//! Structured transcription output with segment and word timings.

use serde::{Deserialize, Serialize};

/// One word with start/end times in seconds (absolute, from file start).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start: f32,
    pub end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prob: Option<f32>,
}

/// A contiguous speech segment with optional word-level breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// Full-file transcription result (JSON-serializable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhisperTranscript {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Audio duration in seconds.
    pub duration: f32,
    pub segments: Vec<TranscriptSegment>,
}

impl WhisperTranscript {
    /// Concatenate segment text with spaces (no timestamps).
    pub fn plain_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Word-level alignment mode for [`crate::pipeline::WhisperPipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordAlignMode {
    #[default]
    Off,
    /// Cross-attention + DTW (whisper.cpp / faster-whisper style).
    Dtw,
    /// Wav2Vec2 CTC forced alignment (WhisperX style).
    Wav2Vec2,
}

impl WordAlignMode {
    /// Parse CLI / config strings: `off`, `dtw`, `wav2vec2`, `w2v`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(Self::Off),
            "dtw" => Some(Self::Dtw),
            "wav2vec2" | "w2v" => Some(Self::Wav2Vec2),
            _ => None,
        }
    }
}

/// Speaker turn from diarization (absolute seconds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub speaker_id: usize,
    pub start: f32,
    pub end: f32,
}
