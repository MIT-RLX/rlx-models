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

//! The chained FunASR production pipeline: **VAD → ASR → punctuation →
//! speaker**. Any stage is optional; with VAD enabled the audio is split into
//! speech segments, each transcribed (and optionally punctuated / embedded)
//! independently.

use anyhow::Result;

use crate::paraformer::Paraformer;
use crate::punc::CtTransformer;
use crate::sensevoice::SenseVoice;
use crate::speaker::CamPlus;
use crate::vad::FsmnVad;

/// The ASR backbone used by the pipeline.
pub enum AsrModel {
    /// Paraformer (CIF) backbone.
    Paraformer(Paraformer),
    /// SenseVoiceSmall (CTC) backbone.
    SenseVoice(SenseVoice),
}

impl AsrModel {
    /// Transcribe a single utterance to plain text.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        match self {
            AsrModel::Paraformer(m) => m.transcribe(pcm),
            AsrModel::SenseVoice(m) => Ok(m.transcribe(pcm, "auto", true)?.text),
        }
    }
}

/// One transcribed speech segment.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment start time (ms).
    pub start_ms: f32,
    /// Segment end time (ms).
    pub end_ms: f32,
    /// Transcribed (and optionally punctuated) text.
    pub text: String,
    /// Optional CAM++ speaker embedding.
    pub speaker: Option<Vec<f32>>,
}

/// The full pipeline result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// All segment texts joined.
    pub text: String,
    /// Per-segment results.
    pub segments: Vec<Segment>,
}

/// A configurable VAD → ASR → punctuation → speaker pipeline.
#[derive(Default)]
pub struct FunPipeline {
    /// Optional voice-activity detector (splits the audio into segments).
    pub vad: Option<FsmnVad>,
    /// Optional ASR backbone.
    pub asr: Option<AsrModel>,
    /// Optional punctuation restorer.
    pub punc: Option<CtTransformer>,
    /// Optional speaker-embedding model.
    pub speaker: Option<CamPlus>,
    /// Input sample rate (Hz).
    pub sample_rate: u32,
}

impl FunPipeline {
    /// An empty pipeline at 16 kHz.
    pub fn new() -> Self {
        Self {
            sample_rate: 16_000,
            ..Default::default()
        }
    }

    /// Attach a VAD stage.
    pub fn with_vad(mut self, vad: FsmnVad) -> Self {
        self.vad = Some(vad);
        self
    }
    /// Attach the ASR backbone.
    pub fn with_asr(mut self, asr: AsrModel) -> Self {
        self.asr = Some(asr);
        self
    }
    /// Attach a punctuation stage.
    pub fn with_punc(mut self, punc: CtTransformer) -> Self {
        self.punc = Some(punc);
        self
    }
    /// Attach a speaker-embedding stage.
    pub fn with_speaker(mut self, spk: CamPlus) -> Self {
        self.speaker = Some(spk);
        self
    }

    /// Run the pipeline over mono PCM at `self.sample_rate`.
    pub fn run(&self, pcm: &[f32]) -> Result<PipelineResult> {
        let sr = self.sample_rate as f32;
        // VAD → list of (start_ms, end_ms); fall back to one whole-clip segment.
        let spans: Vec<(f32, f32)> = match &self.vad {
            Some(vad) => {
                let s = vad.segments(pcm)?;
                if s.is_empty() {
                    vec![(0.0, pcm.len() as f32 / sr * 1000.0)]
                } else {
                    s
                }
            }
            None => vec![(0.0, pcm.len() as f32 / sr * 1000.0)],
        };

        let mut segments = Vec::new();
        for (start_ms, end_ms) in spans {
            let a = ((start_ms / 1000.0 * sr) as usize).min(pcm.len());
            let b = ((end_ms / 1000.0 * sr) as usize).min(pcm.len());
            if b <= a {
                continue;
            }
            let slice = &pcm[a..b];
            let mut text = match &self.asr {
                Some(asr) => asr.transcribe(slice)?,
                None => String::new(),
            };
            if let Some(punc) = &self.punc {
                if !text.is_empty() {
                    text = punc.restore(&text)?;
                }
            }
            let speaker = match &self.speaker {
                Some(spk) => spk.embedding(slice).ok(),
                None => None,
            };
            segments.push(Segment {
                start_ms,
                end_ms,
                text,
                speaker,
            });
        }

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        Ok(PipelineResult { text, segments })
    }
}
