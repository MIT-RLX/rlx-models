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

//! Top-level transcription pipeline: ASR + timestamps + alignment + diarization.
//!
//! Typical usage:
//!
//! ```ignore
//! let mut pipeline = WhisperPipeline::new(runner, WhisperPipelineOpts {
//!     word_align: WordAlignMode::Dtw,
//!     use_silero_vad: true,
//!     ..Default::default()
//! });
//! let transcript = pipeline.run(&pcm)?;
//! ```

use crate::audio::SAMPLE_RATE;
use crate::runner::WhisperRunner;
use crate::transcript::{WhisperTranscript, WordAlignMode};
use anyhow::Result;

#[cfg(feature = "diarize")]
use crate::diarize::assign_speakers;

#[derive(Debug, Clone)]
pub struct WhisperPipelineOpts {
    /// Parse `<|M.SS|>` timestamp tokens into segment start/end times.
    pub timestamps: bool,
    /// Word-level alignment after ASR ([`WordAlignMode::Dtw`] uses cross-attention + DTW).
    pub word_align: WordAlignMode,
    /// Assign speaker labels via `rlx-diarize` (requires `diarize` feature + session).
    pub diarize: bool,
    /// Chunk long audio with Silero VAD before transcribing each region.
    pub use_silero_vad: bool,
    /// Beam width for VAD-region decode (`1` = greedy).
    pub beam_size: usize,
    /// Batched Silero/VAD region encode width (`0` = runner default, usually 10).
    /// Multi-region chunks use batched encode + serial decode for correctness.
    pub max_region_batch: usize,
    /// Parallel CPU DTW across transcript segments when GPU align is unavailable.
    pub parallel_align: bool,
}

impl Default for WhisperPipelineOpts {
    fn default() -> Self {
        Self {
            timestamps: true,
            word_align: WordAlignMode::Off,
            diarize: false,
            use_silero_vad: false,
            beam_size: 1,
            max_region_batch: 0,
            parallel_align: true,
        }
    }
}

pub struct WhisperPipeline {
    pub runner: WhisperRunner,
    pub opts: WhisperPipelineOpts,
    #[cfg(feature = "diarize")]
    pub diarizer: Option<rlx_diarize::DiarizeSession>,
}

impl WhisperPipeline {
    pub fn new(runner: WhisperRunner, opts: WhisperPipelineOpts) -> Self {
        Self {
            runner,
            opts,
            #[cfg(feature = "diarize")]
            diarizer: None,
        }
    }

    #[cfg(feature = "diarize")]
    pub fn with_diarizer(mut self, session: rlx_diarize::DiarizeSession) -> Self {
        self.diarizer = Some(session);
        self
    }

    #[cfg(feature = "tokenizer")]
    pub fn run(&mut self, pcm: &[f32]) -> Result<WhisperTranscript> {
        self.runner.prepare_pcm_geometry(pcm)?;
        if self.opts.max_region_batch > 0 {
            self.runner.set_max_region_batch(self.opts.max_region_batch);
        }
        self.runner.set_parallel_align(self.opts.parallel_align);

        let duration = pcm.len() as f32 / SAMPLE_RATE as f32;
        let mut transcript = if self.opts.use_silero_vad {
            self.runner.transcribe_structured_silero(pcm)?
        } else if self.runner.vad_enabled() {
            self.runner
                .transcribe_structured_vad(pcm, self.opts.beam_size)?
        } else {
            self.runner
                .transcribe_structured(pcm, self.opts.beam_size, 0.0)?
        };

        transcript.duration = duration;

        if self.opts.word_align != WordAlignMode::Off {
            #[cfg(any(feature = "word-dtw", feature = "word-w2v"))]
            self.runner
                .apply_word_alignment(pcm, &mut transcript, self.opts.word_align)?;
            #[cfg(not(any(feature = "word-dtw", feature = "word-w2v")))]
            anyhow::bail!("rebuild with --features word-dtw or word-w2v for word alignment");
        }

        #[cfg(feature = "diarize")]
        if self.opts.diarize {
            if let Some(ref mut diar) = self.diarizer {
                assign_speakers(diar, pcm, &mut transcript)?;
            }
        }

        Ok(transcript)
    }
}
