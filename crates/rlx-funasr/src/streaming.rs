// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Online streaming recognition.
//!
//! [`StreamingRecognizer`] accepts audio chunk by chunk. It buffers the
//! unfinalized tail, runs the FSMN-VAD over the buffer to detect speech
//! segments, and as each segment ends sufficiently before the live edge it is
//! transcribed (and optionally punctuated) by the offline ASR backbone and
//! emitted — so partial results arrive while audio is still streaming in. Once
//! a segment is committed its audio is dropped from the buffer, bounding memory
//! and the per-chunk VAD cost.
//!
//! This is VAD-gated offline streaming (each finalized segment is decoded by the
//! full offline model, which the graph cache makes cheap for repeated lengths).
//! It is not the chunked-cache BiCifParaformer; that is a distinct model.

use anyhow::Result;

use crate::pipeline::{AsrModel, Segment};
use crate::punc::CtTransformer;
use crate::vad::FsmnVad;

/// An online recognizer driven by pushed audio chunks.
pub struct StreamingRecognizer {
    vad: FsmnVad,
    asr: AsrModel,
    punc: Option<CtTransformer>,
    sample_rate: u32,
    /// Unfinalized audio still under consideration.
    buf: Vec<f32>,
    /// Absolute timestamp (ms) of `buf[0]`.
    buf_start_ms: f32,
    /// Trailing audio (ms) kept un-finalized so a segment isn't cut at the edge.
    tail_keep_ms: f32,
}

impl StreamingRecognizer {
    /// Create a recognizer from a VAD and an ASR backbone.
    pub fn new(vad: FsmnVad, asr: AsrModel) -> Self {
        Self {
            vad,
            asr,
            punc: None,
            sample_rate: 16_000,
            buf: Vec::new(),
            buf_start_ms: 0.0,
            tail_keep_ms: 300.0,
        }
    }

    /// Attach a punctuation stage.
    pub fn with_punc(mut self, punc: CtTransformer) -> Self {
        self.punc = Some(punc);
        self
    }

    /// Set how much trailing audio (ms) stays un-finalized.
    pub fn with_tail_keep_ms(mut self, ms: f32) -> Self {
        self.tail_keep_ms = ms;
        self
    }

    fn ms_to_samples(&self, ms: f32) -> usize {
        (ms / 1000.0 * self.sample_rate as f32) as usize
    }

    /// Push the next audio chunk; returns any segments finalized by it.
    pub fn accept(&mut self, chunk: &[f32]) -> Result<Vec<Segment>> {
        self.buf.extend_from_slice(chunk);
        let live_edge_ms = self.buf.len() as f32 / self.sample_rate as f32 * 1000.0;
        let segs = self.vad.segments(&self.buf)?;

        let mut out = Vec::new();
        let mut commit_ms = 0.0f32;
        for (s, e) in segs {
            if e <= live_edge_ms - self.tail_keep_ms {
                out.push(self.transcribe_span(s, e)?);
                commit_ms = e;
            }
        }
        self.drop_committed(commit_ms);
        Ok(out)
    }

    /// Flush: transcribe all remaining buffered speech.
    pub fn finalize(&mut self) -> Result<Vec<Segment>> {
        if self.buf.is_empty() {
            return Ok(Vec::new());
        }
        let segs = self.vad.segments(&self.buf)?;
        let mut out = Vec::new();
        for (s, e) in segs {
            out.push(self.transcribe_span(s, e)?);
        }
        self.buf.clear();
        Ok(out)
    }

    fn transcribe_span(&self, s_ms: f32, e_ms: f32) -> Result<Segment> {
        let a = self.ms_to_samples(s_ms).min(self.buf.len());
        let b = self.ms_to_samples(e_ms).min(self.buf.len());
        let slice = &self.buf[a..b.max(a)];
        let mut text = if slice.is_empty() {
            String::new()
        } else {
            self.asr.transcribe(slice)?
        };
        if let Some(p) = &self.punc {
            if !text.is_empty() {
                text = p.restore(&text)?;
            }
        }
        Ok(Segment {
            start_ms: self.buf_start_ms + s_ms,
            end_ms: self.buf_start_ms + e_ms,
            text,
            speaker: None,
        })
    }

    fn drop_committed(&mut self, commit_ms: f32) {
        if commit_ms <= 0.0 {
            return;
        }
        let drop = self.ms_to_samples(commit_ms).min(self.buf.len());
        self.buf.drain(0..drop);
        self.buf_start_ms += commit_ms;
    }
}
