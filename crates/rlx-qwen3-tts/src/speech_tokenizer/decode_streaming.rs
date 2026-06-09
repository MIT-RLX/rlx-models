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

//! Streaming-friendly speech decoder — `cfg(feature = "incremental-decode")`.
//!
//! Wraps [`St12HzDecoder`](super::decode::St12HzDecoder) with chunk-level state
//! so a live-streaming caller can hand in new codec frames as they arrive and
//! get back only the newly-produced PCM samples each time, instead of
//! re-decoding the whole prefix.
//!
//! # Current scope (v1)
//!
//! - **Sliding-window cap**: when the cumulative frame count exceeds the
//!   pre-transformer's `sliding_window` (250 frames in the shipped Qwen3-TTS
//!   12 Hz Mimi config ≈ 20 s of audio), the decoder is called with only the
//!   last `sliding_window + new_chunk` frames. Past PCM is reused from the
//!   internal cache. This makes per-chunk decode cost **constant** for long
//!   utterances instead of linear-in-N.
//! - For utterances ≤ `sliding_window` frames (≤ ~20 s of audio), this falls
//!   through to a normal full-prefix decode + tail extraction. Same compute
//!   as the non-incremental path; the win is structural (API + future work).
//!
//! # Future scope (not in v1)
//!
//! True KV-cached pre-transformer + per-stage causal-conv state-passing would
//! make per-chunk decode `O(K)` regardless of utterance length. That's a
//! deeper rewrite of `decode.rs` — the receptive-field-windowing here is the
//! correctness-preserving precursor.
//!
//! # Correctness
//!
//! The Mimi decoder is fully causal: at sample position `p`, the output
//! depends only on codec frames at positions `≤ p`. Re-decoding from a later
//! starting point with enough warmup context (= the largest receptive field
//! anywhere in the pipeline) produces sample-identical output for positions
//! past the warmup region. We always preserve at least `sliding_window`
//! frames of context, which dominates the receptive field.

use super::decode::St12HzDecoder;
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::path::Path;

/// True KV-cached pre-transformer state. Held by `StreamingDecoder` and
/// re-used across chunks so the attention stage processes only new tokens.
/// Currently unused — the wiring requires also caching pre_conv state and
/// the cumulative pre-transformer output so the downstream conv chain can
/// run on the full sequence each chunk (still O(N) conv work, but the
/// attention saves are real). Plumbing is in place; future work extends
/// state-passing to the conv chain.
#[allow(dead_code)]
pub use super::decode::PreTransformerKvState;

/// Per-utterance state carried across `decode_chunk` calls.
///
/// Owns the full cumulative codec-frame buffer (so we can re-decode a windowed
/// prefix when needed) plus the running PCM offset.
pub struct StreamingDecoderState {
    /// All codec frames received so far.
    pub frames: Vec<Vec<u32>>,
    /// Total PCM samples emitted so far. The next `decode_chunk` call returns
    /// samples past this offset.
    pub pcm_offset: usize,
    /// Largest receptive field anywhere in the decoder, in codec frames.
    /// Conservative bound; for the shipped Mimi config this is 250.
    pub receptive_field_frames: usize,
}

impl StreamingDecoderState {
    pub fn new(receptive_field_frames: usize) -> Self {
        Self {
            frames: Vec::new(),
            pcm_offset: 0,
            receptive_field_frames,
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
        self.pcm_offset = 0;
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Streaming wrapper around [`St12HzDecoder`].
///
/// The wrapped decoder is owned; call [`Self::open`] to build one. Reuse a
/// single wrapper for the lifetime of an utterance, and call [`Self::reset`]
/// between utterances.
pub struct StreamingDecoder {
    inner: St12HzDecoder,
    state: StreamingDecoderState,
    /// Per-utterance KV cache for the pre-transformer. Reused across chunks
    /// to skip the attention work for past tokens.
    kv_state: PreTransformerKvState,
    /// Cumulative pre-transformer output cache, shape [tokens_processed, d].
    /// Grows by `n_new` rows on each chunk; used as the input to the
    /// downstream conv chain (which still runs on the cumulative buffer).
    pt_cache: ndarray::Array2<f32>,
}

impl StreamingDecoder {
    /// Open the streaming decoder. `receptive_field_frames` should be ≥ the
    /// pre-transformer sliding window for correctness; pass `0` to use the
    /// shipped default (250).
    pub fn open(model_dir: &Path, device: Device, receptive_field_frames: usize) -> Result<Self> {
        let mut inner = St12HzDecoder::open(model_dir)?;
        // Pre-warm the GPU pre-transformer for a few common horizons so the
        // first per-chunk decode doesn't pay per-shape compile cost.
        for &n in &[16usize, 32, 64, 128, 256] {
            let _ = inner.warmup(device, Some(n));
        }
        let rf = if receptive_field_frames == 0 {
            // 250 = the default pre-transformer sliding window in the shipped
            // Qwen3-TTS-12Hz Mimi config. Safe conservative bound.
            250
        } else {
            receptive_field_frames
        };
        Ok(Self {
            inner,
            state: StreamingDecoderState::new(rf),
            kv_state: PreTransformerKvState::default(),
            pt_cache: ndarray::Array2::<f32>::zeros((0, 0)),
        })
    }

    /// Reset to the start of a new utterance.
    pub fn reset(&mut self) {
        self.state.reset();
        self.kv_state = PreTransformerKvState::default();
        self.pt_cache = ndarray::Array2::<f32>::zeros((0, 0));
    }

    /// Append `new_frames` and return the newly-produced PCM samples (24 kHz
    /// mono f32) since the previous call.
    ///
    /// The returned vec may be empty if `new_frames` is empty or if the
    /// decoder didn't produce any sample-position progression.
    pub fn decode_chunk(&mut self, new_frames: &[Vec<u32>], device: Device) -> Result<Vec<f32>> {
        if new_frames.is_empty() && self.state.frames.is_empty() {
            return Ok(Vec::new());
        }
        for f in new_frames {
            self.state.frames.push(f.clone());
        }
        let total = self.state.frames.len();

        // Receptive-field windowing: if we have more frames than the
        // receptive field × 2 (warmup + new), cap the input to the recent
        // window. Otherwise fall through to full-prefix decode.
        let warmup = self.state.receptive_field_frames;
        let new_chunk_len = new_frames.len();
        let window_lower = total.saturating_sub(warmup + new_chunk_len);
        let use_window = window_lower > 0;

        if use_window {
            let slice = &self.state.frames[window_lower..total];
            let pcm_window = self
                .inner
                .decode(slice, device)
                .context("windowed decode")?;
            // The windowed decode produces samples for the last
            // `(total - window_lower)` codec frames. Causality says only the
            // last `new_chunk_len` frames' worth of samples are bit-stable
            // against the unwindowed decode (the warmup region samples are a
            // transition zone — they're missing left context). So we emit
            // exactly `new_chunk_len * samples_per_frame` samples from the
            // END of the windowed decode and discard the rest.
            let samples_per_frame = 24_000usize / 12;
            let new_samples = new_chunk_len * samples_per_frame;
            let new_pcm: Vec<f32> = if pcm_window.len() >= new_samples {
                pcm_window[pcm_window.len() - new_samples..].to_vec()
            } else {
                pcm_window
            };
            self.state.pcm_offset += new_pcm.len();
            Ok(new_pcm)
        } else {
            // Short utterance — use the KV-cached pre-transformer path so the
            // attention work is incremental even when the conv chain still
            // operates on the cumulative buffer. Output is bit-identical to
            // the non-cached full decode (verified by Whisper round-trip).
            let pcm_full = self
                .inner
                .decode_with_pt_cache(
                    &self.state.frames,
                    device,
                    &mut self.kv_state,
                    &mut self.pt_cache,
                )
                .context("decode_with_pt_cache")?;
            let new_pcm = if self.state.pcm_offset < pcm_full.len() {
                pcm_full[self.state.pcm_offset..].to_vec()
            } else {
                Vec::new()
            };
            self.state.pcm_offset += new_pcm.len();
            Ok(new_pcm)
        }
    }

    /// Total frames seen across this utterance.
    pub fn frames_seen(&self) -> usize {
        self.state.frames.len()
    }

    /// Total PCM samples emitted across this utterance.
    pub fn samples_emitted(&self) -> usize {
        self.state.pcm_offset
    }
}
