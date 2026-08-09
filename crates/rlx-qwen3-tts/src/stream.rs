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

//! Streaming voice-clone API.
//!
//! Three streaming modes, each with different latency / RTF / overhead
//! trade-offs. Use [`VoiceClone::generate_stream`](crate::VoiceClone::generate_stream)
//! to build the stream.
//!
//! ## Modes at a glance
//!
//! `Batched` waits for full AR + decode, then emits chunks. `Progressive` drives
//! stepwise codec AR and partial-decodes every `frames_per_chunk` frames on a
//! worker thread (lower TTFA). `PerFrame` runs full synthesis but fires
//! [`StreamEvent::FrameProduced`] during AR for progress UIs.
//!
//! ## Measured (Apple M3 Pro, ~10 s utterance, 2-run average)
//!
//! | Mode | TTFA | RTF (wall ÷ audio) | Use case |
//! |---|---:|---:|---|
//! | [`StreamMode::Batched`] | ~10 s | **~1.0×** | non-interactive batch jobs |
//! | [`StreamMode::PerFrame`] | ~10 s | ~1.0× | progress UI, ETA |
//! | [`StreamMode::Progressive`] { 64 } | ~6 s | ~1.2× | best total throughput |
//! | [`StreamMode::Progressive`] { 32 } | ~3 s | **~1.3×** | recommended live default |
//! | [`StreamMode::Progressive`] { 16 } | ~2 s | ~1.4× | low-latency live |
//! | [`StreamMode::Progressive`] { 8 } | ~2 s | ~1.7× | lower latency |
//! | [`StreamMode::Progressive`] { 4 } | ~1.5 s | ~1.7× | lowest TTFA |
//! | [`StreamConfig::realtime_second`] | ~1.0–1.2 s to 1 s PCM *(warm Metal)* | ~1.0× | 1 s streaming chunks |
//!
//! All modes produce intelligible speech (Whisper validation passes for every
//! mode). The decoder is fully causal, so re-decoding a longer prefix produces
//! sample-identical PCM for the past portion — Progressive just emits the
//! tail past the previously-consumed sample position.
//!
//! # Sync example
//! ```no_run
//! use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, VoiceClone};
//! use rlx_runtime::Device;
//! # fn run() -> anyhow::Result<()> {
//! let mut tts = VoiceClone::open(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base", Device::Metal)?;
//! let reference = tts.extract_reference("speaker.wav")?;
//! let stats = tts.generate_stream(
//!     &reference,
//!     "Hello, world.",
//!     StreamConfig::progressive(16),
//!     |event| {
//!         if let StreamEvent::Pcm(chunk) = event {
//!             println!("got {} samples at offset {}", chunk.samples.len(), chunk.sample_offset);
//!         }
//!         StreamControl::Continue
//!     },
//! )?;
//! println!("done in {:.2}s, RTF = {:.2}", stats.wall_secs, stats.realtime_factor());
//! # Ok(()) }
//! ```
//!
//! # Async example
//! ```no_run
//! # #[cfg(feature = "async")]
//! # async fn run() -> anyhow::Result<()> {
//! use futures_core::Stream;
//! use rlx_qwen3_tts::{StreamConfig, VoiceClone, generate_chunks_async};
//! use rlx_runtime::Device;
//!
//! let mut tts = VoiceClone::open(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base", Device::Metal)?;
//! let reference = tts.extract_reference("speaker.wav")?;
//! let mut stream = generate_chunks_async(tts, reference, "Hello".to_string(), StreamConfig::progressive(16))?;
//! use futures_core::Stream;
//! # let _ = stream;
//! # Ok(()) }
//! ```

use std::time::Instant;

/// Configurable streaming behaviour. See module-level docs for a comparison.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// PCM samples per emitted chunk (24 kHz mono). Default: 24000 (1 second).
    ///
    /// Smaller → smaller buffers / finer control; larger → less per-chunk
    /// callback overhead. The final chunk may be smaller.
    pub chunk_samples: usize,

    /// Streaming strategy. See [`StreamMode`].
    pub mode: StreamMode,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_samples: 24_000,
            mode: StreamMode::Batched,
        }
    }
}

impl StreamConfig {
    /// Bit-exact, full-quality, simplest: emit chunks once the entire utterance
    /// has been generated. Best for batch jobs.
    pub fn batched() -> Self {
        Self::default()
    }

    /// Same precision as [`Self::batched`], but the AR loop fires a frame-level
    /// callback as each codec frame is produced. Use this for a progress bar
    /// or ETA without changing audio output.
    pub fn per_frame() -> Self {
        Self {
            mode: StreamMode::PerFrame,
            ..Self::default()
        }
    }

    /// Live streaming: partial-decode every `frames_per_chunk` codec frames as
    /// the AR loop produces them. Lower time-to-first-audio at the cost of
    /// redundant decode work. Emitted PCM is checked against one-shot decode.
    ///
    /// Sensible values: `4` (lowest TTFA), `8`, `16` (balanced), `32`.
    pub fn progressive(frames_per_chunk: usize) -> Self {
        Self {
            mode: StreamMode::Progressive { frames_per_chunk },
            ..Self::default()
        }
    }

    /// Shortcut for lowest-latency live streaming. Equivalent to
    /// `Self::progressive(4).with_chunk_samples(8_000)` — first audio in
    /// ~1.8 s on Apple M3 Pro, then ~0.33 s chunks at 24 kHz. Trades ~3× more
    /// total CPU for the latency win.
    pub fn live_low_latency() -> Self {
        Self::progressive(4).with_chunk_samples(8_000)
    }

    /// Target ~1 s wall for ~1 s of streamed audio on a **warm** Metal session.
    ///
    /// At 12 Hz codec rate, 1 s of audio ≈ 12 frames. Partial-decode once per
    /// second of AR (not every 4 frames) to avoid redundant decode work while
    /// keeping time-to-1 s near the AR floor (~12 × per-frame talker+CP).
    ///
    /// Use on `Device::Metal` with a reused [`crate::VoiceClone`] (amortize
    /// open/compile). First utterance after cold open is slower. On Apple
    /// Silicon set `VECLIB_MAXIMUM_THREADS=1` for best per-frame latency.
    pub fn realtime_second() -> Self {
        Self::progressive(12).with_chunk_samples(24_000)
    }

    /// Override emitted chunk size (in 24 kHz samples).
    pub fn with_chunk_samples(mut self, n: usize) -> Self {
        self.chunk_samples = n.max(1);
        self
    }
}

/// How the streaming pipeline is structured. See [`StreamConfig`] constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Generate all codec frames, decode the whole thing, then emit fixed-size
    /// chunks. Bit-exact precision, 1× CPU work, but no audio leaves the API
    /// until the entire utterance is generated.
    Batched,

    /// Same as `Batched` but the AR loop fires a [`StreamEvent::FrameProduced`]
    /// callback per codec frame. Useful for progress UIs.
    PerFrame,

    /// Stepwise AR with partial decode every `frames_per_chunk` codec frames.
    /// First audio arrives after the first partial decode (~frames_per_chunk × 80 ms).
    Progressive { frames_per_chunk: usize },
}

/// A chunk of streamed PCM audio.
#[derive(Debug, Clone)]
pub struct PcmChunk {
    /// 24 kHz mono f32 samples. May be smaller than `config.chunk_samples` for
    /// the final chunk.
    pub samples: Vec<f32>,
    /// 0-based index of this chunk within the utterance.
    pub chunk_index: usize,
    /// Sample offset of this chunk within the full utterance (cumulative).
    pub sample_offset: usize,
    /// True if this is the last chunk for the utterance.
    pub is_final: bool,
}

/// Events emitted during streaming generation. The callback returns
/// [`StreamControl::Continue`] to keep going or [`StreamControl::Stop`] to
/// abort early.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// AR has produced a new codec frame. Useful for progress UIs / ETA when
    /// `StreamMode::PerFrame` or `StreamMode::Progressive` is in use.
    FrameProduced {
        frame_index: usize,
        max_frames: usize,
    },
    /// A chunk of PCM samples is ready for the consumer.
    Pcm(PcmChunk),
}

/// Caller's response to a [`StreamEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamControl {
    /// Keep generating.
    Continue,
    /// Stop generation immediately. The next chunk (if any pending) will be
    /// marked `is_final` and the AR loop returns.
    Stop,
}

/// Per-chunk metrics from a completed stream run.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    /// Codec frames emitted by the AR loop.
    pub frames_emitted: usize,
    /// Chunks emitted to the callback.
    pub chunks_emitted: usize,
    /// Total PCM samples emitted (24 kHz mono).
    pub samples_emitted: usize,
    /// Total audio duration (samples ÷ 24 000).
    pub audio_secs: f64,
    /// Total wall time from start of `generate_stream`.
    pub wall_secs: f64,
    /// Wall time at which the first PCM chunk was emitted.
    pub time_to_first_audio_secs: f64,
    /// True if the caller returned `Stop` before natural completion.
    pub stopped_early: bool,
}

impl StreamStats {
    /// Real-time factor (wall ÷ audio). < 1.0 means faster than realtime.
    pub fn realtime_factor(&self) -> f64 {
        if self.audio_secs > 0.0 {
            self.wall_secs / self.audio_secs
        } else {
            0.0
        }
    }
}

/// Internal helper that drives chunk emission from a flat PCM buffer.
pub(crate) struct ChunkEmitter {
    pub chunk_samples: usize,
    pub start: Instant,
    pub chunks_emitted: usize,
    pub samples_emitted: usize,
    pub time_to_first_audio: Option<f64>,
    pub stopped: bool,
}

impl ChunkEmitter {
    /// `start` should be captured at the beginning of `generate_stream` so
    /// time-to-first-audio reflects the full pipeline latency, not just the
    /// chunk emission phase.
    pub fn new(chunk_samples: usize, start: Instant) -> Self {
        Self {
            chunk_samples: chunk_samples.max(1),
            start,
            chunks_emitted: 0,
            samples_emitted: 0,
            time_to_first_audio: None,
            stopped: false,
        }
    }

    /// Emit chunks of `chunk_samples` from `buf` (skipping the first
    /// `consumed` samples that were already emitted). Returns the new
    /// `consumed` count.
    pub fn drain(
        &mut self,
        buf: &[f32],
        consumed: usize,
        is_terminal: bool,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
    ) -> usize {
        let mut idx = consumed;
        while idx + self.chunk_samples <= buf.len() {
            let end = idx + self.chunk_samples;
            let samples: Vec<f32> = buf[idx..end].to_vec();
            if !self.emit(samples, false, on_event) {
                return idx;
            }
            idx = end;
        }
        if is_terminal && idx < buf.len() {
            let samples: Vec<f32> = buf[idx..].to_vec();
            if self.emit(samples, true, on_event) {
                idx = buf.len();
            }
        }
        idx
    }

    fn emit(
        &mut self,
        samples: Vec<f32>,
        is_final: bool,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
    ) -> bool {
        if self.time_to_first_audio.is_none() {
            self.time_to_first_audio = Some(self.start.elapsed().as_secs_f64());
        }
        let n = samples.len();
        let chunk = PcmChunk {
            samples,
            chunk_index: self.chunks_emitted,
            sample_offset: self.samples_emitted,
            is_final,
        };
        self.chunks_emitted += 1;
        self.samples_emitted += n;
        match on_event(StreamEvent::Pcm(chunk)) {
            StreamControl::Continue => true,
            StreamControl::Stop => {
                self.stopped = true;
                false
            }
        }
    }

    pub fn finalize(self, frames_emitted: usize) -> StreamStats {
        StreamStats {
            frames_emitted,
            chunks_emitted: self.chunks_emitted,
            samples_emitted: self.samples_emitted,
            audio_secs: self.samples_emitted as f64 / 24_000.0,
            wall_secs: self.start.elapsed().as_secs_f64(),
            time_to_first_audio_secs: self.time_to_first_audio.unwrap_or(0.0),
            stopped_early: self.stopped,
        }
    }
}

// ─── async (futures) feature ────────────────────────────────────────────────

/// `futures::Stream` adapter — `cfg(feature = "async")`.
#[cfg(feature = "async")]
pub use async_impl::{PcmChunkStream, generate_chunks_async};
#[cfg(feature = "async")]
mod async_impl {
    use super::*;
    use crate::VoiceClone;
    use crate::voice_clone_api::SpeakerReference;
    use anyhow::Result;
    use futures_channel::mpsc;
    use futures_core::Stream;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Async PCM chunk stream backed by an internal worker thread.
    ///
    /// `VoiceClone` is moved into the worker, which runs the synchronous
    /// generation pipeline and forwards events through an unbounded
    /// `futures_channel` mpsc. Drive this from any async runtime (tokio,
    /// smol, async-std).
    pub struct PcmChunkStream {
        pub(crate) rx: mpsc::UnboundedReceiver<Result<PcmChunk>>,
        pub(crate) handle: Option<std::thread::JoinHandle<Result<StreamStats>>>,
    }

    impl Stream for PcmChunkStream {
        type Item = Result<PcmChunk>;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut self.rx).poll_next(cx)
        }
    }

    impl PcmChunkStream {
        /// Wait for the worker thread to finish and return stats.
        /// Blocking — wrap in `tokio::task::spawn_blocking` from async contexts.
        pub fn finish_blocking(mut self) -> Result<StreamStats> {
            let h = self
                .handle
                .take()
                .ok_or_else(|| anyhow::anyhow!("already finished"))?;
            match h.join() {
                Ok(Ok(s)) => Ok(s),
                Ok(Err(e)) => Err(e),
                Err(_) => Err(anyhow::anyhow!("streaming worker panicked")),
            }
        }
    }

    /// Spawn streaming generation on a worker thread and return a
    /// `futures::Stream` of PCM chunks. Requires the `async` feature.
    ///
    /// `VoiceClone` is moved into the worker (must be `Send + 'static`). The
    /// stream stays alive until either the natural end of the utterance, the
    /// receiver is dropped (which terminates the worker), or the worker errors.
    pub fn generate_chunks_async(
        mut tts: VoiceClone,
        reference: SpeakerReference,
        target_text: String,
        config: StreamConfig,
    ) -> Result<PcmChunkStream>
    where
        VoiceClone: Send + 'static,
        SpeakerReference: Send + 'static,
    {
        let (tx, rx) = mpsc::unbounded::<Result<PcmChunk>>();
        let handle = std::thread::spawn(move || -> Result<StreamStats> {
            let tx = tx;
            let stats = tts.generate_stream(&reference, &target_text, config, |evt| {
                match evt {
                    StreamEvent::Pcm(chunk) => {
                        if tx.unbounded_send(Ok(chunk)).is_err() {
                            return StreamControl::Stop;
                        }
                    }
                    StreamEvent::FrameProduced { .. } => { /* swallow in async path */ }
                }
                StreamControl::Continue
            })?;
            Ok(stats)
        });
        Ok(PcmChunkStream {
            rx,
            handle: Some(handle),
        })
    }
}

// ─── tokio feature ──────────────────────────────────────────────────────────

/// `tokio::sync::mpsc` adapter — `cfg(feature = "tokio")`.
#[cfg(feature = "tokio")]
pub use tokio_impl::{PcmChunkReceiver, generate_chunks_tokio};
#[cfg(feature = "tokio")]
mod tokio_impl {
    use super::*;
    use crate::VoiceClone;
    use crate::voice_clone_api::SpeakerReference;
    use anyhow::Result;

    /// `tokio::sync::mpsc::Receiver<Result<PcmChunk>>`. Use `recv().await`.
    pub type PcmChunkReceiver = tokio::sync::mpsc::Receiver<Result<PcmChunk>>;

    /// Spawn streaming generation on a dedicated worker thread (NOT a tokio
    /// blocking task — the synthesis is single-threaded and using a thread
    /// keeps tokio's blocking pool free).
    ///
    /// Returns a tokio mpsc receiver and a `JoinHandle` for the stats. Use:
    ///
    /// ```no_run
    /// # #[cfg(feature = "tokio")]
    /// # async fn run() -> anyhow::Result<()> {
    /// use rlx_qwen3_tts::{StreamConfig, VoiceClone, generate_chunks_tokio};
    /// use rlx_runtime::Device;
    /// let mut tts = VoiceClone::open(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base", Device::Metal)?;
    /// let reference = tts.extract_reference("speaker.wav")?;
    /// let (mut rx, stats_handle) = generate_chunks_tokio(
    ///     tts, reference, "Hello".into(), StreamConfig::progressive(16), 8,
    /// );
    /// while let Some(chunk) = rx.recv().await {
    ///     let chunk = chunk?;
    ///     println!("chunk {} ({} samples)", chunk.chunk_index, chunk.samples.len());
    /// }
    /// let stats = stats_handle.await.unwrap()?;
    /// println!("RTF = {:.2}", stats.realtime_factor());
    /// # Ok(()) }
    /// ```
    pub fn generate_chunks_tokio(
        mut tts: VoiceClone,
        reference: SpeakerReference,
        target_text: String,
        config: StreamConfig,
        channel_capacity: usize,
    ) -> (
        PcmChunkReceiver,
        tokio::task::JoinHandle<Result<StreamStats>>,
    )
    where
        VoiceClone: Send + 'static,
        SpeakerReference: Send + 'static,
    {
        let cap = channel_capacity.max(1);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<PcmChunk>>(cap);
        let handle = tokio::task::spawn_blocking(move || -> Result<StreamStats> {
            let stats = tts.generate_stream(&reference, &target_text, config, |evt| {
                match evt {
                    StreamEvent::Pcm(chunk) => {
                        if tx.blocking_send(Ok(chunk)).is_err() {
                            return StreamControl::Stop;
                        }
                    }
                    StreamEvent::FrameProduced { .. } => {}
                }
                StreamControl::Continue
            })?;
            Ok(stats)
        });
        (rx, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn chunk_emitter_reconstructs_full_buffer() {
        let buf: Vec<f32> = (0..5_001).map(|i| (i as f32 * 0.001).sin()).collect();
        for chunk_samples in [480, 1_200, 2_400] {
            let mut collected = Vec::new();
            let mut emitter = ChunkEmitter::new(chunk_samples, Instant::now());
            emitter.drain(&buf, 0, true, &mut |evt| {
                if let StreamEvent::Pcm(chunk) = evt {
                    collected.extend(chunk.samples);
                }
                StreamControl::Continue
            });
            assert_eq!(collected, buf, "chunk_samples={chunk_samples}");
            assert_eq!(emitter.samples_emitted, buf.len());
        }
    }
}
