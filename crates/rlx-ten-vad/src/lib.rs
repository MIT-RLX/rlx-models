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

//! TEN-VAD ([TEN-framework/ten-vad]) on RLX — a native Rust port, weights
//! embedded, no ONNX Runtime and no C library at run time.
//!
//! The model is a 41-feature-per-frame CRNN: 40 log-mel bands plus an LPC-based
//! pitch estimate, three frames of context, a small separable CNN, two 64-unit
//! LSTMs and a dense head. It scores a 16 ms hop of 16 kHz audio at a time.
//!
//! The DSP frontend (`frontend`, `pitch`, `biquad`, `fft`) is a direct
//! port of the upstream C — it is recursive (IIR filters, a Viterbi pitch
//! track) and stays on the host. The network ([`model`]) is an rlx HIR graph
//! and runs on any rlx backend.
//!
//! ```no_run
//! use rlx_ten_vad::{TenVad, TenVadConfig};
//!
//! # let pcm: Vec<i16> = Vec::new();
//! let mut vad = TenVad::new(TenVadConfig::default())?;
//! for frame in pcm.chunks_exact(vad.hop_size()) {
//!     let out = vad.process_i16(frame)?;
//!     println!("{:.3} {}", out.probability, out.voice);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! [TEN-framework/ten-vad]: https://github.com/TEN-framework/ten-vad

pub mod device;
pub mod model;
pub mod segments;
pub mod session;
pub mod weights;

// The DSP frontend and the Ooura transform live in the `no_std` core crate so
// the MCU and FPGA targets share exactly this code — the parity fixtures here
// are therefore testing *that* implementation, not a copy of it.
pub use rlx_ten_vad_core::{biquad, frontend, math, net, ooura, pitch, synth};

pub mod cli;

// Shapes and rates are the core crate's — re-exported so the public API here
// is unchanged.
pub use rlx_ten_vad_core::{
    CONTEXT_FRAMES, DEFAULT_RESET_FRAMES, FEATURE_LEN, FFT_SIZE, HIDDEN, HOP_SIZE, MEL_BANDS,
    SAMPLE_RATE, SPECTRUM_BINS, WINDOW_SIZE,
};

/// Default voice decision threshold.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
/// Frames scored per graph dispatch by [`TenVadBatch`] (≈2 s).
///
/// Past the knee of the throughput curve on every backend while keeping the
/// padded tail small; see the table on [`TenVadBatch`].
pub const DEFAULT_CHUNK_FRAMES: usize = 128;

pub use device::{
    available_device_labels, available_devices, device_label, ensure_backend_ready,
    parse_device_list, resolve_device,
};
pub use model::{LstmState, Shape as GraphShape, TenVadModel};
pub use segments::{SegmentParams, SpeechSegment, speech_segments};
pub use session::{TenVad, TenVadBatch, TenVadConfig, TenVadFrame};
pub use weights::TenVadWeights;

/// Convert normalized `[-1, 1]` PCM to the int16-scaled floats the model wants.
pub fn to_int16_scale(pcm: &[f32]) -> Vec<f32> {
    pcm.iter().map(|&s| s * 32768.0).collect()
}
