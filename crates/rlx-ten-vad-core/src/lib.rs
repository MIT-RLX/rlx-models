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

//! TEN-VAD without an operating system.
//!
//! The DSP frontend and a scalar CRNN forward, `no_std` + `alloc`, for the
//! targets that cannot host `rlx-runtime`: MCUs, an FPGA soft-core, WASM.
//!
//! Every buffer is a fixed size decided at construction — nothing grows and
//! nothing is allocated per frame — so a bump or static allocator is enough;
//! there is no need for a general-purpose heap. `Net` itself is fully
//! array-backed and allocates nothing at all.
//!
//! [`rlx-ten-vad`] re-exports these modules rather than duplicating them, so
//! the parity fixtures that hold the frontend bit-identical to the upstream C
//! reference are testing *this* code.
//!
//! ```no_run
//! # use rlx_ten_vad_core::{Vad, HOP_SIZE};
//! # let pcm: [f32; 0] = [];
//! let mut vad = Vad::new();
//! for frame in pcm.chunks_exact(HOP_SIZE) {
//!     let p = vad.process(frame);      // frame is int16-scaled f32
//!     let _ = p > 0.5;
//! }
//! ```
//!
//! [`rlx-ten-vad`]: https://docs.rs/rlx-ten-vad

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod biquad;
pub mod fft_fixed;
pub mod fixed;
pub mod fixed_math;
pub mod fixed_tables;
pub mod frontend;
pub mod math;
pub mod net;
pub mod ooura;
pub mod pitch;
#[cfg(feature = "alloc")]
pub mod synth;
pub mod weights;
pub mod weights_layout;

/// The only sample rate TEN-VAD supports.
pub const SAMPLE_RATE: usize = 16_000;
/// Internal analysis hop — 16 ms.
pub const HOP_SIZE: usize = 256;
/// Hann analysis window length.
pub const WINDOW_SIZE: usize = 768;
pub const FFT_SIZE: usize = 1024;
/// `FFT_SIZE / 2 + 1`.
pub const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
pub const MEL_BANDS: usize = 40;
/// Mel bands plus the pitch feature.
pub const FEATURE_LEN: usize = MEL_BANDS + 1;
/// Frames of context the network sees per score.
pub const CONTEXT_FRAMES: usize = 3;
/// LSTM width.
pub const HIDDEN: usize = 64;
/// Frames between LSTM state resets (30 s), matching `resetFrameNum` upstream.
pub const DEFAULT_RESET_FRAMES: usize = 1875;

pub use frontend::Frontend;
pub use net::Net;
pub use weights::{CoreWeights, NetWeights};

/// The whole thing: frontend + network, one 256-sample frame at a time.
///
/// Allocation-free and `Send`. About 14 KB of state (the 768-sample analysis
/// queue, the pitch estimator's buffers, and the LSTM state), so it lives
/// comfortably on an MCU stack or in a static.
pub struct Vad {
    frontend: Frontend,
    net: Net<'static>,
    emph: [f32; HOP_SIZE],
    prev: f32,
    since_reset: usize,
    reset_frames: usize,
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad {
    pub fn new() -> Self {
        Self {
            frontend: Frontend::new(weights::embedded()),
            net: Net::new(weights::embedded_net()),
            emph: [0.0; HOP_SIZE],
            prev: 0.0,
            since_reset: 0,
            reset_frames: DEFAULT_RESET_FRAMES,
        }
    }

    /// Frames between LSTM state resets; `0` disables.
    pub fn set_reset_frames(&mut self, frames: usize) {
        self.reset_frames = frames;
    }

    pub fn reset(&mut self) {
        self.frontend.reset();
        self.net.reset();
        self.prev = 0.0;
        self.since_reset = 0;
    }

    /// Score one 256-sample frame in **int16 units** (`[-32768, 32767]`).
    pub fn process(&mut self, frame: &[f32]) -> f32 {
        debug_assert_eq!(frame.len(), HOP_SIZE);
        frontend::pre_emphasis(frame, &mut self.prev, &mut self.emph);
        self.frontend.push(frame, &self.emph);
        let p = self.net.forward(self.frontend.context());
        self.since_reset += 1;
        if self.reset_frames != 0 && self.since_reset >= self.reset_frames {
            self.net.reset();
            self.since_reset = 0;
        }
        p
    }

    /// Score one frame of raw `i16` PCM.
    pub fn process_i16(&mut self, frame: &[i16]) -> f32 {
        let mut scaled = [0.0f32; HOP_SIZE];
        for (d, &s) in scaled.iter_mut().zip(frame) {
            *d = s as f32;
        }
        self.process(&scaled)
    }
}

#[cfg(all(test, feature = "std"))]
mod footprint {
    use super::*;

    /// Report the embedded footprint, and hold the RAM figure to something an
    /// MCU can actually spare. Flash is dominated by the 305 KB weight blob.
    #[test]
    fn state_and_weights_fit_an_mcu() {
        let vad = core::mem::size_of::<Vad>();
        let net = core::mem::size_of::<Net<'static>>();
        let front = core::mem::size_of::<Frontend>();
        eprintln!(
            "footprint: Vad={vad} B (Frontend={front} B, Net={net} B), \
             all inline — no allocator; weights=305_080 B flash"
        );
        // Every buffer is inline, so this *is* the whole runtime cost — there
        // is no heap behind it. The number is larger than when the same buffers
        // lived in `Vec`s, and the total memory is lower: no allocator, no
        // per-allocation overhead, and nothing to fragment.
        //
        // 40 kB fits an ESP32-C3's 400 kB SRAM many times over. Put it in a
        // `static` rather than a local: 34 kB is more than a default task stack.
        assert!(vad < 40 * 1024, "Vad state grew to {vad} B");
        assert!(net < 8 * 1024, "Net state grew to {net} B");
        assert!(
            front < 34 * 1024,
            "Frontend grew to {front} B — check the mel weight capacity"
        );
    }
}
