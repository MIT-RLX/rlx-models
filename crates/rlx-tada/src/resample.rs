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

//! Band-limited rational resampling, matching `torchaudio.functional.resample`
//! with its default `sinc_interp_hann` kernel — now
//! [`rlx_core::audio::resample_sinc`], shared with the other crates that encode
//! voice prompts.
//!
//! TADA needs the reference audio at two rates at once: 24 kHz for the codec
//! and 16 kHz for the aligner. The two have to describe the *same* signal at
//! the same times, because the aligner's frame indices are used to index the
//! codec's frames. Linear interpolation aliases enough at a 3:2 ratio to shift
//! CTC peaks by a frame, which moves a token's whole latent — so the upstream
//! kernel is reproduced rather than approximated.

pub use rlx_core::audio::resample_sinc as resample;
