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

//! Vocoder submodule metadata (`/decoder/generator/*`).
//!
//! The full vocoder HIR lives in [`crate::graph`] today (quant AdaIN resblocks,
//! source-filter sine gen, noise branches). Shape overrides for long waveforms
//! mirror [`crate::bundle_patches`].

use crate::native::config::KittenTtsConfig;

/// Vocoder hop: waveform samples per alignment frame (matches ONNX mini 0.8).
pub use crate::bundle_patches::SAMPLES_PER_ALIGNMENT_FRAME;

/// Max mel / alignment frames for compile-time buffers on wide sequences.
pub fn max_alignment_frames(sequence_length: usize, max_waveform_samples: usize) -> usize {
    max_waveform_samples
        .div_ceil(SAMPLES_PER_ALIGNMENT_FRAME)
        .max(crate::alignment::alignment_frame_upper_bound(
            sequence_length,
        ))
        .max(1)
}

/// ONNX node names that need explicit waveform-length shapes at compile time.
pub fn explicit_output_shape(
    node_name: &str,
    max_wave: usize,
    cfg: &KittenTtsConfig,
) -> Option<Vec<usize>> {
    let frames = cfg.frame_cap(max_wave);
    let h = cfg.harmonics;
    let table: &[(&str, &[usize])] = &[
        ("/decoder/generator/f0_upsamp/Resize", &[1, 1, max_wave]),
        ("/decoder/generator/Transpose", &[1, max_wave, 1]),
        (
            "/decoder/generator/m_source/l_sin_gen/Greater",
            &[1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Cast",
            &[1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize",
            &[1, h, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize_1",
            &[1, h, max_wave],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose",
            &[1, h, max_wave],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose_1",
            &[1, frames, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/ScatterND_1",
            &[1, max_wave, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Sin",
            &[1, max_wave, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/RandomNormalLike",
            &[1, max_wave, h],
        ),
    ];
    table
        .iter()
        .find(|(name, _)| *name == node_name)
        .map(|(_, shape)| shape.to_vec())
}
