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

//! Native compile sizing and ONNX duration → waveform length mapping.

/// Vocoder samples per duration unit (matches ONNX Kitten mini 0.8).
pub const SAMPLES_PER_DURATION_UNIT: usize = 600;

/// Minimum compiled sequence length for native graphs.
pub const MIN_NATIVE_SEQUENCE_LENGTH: usize = 8;

/// Headroom above `token_len * SAMPLES_PER_DURATION_UNIT` for native waveform buffer.
const WAVEFORM_HEADROOM: usize = 12_000;

/// Recommended `(sequence_length, max_waveform_samples)` for native compile.
pub fn recommended_native_compile_opts(token_len: usize) -> (usize, usize) {
    let token_cap = token_len
        .next_power_of_two()
        .clamp(MIN_NATIVE_SEQUENCE_LENGTH, 512);
    let sequence_length = token_cap;
    let max_waveform_samples = token_len
        .saturating_mul(SAMPLES_PER_DURATION_UNIT)
        .saturating_add(WAVEFORM_HEADROOM)
        .max(sequence_length * SAMPLES_PER_DURATION_UNIT);
    (sequence_length, max_waveform_samples)
}

/// ONNX waveform sample count from the `duration` output and active token count.
pub fn waveform_samples_from_duration(duration: &[i64], token_len: usize) -> Option<usize> {
    if token_len == 0 || duration.is_empty() {
        return None;
    }
    let n = token_len.min(duration.len());
    let sum: i64 = duration[..n]
        .iter()
        .copied()
        .filter(|&d| d > 0 && d < 10_000)
        .sum();
    if sum <= 0 {
        return None;
    }
    Some(sum as usize * SAMPLES_PER_DURATION_UNIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_token_compile_opts() {
        let (seq, wave) = recommended_native_compile_opts(8);
        assert_eq!(seq, 8);
        assert!(wave >= 8 * SAMPLES_PER_DURATION_UNIT);
    }

    #[test]
    fn onnx_hello_duration_sum() {
        let dur = [19i64, 2, 1, 2, 3, 2, 3, 2];
        assert_eq!(
            waveform_samples_from_duration(&dur, 8),
            Some(34 * SAMPLES_PER_DURATION_UNIT)
        );
    }
}
