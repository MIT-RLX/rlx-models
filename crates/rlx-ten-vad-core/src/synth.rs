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

//! A deterministic speech-like clip, so tests and benches need no audio files.
//!
//! Four seconds at 16 kHz alternating silence, harmonic (voiced) stretches with
//! a drifting f0, and broadband noise — enough to move the pitch tracker, the
//! mel features and the decision threshold. The parity fixture is generated
//! from exactly this signal.

use alloc::vec::Vec;

use crate::SAMPLE_RATE;
use crate::math;

/// What a stretch of the clip contains.
#[derive(Debug, Clone, Copy)]
enum Part {
    Silence,
    /// Harmonic tone sweeping `f0` from `.0` to `.1` Hz.
    Voiced(f32, f32),
    /// Broadband noise at the given peak amplitude.
    Noise(f32),
    /// Harmonics plus noise.
    Mixed(f32, f32),
}

/// `(seconds, contents)` — 4 s total.
const SCRIPT: [(f32, Part); 7] = [
    (0.40, Part::Silence),
    (0.90, Part::Voiced(110.0, 155.0)),
    (0.30, Part::Noise(900.0)),
    (0.90, Part::Voiced(205.0, 175.0)),
    (0.35, Part::Silence),
    (0.85, Part::Mixed(130.0, 600.0)),
    (0.30, Part::Silence),
];

/// Number of harmonics in the voiced stretches.
const HARMONICS: usize = 9;

/// Deterministic noise, so the clip is identical on every run.
struct Lcg(u64);

impl Lcg {
    fn next_bipolar(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// Generate the clip as 16-bit PCM at 16 kHz.
pub fn speech_like_clip() -> Vec<i16> {
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let mut out = Vec::new();
    // Phase is carried across parts so voiced stretches have no click at the seam.
    let mut phase = 0.0f32;
    for (seconds, part) in SCRIPT {
        let n = (seconds * SAMPLE_RATE as f32) as usize;
        for i in 0..n {
            let t = i as f32 / n as f32;
            let sample = match part {
                Part::Silence => 0.0,
                Part::Noise(amp) => rng.next_bipolar() * amp,
                Part::Voiced(a, b) | Part::Mixed(a, b) => {
                    let (f0, noise) = match part {
                        Part::Voiced(a, b) => (a + (b - a) * t, 0.0),
                        _ => (a, rng.next_bipolar() * b),
                    };
                    phase += core::f32::consts::TAU * f0 / SAMPLE_RATE as f32;
                    if phase > core::f32::consts::TAU {
                        phase -= core::f32::consts::TAU;
                    }
                    let mut v = 0.0f32;
                    for k in 1..=HARMONICS {
                        // 1/k roll-off is a passable glottal-source stand-in.
                        v += math::sin(phase * k as f32) / k as f32;
                    }
                    // Fade the edges so the envelope is not a hard gate.
                    let env = (t * 12.0).min(1.0).min((1.0 - t) * 12.0).max(0.0);
                    v * 5200.0 * env + noise
                }
            };
            out.push(sample.clamp(-32768.0, 32767.0) as i16);
        }
    }
    out
}

/// The clip as int16-scaled floats, which is what the model consumes.
pub fn speech_like_clip_scaled() -> Vec<f32> {
    speech_like_clip().into_iter().map(|s| s as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_is_deterministic_and_four_seconds() {
        let a = speech_like_clip();
        let b = speech_like_clip();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4 * SAMPLE_RATE);
    }

    #[test]
    fn clip_has_silence_and_signal() {
        let clip = speech_like_clip();
        assert!(
            clip[..1000].iter().all(|&s| s == 0),
            "should open with silence"
        );
        assert!(
            clip.iter().any(|&s| s.abs() > 3000),
            "should contain loud stretches"
        );
    }
}
