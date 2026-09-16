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

//! Parity vs `torchaudio.functional.resample(x, 24000, 16000)`.
//!
//! The 24 k → 16 k hop is where the codec's frame grid and the aligner's frame
//! grid are reconciled, so a resampler that is merely "close" can move a CTC
//! peak across a frame boundary and hand a token the wrong latent. Pinned
//! against the real kernel rather than against a smoothness property.

use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    x: Vec<f32>,
    y: Vec<f32>,
}

#[test]
fn matches_torchaudio_sinc_interp_hann() {
    let raw = include_str!("fixtures/resample_reference.json");
    let f: Fixture = serde_json::from_str(raw).expect("parse resample_reference.json");
    let got = rlx_tada::resample::resample(&f.x, 24_000, 16_000);
    assert_eq!(got.len(), f.y.len(), "output length");

    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in got.iter().zip(&f.y).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    // The fixture is rounded to 6 decimals, so ~1e-6 of the deviation is the
    // fixture itself; anything at 1e-4 is a real kernel disagreement.
    assert!(
        worst < 1e-4,
        "worst deviation {worst} at sample {worst_at} (got {}, want {})",
        got[worst_at],
        f.y[worst_at]
    );
}
