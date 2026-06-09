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

use rlx_vad::earshot::{Detector, FRAME_SAMPLES};

#[test]
fn earshot_silence_low_score() {
    let mut det = Detector::default();
    let frame = [0.0f32; FRAME_SAMPLES];
    let score = det.predict_f32(&frame);
    assert!(score >= 0.0 && score <= 1.0);
    assert!(
        score < 0.5,
        "silence frame should score below 0.5, got {score}"
    );
}

#[test]
fn earshot_reset_clears_state() {
    let mut det = Detector::default();
    let _ = det.predict_f32(&[0.1f32; FRAME_SAMPLES]);
    det.reset();
    let score = det.predict_f32(&[0.0f32; FRAME_SAMPLES]);
    assert!(score < 0.5);
}
