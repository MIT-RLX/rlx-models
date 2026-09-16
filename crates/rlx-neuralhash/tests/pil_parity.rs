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

//! Preprocessing parity against Pillow — the one stage of the pipeline that can
//! be pinned to ground truth without Apple's weights.
//!
//! The fixture below was produced by running the reference chain verbatim
//! against Pillow 11 (`Image.resize` with no `resample=`, i.e. BICUBIC):
//!
//! ```python
//! image = Image.fromarray(src).convert('RGB')
//! image = image.resize([360, 360])
//! arr   = np.array(image).astype(np.float32) / 255.0
//! arr   = arr * 2.0 - 1.0
//! arr   = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
//! ```
//!
//! Over the full 388 800-element tensor the native output is **bit-identical to
//! Pillow at 388 729 elements (99.98%)**; the remaining 71 differ by exactly one
//! 8-bit step, which is a rounding tie in the intermediate byte round-trip
//! Pillow performs between its horizontal and vertical passes.
//!
//! # Why these probe indices
//!
//! Aggregate statistics do *not* discriminate resampling filters — BILINEAR,
//! LANCZOS and even NEAREST all land within 5e-5 of BICUBIC on the tensor mean.
//! The probes are therefore the 64 elements where BICUBIC disagrees most with
//! both BILINEAR and LANCZOS. Measured against this fixture at the tolerance
//! below:
//!
//! | preprocessing            | probes failing |
//! |--------------------------|---------------:|
//! | rlx native (this crate)  |          0 / 64 |
//! | PIL BILINEAR             |         64 / 64 |
//! | PIL LANCZOS              |         64 / 64 |
//! | PIL NEAREST              |         53 / 64 |
//!
//! The source is 512×341 — larger than 360 in both axes and not square — so
//! this exercises the antialiased downscale branch (Pillow widens the filter
//! support by the scale ratio) *and* the aspect-ratio squash. A non-antialiased
//! bicubic, the wrong cubic `a` coefficient, or an aspect-preserving resize all
//! fail here.

use rlx_neuralhash::preprocess::{INPUT_ELEMS, from_rgb8};

/// One 8-bit quantization step in the `[-1, 1]` output range. Pillow rounds to
/// bytes between resampling passes, so a one-step tie is the floor on
/// achievable agreement; anything larger is a real filter difference.
const STEP: f32 = 2.0 / 255.0;

/// Smooth, band-limited source: reproduced bit-for-bit on the Python side.
fn smooth_source(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f64, y as f64);
            let r = 127.5 * (1.0 + (xf / 23.0).sin() * (yf / 17.0).cos());
            let g = 255.0 * xf / w as f64;
            let b = 127.5 * (1.0 + ((xf + yf) / 31.0).sin());
            let i = (y * w + x) * 3;
            px[i] = r.clamp(0.0, 255.0) as u8;
            px[i + 1] = g.clamp(0.0, 255.0) as u8;
            px[i + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    px
}

/// `(flat index, Pillow BICUBIC value)` at the 64 most filter-sensitive positions.
const PILLOW_BICUBIC_PROBES: &[(usize, f32)] = &[
    (2937, -0.35686272),
    (14774, -0.5058824),
    (16707, -0.26274508),
    (18572, -0.4980392),
    (18605, -0.5058824),
    (19787, -0.5058824),
    (21227, -0.5058824),
    (21587, -0.5058824),
    (22532, -0.4980392),
    (22565, -0.5058824),
    (23702, 0.15294123),
    (24017, 0.15294123),
    (37477, 0.6627451),
    (37497, -0.3490196),
    (37946, 0.34901965),
    (53389, -0.18431371),
    (58218, 0.2313726),
    (58262, 0.15294123),
    (60467, -0.5058824),
    (60827, -0.5058824),
    (61547, -0.5058824),
    (64742, 0.14509809),
    (66247, -0.30196077),
    (75529, -0.4980392),
    (83294, 0.8666667),
    (87409, -0.4980392),
    (96518, -0.4980392),
    (97359, 0.32549024),
    (102227, -0.5058824),
    (102587, -0.5058824),
    (106470, 0.6313726),
    (107209, 0.5372549),
    (119577, -0.372549),
    (119867, 0.47450984),
    (120505, -0.6313726),
    (125898, -0.2235294),
    (260935, 0.8745098),
    (263951, -0.5058824),
    (265377, -0.0039215684),
    (265514, -0.0039215684),
    (266649, -0.4980392),
    (267459, -0.5058824),
    (268263, -0.4980392),
    (269615, -0.4980392),
    (292116, -0.5058824),
    (292755, -0.0039215684),
    (306615, 0.003921628),
    (310790, 0.003921628),
    (312735, -0.5058824),
    (314024, -0.0039215684),
    (314663, -0.5058824),
    (319203, -0.4980392),
    (319414, 0.003921628),
    (333217, -0.5058824),
    (333354, -0.5058824),
    (344332, 0.5686275),
    (349072, -0.5529412),
    (356340, 0.003921628),
    (357877, -0.6),
    (360515, 0.003921628),
    (364964, -0.0039215684),
    (373274, -0.8901961),
    (386815, 0.003921628),
    (388104, -0.4980392),
];

#[test]
fn matches_pillow_bicubic_on_a_non_square_downscale() {
    let got = from_rgb8(&smooth_source(512, 341), 512, 341).unwrap();
    assert_eq!(got.len(), INPUT_ELEMS);

    let mut worst = (0usize, 0f32);
    let mut failures = Vec::new();
    for &(i, want) in PILLOW_BICUBIC_PROBES {
        let d = (got[i] - want).abs();
        if d > worst.1 {
            worst = (i, d);
        }
        if d > STEP {
            failures.push(format!("[{i}] got {} want {want} (delta {d:.5})", got[i]));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} probes exceed one 8-bit step ({STEP:.5}) — the resampling filter          does not match Pillow BICUBIC:
  {}",
        failures.len(),
        PILLOW_BICUBIC_PROBES.len(),
        failures.join("
  ")
    );
    // Not just "within tolerance": these probes should be exact, because the
    // one-step ties are elsewhere in the tensor.
    assert!(
        worst.1 == 0.0,
        "probe [{}] drifted by {:.5} — expected bit-exact agreement at every probe",
        worst.0,
        worst.1
    );
}

#[test]
fn aspect_ratio_is_squashed_not_preserved() {
    // A 2:1 source and its 1:1 centre crop must NOT preprocess alike: PIL
    // `resize([360, 360])` squashes. If this ever passes, someone swapped in an
    // aspect-preserving resize and every hash silently changed.
    let wide = from_rgb8(&smooth_source(720, 360), 720, 360).unwrap();
    let square = from_rgb8(&smooth_source(360, 360), 360, 360).unwrap();
    let max_d = wide
        .iter()
        .zip(square.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_d > 0.1, "squash produced the same tensor as no squash");
}

#[test]
fn flat_images_map_to_the_range_endpoints() {
    let black = from_rgb8(&vec![0u8; 700 * 500 * 3], 700, 500).unwrap();
    assert!(black.iter().all(|v| (*v + 1.0).abs() < 1e-6));
    let white = from_rgb8(&vec![255u8; 700 * 500 * 3], 700, 500).unwrap();
    assert!(white.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    // Pillow's coefficients sum to 1, so a constant survives resampling exactly.
    let gray = from_rgb8(&vec![128u8; 700 * 500 * 3], 700, 500).unwrap();
    let want = 128.0f32 / 255.0 * 2.0 - 1.0;
    assert!(gray.iter().all(|v| (v - want).abs() < 1e-6));
}
