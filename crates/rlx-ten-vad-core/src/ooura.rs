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
//
// Derived from Takuya Ooura's FFT package (`fftsg.c`), by way of TEN-VAD's
// `src/fftw.c`:
//
//   Copyright (C) 1996-2001 Takuya OOURA (email: ooura@mmm.t.u-tokyo.ac.jp)
//   You may use, copy, modify and distribute this code for any purpose (include
//   commercial use) and without fee.
//
// The bodies below are a **verbatim transliteration** — same butterflies, same
// order, same `f32` rounding — so that this crate's spectra are bit-identical
// to the reference's. Do not "clean up" the arithmetic: reassociating a single
// expression breaks that guarantee. Regenerate with
// `scripts/transpile_ooura.py` (then `cargo fmt`) instead.

//! Ooura split-radix real FFT, fixed at 1024 points.
//!
//! TEN-VAD's C reference computes its power spectrum with this exact routine.
//! A mathematically-equivalent FFT is not enough for bit-parity: `f32`
//! butterflies round differently depending on the order they are applied in,
//! and the reference's rounding is what its features encode.
//!
//! Layouts, using the reference's own names:
//!
//! * **format1** `[Re₀, Re_nyq, Re₁, Im₁, Re₂, Im₂, …]`
//! * **format2** `[Re₀, Re₁, −Im₁, Re₂, −Im₂, …, Re_nyq]`
//!
//! [`r2c`] and [`c2r`] speak format2; [`inplace_transf`] converts between the
//! two. [`power_spectrum`] chains the whole forward path and squares, which is
//! all the frontend needs.

// Transliterated code: every local is declared `mut` because the C declares them
// all up front, and the index arithmetic is deliberately literal — `a[(ao + (0))]`
// stays as written so each line still reads against `fftw.c`.
#![allow(
    unused_mut,
    clippy::identity_op,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

/// The only transform size TEN-VAD uses.
pub const N: usize = 1024;
/// `N / 2 + 1` — bins returned by [`power_spectrum`].
pub const BINS: usize = N / 2 + 1;
/// `1 / N`, exact in binary, applied before the transform as the reference does.
const SCALE: f32 = 1.0 / N as f32;

/// Ooura's bit-reversal / work-area descriptor for n = 1024 (`ip[0] = nw`,
/// `ip[1] = nc`).
static IP: [i32; 16] = [
    256, 256, 0, 16, 0, 64, 32, 96, 0, 256, 128, 384, 64, 320, 192, 448,
];

/// Ooura's trigonometric work table for n = 1024.
///
/// Spelled exactly as `fftw.c` spells them, so a diff against the C stays
/// meaningful. Trimming the trailing zeros would not change any value.
#[allow(clippy::excessive_precision)]
static W: [f32; 512] = [
    1.000000e+00,
    7.071068e-01,
    5.000377e-01,
    5.003390e-01,
    9.996988e-01,
    2.454123e-02,
    9.972905e-01,
    -7.356456e-02,
    9.987955e-01,
    4.906767e-02,
    9.891765e-01,
    -1.467305e-01,
    9.972905e-01,
    7.356456e-02,
    9.757021e-01,
    -2.191012e-01,
    9.951847e-01,
    9.801714e-02,
    9.569403e-01,
    -2.902847e-01,
    9.924795e-01,
    1.224107e-01,
    9.329928e-01,
    -3.598950e-01,
    9.891765e-01,
    1.467305e-01,
    9.039893e-01,
    -4.275551e-01,
    9.852776e-01,
    1.709619e-01,
    8.700870e-01,
    -4.928982e-01,
    9.807853e-01,
    1.950903e-01,
    8.314696e-01,
    -5.555702e-01,
    9.757021e-01,
    2.191012e-01,
    7.883464e-01,
    -6.152316e-01,
    9.700313e-01,
    2.429802e-01,
    7.409511e-01,
    -6.715590e-01,
    9.637761e-01,
    2.667128e-01,
    6.895405e-01,
    -7.242471e-01,
    9.569403e-01,
    2.902847e-01,
    6.343933e-01,
    -7.730105e-01,
    9.495282e-01,
    3.136817e-01,
    5.758082e-01,
    -8.175848e-01,
    9.415441e-01,
    3.368899e-01,
    5.141027e-01,
    -8.577286e-01,
    9.329928e-01,
    3.598950e-01,
    4.496113e-01,
    -8.932243e-01,
    9.238795e-01,
    3.826834e-01,
    3.826834e-01,
    -9.238795e-01,
    9.142098e-01,
    4.052413e-01,
    3.136817e-01,
    -9.495282e-01,
    9.039893e-01,
    4.275551e-01,
    2.429802e-01,
    -9.700313e-01,
    8.932243e-01,
    4.496113e-01,
    1.709619e-01,
    -9.852776e-01,
    8.819213e-01,
    4.713967e-01,
    9.801714e-02,
    -9.951847e-01,
    8.700870e-01,
    4.928982e-01,
    2.454123e-02,
    -9.996988e-01,
    8.577286e-01,
    5.141027e-01,
    -4.906767e-02,
    -9.987955e-01,
    8.448536e-01,
    5.349976e-01,
    -1.224107e-01,
    -9.924795e-01,
    8.314696e-01,
    5.555702e-01,
    -1.950903e-01,
    -9.807853e-01,
    8.175848e-01,
    5.758082e-01,
    -2.667128e-01,
    -9.637761e-01,
    8.032075e-01,
    5.956993e-01,
    -3.368899e-01,
    -9.415441e-01,
    7.883464e-01,
    6.152316e-01,
    -4.052413e-01,
    -9.142098e-01,
    7.730105e-01,
    6.343933e-01,
    -4.713967e-01,
    -8.819213e-01,
    7.572088e-01,
    6.531728e-01,
    -5.349976e-01,
    -8.448536e-01,
    7.409511e-01,
    6.715590e-01,
    -5.956993e-01,
    -8.032075e-01,
    7.242471e-01,
    6.895405e-01,
    -6.531728e-01,
    -7.572088e-01,
    1.000000e+00,
    7.071068e-01,
    5.001506e-01,
    5.013585e-01,
    9.987955e-01,
    4.906767e-02,
    9.891765e-01,
    -1.467305e-01,
    9.951847e-01,
    9.801714e-02,
    9.569403e-01,
    -2.902847e-01,
    9.891765e-01,
    1.467305e-01,
    9.039893e-01,
    -4.275551e-01,
    9.807853e-01,
    1.950903e-01,
    8.314696e-01,
    -5.555702e-01,
    9.700313e-01,
    2.429802e-01,
    7.409511e-01,
    -6.715590e-01,
    9.569403e-01,
    2.902847e-01,
    6.343933e-01,
    -7.730105e-01,
    9.415441e-01,
    3.368899e-01,
    5.141027e-01,
    -8.577286e-01,
    9.238795e-01,
    3.826834e-01,
    3.826834e-01,
    -9.238795e-01,
    9.039893e-01,
    4.275551e-01,
    2.429802e-01,
    -9.700313e-01,
    8.819213e-01,
    4.713967e-01,
    9.801714e-02,
    -9.951847e-01,
    8.577286e-01,
    5.141027e-01,
    -4.906767e-02,
    -9.987955e-01,
    8.314696e-01,
    5.555702e-01,
    -1.950903e-01,
    -9.807853e-01,
    8.032075e-01,
    5.956993e-01,
    -3.368899e-01,
    -9.415441e-01,
    7.730105e-01,
    6.343933e-01,
    -4.713967e-01,
    -8.819213e-01,
    7.409511e-01,
    6.715590e-01,
    -5.956993e-01,
    -8.032075e-01,
    1.000000e+00,
    7.071068e-01,
    5.006030e-01,
    5.054710e-01,
    9.951847e-01,
    9.801714e-02,
    9.569403e-01,
    -2.902847e-01,
    9.807853e-01,
    1.950903e-01,
    8.314696e-01,
    -5.555702e-01,
    9.569403e-01,
    2.902847e-01,
    6.343933e-01,
    -7.730105e-01,
    9.238795e-01,
    3.826834e-01,
    3.826834e-01,
    -9.238795e-01,
    8.819213e-01,
    4.713967e-01,
    9.801714e-02,
    -9.951847e-01,
    8.314696e-01,
    5.555702e-01,
    -1.950903e-01,
    -9.807853e-01,
    7.730105e-01,
    6.343933e-01,
    -4.713967e-01,
    -8.819213e-01,
    1.000000e+00,
    7.071068e-01,
    5.024193e-01,
    5.224986e-01,
    9.807853e-01,
    1.950903e-01,
    8.314696e-01,
    -5.555702e-01,
    9.238795e-01,
    3.826834e-01,
    3.826834e-01,
    -9.238795e-01,
    8.314696e-01,
    5.555702e-01,
    -1.950903e-01,
    -9.807853e-01,
    1.000000e+00,
    7.071068e-01,
    5.097956e-01,
    6.013449e-01,
    9.238795e-01,
    3.826834e-01,
    3.826834e-01,
    -9.238795e-01,
    1.000000e+00,
    7.071068e-01,
    9.238795e-01,
    3.826834e-01,
    1.000000e+00,
    7.071068e-01,
    0.000000e+00,
    0.000000e+00,
    7.071068e-01,
    4.999906e-01,
    4.999624e-01,
    4.999153e-01,
    4.998494e-01,
    4.997647e-01,
    4.996612e-01,
    4.995389e-01,
    4.993977e-01,
    4.992378e-01,
    4.990591e-01,
    4.988615e-01,
    4.986452e-01,
    4.984101e-01,
    4.981563e-01,
    4.978837e-01,
    4.975924e-01,
    4.972823e-01,
    4.969535e-01,
    4.966060e-01,
    4.962398e-01,
    4.958549e-01,
    4.954513e-01,
    4.950291e-01,
    4.945883e-01,
    4.941288e-01,
    4.936507e-01,
    4.931540e-01,
    4.926388e-01,
    4.921050e-01,
    4.915527e-01,
    4.909819e-01,
    4.903926e-01,
    4.897849e-01,
    4.891587e-01,
    4.885141e-01,
    4.878511e-01,
    4.871697e-01,
    4.864700e-01,
    4.857519e-01,
    4.850156e-01,
    4.842610e-01,
    4.834882e-01,
    4.826972e-01,
    4.818880e-01,
    4.810607e-01,
    4.802153e-01,
    4.793517e-01,
    4.784702e-01,
    4.775706e-01,
    4.766530e-01,
    4.757175e-01,
    4.747641e-01,
    4.737928e-01,
    4.728037e-01,
    4.717967e-01,
    4.707720e-01,
    4.697296e-01,
    4.686695e-01,
    4.675918e-01,
    4.664964e-01,
    4.653835e-01,
    4.642530e-01,
    4.631051e-01,
    4.619398e-01,
    4.607570e-01,
    4.595569e-01,
    4.583395e-01,
    4.571049e-01,
    4.558530e-01,
    4.545840e-01,
    4.532979e-01,
    4.519946e-01,
    4.506744e-01,
    4.493372e-01,
    4.479831e-01,
    4.466122e-01,
    4.452244e-01,
    4.438198e-01,
    4.423985e-01,
    4.409606e-01,
    4.395061e-01,
    4.380350e-01,
    4.365475e-01,
    4.350435e-01,
    4.335231e-01,
    4.319864e-01,
    4.304335e-01,
    4.288643e-01,
    4.272790e-01,
    4.256776e-01,
    4.240602e-01,
    4.224268e-01,
    4.207775e-01,
    4.191124e-01,
    4.174314e-01,
    4.157348e-01,
    4.140225e-01,
    4.122947e-01,
    4.105513e-01,
    4.087924e-01,
    4.070182e-01,
    4.052286e-01,
    4.034238e-01,
    4.016038e-01,
    3.997686e-01,
    3.979185e-01,
    3.960533e-01,
    3.941732e-01,
    3.922783e-01,
    3.903686e-01,
    3.884442e-01,
    3.865052e-01,
    3.845517e-01,
    3.825836e-01,
    3.806012e-01,
    3.786044e-01,
    3.765934e-01,
    3.745682e-01,
    3.725289e-01,
    3.704756e-01,
    3.684083e-01,
    3.663271e-01,
    3.642322e-01,
    3.621235e-01,
    3.600013e-01,
    3.578654e-01,
    3.557161e-01,
    3.535534e-01,
    3.513774e-01,
    3.491881e-01,
    3.469857e-01,
    3.447703e-01,
    3.425418e-01,
    3.403005e-01,
    3.380464e-01,
    3.357795e-01,
    3.335000e-01,
    3.312079e-01,
    3.289033e-01,
    3.265864e-01,
    3.242572e-01,
    3.219158e-01,
    3.195622e-01,
    3.171966e-01,
    3.148191e-01,
    3.124297e-01,
    3.100286e-01,
    3.076158e-01,
    3.051914e-01,
    3.027555e-01,
    3.003082e-01,
    2.978497e-01,
    2.953799e-01,
    2.928989e-01,
    2.904070e-01,
    2.879041e-01,
    2.853904e-01,
    2.828659e-01,
    2.803308e-01,
    2.777851e-01,
    2.752290e-01,
    2.726625e-01,
    2.700857e-01,
    2.674988e-01,
    2.649018e-01,
    2.622948e-01,
    2.596780e-01,
    2.570514e-01,
    2.544151e-01,
    2.517692e-01,
    2.491138e-01,
    2.464491e-01,
    2.437751e-01,
    2.410919e-01,
    2.383996e-01,
    2.356984e-01,
    2.329882e-01,
    2.302694e-01,
    2.275418e-01,
    2.248057e-01,
    2.220611e-01,
    2.193081e-01,
    2.165469e-01,
    2.137775e-01,
    2.110001e-01,
    2.082148e-01,
    2.054216e-01,
    2.026207e-01,
    1.998121e-01,
    1.969960e-01,
    1.941725e-01,
    1.913417e-01,
    1.885037e-01,
    1.856586e-01,
    1.828065e-01,
    1.799475e-01,
    1.770818e-01,
    1.742093e-01,
    1.713304e-01,
    1.684449e-01,
    1.655532e-01,
    1.626551e-01,
    1.597510e-01,
    1.568409e-01,
    1.539248e-01,
    1.510030e-01,
    1.480754e-01,
    1.451423e-01,
    1.422038e-01,
    1.392598e-01,
    1.363107e-01,
    1.333564e-01,
    1.303971e-01,
    1.274328e-01,
    1.244638e-01,
    1.214901e-01,
    1.185118e-01,
    1.155291e-01,
    1.125420e-01,
    1.095506e-01,
    1.065552e-01,
    1.035557e-01,
    1.005523e-01,
    9.754516e-02,
    9.453433e-02,
    9.151994e-02,
    8.850211e-02,
    8.548094e-02,
    8.245656e-02,
    7.942907e-02,
    7.639859e-02,
    7.336524e-02,
    7.032912e-02,
    6.729035e-02,
    6.424906e-02,
    6.120534e-02,
    5.815932e-02,
    5.511110e-02,
    5.206082e-02,
    4.900857e-02,
    4.595448e-02,
    4.289866e-02,
    3.984122e-02,
    3.678228e-02,
    3.372196e-02,
    3.066037e-02,
    2.759762e-02,
    2.453384e-02,
    2.146913e-02,
    1.840361e-02,
    1.533740e-02,
    1.227061e-02,
    9.203365e-03,
    6.135769e-03,
    3.067942e-03,
];

fn cftf1st(n: i32, a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut j: i32;
    let mut j0: i32;
    let mut j1: i32;
    let mut j2: i32;
    let mut j3: i32;
    let mut k: i32;
    let mut m: i32;
    let mut mh: i32;
    let mut wn4r: f32;
    let mut csc1: f32;
    let mut csc3: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (0)) as usize] + a[(ao + (j2)) as usize];
    x0i = a[(ao + (1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (j2)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (0)) as usize] = x0r + x2r;
    a[(ao + (1)) as usize] = x0i + x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i - x2i;
    a[(ao + (j2)) as usize] = x1r - x3i;
    a[(ao + (j2 + 1)) as usize] = x1i + x3r;
    a[(ao + (j3)) as usize] = x1r + x3i;
    a[(ao + (j3 + 1)) as usize] = x1i - x3r;
    wn4r = w[(wo + (1)) as usize];
    csc1 = w[(wo + (2)) as usize];
    csc3 = w[(wo + (3)) as usize];
    wd1r = 1.0;
    wd1i = 0.0;
    wd3r = 1.0;
    wd3i = 0.0;
    k = 0;
    j = 2;
    while j < mh - 2 {
        k += 4;
        wk1r = csc1 * (wd1r + w[(wo + (k)) as usize]);
        wk1i = csc1 * (wd1i + w[(wo + (k + 1)) as usize]);
        wk3r = csc3 * (wd3r + w[(wo + (k + 2)) as usize]);
        wk3i = csc3 * (wd3i + w[(wo + (k + 3)) as usize]);
        wd1r = w[(wo + (k)) as usize];
        wd1i = w[(wo + (k + 1)) as usize];
        wd3r = w[(wo + (k + 2)) as usize];
        wd3i = w[(wo + (k + 3)) as usize];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j)) as usize] + a[(ao + (j2)) as usize];
        x0i = a[(ao + (j + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j)) as usize] - a[(ao + (j2)) as usize];
        x1i = a[(ao + (j + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        y0r = a[(ao + (j + 2)) as usize] + a[(ao + (j2 + 2)) as usize];
        y0i = a[(ao + (j + 3)) as usize] + a[(ao + (j2 + 3)) as usize];
        y1r = a[(ao + (j + 2)) as usize] - a[(ao + (j2 + 2)) as usize];
        y1i = a[(ao + (j + 3)) as usize] - a[(ao + (j2 + 3)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        y2r = a[(ao + (j1 + 2)) as usize] + a[(ao + (j3 + 2)) as usize];
        y2i = a[(ao + (j1 + 3)) as usize] + a[(ao + (j3 + 3)) as usize];
        y3r = a[(ao + (j1 + 2)) as usize] - a[(ao + (j3 + 2)) as usize];
        y3i = a[(ao + (j1 + 3)) as usize] - a[(ao + (j3 + 3)) as usize];
        a[(ao + (j)) as usize] = x0r + x2r;
        a[(ao + (j + 1)) as usize] = x0i + x2i;
        a[(ao + (j + 2)) as usize] = y0r + y2r;
        a[(ao + (j + 3)) as usize] = y0i + y2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i - x2i;
        a[(ao + (j1 + 2)) as usize] = y0r - y2r;
        a[(ao + (j1 + 3)) as usize] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1r * x0r - wk1i * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1r * x0i + wk1i * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[(ao + (j2 + 2)) as usize] = wd1r * x0r - wd1i * x0i;
        a[(ao + (j2 + 3)) as usize] = wd1r * x0i + wd1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3r * x0r + wk3i * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3r * x0i - wk3i * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[(ao + (j3 + 2)) as usize] = wd3r * x0r + wd3i * x0i;
        a[(ao + (j3 + 3)) as usize] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
        x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
        x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        y0r = a[(ao + (j0 - 2)) as usize] + a[(ao + (j2 - 2)) as usize];
        y0i = a[(ao + (j0 - 1)) as usize] + a[(ao + (j2 - 1)) as usize];
        y1r = a[(ao + (j0 - 2)) as usize] - a[(ao + (j2 - 2)) as usize];
        y1i = a[(ao + (j0 - 1)) as usize] - a[(ao + (j2 - 1)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        y2r = a[(ao + (j1 - 2)) as usize] + a[(ao + (j3 - 2)) as usize];
        y2i = a[(ao + (j1 - 1)) as usize] + a[(ao + (j3 - 1)) as usize];
        y3r = a[(ao + (j1 - 2)) as usize] - a[(ao + (j3 - 2)) as usize];
        y3i = a[(ao + (j1 - 1)) as usize] - a[(ao + (j3 - 1)) as usize];
        a[(ao + (j0)) as usize] = x0r + x2r;
        a[(ao + (j0 + 1)) as usize] = x0i + x2i;
        a[(ao + (j0 - 2)) as usize] = y0r + y2r;
        a[(ao + (j0 - 1)) as usize] = y0i + y2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i - x2i;
        a[(ao + (j1 - 2)) as usize] = y0r - y2r;
        a[(ao + (j1 - 1)) as usize] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1i * x0r - wk1r * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1i * x0i + wk1r * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[(ao + (j2 - 2)) as usize] = wd1i * x0r - wd1r * x0i;
        a[(ao + (j2 - 1)) as usize] = wd1i * x0i + wd1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3i * x0r + wk3r * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3i * x0i - wk3r * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[(ao + (j3 - 2)) as usize] = wd3i * x0r + wd3r * x0i;
        a[(ao + (j3 - 1)) as usize] = wd3i * x0i - wd3r * x0r;
        j += 4;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (j0 - 2)) as usize] + a[(ao + (j2 - 2)) as usize];
    x0i = a[(ao + (j0 - 1)) as usize] + a[(ao + (j2 - 1)) as usize];
    x1r = a[(ao + (j0 - 2)) as usize] - a[(ao + (j2 - 2)) as usize];
    x1i = a[(ao + (j0 - 1)) as usize] - a[(ao + (j2 - 1)) as usize];
    x2r = a[(ao + (j1 - 2)) as usize] + a[(ao + (j3 - 2)) as usize];
    x2i = a[(ao + (j1 - 1)) as usize] + a[(ao + (j3 - 1)) as usize];
    x3r = a[(ao + (j1 - 2)) as usize] - a[(ao + (j3 - 2)) as usize];
    x3i = a[(ao + (j1 - 1)) as usize] - a[(ao + (j3 - 1)) as usize];
    a[(ao + (j0 - 2)) as usize] = x0r + x2r;
    a[(ao + (j0 - 1)) as usize] = x0i + x2i;
    a[(ao + (j1 - 2)) as usize] = x0r - x2r;
    a[(ao + (j1 - 1)) as usize] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[(ao + (j2 - 2)) as usize] = wk1r * x0r - wk1i * x0i;
    a[(ao + (j2 - 1)) as usize] = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[(ao + (j3 - 2)) as usize] = wk3r * x0r + wk3i * x0i;
    a[(ao + (j3 - 1)) as usize] = wk3r * x0i - wk3i * x0r;
    x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
    x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
    x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (j0)) as usize] = x0r + x2r;
    a[(ao + (j0 + 1)) as usize] = x0i + x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[(ao + (j2)) as usize] = wn4r * (x0r - x0i);
    a[(ao + (j2 + 1)) as usize] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[(ao + (j3)) as usize] = -wn4r * (x0r + x0i);
    a[(ao + (j3 + 1)) as usize] = -wn4r * (x0i - x0r);
    x0r = a[(ao + (j0 + 2)) as usize] + a[(ao + (j2 + 2)) as usize];
    x0i = a[(ao + (j0 + 3)) as usize] + a[(ao + (j2 + 3)) as usize];
    x1r = a[(ao + (j0 + 2)) as usize] - a[(ao + (j2 + 2)) as usize];
    x1i = a[(ao + (j0 + 3)) as usize] - a[(ao + (j2 + 3)) as usize];
    x2r = a[(ao + (j1 + 2)) as usize] + a[(ao + (j3 + 2)) as usize];
    x2i = a[(ao + (j1 + 3)) as usize] + a[(ao + (j3 + 3)) as usize];
    x3r = a[(ao + (j1 + 2)) as usize] - a[(ao + (j3 + 2)) as usize];
    x3i = a[(ao + (j1 + 3)) as usize] - a[(ao + (j3 + 3)) as usize];
    a[(ao + (j0 + 2)) as usize] = x0r + x2r;
    a[(ao + (j0 + 3)) as usize] = x0i + x2i;
    a[(ao + (j1 + 2)) as usize] = x0r - x2r;
    a[(ao + (j1 + 3)) as usize] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[(ao + (j2 + 2)) as usize] = wk1i * x0r - wk1r * x0i;
    a[(ao + (j2 + 3)) as usize] = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[(ao + (j3 + 2)) as usize] = wk3i * x0r + wk3r * x0i;
    a[(ao + (j3 + 3)) as usize] = wk3i * x0i - wk3r * x0r;
}

fn cftb1st(n: i32, a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut j: i32;
    let mut j0: i32;
    let mut j1: i32;
    let mut j2: i32;
    let mut j3: i32;
    let mut k: i32;
    let mut m: i32;
    let mut mh: i32;
    let mut wn4r: f32;
    let mut csc1: f32;
    let mut csc3: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (0)) as usize] + a[(ao + (j2)) as usize];
    x0i = -a[(ao + (1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (j2)) as usize];
    x1i = -a[(ao + (1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (0)) as usize] = x0r + x2r;
    a[(ao + (1)) as usize] = x0i - x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i + x2i;
    a[(ao + (j2)) as usize] = x1r + x3i;
    a[(ao + (j2 + 1)) as usize] = x1i + x3r;
    a[(ao + (j3)) as usize] = x1r - x3i;
    a[(ao + (j3 + 1)) as usize] = x1i - x3r;
    wn4r = w[(wo + (1)) as usize];
    csc1 = w[(wo + (2)) as usize];
    csc3 = w[(wo + (3)) as usize];
    wd1r = 1.0;
    wd1i = 0.0;
    wd3r = 1.0;
    wd3i = 0.0;
    k = 0;
    j = 2;
    while j < mh - 2 {
        k += 4;
        wk1r = csc1 * (wd1r + w[(wo + (k)) as usize]);
        wk1i = csc1 * (wd1i + w[(wo + (k + 1)) as usize]);
        wk3r = csc3 * (wd3r + w[(wo + (k + 2)) as usize]);
        wk3i = csc3 * (wd3i + w[(wo + (k + 3)) as usize]);
        wd1r = w[(wo + (k)) as usize];
        wd1i = w[(wo + (k + 1)) as usize];
        wd3r = w[(wo + (k + 2)) as usize];
        wd3i = w[(wo + (k + 3)) as usize];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j)) as usize] + a[(ao + (j2)) as usize];
        x0i = -a[(ao + (j + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j)) as usize] - a[(ao + (j2)) as usize];
        x1i = -a[(ao + (j + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        y0r = a[(ao + (j + 2)) as usize] + a[(ao + (j2 + 2)) as usize];
        y0i = -a[(ao + (j + 3)) as usize] - a[(ao + (j2 + 3)) as usize];
        y1r = a[(ao + (j + 2)) as usize] - a[(ao + (j2 + 2)) as usize];
        y1i = -a[(ao + (j + 3)) as usize] + a[(ao + (j2 + 3)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        y2r = a[(ao + (j1 + 2)) as usize] + a[(ao + (j3 + 2)) as usize];
        y2i = a[(ao + (j1 + 3)) as usize] + a[(ao + (j3 + 3)) as usize];
        y3r = a[(ao + (j1 + 2)) as usize] - a[(ao + (j3 + 2)) as usize];
        y3i = a[(ao + (j1 + 3)) as usize] - a[(ao + (j3 + 3)) as usize];
        a[(ao + (j)) as usize] = x0r + x2r;
        a[(ao + (j + 1)) as usize] = x0i - x2i;
        a[(ao + (j + 2)) as usize] = y0r + y2r;
        a[(ao + (j + 3)) as usize] = y0i - y2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i + x2i;
        a[(ao + (j1 + 2)) as usize] = y0r - y2r;
        a[(ao + (j1 + 3)) as usize] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1r * x0r - wk1i * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1r * x0i + wk1i * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[(ao + (j2 + 2)) as usize] = wd1r * x0r - wd1i * x0i;
        a[(ao + (j2 + 3)) as usize] = wd1r * x0i + wd1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3r * x0r + wk3i * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3r * x0i - wk3i * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[(ao + (j3 + 2)) as usize] = wd3r * x0r + wd3i * x0i;
        a[(ao + (j3 + 3)) as usize] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
        x0i = -a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
        x1i = -a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        y0r = a[(ao + (j0 - 2)) as usize] + a[(ao + (j2 - 2)) as usize];
        y0i = -a[(ao + (j0 - 1)) as usize] - a[(ao + (j2 - 1)) as usize];
        y1r = a[(ao + (j0 - 2)) as usize] - a[(ao + (j2 - 2)) as usize];
        y1i = -a[(ao + (j0 - 1)) as usize] + a[(ao + (j2 - 1)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        y2r = a[(ao + (j1 - 2)) as usize] + a[(ao + (j3 - 2)) as usize];
        y2i = a[(ao + (j1 - 1)) as usize] + a[(ao + (j3 - 1)) as usize];
        y3r = a[(ao + (j1 - 2)) as usize] - a[(ao + (j3 - 2)) as usize];
        y3i = a[(ao + (j1 - 1)) as usize] - a[(ao + (j3 - 1)) as usize];
        a[(ao + (j0)) as usize] = x0r + x2r;
        a[(ao + (j0 + 1)) as usize] = x0i - x2i;
        a[(ao + (j0 - 2)) as usize] = y0r + y2r;
        a[(ao + (j0 - 1)) as usize] = y0i - y2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i + x2i;
        a[(ao + (j1 - 2)) as usize] = y0r - y2r;
        a[(ao + (j1 - 1)) as usize] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1i * x0r - wk1r * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1i * x0i + wk1r * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[(ao + (j2 - 2)) as usize] = wd1i * x0r - wd1r * x0i;
        a[(ao + (j2 - 1)) as usize] = wd1i * x0i + wd1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3i * x0r + wk3r * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3i * x0i - wk3r * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[(ao + (j3 - 2)) as usize] = wd3i * x0r + wd3r * x0i;
        a[(ao + (j3 - 1)) as usize] = wd3i * x0i - wd3r * x0r;
        j += 4;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (j0 - 2)) as usize] + a[(ao + (j2 - 2)) as usize];
    x0i = -a[(ao + (j0 - 1)) as usize] - a[(ao + (j2 - 1)) as usize];
    x1r = a[(ao + (j0 - 2)) as usize] - a[(ao + (j2 - 2)) as usize];
    x1i = -a[(ao + (j0 - 1)) as usize] + a[(ao + (j2 - 1)) as usize];
    x2r = a[(ao + (j1 - 2)) as usize] + a[(ao + (j3 - 2)) as usize];
    x2i = a[(ao + (j1 - 1)) as usize] + a[(ao + (j3 - 1)) as usize];
    x3r = a[(ao + (j1 - 2)) as usize] - a[(ao + (j3 - 2)) as usize];
    x3i = a[(ao + (j1 - 1)) as usize] - a[(ao + (j3 - 1)) as usize];
    a[(ao + (j0 - 2)) as usize] = x0r + x2r;
    a[(ao + (j0 - 1)) as usize] = x0i - x2i;
    a[(ao + (j1 - 2)) as usize] = x0r - x2r;
    a[(ao + (j1 - 1)) as usize] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[(ao + (j2 - 2)) as usize] = wk1r * x0r - wk1i * x0i;
    a[(ao + (j2 - 1)) as usize] = wk1r * x0i + wk1i * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[(ao + (j3 - 2)) as usize] = wk3r * x0r + wk3i * x0i;
    a[(ao + (j3 - 1)) as usize] = wk3r * x0i - wk3i * x0r;
    x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
    x0i = -a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
    x1i = -a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (j0)) as usize] = x0r + x2r;
    a[(ao + (j0 + 1)) as usize] = x0i - x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[(ao + (j2)) as usize] = wn4r * (x0r - x0i);
    a[(ao + (j2 + 1)) as usize] = wn4r * (x0i + x0r);
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[(ao + (j3)) as usize] = -wn4r * (x0r + x0i);
    a[(ao + (j3 + 1)) as usize] = -wn4r * (x0i - x0r);
    x0r = a[(ao + (j0 + 2)) as usize] + a[(ao + (j2 + 2)) as usize];
    x0i = -a[(ao + (j0 + 3)) as usize] - a[(ao + (j2 + 3)) as usize];
    x1r = a[(ao + (j0 + 2)) as usize] - a[(ao + (j2 + 2)) as usize];
    x1i = -a[(ao + (j0 + 3)) as usize] + a[(ao + (j2 + 3)) as usize];
    x2r = a[(ao + (j1 + 2)) as usize] + a[(ao + (j3 + 2)) as usize];
    x2i = a[(ao + (j1 + 3)) as usize] + a[(ao + (j3 + 3)) as usize];
    x3r = a[(ao + (j1 + 2)) as usize] - a[(ao + (j3 + 2)) as usize];
    x3i = a[(ao + (j1 + 3)) as usize] - a[(ao + (j3 + 3)) as usize];
    a[(ao + (j0 + 2)) as usize] = x0r + x2r;
    a[(ao + (j0 + 3)) as usize] = x0i - x2i;
    a[(ao + (j1 + 2)) as usize] = x0r - x2r;
    a[(ao + (j1 + 3)) as usize] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[(ao + (j2 + 2)) as usize] = wk1i * x0r - wk1r * x0i;
    a[(ao + (j2 + 3)) as usize] = wk1i * x0i + wk1r * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[(ao + (j3 + 2)) as usize] = wk3i * x0r + wk3r * x0i;
    a[(ao + (j3 + 3)) as usize] = wk3i * x0i - wk3r * x0r;
}

fn cftrec4(n: i32, a: &mut [f32], ao: i32, nw: i32, w: &[f32], wo: i32) {
    let mut isplt: i32;
    let mut j: i32;
    let mut k: i32;
    let mut m: i32;

    m = n;
    while m > 512 {
        m >>= 2;
        cftmdl1(m, a, ao + (n - m), w, wo + (nw - (m >> 1)));
    }
    cftleaf(m, 1, a, ao + (n - m), nw, w, wo);
    k = 0;
    j = n - m;
    while j > 0 {
        k += 1;
        isplt = cfttree(m, j, k, a, ao, nw, w, wo);
        cftleaf(m, isplt, a, ao + (j - m), nw, w, wo);
        j -= m;
    }
}

fn cfttree(n: i32, j: i32, k: i32, a: &mut [f32], ao: i32, nw: i32, w: &[f32], wo: i32) -> i32 {
    let mut i: i32;
    let mut isplt: i32;
    let mut m: i32;

    if (k & 3) != 0 {
        isplt = k & 1;
        if isplt != 0 {
            cftmdl1(n, a, ao + (j - n), w, wo + (nw - (n >> 1)));
        } else {
            cftmdl2(n, a, ao + (j - n), w, wo + (nw - n));
        }
    } else {
        m = n;
        i = k;
        while i & 3 == 0 {
            m <<= 2;
            i >>= 2;
        }
        isplt = i & 1;
        if isplt != 0 {
            while m > 128 {
                cftmdl1(m, a, ao + (j - m), w, wo + (nw - (m >> 1)));
                m >>= 2;
            }
        } else {
            while m > 128 {
                cftmdl2(m, a, ao + (j - m), w, wo + (nw - m));
                m >>= 2;
            }
        }
    }
    isplt
}

fn cftleaf(n: i32, isplt: i32, a: &mut [f32], ao: i32, nw: i32, w: &[f32], wo: i32) {
    if n == 512 {
        cftmdl1(128, a, ao, w, wo + (nw - 64));
        cftf161(a, ao, w, wo + (nw - 8));
        cftf162(a, ao + (32), w, wo + (nw - 32));
        cftf161(a, ao + (64), w, wo + (nw - 8));
        cftf161(a, ao + (96), w, wo + (nw - 8));
        cftmdl2(128, a, ao + (128), w, wo + (nw - 128));
        cftf161(a, ao + (128), w, wo + (nw - 8));
        cftf162(a, ao + (160), w, wo + (nw - 32));
        cftf161(a, ao + (192), w, wo + (nw - 8));
        cftf162(a, ao + (224), w, wo + (nw - 32));
        cftmdl1(128, a, ao + (256), w, wo + (nw - 64));
        cftf161(a, ao + (256), w, wo + (nw - 8));
        cftf162(a, ao + (288), w, wo + (nw - 32));
        cftf161(a, ao + (320), w, wo + (nw - 8));
        cftf161(a, ao + (352), w, wo + (nw - 8));
        if isplt != 0 {
            cftmdl1(128, a, ao + (384), w, wo + (nw - 64));
            cftf161(a, ao + (480), w, wo + (nw - 8));
        } else {
            cftmdl2(128, a, ao + (384), w, wo + (nw - 128));
            cftf162(a, ao + (480), w, wo + (nw - 32));
        }
        cftf161(a, ao + (384), w, wo + (nw - 8));
        cftf162(a, ao + (416), w, wo + (nw - 32));
        cftf161(a, ao + (448), w, wo + (nw - 8));
    } else {
        cftmdl1(64, a, ao, w, wo + (nw - 32));
        cftf081(a, ao, w, wo + (nw - 8));
        cftf082(a, ao + (16), w, wo + (nw - 8));
        cftf081(a, ao + (32), w, wo + (nw - 8));
        cftf081(a, ao + (48), w, wo + (nw - 8));
        cftmdl2(64, a, ao + (64), w, wo + (nw - 64));
        cftf081(a, ao + (64), w, wo + (nw - 8));
        cftf082(a, ao + (80), w, wo + (nw - 8));
        cftf081(a, ao + (96), w, wo + (nw - 8));
        cftf082(a, ao + (112), w, wo + (nw - 8));
        cftmdl1(64, a, ao + (128), w, wo + (nw - 32));
        cftf081(a, ao + (128), w, wo + (nw - 8));
        cftf082(a, ao + (144), w, wo + (nw - 8));
        cftf081(a, ao + (160), w, wo + (nw - 8));
        cftf081(a, ao + (176), w, wo + (nw - 8));
        if isplt != 0 {
            cftmdl1(64, a, ao + (192), w, wo + (nw - 32));
            cftf081(a, ao + (240), w, wo + (nw - 8));
        } else {
            cftmdl2(64, a, ao + (192), w, wo + (nw - 64));
            cftf082(a, ao + (240), w, wo + (nw - 8));
        }
        cftf081(a, ao + (192), w, wo + (nw - 8));
        cftf082(a, ao + (208), w, wo + (nw - 8));
        cftf081(a, ao + (224), w, wo + (nw - 8));
    }
}

fn cftmdl1(n: i32, a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut j: i32;
    let mut j0: i32;
    let mut j1: i32;
    let mut j2: i32;
    let mut j3: i32;
    let mut k: i32;
    let mut m: i32;
    let mut mh: i32;
    let mut wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (0)) as usize] + a[(ao + (j2)) as usize];
    x0i = a[(ao + (1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (j2)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (0)) as usize] = x0r + x2r;
    a[(ao + (1)) as usize] = x0i + x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i - x2i;
    a[(ao + (j2)) as usize] = x1r - x3i;
    a[(ao + (j2 + 1)) as usize] = x1i + x3r;
    a[(ao + (j3)) as usize] = x1r + x3i;
    a[(ao + (j3 + 1)) as usize] = x1i - x3r;
    wn4r = w[(wo + (1)) as usize];
    k = 0;
    j = 2;
    while j < mh {
        k += 4;
        wk1r = w[(wo + (k)) as usize];
        wk1i = w[(wo + (k + 1)) as usize];
        wk3r = w[(wo + (k + 2)) as usize];
        wk3i = w[(wo + (k + 3)) as usize];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j)) as usize] + a[(ao + (j2)) as usize];
        x0i = a[(ao + (j + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j)) as usize] - a[(ao + (j2)) as usize];
        x1i = a[(ao + (j + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        a[(ao + (j)) as usize] = x0r + x2r;
        a[(ao + (j + 1)) as usize] = x0i + x2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1r * x0r - wk1i * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3r * x0r + wk3i * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3r * x0i - wk3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
        x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
        x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
        x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
        a[(ao + (j0)) as usize] = x0r + x2r;
        a[(ao + (j0 + 1)) as usize] = x0i + x2i;
        a[(ao + (j1)) as usize] = x0r - x2r;
        a[(ao + (j1 + 1)) as usize] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[(ao + (j2)) as usize] = wk1i * x0r - wk1r * x0i;
        a[(ao + (j2 + 1)) as usize] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[(ao + (j3)) as usize] = wk3i * x0r + wk3r * x0i;
        a[(ao + (j3 + 1)) as usize] = wk3i * x0i - wk3r * x0r;
        j += 2;
    }
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (j0)) as usize] + a[(ao + (j2)) as usize];
    x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1r = a[(ao + (j0)) as usize] - a[(ao + (j2)) as usize];
    x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2 + 1)) as usize];
    x2r = a[(ao + (j1)) as usize] + a[(ao + (j3)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3r = a[(ao + (j1)) as usize] - a[(ao + (j3)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3 + 1)) as usize];
    a[(ao + (j0)) as usize] = x0r + x2r;
    a[(ao + (j0 + 1)) as usize] = x0i + x2i;
    a[(ao + (j1)) as usize] = x0r - x2r;
    a[(ao + (j1 + 1)) as usize] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[(ao + (j2)) as usize] = wn4r * (x0r - x0i);
    a[(ao + (j2 + 1)) as usize] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[(ao + (j3)) as usize] = -wn4r * (x0r + x0i);
    a[(ao + (j3 + 1)) as usize] = -wn4r * (x0i - x0r);
}

fn cftmdl2(n: i32, a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut j: i32;
    let mut j0: i32;
    let mut j1: i32;
    let mut j2: i32;
    let mut j3: i32;
    let mut k: i32;
    let mut kr: i32;
    let mut m: i32;
    let mut mh: i32;
    let mut wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y2r: f32;
    let mut y2i: f32;

    mh = n >> 3;
    m = 2 * mh;
    wn4r = w[(wo + (1)) as usize];
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (0)) as usize] - a[(ao + (j2 + 1)) as usize];
    x0i = a[(ao + (1)) as usize] + a[(ao + (j2)) as usize];
    x1r = a[(ao + (0)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (j2)) as usize];
    x2r = a[(ao + (j1)) as usize] - a[(ao + (j3 + 1)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3)) as usize];
    x3r = a[(ao + (j1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3)) as usize];
    y0r = wn4r * (x2r - x2i);
    y0i = wn4r * (x2i + x2r);
    a[(ao + (0)) as usize] = x0r + y0r;
    a[(ao + (1)) as usize] = x0i + y0i;
    a[(ao + (j1)) as usize] = x0r - y0r;
    a[(ao + (j1 + 1)) as usize] = x0i - y0i;
    y0r = wn4r * (x3r - x3i);
    y0i = wn4r * (x3i + x3r);
    a[(ao + (j2)) as usize] = x1r - y0i;
    a[(ao + (j2 + 1)) as usize] = x1i + y0r;
    a[(ao + (j3)) as usize] = x1r + y0i;
    a[(ao + (j3 + 1)) as usize] = x1i - y0r;
    k = 0;
    kr = 2 * m;
    j = 2;
    while j < mh {
        k += 4;
        wk1r = w[(wo + (k)) as usize];
        wk1i = w[(wo + (k + 1)) as usize];
        wk3r = w[(wo + (k + 2)) as usize];
        wk3i = w[(wo + (k + 3)) as usize];
        kr -= 4;
        wd1i = w[(wo + (kr)) as usize];
        wd1r = w[(wo + (kr + 1)) as usize];
        wd3i = w[(wo + (kr + 2)) as usize];
        wd3r = w[(wo + (kr + 3)) as usize];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j)) as usize] - a[(ao + (j2 + 1)) as usize];
        x0i = a[(ao + (j + 1)) as usize] + a[(ao + (j2)) as usize];
        x1r = a[(ao + (j)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1i = a[(ao + (j + 1)) as usize] - a[(ao + (j2)) as usize];
        x2r = a[(ao + (j1)) as usize] - a[(ao + (j3 + 1)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3)) as usize];
        x3r = a[(ao + (j1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3)) as usize];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wd1r * x2r - wd1i * x2i;
        y2i = wd1r * x2i + wd1i * x2r;
        a[(ao + (j)) as usize] = y0r + y2r;
        a[(ao + (j + 1)) as usize] = y0i + y2i;
        a[(ao + (j1)) as usize] = y0r - y2r;
        a[(ao + (j1 + 1)) as usize] = y0i - y2i;
        y0r = wk3r * x1r + wk3i * x1i;
        y0i = wk3r * x1i - wk3i * x1r;
        y2r = wd3r * x3r + wd3i * x3i;
        y2i = wd3r * x3i - wd3i * x3r;
        a[(ao + (j2)) as usize] = y0r + y2r;
        a[(ao + (j2 + 1)) as usize] = y0i + y2i;
        a[(ao + (j3)) as usize] = y0r - y2r;
        a[(ao + (j3 + 1)) as usize] = y0i - y2i;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[(ao + (j0)) as usize] - a[(ao + (j2 + 1)) as usize];
        x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2)) as usize];
        x1r = a[(ao + (j0)) as usize] + a[(ao + (j2 + 1)) as usize];
        x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2)) as usize];
        x2r = a[(ao + (j1)) as usize] - a[(ao + (j3 + 1)) as usize];
        x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3)) as usize];
        x3r = a[(ao + (j1)) as usize] + a[(ao + (j3 + 1)) as usize];
        x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3)) as usize];
        y0r = wd1i * x0r - wd1r * x0i;
        y0i = wd1i * x0i + wd1r * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[(ao + (j0)) as usize] = y0r + y2r;
        a[(ao + (j0 + 1)) as usize] = y0i + y2i;
        a[(ao + (j1)) as usize] = y0r - y2r;
        a[(ao + (j1 + 1)) as usize] = y0i - y2i;
        y0r = wd3i * x1r + wd3r * x1i;
        y0i = wd3i * x1i - wd3r * x1r;
        y2r = wk3i * x3r + wk3r * x3i;
        y2i = wk3i * x3i - wk3r * x3r;
        a[(ao + (j2)) as usize] = y0r + y2r;
        a[(ao + (j2 + 1)) as usize] = y0i + y2i;
        a[(ao + (j3)) as usize] = y0r - y2r;
        a[(ao + (j3 + 1)) as usize] = y0i - y2i;
        j += 2;
    }
    wk1r = w[(wo + (m)) as usize];
    wk1i = w[(wo + (m + 1)) as usize];
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[(ao + (j0)) as usize] - a[(ao + (j2 + 1)) as usize];
    x0i = a[(ao + (j0 + 1)) as usize] + a[(ao + (j2)) as usize];
    x1r = a[(ao + (j0)) as usize] + a[(ao + (j2 + 1)) as usize];
    x1i = a[(ao + (j0 + 1)) as usize] - a[(ao + (j2)) as usize];
    x2r = a[(ao + (j1)) as usize] - a[(ao + (j3 + 1)) as usize];
    x2i = a[(ao + (j1 + 1)) as usize] + a[(ao + (j3)) as usize];
    x3r = a[(ao + (j1)) as usize] + a[(ao + (j3 + 1)) as usize];
    x3i = a[(ao + (j1 + 1)) as usize] - a[(ao + (j3)) as usize];
    y0r = wk1r * x0r - wk1i * x0i;
    y0i = wk1r * x0i + wk1i * x0r;
    y2r = wk1i * x2r - wk1r * x2i;
    y2i = wk1i * x2i + wk1r * x2r;
    a[(ao + (j0)) as usize] = y0r + y2r;
    a[(ao + (j0 + 1)) as usize] = y0i + y2i;
    a[(ao + (j1)) as usize] = y0r - y2r;
    a[(ao + (j1 + 1)) as usize] = y0i - y2i;
    y0r = wk1i * x1r - wk1r * x1i;
    y0i = wk1i * x1i + wk1r * x1r;
    y2r = wk1r * x3r - wk1i * x3i;
    y2i = wk1r * x3i + wk1i * x3r;
    a[(ao + (j2)) as usize] = y0r - y2r;
    a[(ao + (j2 + 1)) as usize] = y0i - y2i;
    a[(ao + (j3)) as usize] = y0r + y2r;
    a[(ao + (j3 + 1)) as usize] = y0i + y2i;
}

fn cftf161(a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;
    let mut y4r: f32;
    let mut y4i: f32;
    let mut y5r: f32;
    let mut y5i: f32;
    let mut y6r: f32;
    let mut y6i: f32;
    let mut y7r: f32;
    let mut y7i: f32;
    let mut y8r: f32;
    let mut y8i: f32;
    let mut y9r: f32;
    let mut y9i: f32;
    let mut y10r: f32;
    let mut y10i: f32;
    let mut y11r: f32;
    let mut y11i: f32;
    let mut y12r: f32;
    let mut y12i: f32;
    let mut y13r: f32;
    let mut y13i: f32;
    let mut y14r: f32;
    let mut y14i: f32;
    let mut y15r: f32;
    let mut y15i: f32;

    wn4r = w[(wo + (1)) as usize];
    wk1r = w[(wo + (2)) as usize];
    wk1i = w[(wo + (3)) as usize];
    x0r = a[(ao + (0)) as usize] + a[(ao + (16)) as usize];
    x0i = a[(ao + (1)) as usize] + a[(ao + (17)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (16)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (17)) as usize];
    x2r = a[(ao + (8)) as usize] + a[(ao + (24)) as usize];
    x2i = a[(ao + (9)) as usize] + a[(ao + (25)) as usize];
    x3r = a[(ao + (8)) as usize] - a[(ao + (24)) as usize];
    x3i = a[(ao + (9)) as usize] - a[(ao + (25)) as usize];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y4r = x0r - x2r;
    y4i = x0i - x2i;
    y8r = x1r - x3i;
    y8i = x1i + x3r;
    y12r = x1r + x3i;
    y12i = x1i - x3r;
    x0r = a[(ao + (2)) as usize] + a[(ao + (18)) as usize];
    x0i = a[(ao + (3)) as usize] + a[(ao + (19)) as usize];
    x1r = a[(ao + (2)) as usize] - a[(ao + (18)) as usize];
    x1i = a[(ao + (3)) as usize] - a[(ao + (19)) as usize];
    x2r = a[(ao + (10)) as usize] + a[(ao + (26)) as usize];
    x2i = a[(ao + (11)) as usize] + a[(ao + (27)) as usize];
    x3r = a[(ao + (10)) as usize] - a[(ao + (26)) as usize];
    x3i = a[(ao + (11)) as usize] - a[(ao + (27)) as usize];
    y1r = x0r + x2r;
    y1i = x0i + x2i;
    y5r = x0r - x2r;
    y5i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y9r = wk1r * x0r - wk1i * x0i;
    y9i = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y13r = wk1i * x0r - wk1r * x0i;
    y13i = wk1i * x0i + wk1r * x0r;
    x0r = a[(ao + (4)) as usize] + a[(ao + (20)) as usize];
    x0i = a[(ao + (5)) as usize] + a[(ao + (21)) as usize];
    x1r = a[(ao + (4)) as usize] - a[(ao + (20)) as usize];
    x1i = a[(ao + (5)) as usize] - a[(ao + (21)) as usize];
    x2r = a[(ao + (12)) as usize] + a[(ao + (28)) as usize];
    x2i = a[(ao + (13)) as usize] + a[(ao + (29)) as usize];
    x3r = a[(ao + (12)) as usize] - a[(ao + (28)) as usize];
    x3i = a[(ao + (13)) as usize] - a[(ao + (29)) as usize];
    y2r = x0r + x2r;
    y2i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y10r = wn4r * (x0r - x0i);
    y10i = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y14r = wn4r * (x0r + x0i);
    y14i = wn4r * (x0i - x0r);
    x0r = a[(ao + (6)) as usize] + a[(ao + (22)) as usize];
    x0i = a[(ao + (7)) as usize] + a[(ao + (23)) as usize];
    x1r = a[(ao + (6)) as usize] - a[(ao + (22)) as usize];
    x1i = a[(ao + (7)) as usize] - a[(ao + (23)) as usize];
    x2r = a[(ao + (14)) as usize] + a[(ao + (30)) as usize];
    x2i = a[(ao + (15)) as usize] + a[(ao + (31)) as usize];
    x3r = a[(ao + (14)) as usize] - a[(ao + (30)) as usize];
    x3i = a[(ao + (15)) as usize] - a[(ao + (31)) as usize];
    y3r = x0r + x2r;
    y3i = x0i + x2i;
    y7r = x0r - x2r;
    y7i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y11r = wk1i * x0r - wk1r * x0i;
    y11i = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y15r = wk1r * x0r - wk1i * x0i;
    y15i = wk1r * x0i + wk1i * x0r;
    x0r = y12r - y14r;
    x0i = y12i - y14i;
    x1r = y12r + y14r;
    x1i = y12i + y14i;
    x2r = y13r - y15r;
    x2i = y13i - y15i;
    x3r = y13r + y15r;
    x3i = y13i + y15i;
    a[(ao + (24)) as usize] = x0r + x2r;
    a[(ao + (25)) as usize] = x0i + x2i;
    a[(ao + (26)) as usize] = x0r - x2r;
    a[(ao + (27)) as usize] = x0i - x2i;
    a[(ao + (28)) as usize] = x1r - x3i;
    a[(ao + (29)) as usize] = x1i + x3r;
    a[(ao + (30)) as usize] = x1r + x3i;
    a[(ao + (31)) as usize] = x1i - x3r;
    x0r = y8r + y10r;
    x0i = y8i + y10i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    x3r = y9r - y11r;
    x3i = y9i - y11i;
    a[(ao + (16)) as usize] = x0r + x2r;
    a[(ao + (17)) as usize] = x0i + x2i;
    a[(ao + (18)) as usize] = x0r - x2r;
    a[(ao + (19)) as usize] = x0i - x2i;
    a[(ao + (20)) as usize] = x1r - x3i;
    a[(ao + (21)) as usize] = x1i + x3r;
    a[(ao + (22)) as usize] = x1r + x3i;
    a[(ao + (23)) as usize] = x1i - x3r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x3r = wn4r * (x0r - x0i);
    x3i = wn4r * (x0i + x0r);
    x0r = y4r - y6i;
    x0i = y4i + y6r;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    a[(ao + (8)) as usize] = x0r + x2r;
    a[(ao + (9)) as usize] = x0i + x2i;
    a[(ao + (10)) as usize] = x0r - x2r;
    a[(ao + (11)) as usize] = x0i - x2i;
    a[(ao + (12)) as usize] = x1r - x3i;
    a[(ao + (13)) as usize] = x1i + x3r;
    a[(ao + (14)) as usize] = x1r + x3i;
    a[(ao + (15)) as usize] = x1i - x3r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    x3r = y1r - y3r;
    x3i = y1i - y3i;
    a[(ao + (0)) as usize] = x0r + x2r;
    a[(ao + (1)) as usize] = x0i + x2i;
    a[(ao + (2)) as usize] = x0r - x2r;
    a[(ao + (3)) as usize] = x0i - x2i;
    a[(ao + (4)) as usize] = x1r - x3i;
    a[(ao + (5)) as usize] = x1i + x3r;
    a[(ao + (6)) as usize] = x1r + x3i;
    a[(ao + (7)) as usize] = x1i - x3r;
}

fn cftf162(a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk2r: f32;
    let mut wk2i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;
    let mut y4r: f32;
    let mut y4i: f32;
    let mut y5r: f32;
    let mut y5i: f32;
    let mut y6r: f32;
    let mut y6i: f32;
    let mut y7r: f32;
    let mut y7i: f32;
    let mut y8r: f32;
    let mut y8i: f32;
    let mut y9r: f32;
    let mut y9i: f32;
    let mut y10r: f32;
    let mut y10i: f32;
    let mut y11r: f32;
    let mut y11i: f32;
    let mut y12r: f32;
    let mut y12i: f32;
    let mut y13r: f32;
    let mut y13i: f32;
    let mut y14r: f32;
    let mut y14i: f32;
    let mut y15r: f32;
    let mut y15i: f32;

    wn4r = w[(wo + (1)) as usize];
    wk1r = w[(wo + (4)) as usize];
    wk1i = w[(wo + (5)) as usize];
    wk3r = w[(wo + (6)) as usize];
    wk3i = -w[(wo + (7)) as usize];
    wk2r = w[(wo + (8)) as usize];
    wk2i = w[(wo + (9)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (17)) as usize];
    x1i = a[(ao + (1)) as usize] + a[(ao + (16)) as usize];
    x0r = a[(ao + (8)) as usize] - a[(ao + (25)) as usize];
    x0i = a[(ao + (9)) as usize] + a[(ao + (24)) as usize];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y0r = x1r + x2r;
    y0i = x1i + x2i;
    y4r = x1r - x2r;
    y4i = x1i - x2i;
    x1r = a[(ao + (0)) as usize] + a[(ao + (17)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (16)) as usize];
    x0r = a[(ao + (8)) as usize] + a[(ao + (25)) as usize];
    x0i = a[(ao + (9)) as usize] - a[(ao + (24)) as usize];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y8r = x1r - x2i;
    y8i = x1i + x2r;
    y12r = x1r + x2i;
    y12i = x1i - x2r;
    x0r = a[(ao + (2)) as usize] - a[(ao + (19)) as usize];
    x0i = a[(ao + (3)) as usize] + a[(ao + (18)) as usize];
    x1r = wk1r * x0r - wk1i * x0i;
    x1i = wk1r * x0i + wk1i * x0r;
    x0r = a[(ao + (10)) as usize] - a[(ao + (27)) as usize];
    x0i = a[(ao + (11)) as usize] + a[(ao + (26)) as usize];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y1r = x1r + x2r;
    y1i = x1i + x2i;
    y5r = x1r - x2r;
    y5i = x1i - x2i;
    x0r = a[(ao + (2)) as usize] + a[(ao + (19)) as usize];
    x0i = a[(ao + (3)) as usize] - a[(ao + (18)) as usize];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[(ao + (10)) as usize] + a[(ao + (27)) as usize];
    x0i = a[(ao + (11)) as usize] - a[(ao + (26)) as usize];
    x2r = wk1r * x0r + wk1i * x0i;
    x2i = wk1r * x0i - wk1i * x0r;
    y9r = x1r - x2r;
    y9i = x1i - x2i;
    y13r = x1r + x2r;
    y13i = x1i + x2i;
    x0r = a[(ao + (4)) as usize] - a[(ao + (21)) as usize];
    x0i = a[(ao + (5)) as usize] + a[(ao + (20)) as usize];
    x1r = wk2r * x0r - wk2i * x0i;
    x1i = wk2r * x0i + wk2i * x0r;
    x0r = a[(ao + (12)) as usize] - a[(ao + (29)) as usize];
    x0i = a[(ao + (13)) as usize] + a[(ao + (28)) as usize];
    x2r = wk2i * x0r - wk2r * x0i;
    x2i = wk2i * x0i + wk2r * x0r;
    y2r = x1r + x2r;
    y2i = x1i + x2i;
    y6r = x1r - x2r;
    y6i = x1i - x2i;
    x0r = a[(ao + (4)) as usize] + a[(ao + (21)) as usize];
    x0i = a[(ao + (5)) as usize] - a[(ao + (20)) as usize];
    x1r = wk2i * x0r - wk2r * x0i;
    x1i = wk2i * x0i + wk2r * x0r;
    x0r = a[(ao + (12)) as usize] + a[(ao + (29)) as usize];
    x0i = a[(ao + (13)) as usize] - a[(ao + (28)) as usize];
    x2r = wk2r * x0r - wk2i * x0i;
    x2i = wk2r * x0i + wk2i * x0r;
    y10r = x1r - x2r;
    y10i = x1i - x2i;
    y14r = x1r + x2r;
    y14i = x1i + x2i;
    x0r = a[(ao + (6)) as usize] - a[(ao + (23)) as usize];
    x0i = a[(ao + (7)) as usize] + a[(ao + (22)) as usize];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[(ao + (14)) as usize] - a[(ao + (31)) as usize];
    x0i = a[(ao + (15)) as usize] + a[(ao + (30)) as usize];
    x2r = wk1i * x0r - wk1r * x0i;
    x2i = wk1i * x0i + wk1r * x0r;
    y3r = x1r + x2r;
    y3i = x1i + x2i;
    y7r = x1r - x2r;
    y7i = x1i - x2i;
    x0r = a[(ao + (6)) as usize] + a[(ao + (23)) as usize];
    x0i = a[(ao + (7)) as usize] - a[(ao + (22)) as usize];
    x1r = wk1i * x0r + wk1r * x0i;
    x1i = wk1i * x0i - wk1r * x0r;
    x0r = a[(ao + (14)) as usize] + a[(ao + (31)) as usize];
    x0i = a[(ao + (15)) as usize] - a[(ao + (30)) as usize];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y11r = x1r + x2r;
    y11i = x1i + x2i;
    y15r = x1r - x2r;
    y15i = x1i - x2i;
    x1r = y0r + y2r;
    x1i = y0i + y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    a[(ao + (0)) as usize] = x1r + x2r;
    a[(ao + (1)) as usize] = x1i + x2i;
    a[(ao + (2)) as usize] = x1r - x2r;
    a[(ao + (3)) as usize] = x1i - x2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r - y3r;
    x2i = y1i - y3i;
    a[(ao + (4)) as usize] = x1r - x2i;
    a[(ao + (5)) as usize] = x1i + x2r;
    a[(ao + (6)) as usize] = x1r + x2i;
    a[(ao + (7)) as usize] = x1i - x2r;
    x1r = y4r - y6i;
    x1i = y4i + y6r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[(ao + (8)) as usize] = x1r + x2r;
    a[(ao + (9)) as usize] = x1i + x2i;
    a[(ao + (10)) as usize] = x1r - x2r;
    a[(ao + (11)) as usize] = x1i - x2i;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[(ao + (12)) as usize] = x1r - x2i;
    a[(ao + (13)) as usize] = x1i + x2r;
    a[(ao + (14)) as usize] = x1r + x2i;
    a[(ao + (15)) as usize] = x1i - x2r;
    x1r = y8r + y10r;
    x1i = y8i + y10i;
    x2r = y9r - y11r;
    x2i = y9i - y11i;
    a[(ao + (16)) as usize] = x1r + x2r;
    a[(ao + (17)) as usize] = x1i + x2i;
    a[(ao + (18)) as usize] = x1r - x2r;
    a[(ao + (19)) as usize] = x1i - x2i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    a[(ao + (20)) as usize] = x1r - x2i;
    a[(ao + (21)) as usize] = x1i + x2r;
    a[(ao + (22)) as usize] = x1r + x2i;
    a[(ao + (23)) as usize] = x1i - x2r;
    x1r = y12r - y14i;
    x1i = y12i + y14r;
    x0r = y13r + y15i;
    x0i = y13i - y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[(ao + (24)) as usize] = x1r + x2r;
    a[(ao + (25)) as usize] = x1i + x2i;
    a[(ao + (26)) as usize] = x1r - x2r;
    a[(ao + (27)) as usize] = x1i - x2i;
    x1r = y12r + y14i;
    x1i = y12i - y14r;
    x0r = y13r - y15i;
    x0i = y13i + y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[(ao + (28)) as usize] = x1r - x2i;
    a[(ao + (29)) as usize] = x1i + x2r;
    a[(ao + (30)) as usize] = x1r + x2i;
    a[(ao + (31)) as usize] = x1i - x2r;
}

fn cftf081(a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut wn4r: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;
    let mut y4r: f32;
    let mut y4i: f32;
    let mut y5r: f32;
    let mut y5i: f32;
    let mut y6r: f32;
    let mut y6i: f32;
    let mut y7r: f32;
    let mut y7i: f32;

    wn4r = w[(wo + (1)) as usize];
    x0r = a[(ao + (0)) as usize] + a[(ao + (8)) as usize];
    x0i = a[(ao + (1)) as usize] + a[(ao + (9)) as usize];
    x1r = a[(ao + (0)) as usize] - a[(ao + (8)) as usize];
    x1i = a[(ao + (1)) as usize] - a[(ao + (9)) as usize];
    x2r = a[(ao + (4)) as usize] + a[(ao + (12)) as usize];
    x2i = a[(ao + (5)) as usize] + a[(ao + (13)) as usize];
    x3r = a[(ao + (4)) as usize] - a[(ao + (12)) as usize];
    x3i = a[(ao + (5)) as usize] - a[(ao + (13)) as usize];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y2r = x0r - x2r;
    y2i = x0i - x2i;
    y1r = x1r - x3i;
    y1i = x1i + x3r;
    y3r = x1r + x3i;
    y3i = x1i - x3r;
    x0r = a[(ao + (2)) as usize] + a[(ao + (10)) as usize];
    x0i = a[(ao + (3)) as usize] + a[(ao + (11)) as usize];
    x1r = a[(ao + (2)) as usize] - a[(ao + (10)) as usize];
    x1i = a[(ao + (3)) as usize] - a[(ao + (11)) as usize];
    x2r = a[(ao + (6)) as usize] + a[(ao + (14)) as usize];
    x2i = a[(ao + (7)) as usize] + a[(ao + (15)) as usize];
    x3r = a[(ao + (6)) as usize] - a[(ao + (14)) as usize];
    x3i = a[(ao + (7)) as usize] - a[(ao + (15)) as usize];
    y4r = x0r + x2r;
    y4i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    x2r = x1r + x3i;
    x2i = x1i - x3r;
    y5r = wn4r * (x0r - x0i);
    y5i = wn4r * (x0r + x0i);
    y7r = wn4r * (x2r - x2i);
    y7i = wn4r * (x2r + x2i);
    a[(ao + (8)) as usize] = y1r + y5r;
    a[(ao + (9)) as usize] = y1i + y5i;
    a[(ao + (10)) as usize] = y1r - y5r;
    a[(ao + (11)) as usize] = y1i - y5i;
    a[(ao + (12)) as usize] = y3r - y7i;
    a[(ao + (13)) as usize] = y3i + y7r;
    a[(ao + (14)) as usize] = y3r + y7i;
    a[(ao + (15)) as usize] = y3i - y7r;
    a[(ao + (0)) as usize] = y0r + y4r;
    a[(ao + (1)) as usize] = y0i + y4i;
    a[(ao + (2)) as usize] = y0r - y4r;
    a[(ao + (3)) as usize] = y0i - y4i;
    a[(ao + (4)) as usize] = y2r - y6i;
    a[(ao + (5)) as usize] = y2i + y6r;
    a[(ao + (6)) as usize] = y2r + y6i;
    a[(ao + (7)) as usize] = y2i - y6r;
}

fn cftf082(a: &mut [f32], ao: i32, w: &[f32], wo: i32) {
    let mut wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;
    let mut y4r: f32;
    let mut y4i: f32;
    let mut y5r: f32;
    let mut y5i: f32;
    let mut y6r: f32;
    let mut y6i: f32;
    let mut y7r: f32;
    let mut y7i: f32;

    wn4r = w[(wo + (1)) as usize];
    wk1r = w[(wo + (2)) as usize];
    wk1i = w[(wo + (3)) as usize];
    y0r = a[(ao + (0)) as usize] - a[(ao + (9)) as usize];
    y0i = a[(ao + (1)) as usize] + a[(ao + (8)) as usize];
    y1r = a[(ao + (0)) as usize] + a[(ao + (9)) as usize];
    y1i = a[(ao + (1)) as usize] - a[(ao + (8)) as usize];
    x0r = a[(ao + (4)) as usize] - a[(ao + (13)) as usize];
    x0i = a[(ao + (5)) as usize] + a[(ao + (12)) as usize];
    y2r = wn4r * (x0r - x0i);
    y2i = wn4r * (x0i + x0r);
    x0r = a[(ao + (4)) as usize] + a[(ao + (13)) as usize];
    x0i = a[(ao + (5)) as usize] - a[(ao + (12)) as usize];
    y3r = wn4r * (x0r - x0i);
    y3i = wn4r * (x0i + x0r);
    x0r = a[(ao + (2)) as usize] - a[(ao + (11)) as usize];
    x0i = a[(ao + (3)) as usize] + a[(ao + (10)) as usize];
    y4r = wk1r * x0r - wk1i * x0i;
    y4i = wk1r * x0i + wk1i * x0r;
    x0r = a[(ao + (2)) as usize] + a[(ao + (11)) as usize];
    x0i = a[(ao + (3)) as usize] - a[(ao + (10)) as usize];
    y5r = wk1i * x0r - wk1r * x0i;
    y5i = wk1i * x0i + wk1r * x0r;
    x0r = a[(ao + (6)) as usize] - a[(ao + (15)) as usize];
    x0i = a[(ao + (7)) as usize] + a[(ao + (14)) as usize];
    y6r = wk1i * x0r - wk1r * x0i;
    y6i = wk1i * x0i + wk1r * x0r;
    x0r = a[(ao + (6)) as usize] + a[(ao + (15)) as usize];
    x0i = a[(ao + (7)) as usize] - a[(ao + (14)) as usize];
    y7r = wk1r * x0r - wk1i * x0i;
    y7i = wk1r * x0i + wk1i * x0r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y4r + y6r;
    x1i = y4i + y6i;
    a[(ao + (0)) as usize] = x0r + x1r;
    a[(ao + (1)) as usize] = x0i + x1i;
    a[(ao + (2)) as usize] = x0r - x1r;
    a[(ao + (3)) as usize] = x0i - x1i;
    x0r = y0r - y2r;
    x0i = y0i - y2i;
    x1r = y4r - y6r;
    x1i = y4i - y6i;
    a[(ao + (4)) as usize] = x0r - x1i;
    a[(ao + (5)) as usize] = x0i + x1r;
    a[(ao + (6)) as usize] = x0r + x1i;
    a[(ao + (7)) as usize] = x0i - x1r;
    x0r = y1r - y3i;
    x0i = y1i + y3r;
    x1r = y5r - y7r;
    x1i = y5i - y7i;
    a[(ao + (8)) as usize] = x0r + x1r;
    a[(ao + (9)) as usize] = x0i + x1i;
    a[(ao + (10)) as usize] = x0r - x1r;
    a[(ao + (11)) as usize] = x0i - x1i;
    x0r = y1r + y3i;
    x0i = y1i - y3r;
    x1r = y5r + y7r;
    x1i = y5i + y7i;
    a[(ao + (12)) as usize] = x0r - x1i;
    a[(ao + (13)) as usize] = x0i + x1r;
    a[(ao + (14)) as usize] = x0r + x1i;
    a[(ao + (15)) as usize] = x0i - x1r;
}

fn bitrv2(n: i32, ip: &[i32], ipo: i32, a: &mut [f32], ao: i32) {
    let mut j: i32;
    let mut j1: i32;
    let mut k: i32;
    let mut k1: i32;
    let mut l: i32;
    let mut m: i32;
    let mut nh: i32;
    let mut nm: i32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = 1;
    l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    nh = n >> 1;
    nm = 4 * m;
    if l == 8 {
        k = 0;
        while k < m {
            j = 0;
            while j < k {
                j1 = 4 * j + 2 * ip[(ipo + (m + k)) as usize];
                k1 = 4 * k + 2 * ip[(ipo + (m + j)) as usize];
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j += 1;
            }
            k1 = 4 * k + 2 * ip[(ipo + (m + k)) as usize];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nm;
            k1 += 2 * nm;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            k += 1;
        }
    } else {
        k = 0;
        while k < m {
            j = 0;
            while j < k {
                j1 = 4 * j + ip[(ipo + (m + k)) as usize];
                k1 = 4 * k + ip[(ipo + (m + j)) as usize];
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j += 1;
            }
            k1 = 4 * k + ip[(ipo + (m + k)) as usize];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nm;
            k1 += nm;
            xr = a[(ao + (j1)) as usize];
            xi = a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            k += 1;
        }
    }
}

fn bitrv2conj(n: i32, ip: &[i32], ipo: i32, a: &mut [f32], ao: i32) {
    let mut j: i32;
    let mut j1: i32;
    let mut k: i32;
    let mut k1: i32;
    let mut l: i32;
    let mut m: i32;
    let mut nh: i32;
    let mut nm: i32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = 1;
    l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    nh = n >> 1;
    nm = 4 * m;
    if l == 8 {
        k = 0;
        while k < m {
            j = 0;
            while j < k {
                j1 = 4 * j + 2 * ip[(ipo + (m + k)) as usize];
                k1 = 4 * k + 2 * ip[(ipo + (m + j)) as usize];
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j += 1;
            }
            k1 = 4 * k + 2 * ip[(ipo + (m + k)) as usize];
            j1 = k1 + 2;
            k1 += nh;
            a[(ao + (j1 - 1)) as usize] = -a[(ao + (j1 - 1)) as usize];
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            a[(ao + (k1 + 3)) as usize] = -a[(ao + (k1 + 3)) as usize];
            j1 += nm;
            k1 += 2 * nm;
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            a[(ao + (j1 - 1)) as usize] = -a[(ao + (j1 - 1)) as usize];
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            a[(ao + (k1 + 3)) as usize] = -a[(ao + (k1 + 3)) as usize];
            k += 1;
        }
    } else {
        k = 0;
        while k < m {
            j = 0;
            while j < k {
                j1 = 4 * j + ip[(ipo + (m + k)) as usize];
                k1 = 4 * k + ip[(ipo + (m + j)) as usize];
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[(ao + (j1)) as usize];
                xi = -a[(ao + (j1 + 1)) as usize];
                yr = a[(ao + (k1)) as usize];
                yi = -a[(ao + (k1 + 1)) as usize];
                a[(ao + (j1)) as usize] = yr;
                a[(ao + (j1 + 1)) as usize] = yi;
                a[(ao + (k1)) as usize] = xr;
                a[(ao + (k1 + 1)) as usize] = xi;
                j += 1;
            }
            k1 = 4 * k + ip[(ipo + (m + k)) as usize];
            j1 = k1 + 2;
            k1 += nh;
            a[(ao + (j1 - 1)) as usize] = -a[(ao + (j1 - 1)) as usize];
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            a[(ao + (k1 + 3)) as usize] = -a[(ao + (k1 + 3)) as usize];
            j1 += nm;
            k1 += nm;
            a[(ao + (j1 - 1)) as usize] = -a[(ao + (j1 - 1)) as usize];
            xr = a[(ao + (j1)) as usize];
            xi = -a[(ao + (j1 + 1)) as usize];
            yr = a[(ao + (k1)) as usize];
            yi = -a[(ao + (k1 + 1)) as usize];
            a[(ao + (j1)) as usize] = yr;
            a[(ao + (j1 + 1)) as usize] = yi;
            a[(ao + (k1)) as usize] = xr;
            a[(ao + (k1 + 1)) as usize] = xi;
            a[(ao + (k1 + 3)) as usize] = -a[(ao + (k1 + 3)) as usize];
            k += 1;
        }
    }
}

fn rftfsub(n: i32, a: &mut [f32], ao: i32, nc: i32, c: &[f32], co: i32) {
    let mut j: i32;
    let mut k: i32;
    let mut kk: i32;
    let mut ks: i32;
    let mut m: i32;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    j = 2;
    while j < m {
        k = n - j;
        kk += ks;
        wkr = 0.5f32 - c[(co + (nc - kk)) as usize];
        wki = c[(co + (kk)) as usize];
        xr = a[(ao + (j)) as usize] - a[(ao + (k)) as usize];
        xi = a[(ao + (j + 1)) as usize] + a[(ao + (k + 1)) as usize];
        yr = wkr * xr - wki * xi;
        yi = wkr * xi + wki * xr;
        a[(ao + (j)) as usize] -= yr;
        a[(ao + (j + 1)) as usize] -= yi;
        a[(ao + (k)) as usize] += yr;
        a[(ao + (k + 1)) as usize] -= yi;
        j += 2;
    }
}

fn rftbsub(n: i32, a: &mut [f32], ao: i32, nc: i32, c: &[f32], co: i32) {
    let mut j: i32;
    let mut k: i32;
    let mut kk: i32;
    let mut ks: i32;
    let mut m: i32;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    j = 2;
    while j < m {
        k = n - j;
        kk += ks;
        wkr = 0.5f32 - c[(co + (nc - kk)) as usize];
        wki = c[(co + (kk)) as usize];
        xr = a[(ao + (j)) as usize] - a[(ao + (k)) as usize];
        xi = a[(ao + (j + 1)) as usize] + a[(ao + (k + 1)) as usize];
        yr = wkr * xr + wki * xi;
        yi = wkr * xi - wki * xr;
        a[(ao + (j)) as usize] -= yr;
        a[(ao + (j + 1)) as usize] -= yi;
        a[(ao + (k)) as usize] += yr;
        a[(ao + (k + 1)) as usize] -= yi;
        j += 2;
    }
}

// ── driver (short enough to hand-write; mirrors `AUP_FFTW_rdft` etc.) ───────

fn cftfsub(n: i32, a: &mut [f32], ao: i32, ip: &[i32], ipo: i32, nw: i32, w: &[f32], wo: i32) {
    if n > 8 {
        if n > 32 {
            cftf1st(n, a, ao, w, wo + nw - (n >> 2));
            if n > 512 {
                cftrec4(n, a, ao, nw, w, wo);
            } else if n > 128 {
                cftleaf(n, 1, a, ao, nw, w, wo);
            } else {
                unreachable!("n = {n}: only the 1024-point path is ported");
            }
            bitrv2(n, ip, ipo, a, ao);
        } else {
            unreachable!("n = {n}: only the 1024-point path is ported");
        }
    }
}

fn cftbsub(n: i32, a: &mut [f32], ao: i32, ip: &[i32], ipo: i32, nw: i32, w: &[f32], wo: i32) {
    if n > 8 {
        if n > 32 {
            cftb1st(n, a, ao, w, wo + nw - (n >> 2));
            if n > 512 {
                cftrec4(n, a, ao, nw, w, wo);
            } else if n > 128 {
                cftleaf(n, 1, a, ao, nw, w, wo);
            } else {
                unreachable!("n = {n}: only the 1024-point path is ported");
            }
            bitrv2conj(n, ip, ipo, a, ao);
        } else {
            unreachable!("n = {n}: only the 1024-point path is ported");
        }
    }
}

/// `AUP_FFTW_rdft`. `isgn >= 0` is the forward transform.
fn rdft(n: i32, isgn: i32, a: &mut [f32]) {
    let nw = IP[0];
    let nc = IP[1];
    if isgn >= 0 {
        cftfsub(n, a, 0, &IP, 0, nw, &W, 0);
        rftfsub(n, a, 0, nc, &W, nw);
        let xi = a[0] - a[1];
        a[0] += a[1];
        a[1] = xi;
    } else {
        a[1] = 0.5 * (a[0] - a[1]);
        a[0] -= a[1];
        rftbsub(n, a, 0, nc, &W, nw);
        cftbsub(n, a, 0, &IP, 0, nw, &W, 0);
    }
}

/// `AUP_FFTW_r2c_1024`: real input → format2 spectrum.
pub fn r2c(input: &[f32], out: &mut [f32]) {
    assert!(input.len() >= N && out.len() >= N);
    let mut tmp = [0f32; N];
    for i in 0..N {
        tmp[i] = input[i] * SCALE;
    }
    rdft(N as i32, 1, &mut tmp);
    out[0] = tmp[0];
    out[N - 1] = tmp[1];
    let mut i = 1;
    while i < N - 1 {
        out[i] = tmp[i + 1];
        out[i + 1] = -tmp[i + 2];
        i += 2;
    }
}

/// `AUP_FFTW_c2r_1024`: format2 spectrum → real output (unscaled; pair with
/// [`rescale_ifft_out`]).
pub fn c2r(input: &[f32], out: &mut [f32]) {
    assert!(input.len() >= N && out.len() >= N);
    out[0] = input[0];
    out[1] = input[N - 1];
    let mut i = 2;
    while i < N {
        out[i] = input[i - 1];
        out[i + 1] = -input[i];
        i += 2;
    }
    rdft(N as i32, -1, out);
    for v in out[..N].iter_mut() {
        *v *= 2.0;
    }
}

/// `AUP_FFTW_InplaceTransf`. `direction = 0` is format1 → format2, `1` the
/// reverse.
pub fn inplace_transf(direction: i32, buf: &mut [f32]) {
    assert!(buf.len() >= N);
    if direction == 0 {
        let nyq = buf[1];
        let mut i = 1;
        while i < N - 1 {
            buf[i] = buf[i + 1];
            buf[i + 1] = -buf[i + 2];
            i += 2;
        }
        buf[N - 1] = nyq;
    } else {
        let nyq = buf[N - 1];
        let mut i = N - 1;
        while i > 2 {
            buf[i] = -buf[i - 1];
            buf[i - 1] = buf[i - 2];
            i -= 2;
        }
        buf[1] = nyq;
    }
}

/// `AUP_FFTW_RescaleFFTOut` — undoes the `SCALE` (`1/N`) factor, leaving the
/// plain unnormalized DFT.
pub fn rescale_fft_out(buf: &mut [f32]) {
    for v in buf[..N].iter_mut() {
        *v *= N as f32;
    }
}

/// `AUP_FFTW_RescaleIFFTOut`.
pub fn rescale_ifft_out(buf: &mut [f32]) {
    for v in buf[..N].iter_mut() {
        *v *= 0.5;
    }
}

/// The frontend's forward path: window → `|X[k]|²`, exactly as `stft.cc`
/// followed by `AUP_Aed_CalcBinPow` does it.
///
/// The reference gets here via `r2c` → `inplace_transf(1)` → `rescale_fft_out`,
/// but `r2c` permutes Ooura's packed output into format2 and `inplace_transf(1)`
/// permutes it straight back — format1 *is* Ooura's layout, `[R₀, R_nyq, R₁,
/// I₁, …]`. Both passes are pure permutation, so dropping them changes no
/// rounding; `frontend_is_bit_identical_to_the_reference_dsp` holds it to that.
///
/// `input` is zero-padded to [`N`]; `out` receives [`BINS`] powers.
pub fn power_spectrum(input: &[f32], out: &mut [f32]) {
    assert!(input.len() <= N, "input longer than the transform");
    assert_eq!(out.len(), BINS);
    let mut buf = [0f32; N];
    for (dst, &src) in buf[..input.len()].iter_mut().zip(input) {
        *dst = src * SCALE;
    }
    rdft(N as i32, 1, &mut buf);
    // `rescale_fft_out` folded in: undo SCALE, then square.
    let n = N as f32;
    let (dc, nyq) = (buf[0] * n, buf[1] * n);
    out[0] = dc * dc;
    out[BINS - 1] = nyq * nyq;
    for k in 1..BINS - 1 {
        let (re, im) = (buf[2 * k] * n, buf[2 * k + 1] * n);
        out[k] = re * re + im * im;
    }
}

/// The pitch estimator's `lpc_from_bands` inverse transform: a real, symmetric
/// spectrum `xr` (Nyquist already zeroed) back to `out.len()` autocorrelation lags.
///
/// Same shortcut as [`power_spectrum`] in the other direction — the reference's
/// `inplace_transf(0)` followed by `c2r`'s unpacking is the identity on
/// format1, so the spectrum is laid out for `rdft` directly. The `×2` from
/// `c2r` and the `×0.5` from `rescale_ifft_out` are kept explicit; both are
/// exact in binary and cancel.
pub fn real_spectrum_to_autocorrelation(xr: &[f32], out: &mut [f32]) {
    assert_eq!(xr.len(), BINS);
    assert!(out.len() <= N);
    let mut buf = [0f32; N];
    buf[0] = xr[0];
    buf[1] = xr[BINS - 1];
    for i in 1..BINS - 1 {
        buf[i << 1] = xr[i]; // imaginary parts stay zero
    }
    rdft(N as i32, -1, &mut buf);
    for (dst, &v) in out.iter_mut().zip(buf.iter()) {
        *dst = v * 2.0 * 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use alloc::{vec, vec::Vec};

    /// A real, symmetric spectrum must invert to a real, symmetric sequence.
    #[test]
    fn round_trip_of_a_single_bin_is_a_cosine() {
        let mut xr = vec![0f32; BINS];
        xr[4] = 1.0;
        let mut lags = vec![0f32; 17];
        real_spectrum_to_autocorrelation(&xr, &mut lags);
        for (j, &got) in lags.iter().enumerate() {
            let want = (2.0 * core::f64::consts::PI * (4 * j) as f64 / N as f64).cos() as f32;
            assert!((got - want).abs() < 1e-5, "lag {j}: {got} vs {want}");
        }
    }

    /// The forward transform against a naive DFT, on a signal with enough
    /// dynamic range to catch an indexing slip.
    #[test]
    fn power_spectrum_matches_a_naive_dft() {
        let x: Vec<f32> = (0..768)
            .map(|i| (i as f32 * 0.37).sin() * 1000.0 + (i as f32 * 0.05).cos() * 200.0)
            .collect();
        let mut got = vec![0f32; BINS];
        power_spectrum(&x, &mut got);
        for k in [0usize, 1, 7, 64, 300, BINS - 1] {
            let (mut re, mut im) = (0f64, 0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * core::f64::consts::PI * (k * t) as f64 / N as f64;
                re += v as f64 * a.cos();
                im += v as f64 * a.sin();
            }
            let want = re * re + im * im;
            let rel = (got[k] as f64 - want).abs() / want.max(1.0);
            assert!(rel < 1e-4, "bin {k}: {} vs {want} (rel {rel:.2e})", got[k]);
        }
    }
}
