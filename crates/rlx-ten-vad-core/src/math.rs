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

//! The handful of transcendentals the pipeline needs, behind one seam.
//!
//! `core` has no `f32::ln` / `sqrt` / `cos` — those live in `std`. So every call
//! site goes through this module, which routes to the platform libm under `std`
//! and to the `libm` crate otherwise.
//!
//! The distinction is not cosmetic. The log-mel and the pitch band energies are
//! held **bit-identical** to the upstream C reference, and that holds against
//! *the platform's* libm — Apple's and glibc's already differ in the last place.
//! Swapping in the `libm` crate changes them again. `std` builds therefore keep
//! the platform path so the parity fixtures still hold; `no_std` builds take
//! `libm`'s answer, which has the compensating virtue of being identical on
//! every target.

#![allow(dead_code)]

macro_rules! shim {
    ($($(#[$m:meta])* $name:ident($($arg:ident),*) => $std:expr, $libm:expr;)*) => {
        $(
            $(#[$m])*
            #[inline]
            pub fn $name($($arg: f32),*) -> f32 {
                #[cfg(feature = "std")]
                { $std }
                #[cfg(not(feature = "std"))]
                { $libm }
            }
        )*
    };
}

shim! {
    ln(x)      => x.ln(),    libm::logf(x);
    log10(x)   => x.log10(), libm::log10f(x);
    exp(x)     => x.exp(),   libm::expf(x);
    tanh(x)    => x.tanh(),  libm::tanhf(x);
    sqrt(x)    => x.sqrt(),  libm::sqrtf(x);
    cos(x)     => x.cos(),   libm::cosf(x);
    sin(x)     => x.sin(),   libm::sinf(x);
    round(x)   => x.round(), libm::roundf(x);
    abs(x)     => x.abs(),   libm::fabsf(x);
    /// `x^y`. Prefer [`pow10`] for base 10 — spelled as the reference spells it.
    powf(x, y) => x.powf(y), libm::powf(x, y);
}

/// `10^x`, spelled as the C reference spells it (`powf(10.f, x)`) so the last
/// place matches; `exp10` is a different function with a different answer.
#[inline]
pub fn pow10(x: f32) -> f32 {
    powf(10.0, x)
}

/// `f64` sine — the companion to [`cos64`], for the same reason.
#[inline]
pub fn sin64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(x)
    }
}

/// `f64` cosine — builds twiddle tables and the test references.
///
/// Not a stylistic choice: an `f32` has 24 mantissa bits and a Q30 twiddle
/// needs 30, so building one through `cos` leaves its low six bits as noise.
#[inline]
pub fn cos64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cos(x)
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}
