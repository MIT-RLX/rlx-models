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

//! A minimal `f64` complex number, for the frequency-space representation.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A complex number with `f64` parts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// `re + i·im`.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `0`.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// A purely real number.
    pub const fn real(re: f64) -> Self {
        Self::new(re, 0.0)
    }

    /// `e^{iθ}`.
    pub fn from_angle(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    /// `|z|²`.
    pub fn norm_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// `|z|`.
    pub fn abs(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// `z̄`.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// `z · s` for real `s`.
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// The normalised cardinal sine `sin(πx)/(πx)`, with `sinc(0) = 1`.
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_follows_the_usual_rules() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(-3.0, 0.5);
        assert_eq!(a * b, Complex::new(-3.0 - 1.0, 0.5 - 6.0));
        assert_eq!(a + b, Complex::new(-2.0, 2.5));
        assert_eq!(a - b, Complex::new(4.0, 1.5));
        assert_eq!((a * a.conj()).re, a.norm_squared());
        assert!((a * a.conj()).im.abs() < 1e-15);
        assert!((a.abs() - 5f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn from_angle_lies_on_the_unit_circle_and_adds_phases() {
        for theta in [-2.0, 0.0, 0.7, 3.5] {
            assert!((Complex::from_angle(theta).abs() - 1.0).abs() < 1e-15);
        }
        let sum = Complex::from_angle(0.4) * Complex::from_angle(1.1);
        let direct = Complex::from_angle(1.5);
        assert!((sum - direct).abs() < 1e-12);
    }

    /// `sinc` is the transform of a box, so the `evaluate_spatial_grid`
    /// envelope depends on the normalised convention and on `sinc(0) = 1`.
    #[test]
    fn sinc_is_the_normalised_cardinal_sine() {
        assert_eq!(sinc(0.0), 1.0);
        for k in 1..6 {
            assert!(sinc(k as f64).abs() < 1e-15, "sinc({k}) should vanish");
        }
        assert!((sinc(0.5) - 2.0 / std::f64::consts::PI).abs() < 1e-15);
        // Continuity through the removable singularity.
        assert!((sinc(1e-9) - 1.0).abs() < 1e-15);
    }
}
