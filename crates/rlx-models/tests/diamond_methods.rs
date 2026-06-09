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

//! Diamond method parsing and weighted renoise math.

use rlx_diamond::{renoise, renoise_params, t_prime_from_snr};
use rlx_models::flux2::DiamondMethod;

#[test]
fn diamond_method_parse() {
    assert_eq!(DiamondMethod::parse("glass"), Some(DiamondMethod::Glass));
    assert_eq!(
        DiamondMethod::parse("weighted_diamond"),
        Some(DiamondMethod::Weighted)
    );
    assert_eq!(DiamondMethod::parse("dps"), Some(DiamondMethod::Dps));
    assert!(DiamondMethod::parse("unknown").is_none());
}

#[test]
fn renoise_shape() {
    let x = vec![1.0f32; 4];
    let eps = vec![0.1; 4];
    let (scale, std) = renoise_params(0.6, 0.3);
    let y = renoise(&x, scale, std, &eps);
    assert_eq!(y.len(), 4);
    assert!(y[0].is_finite());
}

#[test]
fn t_prime_snr_increases_with_snr() {
    let t = 0.5f32;
    let tp_low = t_prime_from_snr(t, 0.25);
    let tp_high = t_prime_from_snr(t, 4.0);
    assert!(tp_low < tp_high);
}
