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

//! Target encoding of categorical covariates (upstream `timestral.covariate_encoding`).
//!
//! Timestral was trained on numeric covariates, so in series `i` category `c` becomes
//! `(smooth * mean_i + sum_{i,c}) / (smooth + count_{i,c})` over the series' context,
//! with `smooth = 1`. NaN is a category of its own; a category never seen in the
//! context (a new value in a known covariate's horizon) resolves to the series mean.
//! In f64, as upstream computes it.

/// Weight of the series mean in each category mean.
pub const SMOOTH: f64 = 1.0;

/// Encode `past_codes` (aligned with `target`) and optionally `future_codes`.
///
/// Non-finite `target` entries are left out of the means.
pub fn target_encode(
    target: &[f64],
    past_codes: &[f64],
    future_codes: Option<&[f64]>,
) -> (Vec<f64>, Option<Vec<f64>>) {
    assert_eq!(
        target.len(),
        past_codes.len(),
        "target and codes must align"
    );
    let observed: Vec<bool> = target.iter().map(|t| t.is_finite()).collect();
    let count = observed.iter().filter(|&&o| o).count() as f64;
    // Sums are taken around the first observed value, so a large target level does not
    // swamp them. `lookup` adds the pivot back.
    let pivot = target
        .iter()
        .copied()
        .find(|t| t.is_finite())
        .unwrap_or(0.0);
    let total: f64 = target
        .iter()
        .filter(|t| t.is_finite())
        .map(|t| t - pivot)
        .sum();
    let mean_dev = if count > 0.0 { total / count } else { 0.0 };

    let mut categories: Vec<f64> = past_codes
        .iter()
        .copied()
        .filter(|c| c.is_finite())
        .collect();
    categories.sort_by(|a, b| a.partial_cmp(b).unwrap());
    categories.dedup();
    let nan_slot = categories.len();
    let slot = |c: f64| -> Option<usize> {
        if c.is_nan() {
            Some(nan_slot)
        } else {
            categories
                .binary_search_by(|x| x.partial_cmp(&c).unwrap())
                .ok()
        }
    };

    let mut sums = vec![0f64; nan_slot + 1];
    let mut counts = vec![0f64; nan_slot + 1];
    let past_slots: Vec<usize> = past_codes
        .iter()
        .map(|&c| slot(c).expect("every past code is a category"))
        .collect();
    for ((&s, &t), &o) in past_slots.iter().zip(target).zip(&observed) {
        if o {
            sums[s] += t - pivot;
            counts[s] += 1.0;
        }
    }
    let mut lookup: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .map(|(s, c)| pivot + (SMOOTH * mean_dev + s) / (SMOOTH + c))
        .collect();
    lookup.push(pivot + mean_dev);
    let unseen = nan_slot + 1;

    let past = past_slots.iter().map(|&s| lookup[s]).collect();
    let future = future_codes.map(|codes| {
        codes
            .iter()
            .map(|&c| lookup[slot(c).unwrap_or(unseen)])
            .collect()
    });
    (past, future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothed_means_and_unseen_categories() {
        let target = [1.0, 3.0, 5.0, f64::NAN];
        let codes = [0.0, 0.0, 7.0, 7.0];
        let (past, fut) = target_encode(&target, &codes, Some(&[7.0, 9.0, f64::NAN]));
        // Series mean 3; category 0: (3 + 4) / 3; category 7: (3 + 5) / 2.
        let close = |got: &[f64], want: &[f64]| {
            got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6)
        };
        assert!(close(&past, &[7.0 / 3.0, 7.0 / 3.0, 4.0, 4.0]), "{past:?}");
        // NaN never occurred in the context: its slot holds the prior alone (3 / 1).
        let fut = fut.unwrap();
        assert!(close(&fut, &[4.0, 3.0, 3.0]), "{fut:?}");
    }
}
