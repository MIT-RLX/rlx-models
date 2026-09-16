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

//! Character n-gram F1 (chrF), matching `rlx-translate::score::chrf`.

fn ngrams(s: &str, n: usize) -> Vec<String> {
    let c: Vec<char> = s
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if c.len() < n {
        return Vec::new();
    }
    (0..=c.len() - n)
        .map(|i| c[i..i + n].iter().collect())
        .collect()
}

/// chrF: character n-gram F1 averaged over n = 1..=4, in `0.0..=1.0`.
pub fn chrf(got: &str, want: &str) -> f64 {
    let mut total = 0.0;
    let mut used = 0usize;
    for n in 1..=4 {
        let (g, w) = (ngrams(got, n), ngrams(want, n));
        if g.is_empty() || w.is_empty() {
            continue;
        }
        let mut pool = g.clone();
        let mut hits = 0usize;
        for t in &w {
            if let Some(i) = pool.iter().position(|x| x == t) {
                pool.remove(i);
                hits += 1;
            }
        }
        let (p, r) = (hits as f64 / g.len() as f64, hits as f64 / w.len() as f64);
        if p + r > 0.0 {
            total += 2.0 * p * r / (p + r);
        }
        used += 1;
    }
    if used == 0 { 0.0 } else { total / used as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_one() {
        assert!((chrf("hello", "hello") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn unrelated_is_low() {
        assert!(chrf("abc", "xyz") < 0.2);
    }
}
