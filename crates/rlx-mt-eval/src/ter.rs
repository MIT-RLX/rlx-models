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

//! Translation Edit Rate (word-level), lower is better.

fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '\'' || c == '’' {
            cur.extend(c.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Word TER = edit distance(hyp, ref) / |ref|. Identical → 0.0.
pub fn ter(hypothesis: &str, reference: &str) -> f64 {
    let ht = tokens(hypothesis);
    let rt = tokens(reference);
    if rt.is_empty() {
        return if ht.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&ht, &rt) as f64 / rt.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_zero() {
        assert!((ter("a b c", "a b c") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn total_mismatch_high() {
        assert!(ter("x y z", "a b c") > 0.9);
    }
}
