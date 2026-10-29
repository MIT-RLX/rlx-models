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

//! End to end: the native pipeline vs upstream `indra-ocr --format json` on
//! the same images (`scripts/gen_reference.sh`).
//!
//! `INDRA_MODEL_DIR` = converted models, `INDRA_REFERENCE_DIR` = the upstream
//! `.jsonl` files (each names its image in `"file"`). Skips when unset.
//! Reports, per page, the character error rate of the page text and how many
//! lines match exactly; fails above 1% CER.
//!
//! ```sh
//! INDRA_MODEL_DIR=.cache/indra INDRA_REFERENCE_DIR=.cache/indra-ref \
//!   cargo test -p rlx-indra --release --test indra_reference -- --nocapture
//! ```

use rlx_indra::{Indra, Options, load_gray};
use serde_json::Value;

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[test]
fn pages_match_upstream() {
    let (Some(models), Some(refs)) = (
        std::env::var_os("INDRA_MODEL_DIR"),
        std::env::var_os("INDRA_REFERENCE_DIR"),
    ) else {
        eprintln!("skip: set INDRA_MODEL_DIR and INDRA_REFERENCE_DIR");
        return;
    };
    let ocr = Indra::load(&models, Options::default()).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&refs)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .jsonl files in INDRA_REFERENCE_DIR");
    let (mut errs, mut total) = (0usize, 0usize);
    for f in files {
        for line in std::fs::read_to_string(&f).unwrap().lines() {
            let want: Value = serde_json::from_str(line).unwrap();
            let image = want["file"].as_str().unwrap();
            let page = ocr.read(&load_gray(image.as_ref()).unwrap(), true).unwrap();
            let want_lines: Vec<&str> = want["lines"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l["text"].as_str().unwrap_or(""))
                .collect();
            let got_lines: Vec<&str> = page.lines.iter().map(|l| l.text.as_str()).collect();
            let a: Vec<char> = want_lines.join("\n").chars().collect();
            let b: Vec<char> = got_lines.join("\n").chars().collect();
            let d = edit_distance(&a, &b);
            let same = want_lines
                .iter()
                .zip(&got_lines)
                .filter(|(x, y)| x == y)
                .count();
            eprintln!(
                "{image}: CER {:.3}% ({d}/{}), lines {same}/{} identical (native {}), rotation {} vs {}, \
                 markdown {}",
                100.0 * d as f64 / a.len().max(1) as f64,
                a.len(),
                want_lines.len(),
                got_lines.len(),
                want["rotation"],
                page.rotation,
                if want["markdown"].as_str() == page.markdown.as_deref() {
                    "identical"
                } else {
                    "differs"
                },
            );
            errs += d;
            total += a.len();
        }
    }
    let cer = errs as f64 / total.max(1) as f64;
    eprintln!("overall CER {:.3}%", 100.0 * cer);
    assert!(cer < 0.01, "CER {cer} vs upstream");
}
