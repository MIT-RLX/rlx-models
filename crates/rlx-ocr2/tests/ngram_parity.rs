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

//! The native n-gram model matches the stored expected scores (`ngram_ref.tsv`),
//! which are exact for context length ≤ 2 (order-3 table).

use rlx_ocr2::NgramModel;
use std::path::PathBuf;

#[test]
fn ngram_reproduces_expected() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let bin = dir.join("ngram.bin");
    if !bin.is_file() {
        eprintln!("ngram.bin fixture missing — skipping");
        return;
    }
    let model = NgramModel::load(&bin).unwrap();
    let refs = std::fs::read_to_string(dir.join("ngram_ref.tsv")).unwrap();
    let mut max_err = 0f32;
    let mut n = 0;
    for line in refs.lines() {
        let (seq_s, exp_s) = line.split_once('\t').unwrap();
        let seq: Vec<u32> = seq_s.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let exp: f32 = exp_s.parse().unwrap();
        let got = model.joint(&seq);
        max_err = max_err.max((got - exp).abs());
        n += 1;
    }
    println!("n-gram parity: order={} {} seqs, max_abs_err={:.6}", model.order, n, max_err);
    assert!(max_err < 1e-3, "n-gram native vs expected max err {max_err}");
}
