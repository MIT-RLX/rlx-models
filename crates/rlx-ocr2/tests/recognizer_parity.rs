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

//! Numeric parity of the native rlx recognizer vs a validated numpy fixture.
//! Env-gated so CI without fixtures skips. Set:
//!   OCR2_FIXTURES = dir with rec_input.bin ([1,1,32,W] f32) + rec_logits.bin ([seq,439] f32)
//!   OCR2_WEIGHTS  = recognizer.safetensors
//!   OCR2_CODEMAP  = codemap.txt
//! Sweeps OCR2_DEVICES (default cpu).

mod common;
use common::{cosine, devices, max_abs, read_f32};
use rlx_ocr2::Recognizer;
use std::path::PathBuf;

#[test]
fn recognizer_cpu_parity_vs_numpy() {
    let Ok(fix) = std::env::var("OCR2_FIXTURES") else {
        eprintln!("OCR2_FIXTURES unset — skipping");
        return;
    };
    let fix = PathBuf::from(fix);
    let weights = PathBuf::from(std::env::var("OCR2_WEIGHTS").expect("OCR2_WEIGHTS"));
    let codemap = PathBuf::from(std::env::var("OCR2_CODEMAP").expect("OCR2_CODEMAP"));

    let input = read_f32(fix.join("rec_input.bin"));
    let expected = read_f32(fix.join("rec_logits.bin"));
    let width = input.len() / 32;

    for (name, device) in devices() {
        let rec = Recognizer::load(weights.as_path(), codemap.as_path(), device).unwrap();
        let logits = rec.forward_logits(&input, width).unwrap();
        assert_eq!(logits.len(), expected.len(), "logit count mismatch");
        let cos = cosine(&logits, &expected);
        let mad = max_abs(&logits, &expected);
        println!(
            "recognizer [{name:5}] cos={cos:.6} max_abs={mad:.4e} (n={})",
            logits.len()
        );
        assert!(
            cos > 0.999,
            "recognizer[{name}] cos {cos} below 0.999 (max_abs {mad})"
        );
    }
}
