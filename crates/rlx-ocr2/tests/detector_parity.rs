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

//! Detector head parity: native rlx graph vs a validated numeric fixture.
//! Env: OCR2_DET_FIXTURES (dir with det_input.bin + <head>.bin), OCR2_DET_RECIPE,
//! OCR2_DET_WEIGHTS. Sweeps OCR2_DEVICES (default cpu).

mod common;
use common::{cosine, devices, read_f32};
use rlx_ocr2::Detector;
use std::path::PathBuf;

#[test]
fn detector_cpu_parity_vs_expected() {
    let Ok(fix) = std::env::var("OCR2_DET_FIXTURES") else {
        eprintln!("OCR2_DET_FIXTURES unset — skipping");
        return;
    };
    let fix = PathBuf::from(fix);
    let recipe = PathBuf::from(std::env::var("OCR2_DET_RECIPE").expect("OCR2_DET_RECIPE"));
    let weights = PathBuf::from(std::env::var("OCR2_DET_WEIGHTS").expect("OCR2_DET_WEIGHTS"));

    let input = read_f32(fix.join("det_input.bin"));
    for (name, device) in devices() {
        let det = Detector::load(recipe.as_path(), weights.as_path(), device).unwrap();
        let outs = det.forward(&input).unwrap();
        let mut worst = 1.0f32;
        println!("device [{name}]:");
        for (head, data) in &outs {
            let expected = read_f32(fix.join(format!("{head}.bin")));
            assert_eq!(data.len(), expected.len(), "{head} size mismatch");
            let cos = cosine(data, &expected);
            let mad = data
                .iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).abs())
                .fold(0., f32::max);
            println!("  {head:24} cos={cos:.5} maxabs={mad:.4e}");
            if cos.is_finite() {
                worst = worst.min(cos);
            }
        }
        assert!(
            worst > 0.999,
            "[{name}] worst detector head cos {worst} below 0.999"
        );
    }
}
