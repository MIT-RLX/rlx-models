#![cfg(feature = "rten-inference")]
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

use rlx_ocr::DetectionParams;
use rlx_ocr::inference::RtenTextDetector;

#[test]
fn probe_detection_hw() {
    let p = std::env::var("OCR_DET_RTEN")
        .unwrap_or_else(|_| "/tmp/rlx-ocr-models/text-detection-ssfbcj81.rten".into());
    if !std::path::Path::new(&p).is_file() {
        eprintln!("skip: no detection rten at {p}");
        return;
    }
    let d = RtenTextDetector::from_path(&p, DetectionParams::default()).unwrap();
    println!("fixed_input_hw={:?}", d.fixed_input_hw());
}
