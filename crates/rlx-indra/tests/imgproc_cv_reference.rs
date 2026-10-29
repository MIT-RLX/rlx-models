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

//! `imgproc` vs OpenCV on fixtures from `scripts/gen_cv_fixtures.py`
//! (`INDRA_CV_FIXTURES=<dir>`; skips when unset). Reports the share of
//! pixels that differ and the largest difference per case.
//!
//! The warp and area paths follow OpenCV's generic code and agree to a level
//! (float accumulation order). 8-bit `INTER_LINEAR` is platform-dependent in
//! OpenCV itself: arm64 builds hand it to a HAL (Carotene / KleidiCV), x86
//! builds run the generic fixed-point path implemented here, so linear cases
//! get a looser bound.

use rlx_indra::imgproc::{Border, Gray, Interp, resize, rotate};
use serde_json::Value;

#[test]
fn imgproc_matches_opencv() {
    let Some(dir) = std::env::var_os("INDRA_CV_FIXTURES") else {
        eprintln!("skip: set INDRA_CV_FIXTURES");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let (mut worst, mut worst_linear) = (0u8, 0u8);
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let dims = |k: &str| {
            (
                c[k][0].as_u64().unwrap() as usize,
                c[k][1].as_u64().unwrap() as usize,
            )
        };
        let (iw, ih) = dims("in");
        let (ow, oh) = dims("out");
        let src = Gray::new(
            iw,
            ih,
            std::fs::read(dir.join(format!("{name}.in.u8"))).unwrap(),
        );
        let want = std::fs::read(dir.join(format!("{name}.out.u8"))).unwrap();
        let got = match c["op"].as_str().unwrap() {
            "resize" => {
                let interp = match c["interp"].as_str().unwrap() {
                    "linear" => Interp::Linear,
                    _ => Interp::Area,
                };
                resize(&src, ow, oh, interp)
            }
            _ => {
                let border = match c["border"].as_str().unwrap() {
                    "replicate" => Border::Replicate,
                    _ => Border::Constant(0),
                };
                rotate(&src, c["degrees"].as_f64().unwrap(), border)
            }
        };
        let diff: Vec<u8> = got
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let n = diff.iter().filter(|&&d| d > 0).count();
        let mx = diff.iter().copied().max().unwrap_or(0);
        if c["interp"].as_str() == Some("linear") {
            worst_linear = worst_linear.max(mx);
        } else {
            worst = worst.max(mx);
        }
        eprintln!("{name:24} {ow}×{oh}: {n} px differ, max {mx}");
    }
    assert!(
        worst <= 1,
        "an area/warp case differs by {worst} gray levels"
    );
    assert!(
        worst_linear <= 3,
        "a linear case differs by {worst_linear} gray levels"
    );
}
