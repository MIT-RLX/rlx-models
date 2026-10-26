// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Subject lifting exported as a package gives what `SubjectLift` does: the
//! model macOS ships (skipped where absent), images of two sizes, the alpha
//! (`Lifted::alpha`) and the gating confidence.

#![cfg(feature = "export")]

use std::path::Path;

use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use rlx_subject_lift::{LiftOptions, SYSTEM_MODEL, SubjectLift};
use serde_json::json;

fn image(w: usize, h: usize) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (y, x) = ((p / w) as f32 / h as f32, (p % w) as f32 / w as f32);
            let inside = (x - 0.5).powi(2) + (y - 0.45).powi(2) < 0.06;
            let base = if inside { [230, 120, 40][c] } else { 245 };
            (base - ((p * 7 + c * 31) % 13) as i32) as u8
        })
        .collect()
}

#[test]
fn exported_subject_lifting_gives_the_runners_alpha() {
    if !Path::new(SYSTEM_MODEL).is_file() {
        eprintln!("skipped: no {SYSTEM_MODEL}");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("subject-lift.rlxp");
    let report =
        rlx_subject_lift::export::export_rlxp(Path::new(""), &out, &Default::default()).unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut lift = SubjectLift::open(None, Device::Cpu, LiftOptions::default()).unwrap();
    for (w, h) in [(300, 200), (640, 640)] {
        let px = image(w, h);
        let img = image::RgbImage::from_raw(w as u32, h as u32, px.clone()).unwrap();
        let lifted = lift.run(&img).unwrap();
        let want = lifted.alpha(w as u32, h as u32);
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], px);
        let outs = m.run(&[input], &json!({})).unwrap();
        assert_eq!(outs[0].shape, vec![h, w]);
        let got = outs[0].to_f32().unwrap();
        let gating = outs[1].to_f32().unwrap()[0];
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let mean = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / got.len() as f32;
        eprintln!(
            "{w}x{h}: alpha off by {worst} (mean {mean}); gating {gating} vs {}",
            lifted.gating
        );
        // The resize graph sums its taps in another order than the `image`
        // crate: a few pixels in a hundred thousand land a level apart
        // (none at 300 × 200), which moves the alpha near them.
        assert!(
            worst < 5e-3 && mean < 1e-5,
            "{w}x{h}: alpha off by {worst} (mean {mean})"
        );
        assert!((gating - lifted.gating).abs() < 1e-4, "{w}x{h}: gating");
    }
}
