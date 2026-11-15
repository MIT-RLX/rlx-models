// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! The RLX graph against ONNX Runtime running the same YuNet export, with Pollen's
//! `face_detector.py` decode: every head output, and the faces. Needs `YUNET_MODEL` (the .onnx,
//! converted here) and `YUNET_FIXTURES` (a dir of `NAME.bgr` frames, `NAME.heads` reference
//! outputs in the export's output order, and `NAME.json` reference faces); skipped without them.
//! `YUNET_DEVICE=metal` (feature `metal`) runs the graph on the GPU instead.

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_yunet::{OUTPUTS, YuNet};

#[test]
fn matches_onnxruntime() {
    let (Some(model), Some(dir)) = (
        std::env::var_os("YUNET_MODEL"),
        std::env::var_os("YUNET_FIXTURES"),
    ) else {
        eprintln!("YUNET_MODEL / YUNET_FIXTURES unset: skipped");
        return;
    };
    let weights = rlx_yunet::onnx::weights_from_onnx(&std::fs::read(model).unwrap()).unwrap();
    let device = match std::env::var("YUNET_DEVICE").as_deref() {
        Ok("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let mut net = YuNet::new(weights, device);
    let dir = PathBuf::from(dir);
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        let (w, h) = (
            meta["width"].as_u64().unwrap() as usize,
            meta["height"].as_u64().unwrap() as usize,
        );
        let bgr = std::fs::read(p.with_extension("bgr")).unwrap();
        let names: Vec<&str> = meta["heads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, OUTPUTS, "reference output order");

        // Heads, against the reference.
        let (pw, ph) = (w.div_ceil(32) * 32, h.div_ceil(32) * 32);
        let mut x = vec![0.0f32; 3 * ph * pw];
        for y in 0..h {
            for i in 0..w {
                for c in 0..3 {
                    x[(c * ph + y) * pw + i] = f32::from(bgr[(y * w + i) * 3 + c]);
                }
            }
        }
        let got: Vec<f32> = net.run(&x, ph, pw).unwrap().concat();
        let want: Vec<f32> = std::fs::read(p.with_extension("heads"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(got.len(), want.len(), "{}", p.display());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{}: heads differ by {worst}", p.display());

        // Faces.
        net.score_threshold = meta["threshold"].as_f64().unwrap() as f32;
        let faces = net.detect_bgr(&bgr, w, h, 3 * w).unwrap();
        let refs = meta["faces"].as_array().unwrap();
        assert_eq!(faces.len(), refs.len(), "{}: face count", p.display());
        for (f, r) in faces.iter().zip(refs) {
            let close = |a: &[f32], b: &serde_json::Value| {
                a.iter()
                    .zip(b.as_array().unwrap())
                    .all(|(x, y)| (x - y.as_f64().unwrap() as f32).abs() < 1e-2)
            };
            assert!(
                close(&f.bbox, &r["bbox"]),
                "{}: bbox {:?} vs {}",
                p.display(),
                f.bbox,
                r["bbox"]
            );
            assert!(
                close(&f.right_eye(), &r["right_eye"])
                    && close(&f.left_eye(), &r["left_eye"])
                    && close(&f.nose(), &r["nose"]),
                "{}: landmarks",
                p.display()
            );
        }
        eprintln!(
            "{}: {} faces, heads within {worst:.1e}",
            p.file_stem().unwrap().to_string_lossy(),
            faces.len()
        );
        n += 1;
    }
    assert!(n > 0, "no fixtures in {}", dir.display());
}
