// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A d1 package answers what `D1::system_one` does on every text case of
//! `tests/fixtures/d1_cases.json` (nouls, choices, many options, a JSON
//! state, Unicode; one question and several) — on the bf16 checkpoint
//! where it is on disk (`weights/lm/d1-3b`, or `RLX_D1_DIR`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_d1::export::ExportOptions;
use rlx_d1::{D1, Json, RgbImage};
use rlx_runtime::Device;
use serde_json::json;

const CASES: &str = include_str!("fixtures/d1_cases.json");

fn dir() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_D1_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/lm/d1-3b"));
    p.join("model.safetensors").exists().then_some(p)
}

/// The largest difference between two decisions' numbers; an error where
/// anything else differs.
fn compare(a: &Json, b: &Json, at: &str, worst: &mut f64) -> Result<(), String> {
    let num = |j: &Json| match j {
        Json::Float(f) => Some(*f),
        Json::Int(i) => Some(*i as f64),
        Json::UInt(u) => Some(*u as f64),
        _ => None,
    };
    match (a, b) {
        (Json::Object(x), Json::Object(y)) => {
            if x.len() != y.len() {
                return Err(format!("{at}: {} keys vs {}", x.len(), y.len()));
            }
            for ((ka, va), (kb, vb)) in x.iter().zip(y) {
                if ka != kb {
                    return Err(format!("{at}: key {ka} vs {kb}"));
                }
                compare(va, vb, &format!("{at}.{ka}"), worst)?;
            }
            Ok(())
        }
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => {
                *worst = worst.max((x - y).abs());
                Ok(())
            }
            _ if a == b => Ok(()),
            _ => Err(format!("{at}: {a:?} vs {b:?}")),
        },
    }
}

#[test]
fn a_package_decides_what_d1_does() {
    let Some(dir) = dir() else {
        eprintln!("no d1 checkpoint; skipped");
        return;
    };
    let cases: Vec<Json> = Json::parse(CASES)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c.get("images").is_none())
        .cloned()
        .collect();

    // The runner first, then dropped: the two together hold the checkpoint twice.
    let mut wants = Vec::new();
    {
        let mut d = D1::open(&dir, Device::Cpu).unwrap();
        for c in &cases {
            let got = d
                .system_one(c.get("state").unwrap(), c.get("questions").unwrap(), &[])
                .unwrap();
            wants.push(got);
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("d1.rlxp");
    let opts = ExportOptions {
        uppers: vec![512],
        ..ExportOptions::default()
    };
    let report = rlx_d1::export::export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (c, want) in cases.iter().zip(&wants) {
        let name = c.get("name").and_then(Json::as_str).unwrap_or("?");
        let request = Json::Object(vec![
            ("state".into(), c.get("state").unwrap().clone()),
            ("questions".into(), c.get("questions").unwrap().clone()),
        ]);
        let outs = m
            .run(&[], &json!({"request": request.to_python_json(None)}))
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let got = Json::parse(&String::from_utf8(outs[0].data.clone()).unwrap()).unwrap();
        let mut worst = 0f64;
        compare(&got, want, name, &mut worst).unwrap();
        eprintln!(
            "{name}: {} — numbers within {worst:.2e}",
            want.get("usage").unwrap().to_python_json(None)
        );
        assert!(worst < 2e-3, "{name}: off by {worst}");
    }
}

/// Standard base64, padded.
fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i <= c.len() {
                A[((n >> shift) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

/// The image cases of `d1_cases.json` — one picture under one question and
/// several, a tiled 1400 × 900 picture, a small one — with the decider's
/// fixture pictures (the cats one standing in for `cats.jpg`), given to the
/// package as `data:` URLs.
#[test]
fn a_package_decides_on_pictures_what_d1_does() {
    let Some(dir) = dir() else {
        eprintln!("no d1 checkpoint; skipped");
        return;
    };
    let pics = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-pplx-decider/tests/fixtures/img");
    let file = |name: &str| {
        pics.join(match name {
            "cats.jpg" => "cats_320x240.jpg",
            other => other,
        })
    };
    let cases: Vec<Json> = Json::parse(CASES)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c.get("images").is_some())
        .cloned()
        .collect();
    let images_of = |c: &Json| -> Vec<std::path::PathBuf> {
        c.get("images")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|n| file(n.as_str().unwrap()))
            .collect()
    };
    let mut wants = Vec::new();
    {
        let mut d = D1::open(&dir, Device::Cpu).unwrap();
        for c in &cases {
            let imgs: Vec<RgbImage> = images_of(c)
                .iter()
                .map(|p| RgbImage::open(p).unwrap())
                .collect();
            wants.push(
                d.system_one(c.get("state").unwrap(), c.get("questions").unwrap(), &imgs)
                    .unwrap(),
            );
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("d1.rlxp");
    let opts = ExportOptions {
        uppers: vec![2048],
        ..ExportOptions::default()
    };
    let report = rlx_d1::export::export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (c, want) in cases.iter().zip(&wants) {
        let name = c.get("name").and_then(Json::as_str).unwrap_or("?");
        let urls = images_of(c)
            .iter()
            .map(|p| {
                let kind = if p.extension().is_some_and(|e| e == "png") {
                    "png"
                } else {
                    "jpeg"
                };
                Json::Str(format!(
                    "data:image/{kind};base64,{}",
                    base64(&std::fs::read(p).unwrap())
                ))
            })
            .collect();
        let request = Json::Object(vec![
            ("state".into(), c.get("state").unwrap().clone()),
            ("questions".into(), c.get("questions").unwrap().clone()),
            ("images".into(), Json::Array(urls)),
        ]);
        let outs = m
            .run(&[], &json!({"request": request.to_python_json(None)}))
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let got = Json::parse(&String::from_utf8(outs[0].data.clone()).unwrap()).unwrap();
        let mut worst = 0f64;
        compare(&got, want, name, &mut worst).unwrap();
        eprintln!(
            "{name}: {} — numbers within {worst:.2e}",
            want.get("usage").unwrap().to_python_json(None)
        );
        assert!(worst < 2e-3, "{name}: off by {worst}");
    }
}
