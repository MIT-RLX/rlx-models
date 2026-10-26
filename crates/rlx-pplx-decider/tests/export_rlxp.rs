// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A pplx-decider package answers what `PplxDecider::decide` does: every
//! text request of `tests/fixtures/requests.json` (choices, nouls, scores,
//! several questions to a state), and one past the package's longest
//! bucket, split into question chunks as the runner splits it — on the
//! Q8_0 GGUF where it is on disk (`weights/lm/pplx-decider-flash-gguf`, or
//! `PPLX_DECIDER_GGUF`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_pplx_decider::export::ExportOptions;
use rlx_pplx_decider::{Json, OpenOptions, PplxDecider};
use rlx_runtime::Device;
use serde_json::json;

const REQUESTS: &str = include_str!("fixtures/requests.json");
const IMAGE_REQUESTS: &str = include_str!("fixtures/requests_images.json");
const LONGEST: usize = 1024;

fn gguf() -> Option<PathBuf> {
    let p = rlx_ir::env::var("PPLX_DECIDER_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/lm/pplx-decider-flash-gguf/decider-flash-v1.2-q8_0.gguf")
        });
    p.exists().then_some(p)
}

/// The largest difference between two decisions' numbers, or the first
/// place they differ otherwise.
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
        (Json::Array(x), Json::Array(y)) if x.len() == y.len() => {
            for (i, (va, vb)) in x.iter().zip(y).enumerate() {
                compare(va, vb, &format!("{at}[{i}]"), worst)?;
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
fn a_package_decides_what_the_decider_does() {
    let Some(path) = gguf() else {
        eprintln!("no pplx-decider GGUF; skipped");
        return;
    };
    let mut requests: Vec<Json> = Json::parse(REQUESTS)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r.get("images").is_none())
        .cloned()
        .collect();
    // A state long enough that its questions go in two sequences.
    let first = requests[0].clone();
    let long = "The ticket history follows. ".repeat(178);
    let Json::Object(mut fields) = first else {
        panic!("a request is an object")
    };
    fields[0].1 = Json::Str(long);
    requests.push(Json::Object(fields));

    // The runner first, then dropped: the two together would hold the
    // checkpoint twice.
    let opts = OpenOptions {
        max_length: Some(LONGEST),
        ..OpenOptions::default()
    };
    let mut wants = Vec::new();
    {
        let mut d = PplxDecider::open_with(&path, Device::Cpu, &opts).unwrap();
        for r in &requests {
            let req = d.request(r).unwrap();
            wants.push(d.decide(&req, None).unwrap());
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("decider.rlxp");
    let export = ExportOptions {
        uppers: vec![LONGEST],
        ..ExportOptions::default()
    };
    let report = rlx_pplx_decider::export::export_rlxp(&path, &out, &export).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (i, (r, want)) in requests.iter().zip(&wants).enumerate() {
        let outs = m
            .run(&[], &json!({"request": r.to_python_json(None)}))
            .unwrap_or_else(|e| panic!("request {i}: {e:#}"));
        let text = String::from_utf8(outs[0].data.clone()).unwrap();
        let got = Json::parse(&text).unwrap();
        let mut worst = 0f64;
        compare(&got, want, &format!("request {i}"), &mut worst).unwrap();
        eprintln!(
            "request {i}: {} — numbers within {worst:.2e}",
            want.get("usage").unwrap().to_python_json(None)
        );
        assert!(worst < 2e-3, "request {i}: off by {worst}");
        if i == requests.len() - 1 {
            let n = want.get("usage").unwrap().get("sequences").unwrap();
            assert_eq!(n, &Json::UInt(2), "the long request goes in two sequences");
        }
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

/// Every image request of `tests/fixtures/requests_images.json` (one image
/// and two, a tiled document past 1500 tokens), its pictures given as
/// `data:` URLs: the package's vision tower and M-RoPE positions against
/// the runner's.
#[test]
fn a_package_decides_on_images_what_the_decider_does() {
    let Some(path) = gguf() else {
        eprintln!("no pplx-decider GGUF; skipped");
        return;
    };
    if crate_mmproj(&path).is_none() {
        eprintln!("no mmproj beside the GGUF; skipped");
        return;
    }
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let requests: Vec<Json> = Json::parse(IMAGE_REQUESTS)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let Json::Object(fields) = r else {
                panic!("a request is an object")
            };
            Json::Object(
                fields
                    .iter()
                    .map(|(k, v)| {
                        if k != "images" {
                            return (k.clone(), v.clone());
                        }
                        let urls = v
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|p| {
                                let p = p.as_str().unwrap();
                                let bytes = std::fs::read(fixtures.join(p)).unwrap();
                                let kind = if p.ends_with(".png") { "png" } else { "jpeg" };
                                Json::Str(format!("data:image/{kind};base64,{}", base64(&bytes)))
                            })
                            .collect();
                        (k.clone(), Json::Array(urls))
                    })
                    .collect(),
            )
        })
        .collect();
    const LONGEST_IMAGES: usize = 2048;
    let opts = OpenOptions {
        max_length: Some(LONGEST_IMAGES),
        ..OpenOptions::default()
    };
    let mut wants = Vec::new();
    {
        let mut d = PplxDecider::open_with(&path, Device::Cpu, &opts).unwrap();
        for r in &requests {
            let req = d.request(r).unwrap();
            wants.push(d.decide(&req, None).unwrap());
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("decider.rlxp");
    let export = ExportOptions {
        uppers: vec![LONGEST_IMAGES],
        ..ExportOptions::default()
    };
    let report = rlx_pplx_decider::export::export_rlxp(&path, &out, &export).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (i, (r, want)) in requests.iter().zip(&wants).enumerate() {
        let outs = m
            .run(&[], &json!({"request": r.to_python_json(None)}))
            .unwrap_or_else(|e| panic!("image request {i}: {e:#}"));
        let got = Json::parse(&String::from_utf8(outs[0].data.clone()).unwrap()).unwrap();
        let mut worst = 0f64;
        compare(&got, want, &format!("image request {i}"), &mut worst).unwrap();
        eprintln!(
            "image request {i}: {} — numbers within {worst:.2e}",
            want.get("usage").unwrap().to_python_json(None)
        );
        assert!(worst < 2e-3, "image request {i}: off by {worst}");
    }
}

/// The `mmproj-*.gguf` the runner picks beside `gguf`.
fn crate_mmproj(gguf: &Path) -> Option<PathBuf> {
    std::fs::read_dir(gguf.parent()?)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
        })
}
