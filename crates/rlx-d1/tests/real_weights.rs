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

//! Parity with LiquidAI's own d1 code on the real checkpoint.
//!
//! Env-gated; skipped (with a note) when unset:
//!
//! * `RLX_D1_DIR` — the `LiquidAI/d1-3B` directory (`just fetch-d1`);
//! * `RLX_D1_IMAGES` — the case images (`just d1-test-images`); without it
//!   the picture cases are skipped;
//! * `RLX_D1_REF_DIR` — `scripts/d1_reference.py --out` (the per-image
//!   processor and feature dumps) for the stage-by-stage vision test. With
//!   it, pictures are taken as Pillow decoded them: the `image` crate's JPEG
//!   decoder differs from libjpeg-turbo on ~1% of bytes;
//! * `RLX_D1_DEVICE` — backend for the forward passes (default `cpu`).
//!
//! The probabilities are compared with `fixtures/d1_reference.json`, the
//! reference's fp32 CPU output. Run with `--release`: the backbone is 3B.

use std::path::PathBuf;

use rlx_d1::vision::{self, ProcessorConfig, RgbImage, VisionEncoder};
use rlx_d1::{D1, Json};
use rlx_runtime::Device;

fn env_path(key: &str) -> Option<PathBuf> {
    rlx_ir::env::var(key)
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// `RLX_D1_WEIGHT_STORAGE` (`auto` by default): how dense weights are stored.
fn storage() -> rlx_d1::WeightStorage {
    rlx_ir::env::var("RLX_D1_WEIGHT_STORAGE")
        .map(|s| s.parse().expect("RLX_D1_WEIGHT_STORAGE"))
        .unwrap_or_default()
}

fn device() -> Device {
    rlx_ir::env::var("RLX_D1_DEVICE")
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_D1_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fixture(name: &str) -> Json {
    Json::parse(match name {
        "cases" => include_str!("fixtures/d1_cases.json"),
        _ => include_str!("fixtures/d1_reference.json"),
    })
    .expect("fixture parses")
}

/// A case's pictures. With `RLX_D1_REF_DIR`, a single picture comes from the
/// reference dump (Pillow's decode, already capped — capping is idempotent),
/// so a JPEG decoder difference does not count against the port.
fn images_of(case: &Json, dir: Option<&PathBuf>) -> Option<Vec<RgbImage>> {
    let names = case.get("images").and_then(Json::as_array).unwrap_or(&[]);
    if names.is_empty() {
        return Some(Vec::new());
    }
    let name = case.get("name").unwrap().as_str().unwrap();
    if names.len() == 1
        && let Some(refs) = env_path("RLX_D1_REF_DIR")
        && let Ok(bytes) = std::fs::read(refs.join(format!("vision_{name}.safetensors")))
    {
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let size: Vec<usize> = st
            .tensor("capped_size")
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as usize)
            .collect();
        let rgb = st.tensor("capped_rgb").unwrap().data().to_vec();
        return Some(vec![RgbImage::new(size[0], size[1], rgb).unwrap()]);
    }
    let dir = dir?;
    Some(
        names
            .iter()
            .map(|n| RgbImage::open(&dir.join(n.as_str().unwrap())).expect("image"))
            .collect(),
    )
}

fn f(v: &Json) -> f64 {
    match v {
        Json::Float(x) => *x,
        Json::Int(i) => *i as f64,
        Json::UInt(u) => *u as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// Every probability (and `score`) in two `answers` objects, paired.
fn answer_values(a: &Json) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for (q, ans) in a.as_object().unwrap() {
        for (k, v) in ans.as_object().unwrap() {
            match k.as_str() {
                "noul" | "score" | "confidence" => out.push((format!("{q}.{k}"), f(v))),
                "probabilities" => {
                    for (o, p) in v.as_object().unwrap() {
                        out.push((format!("{q}.p[{o}]"), f(p)));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

#[test]
fn end_to_end_parity() {
    let Some(dir) = env_path("RLX_D1_DIR") else {
        eprintln!("skip: set RLX_D1_DIR to the LiquidAI/d1-3B directory");
        return;
    };
    let images_dir = env_path("RLX_D1_IMAGES");
    let opts = rlx_d1::OpenOptions {
        weight_storage: storage(),
        ..Default::default()
    };
    let mut d1 = D1::open_with(&dir, device(), &opts).expect("open d1");
    eprintln!("weight storage: {}", storage());
    let cases = fixture("cases");
    let reference = fixture("reference");
    let mut worst = 0f64;
    let mut ran = 0;
    for (case, refc) in cases
        .as_array()
        .unwrap()
        .iter()
        .zip(reference.as_array().unwrap())
    {
        let name = case.get("name").unwrap().as_str().unwrap();
        let Some(images) = images_of(case, images_dir.as_ref()) else {
            eprintln!("skip {name}: set RLX_D1_IMAGES for the picture cases");
            continue;
        };
        // Token ids of every prompt, joint-tokenized as the reference rows are.
        for row in refc.get("rows").unwrap().as_array().unwrap() {
            let text = row.get("text").unwrap().as_str().unwrap();
            let want: Vec<u32> = row
                .get("ids")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f(v) as u32)
                .collect();
            let got = d1.prompt_ids(text, &images).expect("prompt ids");
            assert_eq!(got, want, "{name}: token ids");
        }

        let t = std::time::Instant::now();
        let out = d1
            .system_one(
                case.get("state").unwrap(),
                case.get("questions").unwrap(),
                &images,
            )
            .expect("system_one");
        let official = refc.get("system_one").unwrap();
        assert_eq!(
            out.get("usage").unwrap().to_python_json(None),
            official.get("usage").unwrap().to_python_json(None),
            "{name}: usage"
        );
        let got = answer_values(out.get("answers").unwrap());
        let want = answer_values(official.get("answers").unwrap());
        assert_eq!(got.len(), want.len());
        let mut case_worst = 0f64;
        for ((k, g), (k2, w)) in got.iter().zip(&want) {
            assert_eq!(k, k2);
            // `score` is an expectation over up to 10 levels: compare it per level.
            let d = if k.ends_with(".score") {
                (g - w).abs() / 9.0
            } else {
                (g - w).abs()
            };
            case_worst = case_worst.max(d);
        }
        // Same picks.
        for ((q, a), (_, b)) in out
            .get("answers")
            .unwrap()
            .as_object()
            .unwrap()
            .iter()
            .zip(official.get("answers").unwrap().as_object().unwrap())
        {
            if let (Some(x), Some(y)) = (a.get("choice"), b.get("choice")) {
                assert_eq!(x, y, "{name}/{q}: pick");
            }
        }
        eprintln!(
            "{name:18} max |dp| = {case_worst:.2e}  ({:.2}s)",
            t.elapsed().as_secs_f64()
        );
        worst = worst.max(case_worst);
        ran += 1;
    }
    eprintln!("worst max |dp| over {ran} cases = {worst:.2e}");
    assert!(ran > 0);
    assert!(
        worst < 2e-4,
        "probabilities differ from the reference by {worst:.2e}"
    );
}

/// The vision path stage by stage against the reference dumps: the capped
/// picture, every tile's normalized patches, and the projected features.
#[test]
fn vision_stages_match_the_reference() {
    let (Some(dir), Some(images), Some(refs)) = (
        env_path("RLX_D1_DIR"),
        env_path("RLX_D1_IMAGES"),
        env_path("RLX_D1_REF_DIR"),
    ) else {
        eprintln!("skip: needs RLX_D1_DIR, RLX_D1_IMAGES and RLX_D1_REF_DIR");
        return;
    };
    let cfg = ProcessorConfig::from_dir(&dir).unwrap();
    let mut enc = VisionEncoder::load(&dir, device(), storage()).expect("vision tower");
    for case in fixture("cases").as_array().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let Some(first) = case
            .get("images")
            .and_then(Json::as_array)
            .and_then(|a| a.first())
        else {
            continue;
        };
        let dump = refs.join(format!("vision_{name}.safetensors"));
        let bytes = std::fs::read(&dump).expect("reference dump");
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let tensor_f32 = |k: &str| -> Vec<f32> {
            st.tensor(k)
                .unwrap()
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let tensor_i64 = |k: &str| -> Vec<i64> {
            st.tensor(k)
                .unwrap()
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect()
        };

        let img = RgbImage::open(&images.join(first.as_str().unwrap())).unwrap();
        let capped = vision::cap_pixels(&img, vision::VISION_MAX_PIXELS);
        let size = tensor_i64("capped_size");
        assert_eq!(
            (capped.width as i64, capped.height as i64),
            (size[0], size[1]),
            "{name}: capped size"
        );
        let want_rgb = st.tensor("capped_rgb").unwrap().data().to_vec();
        let rgb_diff = capped
            .data
            .iter()
            .zip(&want_rgb)
            .filter(|(a, b)| a != b)
            .count();
        // A JPEG decodes differently in the `image` crate than in Pillow's
        // libjpeg-turbo (IDCT, chroma upsampling), so from here on the
        // reference's own decoded pixels are the input: what is measured is
        // the port, not the decoder. PNG sources must match exactly.
        let is_jpeg = first.as_str().unwrap().ends_with(".jpg");
        if !is_jpeg {
            assert_eq!(rgb_diff, 0, "{name}: capped picture");
        }
        let capped = RgbImage::new(capped.width, capped.height, want_rgb).unwrap();

        let p = vision::process_image(&capped, &cfg).unwrap();
        let shapes = tensor_i64("spatial_shapes");
        let pv = tensor_f32("pixel_values");
        let max_patches = cfg.max_num_patches();
        let pd = pv.len() / (p.tiles.len() * max_patches);
        assert_eq!(shapes.len(), 2 * p.tiles.len(), "{name}: tile count");
        let mut pv_diff = 0f32;
        for (i, tile) in p.tiles.iter().enumerate() {
            assert_eq!(
                (tile.ph as i64, tile.pw as i64),
                (shapes[2 * i], shapes[2 * i + 1]),
                "{name}: tile {i} grid"
            );
            let want = &pv[i * max_patches * pd..][..tile.pixel_values.len()];
            for (a, b) in tile.pixel_values.iter().zip(want) {
                pv_diff = pv_diff.max((a - b).abs());
            }
        }

        let got = enc.encode(&p).unwrap();
        let want = tensor_f32("image_features");
        assert_eq!(got.len(), want.len(), "{name}: feature count");
        let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
        for (a, b) in got.iter().zip(&want) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
            maxd = maxd.max((a - b).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!(
            "{name:18} decoded/capped bytes differing {rgb_diff}{}, pixel max |d| {pv_diff:.2e}, \
             features cos {cos:.7} max |d| {maxd:.2e}",
            if is_jpeg {
                " (JPEG decoder; not compared)"
            } else {
                ""
            }
        );
        assert!(pv_diff < 1e-6, "{name}: pixel values");
        assert!(cos > 0.99999, "{name}: features cos {cos}");
    }
}
