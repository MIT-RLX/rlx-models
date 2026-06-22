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

//! BioCLIP-2 parity tests.
//!
//! `reference_parity_cpu` compares rlx CPU image/text features against an
//! OpenCLIP reference dumped by `scripts/bioclip2_dump_reference.py`.
//! Gated on `RLX_BIOCLIP2_FIXTURE` (reference.json) + a model directory
//! (`RLX_BIOCLIP2_MODEL`, default `weights/bioclip-2`).
//!
//! The cross-backend tests reuse the same model + reference but run the
//! towers on a GPU backend; they are feature-gated and skip when the
//! backend is unavailable at runtime.

use rlx_bioclip2::BioClip2Runner;
use rlx_runtime::Device;
use std::path::PathBuf;

struct Reference {
    pixel_values: Vec<f32>,
    image_features: Vec<f32>,
    token_ids: Vec<Vec<u32>>,
    text_features: Vec<Vec<f32>>,
    logits_per_image: Vec<f32>,
}

fn fixture_paths() -> Option<(PathBuf, PathBuf)> {
    let fixture = std::env::var_os("RLX_BIOCLIP2_FIXTURE")?;
    let model = std::env::var_os("RLX_BIOCLIP2_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/bioclip-2"));
    let ref_json = PathBuf::from(fixture).join("reference.json");
    if !ref_json.exists() || !model.join("open_clip_model.safetensors").exists() {
        return None;
    }
    Some((ref_json, model))
}

fn load_reference(path: &PathBuf) -> Reference {
    let data = std::fs::read_to_string(path).expect("read reference.json");
    let v: serde_json::Value = serde_json::from_str(&data).expect("parse reference.json");
    let farr = |x: &serde_json::Value| -> Vec<f32> {
        x.as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_f64().unwrap() as f32)
            .collect()
    };
    Reference {
        pixel_values: farr(&v["pixel_values"]),
        image_features: farr(&v["image_features"]),
        token_ids: v["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect(),
        text_features: v["text_features"]
            .as_array()
            .unwrap()
            .iter()
            .map(farr)
            .collect(),
        logits_per_image: farr(&v["logits_per_image"]),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn run_parity_on(device: Device, label: &str) {
    let Some((ref_json, model)) = fixture_paths() else {
        eprintln!("[bioclip2 parity] RLX_BIOCLIP2_FIXTURE / model dir unset — skip ({label})");
        return;
    };
    let reference = load_reference(&ref_json);

    let mut runner = BioClip2Runner::builder()
        .model_dir(&model)
        .device(device)
        .build()
        .expect("build runner");

    // --- Image features ---
    let img = runner
        .encode_image_nchw(&reference.pixel_values)
        .expect("encode image");
    let img_dmax = max_abs(&img, &reference.image_features);
    let img_cos = cosine(&img, &reference.image_features);
    eprintln!("[bioclip2 parity/{label}] image: max|Δ|={img_dmax:.3e} cos={img_cos:.6}");
    assert!(img_dmax < 1e-3, "image max|Δ|={img_dmax:.3e} >= 1e-3");
    assert!(img_cos > 0.999, "image cosine {img_cos:.6} <= 0.999");

    // --- Text features ---
    for (i, ids) in reference.token_ids.iter().enumerate() {
        let txt = runner.encode_text_ids(ids).expect("encode text");
        let dmax = max_abs(&txt, &reference.text_features[i]);
        let cos = cosine(&txt, &reference.text_features[i]);
        eprintln!("[bioclip2 parity/{label}] text[{i}]: max|Δ|={dmax:.3e} cos={cos:.6}");
        assert!(dmax < 1e-3, "text[{i}] max|Δ|={dmax:.3e} >= 1e-3");
        assert!(cos > 0.999, "text[{i}] cosine {cos:.6} <= 0.999");
    }

    // --- Zero-shot logits (full pipeline) ---
    let scale = runner.logit_scale().exp();
    let img_n = l2(&img);
    let logits: Vec<f32> = reference
        .text_features
        .iter()
        .map(|t| {
            let txt = l2(t);
            scale * img_n.iter().zip(&txt).map(|(a, b)| a * b).sum::<f32>()
        })
        .collect();
    let logit_dmax = max_abs(&logits, &reference.logits_per_image);
    eprintln!("[bioclip2 parity/{label}] logits: max|Δ|={logit_dmax:.3e}");
    assert!(logit_dmax < 5e-2, "logits max|Δ|={logit_dmax:.3e} >= 5e-2");
}

fn l2(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

#[test]
fn reference_parity_cpu() {
    run_parity_on(Device::Cpu, "cpu");
}

/// Batched encoding (`.batch(n)`) must match the open_clip reference
/// exactly — same numerics as batch=1, just more items per graph run.
#[test]
fn batched_parity_cpu() {
    let Some((ref_json, model)) = fixture_paths() else {
        eprintln!("[bioclip2 batched] fixture/model unset — skip");
        return;
    };
    let reference = load_reference(&ref_json);
    let n = reference.token_ids.len();

    let mut runner = BioClip2Runner::builder()
        .model_dir(&model)
        .device(Device::Cpu)
        .batch(n)
        .build()
        .expect("build batched runner");
    assert_eq!(runner.batch(), n);

    // All texts in a single batched graph run.
    let id_refs: Vec<&[u32]> = reference.token_ids.iter().map(|v| v.as_slice()).collect();
    let txt = runner
        .encode_texts_ids(&id_refs)
        .expect("batched text encode");
    for (i, feat) in txt.iter().enumerate() {
        let dmax = max_abs(feat, &reference.text_features[i]);
        eprintln!("[bioclip2 batched/cpu] text[{i}]: max|Δ|={dmax:.3e}");
        assert!(dmax < 1e-3, "batched text[{i}] max|Δ|={dmax:.3e}");
    }

    // Image through the batched path (padded chunk) must match too.
    let img = runner
        .encode_images_nchw(&[reference.pixel_values.as_slice()])
        .expect("batched image encode")
        .remove(0);
    let dmax = max_abs(&img, &reference.image_features);
    eprintln!("[bioclip2 batched/cpu] image: max|Δ|={dmax:.3e}");
    assert!(dmax < 1e-3, "batched image max|Δ|={dmax:.3e}");
}

/// Verify the pure-Rust PIL-equivalent preprocessing (shared
/// `rlx_core::image_preprocess`) against open_clip's actual
/// post-preprocess `pixel_values`, by reproducing the dumper's
/// deterministic synthetic image and comparing the normalized tensors.
#[test]
fn preprocess_parity_cpu() {
    let Some((ref_json, _model)) = fixture_paths() else {
        eprintln!("[bioclip2 preprocess] fixture unset — skip");
        return;
    };
    let reference = load_reference(&ref_json);

    // Same synthetic image as scripts/bioclip2_dump_reference.py.
    let size = 256usize;
    let mut rgb = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let i = (y * size + x) * 3;
            rgb[i] = (x * 255 / size) as u8;
            rgb[i + 1] = (y * 255 / size) as u8;
            rgb[i + 2] = (((x / 16 + y / 16) % 2) * 200 + 30) as u8;
        }
    }
    let nchw = rlx_bioclip2::clip_normalize_nchw(&rgb, size, size, 224);

    let dmax = max_abs(&nchw, &reference.pixel_values);
    let n = nchw.len() as f32;
    let mean: f32 = nchw
        .iter()
        .zip(&reference.pixel_values)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / n;
    eprintln!("[bioclip2 preprocess/cpu] pixels: max|Δ|={dmax:.3e} mean|Δ|={mean:.3e}");
    // PIL's 8-bit resampler rounding can differ by up to ~1 LSB on a few
    // pixels (≈ 1/255 / std ≈ 1.5e-2 normalized); the mean must be tiny.
    assert!(dmax < 3e-2, "preprocess max|Δ|={dmax:.3e} too large");
    assert!(mean < 1e-3, "preprocess mean|Δ|={mean:.3e} too large");
}

#[cfg(feature = "metal")]
#[test]
fn reference_parity_metal() {
    if !rlx_runtime::device_ext::is_available(Device::Metal) {
        eprintln!("[bioclip2 parity] Metal unavailable — skip");
        return;
    }
    run_parity_on(Device::Metal, "metal");
}

#[cfg(feature = "mlx")]
#[test]
fn reference_parity_mlx() {
    if !rlx_runtime::device_ext::is_available(Device::Mlx) {
        eprintln!("[bioclip2 parity] MLX unavailable — skip");
        return;
    }
    run_parity_on(Device::Mlx, "mlx");
}

/// Regression test for the MLX `Op::Gather` fix (rlx-mlx
/// `mlx_indices_i64` must convert indices lazily, not via a host
/// `to_bytes()` round-trip, or compiled graphs with a dynamic-index
/// gather panic with "Attempting to eval an array during function
/// transformations"). BioCLIP-2 itself embeds on host, but this guards
/// the underlying backend fix for every gather-using model.
#[cfg(feature = "mlx")]
#[test]
fn mlx_compiled_gather_matches_cpu() {
    use rlx_flow::{CompileProfile, MapWeights, ModelFlow};
    use rlx_ir::hir::HirMut;
    use rlx_ir::{DType, HirGraphExt, Shape};

    if !rlx_runtime::device_ext::is_available(Device::Mlx) {
        eprintln!("[bioclip2 gather] MLX unavailable — skip");
        return;
    }

    let (vocab, d, n) = (5usize, 3usize, 4usize);
    let table: Vec<f32> = (0..vocab * d).map(|i| i as f32).collect();

    let build = |device: Device| {
        let table = table.clone();
        let flow = ModelFlow::new("gather_smoke")
            .with_profile(CompileProfile::encoder())
            .input("ids", Shape::new(&[1, n], DType::F32))
            .plugin_named("embed", move |emit, _prev| {
                let ids = emit.flow_input("ids")?;
                let w =
                    emit.synth_param("embed", table.clone(), Shape::new(&[vocab, d], DType::F32));
                let mut gb = HirMut::new(emit.hir());
                let out = gb.gather_(w, ids.hir_id(), 0);
                Ok(Some(emit.wrap(out, Shape::new(&[1, n, d], DType::F32))))
            })
            .output("y");
        let built = flow
            .build(&mut MapWeights::default())
            .expect("build gather flow");
        rlx_core::flow_util::compile_built(built, device).expect("compile gather flow")
    };

    let ids = [1.0f32, 3.0, 0.0, 4.0];
    let cpu = build(Device::Cpu)
        .run(&[("ids", &ids)])
        .into_iter()
        .next()
        .unwrap();
    let mlx = build(Device::Mlx)
        .run(&[("ids", &ids)])
        .into_iter()
        .next()
        .unwrap();

    let expect: Vec<f32> = ids
        .iter()
        .flat_map(|&t| {
            let r = t as usize * d;
            table[r..r + d].to_vec()
        })
        .collect();
    eprintln!("[bioclip2 gather] cpu={cpu:?} mlx={mlx:?}");
    assert_eq!(cpu, expect, "cpu gather wrong");
    let dmax = max_abs(&mlx, &expect);
    assert!(dmax < 1e-5, "mlx compiled gather max|Δ|={dmax:.3e}");
}

#[cfg(feature = "gpu")]
#[test]
fn reference_parity_gpu() {
    if !rlx_runtime::device_ext::is_available(Device::Gpu) {
        eprintln!("[bioclip2 parity] wgpu unavailable — skip");
        return;
    }
    run_parity_on(Device::Gpu, "gpu");
}

#[cfg(feature = "cuda")]
#[test]
fn reference_parity_cuda() {
    if !rlx_runtime::device_ext::is_available(Device::Cuda) {
        eprintln!("[bioclip2 parity] CUDA unavailable — skip");
        return;
    }
    run_parity_on(Device::Cuda, "cuda");
}

#[cfg(feature = "rocm")]
#[test]
fn reference_parity_rocm() {
    if !rlx_runtime::device_ext::is_available(Device::Rocm) {
        eprintln!("[bioclip2 parity] ROCm unavailable — skip");
        return;
    }
    run_parity_on(Device::Rocm, "rocm");
}

#[cfg(feature = "vulkan")]
#[test]
fn reference_parity_vulkan() {
    if !rlx_runtime::device_ext::is_available(Device::Vulkan) {
        eprintln!("[bioclip2 parity] Vulkan unavailable — skip");
        return;
    }
    run_parity_on(Device::Vulkan, "vulkan");
}
