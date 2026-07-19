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

//! SigLIP 2 parity tests.
//!
//! `reference_parity_cpu` compares rlx CPU image/text embeddings against the
//! HuggingFace reference dumped by `scripts/siglip2_hf_dump.py`. Defaults to
//! `weights/siglip2-base-224` (+ `fixture/reference.json`); override with
//! `RLX_SIGLIP2_MODEL` / `RLX_SIGLIP2_FIXTURE`. Skips cleanly when absent.
//!
//! Cross-backend tests reuse the same model + reference on a GPU backend;
//! they are feature-gated and skip when the backend is unavailable.

use rlx_runtime::Device;
use rlx_siglip2::Siglip2Runner;
use std::path::PathBuf;

struct Reference {
    pixel_values: Vec<f32>,
    token_ids: Vec<Vec<u32>>,
    image_features: Vec<f32>,
    text_features: Vec<Vec<f32>>,
    logits_per_image: Vec<f32>,
    prompts: Vec<String>,
    spatial_shapes: Option<(usize, usize)>,
}

fn fixture_paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("RLX_SIGLIP2_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/siglip2-base-224"));
    let fixture = std::env::var_os("RLX_SIGLIP2_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| model.join("fixture"));
    let ref_json = fixture.join("reference.json");
    if !ref_json.exists() || !model.join("model.safetensors").exists() {
        return None;
    }
    Some((ref_json, model))
}

fn naflex_fixture_paths() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("RLX_SIGLIP2_NAFLEX_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/siglip2-base-naflex"));
    let fixture = std::env::var_os("RLX_SIGLIP2_NAFLEX_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| model.join("fixture"));
    let ref_json = fixture.join("reference.json");
    if !ref_json.exists() || !model.join("model.safetensors").exists() {
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
        image_features: farr(&v["image_features"]),
        text_features: v["text_features"]
            .as_array()
            .unwrap()
            .iter()
            .map(farr)
            .collect(),
        logits_per_image: farr(&v["logits_per_image"]),
        prompts: v["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect(),
        spatial_shapes: v.get("spatial_shapes").and_then(|s| s.as_array()).map(|a| {
            (
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            )
        }),
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

fn l2(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn run_parity_on(device: Device, label: &str) {
    let Some((ref_json, model)) = fixture_paths() else {
        eprintln!("[siglip2 parity] model/fixture unset — skip ({label})");
        return;
    };
    let reference = load_reference(&ref_json);

    let mut runner = Siglip2Runner::builder()
        .model_dir(&model)
        .device(device)
        .build()
        .expect("build runner");

    // --- Image embedding (from HF pixel_values, isolating the encoder) ---
    let img = runner
        .encode_image_nchw(&reference.pixel_values)
        .expect("encode image");
    let img_dmax = max_abs(&img, &reference.image_features);
    let img_cos = cosine(&img, &reference.image_features);
    eprintln!("[siglip2 parity/{label}] image: max|Δ|={img_dmax:.3e} cos={img_cos:.6}");
    // cos > 0.999 is the parity gate; the raw-magnitude bound is loose enough
    // to absorb CUDA's matmul accumulation order (its cos is still 1.000000).
    assert!(img_dmax < 2e-3, "image max|Δ|={img_dmax:.3e} >= 2e-3");
    assert!(img_cos > 0.999, "image cosine {img_cos:.6} <= 0.999");

    // --- Text embeddings ---
    for (i, ids) in reference.token_ids.iter().enumerate() {
        let txt = runner.encode_text_ids(ids).expect("encode text");
        let dmax = max_abs(&txt, &reference.text_features[i]);
        let cos = cosine(&txt, &reference.text_features[i]);
        eprintln!("[siglip2 parity/{label}] text[{i}]: max|Δ|={dmax:.3e} cos={cos:.6}");
        assert!(dmax < 2e-3, "text[{i}] max|Δ|={dmax:.3e} >= 2e-3");
        assert!(cos > 0.999, "text[{i}] cosine {cos:.6} <= 0.999");
    }

    // --- Zero-shot logits (full pipeline: scale·⟨î,t̂⟩ + bias) ---
    let scale = runner.logit_scale().exp();
    let bias = runner.logit_bias();
    let img_n = l2(&img);
    let logits: Vec<f32> = reference
        .text_features
        .iter()
        .map(|t| {
            let txt = l2(t);
            scale * img_n.iter().zip(&txt).map(|(a, b)| a * b).sum::<f32>() + bias
        })
        .collect();
    let logit_dmax = max_abs(&logits, &reference.logits_per_image);
    eprintln!("[siglip2 parity/{label}] logits: max|Δ|={logit_dmax:.3e}");
    assert!(logit_dmax < 5e-2, "logits max|Δ|={logit_dmax:.3e} >= 5e-2");
}

/// NaFlex parity: feed HF `pixel_values` + `spatial_shapes` into
/// `encode_naflex_patches` (isolating the encoder + host position-embed
/// resize + padding mask from image preprocessing).
fn run_naflex_parity_on(device: Device, label: &str) {
    let Some((ref_json, model)) = naflex_fixture_paths() else {
        eprintln!("[siglip2 naflex] model/fixture unset — skip ({label})");
        return;
    };
    let reference = load_reference(&ref_json);
    let (nph, npw) = reference
        .spatial_shapes
        .expect("naflex reference needs spatial_shapes");

    let mut runner = Siglip2Runner::builder()
        .model_dir(&model)
        .device(device)
        .build()
        .expect("build naflex runner");
    assert_eq!(runner.config().variant, rlx_siglip2::Variant::NaFlex);

    let img = runner
        .encode_naflex_patches(&reference.pixel_values, nph, npw)
        .expect("encode naflex patches");
    let img_dmax = max_abs(&img, &reference.image_features);
    let img_cos = cosine(&img, &reference.image_features);
    eprintln!("[siglip2 naflex/{label}] image: max|Δ|={img_dmax:.3e} cos={img_cos:.6}");
    assert!(img_cos > 0.999, "naflex image cosine {img_cos:.6} <= 0.999");
    assert!(
        img_dmax < 2e-3,
        "naflex image max|Δ|={img_dmax:.3e} >= 2e-3"
    );

    for (i, ids) in reference.token_ids.iter().enumerate() {
        let txt = runner.encode_text_ids(ids).expect("encode text");
        let cos = cosine(&txt, &reference.text_features[i]);
        assert!(cos > 0.999, "naflex text[{i}] cosine {cos:.6} <= 0.999");
    }

    let scale = runner.logit_scale().exp();
    let bias = runner.logit_bias();
    let img_n = l2(&img);
    let logits: Vec<f32> = reference
        .text_features
        .iter()
        .map(|t| {
            let txt = l2(t);
            scale * img_n.iter().zip(&txt).map(|(a, b)| a * b).sum::<f32>() + bias
        })
        .collect();
    let logit_dmax = max_abs(&logits, &reference.logits_per_image);
    eprintln!("[siglip2 naflex/{label}] logits: max|Δ|={logit_dmax:.3e}");
    assert!(
        logit_dmax < 1e-1,
        "naflex logits max|Δ|={logit_dmax:.3e} >= 1e-1"
    );
}

#[test]
fn reference_parity_cpu() {
    run_parity_on(Device::Cpu, "cpu");
}

#[test]
fn naflex_parity_cpu() {
    run_naflex_parity_on(Device::Cpu, "cpu");
}

/// The Gemma tokenizer (fast, via `tokenizer.json`) must reproduce the HF
/// processor token ids exactly (`<eos>` appended, right-`<pad>` to 64).
#[test]
fn tokenizer_parity_cpu() {
    let Some((ref_json, model)) = fixture_paths() else {
        eprintln!("[siglip2 tokenizer] fixture unset — skip");
        return;
    };
    let reference = load_reference(&ref_json);
    let tk = rlx_siglip2::SiglipTokenizer::from_path(&model, 64).expect("tokenizer");
    for (i, prompt) in reference.prompts.iter().enumerate() {
        let ids = tk.encode(prompt).expect("encode");
        assert_eq!(
            ids, reference.token_ids[i],
            "tokenizer mismatch for {prompt:?}"
        );
    }
    eprintln!(
        "[siglip2 tokenizer/cpu] {} prompts match",
        reference.prompts.len()
    );
}

/// Pure-Rust PIL-faithful preprocessing vs HF `pixel_values` (slow processor).
#[test]
fn preprocess_parity_cpu() {
    let Some((ref_json, _model)) = fixture_paths() else {
        eprintln!("[siglip2 preprocess] fixture unset — skip");
        return;
    };
    let reference = load_reference(&ref_json);

    // Same synthetic image as scripts/siglip2_hf_dump.py::synthetic_image.
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
    let nchw = rlx_siglip2::siglip_normalize_nchw(&rgb, size, size, 224);
    let dmax = max_abs(&nchw, &reference.pixel_values);
    let mean: f32 = nchw
        .iter()
        .zip(&reference.pixel_values)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / nchw.len() as f32;
    eprintln!("[siglip2 preprocess/cpu] pixels: max|Δ|={dmax:.3e} mean|Δ|={mean:.3e}");
    assert!(dmax < 3e-2, "preprocess max|Δ|={dmax:.3e} too large");
    assert!(mean < 1e-3, "preprocess mean|Δ|={mean:.3e} too large");
}

#[cfg(feature = "metal")]
#[test]
fn reference_parity_metal() {
    if !rlx_runtime::device_ext::is_available(Device::Metal) {
        eprintln!("[siglip2 parity] Metal unavailable — skip");
        return;
    }
    run_parity_on(Device::Metal, "metal");
    run_naflex_parity_on(Device::Metal, "metal");
}

#[cfg(feature = "mlx")]
#[test]
fn reference_parity_mlx() {
    if !rlx_runtime::device_ext::is_available(Device::Mlx) {
        eprintln!("[siglip2 parity] MLX unavailable — skip");
        return;
    }
    run_parity_on(Device::Mlx, "mlx");
    run_naflex_parity_on(Device::Mlx, "mlx");
}

#[cfg(feature = "gpu")]
#[test]
fn reference_parity_gpu() {
    if !rlx_runtime::device_ext::is_available(Device::Gpu) {
        eprintln!("[siglip2 parity] wgpu unavailable — skip");
        return;
    }
    run_parity_on(Device::Gpu, "gpu");
    run_naflex_parity_on(Device::Gpu, "gpu");
}

#[cfg(feature = "cuda")]
#[test]
fn reference_parity_cuda() {
    if !rlx_runtime::device_ext::is_available(Device::Cuda) {
        eprintln!("[siglip2 parity] CUDA unavailable — skip");
        return;
    }
    run_parity_on(Device::Cuda, "cuda");
    run_naflex_parity_on(Device::Cuda, "cuda");
}

#[cfg(feature = "vulkan")]
#[test]
fn reference_parity_vulkan() {
    if !rlx_runtime::device_ext::is_available(Device::Vulkan) {
        eprintln!("[siglip2 parity] Vulkan unavailable — skip");
        return;
    }
    run_parity_on(Device::Vulkan, "vulkan");
    run_naflex_parity_on(Device::Vulkan, "vulkan");
}
