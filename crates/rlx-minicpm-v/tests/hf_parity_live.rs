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

//! Parity vs HF transformers (MiniCPM-V 4.7 / 4.6), stage by stage:
//! slicing + pixels, each crop's vision features, prompt ids, canvas
//! positions, prefill logits, greedy tokens.
//!
//! ```sh
//! .venv-minicpmv/bin/python -I scripts/minicpmv_reference.py \
//!     weights/vision/minicpm-v-4.7-1b crates/rlx-locateanything/fixtures/sample.jpg /tmp/minicpmv47
//! RLX_MINICPMV_DIR=$PWD/weights/vision/minicpm-v-4.7-1b RLX_MINICPMV_REF=/tmp/minicpmv47 \
//!     cargo test -p rlx-minicpm-v --release --test hf_parity_live -- --nocapture
//! ```

use rlx_minicpm_v::preprocess::unpack_navit;
use rlx_minicpm_v::{MiniCpmVRunner, RgbImage, chat_prompt};
use rlx_qwen35::{MultimodalPrefill, SampleOpts};
use rlx_runtime::Device;
use safetensors::SafeTensors;

fn test_device() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn f32s(st: &SafeTensors, k: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st
        .tensor(k)
        .unwrap_or_else(|_| panic!("reference has no {k}"));
    let v = t
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn ints(st: &SafeTensors, k: &str) -> Vec<i64> {
    let t = st
        .tensor(k)
        .unwrap_or_else(|_| panic!("reference has no {k}"));
    match t.dtype() {
        safetensors::Dtype::I64 => t
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect(),
        safetensors::Dtype::I32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as i64)
            .collect(),
        d => panic!("{k}: unexpected dtype {d:?}"),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// How close each stage must be. `None` reports without asserting (a
/// quantized GGUF: its features and logits legitimately move).
struct Tol {
    features_cos: Option<f64>,
    logits_abs: Option<f32>,
    /// Greedy may diverge only where the reference's top-2 margin is below this.
    greedy_margin: Option<f64>,
}

const HF_TOL: Tol = Tol {
    features_cos: Some(0.99999),
    logits_abs: Some(2e-2),
    greedy_margin: Some(0.05),
};

/// Prompt plus generated tokens: the context the runner is compiled for. The
/// prefill arena grows with it (on wgpu past its 4 GiB buffers at 1024).
fn context_len(meta: &serde_json::Value, st: &SafeTensors) -> usize {
    st.tensor("input_ids").unwrap().shape()[0] + meta["greedy"].as_array().unwrap().len()
}

fn load_reference(prefix: &str) -> (serde_json::Value, Vec<u8>) {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.json")).unwrap()).unwrap();
    (
        meta,
        std::fs::read(format!("{prefix}.safetensors")).unwrap(),
    )
}

#[test]
fn hf_reference_parity() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_MINICPMV_DIR"),
        rlx_ir::env::var_result("RLX_MINICPMV_REF"),
    ) else {
        eprintln!(
            "skip: set RLX_MINICPMV_DIR and RLX_MINICPMV_REF (scripts/minicpmv_reference.py)"
        );
        return;
    };
    let (meta, bytes) = load_reference(&prefix);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut runner = MiniCpmVRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .max_seq(context_len(&meta, &st))
        .build()
        .expect("build runner");
    check(&mut runner, &meta, &st, &HF_TOL);
}

/// The official GGUF pair against the same HF reference. `F16` (the default)
/// carries the bf16 weights almost exactly, so it is held to near-HF
/// tolerances; any other `RLX_MINICPMV_GGUF_PREFER` (e.g. `Q4_K_M`) checks the
/// exact stages — config, slicing, ids, positions — and reports the rest.
///
/// ```sh
/// RLX_MINICPMV_GGUF_DIR=$PWD/weights/vision/minicpm-v-4.7-1b-gguf RLX_MINICPMV_REF=/tmp/minicpmv47png \
///     cargo test -p rlx-minicpm-v --release --test hf_parity_live gguf -- --nocapture
/// ```
#[test]
fn gguf_reference_parity() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_MINICPMV_GGUF_DIR"),
        rlx_ir::env::var_result("RLX_MINICPMV_REF"),
    ) else {
        eprintln!("skip: set RLX_MINICPMV_GGUF_DIR and RLX_MINICPMV_REF");
        return;
    };
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("skip: {dir} is not a directory");
        return;
    }
    let prefer =
        rlx_ir::env::var_result("RLX_MINICPMV_GGUF_PREFER").unwrap_or_else(|_| "F16".into());
    let (meta, bytes) = load_reference(&prefix);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    // `RLX_MINICPMV_GGUF_PACKED=1` keeps the trunk's weights packed on the
    // device (half the memory of widening them, same values for F16).
    let mut runner = MiniCpmVRunner::builder()
        .model_dir(&dir)
        .prefer(&prefer)
        .device(test_device())
        .packed(rlx_ir::env::flag("RLX_MINICPMV_GGUF_PACKED"))
        .max_seq(context_len(&meta, &st))
        .build()
        .expect("build runner from GGUF");
    // The GGUF-derived config must describe the same model as the HF one.
    if let Ok(hf_dir) = rlx_ir::env::var_result("RLX_MINICPMV_DIR") {
        let hf = rlx_minicpm_v::MiniCpmVConfig::from_dir(std::path::Path::new(&hf_dir)).unwrap();
        let g = runner.config();
        assert_eq!(g.vision, hf.vision, "vision config");
        assert_eq!(g.slice, hf.slice, "slice config");
        assert_eq!(g.markers, hf.markers, "marker ids");
        assert_eq!(
            (g.downsample, g.mrope, g.text_hidden_size),
            (hf.downsample, hf.mrope, hf.text_hidden_size)
        );
        eprintln!("GGUF config matches config.json");
    }
    let tol = if prefer == "F16" {
        Tol {
            features_cos: Some(0.9999),
            logits_abs: Some(0.1),
            greedy_margin: Some(0.5),
        }
    } else {
        Tol {
            features_cos: None,
            logits_abs: None,
            greedy_margin: None,
        }
    };
    check(&mut runner, &meta, &st, &tol);
}

/// The scripts record paths as given, usually relative to the repo root.
fn repo_path(p: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(p)
    }
}

/// Positions `[t, h, w]` of the reference, `[3, S]` row-major.
fn assert_positions(ours: &[rlx_minicpm_v::positions::Pos], st: &SafeTensors) {
    let pos = ints(st, "position_ids");
    let s = ours.len();
    assert_eq!(pos.len(), 3 * s, "position count");
    for (t, p) in ours.iter().enumerate() {
        let want = [
            pos[t] as usize,
            pos[s + t] as usize,
            pos[2 * s + t] as usize,
        ];
        assert_eq!([p[0], p[1], p[2]], want, "position of token {t}");
    }
}

/// A clip against `scripts/minicpmv_video_reference.py`: sampled frame
/// indices, frame grids and pixels (both sides decode with `ffmpeg`), every
/// crop's features, ids, canvas positions, prefill logits, greedy tokens.
///
/// ```sh
/// .venv-minicpmv/bin/python -I scripts/minicpmv_video_reference.py \
///     weights/vision/minicpm-v-4.7-1b clip.mp4 .cache/minicpmv/video47 --max-num-frames 16
/// RLX_MINICPMV_DIR=$PWD/weights/vision/minicpm-v-4.7-1b \
///     RLX_MINICPMV_VIDEO_REF=$PWD/.cache/minicpmv/video47 \
///     cargo test -p rlx-minicpm-v --release --test hf_parity_live video -- --nocapture
/// ```
#[test]
fn video_reference_parity() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_MINICPMV_DIR"),
        rlx_ir::env::var_result("RLX_MINICPMV_VIDEO_REF"),
    ) else {
        eprintln!("skip: set RLX_MINICPMV_DIR and RLX_MINICPMV_VIDEO_REF");
        return;
    };
    let (meta, bytes) = load_reference(&prefix);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut runner = MiniCpmVRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .max_seq(context_len(&meta, &st))
        .build()
        .expect("build runner");
    let us = |k: &str| meta[k].as_u64().unwrap() as usize;

    // 1. Frame sampling.
    let path = repo_path(meta["video"].as_str().unwrap());
    let vmeta = rlx_minicpm_v::video::probe(&path).unwrap();
    assert_eq!(vmeta.total_frames, us("total_frames"), "frame count");
    let indices = rlx_minicpm_v::video::sample_frames(&vmeta, us("max_num_frames"));
    let hf_indices: Vec<usize> = meta["frame_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(indices, hf_indices, "sampled frames");

    // 2. Each frame sliced: grids and pixels.
    let frames = rlx_minicpm_v::video::decode_frames(&path, &vmeta, &indices).unwrap();
    let sliced: Vec<rlx_minicpm_v::SlicedImage> = frames
        .iter()
        .map(|f| runner.slice_frame(f, us("max_slice_nums")).unwrap())
        .collect();
    let hf_grids: Vec<(usize, usize)> = ints(&st, "target_sizes_videos")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let grids: Vec<(usize, usize)> = sliced
        .iter()
        .flat_map(|f| f.crops.iter().map(|c| (c.grid_h, c.grid_w)))
        .collect();
    assert_eq!(grids, hf_grids, "crop grids");
    let (pv, pv_shape) = f32s(&st, "pixel_values_videos");
    let hf_crops = unpack_navit(&pv, pv_shape[3], 14, &hf_grids);
    let ours: Vec<&rlx_minicpm_v::Crop> = sliced.iter().flat_map(|f| f.crops.iter()).collect();
    let mut worst = 0f32;
    for (a, b) in ours.iter().zip(&hf_crops) {
        worst = worst.max(max_abs(&a.patches, &b.patches));
    }
    eprintln!("{} crops: pixel max |rlx - hf| = {worst:.3e}", ours.len());
    // Pixels must agree when both sides ran the same ffmpeg build; another
    // build converts YUV to RGB its own way, so then they are only reported
    // (every later stage runs on the reference's pixels).
    let ours_ff = rlx_minicpm_v::video::ffmpeg_version().unwrap();
    if meta["ffmpeg"].as_str() == Some(ours_ff.as_str()) {
        assert!(worst <= 1.0 / 127.5 + 1e-6, "pixels differ by {worst}");
    } else {
        eprintln!(
            "ffmpeg differs from the reference's ({ours_ff:?} vs {:?}): pixels not held",
            meta["ffmpeg"]
        );
    }

    // 3. Vision features of every crop.
    let mut min_cos = 1f64;
    let mut worst = 0f32;
    for (i, c) in hf_crops.iter().enumerate() {
        let ours = runner.encode_crop(c).unwrap();
        let (theirs, _) = f32s(&st, &format!("features.{i}"));
        worst = worst.max(max_abs(&ours, &theirs));
        min_cos = min_cos.min(cosine(&ours, &theirs));
    }
    eprintln!("features: max |rlx - hf| = {worst:.3e}, min cos {min_cos:.7}");
    assert!(
        min_cos > HF_TOL.features_cos.unwrap(),
        "features cos {min_cos}"
    );

    // 4. Prompt ids and canvas positions, from the reference's pixels.
    let prompt = rlx_minicpm_v::chat_prompt_media(
        meta["question"].as_str().unwrap(),
        &[rlx_minicpm_v::Media::Video],
        false,
    );
    assert_eq!(prompt, meta["prompt"].as_str().unwrap(), "chat template");
    let mut hf_iter = hf_crops.into_iter();
    let hf_frames: Vec<rlx_minicpm_v::SlicedImage> = sliced
        .iter()
        .map(|f| rlx_minicpm_v::SlicedImage {
            crops: hf_iter.by_ref().take(f.crops.len()).collect(),
            grid: f.grid,
        })
        .collect();
    let assembled = runner
        .assemble_media(&prompt, &[rlx_minicpm_v::Visual::Video(hf_frames)])
        .unwrap();
    let hf_ids: Vec<u32> = ints(&st, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert_eq!(assembled.ids, hf_ids, "prompt ids");
    assert_positions(&assembled.positions, &st);
    eprintln!(
        "ids and canvas positions match ({} tokens, {} frames)",
        hf_ids.len(),
        indices.len()
    );
    check_lm(&mut runner, assembled, &meta, &st, &HF_TOL);
}

/// Stage-by-stage comparison against an HF reference dump.
fn check(runner: &mut MiniCpmVRunner, meta: &serde_json::Value, st: &SafeTensors, tol: &Tol) {
    // 1. Slicing: same grids, and pixels within one 8-bit level.
    let img = RgbImage::open(repo_path(meta["image"].as_str().unwrap())).unwrap();
    let sliced = runner.slice(&img).unwrap();
    let ts = ints(st, "target_sizes");
    let hf_grids: Vec<(usize, usize)> = ts
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let grids: Vec<(usize, usize)> = sliced.crops.iter().map(|c| (c.grid_h, c.grid_w)).collect();
    assert_eq!(grids, hf_grids, "crop grids");
    let (pv, pv_shape) = f32s(st, "pixel_values");
    let hf_crops = unpack_navit(&pv, pv_shape[3], 14, &hf_grids);
    for (i, (a, b)) in sliced.crops.iter().zip(&hf_crops).enumerate() {
        let d = max_abs(&a.patches, &b.patches);
        eprintln!("crop {i}: pixel max |rlx - hf| = {d:.3e}");
        // 1/127.5 is one uint8 level after (x/255 - 0.5)/0.5.
        assert!(d <= 1.0 / 127.5 + 1e-6, "crop {i} pixels differ by {d}");
    }

    // 2. Vision features per crop, fed the reference processor's pixels.
    for (i, c) in hf_crops.iter().enumerate() {
        let ours = runner.encode_crop(c).unwrap();
        let (theirs, _) = f32s(st, &format!("features.{i}"));
        let d = max_abs(&ours, &theirs);
        let cos = cosine(&ours, &theirs);
        let scale = theirs.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!(
            "crop {i}: features max |rlx - hf| = {d:.3e} (max |hf| {scale:.2}), cos {cos:.7}"
        );
        if let Some(min) = tol.features_cos {
            assert!(cos > min, "crop {i} features cos {cos}");
        }
    }

    // 3. Prompt ids and canvas positions.
    let prompt = chat_prompt(meta["question"].as_str().unwrap(), 1, false);
    assert_eq!(prompt, meta["prompt"].as_str().unwrap(), "chat template");
    let hf_sliced = rlx_minicpm_v::SlicedImage {
        crops: hf_crops,
        grid: sliced.grid,
    };
    let assembled = runner.assemble(&prompt, &[hf_sliced]).unwrap();
    let hf_ids: Vec<u32> = ints(st, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert_eq!(assembled.ids, hf_ids, "prompt ids");
    assert_positions(&assembled.positions, st);
    eprintln!("ids and canvas positions match ({} tokens)", hf_ids.len());
    check_lm(runner, assembled, meta, st, tol);
}

/// Prefill logits at the last prompt token, then greedy decode.
fn check_lm(
    runner: &mut MiniCpmVRunner,
    assembled: rlx_minicpm_v::AssembledPrompt,
    meta: &serde_json::Value,
    st: &SafeTensors,
    tol: &Tol,
) {
    let s = assembled.ids.len();
    // The features are spliced in; give the trunk the device memory.
    runner.release_vision_graphs();
    // 4. Prefill logits at the last prompt token.
    let (hf_logits, _) = f32s(st, "last_logits");
    let seed = runner
        .lm()
        .prefill_from_assembled(MultimodalPrefill {
            hidden: assembled.embeds.clone(),
            mrope_sections: assembled.positions.clone(),
            last_token_idx: s - 1,
            seq: assembled.ids.clone(),
        })
        .unwrap();
    let vocab = hf_logits.len();
    let last = &seed.trunk_logits[seed.trunk_logits.len() - vocab..];
    let d = max_abs(last, &hf_logits);
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
            .0
    };
    eprintln!(
        "last logits: max |rlx - hf| = {d:.3e}, argmax rlx {} hf {}",
        am(last),
        am(&hf_logits)
    );
    if let Some(max) = tol.logits_abs {
        assert!(d < max, "prefill logits differ by {d}");
    }

    // 5. Greedy decode agrees up to a near-tie in the reference.
    let greedy: Vec<u32> = meta["greedy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let margins: Vec<f64> = meta["margins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let got = runner
        .generate_assembled(assembled, greedy.len(), SampleOpts::greedy(), |_| true)
        .unwrap();
    eprintln!("rlx: {:?}\nrlx text: {}", got, runner.decode(&got).unwrap());
    eprintln!(
        "hf : {:?}\nhf  text: {}",
        greedy,
        meta["text"].as_str().unwrap()
    );
    if let Some(i) = got.iter().zip(&greedy).position(|(a, b)| a != b) {
        eprintln!(
            "first divergence at step {i}: reference margin {:.4}",
            margins[i]
        );
        if let Some(max) = tol.greedy_margin {
            assert!(
                margins[i] < max,
                "greedy diverged at step {i}, margin {}",
                margins[i]
            );
        }
    } else {
        let n = got.len().min(greedy.len());
        assert!(n == greedy.len() || got.len() < greedy.len(), "length");
    }
}

/// The resampler generation (MiniCPM-V-4) against
/// `scripts/minicpmv_legacy_reference.py`: crop grids and pixels, each crop's
/// resampler output, prompt ids, prefill logits, greedy tokens.
///
/// ```sh
/// .venv-sala/bin/python -I scripts/minicpmv_legacy_reference.py \
///     weights/vision/minicpm-v-4 crates/rlx-locateanything/fixtures/sample.jpg .cache/minicpmv/v4
/// RLX_MINICPMV_V4_DIR=$PWD/weights/vision/minicpm-v-4 RLX_MINICPMV_V4_REF=$PWD/.cache/minicpmv/v4 \
///     cargo test -p rlx-minicpm-v --release --test hf_parity_live resampler -- --nocapture
/// ```
#[test]
fn resampler_reference_parity() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_MINICPMV_V4_DIR"),
        rlx_ir::env::var_result("RLX_MINICPMV_V4_REF"),
    ) else {
        eprintln!("skip: set RLX_MINICPMV_V4_DIR and RLX_MINICPMV_V4_REF");
        return;
    };
    let (meta, bytes) = load_reference(&prefix);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut runner = rlx_minicpm_v::ResamplerRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .max_seq(context_len(&meta, &st))
        .build()
        .expect("build runner");

    // 1. Slicing: grids, and pixels (Pillow-exact resizes).
    let img = RgbImage::open(repo_path(meta["image"].as_str().unwrap())).unwrap();
    let sliced = runner.slice(&img).unwrap();
    let hf_grids: Vec<(usize, usize)> = ints(&st, "tgt_sizes")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let grids: Vec<(usize, usize)> = sliced.crops.iter().map(|c| (c.grid_h, c.grid_w)).collect();
    assert_eq!(grids, hf_grids, "crop grids");
    let mut hf_crops = Vec::new();
    for (i, (ours, &g)) in sliced.crops.iter().zip(&hf_grids).enumerate() {
        let (pv, shape) = f32s(&st, &format!("pixel.{i}"));
        let hf = unpack_navit(&pv, shape[2], 14, &[g]).remove(0);
        let d = max_abs(&ours.patches, &hf.patches);
        eprintln!("crop {i}: pixel max |rlx - hf| = {d:.3e}");
        assert!(d <= 1e-6, "crop {i} pixels differ by {d}");
        hf_crops.push(hf);
    }

    // 2. Resampler output per crop, from the reference's pixels.
    let refs: Vec<&rlx_minicpm_v::Crop> = hf_crops.iter().collect();
    let feats = runner.encode_crops(&refs).unwrap();
    for (i, ours) in feats.iter().enumerate() {
        let (theirs, _) = f32s(&st, &format!("features.{i}"));
        let d = max_abs(ours, &theirs);
        let cos = cosine(ours, &theirs);
        let scale = theirs.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!(
            "crop {i}: features max |rlx - hf| = {d:.3e} (max |hf| {scale:.2}), cos {cos:.7}"
        );
        assert!(
            cos > HF_TOL.features_cos.unwrap(),
            "crop {i} features cos {cos}"
        );
    }

    // 3. Prompt ids.
    let prompt = runner.chat_prompt(meta["question"].as_str().unwrap(), 1, false);
    assert_eq!(prompt, meta["prompt"].as_str().unwrap(), "chat template");
    let hf_sliced = rlx_minicpm_v::SlicedImage {
        crops: hf_crops,
        grid: sliced.grid,
    };
    let (ids, embeds) = runner.assemble(&prompt, &[hf_sliced]).unwrap();
    let hf_ids: Vec<u32> = ints(&st, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert_eq!(ids, hf_ids, "prompt ids");
    eprintln!("prompt ids match ({} tokens)", ids.len());

    // 4. Prefill logits.
    runner.release_vision_graphs();
    let (hf_logits, _) = f32s(&st, "last_logits");
    runner.lm().reset();
    let logits = runner.lm().feed_embeds(&embeds).unwrap();
    let d = max_abs(&logits, &hf_logits);
    let scale = hf_logits.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!("last logits: max |rlx - hf| = {d:.3e} (max |hf| {scale:.2})");
    assert!(d < 5e-2, "prefill logits differ by {d}");

    // 5. Greedy tokens, up to a near-tie in the reference.
    let greedy: Vec<u32> = meta["greedy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let margins: Vec<f64> = meta["margins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let got = runner
        .generate_embeds(&embeds, greedy.len(), SampleOpts::greedy(), |_| true)
        .unwrap();
    eprintln!("rlx: {got:?}\nrlx text: {}", runner.decode(&got).unwrap());
    eprintln!(
        "hf : {greedy:?}\nhf  text: {}",
        meta["text"].as_str().unwrap()
    );
    // The reference stops after the end-of-turn token; ours does not return it.
    if let Some(i) = got.iter().zip(&greedy).position(|(a, b)| a != b) {
        eprintln!(
            "first divergence at step {i}: reference margin {:.4}",
            margins[i]
        );
        assert!(
            margins[i] < 0.05,
            "greedy diverged at step {i}, margin {}",
            margins[i]
        );
    } else {
        assert!(greedy.len() - got.len() <= 1, "rlx stopped early");
    }
}

/// MiniCPM-V 4.5's packed video against
/// `scripts/minicpmv_legacy_reference.py --video`: sampled frames and their
/// temporal groups, frame grids and pixels (when both sides ran the same
/// ffmpeg), each group's resampler output, ids, prefill logits, greedy tokens.
///
/// ```sh
/// .venv-sala/bin/python -I scripts/minicpmv_legacy_reference.py weights/vision/minicpm-v-4_5 \
///     .cache/minicpmv/v45video --video clip.mp4 --fps 1 --force-packing 3 --widen
/// RLX_MINICPMV_V4_DIR=$PWD/weights/vision/minicpm-v-4_5 \
///     RLX_MINICPMV_V45_VIDEO_REF=$PWD/.cache/minicpmv/v45video \
///     cargo test -p rlx-minicpm-v --release --test hf_parity_live packed_video -- --nocapture
/// ```
#[test]
fn packed_video_reference_parity() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_MINICPMV_V4_DIR"),
        rlx_ir::env::var_result("RLX_MINICPMV_V45_VIDEO_REF"),
    ) else {
        eprintln!("skip: set RLX_MINICPMV_V4_DIR (4.5) and RLX_MINICPMV_V45_VIDEO_REF");
        return;
    };
    let (meta, bytes) = load_reference(&prefix);
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut runner = rlx_minicpm_v::ResamplerRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .max_seq(context_len(&meta, &st))
        .build()
        .expect("build runner");
    let list = |k: &str| -> Vec<usize> {
        meta[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };

    // 1. Sampling and temporal groups.
    let path = repo_path(meta["video"].as_str().unwrap());
    let vmeta = rlx_minicpm_v::video::probe(&path).unwrap();
    let opts = rlx_minicpm_v::video::PackOpts {
        fps: meta["choose_fps"].as_u64().unwrap() as usize,
        force_packing: meta["force_packing"].as_u64().map(|x| x as usize),
        ..Default::default()
    };
    let (idx, groups) = rlx_minicpm_v::video::sample_packed(&vmeta, &opts);
    assert_eq!(idx, list("frame_indices"), "sampled frames");
    let hf_groups: Vec<Vec<usize>> = meta["temporal_groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    assert_eq!(groups, hf_groups, "temporal groups");
    eprintln!("{} frames in {} groups match", idx.len(), groups.len());

    // 2. Frames: grids, and pixels under the same ffmpeg.
    let frames = rlx_minicpm_v::video::decode_frames(&path, &vmeta, &idx).unwrap();
    let packed = runner.pack_frames(&frames, &groups).unwrap();
    let hf_grids: Vec<(usize, usize)> = ints(&st, "tgt_sizes")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0] as usize, c[1] as usize))
        .collect();
    let ours: Vec<&rlx_minicpm_v::Crop> = packed.groups.iter().flatten().map(|(c, _)| c).collect();
    let grids: Vec<(usize, usize)> = ours.iter().map(|c| (c.grid_h, c.grid_w)).collect();
    assert_eq!(grids, hf_grids, "frame grids");
    let mut hf_frames = Vec::new();
    let mut worst = 0f32;
    for (i, (c, &g)) in ours.iter().zip(&hf_grids).enumerate() {
        let (pv, shape) = f32s(&st, &format!("pixel.{i}"));
        let hf = unpack_navit(&pv, shape[2], 14, &[g]).remove(0);
        worst = worst.max(max_abs(&c.patches, &hf.patches));
        hf_frames.push(hf);
    }
    eprintln!("{} frames: pixel max |rlx - hf| = {worst:.3e}", ours.len());
    if meta["ffmpeg"].as_str() == Some(rlx_minicpm_v::video::ffmpeg_version().unwrap().as_str()) {
        assert!(worst <= 1e-6, "pixels differ by {worst}");
    }

    // 3. One resampler pass per group, from the reference's pixels.
    let mut it = hf_frames.into_iter();
    let hf_packed = rlx_minicpm_v::PackedVideo {
        groups: groups
            .iter()
            .map(|g| g.iter().map(|&t| (it.next().unwrap(), t)).collect())
            .collect(),
    };
    let mut min_cos = 1f64;
    let mut worst = 0f32;
    for (g, group) in hf_packed.groups.iter().enumerate() {
        let feats = runner.encode_group(group).unwrap();
        let (theirs, _) = f32s(&st, &format!("features.{g}"));
        worst = worst.max(max_abs(&feats, &theirs));
        min_cos = min_cos.min(cosine(&feats, &theirs));
    }
    eprintln!("group features: max |rlx - hf| = {worst:.3e}, min cos {min_cos:.7}");
    assert!(
        min_cos > HF_TOL.features_cos.unwrap(),
        "group features cos {min_cos}"
    );

    // 4. Ids, prefill logits, greedy.
    let (ids, embeds) = runner
        .assemble_video(meta["question"].as_str().unwrap(), &hf_packed, false)
        .unwrap();
    let hf_ids: Vec<u32> = ints(&st, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert_eq!(ids, hf_ids, "prompt ids");
    eprintln!("prompt ids match ({} tokens)", ids.len());
    runner.release_vision_graphs();
    let (hf_logits, _) = f32s(&st, "last_logits");
    runner.lm().reset();
    let logits = runner.lm().feed_embeds(&embeds).unwrap();
    let d = max_abs(&logits, &hf_logits);
    eprintln!("last logits: max |rlx - hf| = {d:.3e}");
    assert!(d < 5e-2, "prefill logits differ by {d}");
    let greedy: Vec<u32> = list("greedy").into_iter().map(|x| x as u32).collect();
    let margins: Vec<f64> = meta["margins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let got = runner
        .generate_embeds(&embeds, greedy.len(), SampleOpts::greedy(), |_| true)
        .unwrap();
    eprintln!("rlx text: {}", runner.decode(&got).unwrap());
    eprintln!("hf  text: {}", meta["text"].as_str().unwrap());
    if let Some(i) = got.iter().zip(&greedy).position(|(a, b)| a != b) {
        eprintln!(
            "first divergence at step {i}: reference margin {:.4}",
            margins[i]
        );
        assert!(
            margins[i] < 0.05,
            "greedy diverged at step {i}, margin {}",
            margins[i]
        );
    } else {
        assert!(greedy.len() - got.len() <= 1, "rlx stopped early");
    }
}
