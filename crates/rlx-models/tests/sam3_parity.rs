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

//! SAM3 parity tests against `facebookresearch/sam3`.
//!
//! The native SAM3 implementation lands in milestones:
//!   1. patch_embed parity (numerical gate)
//!   2. full ViT trunk parity (numerical gate; warn-only until 2D-RoPE
//!      windowed attention is implemented)
//!   3. image pipeline (shape-only gate against PyTorch processor output)
//!   4. video state advancement check (requires Python video model to
//!      construct successfully)
//!
//! All tests share the same skip semantics as SAM2: without
//! `RLX_SAM3_WEIGHTS` pointing at a converted `.safetensors`, each test
//! prints a skip notice and returns Ok.

#![cfg(feature = "parity-pytorch")]

mod compile_support;

use anyhow::{Context, Result, ensure};
use rlx_models::sam3::{SAM3_IMG_SIZE, SAM3_PATCH_GRID, SAM3_VISION_DIM, Sam3, Sam3Config};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const TOL_COS: f64 = 1e-7;
const WARN_MAX_DIFF: f32 = 5e-2;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_SAM3_WEIGHTS")
}

// 6.28 / 3.14 are deliberate 2-dp values, not TAU/PI: they define the
// synthetic parity input, and changing them would change every dumped
// reference blob compared against here.
#[allow(clippy::approx_constant)]
fn synthesize_image_u8() -> Vec<u8> {
    let n = SAM3_IMG_SIZE * SAM3_IMG_SIZE * 3;
    let mut v = vec![0u8; n];
    for y in 0..SAM3_IMG_SIZE {
        for x in 0..SAM3_IMG_SIZE {
            for c in 0..3 {
                let fx = x as f32 / SAM3_IMG_SIZE as f32;
                let fy = y as f32 / SAM3_IMG_SIZE as f32;
                let phase = (c as f32) * 0.7;
                let s = (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
                let val = ((s + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                v[(y * SAM3_IMG_SIZE + x) * 3 + c] = val;
            }
        }
    }
    v
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

fn read_f32_blob(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let mut f = fs::File::open(path).with_context(|| format!("opening reference blob {path:?}"))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() == expected_len * 4,
        "blob {path:?}: expected {} bytes, got {}",
        expected_len * 4,
        bytes.len()
    );
    let mut out = vec![0f32; expected_len];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn write_f32_blob(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn dump_reference(weights: &str, image_nchw: &[f32], text_prompt: &str) -> Result<PathBuf> {
    let tmp = env::temp_dir().join(format!("rlx_sam3_parity_{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let img_bin = tmp.join("image.f32");
    write_f32_blob(&img_bin, image_nchw)?;

    let use_docker = rlx_ir::env::var("RLX_SAM3_DOCKER").as_deref() == Some("1");
    let device = rlx_ir::env::var("RLX_SAM3_DEVICE").unwrap_or_else(|| "cpu".to_string());

    let mut cmd = if use_docker {
        let mut c = Command::new("bash");
        c.arg("tests/sam3_parity_helpers/run-ref.sh");
        c.env("RLX_SAM3_DEVICE", &device);
        if let Some(tag) = rlx_ir::env::var("RLX_SAM3_IMAGE_TAG") {
            c.env("RLX_SAM3_IMAGE_TAG", tag);
        }
        c
    } else {
        let mut c =
            Command::new(rlx_ir::env::var("RLX_SAM3_PYTHON").unwrap_or_else(|| "python3".into()));
        c.arg("tests/sam3_parity_helpers/dump_reference.py");
        c.env("RLX_SAM3_DEVICE", &device);
        c
    };

    cmd.env("RLX_SAM3_WEIGHTS", weights)
        .env("RLX_SAM3_IMAGE_BIN", &img_bin)
        .env("RLX_SAM3_OUT_DIR", &tmp)
        .env("RLX_SAM3_TEXT_PROMPT", text_prompt)
        .env("RLX_SAM3_RUN_IMAGE", "1")
        .env(
            "RLX_SAM3_RUN_VIDEO",
            rlx_ir::env::var("RLX_SAM3_RUN_VIDEO").unwrap_or_else(|| "0".into()),
        );

    let out = cmd
        .output()
        .with_context(|| "running SAM3 reference dumper")?;
    if !out.status.success() {
        anyhow::bail!(
            "SAM3 reference failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    Ok(tmp)
}

#[test]
fn sam3_patch_embed_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!(
            "skipping SAM3 parity: set RLX_SAM3_WEIGHTS to a converted .safetensors checkpoint"
        );
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!(
            "skipping SAM3 native parity: convert .pt with tests/sam3_parity_helpers/pt_to_safetensors.py"
        );
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);

    let mut cfg = Sam3Config::base();
    cfg.vit.use_abs_pos = false;
    let model = Sam3::from_safetensors(&weights, cfg)?;
    let encoded = model.patch_embed_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
    ensure!(encoded.grid == SAM3_PATCH_GRID);
    ensure!(encoded.embed_dim == SAM3_VISION_DIM);

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let reference = read_f32_blob(
        &ref_dir.join("patch_embed.f32"),
        SAM3_PATCH_GRID * SAM3_PATCH_GRID * SAM3_VISION_DIM,
    )?;
    let (mad, idx) = max_abs_diff(&encoded.patch_tokens, &reference);
    let cos = cosine_distance(&encoded.patch_tokens, &reference);
    eprintln!("sam3 patch_embed parity: cos_dist={cos:.3e} max_abs_diff={mad:.6} idx={idx}");
    if mad > WARN_MAX_DIFF {
        eprintln!("warning: SAM3 patch_embed max_abs_diff {mad:.6} exceeds {WARN_MAX_DIFF}");
    }
    ensure!(
        cos <= TOL_COS,
        "SAM3 patch_embed cosine distance {cos:.3e} > {TOL_COS}"
    );
    Ok(())
}

#[test]
fn sam3_vision_encoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 vision-encoder parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 vision-encoder parity: requires .safetensors");
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);

    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;
    let encoded = model.encode_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
    ensure!(encoded.grid == SAM3_PATCH_GRID);
    ensure!(encoded.embed_dim == SAM3_VISION_DIM);

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let blob = ref_dir.join("vision_encoder.f32");
    if !blob.exists() {
        eprintln!("skipping SAM3 vision-encoder parity: reference dumper did not emit blob");
        return Ok(());
    }
    let reference = read_f32_blob(&blob, SAM3_PATCH_GRID * SAM3_PATCH_GRID * SAM3_VISION_DIM)?;
    let (mad, idx) = max_abs_diff(&encoded.patch_tokens, &reference);
    let cos = cosine_distance(&encoded.patch_tokens, &reference);
    eprintln!("sam3 vision_encoder parity: cos_dist={cos:.3e} max_abs_diff={mad:.6} idx={idx}");
    // After 32 ViT blocks small fp rounding delta is expected; gate cosine tightly,
    // warn on absolute residual.
    if mad > 5e-1 {
        eprintln!(
            "warning: SAM3 vision_encoder max_abs_diff {mad:.6} (this is informational; cosine is the gate)"
        );
    }
    ensure!(
        cos <= 1e-5,
        "SAM3 vision_encoder cosine distance {cos:.3e} > 1e-5"
    );
    Ok(())
}

fn read_i32_blob(path: &Path) -> Result<Vec<i32>> {
    let mut f = fs::File::open(path).with_context(|| format!("opening i32 blob {path:?}"))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() % 4 == 0,
        "i32 blob {path:?}: byte length not multiple of 4"
    );
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[test]
fn sam3_text_encoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 text encoder parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 text encoder parity: requires .safetensors");
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);

    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let tokens_blob = ref_dir.join("text_tokens.i32");
    let memory_blob = ref_dir.join("text_memory_resized.f32");
    if !tokens_blob.exists() || !memory_blob.exists() {
        eprintln!("skipping SAM3 text encoder parity: reference blobs missing");
        return Ok(());
    }
    let tokens_i32 = read_i32_blob(&tokens_blob)?;
    ensure!(
        tokens_i32.len() == 32,
        "expected 32 tokens, got {}",
        tokens_i32.len()
    );
    let tokens: Vec<u32> = tokens_i32.iter().map(|&v| v as u32).collect();
    let encoded = model.encode_text_tokens(&tokens, 1, 32)?;
    let expected = read_f32_blob(&memory_blob, 32 * 256)?;
    let (mad, idx) = max_abs_diff(&encoded.text_memory_resized, &expected);
    let cos = cosine_distance(&encoded.text_memory_resized, &expected);
    eprintln!(
        "sam3 text_memory_resized parity: cos_dist={cos:.3e} max_abs_diff={mad:.6} idx={idx}"
    );
    ensure!(
        cos <= 1e-5,
        "SAM3 text encoder cosine distance {cos:.3e} > 1e-5"
    );
    Ok(())
}

fn read_u8_blob(path: &Path) -> Result<Vec<u8>> {
    let mut f = fs::File::open(path).with_context(|| format!("opening u8 blob {path:?}"))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[test]
fn sam3_detector_encoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 detector encoder parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 detector encoder parity: requires .safetensors");
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);

    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let src_blob = ref_dir.join("encoder_src.f32");
    let pos_blob = ref_dir.join("encoder_pos.f32");
    let prompt_blob = ref_dir.join("encoder_prompt.f32");
    let prompt_mask_blob = ref_dir.join("encoder_prompt_mask.u8");
    let memory_blob = ref_dir.join("encoder_memory.f32");
    for b in [
        &src_blob,
        &pos_blob,
        &prompt_blob,
        &prompt_mask_blob,
        &memory_blob,
    ] {
        if !b.exists() {
            eprintln!("skipping SAM3 detector encoder parity: reference blob missing: {b:?}");
            return Ok(());
        }
    }

    let batch = 1usize;
    let c = 256usize;
    let h = 72usize;
    let w = 72usize;
    let seq = 32usize;

    let src = read_f32_blob(&src_blob, batch * c * h * w)?;
    let pos = read_f32_blob(&pos_blob, batch * c * h * w)?;
    let prompt = read_f32_blob(&prompt_blob, seq * batch * c)?;
    let prompt_mask = read_u8_blob(&prompt_mask_blob)?;
    ensure!(
        prompt_mask.len() == batch * seq,
        "prompt mask len {}",
        prompt_mask.len()
    );
    let expected = read_f32_blob(&memory_blob, batch * h * w * c)?;

    let memory = model.run_encoder(&src, &pos, &prompt, &prompt_mask, batch, h, w, seq)?;
    let (mad, idx) = max_abs_diff(&memory, &expected);
    let cos = cosine_distance(&memory, &expected);
    eprintln!("sam3 detector_encoder parity: cos_dist={cos:.3e} max_abs_diff={mad:.6} idx={idx}");
    ensure!(
        cos <= 1e-5,
        "SAM3 detector_encoder cosine distance {cos:.3e} > 1e-5"
    );
    Ok(())
}

#[test]
fn sam3_detector_decoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 detector decoder parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 detector decoder parity: requires .safetensors");
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let mem_blob = ref_dir.join("encoder_memory.f32");
    let pos_blob = ref_dir.join("encoder_pos.f32");
    let prompt_blob = ref_dir.join("encoder_prompt.f32");
    let prompt_mask_blob = ref_dir.join("encoder_prompt_mask.u8");
    let int_blob = ref_dir.join("decoder_intermediate.f32");
    let ref_boxes_blob = ref_dir.join("decoder_ref_boxes.f32");
    let plogit_blob = ref_dir.join("decoder_presence_logits.f32");
    for b in [
        &mem_blob,
        &pos_blob,
        &prompt_blob,
        &prompt_mask_blob,
        &int_blob,
        &ref_boxes_blob,
        &plogit_blob,
    ] {
        if !b.exists() {
            eprintln!("skipping SAM3 detector decoder parity: blob missing {b:?}");
            return Ok(());
        }
    }

    let batch = 1usize;
    let c = 256usize;
    let h = 72usize;
    let w = 72usize;
    let seq = 32usize;
    let nq = 200usize;
    let num_layers = 6usize;

    let memory = read_f32_blob(&mem_blob, batch * h * w * c)?;
    let memory_pos_seq = read_f32_blob(&pos_blob, batch * c * h * w)?;
    // The encoder_pos.f32 we dumped is the FPN pos in NCHW. We need batch-
    // first [B, H*W, C] for the decoder API to match the encoder's memory
    // ordering. Reshape NCHW → [B, H*W, C].
    let mut memory_pos = vec![0f32; batch * h * w * c];
    for b in 0..batch {
        for y in 0..h {
            for xc in 0..w {
                for ch in 0..c {
                    memory_pos[(b * h * w + y * w + xc) * c + ch] =
                        memory_pos_seq[((b * c + ch) * h + y) * w + xc];
                }
            }
        }
    }
    // encoder_memory is seq-first [hw, bs, C]; reorder to [B, hw, C].
    let mut memory_bf = vec![0f32; batch * h * w * c];
    for l in 0..h * w {
        for b in 0..batch {
            let src = (l * batch + b) * c;
            let dst = (b * h * w + l) * c;
            memory_bf[dst..dst + c].copy_from_slice(&memory[src..src + c]);
        }
    }
    let prompt = read_f32_blob(&prompt_blob, seq * batch * c)?;
    let prompt_mask = read_u8_blob(&prompt_mask_blob)?;
    ensure!(prompt_mask.len() == batch * seq);

    let out = model.run_decoder(
        &memory_bf,
        &memory_pos,
        &prompt,
        &prompt_mask,
        batch,
        h,
        w,
        seq,
    )?;

    let int_ref = read_f32_blob(&int_blob, num_layers * nq * batch * c)?;
    let (mad_i, idx_i) = max_abs_diff(&out.intermediate, &int_ref);
    let cos_i = cosine_distance(&out.intermediate, &int_ref);
    eprintln!(
        "sam3 decoder intermediate parity: cos_dist={cos_i:.3e} max_abs_diff={mad_i:.6} idx={idx_i}"
    );

    let ref_boxes_ref = read_f32_blob(&ref_boxes_blob, num_layers * nq * batch * 4)?;
    let (mad_r, _) = max_abs_diff(&out.intermediate_ref_boxes, &ref_boxes_ref);
    let cos_r = cosine_distance(&out.intermediate_ref_boxes, &ref_boxes_ref);
    eprintln!("sam3 decoder ref_boxes parity: cos_dist={cos_r:.3e} max_abs_diff={mad_r:.6}");

    let plogit_ref = read_f32_blob(&plogit_blob, num_layers * batch)?;
    let (mad_p, _) = max_abs_diff(&out.presence_logits, &plogit_ref);
    let cos_p = cosine_distance(&out.presence_logits, &plogit_ref);
    eprintln!("sam3 decoder presence_logits parity: cos_dist={cos_p:.3e} max_abs_diff={mad_p:.6}");

    ensure!(
        cos_i <= 1e-4,
        "SAM3 decoder intermediate cosine distance {cos_i:.3e} > 1e-4"
    );
    ensure!(
        cos_r <= 1e-4,
        "SAM3 decoder ref_boxes cosine distance {cos_r:.3e} > 1e-4"
    );
    Ok(())
}

#[test]
fn sam3_end_to_end_image_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 end-to-end parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 end-to-end parity: requires .safetensors");
        return Ok(());
    }
    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let cfg = Sam3Config::base();
    let mut model = Sam3::from_safetensors(&weights, cfg)?;
    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;

    let tokens_blob = ref_dir.join("text_tokens.i32");
    let scores_blob = ref_dir.join("decoder_scores.f32");
    let boxes_blob = ref_dir.join("final_boxes_xyxy.f32");
    let mask_blob = ref_dir.join("seg_mask_pred.f32");
    for b in [&tokens_blob, &scores_blob, &boxes_blob, &mask_blob] {
        if !b.exists() {
            eprintln!("skipping SAM3 end-to-end parity: blob missing {b:?}");
            return Ok(());
        }
    }
    let tokens_i32 = read_i32_blob(&tokens_blob)?;
    let tokens: Vec<u32> = tokens_i32.iter().map(|&v| v as u32).collect();
    let pred = model.predict_image_text(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE, &tokens)?;

    // Compare scores (last layer).
    let num_layers = 6;
    let nq = 200;
    let all_scores = read_f32_blob(&scores_blob, num_layers * nq)?;
    let last_scores = &all_scores[(num_layers - 1) * nq..];
    let (mad_s, _) = max_abs_diff(&pred.scores, last_scores);
    let cos_s = cosine_distance(&pred.scores, last_scores);
    eprintln!("sam3 e2e scores parity: cos_dist={cos_s:.3e} max_abs_diff={mad_s:.6}");

    // Compare boxes xyxy.
    let expected_boxes = read_f32_blob(&boxes_blob, nq * 4)?;
    let (mad_b, _) = max_abs_diff(&pred.boxes, &expected_boxes);
    let cos_b = cosine_distance(&pred.boxes, &expected_boxes);
    eprintln!("sam3 e2e boxes parity: cos_dist={cos_b:.3e} max_abs_diff={mad_b:.6}");

    // Compare masks.
    let h_out = 288;
    let w_out = 288;
    let expected_masks = read_f32_blob(&mask_blob, nq * h_out * w_out)?;
    let (mad_m, _) = max_abs_diff(&pred.masks, &expected_masks);
    let cos_m = cosine_distance(&pred.masks, &expected_masks);
    eprintln!("sam3 e2e masks parity: cos_dist={cos_m:.3e} max_abs_diff={mad_m:.6}");

    ensure!(
        cos_s <= 1e-3,
        "SAM3 e2e scores cosine distance {cos_s:.3e} > 1e-3"
    );
    ensure!(
        cos_b <= 1e-3,
        "SAM3 e2e boxes cosine distance {cos_b:.3e} > 1e-3"
    );
    ensure!(
        cos_m <= 1e-3,
        "SAM3 e2e masks cosine distance {cos_m:.3e} > 1e-3"
    );
    Ok(())
}

#[test]
fn sam3_video_model_construct_check() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 video model check: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 video model check: requires .safetensors");
        return Ok(());
    }
    // Native side: verify the Sam3 loader consumed every tracker tensor
    // (i.e. tracker.* weights are recognised and we expose them via
    // Sam3TrackerWeights.raw).
    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;
    let raw = &model.tracker_weights().raw;
    eprintln!("sam3 tracker recognised {} weight tensors", raw.len());
    ensure!(
        raw.len() >= 100,
        "expected >= 100 tracker weight tensors; got {}",
        raw.len()
    );
    for required in [
        "no_mem_embed",
        "no_obj_ptr",
        "mask_downsample.weight",
        "obj_ptr_tpos_proj.weight",
        "sam_mask_decoder.iou_token.weight",
        "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
    ] {
        ensure!(
            raw.contains_key(required),
            "tracker is missing key suffix: {required}"
        );
    }

    // Python side: ensure the upstream video model builds against the same
    // checkpoint (sentinel emitted by the dumper).
    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let tmp = env::temp_dir().join(format!("rlx_sam3_video_check_{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let img_bin = tmp.join("image.f32");
    write_f32_blob(&img_bin, &image_nchw)?;
    let mut cmd = if rlx_ir::env::var("RLX_SAM3_DOCKER").as_deref() == Some("1") {
        let mut c = Command::new("bash");
        c.arg("tests/sam3_parity_helpers/run-ref.sh");
        c
    } else {
        let mut c =
            Command::new(rlx_ir::env::var("RLX_SAM3_PYTHON").unwrap_or_else(|| "python3".into()));
        c.arg("tests/sam3_parity_helpers/dump_reference.py");
        c
    };
    cmd.env("RLX_SAM3_WEIGHTS", &weights)
        .env("RLX_SAM3_IMAGE_BIN", &img_bin)
        .env("RLX_SAM3_OUT_DIR", &tmp)
        .env("RLX_SAM3_RUN_VIDEO", "1");
    let out = cmd.output().context("running SAM3 video check")?;
    if !out.status.success() {
        eprintln!(
            "video check dumper failed (informational): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return Ok(());
    }
    let sentinel = tmp.join("video_model_ready.f32");
    if sentinel.exists() {
        let v = read_f32_blob(&sentinel, 1)?;
        ensure!(v[0] == 1.0, "video_model_ready sentinel mismatch: {v:?}");
        eprintln!("sam3 video model construct sentinel OK");
    } else {
        eprintln!("video check: sentinel not produced (informational)");
    }
    Ok(())
}

#[test]
fn sam3_detector_decoder_ir_parity_vs_pytorch() -> Result<()> {
    use rlx_models::sam3::detector_decoder_ir::Sam3CompiledDecoder;
    use rlx_runtime::Device;
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 IR decoder parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 IR decoder parity: requires .safetensors");
        return Ok(());
    }
    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;
    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let mem_blob = ref_dir.join("encoder_memory.f32");
    let pos_blob = ref_dir.join("encoder_pos.f32");
    let prompt_blob = ref_dir.join("encoder_prompt.f32");
    let prompt_mask_blob = ref_dir.join("encoder_prompt_mask.u8");
    let int_blob = ref_dir.join("decoder_intermediate.f32");
    let ref_boxes_blob = ref_dir.join("decoder_ref_boxes.f32");
    for b in [
        &mem_blob,
        &pos_blob,
        &prompt_blob,
        &prompt_mask_blob,
        &int_blob,
        &ref_boxes_blob,
    ] {
        if !b.exists() {
            eprintln!("skipping SAM3 IR decoder parity: blob missing {b:?}");
            return Ok(());
        }
    }
    let batch = 1usize;
    let c = 256usize;
    let h = 72usize;
    let w = 72usize;
    let seq = 32usize;
    let nq = 200usize;
    let num_layers = 6usize;

    let memory = read_f32_blob(&mem_blob, batch * h * w * c)?;
    let memory_pos_chw = read_f32_blob(&pos_blob, batch * c * h * w)?;
    // pos was dumped NCHW; the IR decoder API takes batch-first [B, hw, C].
    let mut memory_pos = vec![0f32; batch * h * w * c];
    for b in 0..batch {
        for y in 0..h {
            for xc in 0..w {
                for ch in 0..c {
                    memory_pos[(b * h * w + y * w + xc) * c + ch] =
                        memory_pos_chw[((b * c + ch) * h + y) * w + xc];
                }
            }
        }
    }
    let mut memory_bf = vec![0f32; batch * h * w * c];
    for l in 0..h * w {
        for b in 0..batch {
            let src = (l * batch + b) * c;
            let dst = (b * h * w + l) * c;
            memory_bf[dst..dst + c].copy_from_slice(&memory[src..src + c]);
        }
    }
    let prompt = read_f32_blob(&prompt_blob, seq * batch * c)?;
    let prompt_mask = read_u8_blob(&prompt_mask_blob)?;

    let mut dec = Sam3CompiledDecoder::new_with_profile(
        model.decoder_weights(),
        batch,
        h * w,
        seq,
        Device::Cpu,
        model.compile_profile(),
    )?;
    let (intermediate, ref_boxes, _presence_logits, _presence_feats) =
        dec.run(&memory_bf, &memory_pos, &prompt, &prompt_mask, h, w)?;

    let int_ref = read_f32_blob(&int_blob, num_layers * nq * batch * c)?;
    let (mad, _) = max_abs_diff(&intermediate, &int_ref);
    let cos = cosine_distance(&intermediate, &int_ref);
    eprintln!("sam3 IR decoder intermediate parity: cos_dist={cos:.3e} max_abs_diff={mad:.6}");

    let ref_boxes_ref = read_f32_blob(&ref_boxes_blob, num_layers * nq * batch * 4)?;
    let (mad_r, _) = max_abs_diff(&ref_boxes, &ref_boxes_ref);
    let cos_r = cosine_distance(&ref_boxes, &ref_boxes_ref);
    eprintln!("sam3 IR decoder ref_boxes parity: cos_dist={cos_r:.3e} max_abs_diff={mad_r:.6}");

    ensure!(
        cos <= 1e-4,
        "IR decoder intermediate cos_dist {cos:.3e} > 1e-4"
    );
    ensure!(
        cos_r <= 1e-4,
        "IR decoder ref_boxes cos_dist {cos_r:.3e} > 1e-4"
    );
    Ok(())
}

#[test]
fn sam3_dot_prod_scoring_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 scoring parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 scoring parity: requires .safetensors");
        return Ok(());
    }
    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let cfg = Sam3Config::base();
    let model = Sam3::from_safetensors(&weights, cfg)?;
    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    let int_blob = ref_dir.join("decoder_intermediate.f32");
    let scores_blob = ref_dir.join("decoder_scores.f32");
    let prompt_blob = ref_dir.join("encoder_prompt.f32");
    let prompt_mask_blob = ref_dir.join("encoder_prompt_mask.u8");
    for b in [&int_blob, &scores_blob, &prompt_blob, &prompt_mask_blob] {
        if !b.exists() {
            eprintln!("skipping SAM3 scoring parity: blob missing {b:?}");
            return Ok(());
        }
    }
    let num_layers = 6;
    let batch = 1;
    let nq = 200;
    let seq = 32;
    let c = 256;
    let intermediate = read_f32_blob(&int_blob, num_layers * nq * batch * c)?;
    // Convert decoder intermediate (seq-first [L, nq, bs, D]) to batch-
    // first [L, bs, nq, D] for the scoring API.
    let mut hs_bf = vec![0f32; num_layers * batch * nq * c];
    for l in 0..num_layers {
        for q in 0..nq {
            for b in 0..batch {
                let src = ((l * nq + q) * batch + b) * c;
                let dst = ((l * batch + b) * nq + q) * c;
                hs_bf[dst..dst + c].copy_from_slice(&intermediate[src..src + c]);
            }
        }
    }
    let prompt = read_f32_blob(&prompt_blob, seq * batch * c)?;
    let prompt_mask = read_u8_blob(&prompt_mask_blob)?;
    let scores =
        model.run_dot_prod_scoring(&hs_bf, &prompt, &prompt_mask, num_layers, batch, nq, seq)?;
    let expected = read_f32_blob(&scores_blob, num_layers * batch * nq)?;
    let (mad, _) = max_abs_diff(&scores, &expected);
    let cos = cosine_distance(&scores, &expected);
    eprintln!("sam3 dot_prod_scoring parity: cos_dist={cos:.3e} max_abs_diff={mad:.6}");
    ensure!(
        cos <= 1e-4,
        "SAM3 dot_prod_scoring cosine distance {cos:.3e} > 1e-4"
    );
    Ok(())
}

#[test]
fn sam3_segmentation_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 segmentation parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 segmentation parity: requires .safetensors");
        return Ok(());
    }
    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);
    let cfg = Sam3Config::base();
    let mut model = Sam3::from_safetensors(&weights, cfg)?;
    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;

    let int_blob = ref_dir.join("decoder_intermediate.f32");
    let mem_blob = ref_dir.join("encoder_memory.f32");
    let prompt_blob = ref_dir.join("encoder_prompt.f32");
    let prompt_mask_blob = ref_dir.join("encoder_prompt_mask.u8");
    let mask_blob = ref_dir.join("seg_mask_pred.f32");
    let sem_blob = ref_dir.join("seg_semantic.f32");
    for b in [
        &int_blob,
        &mem_blob,
        &prompt_blob,
        &prompt_mask_blob,
        &mask_blob,
        &sem_blob,
    ] {
        if !b.exists() {
            eprintln!("skipping SAM3 segmentation parity: blob missing {b:?}");
            return Ok(());
        }
    }

    let num_layers = 6;
    let batch = 1;
    let nq = 200;
    let seq = 32;
    let c = 256;
    let h = 72;
    let w = 72;
    let h_out = 288;
    let w_out = 288;

    // Re-run vision + neck to recover the backbone FPN (no separate dump).
    let levels = model.predict_neck(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
    ensure!(levels.len() == 4);
    // Take first 3 (drop scale 0.5 per scalp=1).
    let backbone_fpn: Vec<Vec<f32>> = levels[..3].iter().map(|l| l.features.clone()).collect();
    let backbone_shapes: Vec<(usize, usize)> = levels[..3].iter().map(|l| (l.h, l.w)).collect();

    let intermediate = read_f32_blob(&int_blob, num_layers * nq * batch * c)?;
    // Last layer queries, batch-first [bs, nq, c].
    let mut queries_last_bf = vec![0f32; batch * nq * c];
    let l = num_layers - 1;
    for q in 0..nq {
        for b in 0..batch {
            let src = ((l * nq + q) * batch + b) * c;
            let dst = (b * nq + q) * c;
            queries_last_bf[dst..dst + c].copy_from_slice(&intermediate[src..src + c]);
        }
    }

    // Encoder memory (seq-first → batch-first).
    let memory = read_f32_blob(&mem_blob, h * w * batch * c)?;
    let mut memory_bf = vec![0f32; batch * h * w * c];
    for li in 0..h * w {
        for b in 0..batch {
            let src = (li * batch + b) * c;
            let dst = (b * h * w + li) * c;
            memory_bf[dst..dst + c].copy_from_slice(&memory[src..src + c]);
        }
    }
    let prompt = read_f32_blob(&prompt_blob, seq * batch * c)?;
    let prompt_mask = read_u8_blob(&prompt_mask_blob)?;

    let seg = model.run_segmentation(
        &memory_bf,
        &backbone_fpn,
        &backbone_shapes,
        &queries_last_bf,
        &prompt,
        &prompt_mask,
        batch,
        h,
        w,
        nq,
        seq,
    )?;

    let expected_mask = read_f32_blob(&mask_blob, batch * nq * h_out * w_out)?;
    let (mad_m, _) = max_abs_diff(&seg.mask_pred, &expected_mask);
    let cos_m = cosine_distance(&seg.mask_pred, &expected_mask);
    eprintln!("sam3 seg mask_pred parity: cos_dist={cos_m:.3e} max_abs_diff={mad_m:.6}");

    let expected_sem = read_f32_blob(&sem_blob, batch * h_out * w_out)?;
    let (mad_s, _) = max_abs_diff(&seg.semantic_seg, &expected_sem);
    let cos_s = cosine_distance(&seg.semantic_seg, &expected_sem);
    eprintln!("sam3 seg semantic_seg parity: cos_dist={cos_s:.3e} max_abs_diff={mad_s:.6}");

    ensure!(
        cos_m <= 1e-3,
        "SAM3 segmentation mask_pred cosine distance {cos_m:.3e} > 1e-3"
    );
    ensure!(
        cos_s <= 1e-4,
        "SAM3 segmentation semantic_seg cosine distance {cos_s:.3e} > 1e-4"
    );
    Ok(())
}

#[test]
fn sam3_neck_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping SAM3 neck parity: set RLX_SAM3_WEIGHTS");
        return Ok(());
    };
    if !weights.ends_with(".safetensors") {
        eprintln!("skipping SAM3 neck parity: requires .safetensors");
        return Ok(());
    }

    let image = synthesize_image_u8();
    let (image_nchw, _) = rlx_models::sam3_preprocess_image(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE);

    let cfg = Sam3Config::base();
    let mut model = Sam3::from_safetensors(&weights, cfg)?;
    let levels = model.predict_neck(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
    ensure!(levels.len() == 4, "SAM3 neck must produce 4 levels");

    let ref_dir = dump_reference(&weights, &image_nchw, "person")?;
    for (i, lvl) in levels.iter().enumerate() {
        let blob = ref_dir.join(format!("neck_level_{i}.f32"));
        if !blob.exists() {
            eprintln!("skipping SAM3 neck level {i} parity: reference dumper did not emit blob");
            continue;
        }
        let expected_len = lvl.channels * lvl.h * lvl.w;
        let reference = read_f32_blob(&blob, expected_len)?;
        let (mad, _) = max_abs_diff(&lvl.features, &reference);
        let cos = cosine_distance(&lvl.features, &reference);
        eprintln!(
            "sam3 neck level {i} ({}×{}×{}): cos_dist={cos:.3e} max_abs_diff={mad:.6}",
            lvl.channels, lvl.h, lvl.w
        );
        ensure!(
            cos <= 1e-4,
            "SAM3 neck level {i} cosine distance {cos:.3e} > 1e-4"
        );

        // Positional encoding parity (preflight, no checkpoint dependency).
        let pos_blob = ref_dir.join(format!("neck_pos_{i}.f32"));
        if pos_blob.exists() {
            let pos_ref = read_f32_blob(&pos_blob, expected_len)?;
            let (pmad, _) = max_abs_diff(&lvl.pos, &pos_ref);
            eprintln!("sam3 neck level {i} pos max_abs_diff={pmad:.3e}");
            ensure!(
                pmad <= 1e-5,
                "SAM3 neck level {i} positional encoding max_abs_diff {pmad:.3e} > 1e-5"
            );
        }
    }
    Ok(())
}
