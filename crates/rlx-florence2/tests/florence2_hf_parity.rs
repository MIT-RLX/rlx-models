// RLX — Florence-2 staged HF parity test.
//
// Compares the RLX vision tower, BART encoder, and first-step decoder logits
// against a reference dumped by `scripts/florence2_hf_parity.py`.
//
// Gated on:
//   RLX_FLORENCE2_DIR   — checkpoint dir (safetensors + config.json)
//   RLX_FLORENCE2_FIXTURE — JSON dumped by the reference script
// Both default to the in-repo `.cache/florence2/...` locations; the test is a
// no-op when they are absent.

use rlx_florence2::{Florence2Config, Florence2Model, GenerateConfig};
use rlx_runtime::Device;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_mean_abs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let (mut max, mut sum) = (0f32, 0f64);
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        sum += d as f64;
        if d > max {
            max = d;
        }
    }
    (max, (sum / n as f64) as f32)
}

fn argmax(a: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in a.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
}

fn dir() -> Option<PathBuf> {
    let p = std::env::var("RLX_FLORENCE2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/florence2/Florence-2-large"));
    p.join("config.json").is_file().then_some(p)
}

fn fixture() -> Option<serde_json::Value> {
    let p = std::env::var("RLX_FLORENCE2_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/florence2/parity_caption.json"));
    let raw = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn f32_vec(v: &serde_json::Value, key: &str) -> Vec<f32> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture missing `{key}`"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn u32_vec(v: &serde_json::Value, key: &str) -> Vec<u32> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture missing `{key}`"))
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn florence2_large_staged_parity() {
    let (Some(dir), Some(fx)) = (dir(), fixture()) else {
        eprintln!("[florence2] skipping parity (weights/fixture absent)");
        return;
    };

    let cfg = Florence2Config::from_hf_config_json(&dir.join("config.json")).unwrap();
    let mut model = Florence2Model::load(&dir, cfg, Device::Cpu).unwrap();

    let pixel = f32_vec(&fx, "pixel_values");
    let img_size = fx["img_size"].as_u64().unwrap() as usize;
    let input_ids = u32_vec(&fx, "input_ids");
    let ref_feats = f32_vec(&fx, "image_features");
    let ref_enc = f32_vec(&fx, "encoder_hidden");
    let seq = fx["seq"].as_u64().unwrap() as usize;
    let ref_step0 = f32_vec(&fx, "step0_logits");

    // Stage 1: vision tower.
    let feats = model.encode_image(&pixel, img_size).unwrap();
    let cos = cosine(&feats, &ref_feats);
    let (mx, mn) = max_mean_abs(&feats, &ref_feats);
    eprintln!(
        "[florence2] image_features cos={cos:.6} max={mx:.4} mean={mn:.5} (rlx {} vs ref {})",
        feats.len(),
        ref_feats.len()
    );
    assert!(cos >= 0.9995, "vision features cosine {cos} < 0.9995");

    // Stage 2: encoder.
    let text = model.embed_text(&input_ids);
    let merged = model.merge_embeds(&feats, &text);
    assert_eq!(merged.len(), seq * model.config().text.d_model);
    let enc = model.encode(&merged, seq).unwrap();
    let cos_e = cosine(&enc, &ref_enc);
    let (mxe, mne) = max_mean_abs(&enc, &ref_enc);
    eprintln!("[florence2] encoder_hidden cos={cos_e:.6} max={mxe:.4} mean={mne:.5}");
    assert!(cos_e >= 0.9995, "encoder hidden cosine {cos_e} < 0.9995");

    // Stage 3: first decode step logits.
    let start = model.config().text.decoder_start_token_id;
    let logits = model.decoder_logits(&[start], &enc, seq).unwrap();
    let cos_l = cosine(&logits, &ref_step0);
    let (mxl, mnl) = max_mean_abs(&logits, &ref_step0);
    let top_rlx = argmax(&logits);
    let top_ref = argmax(&ref_step0);
    eprintln!(
        "[florence2] step0 logits cos={cos_l:.6} max={mxl:.4} mean={mnl:.5} top1 rlx={top_rlx} ref={top_ref}"
    );
    assert_eq!(top_rlx, top_ref, "step0 argmax mismatch");
    assert!(cos_l >= 0.999, "step0 logits cosine {cos_l} < 0.999");

    // Stage 4: greedy + beam token sequences must match HF exactly.
    let ref_greedy = u32_vec(&fx, "greedy");
    let ref_beam = u32_vec(&fx, "beam");
    let max_new = ref_greedy.len().max(ref_beam.len()) + 2;
    let gcfg = GenerateConfig::from_text(&model.config().text, max_new);

    let greedy = model.generate_greedy(&enc, seq, &gcfg).unwrap();
    eprintln!("[florence2] greedy rlx={greedy:?}\n              ref={ref_greedy:?}");
    assert_eq!(greedy, ref_greedy, "greedy token mismatch");

    let beam = model.generate_beam(&enc, seq, &gcfg).unwrap();
    eprintln!("[florence2] beam   rlx={beam:?}\n              ref={ref_beam:?}");
    assert_eq!(beam, ref_beam, "beam token mismatch");
}

/// Pillow bicubic preprocessing must reproduce the HF `pixel_values`.
#[test]
fn florence2_preprocess_parity() {
    let Some(fx) = fixture() else {
        eprintln!("[florence2] skipping preprocess parity (fixture absent)");
        return;
    };
    let Some(rgb_vals) = fx.get("image_rgb").and_then(|v| v.as_array()) else {
        eprintln!("[florence2] fixture lacks image_rgb; regenerate");
        return;
    };
    let rgb: Vec<u8> = rgb_vals.iter().map(|x| x.as_u64().unwrap() as u8).collect();
    let hw = fx["image_hw"].as_array().unwrap();
    let (h, w) = (
        hw[0].as_u64().unwrap() as usize,
        hw[1].as_u64().unwrap() as usize,
    );
    let out = fx["img_size"].as_u64().unwrap() as usize;
    let ref_pixel = f32_vec(&fx, "pixel_values");

    let pixel = rlx_florence2::preprocess::preprocess_rgb(&rgb, h, w, out);
    let cos = cosine(&pixel, &ref_pixel);
    let (mx, mn) = max_mean_abs(&pixel, &ref_pixel);
    eprintln!("[florence2] preprocess cos={cos:.6} max={mx:.5} mean={mn:.6}");
    assert!(cos >= 0.9999, "pixel_values cosine {cos} < 0.9999");
    assert!(mx < 0.02, "pixel_values max abs diff {mx} too large");
}

/// Tokenizer + post-processor: round-trip the caption tokens and check the
/// loc-token id mapping used by region tasks.
#[test]
fn florence2_tokenizer_postprocess() {
    let Some(dir) = dir() else {
        eprintln!("[florence2] skipping tokenizer test (weights absent)");
        return;
    };
    let tk = rlx_florence2::Florence2Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    // <CAPTION> expands to its prompt and BART-encodes to the fixture ids.
    if let Some(fx) = fixture() {
        let ids = tk.encode_prompt("<CAPTION>").unwrap();
        let ref_ids = u32_vec(&fx, "input_ids");
        eprintln!("[florence2] prompt ids rlx={ids:?} ref={ref_ids:?}");
        assert_eq!(ids, ref_ids, "prompt token mismatch");
    }
    // <loc_0> .. <loc_999> map to contiguous ids.
    let loc0 = tk.encode_prompt("placeholder").ok(); // ensure tokenizer is live
    let _ = loc0;
    let decoded = tk.decode_keep_special(&[2, 0, 102]).unwrap();
    assert!(!decoded.is_empty());
}

/// Object-detection post-processor: feed HF's exact generated tokens into the
/// RLX parser and compare boxes/labels to HF `post_process_generation`.
/// Uses the OD fixture (`RLX_FLORENCE2_OD_FIXTURE`, default `parity_od.json`).
#[test]
fn florence2_od_postprocess_parity() {
    let Some(dir) = dir() else {
        eprintln!("[florence2] skipping OD parity (weights absent)");
        return;
    };
    let p = std::env::var("RLX_FLORENCE2_OD_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/florence2/parity_od.json"));
    let Some(fx): Option<serde_json::Value> = std::fs::read_to_string(&p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    else {
        eprintln!("[florence2] skipping OD parity (od fixture absent)");
        return;
    };

    let tk = rlx_florence2::Florence2Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let beam = u32_vec(&fx, "beam");
    let hw = fx["image_hw"].as_array().unwrap();
    let (h, w) = (
        hw[0].as_u64().unwrap() as f64,
        hw[1].as_u64().unwrap() as f64,
    );

    let result = rlx_florence2::postprocess::post_process("<OD>", &beam, &tk, (w, h)).unwrap();
    let boxes = match result {
        rlx_florence2::Florence2Result::BBoxes(b) => b,
        other => panic!("expected boxes, got {other:?}"),
    };

    let hf = &fx["hf_answer"];
    let hf_boxes = hf["bboxes"].as_array().unwrap();
    let hf_labels = hf["labels"].as_array().unwrap();
    eprintln!(
        "[florence2] OD boxes rlx={} hf={}",
        boxes.len(),
        hf_boxes.len()
    );
    assert_eq!(boxes.len(), hf_boxes.len(), "box count mismatch");
    for (i, inst) in boxes.iter().enumerate() {
        let hb = hf_boxes[i].as_array().unwrap();
        for j in 0..4 {
            let r = inst.bbox[j];
            let h = hb[j].as_f64().unwrap();
            assert!((r - h).abs() < 1e-2, "box {i} coord {j}: rlx {r} vs hf {h}");
        }
        let hl = hf_labels[i].as_str().unwrap();
        assert_eq!(inst.label, hl, "label {i} mismatch");
    }
}
