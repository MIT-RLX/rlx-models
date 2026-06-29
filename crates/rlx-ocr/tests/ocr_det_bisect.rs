// SPDX-License-Identifier: GPL-3.0-only
//! Bisect the detection U-Net Metal vs CPU stage-by-stage to name the op that
//! diverges (CPU matches rten). `OCR_MODEL_DIR=<dir> cargo test -p rlx-ocr
//! --release --features metal --test ocr_det_bisect -- --ignored --nocapture`
#![cfg(feature = "rlx")]

use image::GenericImageView;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_ocr::host_resize::{pad_hw_end, resize_bilinear};
use rlx_ocr::model::{DetectionGraphConfig, build_detection_graph_to_stage};
use rlx_ocr::weights::{
    HF_DETECTION_ST, HF_DETECTION_ST_FULL, SafetensorsFile, prefer_safetensors_path,
};
use rlx_ocr::{BLACK_VALUE, ImageSource, OcrEngine, input_image};
use rlx_runtime::{Device, Session};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use std::path::PathBuf;

#[test]
#[ignore]
fn detection_metal_bisect() {
    let dir = PathBuf::from(std::env::var("OCR_MODEL_DIR").expect("OCR_MODEL_DIR"));
    let path = prefer_safetensors_path(&dir, HF_DETECTION_ST, HF_DETECTION_ST_FULL);
    let st = SafetensorsFile::open(&path).expect("open detection safetensors");
    let cfg = DetectionGraphConfig::default();
    let (in_h, in_w) = (cfg.height, cfg.width);

    // Real preprocessed detection input: grayscale → pad → resize, exactly as
    // `RlxTextDetector::detect_text_pixels` does it.
    let img_path = std::env::var("OCR_TEST_IMAGE").expect("OCR_TEST_IMAGE");
    let engine = OcrEngine::from_model_dir_on_device(&dir, Device::Cpu).unwrap();
    let dynimg = image::open(&img_path).unwrap();
    let (iw, ih) = dynimg.dimensions();
    let rgb = dynimg.to_rgb8().into_raw();
    let oinput = engine
        .prepare_input(ImageSource::from_bytes(&rgb, (iw, ih)).unwrap())
        .unwrap();
    let grey = input_image(&oinput);
    let [c, gh, gw] = grey.shape();
    let grey4 = grey.reshaped([1, c, gh, gw]);
    let (pad_b, pad_r) = (in_h.saturating_sub(gh), in_w.saturating_sub(gw));
    let padded: NdTensor<f32, 4> = if pad_b > 0 || pad_r > 0 {
        pad_hw_end(grey4.view(), pad_b, pad_r, BLACK_VALUE)
    } else {
        NdTensor::from_data(grey4.shape(), grey4.to_vec())
    };
    let resized: NdTensor<f32, 4> = if padded.size(2) != in_h || padded.size(3) != in_w {
        resize_bilinear(padded.view(), in_h, in_w)
    } else {
        padded
    };
    let input: Vec<f32> = resized.iter().copied().collect();
    eprintln!("[det-bisect] REAL input {in_h}x{in_w} (from {gh}x{gw} grey)");

    let run = |stage: Option<u8>, device: Device| -> Vec<f32> {
        let mut wm = st.weight_map().unwrap();
        let (graph, params) = build_detection_graph_to_stage(&mut wm, cfg, stage).unwrap();
        let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &[]);
        compiled.run(&[("image", input.as_slice())]).pop().unwrap()
    };

    let cmp = |label: &str, stage: Option<u8>| {
        let cpu = run(stage, Device::Cpu);
        let metal = run(stage, Device::Metal);
        let maxd = cpu
            .iter()
            .zip(&metal)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mark = if maxd > 1e-3 { "  <<< DIVERGES" } else { "" };
        eprintln!(
            "[det-bisect] {label:<22}: n={:>8} max_abs_diff={maxd:.5}{mark}",
            cpu.len()
        );
    };
    cmp("stage 12 (decoder end)", Some(12));
    cmp("stage 13 (logits/out_conv)", Some(13));
    cmp("mask (sigmoid, full)", None);
}
