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

//! LocateAnything-3B config + safetensors layout (env-gated).
//!
//! ```bash
//! just fetch-locateanything
//! RLX_LOCATEANYTHING_DIR=.cache/locateanything/LocateAnything-3B \
//!   cargo test -p rlx-models --test locateanything_checkpoint --release
//! ```

use rlx_locateanything::load::EXPECTED_TENSOR_COUNT;
use rlx_locateanything::{
    LocateAnythingConfig, LocateAnythingWeightStore, PREFIX_LANGUAGE_MODEL, PREFIX_PROJECTOR,
    PREFIX_VISION,
};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_LOCATEANYTHING_DIR")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn locateanything_config_matches_hf_card() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let cfg = LocateAnythingConfig::from_model_dir(&dir).expect("config.json");
    cfg.validate().expect("validate");
    assert_eq!(cfg.model_type, "locateanything");
    assert_eq!(cfg.text_config.num_hidden_layers, 36);
    assert_eq!(cfg.text_config.hidden_size, 2048);
    assert_eq!(cfg.vision_config.num_hidden_layers, 27);
    assert_eq!(cfg.vision_config.hidden_size, 1152);
    assert_eq!(cfg.projector_input_dim(), 1152 * 4);
    assert_eq!(cfg.text_config.block_size, 6);
}

#[test]
fn locateanything_safetensors_layout() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let store = LocateAnythingWeightStore::open(&dir).expect("open checkpoint");
    store.validate_tensor_layout().expect("tensor counts");
    assert_eq!(store.keys().len(), EXPECTED_TENSOR_COUNT);

    let vision = store.load_vision_weights().expect("vision");
    assert!(vision.has(rlx_locateanything::LocateAnythingWeightPrefix::vision_patch_proj_w()));

    let proj = store.load_projector_weights().expect("projector");
    assert_eq!(proj.len(), 6);

    let lm = store.load_language_model_weights().expect("lm");
    assert!(lm.has(rlx_locateanything::LocateAnythingWeightPrefix::lm_embed_tokens()));

    assert_eq!(store.count_keys_with_prefix(PREFIX_VISION), 329);
    assert_eq!(store.count_keys_with_prefix(PREFIX_PROJECTOR), 6);
    assert_eq!(store.count_keys_with_prefix(PREFIX_LANGUAGE_MODEL), 435);
}

/// MoonViT + projector on a tiny image (CPU, real weights).
#[test]
fn locateanything_vision_encode_cpu() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(&dir)
        .device(rlx_runtime::Device::Cpu)
        .build()
        .expect("runner");
    let rgb = image::RgbImage::from_fn(128, 128, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    });
    let img = image::DynamicImage::ImageRgb8(rgb);
    let pre = runner.preprocess_image(&img).expect("preprocess");
    let out = runner.encode_vision_cached(&pre).expect("encode");
    let h = runner.cfg.text_config.hidden_size;
    assert_eq!(out.len() % h, 0);
    assert!(out.iter().all(|v| v.is_finite()));
}
