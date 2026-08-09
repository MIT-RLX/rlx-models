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

// Env-gated: real GGUF + mmproj multimodal prefill (+ optional decode).
//
//   QWEN35_GGUF_PATH=/path/to/model.gguf \
//   QWEN35_MMPROJ_PATH=/path/to/mmproj.gguf \
//   QWEN35_VLM_IMAGE=/path/to/image.jpg \  # optional; synthetic RGB if omitted
//     cargo test -p rlx-models --test qwen35_vlm_gguf_quick_check --features qwen35-vlm --release -- --nocapture

#[path = "qwen35_gguf_support.rs"]
mod support;

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen35::{MEDIA_MARKER, MultimodalPrompt, Qwen35VisionEncoder};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use support::gguf_path;

const DEFAULT_MMPROJ: &str = "/tmp/rlx-models/Qwen3.5-0.8B-mmproj.gguf";

fn mmproj_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_MMPROJ_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = PathBuf::from(DEFAULT_MMPROJ);
    path.is_file().then_some(path)
}

fn load_rgb(path: Option<&Path>) -> (Vec<u8>, usize, usize) {
    if let Some(p) = path {
        #[cfg(feature = "qwen35-vlm")]
        {
            let (rgb, w, h) =
                rlx_models::qwen35::load_rgb_image(p.to_str().expect("utf-8 image path"))
                    .expect("load image");
            return (rgb, w, h);
        }
        #[cfg(not(feature = "qwen35-vlm"))]
        {
            let _ = p;
            panic!("QWEN35_VLM_IMAGE set but rebuild with feature qwen35-vlm");
        }
    }
    let w = 224;
    let h = 224;
    let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 251) as u8).collect();
    (rgb, w, h)
}

#[test]
fn qwen35_real_gguf_mmproj_multimodal_prefill() {
    let weights = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_vlm_gguf_quick_check: set QWEN35_GGUF_PATH");
            return;
        }
    };
    let mmproj = match mmproj_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_vlm_gguf_quick_check: set QWEN35_MMPROJ_PATH");
            return;
        }
    };

    let image_path = std::env::var("QWEN35_VLM_IMAGE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let (rgb, img_w, img_h) = load_rgb(image_path.as_deref());

    let prompt = format!("Describe this image. {MEDIA_MARKER}");
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(&weights)
        .mmproj(&mmproj)
        .device(Device::Cpu)
        .batch(1)
        .packed_weights(true)
        .max_seq(512)
        .last_logits_only(true)
        .build()
        .expect("VLM runner");

    assert!(runner.has_vision());

    let seed = runner
        .prefill_multimodal(&prompt, &rgb, img_w, img_h, None)
        .expect("multimodal prefill");
    assert!(!seed.trunk_logits.is_empty());
    assert!(seed.trunk_logits.iter().all(|v| v.is_finite()));

    let _ = runner
        .decode_get_logits(1)
        .expect("one decode step after multimodal prefill");

    eprintln!(
        "qwen35 VLM gguf quick check ok: weights={} mmproj={} image={}x{} logits={}",
        weights.display(),
        mmproj.display(),
        img_w,
        img_h,
        seed.trunk_logits.len()
    );
}

/// Assembly-only path when tokenizer is unavailable (manual vision + fake tok ids).
#[test]
fn qwen35_real_gguf_mmproj_assembled_prefill() {
    let weights = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let mmproj = match mmproj_path() {
        Some(p) => p,
        None => return,
    };

    let (rgb, img_w, img_h) = load_rgb(None);
    let mut enc = Qwen35VisionEncoder::from_mmproj(&mmproj, img_w, img_h, Device::Cpu)
        .expect("vision encoder");
    let vision = enc.encode_rgb(&rgb, img_w, img_h).expect("encode");

    let mut loader =
        rlx_models::weight_loader::GgufLoader::from_file(weights.to_str().expect("utf-8 weights"))
            .expect("loader");
    let cfg = rlx_models::qwen35::Qwen35Config::from_gguf(loader.file()).expect("cfg");
    let w = rlx_models::qwen35::Qwen35Weights::from_loader(&mut loader, &cfg).expect("weights");

    let prompt = format!("before{MEDIA_MARKER}after");
    let mm = MultimodalPrompt {
        prompt: &prompt,
        vision: &vision,
    };
    let prefill = mm
        .assemble(
            |s| Ok(s.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect()),
            &w.token_embd,
            cfg.hidden_size,
            0,
        )
        .expect("assemble");

    let mut runner = Qwen35RunnerBuilder::default()
        .weights(&weights)
        .mmproj(&mmproj)
        .device(Device::Cpu)
        .batch(1)
        .packed_weights(true)
        .max_seq(prefill.seq.len() + 8)
        .last_logits_only(true)
        .build()
        .expect("runner");

    let seed = runner
        .prefill_from_assembled(prefill)
        .expect("assembled prefill");
    assert!(seed.trunk_logits.iter().all(|v| v.is_finite()));
    eprintln!(
        "qwen35 VLM assembled prefill ok: seq logits={}",
        seed.trunk_logits.len()
    );
}
