// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! WeSpeaker ResNet34-LM speaker embeddings (256-d) on **native RLX**.
//!
//! Default path: Kaldi-style fbank → TinyModel (`graphs/wespeaker.rlxp` or
//! `onnx/wespeaker.onnx` imported via `rlx-onnx-import`) on any RLX device.
//!
//! The graph is baked for a fixed **148** frame window (~1.5 s @ 16 kHz,
//! 25 ms / 10 ms fbank). Shorter/longer windows are padded or truncated.
//!
//! Optional `onnx` feature keeps ORT for parity against the upstream packed
//! ONNX — not used on the ship path.

pub mod fbank;
mod native;

#[cfg(feature = "onnx")]
mod ort_ref;

pub use fbank::log_mel_fbank;
pub use native::{FIXED_FRAMES, WeSpeaker, resolve_model_dir};

#[cfg(feature = "onnx")]
pub use ort_ref::OrtWeSpeaker;

/// Embedding dimensionality of WeSpeaker ResNet34-LM.
pub const EMBED_DIM: usize = 256;

/// Cosine similarity of two L2-normalized (or raw) vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(all(test, feature = "native"))]
mod native_tests {
    use super::*;
    use rlx_runtime::Device;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest.join("../../weights/wespeaker-voxceleb-resnet34-LM"),
            PathBuf::from("/Volumes/FOUR/weights/wespeaker-voxceleb-resnet34-LM"),
            PathBuf::from("/Users/Shared/translator/models/wespeaker-voxceleb-resnet34-LM"),
        ];
        candidates.into_iter().find(|p| {
            p.join("graphs/wespeaker.rlxp").is_file() || p.join("onnx/wespeaker.onnx").is_file()
        })
    }

    #[test]
    fn embed_tone_native_cpu() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: WeSpeaker RLX weights not found");
            return;
        };
        let mut spk = WeSpeaker::open_on(&dir, Device::Cpu).expect("open");
        let sr = 16_000usize;
        let n = sr + sr / 2;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 180.0 * i as f32 / sr as f32).sin() * 0.1)
            .collect();
        let emb = spk.embed_pcm(&pcm).expect("embed");
        assert_eq!(emb.len(), EMBED_DIM);
        let nrm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-3, "nrm={nrm}");
    }
}
