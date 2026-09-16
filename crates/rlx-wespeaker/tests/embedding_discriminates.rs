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

//! The embedder must actually depend on its input.
//!
//! This is not a parity test and deliberately so. For a long time the native
//! path returned a byte-identical 256-d vector for speech, white noise and a
//! pure tone — the audio never reached the network — and every check in the
//! crate passed, because comparing a constant against itself on five backends
//! is perfectly self-consistent. The cause was in `rlx-onnx-import`: its conv
//! shape propagation kept only the last spatial axis, so a 3x3 stride-1 pad-1
//! conv on `[1, 1, 80, 148]` produced `[1, 32, 1, 148]` and the ResNet ran on a
//! one-bin spectrogram. Only the debug-build IR verifier objected
//! (`MatMul: matmul K mismatch: 19 vs 5120`); release shipped the constant.
//!
//! So: assert the property that was violated, not the numbers.

use rlx_wespeaker::{WeSpeaker, cosine, resolve_model_dir};
use std::path::Path;

const SR: usize = 16_000;

fn tone(hz: f32, seconds: f32) -> Vec<f32> {
    let n = (SR as f32 * seconds) as usize;
    (0..n)
        .map(|i| 0.3 * (i as f32 * hz * std::f32::consts::TAU / SR as f32).sin())
        .collect()
}

fn pseudo_noise(seconds: f32) -> Vec<f32> {
    let n = (SR as f32 * seconds) as usize;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

#[test]
fn different_audio_gives_different_embeddings() {
    let Some(dir) = resolve_model_dir(&[Path::new("weights/wespeaker-voxceleb-resnet34-LM")])
    else {
        eprintln!("no WeSpeaker model dir — skipping");
        return;
    };
    let mut spk = WeSpeaker::open(&dir).expect("open WeSpeaker");

    let a = spk.embed_pcm(&tone(220.0, 3.0)).expect("embed tone");
    let b = spk.embed_pcm(&pseudo_noise(3.0)).expect("embed noise");
    let c = spk.embed_pcm(&tone(880.0, 3.0)).expect("embed high tone");

    assert_ne!(a, b, "tone and noise produced the SAME embedding");
    assert_ne!(a, c, "220 Hz and 880 Hz produced the SAME embedding");

    // Not merely unequal — unrelated audio must be far apart. A graph that
    // leaks only a trace of its input would still pass `assert_ne!`.
    let cos = cosine(&a, &b);
    assert!(
        cos < 0.9,
        "a tone and white noise should not look like the same speaker (cosine {cos:.4})"
    );

    // And the embedder must be deterministic, or the comparison above is noise.
    let again = spk.embed_pcm(&tone(220.0, 3.0)).expect("re-embed");
    assert_eq!(a, again, "embedding is not reproducible for one input");
}
