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

//! Codec parity against `scripts/fish_reference.py codec`.
//!
//! Needs `RLX_FISH_DIR` (a directory with `codec.pth`) and `RLX_FISH_REF`
//! (the script's `--out`); skips otherwise. `RLX_FISH_DEVICE` picks the
//! backend (default `cpu`).

mod common;

use common::{Fixture, close, setup};
use rlx_fish::FishCodec;

#[test]
fn codec_decode_matches_reference() {
    let Some((dir, reference, device)) = setup("codec.safetensors") else {
        return;
    };
    let fx = Fixture::open(&reference.join("codec.safetensors"));
    let (codes, shape) = fx.i64("codes");
    let frames = shape[1];
    let rows: Vec<Vec<u32>> = codes
        .chunks(frames)
        .map(|r| r.iter().map(|&c| c as u32).collect())
        .collect();
    let mut codec = FishCodec::open(&dir, device).unwrap();

    // The summed codebook vectors are what the post transformer sees.
    let z = codec.dequantize(&rows).unwrap();
    let wav = codec.decode_latent(&z, frames).unwrap();
    let (want, _) = fx.f32("wav");
    assert_eq!(wav.len(), want.len());
    close("wav", &wav, &want, 2e-3);
}

#[test]
fn codec_encode_matches_reference() {
    let Some((dir, reference, device)) = setup("codec.safetensors") else {
        return;
    };
    let fx = Fixture::open(&reference.join("codec.safetensors"));
    let (pcm, _) = fx.f32("enc_pcm");
    let mut codec = FishCodec::open(&dir, device).unwrap();

    let latent = codec.encode_latent(&pcm).unwrap();
    let (want, shape) = fx.f32("enc_latent");
    // Reference latent is `[C, T]`; ours is rows `[T, C]`.
    let (c, t) = (shape[0], shape[1]);
    let mut want_rows = vec![0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            want_rows[ti * c + ci] = want[ci * t + ti];
        }
    }
    close("encode latent", &latent, &want_rows, 1e-3);

    let codes = codec.encode(&pcm).unwrap();
    let (want_codes, shape) = fx.i64("enc_codes");
    let frames = shape[1];
    assert_eq!(codes[0].len(), frames);
    let mut same = 0;
    for (cb, row) in codes.iter().enumerate() {
        for (t, &c) in row.iter().enumerate() {
            same += usize::from(c as i64 == want_codes[cb * frames + t]);
        }
    }
    let total = codes.len() * frames;
    eprintln!("encode codes: {same}/{total} identical");
    // A nearest-neighbour search can flip on a near-tie, and a flip in one
    // level changes the residual every later level sees; allow a little.
    assert!(same * 100 >= total * 98, "codes {same}/{total}");
}
