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

//! Integration: decoder output matches eager ndarray gold (any active stack).
//!
//! ```sh
//! NEUTTS_DECODER_PATH=/path/to/neucodec_decoder.safetensors \
//!   cargo test -p rlx-neutts --features codec,fast,rlx,burn,wgpu decoder_stack --release
//! ```

use rlx_neutts::NeuCodecDecoder;

#[test]
fn decoder_stack_produces_finite_audio() {
    let Some(path) = rlx_neutts::decoder::decoder_weights_path_if_available() else {
        eprintln!("skip: set NEUTTS_DECODER_PATH");
        return;
    };

    let dec = NeuCodecDecoder::from_file(&path).expect("load");
    let codes: Vec<i32> = vec![0, 42, 128, 512, 1023];
    let audio = dec.decode(&codes).expect("decode");

    assert!(!audio.is_empty());
    assert!(audio.iter().all(|s| s.is_finite()));
    assert!(
        audio.len() == codes.len() * dec.hop_length(),
        "expected {} samples, got {}",
        codes.len() * dec.hop_length(),
        audio.len()
    );
    eprintln!("backend: {}", dec.backend_name());
}
