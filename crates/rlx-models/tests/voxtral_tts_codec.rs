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

//! Env-gated codec decode test against real weights.

use rlx_voxtral_tts::{CodecDecoder, VoxtralTtsConfig, VoxtralTtsWeightStore};
use std::path::PathBuf;

#[test]
fn codec_decoder_loads_from_checkpoint() {
    let dir = match std::env::var("RLX_VOXTRAL_TTS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("skip codec_decoder_loads_from_checkpoint: set RLX_VOXTRAL_TTS_DIR");
            return;
        }
    };
    let store = VoxtralTtsWeightStore::open(&dir).expect("open store");
    let cfg = VoxtralTtsConfig::from_model_dir(store.model_dir()).expect("config");
    let tensors = store
        .tensor_snapshot(rlx_voxtral_tts::load::PREFIX_CODEC)
        .expect("codec tensors");
    let decoder = CodecDecoder::from_tensors(
        rlx_voxtral_tts::load::PREFIX_CODEC,
        &tensors,
        &cfg.audio_config.codec_args,
    )
    .expect("build decoder");
    // Minimal single-frame decode (semantic token 2 = codebook 0, acoustic zeros + offset).
    let codes = vec![2u32; 37];
    let pcm = decoder.decode_codes(&codes, 1).expect("decode");
    assert!(!pcm.is_empty(), "expected non-empty PCM");
}
