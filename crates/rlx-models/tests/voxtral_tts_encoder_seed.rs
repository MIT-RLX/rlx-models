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

//! Env-gated encoder load from decoder-seeded weights (public checkpoints omit encoder).

use rlx_voxtral_tts::codec::encoder::CodecEncoder;
use rlx_voxtral_tts::codec::encoder_seed::seed_encoder_from_decoder;
use rlx_voxtral_tts::load::PREFIX_CODEC;
use rlx_voxtral_tts::{VoxtralTtsConfig, VoxtralTtsWeightStore};
use std::path::PathBuf;

#[test]
fn encoder_loads_from_decoder_seed_and_encodes_pcm() {
    let dir = match std::env::var("RLX_VOXTRAL_TTS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("skip encoder_loads_from_decoder_seed: set RLX_VOXTRAL_TTS_DIR");
            return;
        }
    };
    let store = VoxtralTtsWeightStore::open(&dir).expect("open store");
    let cfg = VoxtralTtsConfig::from_model_dir(store.model_dir()).expect("config");
    let mut tensors = store.tensor_snapshot(PREFIX_CODEC).expect("codec tensors");

    seed_encoder_from_decoder(&mut tensors, &cfg.audio_config.codec_args).expect("seed encoder");

    let encoder = CodecEncoder::from_tensors(PREFIX_CODEC, &tensors, &cfg.audio_config.codec_args)
        .expect("build encoder from seeded weights");

    let embed_tensors = store.tensor_snapshot_for_embed().expect("embed");
    let embed = rlx_voxtral_tts::backbone::embed::EmbeddingTables::from_tensors(
        &embed_tensors,
        &cfg.text_config,
        &cfg.audio_config.audio_model_args,
    )
    .expect("embed");

    let rate = cfg.audio_config.codec_args.sampling_rate;
    let n = rate / 2;
    let pcm: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin() * 0.2)
        .collect();

    let voice = encoder
        .encode_pcm_to_voice_embedding(&pcm, &embed, "seed_test")
        .expect("encode reference pcm");
    assert!(voice.n_tokens > 0, "expected at least one voice frame");
    assert_eq!(voice.data.len(), voice.n_tokens * voice.hidden);
}
