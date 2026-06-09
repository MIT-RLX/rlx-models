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

//! Env-gated native synthesis after clone weights are present.

use rlx_runtime::Device;
use rlx_voxtral_tts::backbone::NativeTtsEngine;
use rlx_voxtral_tts::codec::encoder::load_mono_wav;
use rlx_voxtral_tts::generation::GenerationConfig;
use rlx_voxtral_tts::options::VoxtralTtsOptions;
use rlx_voxtral_tts::speech_tokenizer::SpeechTokenizer;
use rlx_voxtral_tts::{VoxtralTtsWeightStore, encode_reference_wav};
use rlx_voxtral_tts_train::audio_metrics::{cosine_similarity, mel_similarity};
use std::path::PathBuf;

#[test]
fn synthesize_with_cloned_voice_env_gated() {
    let model_dir = match std::env::var("RLX_VOXTRAL_TTS_DIR") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => {
            eprintln!("skip synthesize_with_cloned_voice_env_gated (RLX_VOXTRAL_TTS_DIR unset)");
            return;
        }
    };
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_RIG").ok().as_deref() != Some("1") {
        eprintln!("skip synthesize_with_cloned_voice_env_gated (RLX_VOXTRAL_TTS_TRAIN_RIG!=1)");
        return;
    }
    let ref_wav = std::env::var("RLX_VOXTRAL_TTS_REF_WAV")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    let Some(ref_wav) = ref_wav else {
        eprintln!("skip synthesize_with_cloned_voice_env_gated (RLX_VOXTRAL_TTS_REF_WAV unset)");
        return;
    };

    let store = VoxtralTtsWeightStore::open(&model_dir).expect("open store");
    let cfg =
        rlx_voxtral_tts::config::VoxtralTtsConfig::from_model_dir(&model_dir).expect("config");
    let voice = encode_reference_wav(&store, &cfg, &ref_wav, "neutral_female").expect("encode");
    let options = VoxtralTtsOptions {
        device: Device::Cpu,
        eager_lm: true,
        eager_acoustic: true,
    };
    let mut engine = NativeTtsEngine::open(&store, &cfg, &options).expect("engine");
    let tok = SpeechTokenizer::from_model_dir(&model_dir).expect("tokenizer");
    let ids = tok
        .encode_speech("Hello.", "neutral_female")
        .expect("prompt");
    let pcm = engine
        .synthesize(&ids, &voice, &GenerationConfig::default())
        .expect("synthesize");
    assert!(pcm.len() > cfg.audio_config.codec_args.sampling_rate / 2);
    let peak = pcm.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(peak > 1e-4, "silent synthesis");

    let rate = cfg.audio_config.codec_args.sampling_rate as u32;
    let ref_pcm = load_mono_wav(&ref_wav, rate).expect("load ref pcm");
    let mel_sim = mel_similarity(&ref_pcm, &pcm);
    eprintln!("[rig] mel_similarity={mel_sim:.4}");
    assert!(mel_sim > 0.05, "mel similarity too low ({mel_sim:.4})");

    let min_len = ref_pcm.len().min(pcm.len());
    if min_len > rate as usize {
        let pcm_corr = cosine_similarity(&ref_pcm[..min_len], &pcm[..min_len]);
        eprintln!("[rig] pcm_cosine={pcm_corr:.4}");
        assert!(pcm_corr > 0.0, "negative pcm correlation");
    }

    let voice_sim = cosine_similarity(&voice.data, &voice.data);
    assert!((voice_sim - 1.0).abs() < 1e-5);
}
