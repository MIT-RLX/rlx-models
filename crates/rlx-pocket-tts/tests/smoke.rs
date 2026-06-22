// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Smoke test — only runs if `POCKET_TTS_WEIGHTS` and `POCKET_TTS_TOKENIZER`
//! env vars are set and a `POCKET_TTS_VOICE_FILE` voice safetensors is given.
//! Skips silently otherwise so CI without weights does not fail.

use rlx_pocket_tts::{GenerationOptions, TtsModel};

#[test]
fn generate_short_utterance() {
    let weights = match std::env::var("POCKET_TTS_WEIGHTS") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: POCKET_TTS_WEIGHTS not set");
            return;
        }
    };
    let tokenizer = match std::env::var("POCKET_TTS_TOKENIZER") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: POCKET_TTS_TOKENIZER not set");
            return;
        }
    };
    let voice_path = match std::env::var("POCKET_TTS_VOICE_FILE") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: POCKET_TTS_VOICE_FILE not set");
            return;
        }
    };

    let model = TtsModel::open(&weights, &tokenizer).expect("load model");
    let voice = model.load_voice(&voice_path).expect("load voice");
    assert_eq!(voice.embed_dim(), 1024);

    let opts = GenerationOptions {
        max_frames: 12, // ~ 1 s of audio for a quick smoke check
        ..Default::default()
    };
    let audio = model.generate("Hello.", &voice, opts).expect("generate");
    assert!(!audio.samples.is_empty(), "empty audio");
    assert_eq!(audio.sample_rate, 24_000);
    eprintln!(
        "smoke: {} samples ({:.2}s)",
        audio.samples.len(),
        audio.duration_secs()
    );
}
