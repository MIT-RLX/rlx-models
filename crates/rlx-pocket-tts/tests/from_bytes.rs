// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! The model from bytes in memory (as a browser has it) says the same as
//! the model from files. Runs when `POCKET_TTS_ROOT` holds a checkout with
//! `languages/<POCKET_TTS_LANGUAGE>` (default french) and its `estelle` voice.

use rlx_pocket_tts::engine::PocketTts;
use rlx_runtime::Device;

#[test]
fn from_bytes_says_what_from_files_does() {
    let Ok(root) = std::env::var("POCKET_TTS_ROOT") else {
        eprintln!("skipping: POCKET_TTS_ROOT not set");
        return;
    };
    let language = std::env::var("POCKET_TTS_LANGUAGE").unwrap_or_else(|_| "french".into());
    let dir = std::path::Path::new(&root)
        .join("languages")
        .join(&language);
    let text = "Bonjour, comment allez-vous?";

    let mut files = PocketTts::open(&root, &language, Device::Cpu).unwrap();
    let voice = files.load_voice("estelle").unwrap();
    let want = files.generate(text, &voice).unwrap();

    let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
    let tokenizer = std::fs::read(dir.join("tokenizer.model")).unwrap();
    let mut memory = PocketTts::from_bytes(&language, weights, &tokenizer, Device::Cpu).unwrap();
    let voice = memory
        .voice_from_bytes(std::fs::read(dir.join("embeddings/estelle.safetensors")).unwrap())
        .unwrap();
    let got = memory.generate(text, &voice).unwrap();

    assert_eq!(got.sample_rate, want.sample_rate);
    assert_eq!(
        got.samples, want.samples,
        "the same audio, sample for sample"
    );
}
