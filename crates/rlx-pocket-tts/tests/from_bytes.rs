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

/// Streamed, the audio comes in decoder blocks that add up to what
/// `generate` returns, the first long before the end.
#[test]
fn streaming_gives_the_same_samples_in_blocks() {
    let Ok(root) = std::env::var("POCKET_TTS_ROOT") else {
        eprintln!("skipping: POCKET_TTS_ROOT not set");
        return;
    };
    let language = std::env::var("POCKET_TTS_LANGUAGE").unwrap_or_else(|_| "french".into());
    let mut tts = PocketTts::open(&root, &language, Device::Cpu).unwrap();
    let voice = tts.load_voice("estelle").unwrap();
    let text = "Les enfants jouent dans le jardin, et leur mère prépare le dîner.";
    let whole = tts.generate(text, &voice).unwrap();
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    let streamed = tts
        .generate_streaming(text, &voice, &mut |b| blocks.push(b.to_vec()))
        .unwrap();
    assert!(blocks.len() > 1, "{} blocks", blocks.len());
    assert!(
        blocks[0].len() < whole.samples.len() / 2,
        "the first block comes early"
    );
    assert_eq!(blocks.concat(), whole.samples, "the blocks are the audio");
    assert_eq!(streamed.samples, whole.samples);
}
