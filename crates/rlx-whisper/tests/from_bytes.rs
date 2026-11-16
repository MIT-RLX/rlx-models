//! The runner built from bytes (as in a browser) hears what the one built
//! from files does. Runs where Whisper tiny is in `weights/asr/whisper-tiny`
//! and `RLX_WHISPER_TEST_WAV` names a 16 kHz WAV; skipped otherwise.

use std::path::Path;

#[test]
fn from_bytes_hears_what_from_files_does() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/asr/whisper-tiny");
    let Ok(wav) = std::env::var("RLX_WHISPER_TEST_WAV") else {
        return;
    };
    if !dir.join("model.safetensors").exists() {
        return;
    }
    let pcm = rlx_whisper::audio::load_wav_mono_f32(Path::new(&wav)).unwrap();
    let from_files = rlx_whisper::WhisperRunnerBuilder::default()
        .weights(dir.join("model.safetensors"))
        .language("fr")
        .build()
        .unwrap()
        .transcribe_greedy(&pcm)
        .unwrap();
    let config = rlx_whisper::config::WhisperConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let from_bytes = rlx_whisper::WhisperRunnerBuilder::default()
        .weights_bytes(std::fs::read(dir.join("model.safetensors")).unwrap())
        .tokenizer_bytes(std::fs::read(dir.join("tokenizer.json")).unwrap())
        .config(config)
        .language("fr")
        .build()
        .unwrap()
        .transcribe_greedy(&pcm)
        .unwrap();
    assert!(!from_files.trim().is_empty());
    assert_eq!(from_bytes, from_files);
    eprintln!("heard: {from_bytes}");
}
