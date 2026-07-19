//! Text → StyleTTS2 (Kokoro native) → Whisper fox word coverage.
//!
//! ```bash
//! cargo test -p rlx-styletts2 --release --features apple-silicon --test whisper_roundtrip -- --nocapture
//! ```

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_styletts2::{STYLETTS2_SAMPLE_RATE, StyleTTS2, default_model_dir, peak_amplitude};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_KOKORO_DIR") {
        let p = PathBuf::from(d);
        return has_split(&p).then_some(p);
    }
    let p = default_model_dir();
    if has_split(&p) {
        return Some(p);
    }
    let alt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/kokoro-82m");
    has_split(&alt).then_some(alt)
}

fn has_split(model_dir: &std::path::Path) -> bool {
    model_dir.join("onnx/rlx-split/encoder.onnx").is_file()
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        return whisper_ready(&p).then_some(p);
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = cache.join(name);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    None
}

fn whisper_ready(dir: &std::path::Path) -> bool {
    dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file()
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[test]
fn test_styletts2_whisper_fox() {
    let Some(model) = model_dir() else {
        eprintln!("skip: Kokoro split bundle missing (just fetch-kokoro + split_kokoro.py)");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR or fetch .cache/whisper-*");
        return;
    };

    let tts = StyleTTS2::load(&model, Device::Cpu).expect("load StyleTTS2/Kokoro");
    let audio = tts.generate(TEXT, "af_heart", 1.0).expect("synthesize");
    assert!(audio.len() > STYLETTS2_SAMPLE_RATE as usize);
    assert!(peak_amplitude(&audio) > 0.05);

    let pcm_16k = resample_linear(&audio, STYLETTS2_SAMPLE_RATE, WHISPER_RATE as u32);
    let mut runner = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let transcript = runner.transcribe_greedy(&pcm_16k).expect("transcribe");
    let lower = transcript.to_lowercase();
    let hits = FOX_WORDS.iter().filter(|w| lower.contains(*w)).count();
    eprintln!("reference: {TEXT}");
    eprintln!("whisper:   {transcript}");
    eprintln!("fox:       {hits}/6");
    assert!(
        hits >= 5,
        "StyleTTS2 whisper fox {hits}/6 too low.\nref: {TEXT}\ngot: {transcript}"
    );
}
