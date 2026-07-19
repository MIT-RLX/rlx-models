//! Fox sentence Whisper round-trip for MioTTS.

use std::path::PathBuf;

use rlx_miotts::{GenerateOpts, MioSession, default_codec_dir, default_model_dir};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const FOX: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_WHISPER_DIR") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["whisper-tiny", "whisper-tiny.en", "whisper-base.en"] {
        let p = root.join(".cache").join(name);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

fn resample_24k_to_16k(pcm: &[f32]) -> Vec<f32> {
    // Simple linear resample 24000 → 16000 (3:2)
    let out_len = pcm.len() * 2 / 3;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * 3.0 / 2.0;
        let j = src.floor() as usize;
        let f = (src - j as f64) as f32;
        let a = pcm.get(j).copied().unwrap_or(0.0);
        let b = pcm.get(j + 1).copied().unwrap_or(a);
        out.push(a * (1.0 - f) + b * f);
    }
    out
}

#[test]
fn test_miotts_whisper_fox() {
    let model = default_model_dir();
    let codec = default_codec_dir();
    if !model.join("model.safetensors").is_file() || !codec.join("decoder_body.onnx").is_file() {
        eprintln!("skip: weights missing (just fetch-miotts + export script)");
        return;
    }
    let Some(wdir) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR or drop whisper-tiny in .cache");
        return;
    };

    let mut session = MioSession::open(&model, &codec, Device::Cpu).expect("open");
    let opts = GenerateOpts {
        seed: 42,
        max_new_tokens: 400,
        preset: "en_female".into(),
    };
    let result = session.synthesize(FOX, &opts).expect("synth");
    eprintln!(
        "synth: {} samples @ {} Hz, {} codes, peak={:.3}",
        result.samples.len(),
        result.sample_rate,
        result.content_codes.len(),
        result
            .samples
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
    );
    assert!(result.content_codes.len() > 10, "too few speech codes");

    let pcm16 = resample_24k_to_16k(&result.samples);
    assert_eq!(WHISPER_RATE, 16_000);

    let mut whisper = WhisperRunner::builder()
        .weights(wdir.join("model.safetensors"))
        .config_path(wdir.join("config.json"))
        .tokenizer_path(wdir.join("tokenizer.json"))
        .language("en")
        .build()
        .expect("whisper");
    let transcript = whisper.transcribe_greedy(&pcm16).expect("asr");
    let lower = transcript.to_lowercase();
    let hits = FOX_WORDS.iter().filter(|w| lower.contains(*w)).count();
    eprintln!("whisper heard: {transcript:?}  hits={hits}/6");
    assert!(
        hits >= 5,
        "fox coverage {hits}/6 too low (heard {transcript:?})"
    );
}
