//! ChatterBox native → Whisper fox word coverage.
//!
//! ```bash
//! just chatterbox-whisper
//! ```

use std::path::PathBuf;

use rlx_chatterbox::{DEFAULT_LOCAL_DIR, NativeChatterBox, SAMPLE_RATE, SynthOpts, peak_amplitude};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("RLX_CHATTERBOX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join(DEFAULT_LOCAL_DIR));
    p.join("onnx/speech_encoder.onnx").is_file().then_some(p)
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        return whisper_ready(&p).then_some(p);
    }
    let cache = root().join(".cache");
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

fn read_ref() -> (Vec<f32>, u32) {
    let p = root().join("crates/rlx-luxtts/tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).expect("prompt.wav");
    let sr = r.spec().sample_rate;
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    (
        r.samples::<i32>()
            .map(|s| s.unwrap() as f32 / max)
            .collect(),
        sr,
    )
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
fn test_chatterbox_whisper_fox() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: need weights/tts/chatterbox (just fetch-tts-validation-bundles)");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR or fetch .cache/whisper-*");
        return;
    };

    let tts = NativeChatterBox::load_on(&dir, Device::Cpu).expect("load ChatterBox");
    let (reference, ref_sr) = read_ref();
    let opts = SynthOpts {
        greedy: true,
        max_frames: 128,
        ..Default::default()
    };
    let audio = tts
        .synthesize(TEXT, &reference, ref_sr, &opts)
        .expect("synthesize");
    assert!(audio.len() > SAMPLE_RATE as usize / 2);
    assert!(peak_amplitude(&audio) > 0.03);

    let pcm_16k = resample_linear(&audio, SAMPLE_RATE, WHISPER_RATE as u32);
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
        "ChatterBox whisper fox {hits}/6 too low.\nref: {TEXT}\ngot: {transcript}"
    );
}
