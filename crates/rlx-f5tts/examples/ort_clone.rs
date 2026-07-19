//! ORT reference clone (needs `--features onnx`). Env: RLX_F5TTS_DIR, NFE, REF_SECS.
use rlx_f5tts::{F5Tts, InferOpts};
use rlx_runtime::Device;
use std::path::PathBuf;

const REF_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::var("RLX_F5TTS_DIR").unwrap_or_else(|_| "weights/tts/f5tts".into()),
    );
    let tts = F5Tts::load_on(&dir, Device::Cpu)?;
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&p)?;
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let mut refa: Vec<f32> = r
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect();
    if let Ok(secs) = std::env::var("REF_SECS") {
        let n = (secs.parse::<f32>().unwrap_or(2.0) * 24000.0) as usize;
        refa.truncate(n.min(refa.len()));
    }
    let nfe: usize = std::env::var("NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let t0 = std::time::Instant::now();
    let audio = tts.synthesize(TEXT, &refa, REF_TEXT, &InferOpts { nfe, speed: 1.0 })?;
    let dt = t0.elapsed().as_secs_f32();
    let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let secs = audio.len() as f32 / 24000.0;
    eprintln!(
        "ort f5: {} samples ({secs:.2}s) peak={peak:.3} in {dt:.1}s (nfe={nfe}, rtf={:.2})",
        audio.len(),
        dt / secs
    );
    tts.write_wav(&audio, std::path::Path::new("tmp/f5tts_wavs/ort_ref.wav"))?;
    Ok(())
}
