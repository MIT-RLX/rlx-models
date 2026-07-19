use anyhow::{Context, Result};
use rlx_kokoro::{Device, Kokoro};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model = PathBuf::from("weights/tts/kokoro-82m");
    let tts = Kokoro::load_on(&model, "model.onnx", Device::Cpu).context("load ort")?;
    let text = "The quick brown fox jumps over the lazy dog.";
    let audio = tts.generate_from_text(text, "af_heart", 1.0)?;
    let peak = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "ort samples={} peak={peak:.4} dur={:.2}s",
        audio.len(),
        audio.len() as f32 / 24000.0
    );
    tts.write_wav(&audio, std::path::Path::new("/tmp/kokoro_ort_rust.wav"))?;
    Ok(())
}
