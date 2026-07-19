use half::f16;
use rlx_f5tts::DEFAULT_LOCAL_DIR;
use rlx_f5tts::config::{HOP_LENGTH, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&ref_path)?;
    let sr = r.spec().sample_rate;
    let m = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let reference: Vec<f32> = r.samples::<i32>().map(|s| s.unwrap() as f32 / m).collect();
    let text = "The quick brown fox jumps over the lazy dog.";
    let ref_text =
        normalize_ref_text("Hello from Kokoro. This is a test of speech synthesis in Rust.");
    let reference = preprocess_ref_audio(&reference, sr);
    let vocab = Vocab::load(&model)?;
    let text_ids = encode(&ref_text, text, &vocab);
    let n = reference.len();
    let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
    let ref_tl = text_len(&ref_text).max(1) as f64;
    let gen_tl = text_len(text) as f64;
    let max_duration = (ref_audio_len + (ref_audio_len / ref_tl * gen_tl)) as usize;

    std::fs::write(
        "/tmp/probe_audio_f32.bin",
        reference
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(
        "/tmp/probe_audio_f16.bin",
        reference
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(
        "/tmp/probe_text_ids_i32.bin",
        text_ids
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    eprintln!(
        "n={n} t={} max_duration={max_duration} text_ids={:?}",
        text_ids.len(),
        text_ids
    );
    Ok(())
}
