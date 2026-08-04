// Native-path generate → WAV, for whisper round-trip validation (no ort).
use rlx_runtime::Device;
use rlx_supertonic::{InferOpts, Supertonic, Voice};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/st_native.wav".into());
    let text = std::env::var("ST_TEXT").unwrap_or_else(|_| {
        "The quick brown fox jumps over the lazy dog near the river bank.".into()
    });
    let tts = Supertonic::load_on(&dir, Device::Cpu)?;
    let voice = Voice::load(&dir.join("voice_styles/F1.json"))?;
    let opts = InferOpts {
        total_step: 8,
        speed: 1.0,
        seed: 42,
    };
    let audio = tts.synthesize(&text, "en", &voice, &opts)?;
    let sr = tts.sample_rate();
    eprintln!(
        "samples={} sr={sr} peak={:.3}",
        audio.len(),
        rlx_supertonic::peak_amplitude(&audio)
    );

    // Minimal PCM16 mono WAV writer.
    let mut f = std::fs::File::create(&out)?;
    let n = audio.len();
    let byte_rate = sr * 2;
    let data_bytes = (n * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_bytes).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&sr.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    for &s in &audio {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    eprintln!("wrote {out}");
    Ok(())
}
