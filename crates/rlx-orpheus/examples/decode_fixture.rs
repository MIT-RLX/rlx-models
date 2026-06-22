//! Decode golden SNAC fixture to WAV (no LM).
//!
//! Set `ORPHEUS_SNAC_COREML=1` to exercise the native RLX Ane SNAC path.
use std::path::PathBuf;

fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    use anyhow::Context;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in samples {
        let int_s = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(int_s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let snac = std::env::var("ORPHEUS_SNAC_PATH")
        .unwrap_or_else(|_| "/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors".into());
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/orpheus-demos/golden-hi.wav".into()),
    );
    let (count, codes): (usize, Vec<i32>) = if let Ok(path) = std::env::var("ORPHEUS_CODES_PATH") {
        let text = std::fs::read_to_string(&path)?;
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let count: usize = lines.next().unwrap().trim().parse()?;
        let codes: Vec<i32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        (count, codes)
    } else {
        let fixture = include_str!("../tests/fixtures/orpheus_hi_codes.txt");
        let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
        let count: usize = lines.next().unwrap().trim().parse()?;
        let codes: Vec<i32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        (count, codes)
    };
    let _count = count;
    let coreml = std::env::var("ORPHEUS_SNAC_COREML")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE"));
    let snac_dec = rlx_orpheus::SnacBackend::open(
        std::path::Path::new(&snac),
        rlx_orpheus::SnacLoadOptions { coreml },
    )?;
    let samples = rlx_orpheus::decode_orpheus_codes(&snac_dec, &codes)?;
    write_wav(&out, &samples, rlx_orpheus::SAMPLE_RATE)?;
    eprintln!(
        "wrote {} ({} samples, {:.2}s)",
        out.display(),
        samples.len(),
        samples.len() as f64 / rlx_orpheus::SAMPLE_RATE as f64
    );
    Ok(())
}
