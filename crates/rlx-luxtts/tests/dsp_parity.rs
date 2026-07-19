// Validate the Rust VocosFbank mel + ISTFT against Python (torchaudio/torch)
// goldens. Fixtures generated once from a 1s clip (see tests/fixtures/).

use std::path::PathBuf;

use rlx_luxtts::dsp::{VocosFbank, istft};

fn fx(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_f32(name: &str) -> Vec<f32> {
    std::fs::read(fx(name))
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let maxabs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    (dot / (na * nb + 1e-12), maxabs)
}

#[test]
fn mel_matches_torchaudio_golden() {
    let wav = read_f32("golden_wav_in.f32");
    if wav.is_empty() {
        eprintln!("skip: missing golden fixtures");
        return;
    }
    let golden = read_f32("golden_mel.f32");
    let (mel, t) = VocosFbank::new().log_mel(&wav);
    eprintln!("mel T={t}, golden {} vs ours {}", golden.len(), mel.len());
    let (cos, maxabs) = cosine(&mel, &golden);
    eprintln!("mel parity: cosine={cos:.6}, max_abs={maxabs:.4}");
    assert!(cos > 0.999, "mel cosine {cos} too low");
    assert!(maxabs < 0.2, "mel max_abs {maxabs} too high");
}

#[test]
fn istft_matches_torch_golden() {
    let real = read_f32("golden_real.f32");
    let imag = read_f32("golden_imag.f32");
    if real.is_empty() {
        eprintln!("skip: missing golden fixtures");
        return;
    }
    let golden = read_f32("golden_istft.f32");
    let t = real.len() / 513;
    let out = istft(&real, &imag, t);
    eprintln!("istft len {} vs golden {}", out.len(), golden.len());
    let (cos, maxabs) = cosine(&out, &golden);
    eprintln!("istft parity: cosine={cos:.6}, max_abs={maxabs:.4}");
    assert!(cos > 0.999, "istft cosine {cos} too low");
    assert!(maxabs < 1e-2, "istft max_abs {maxabs} too high");
}
