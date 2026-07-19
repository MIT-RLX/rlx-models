//! Diagnose the Metal in-process graph-reuse regression: synthesize the same
//! text twice on one model instance, dumping the text_encoder tensors each run,
//! and report which one diverges between run 1 and run 2.
//!
//! Run: cargo run -p rlx-tiny-tts --release --features metal --example reuse_probe -- weights/tiny-tts-rlx metal

use std::path::PathBuf;

use rlx_tiny_tts::{InferOpts, TinyTts};

fn read_f32(dir: &str, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(format!("{dir}/{name}.f32")).unwrap_or_default();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f32::NAN;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "weights/tiny-tts-rlx".into());
    let dev = args.next().unwrap_or_else(|| "metal".into());
    let device = rlx_runtime::parse_device(&dev).map_err(|e| anyhow::anyhow!("{e}"))?;
    let text = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant mountains.";

    let model = TinyTts::load(&PathBuf::from(&dir))?;
    let opts = InferOpts::from_config(model.config());
    let base = std::env::temp_dir();
    let d1 = base.join("reuse_run1");
    let d2 = base.join("reuse_run2");
    std::fs::create_dir_all(&d1).ok();
    std::fs::create_dir_all(&d2).ok();

    unsafe { std::env::set_var("RLX_TTS_DUMP", d1.to_str().unwrap()) };
    let w1 = model.synthesize_on(text, device, &opts)?;
    unsafe { std::env::set_var("RLX_TTS_DUMP", d2.to_str().unwrap()) };
    let w2 = model.synthesize_on(text, device, &opts)?;

    let (s1, s2) = (d1.to_str().unwrap(), d2.to_str().unwrap());
    println!(
        "run1 samples={} run2 samples={}\n",
        w1.samples.len(),
        w2.samples.len()
    );
    for name in [
        "x_enc",
        "m_p",
        "logs_p",
        "x_mask_graph",
        "g_enc",
        "logw",
        "z",
        "dec_out",
    ] {
        let a = read_f32(s1, name);
        let b = read_f32(s2, name);
        if a.is_empty() && b.is_empty() {
            continue;
        }
        println!(
            "{name:14} run1_len={:6} run2_len={:6} cos={:.5}",
            a.len(),
            b.len(),
            cos(&a, &b)
        );
    }
    Ok(())
}
