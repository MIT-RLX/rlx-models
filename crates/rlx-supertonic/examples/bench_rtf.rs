//! Steady-state RTF benchmark for Supertonic-3. Loads the 4 sessions once and
//! synthesizes the same text several times (real-time factor = audio-seconds /
//! synth-seconds).
//!
//! Run: cargo run -p rlx-supertonic --release --example bench_rtf -- \
//!        weights/tts/supertonic-3 cpu F1 8 6

use rlx_supertonic::{InferOpts, Supertonic, Voice};
use rlx_runtime::parse_device;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut a = std::env::args().skip(1);
    let dir = PathBuf::from(a.next().unwrap_or_else(|| "weights/tts/supertonic-3".into()));
    let dev = a.next().unwrap_or_else(|| "cpu".into());
    let voice = a.next().unwrap_or_else(|| "F1".into());
    let steps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let text = "The quick brown fox jumps over the lazy dog while the sun sets \
                slowly behind the distant mountains.";

    let device = parse_device(&dev).map_err(|e| anyhow::anyhow!("{e}"))?;
    let tts = Supertonic::load_on(&dir, device)?;
    let v = Voice::load(&dir.join("voice_styles").join(format!("{voice}.json")))?;
    let opts = InferOpts { total_step: steps, ..Default::default() };

    println!("device={device:?} ep={} voice={voice} steps={steps} iters={iters}\n", tts.ort_ep());
    let mut warm = Vec::new();
    for i in 0..iters {
        let t0 = std::time::Instant::now();
        let audio = tts.synthesize(text, "en", &v, &opts)?;
        let synth = t0.elapsed().as_secs_f32();
        let secs = audio.len() as f32 / tts.sample_rate() as f32;
        let rtf = secs / synth.max(1e-6);
        let tag = if i == 0 { "cold" } else { warm.push(rtf); "warm" };
        println!("iter {i} {tag}: {secs:.2}s audio, {synth:.3}s synth → {rtf:.1}× RT");
    }
    if !warm.is_empty() {
        println!("\nwarm mean: {:.1}× real-time on {device:?}", warm.iter().sum::<f32>() / warm.len() as f32);
    }
    Ok(())
}
