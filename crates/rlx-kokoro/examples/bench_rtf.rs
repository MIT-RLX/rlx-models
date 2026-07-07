//! Steady-state RTF benchmark for Kokoro-82M. Loads the model once, synthesizes
//! the same text several times on one session. The first call warms ONNX
//! Runtime; subsequent calls measure pure inference throughput (real-time
//! factor = seconds-of-audio / seconds-to-synthesize).
//!
//! Run:
//!   cargo run -p rlx-kokoro --release --example bench_rtf -- \
//!       weights/tts/kokoro-82m cpu af_heart 6

use rlx_kokoro::{Kokoro, config::SAMPLE_RATE};
use rlx_runtime::parse_device;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let data = args
        .next()
        .unwrap_or_else(|| "weights/tts/kokoro-82m".into());
    let dev = args.next().unwrap_or_else(|| "cpu".into());
    let voice = args.next().unwrap_or_else(|| "af_heart".into());
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let text = "The quick brown fox jumps over the lazy dog while the sun sets \
                slowly behind the distant mountains.";

    let device = parse_device(&dev).map_err(|e| anyhow::anyhow!("{e}"))?;
    let model = Kokoro::load_on(&PathBuf::from(&data), "model.onnx", device)?;

    println!(
        "device={device:?} ep={} voice={voice} iters={iters}\ntext: {text:?}\n",
        model.ort_ep()
    );
    let mut warm_rtf = Vec::new();
    for i in 0..iters {
        let t0 = std::time::Instant::now();
        let audio = model.generate_from_text(text, &voice, 1.0)?;
        let synth = t0.elapsed().as_secs_f32();
        let secs_audio = audio.len() as f32 / SAMPLE_RATE as f32;
        let rtf = secs_audio / synth.max(1e-6);
        let tag = if i == 0 {
            "cold (ort warmup)  "
        } else {
            warm_rtf.push(rtf);
            "warm               "
        };
        println!("iter {i} {tag}: {secs_audio:.2}s audio, {synth:.3}s synth → {rtf:.1}× RT");
    }
    if !warm_rtf.is_empty() {
        let avg = warm_rtf.iter().sum::<f32>() / warm_rtf.len() as f32;
        println!("\nwarm mean: {avg:.1}× real-time on {device:?}");
    }
    Ok(())
}
