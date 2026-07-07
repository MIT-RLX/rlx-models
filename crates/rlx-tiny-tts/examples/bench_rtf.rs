//! Steady-state RTF benchmark: load the model once, synthesize the same text
//! several times on one `TinyTts` instance. The first call imports + compiles
//! each subgraph; subsequent calls hit the in-memory graph cache (as a real
//! long-lived service would), so they measure pure inference RTF.
//!
//! Run: cargo run -p rlx-tiny-tts --release --features mlx --example bench_rtf -- \
//!        weights/tiny-tts-rlx mlx "The quick brown fox ..."

use std::path::PathBuf;

use rlx_tiny_tts::{InferOpts, TinyTts};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let data = args.next().unwrap_or_else(|| "weights/tiny-tts-rlx".into());
    let dev = args.next().unwrap_or_else(|| "mlx".into());
    let text = args.next().unwrap_or_else(|| {
        "The quick brown fox jumps over the lazy dog while the sun sets slowly \
         behind the distant mountains."
            .into()
    });
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let device = rlx_runtime::parse_device(&dev).map_err(|e| anyhow::anyhow!("{e}"))?;

    let model = TinyTts::load_from_dir(&PathBuf::from(&data))?;
    let opts = InferOpts::from_config(model.config());

    // Warm the frontend (tokenizer/g2p/BERT load) so it doesn't skew the timing.
    let _ = model.text_to_ids(&text)?;

    println!("device={device:?} iters={iters}\ntext: {text:?}\n");
    for i in 0..iters {
        let t0 = std::time::Instant::now();
        let wav = model.synthesize_on(&text, device, &opts)?;
        let synth = t0.elapsed().as_secs_f32();
        let secs_audio = wav.samples.len() as f32 / wav.sample_rate as f32;
        let tag = if i == 0 { "cold (import+compile)" } else { "warm (in-mem cache)  " };
        println!(
            "iter {i} {tag}: {secs_audio:.2}s audio, {synth:.3}s synth → {:.1}× RT",
            secs_audio / synth.max(1e-6)
        );
    }
    Ok(())
}
