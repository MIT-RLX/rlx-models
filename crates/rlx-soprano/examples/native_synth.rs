// End-to-end soprano with the NATIVE rlx-qwen3 backbone (ort-free) + the ONNX
// Vocos decoder → WAV. Validates the native backbone fixes soprano on CUDA.
//
//   RLX_SOPRANO_DIR=$HOME/rlx-models/weights/tts/soprano \
//   SOPRANO_BACKBONE_ST=.../ekwek/model.safetensors \
//   cargo run -p rlx-soprano --release --features cuda --example native_synth -- \
//     cuda "The quick brown fox jumps over the lazy dog." /tmp/soprano_native.wav

use anyhow::{Context, Result};
use rlx_soprano::native_qwen3::SopranoQwen3;
use rlx_soprano::{NativeSoprano, parse_device};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dev = args.get(1).cloned().unwrap_or_else(|| "cpu".into());
    let text = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/tmp/soprano_native.wav".into());
    let max_new: usize = std::env::var("SOPRANO_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let dir = std::env::var("RLX_SOPRANO_DIR").unwrap_or_else(|_| "weights/tts/soprano".into());
    let st = std::env::var("SOPRANO_BACKBONE_ST").context("set SOPRANO_BACKBONE_ST")?;
    let device = parse_device(&dev).context("device")?;

    let onnx = NativeSoprano::open(&dir, device).context("open onnx (tokenizer + decoder)")?;
    let native =
        SopranoQwen3::open(std::path::Path::new(&st), device).context("open native backbone")?;

    let ids = onnx.encode_prompt(&text)?;
    eprintln!(
        "[native_synth] device={dev} prompt='{text}' ({} tok) max_new={max_new}",
        ids.len()
    );
    let t0 = std::time::Instant::now();
    let (latents, toks) = native.generate_latents_greedy(&ids, max_new)?;
    eprintln!(
        "[native_synth] generated {} audio tokens, {} latents in {:.1}s",
        toks.len(),
        latents.len(),
        t0.elapsed().as_secs_f32()
    );
    anyhow::ensure!(!latents.is_empty(), "no latents produced");

    // ONNX Vocos decoder (clean-named weights) — swapped for a native one in M3b.
    let pcm = onnx
        .decode_latents(&latents, true)
        .context("decode latents")?;
    let peak = pcm.iter().fold(0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "[native_synth] pcm {} samples ({:.2}s) peak={peak:.3}",
        pcm.len(),
        pcm.len() as f32 / 32000.0
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 32_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&out, spec)?;
    for &s in &pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    eprintln!("[native_synth] wrote {out}");
    Ok(())
}
