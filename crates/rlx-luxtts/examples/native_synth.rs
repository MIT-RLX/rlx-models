// Native (ort-free) LuxTTS voice-clone synthesis, timed. Isolates the native
// path (no whisper) to measure/verify it.
//   cargo run -p rlx-luxtts --features espeak --example native_synth
use std::path::PathBuf;
use std::time::Instant;

use rlx_luxtts::{InferOpts, LuxTts};
use rlx_runtime::{Device, parse_device};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(arg("--dir", "weights/tts/luxtts"));
    let text = arg("--text", "The quick brown fox jumps over the lazy dog.");
    let device: Device = parse_device(&arg("--device", "cpu"))?;
    let steps: usize = arg("--steps", "0").parse().unwrap_or(0);

    let prompt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&prompt_path)?;
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let prompt: Vec<f32> = r
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect();
    let prompt_text = "Hello from Kokoro. This is a test of speech synthesis in Rust.";

    let t0 = Instant::now();
    let tts = LuxTts::load_on(&dir, device)?;
    eprintln!("load: {:.2}s", t0.elapsed().as_secs_f32());

    let mut opts = InferOpts::default();
    if steps > 0 {
        opts.num_step = steps;
    }
    eprintln!("num_step={}", opts.num_step);

    let t1 = Instant::now();
    let audio = tts.synthesize(&text, &prompt, prompt_text, &opts)?;
    let dt = t1.elapsed().as_secs_f32();
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    let peak = audio.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    eprintln!(
        "synth: {:.2}s → {:.2}s audio, peak {:.3}, RTF {:.2}× on {device:?}",
        dt,
        secs,
        peak,
        secs / dt
    );
    let out = arg("--out", "");
    if !out.is_empty() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: tts.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&out, spec)?;
        for &s in &audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        eprintln!("wrote {out}");
    }
    Ok(())
}
