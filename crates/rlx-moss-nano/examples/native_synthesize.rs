//! End-to-end NATIVE (rlx, no ORT) MOSS-TTS-Nano synthesis → WAV.
//! `RLX_MOSS_DIR=... TEXT="..." VOICE=Junhao MAXF=48 cargo run -p rlx-moss-nano --example native_synthesize`
use rlx_moss_nano::{MossNative, NativeOpts};
use rlx_runtime::Device;
use std::path::PathBuf;

fn parse_device(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let text = std::env::var("TEXT").unwrap_or("Hello, this is a native test.".into());
    let voice = std::env::var("VOICE").unwrap_or_else(|_| String::new());
    let maxf: usize = std::env::var("MAXF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device = parse_device(&std::env::var("DEVICE").unwrap_or_default());

    let t0 = std::time::Instant::now();
    eprintln!("device: {}", device.name());
    let m = MossNative::load_on(&dir, device)?;
    eprintln!(
        "loaded in {:.1}s; voices: {:?}",
        t0.elapsed().as_secs_f32(),
        &m.voice_names()[..3.min(m.voice_names().len())]
    );
    let voice = if voice.is_empty() {
        m.voice_names()[0].clone()
    } else {
        voice
    };
    let opts = NativeOpts {
        seed,
        max_frames: maxf,
        ..Default::default()
    };

    let t1 = std::time::Instant::now();
    let audio = m.synthesize(&text, &voice, &opts)?;
    let secs = audio.len() as f32 / (m.sample_rate() as f32 * m.channels() as f32);
    let peak = audio.iter().fold(0f32, |a, &x| a.max(x.abs()));
    eprintln!(
        "synthesized {:.2}s audio ({} samples, peak {peak:.3}) in {:.1}s (voice={voice}, text={text:?})",
        secs,
        audio.len(),
        t1.elapsed().as_secs_f32()
    );
    let out = std::env::var("OUT").unwrap_or("/tmp/moss_native.wav".into());
    m.write_wav(&audio, std::path::Path::new(&out))?;
    eprintln!("wrote {out}");
    Ok(())
}
