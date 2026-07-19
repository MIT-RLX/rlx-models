//! NATIVE (rlx, no ORT) MOSS-TTS-Nano → Whisper round-trip: synthesize, transcribe,
//! report word coverage. `RLX_MOSS_DIR=... RLX_WHISPER_DIR=... MAXF=64 cargo run
//! -p rlx-moss-nano --example native_whisper`
use rlx_moss_nano::{MossNative, NativeOpts};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};
use std::path::PathBuf;

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let wd =
        PathBuf::from(std::env::var("RLX_WHISPER_DIR").unwrap_or(".cache/whisper-tiny".into()));
    let text =
        std::env::var("TEXT").unwrap_or("The quick brown fox jumps over the lazy dog.".into());
    let maxf: usize = std::env::var("MAXF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let tts = MossNative::load(&dir)?;
    let voice = tts.voice_names()[0].clone();
    let t0 = std::time::Instant::now();
    let audio = tts.synthesize(
        &text,
        &voice,
        &NativeOpts {
            seed: 0,
            max_frames: maxf,
            ..Default::default()
        },
    )?;
    let secs = audio.len() as f32 / (tts.sample_rate() as f32 * tts.channels() as f32);
    eprintln!(
        "synthesized {secs:.2}s in {:.0}s (peak {:.3})",
        t0.elapsed().as_secs_f32(),
        audio.iter().fold(0f32, |a, &x| a.max(x.abs()))
    );
    tts.write_wav(&audio, std::path::Path::new("/tmp/moss_native_whisper.wav"))?;

    // interleaved-stereo 48k → mono 16k
    let ch = tts.channels() as usize;
    let nfr = audio.len() / ch;
    let mono: Vec<f32> = (0..nfr)
        .map(|i| (0..ch).map(|c| audio[i * ch + c]).sum::<f32>() / ch as f32)
        .collect();
    let n = (nfr as u64 * WR as u64 / tts.sample_rate() as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * tts.sample_rate() as f64 / WR as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = mono[idx.min(mono.len() - 1)];
            let b = mono[(idx + 1).min(mono.len() - 1)];
            a + (b - a) * f
        })
        .collect();

    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let transcript = w.transcribe_greedy(&pcm)?;
    let (refs, heard) = (words(&text), words(&transcript));
    let hits = refs
        .iter()
        .filter(|x| heard.iter().any(|h| h == *x || h.contains(x.as_str())))
        .count();
    let cov = if refs.is_empty() {
        0.0
    } else {
        hits as f32 / refs.len() as f32
    };
    eprintln!("target:   {text}");
    eprintln!("whisper:  {transcript}");
    eprintln!("coverage: {cov:.2}");
    Ok(())
}
