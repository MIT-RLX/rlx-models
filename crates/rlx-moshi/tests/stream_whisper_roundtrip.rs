//! Streaming duplex: chunked PCM in → Moshi PCM out → WAV → Whisper ASR.
//!
//! ```sh
//! just fetch-moshi fetch-mimi fetch-whisper-base
//! RLX_MOSHI_STREAM_E2E=1 just test-moshi-stream-whisper
//! ```

use anyhow::{Context, Result, ensure};
use rlx_mimi::SAMPLE_RATE as MIMI_RATE;
use rlx_mimi::audio::{load_wav_mono, write_wav_mono};
use rlx_moshi::{
    FRAME_SAMPLES, GenerationConfig, MoshiSession, MoshiVariant, StreamCommand, StreamEvent,
    device_ready, spawn_duplex_stream, test_devices,
};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner, normalize_transcript};
use std::path::{Path, PathBuf};

const MIN_OUTPUT_SAMPLES: usize = 4800;
const MIN_PEAK: f32 = 1e-3;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn moshi_dir() -> Option<PathBuf> {
    std::env::var("RLX_MOSHI_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            let d = repo_root().join(".cache/moshiko");
            d.join("model.safetensors").is_file().then_some(d)
        })
}

fn mimi_dir() -> Option<PathBuf> {
    std::env::var("RLX_MIMI_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            let d = repo_root().join(".cache/mimi");
            d.join("model.safetensors").is_file().then_some(d)
        })
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    let cache = repo_root().join(".cache");
    for name in ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"] {
        let p = cache.join(name);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

fn input_wav() -> PathBuf {
    if let Ok(p) = std::env::var("RLX_MOSHI_STREAM_WAV") {
        return PathBuf::from(p);
    }
    repo_root().join("crates/rlx-qwen3-tts/examples/audio/ask_not.wav")
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn whisper_runner(dir: &Path, device: Device) -> Result<WhisperRunner> {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(device)
        .language("en")
        .build()
}

fn stream_duplex_to_pcm(device: Device, pcm_in: &[f32], max_steps: usize) -> Result<Vec<f32>> {
    let moshi_dir = moshi_dir().context("run `just fetch-moshi`")?;
    let mimi_dir = mimi_dir().context("run `just fetch-mimi`")?;
    let session = MoshiSession::open_on(&moshi_dir, &mimi_dir, MoshiVariant::Moshiko, device)?;
    let cfg = GenerationConfig {
        max_steps,
        ..GenerationConfig::default()
    };
    let handle = spawn_duplex_stream(session, "", cfg)?;
    let mut out_pcm = Vec::new();
    let feed_len = (max_steps * FRAME_SAMPLES).min(pcm_in.len());

    loop {
        match handle
            .event_rx
            .recv_timeout(std::time::Duration::from_secs(600))
        {
            Ok(StreamEvent::Ready) => break,
            Ok(StreamEvent::Error(e)) => anyhow::bail!("stream error: {e}"),
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                anyhow::bail!("timed out waiting for stream Ready")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("stream worker exited before Ready")
            }
        }
    }

    let mut feed_off = 0usize;
    while feed_off < feed_len {
        let end = (feed_off + FRAME_SAMPLES).min(feed_len);
        handle
            .cmd_tx
            .send(StreamCommand::Pcm(pcm_in[feed_off..end].to_vec()))?;
        feed_off = end;
        let mut got_step = false;
        while !got_step {
            match handle
                .event_rx
                .recv_timeout(std::time::Duration::from_secs(300))
            {
                Ok(StreamEvent::OutputPcm { samples, .. }) => out_pcm.extend(samples),
                Ok(StreamEvent::Step(_)) => got_step = true,
                Ok(StreamEvent::Error(e)) => anyhow::bail!("stream error: {e}"),
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    anyhow::bail!("timed out waiting for stream step")
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("stream worker exited mid-step")
                }
            }
        }
    }
    handle.cmd_tx.send(StreamCommand::Finish)?;
    while let Ok(ev) = handle.event_rx.recv() {
        match ev {
            StreamEvent::OutputPcm { samples, .. } => out_pcm.extend(samples),
            StreamEvent::Finished(_) => break,
            StreamEvent::Error(e) => anyhow::bail!("stream error: {e}"),
            _ => {}
        }
    }
    handle.stop();
    Ok(out_pcm)
}

fn streaming_roundtrip_on_device(device: Device) -> Result<()> {
    let whisper_dir = whisper_dir().context("run `just fetch-whisper-base`")?;
    let wav_in = input_wav();
    ensure!(wav_in.is_file(), "missing input wav {}", wav_in.display());

    let pcm_in = load_wav_mono(&wav_in, MIMI_RATE)?;
    ensure!(!pcm_in.is_empty(), "empty input wav");
    let max_steps = std::env::var("RLX_MOSHI_STREAM_MAX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (pcm_in.len() / FRAME_SAMPLES).clamp(8, 32));

    eprintln!(
        "[{device:?}] streaming {max_steps} frames from {} ({} samples)",
        wav_in.display(),
        pcm_in.len()
    );

    let out_pcm = stream_duplex_to_pcm(device, &pcm_in, max_steps)?;
    ensure!(
        out_pcm.len() >= MIN_OUTPUT_SAMPLES,
        "[{device:?}] output too short: {} samples",
        out_pcm.len()
    );
    ensure!(
        peak(&out_pcm) >= MIN_PEAK,
        "[{device:?}] output near silence (peak {})",
        peak(&out_pcm)
    );

    let out_wav =
        std::env::temp_dir().join(format!("moshi-stream-{device:?}.wav").replace(' ', "_"));
    write_wav_mono(&out_wav, &out_pcm, MIMI_RATE)?;
    eprintln!("[{device:?}] wrote {}", out_wav.display());

    let mut whisper = whisper_runner(&whisper_dir, device)?;
    let pcm_16k = resample_linear(&out_pcm, MIMI_RATE, WHISPER_RATE as u32);
    let text = whisper.transcribe_greedy(&pcm_16k)?;
    let norm = normalize_transcript(&text);
    eprintln!("[{device:?}] whisper: {norm}");
    ensure!(
        !norm.trim().is_empty(),
        "[{device:?}] whisper returned empty transcript"
    );
    Ok(())
}

#[test]
fn streaming_duplex_pcm_whisper_roundtrip() -> Result<()> {
    if std::env::var("RLX_MOSHI_STREAM_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_MOSHI_STREAM_E2E=1 and fetch weights");
        return Ok(());
    }
    if moshi_dir().is_none() || mimi_dir().is_none() {
        eprintln!("skip: run `just fetch-moshi` and `just fetch-mimi`");
        return Ok(());
    }
    if whisper_dir().is_none() {
        eprintln!("skip: run `just fetch-whisper-base`");
        return Ok(());
    }

    let devices = test_devices();
    ensure!(!devices.is_empty(), "no test devices");
    for device in devices {
        if !device_ready(device) {
            eprintln!("skip unavailable device {device:?}");
            continue;
        }
        streaming_roundtrip_on_device(device)?;
    }
    Ok(())
}
