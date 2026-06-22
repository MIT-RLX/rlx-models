// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Live **audio-to-audio** voice chat with Kyutai **Moshi** (full-duplex
//! speech↔speech).
//!
//! Unlike `qwen_voice_chat` (a Qwen3-ASR → Qwen3 LM → Qwen3-TTS *pipeline*),
//! Moshi is a **single** model: user speech goes in, Moshi speech comes out,
//! continuously and simultaneously — no separate ASR/LLM/TTS stages, no turn
//! gate. We just stream 24 kHz mic PCM into [`DuplexStreamEngine::feed_pcm`] and
//! play the [`StreamStepOutput::moshi_pcm`] it returns; Moshi's text "inner
//! monologue" is printed as it goes.
//!
//! Moshi is a 7B model at 12.5 Hz (80 ms/frame): real-time needs a GPU
//! (`--features apple-silicon` → Metal/MLX). On CPU it runs but far slower than
//! real-time.
//!
//! Quick run (download ~8 GB Q8 + ~385 MB Mimi first):
//! ```sh
//! just fetch-mimi && just fetch-moshi-q8
//! # batch: drive Moshi from a WAV, write its reply
//! cargo run --release -p rlx-moshi --features apple-silicon \
//!   --example moshi_voice_chat -- --device metal \
//!   --input-wav some_speech.wav --out-wav /tmp/moshi_reply.wav
//! # live mic ↔ speaker (use headphones — Moshi is full-duplex and will hear itself):
//! cargo run --release -p rlx-moshi --features apple-silicon,mic \
//!   --example moshi_voice_chat -- --device metal --mic --secs 60
//! ```

#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use rlx_moshi::{
    DuplexStreamEngine, GenerationConfig, MoshiSession, MoshiVariant, default_mimi_dir,
    default_moshi_dir, parse_moshi_device,
};
use rlx_runtime::{Device, is_available};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const MOSHI_RATE: u32 = 24_000; // Mimi codec PCM rate

struct Args {
    moshi_dir: PathBuf,
    mimi_dir: PathBuf,
    device: Device,
    variant: MoshiVariant,
    /// Conversation length in seconds (→ max_steps at 12.5 Hz).
    secs: f32,
    input_wav: Option<PathBuf>,
    out_wav: PathBuf,
    mic: bool,
    mic_chunk_ms: u32,
}

fn pick_device(name: &str) -> Result<Device> {
    Ok(if name == "auto" {
        if is_available(Device::Metal) {
            Device::Metal
        } else if is_available(Device::Cuda) {
            Device::Cuda
        } else {
            Device::Cpu
        }
    } else {
        parse_moshi_device(name)?
    })
}

fn parse_variant(s: &str) -> Result<MoshiVariant> {
    Ok(match s.to_lowercase().as_str() {
        "moshiko" | "male" => MoshiVariant::Moshiko,
        "moshika" | "female" => MoshiVariant::Moshika,
        other => bail!("--variant: expected moshiko|moshika (full-duplex), got {other:?}"),
    })
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        moshi_dir: default_moshi_dir(),
        mimi_dir: default_mimi_dir(),
        device: Device::Cpu,
        variant: MoshiVariant::Moshiko,
        secs: 30.0,
        input_wav: None,
        out_wav: PathBuf::from("/tmp/moshi_voice_chat.wav"),
        mic: false,
        mic_chunk_ms: 80,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let val = |i: usize| -> Result<String> {
        raw.get(i + 1)
            .cloned()
            .with_context(|| format!("missing value for {}", raw[i]))
    };
    let mut i = 0;
    while i < raw.len() {
        let mut step = 1;
        match raw[i].as_str() {
            "--moshi-dir" => {
                a.moshi_dir = PathBuf::from(val(i)?);
                step = 2;
            }
            "--mimi-dir" => {
                a.mimi_dir = PathBuf::from(val(i)?);
                step = 2;
            }
            "--device" => {
                a.device = pick_device(&val(i)?)?;
                step = 2;
            }
            "--variant" => {
                a.variant = parse_variant(&val(i)?)?;
                step = 2;
            }
            "--secs" => {
                a.secs = val(i)?.parse().context("--secs")?;
                step = 2;
            }
            "--input-wav" => {
                a.input_wav = Some(PathBuf::from(val(i)?));
                step = 2;
            }
            "--out-wav" => {
                a.out_wav = PathBuf::from(val(i)?);
                step = 2;
            }
            "--mic" => a.mic = true,
            "--mic-chunk-ms" => {
                a.mic_chunk_ms = val(i)?.parse().context("--mic-chunk-ms")?;
                step = 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: moshi_voice_chat [--device auto|metal|mlx|cpu] [--variant moshiko|moshika]\n\
                     \x20 [--secs N] [--input-wav WAV] [--out-wav WAV] [--mic] [--mic-chunk-ms N]\n\
                     \x20 [--moshi-dir DIR] [--mimi-dir DIR]\n\n\
                     \x20 Audio-to-audio: speech in → Moshi speech out (one full-duplex model).\n\
                     \x20 --mic needs the `mic` cargo feature; real-time needs a GPU (--device metal)."
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
        i += step;
    }
    Ok(a)
}

fn gen_config(secs: f32) -> GenerationConfig {
    GenerationConfig {
        max_steps: ((secs * 12.5).round() as usize).max(1),
        ..GenerationConfig::default()
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    println!("┌─ Moshi voice chat (audio → audio, one full-duplex model) ─────");
    println!("│ Moshi:  {}", args.moshi_dir.display());
    println!("│ Mimi:   {}", args.mimi_dir.display());
    println!("│ device: {:?}  variant: {:?}", args.device, args.variant);
    println!(
        "│ mode:   {}",
        if args.mic {
            "🎙  live microphone"
        } else {
            "WAV batch"
        }
    );
    println!(
        "│ length: {:.0}s ({} frames @ 12.5 Hz)",
        args.secs,
        gen_config(args.secs).max_steps
    );
    println!("└───────────────────────────────────────────────────────────────");
    if args.device == Device::Cpu {
        eprintln!(
            "warning: Moshi 7B on CPU is far below real-time — use `--device metal` (apple-silicon build)"
        );
    }

    if args.mic {
        return run_mic(&args);
    }
    run_wav(&args)
}

/// Batch: drive Moshi from an input WAV (or silence) and write its reply.
fn run_wav(args: &Args) -> Result<()> {
    let cfg = gen_config(args.secs);
    let t0 = Instant::now();
    let session =
        MoshiSession::open_on(&args.moshi_dir, &args.mimi_dir, args.variant, args.device)?;
    println!("opened Moshi + Mimi in {:.2}s", t0.elapsed().as_secs_f64());
    let mut engine = DuplexStreamEngine::from_session(session, "", &cfg)?;

    // User audio: the input WAV (24 kHz mono), or silence if none.
    let user: Vec<f32> = match &args.input_wav {
        Some(p) => load_wav_mono_24k(p)?,
        None => vec![0.0; cfg.max_steps * engine.frame_samples()],
    };

    let mut out = Vec::new();
    let t = Instant::now();
    for chunk in user.chunks(engine.frame_samples()) {
        for s in engine.feed_pcm(chunk)? {
            if let Some(delta) = &s.transcript_delta {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            out.extend_from_slice(&s.moshi_pcm);
        }
    }
    for s in engine.finish()? {
        if let Some(delta) = &s.transcript_delta {
            print!("{delta}");
        }
        out.extend_from_slice(&s.moshi_pcm);
    }
    println!();
    let audio_s = out.len() as f64 / MOSHI_RATE as f64;
    let wall = t.elapsed().as_secs_f64();
    write_wav_mono_24k(&args.out_wav, &out)?;
    println!(
        "✓ wrote {} ({:.2}s audio, synth {:.2}s, RTF {:.2}×, {} steps)",
        args.out_wav.display(),
        audio_s,
        wall,
        audio_s / wall.max(1e-9),
        engine.steps_done()
    );
    Ok(())
}

/// Live mic ↔ speaker: stream mic PCM into Moshi and play its reply, continuously.
#[cfg(feature = "mic")]
fn run_mic(args: &Args) -> Result<()> {
    use std::time::Duration;
    // Grab the audio devices BEFORE the (heavy) model load so the CoreAudio
    // session stays live throughout.
    let cap = mic::MicCapture::start()?;
    let player = mic::StreamPlayer::start(MOSHI_RATE)?;
    println!(
        "🎙  mic + speaker acquired ({} Hz in); loading Moshi…",
        cap.in_rate
    );

    let cfg = gen_config(args.secs);
    let t0 = Instant::now();
    let session =
        MoshiSession::open_on(&args.moshi_dir, &args.mimi_dir, args.variant, args.device)?;
    println!("opened Moshi + Mimi in {:.2}s", t0.elapsed().as_secs_f64());
    let mut engine = DuplexStreamEngine::from_session(session, "", &cfg)?;

    cap.clear(); // discard audio buffered during load
    println!(
        "\n🎙  talk to Moshi (full-duplex). Use HEADPHONES — it hears the mic continuously. Ctrl-C to quit.\n"
    );
    let chunk_ms = args.mic_chunk_ms.max(20) as u64;
    loop {
        std::thread::sleep(Duration::from_millis(chunk_ms));
        let raw = cap.drain();
        if raw.is_empty() {
            continue;
        }
        let pcm24 = resample_linear(&raw, cap.in_rate, MOSHI_RATE);
        for s in engine.feed_pcm(&pcm24)? {
            if let Some(delta) = &s.transcript_delta {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            player.push(&s.moshi_pcm);
        }
        if engine.steps_done() >= cfg.max_steps {
            println!("\n⏱  reached {:.0}s — done.", args.secs);
            player.wait_drained();
            break;
        }
    }
    Ok(())
}

#[cfg(not(feature = "mic"))]
fn run_mic(_args: &Args) -> Result<()> {
    bail!("--mic requires the `mic` cargo feature: rebuild with `--features apple-silicon,mic`")
}

// ── WAV + resample helpers ───────────────────────────────────────────────────

fn load_wav_mono_24k(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };
    // Downmix to mono.
    let mono: Vec<f32> = if ch <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok(resample_linear(&mono, spec.sample_rate, MOSHI_RATE))
}

fn write_wav_mono_24k(path: &std::path::Path, pcm: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: MOSHI_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
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

// ── live microphone + speaker (opt-in `mic` cargo feature) ───────────────────

#[cfg(feature = "mic")]
mod mic {
    use anyhow::{Context, Result, bail};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{FromSample, Sample, SampleFormat, SizedSample};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `default_*_config()` intermittently fails on macOS CoreAudio — retry then
    /// fall back to enumerating supported ranges (prefer f32 @ 48 k / 24 k).
    fn pick_from_ranges<I: Iterator<Item = cpal::SupportedStreamConfigRange>>(
        ranges: I,
    ) -> Option<cpal::SupportedStreamConfig> {
        let mut best: Option<cpal::SupportedStreamConfig> = None;
        for r in ranges {
            let (min, max) = (r.min_sample_rate().0, r.max_sample_rate().0);
            let want = if (min..=max).contains(&48_000) {
                48_000
            } else if (min..=max).contains(&24_000) {
                24_000
            } else {
                max
            };
            let cfg = r.with_sample_rate(cpal::SampleRate(want));
            if cfg.sample_format() == SampleFormat::F32 {
                return Some(cfg);
            }
            best.get_or_insert(cfg);
        }
        best
    }

    fn pick_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
        for _ in 0..3 {
            if let Ok(c) = device.default_input_config() {
                return Ok(c);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let ranges = device.supported_input_configs().context(
            "no input configs — grant mic permission in System Settings → Privacy → Microphone",
        )?;
        pick_from_ranges(ranges).context("no usable input config (mic permission?)")
    }

    fn pick_output_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
        for _ in 0..3 {
            if let Ok(c) = device.default_output_config() {
                return Ok(c);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let ranges = device
            .supported_output_configs()
            .context("no output configs")?;
        pick_from_ranges(ranges).context("no usable output config")
    }

    /// Background mic capture into a shared mono f32 buffer at the device rate.
    pub struct MicCapture {
        _stream: cpal::Stream,
        buf: Arc<Mutex<Vec<f32>>>,
        pub in_rate: u32,
    }

    impl MicCapture {
        pub fn start() -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .context("no default input device (grant mic permission?)")?;
            let supported = pick_input_config(&device)?;
            let in_rate = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let cfg: cpal::StreamConfig = supported.config();
            let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
            let stream = match supported.sample_format() {
                SampleFormat::F32 => build_input::<f32>(&device, &cfg, channels, buf.clone())?,
                SampleFormat::I16 => build_input::<i16>(&device, &cfg, channels, buf.clone())?,
                SampleFormat::U16 => build_input::<u16>(&device, &cfg, channels, buf.clone())?,
                other => bail!("unsupported mic sample format {other:?}"),
            };
            stream.play().context("start mic stream")?;
            Ok(Self {
                _stream: stream,
                buf,
                in_rate,
            })
        }

        pub fn drain(&self) -> Vec<f32> {
            std::mem::take(&mut *lock(&self.buf))
        }
        pub fn clear(&self) {
            lock(&self.buf).clear();
        }
    }

    fn build_input<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        buf: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let stream = device.build_input_stream(
            cfg,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mut b = lock(&buf);
                for frame in data.chunks(channels.max(1)) {
                    let mut acc = 0.0f32;
                    for &s in frame {
                        acc += f32::from_sample(s);
                    }
                    b.push(acc / channels.max(1) as f32);
                }
            },
            |e| eprintln!("mic stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }

    /// Persistent streaming speaker: drains a shared queue continuously.
    pub struct StreamPlayer {
        _stream: cpal::Stream,
        queue: Arc<Mutex<VecDeque<f32>>>,
        src_rate: u32,
        out_rate: u32,
    }

    impl StreamPlayer {
        pub fn start(src_rate: u32) -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .context("no default output device")?;
            let supported = pick_output_config(&device)?;
            let out_rate = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let cfg: cpal::StreamConfig = supported.config();
            let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
            let stream = match supported.sample_format() {
                SampleFormat::F32 => build_out::<f32>(&device, &cfg, channels, queue.clone())?,
                SampleFormat::I16 => build_out::<i16>(&device, &cfg, channels, queue.clone())?,
                SampleFormat::U16 => build_out::<u16>(&device, &cfg, channels, queue.clone())?,
                other => bail!("unsupported speaker sample format {other:?}"),
            };
            stream.play().context("start speaker stream")?;
            Ok(Self {
                _stream: stream,
                queue,
                src_rate,
                out_rate,
            })
        }

        pub fn push(&self, pcm_src: &[f32]) {
            if pcm_src.is_empty() {
                return;
            }
            let resampled = super::resample_linear(pcm_src, self.src_rate, self.out_rate);
            lock(&self.queue).extend(resampled);
        }

        pub fn wait_drained(&self) {
            while !lock(&self.queue).is_empty() {
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(Duration::from_millis(180));
        }
    }

    fn build_out<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        queue: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample + FromSample<f32>,
    {
        let stream = device.build_output_stream(
            cfg,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut q = lock(&queue);
                for frame in out.chunks_mut(channels.max(1)) {
                    let s = q.pop_front().unwrap_or(0.0);
                    let v = T::from_sample(s);
                    for c in frame.iter_mut() {
                        *c = v;
                    }
                }
            },
            |e| eprintln!("speaker stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }
}
