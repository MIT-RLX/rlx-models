// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CLI for VibeVoice-ASR:
//   BitNet:    --vae <vae.gguf> --lm <lm.gguf> --audio <in.wav>
//   Streaming: --model-dir <hf-dir> --audio <in.wav> [--context-info …]

#[cfg(feature = "tokenizer")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, Result, bail};
    use rlx_runtime::Device;
    use rlx_vibevoice_asr::streaming::DEFAULT_MAX_NEW_PER_CHUNK;
    use rlx_vibevoice_asr::{EncodeMode, StreamingAsr, VibeAsr};
    use std::path::PathBuf;

    let mut vae: Option<PathBuf> = None;
    let mut lm: Option<PathBuf> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut audio: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut context_info: Option<String> = None;
    let mut encode_mode: Option<EncodeMode> = None;
    let mut json = false;
    let mut stream_json = false;
    let mut max_new = DEFAULT_MAX_NEW_PER_CHUNK;
    let mut device_name = "cpu".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--vae" => vae = args.next().map(PathBuf::from),
            "--lm" => lm = args.next().map(PathBuf::from),
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--audio" | "--wav" => audio = args.next().map(PathBuf::from),
            "--tokenizer" => tokenizer = args.next().map(PathBuf::from),
            "--context-info" => context_info = args.next(),
            "--encode" => {
                let v = args.next().unwrap_or_default();
                encode_mode = Some(match v.as_str() {
                    "encode_then_split" | "file" => EncodeMode::EncodeThenSplit,
                    "split_then_encode" | "stream" | "split" => EncodeMode::SplitThenEncode,
                    other => {
                        bail!("unknown --encode {other} (use encode_then_split|split_then_encode)")
                    }
                });
            }
            "--max-new-tokens" => {
                max_new = args
                    .next()
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_MAX_NEW_PER_CHUNK);
            }
            "--device" => device_name = args.next().unwrap_or_default(),
            "--json" => json = true,
            "--stream-json" => stream_json = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage:\n  \
                     rlx-vibevoice-asr --model-dir <hf-dir> --audio <in.wav> \\\n    \
                       [--device cpu|metal|mlx|gpu|cuda|vulkan|rocm|ane] \\\n    \
                       [--encode encode_then_split|split_then_encode] \\\n    \
                       [--context-info TEXT] [--max-new-tokens N] [--stream-json]\n  \
                     rlx-vibevoice-asr --vae <vae.gguf> --lm <lm.gguf> --audio <in.wav> \\\n    \
                       [--tokenizer <tokenizer.json>] [--device …] [--json]\n\n  \
                     Env: RLX_VIBEVOICE_ASR_TIMING=1  RLX_VIBEVOICE_ASR_ENCODE=split"
                );
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        "cuda" => Device::Cuda,
        "vulkan" => Device::Vulkan,
        "rocm" => Device::Rocm,
        "coreml" | "ane" => Device::Ane,
        other => bail!("unknown --device {other}"),
    };

    let audio_path = audio.context("--audio / --wav required")?;
    let (mono, sr) = load_wav_mono(&audio_path)?;
    eprintln!(
        "[vibeasr] audio: {} samples @ {sr} Hz ({:.2}s) device={device:?}",
        mono.len(),
        mono.len() as f32 / sr as f32
    );

    if let Some(dir) = model_dir {
        let mut asr = StreamingAsr::load(&dir, device)?;
        if let Some(mode) = encode_mode {
            asr.set_encode_mode(mode);
        }
        let chunks = asr.streaming_generate(&mono, sr, context_info.as_deref(), max_new)?;
        if stream_json {
            for c in &chunks {
                println!(
                    "{{\"index\":{},\"total\":{},\"text\":{}}}",
                    c.index,
                    c.total,
                    serde_json::to_string(&c.text)?
                );
            }
        } else {
            for c in &chunks {
                eprintln!("[{}/{}] {}", c.index + 1, c.total, c.text);
            }
            print!("{}", chunks.into_iter().map(|c| c.text).collect::<String>());
            if !json {
                println!();
            }
        }
        return Ok(());
    }

    let vae = vae.context("--vae required (or pass --model-dir for Streaming)")?;
    let lm = lm.context("--lm required (or pass --model-dir for Streaming)")?;
    let tokenizer = tokenizer.unwrap_or_else(|| {
        lm.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("tokenizer.json")
    });

    let mut asr = VibeAsr::load(&vae, &lm, &tokenizer, device)?;
    let text = asr.transcribe(&mono, sr, json)?;
    println!("{text}");
    return Ok(());

    /// Minimal RIFF/WAVE reader: 16-bit PCM (and 32-bit float) → mono f32 in
    /// [-1, 1], returning `(samples, sample_rate)`.
    fn load_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, usize)> {
        let bytes = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
        if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            bail!("not a RIFF/WAVE file: {path:?}");
        }
        let u16le = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let u32le =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

        let mut pos = 12usize;
        let mut fmt_tag = 1u16;
        let mut channels = 1usize;
        let mut sample_rate = 24000usize;
        let mut bits = 16u16;
        let mut data: Option<(usize, usize)> = None;

        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32le(pos + 4) as usize;
            let body = pos + 8;
            if id == b"fmt " && body + 16 <= bytes.len() {
                fmt_tag = u16le(body);
                channels = u16le(body + 2).max(1) as usize;
                sample_rate = u32le(body + 4) as usize;
                bits = u16le(body + 14);
            } else if id == b"data" {
                data = Some((body, size.min(bytes.len().saturating_sub(body))));
            }
            pos = body + size + (size & 1); // chunks are word-aligned
        }

        let (doff, dlen) = data.context("no data chunk")?;
        let seg = &bytes[doff..doff + dlen];
        let mut interleaved: Vec<f32> = Vec::new();
        match (fmt_tag, bits) {
            (1, 16) => {
                for c in seg.chunks_exact(2) {
                    interleaved.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
                }
            }
            (3, 32) => {
                for c in seg.chunks_exact(4) {
                    interleaved.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (1, 8) => {
                for &b in seg {
                    interleaved.push((b as f32 - 128.0) / 128.0);
                }
            }
            (t, b) => bail!("unsupported WAV format tag {t} / {b}-bit"),
        }

        let mono: Vec<f32> = if channels <= 1 {
            interleaved
        } else {
            interleaved
                .chunks_exact(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        Ok((mono, sample_rate))
    }
}

#[cfg(not(feature = "tokenizer"))]
fn main() {
    eprintln!("rlx-vibevoice-asr: rebuild with `--features tokenizer` to run transcription.");
}
