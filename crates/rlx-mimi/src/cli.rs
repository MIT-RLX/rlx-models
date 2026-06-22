use crate::audio::{load_wav_mono, parse_codes_text, write_wav_mono};
use crate::codec::{MimiCodec, RoundtripStats, SAMPLE_RATE};
use crate::codes::MimiCodes;
use crate::download::{ensure_weights, fetch_mimi, resolve_model_dir};
use anyhow::{Context, Result, bail};
use rlx_cli::req;
use std::fs;
use std::path::{Path, PathBuf};

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut in_wav: Option<PathBuf> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut codes_in: Option<PathBuf> = None;
    let mut codes_out: Option<PathBuf> = None;
    let mut fetch = false;
    let mut bench = false;
    let mut num_quantizers: Option<usize> = None;
    let mut target_bitrate: Option<f32> = None;
    let mut device = crate::device::parse_mimi_device("cpu")?;
    let mut mode = Mode::Roundtrip;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Roundtrip,
        Encode,
        Decode,
    }

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => model_dir = Some(req(args, &mut i)?.into()),
            "--device" => device = crate::device::parse_mimi_device(&req(args, &mut i)?)?,
            "--in-wav" | "--wav" => in_wav = Some(req(args, &mut i)?.into()),
            "--out-wav" | "--out" => out_wav = Some(req(args, &mut i)?.into()),
            "--codes-in" => {
                codes_in = Some(req(args, &mut i)?.into());
                mode = Mode::Decode;
            }
            "--codes-out" => codes_out = Some(req(args, &mut i)?.into()),
            "--num-quantizers" => {
                num_quantizers = Some(req(args, &mut i)?.parse().context("--num-quantizers")?);
            }
            "--target-bitrate" => {
                target_bitrate = Some(
                    req(args, &mut i)?
                        .parse()
                        .context("--target-bitrate (bits/sec)")?,
                );
            }
            "--encode" => {
                mode = Mode::Encode;
                i += 1;
            }
            "--decode" => {
                mode = Mode::Decode;
                i += 1;
            }
            "--bench" => {
                bench = true;
                i += 1;
            }
            "--fetch" | "--download" => {
                fetch = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown flag `{other}` (try --help)"),
        }
    }

    let dir = resolve_model_dir(model_dir.as_deref());
    if fetch {
        fetch_mimi(&dir)?;
        if mode == Mode::Roundtrip && in_wav.is_none() && codes_in.is_none() && !bench {
            return Ok(());
        }
    }
    ensure_weights(&dir)?;
    let codec = MimiCodec::open_on(&dir, device)?;
    if let Some(bps) = target_bitrate {
        use rlx_core::AudioCodec;
        let nq = codec.info().quantizers_for_bitrate(bps);
        eprintln!(
            "mimi: target {bps:.0} bps → {nq} codebooks ({:.0} bps)",
            codec.info().bitrate_bps(Some(nq))
        );
        num_quantizers = Some(nq);
    }

    if bench {
        let wav = in_wav.unwrap_or_else(repo_jfk_24k);
        let pcm = load_wav_mono(&wav, SAMPLE_RATE)?;
        let (_codes, _recon, stats) = codec.roundtrip_pcm(&pcm, num_quantizers)?;
        print_bench(&codec, &wav, &stats);
        return Ok(());
    }

    match mode {
        Mode::Encode => {
            let wav = in_wav.context("--in-wav required for --encode")?;
            let codes = codec.encode_wav(&wav, num_quantizers)?;
            let out = codes_out.context("--codes-out required for --encode")?;
            write_codes(&out, &codes)?;
            eprintln!(
                "encoded {} frames × {} codebooks → {}",
                codes.num_frames(),
                codes.num_quantizers,
                out.display()
            );
        }
        Mode::Decode => {
            let codes_path = codes_in.context("--codes-in required for --decode")?;
            let text = fs::read_to_string(&codes_path)
                .with_context(|| format!("read {}", codes_path.display()))?;
            let nq = num_quantizers.unwrap_or(codec.config().num_quantizers);
            let frames = parse_codes_text(&text, nq)?;
            let codes = MimiCodes {
                num_quantizers: nq,
                frames,
            };
            let pcm = codec.decode_codes(&codes)?;
            let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/mimi-decode.wav"));
            write_wav_mono(&out, &pcm, SAMPLE_RATE)?;
            eprintln!(
                "decoded {} frames → {} ({} samples)",
                codes.num_frames(),
                out.display(),
                pcm.len()
            );
        }
        Mode::Roundtrip => {
            let wav = in_wav.context("--in-wav required")?;
            let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/mimi-roundtrip.wav"));
            let pcm_len = load_wav_mono(&wav, SAMPLE_RATE)?.len();
            let codes = codec.encode_wav(&wav, num_quantizers)?;
            codec.decode_wav(&codes, &out, Some(pcm_len))?;
            eprintln!(
                "roundtrip {} → {} ({} codec frames)",
                wav.display(),
                out.display(),
                codes.num_frames()
            );
        }
    }
    Ok(())
}

fn repo_jfk_24k() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/rlx-qwen3-tts/examples/audio/ask_not.wav")
}

fn print_bench(codec: &MimiCodec, wav: &Path, stats: &RoundtripStats) {
    let cfg = codec.config();
    let dur_s = stats.pcm_samples as f64 / SAMPLE_RATE as f64;
    let rtf_encode = stats.encode_ms / 1000.0 / dur_s;
    let rtf_decode = stats.decode_ms / 1000.0 / dur_s;
    eprintln!(
        "bench {} ({:.2}s, {} samples)\n  encode: {:.1} ms (RTF {:.3})\n  decode: {:.1} ms (RTF {:.3})\n  frames: {} @ {:.1} Hz ({} samples/frame)\n  nominal bitrate: {:.0} bps",
        wav.display(),
        dur_s,
        stats.pcm_samples,
        stats.encode_ms,
        rtf_encode,
        stats.decode_ms,
        rtf_decode,
        stats.num_frames,
        cfg.frame_rate,
        cfg.samples_per_codec_frame(),
        cfg.bitrate_bps(),
    );
}

fn write_codes(path: &PathBuf, codes: &MimiCodes) -> Result<()> {
    let mut text = String::new();
    for row in &codes.frames {
        for (i, c) in row.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(&c.to_string());
        }
        text.push('\n');
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn print_help() {
    eprintln!(
        "rlx-mimi — Kyutai Mimi codec (12.5 Hz @ 24 kHz)

Usage:
  rlx-mimi --in-wav speech.wav --out-wav /tmp/out.wav
  rlx-mimi --encode --in-wav speech.wav --codes-out codes.txt
  rlx-mimi --decode --codes-in codes.txt --out-wav /tmp/out.wav
  rlx-mimi --bench --in-wav speech.wav
  rlx-mimi --fetch

Options:
  --model-dir DIR     HF checkpoint dir (default: RLX_MIMI_DIR or .cache/mimi)
  --device DEV        cpu|metal|mlx|cuda|rocm|gpu|wgpu|vulkan (default cpu)
  --in-wav PATH       Input mono/stereo WAV (resampled to 24 kHz)
  --out-wav PATH      Output WAV path
  --codes-in PATH     Text file: one frame per line, space-separated codebook indices
  --codes-out PATH    Write encoded codes (text)
  --num-quantizers N  Truncate RVQ codebooks (default: 32)
  --target-bitrate B  Pick codebooks to fit a target bitrate (bits/sec)
  --encode            Encode only
  --decode            Decode only
  --bench             Print encode/decode timing + RTF
  --fetch             Download kyutai/mimi into --model-dir
"
    );
}
