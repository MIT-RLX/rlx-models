use crate::SAMPLE_RATE;
use crate::codec::{RoundtripStats, TsacBackendKind, TsacCodec, TsacOptions};
use crate::device::parse_tsac_device;
use crate::download::{ensure_tsac, fetch_tsac, resolve_install_dir};
use crate::platform::platform_note;
use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: &[String]) -> Result<()> {
    let mut install_dir: Option<PathBuf> = None;
    let mut in_wav: Option<PathBuf> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut in_tsac: Option<PathBuf> = None;
    let mut out_tsac: Option<PathBuf> = None;
    let mut fetch = false;
    let mut bench = false;
    let mut device = Device::Cpu;
    let mut quality: Option<u8> = None;
    let mut fast = false;
    let mut separate_stereo = false;
    let mut channels: Option<u8> = None;
    let mut verbose = false;
    let mut backend = TsacBackendKind::Auto;
    let mut device_set = false;
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
            "--install-dir" | "--model-dir" => install_dir = Some(req(args, &mut i)?.into()),
            "--in-wav" | "--wav" => in_wav = Some(req(args, &mut i)?.into()),
            "--out-wav" | "--out" => out_wav = Some(req(args, &mut i)?.into()),
            "--in-tsac" => {
                in_tsac = Some(req(args, &mut i)?.into());
                mode = Mode::Decode;
            }
            "--out-tsac" => out_tsac = Some(req(args, &mut i)?.into()),
            "--device" => {
                device = parse_tsac_device(&req(args, &mut i)?)?;
                device_set = true;
            }
            "--quality" | "-q" => {
                quality = Some(req(args, &mut i)?.parse().context("--quality")?);
            }
            "--fast" | "-f" => {
                fast = true;
                i += 1;
            }
            "--separate-stereo" | "-s" => {
                separate_stereo = true;
                i += 1;
            }
            "--channels" | "-c" => {
                channels = Some(req(args, &mut i)?.parse().context("--channels")?);
            }
            "--verbose" | "-v" => {
                verbose = true;
                i += 1;
            }
            "--correct" => {
                backend = TsacBackendKind::Correct;
                i += 1;
            }
            "--native" => {
                backend = TsacBackendKind::Native;
                i += 1;
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

    let dir = resolve_install_dir(install_dir.as_deref());
    if fetch {
        fetch_tsac(&dir)?;
        if mode == Mode::Roundtrip && in_wav.is_none() && in_tsac.is_none() && !bench {
            return Ok(());
        }
    }
    // The correct codec uses descript-DAC weights (its own dir), not the q8 bundle.
    if backend != TsacBackendKind::Correct {
        ensure_tsac(&dir)?;
    }

    // Default device: for the correct codec, pick the fastest available backend
    // (MLX ≫ Metal > wgpu > CPU) rather than CPU — huge speedup (14–26× vs 0.02×).
    let correct_active = matches!(backend, TsacBackendKind::Correct)
        || (matches!(backend, TsacBackendKind::Auto)
            && crate::correct::weights_available(&crate::correct::default_dir())
            && std::env::var("RLX_TSAC_ENGINE").ok().as_deref() != Some("native"));
    if !device_set && correct_active {
        device = crate::correct::best_device();
    }

    let options = TsacOptions {
        device,
        backend,
        quality,
        fast,
        separate_stereo,
        channels,
        verbose,
    };
    let codec = TsacCodec::open_with_options(&dir, options)?;
    eprintln!("tsac: device={:?} (resolved {:?})", device, codec.device());

    if bench {
        let wav = in_wav.context("--in-wav required for --bench")?;
        let stats = codec.roundtrip(&wav, PathBuf::from("/tmp/rlx-tsac-bench.wav").as_path())?;
        print_bench(&wav, &stats);
        return Ok(());
    }

    match mode {
        Mode::Encode => {
            let wav = in_wav.context("--in-wav required for --encode")?;
            let out = out_tsac.context("--out-tsac required for --encode")?;
            let stats = codec.encode(&wav, &out)?;
            eprintln!(
                "encoded {} → {} ({:.1} ms, {} bytes, ~{:.0} bps @ {} Hz)",
                wav.display(),
                out.display(),
                stats.encode_ms,
                stats.output_bytes,
                tsac_bitrate_bps(stats.output_bytes, &wav)?,
                SAMPLE_RATE
            );
        }
        Mode::Decode => {
            let tsac_path = in_tsac.context("--in-tsac required for --decode")?;
            let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/tsac-decode.wav"));
            let t0 = Instant::now();
            codec.decode(&tsac_path, &out)?;
            eprintln!(
                "decoded {} → {} ({:.1} ms)",
                tsac_path.display(),
                out.display(),
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Mode::Roundtrip => {
            let wav = in_wav.context("--in-wav required")?;
            let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/tsac-roundtrip.wav"));
            let stats = codec.roundtrip(&wav, &out)?;
            eprintln!(
                "roundtrip {} → {} ({:.1} ms encode, {:.1} ms decode, {} bytes compressed)",
                wav.display(),
                out.display(),
                stats.encode_ms,
                stats.decode_ms,
                stats.tsac_bytes
            );
        }
    }
    Ok(())
}

fn tsac_bitrate_bps(tsac_bytes: u64, wav_path: &PathBuf) -> Result<f64> {
    let reader = hound::WavReader::open(wav_path)
        .with_context(|| format!("open wav {}", wav_path.display()))?;
    let spec = reader.spec();
    let samples = reader.len() as f64;
    let dur_s = samples / spec.sample_rate as f64 / spec.channels as f64;
    if dur_s <= 0.0 {
        return Ok(0.0);
    }
    Ok(tsac_bytes as f64 * 8.0 / dur_s)
}

fn print_bench(wav: &PathBuf, stats: &RoundtripStats) {
    let bitrate = tsac_bitrate_bps(stats.tsac_bytes, wav).unwrap_or(0.0);
    eprintln!(
        "bench {} @ {} Hz\n  encode: {:.1} ms\n  decode: {:.1} ms\n  compressed: {} bytes (~{:.1} kb/s)",
        wav.display(),
        SAMPLE_RATE,
        stats.encode_ms,
        stats.decode_ms,
        stats.tsac_bytes,
        bitrate / 1000.0
    );
}

fn print_help() {
    eprintln!(
        "rlx-tsac — Bellard TSAC codec (44.1 kHz, very low bitrate)

Usage:
  rlx-tsac --in-wav speech.wav --out-wav /tmp/out.wav
  rlx-tsac --encode --in-wav speech.wav --out-tsac speech.tsac
  rlx-tsac --decode --in-tsac speech.tsac --out-wav /tmp/out.wav
  rlx-tsac --bench --in-wav speech.wav --device metal
  rlx-tsac --fetch

Options:
  --install-dir DIR   Weight dir (default: RLX_TSAC_DIR or .cache/tsac)
  --in-wav PATH       Input audio (44.1 kHz WAV; use ffmpeg to convert)
  --out-wav PATH      Output WAV (decode / roundtrip)
  --in-tsac PATH      Input .tsac file (decode)
  --out-tsac PATH     Output .tsac file (encode)
  --device NAME       RLX backend: {STANDARD_DEVICE_NAMES}
  --quality N         Bitrate/quality (-q, 1–12 joint stereo)
  --fast              Disable Transformer (-f, Bellard binary only)
  --separate-stereo   Encode channels separately (-s, Bellard binary only)
  --channels N        Force channel count (-c, Bellard binary only)
  --verbose           Print statistics (-v)
  --correct           Use the correct codec (Descript-DAC-44kHz weights) on the
                      chosen RLX backend — recommended. Auto-selected when those
                      weights are present. (Bellard's q8 bundle uses an
                      un-dequantizable libnc BF8 format; descript = same model.)
  --native            Force the faithful tsac-ng C codec (decodes tsac-ng files)
  --encode            Encode only
  --decode            Decode only
  --bench             Time encode+decode roundtrip
  --fetch             Download official weight bundle

Native codec runs on every platform; set RLX_TSAC_ENGINE=bellard on Linux x86_64
to force the upstream binary. {}
Env: RLX_TSAC_DIR, RLX_TSAC_BIN
",
        platform_note()
    );
}
