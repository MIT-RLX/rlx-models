use crate::audio::load_wav_mono;
use crate::codec::DacCodec;
use crate::download::{ensure_weights, fetch_dac, resolve_model_dir};
use anyhow::{Context, Result, bail};
use rlx_cli::req;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut in_wav: Option<PathBuf> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut fetch = false;
    let mut bench = false;
    let mut num_quantizers: Option<usize> = None;
    let mut target_bitrate: Option<f32> = None;
    let mut model_type = "24khz".to_string();
    let mut device = rlx_runtime::Device::Cpu;
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
            "--model-type" => model_type = req(args, &mut i)?,
            "--device" => device = rlx_cli::parse_standard_device("dac", &req(args, &mut i)?)?,
            "--in-wav" | "--wav" => in_wav = Some(req(args, &mut i)?.into()),
            "--out-wav" | "--out" => out_wav = Some(req(args, &mut i)?.into()),
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

    let dir = model_dir.unwrap_or_else(|| resolve_model_dir(None).join(&model_type));
    if fetch {
        fetch_dac(&dir, &model_type)?;
        if mode == Mode::Roundtrip && in_wav.is_none() && !bench {
            return Ok(());
        }
    }
    ensure_weights(&dir)?;
    let codec = DacCodec::open_on(&dir, device)?;
    if let Some(bps) = target_bitrate {
        use rlx_core::AudioCodec;
        let nq = codec.info().quantizers_for_bitrate(bps);
        eprintln!(
            "dac: target {bps:.0} bps → {nq} codebooks ({:.0} bps)",
            codec.info().bitrate_bps(Some(nq))
        );
        num_quantizers = Some(nq);
    }

    if bench {
        let wav = in_wav.context("--bench requires --in-wav")?;
        let pcm = load_wav_mono(&wav, codec.sample_rate())?;
        let (_codes, _recon, stats) = codec.roundtrip_pcm(&pcm, num_quantizers)?;
        eprintln!(
            "dac {} encode {:.1} ms decode {:.1} ms {} frames {} samples",
            model_type, stats.encode_ms, stats.decode_ms, stats.num_frames, stats.pcm_samples
        );
        return Ok(());
    }

    match mode {
        Mode::Roundtrip => {
            let in_path = in_wav.context("roundtrip requires --in-wav")?;
            let out_path = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/dac-roundtrip.wav"));
            let codes = codec.roundtrip_wav(&in_path, &out_path, num_quantizers)?;
            eprintln!(
                "wrote {} ({} frames × {} codebooks)",
                out_path.display(),
                codes.num_frames(),
                codes.num_quantizers
            );
        }
        Mode::Encode => {
            bail!("encode-only mode not yet exposed — use roundtrip for now");
        }
        Mode::Decode => {
            bail!("decode-only mode not yet exposed — use roundtrip for now");
        }
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "rlx-dac — Descript Audio Codec (Rust/RLX)\n\
         \n\
         Usage:\n\
           rlx-dac --fetch --model-type 24khz\n\
           rlx-dac --model-dir .cache/dac/24khz --in-wav speech.wav --out-wav out.wav\n\
           rlx-dac --bench --in-wav speech.wav\n\
         \n\
         Options:\n\
           --model-dir DIR       weights directory (default: RLX_DAC_DIR or .cache/dac/24khz)\n\
           --model-type TYPE     24khz | 16khz | 44khz (default 24khz)\n\
           --device DEV          cpu|metal|mlx|cuda|rocm|gpu|wgpu|vulkan (default cpu)\n\
           --fetch               download safetensors from HuggingFace\n\
           --num-quantizers N    RVQ codebooks to use (default: all)\n\
           --target-bitrate BPS  pick codebooks to fit a target bitrate (bits/sec)\n\
           --bench               print encode/decode timing\n"
    );
}
