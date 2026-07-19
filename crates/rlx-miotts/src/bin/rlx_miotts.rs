use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rlx_miotts::{GenerateOpts, MioSession, default_codec_dir, default_model_dir};
use rlx_runtime::{Device, parse_device};

#[derive(Parser, Debug)]
#[command(name = "rlx-miotts", about = "MioTTS-0.6B (Qwen3 + MioCodec) on RLX")]
struct Args {
    #[arg(long, default_value_t = default_text())]
    text: String,
    #[arg(long, default_value = "en_female")]
    preset: String,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "/tmp/miotts.wav")]
    output: PathBuf,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    codec_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 400)]
    max_tokens: usize,
}

fn default_text() -> String {
    "The quick brown fox jumps over the lazy dog.".into()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: Device = parse_device(&args.device)?;
    let model = args.model_dir.unwrap_or_else(default_model_dir);
    let codec = args.codec_dir.unwrap_or_else(default_codec_dir);
    let mut session = MioSession::open(&model, &codec, device)?;
    let opts = GenerateOpts {
        seed: args.seed,
        max_new_tokens: args.max_tokens,
        preset: args.preset.clone(),
    };
    eprintln!(
        "MioTTS synthesize text={:?} preset={} device={device:?} seed={}",
        args.text, args.preset, args.seed
    );
    let result = session.synthesize(&args.text, &opts)?;
    write_wav(&args.output, &result.samples, result.sample_rate)?;
    eprintln!(
        "wrote {} ({} samples @ {} Hz, {} codes)",
        args.output.display(),
        result.samples.len(),
        result.sample_rate,
        result.content_codes.len()
    );
    Ok(())
}

fn write_wav(path: &PathBuf, pcm: &[f32], sr: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}
