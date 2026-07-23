use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rlx_styletts2::{StyleTTS2, default_model_dir, parse_device};

#[derive(Parser, Debug)]
#[command(
    name = "rlx-styletts2",
    about = "StyleTTS2-family TTS via Kokoro-82M (native graph-split RLX path)"
)]
struct Args {
    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog.")]
    text: String,
    #[arg(long, default_value = "af_heart")]
    voice: String,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "/tmp/styletts2.wav")]
    output: PathBuf,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    list_voices: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = parse_device(&args.device)?;
    let model_dir = args.model_dir.unwrap_or_else(default_model_dir);
    let tts = StyleTTS2::load(&model_dir, device)?;

    if args.list_voices {
        for name in tts.voice_names() {
            println!("{name}");
        }
        return Ok(());
    }

    eprintln!(
        "StyleTTS2 {} text={:?} voice={} requested={device:?} speed={}",
        tts.path(),
        args.text,
        args.voice,
        args.speed
    );
    let audio = tts.generate(&args.text, &args.voice, args.speed)?;
    tts.write_wav(&audio, &args.output)?;
    eprintln!(
        "wrote {} ({} samples @ {} Hz)",
        args.output.display(),
        audio.len(),
        tts.sample_rate()
    );
    Ok(())
}
