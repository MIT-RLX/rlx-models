// metavoice Metal smoke: partial generation + wav, for a fast whisper check.
//   RLX_DEV=metal RLX_METAVOICE_DIR=... RLX_ENCODEC_PATH=...gguf MAX_NEW=300 \
//     WAV_OUT=/tmp/mv.wav cargo run -p rlx-metavoice --example mv_probe --features metal
use rlx_metavoice::{InferOpts, MetaVoice, peak_amplitude};
use rlx_runtime::Device;
use std::io::Write;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_METAVOICE_DIR")?;
    let enc = std::env::var("RLX_ENCODEC_PATH")?;
    let dev = match std::env::var("RLX_DEV").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        _ => Device::Cpu,
    };
    eprintln!("[mv] device={dev:?}");
    let mv = MetaVoice::open_with_encodec(Path::new(&dir), Path::new(&enc), dev)?;
    let opts = InferOpts {
        max_new_tokens: std::env::var("MAX_NEW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
        ..InferOpts::default()
    };
    let reference = format!("{dir}/bria_16k.wav");
    let text = "The quick brown fox jumps over the lazy dog.";
    let pcm = mv.synthesize(text, Some(Path::new(&reference)), &opts)?;
    eprintln!(
        "[mv] samples={} sr={} peak={:.4} max_new={}",
        pcm.len(),
        mv.sample_rate(),
        peak_amplitude(&pcm),
        opts.max_new_tokens
    );
    if let Ok(out) = std::env::var("WAV_OUT") {
        let sr = mv.sample_rate();
        let mut f = std::fs::File::create(&out)?;
        let data = (pcm.len() * 2) as u32;
        f.write_all(b"RIFF")?;
        f.write_all(&(36 + data).to_le_bytes())?;
        f.write_all(b"WAVEfmt ")?;
        f.write_all(&16u32.to_le_bytes())?;
        f.write_all(&1u16.to_le_bytes())?;
        f.write_all(&1u16.to_le_bytes())?;
        f.write_all(&sr.to_le_bytes())?;
        f.write_all(&(sr * 2).to_le_bytes())?;
        f.write_all(&2u16.to_le_bytes())?;
        f.write_all(&16u16.to_le_bytes())?;
        f.write_all(b"data")?;
        f.write_all(&data.to_le_bytes())?;
        for &s in &pcm {
            f.write_all(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())?;
        }
        eprintln!("[mv] wrote {out}");
    }
    Ok(())
}
