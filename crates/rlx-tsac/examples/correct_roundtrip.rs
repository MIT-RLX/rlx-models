//! Correct TSAC (= Descript-DAC-44kHz) encode→decode on an RLX backend.
//! ```bash
//! RLX_DAC_DIR=$PWD/.cache/dac44 cargo run -p rlx-tsac --example correct_roundtrip \
//!   --release --features native-codec,oracle,metal -- /tmp/tcmp/in44.wav metal
//! ```
use anyhow::Result;
use rlx_tsac::{SAMPLE_RATE, audio, correct, parse_tsac_device};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let in_wav = args
        .first()
        .cloned()
        .unwrap_or_else(|| "/tmp/tcmp/in44.wav".into());
    let device = args
        .get(1)
        .map(|s| parse_tsac_device(s))
        .transpose()?
        .unwrap_or(rlx_runtime::Device::Cpu);

    let (input, sr) = audio::load_wav_f32(std::path::Path::new(&in_wav), SAMPLE_RATE)?;
    eprintln!(
        "input: {} samples @ {sr} Hz, device={device:?}",
        input.len()
    );

    let codec = correct::open(device)?;
    let (codes, recon, _) = codec.roundtrip_pcm(&input, None)?;
    let n = input.len().min(recon.len());
    let corr = audio::correlation(&input[..n], &recon[..n]);
    eprintln!(
        "correct TSAC roundtrip on {device:?}: {} frames, recon corr={corr:.4} (n={n})",
        codes.num_frames(),
    );
    Ok(())
}
