// Native decoder_body dump for parity bisection: fixed tokens + en_female emb →
// mag/phase (via RLX_MIO_DUMP). A Python onnxruntime run on the SAME dumped
// tokens+emb then diffs mag/phase to isolate decoder_body vs the host ISTFT.
//   RLX_MIO_DUMP=/tmp/mio cargo run -p rlx-miotts --example mio_dump
use rlx_miotts::codec::{MioCodec, load_preset_embedding};
use rlx_miotts::tokens::{SPEECH_LEN, fit_speech_len};
use rlx_runtime::Device;

fn main() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let codec_dir = root.join("weights/tts/miocodec");
    let emb = load_preset_embedding(&root.join("weights/tts/miotts/presets"), "en_female")?;
    // Deterministic content codes in a plausible range (0..4096), length 100.
    let tokens: Vec<u32> = (0..SPEECH_LEN as u32)
        .map(|i| (i * 37 + 11) % 4096)
        .collect();
    let codec = MioCodec::load(&codec_dir, Device::Cpu)?;
    let wav = codec.decode(&fit_speech_len(&tokens), &emb)?;
    let peak = wav.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    eprintln!("wav samples={} peak={peak:.4}", wav.len());
    Ok(())
}
