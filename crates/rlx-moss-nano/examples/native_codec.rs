use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap());
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 48000,
        add_blank: false,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        length_scale: 1.0,
        inter_channels: 0,
        gin_channels: 0,
    };
    let model = TinyModel::new(dir.join("codec"), cfg);
    let n = 4usize;
    let codes: Vec<u8> = (0..n * 16)
        .flat_map(|i| ((i as i32 * 37) % 1024).to_le_bytes())
        .collect();
    let lens: Vec<u8> = (n as i32).to_le_bytes().to_vec();
    let mut g = model
        .compile_named(
            "moss_audio_tokenizer_decode_full",
            Device::Cpu,
            n,
            &[("batch", 1), ("code_length", n)],
        )
        .map_err(|e| anyhow::anyhow!("compile: {e:#}"))?;
    let out = g.run_typed(&[
        ("audio_codes", &codes, DType::I32),
        ("audio_code_lengths", &lens, DType::I32),
    ]);
    let (b, _) = &out[0];
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let pk = v
        .iter()
        .fold(0f32, |m, &x| if x.is_nan() { m } else { m.max(x.abs()) });
    let nan = v.iter().filter(|x| x.is_nan()).count();
    eprintln!("  audio n={} peak={pk:.4} nans={nan}", v.len());
    if let Ok(p) = std::env::var("DUMP") {
        let _ = std::fs::write(p, b);
    }
    Ok(())
}
