//! Check whether F5_Transformer output aliases the noise input on wgpu.
use half::f16;
use rlx_f5tts::config::{HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use rlx_f5tts::{DEFAULT_LOCAL_DIR, InferOpts};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

fn main() -> anyhow::Result<()> {
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let ref_path = PathBuf::from("crates/rlx-f5tts/tests/fixtures/prompt.wav");
    let (reference, sr) = {
        let mut r = hound::WavReader::open(&ref_path)?;
        let spec = r.spec();
        let samples: Vec<f32> = r
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect();
        (samples, spec.sample_rate)
    };
    let text = "The quick brown fox jumps over the lazy dog.";
    let ref_text =
        normalize_ref_text("Hello from Kokoro. This is a test of speech synthesis in Rust.");
    let reference = preprocess_ref_audio(&reference, sr);
    let vocab = Vocab::load(&model)?;
    let text_ids = encode(&ref_text, text, &vocab);
    let n = reference.len();
    let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
    let ref_tl = text_len(&ref_text).max(1) as f64;
    let gen_tl = text_len(text) as f64;
    let opts = InferOpts {
        nfe: 32,
        speed: 1.0,
    };
    let d = (ref_audio_len + (ref_audio_len / ref_tl * gen_tl / opts.speed as f64)) as usize;
    let t = text_ids.len();
    let layout = Layout::resolve(&model)?;
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: rlx_f5tts::SAMPLE_RATE,
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
    let tiny = TinyModel::new(layout.dir, cfg);
    let named = [
        ("audio_len", n),
        ("text_ids_len", t),
        ("max_duration", d),
        ("text_embed_len", 612),
    ];
    let audio_b: Vec<u8> = reference
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();
    let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_b = (d as i64).to_le_bytes().to_vec();
    let mut cpu_pre = tiny.compile_named("F5_Preprocess", Device::Cpu, d, &named)?;
    let pre = cpu_pre.run_typed(&[
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ]);
    let noise: Vec<u8> = to_f32(&pre[0].0, pre[0].1)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // Corrupt noise so identity is obvious: set first element to 123.0
    let mut noise_marked = noise.clone();
    noise_marked[..4].copy_from_slice(&123.0f32.to_le_bytes());
    let names = [
        "noise",
        "rope_cos",
        "rope_sin",
        "cat_mel_text",
        "cat_mel_text_drop",
        "qk_rotated_empty",
    ];
    let cond: Vec<(String, Vec<u8>)> = (0..6)
        .map(|i| {
            let b = if i == 0 {
                noise_marked.clone()
            } else {
                to_f32(&pre[i].0, pre[i].1)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            };
            (names[i].to_string(), b)
        })
        .collect();
    let mut tf = tiny.compile_named(
        "F5_Transformer",
        Device::Gpu,
        d,
        &[("max_duration", d), ("text_embed_len", 612)],
    )?;
    let ts = 0i32.to_le_bytes().to_vec();
    let feed: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], cond[i].1.as_slice(), DType::F32))
        .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
        .collect();
    let out = tf.run_typed(&feed);
    let v = to_f32(&out[0].0, out[0].1);
    let noise_f = to_f32(&noise_marked, DType::F32);
    let delta = v
        .iter()
        .zip(noise_f.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let peak = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    println!(
        "out[0]={:.4} (marked input was 123.0) peak={:.4} |out-noise|_inf={:.6} n={}",
        v[0],
        peak,
        delta,
        v.len()
    );
    // At small CFG residual, out[0]≈123 is expected; a dead DiT has delta≈0.
    println!("identity={}", delta < 1e-4);
    Ok(())
}
