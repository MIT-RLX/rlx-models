//! Multi-step CPU vs CUDA DiT cosine (CPU preprocess reused).
use half::f16;
use rlx_f5tts::config::{HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use rlx_f5tts::{DEFAULT_LOCAL_DIR, InferOpts};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

const TEXT_EMBED_LEN: usize = 612;

fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt() + 1e-12)
}
fn peak(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
fn as_f32_bytes(b: &[u8], dt: DType) -> Vec<u8> {
    to_f32(b, dt).iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
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
    let opts = InferOpts {
        nfe: 32,
        speed: 1.0,
    };
    let d = {
        let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
        let ref_tl = text_len(&ref_text).max(1) as f64;
        let gen_tl = text_len(text) as f64;
        (ref_audio_len + (ref_audio_len / ref_tl * gen_tl / opts.speed as f64)) as usize
    };
    let t = text_ids.len();
    let layout = Layout::resolve(&model)?;
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 24_000,
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
        ("text_embed_len", TEXT_EMBED_LEN),
    ];
    let audio_b: Vec<u8> = reference
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();
    let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_b = (d as i64).to_le_bytes().to_vec();
    let pre_inputs: [(&str, &[u8], DType); 3] = [
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ];
    let mut cpu_pre = tiny.compile_named("F5_Preprocess", Device::Cpu, d, &named)?;
    let pre = cpu_pre.run_typed(&pre_inputs);
    let names = [
        "noise",
        "rope_cos",
        "rope_sin",
        "cat_mel_text",
        "cat_mel_text_drop",
        "qk_rotated_empty",
    ];
    let tf_named = &[("max_duration", d), ("text_embed_len", TEXT_EMBED_LEN)];
    let cond_f32: Vec<(String, Vec<u8>)> = (0..6)
        .map(|i| (names[i].to_string(), as_f32_bytes(&pre[i].0, pre[i].1)))
        .collect();

    let mut tf_cpu = tiny.compile_named("F5_Transformer", Device::Cpu, d, tf_named)?;
    let mut tf_cuda = tiny.compile_named("F5_Transformer", Device::Cuda, d, tf_named)?;

    let mut noise_cpu = pre[0].0.clone();
    let mut noise_cpu_dt = pre[0].1;
    let mut noise_cuda = cond_f32[0].1.clone();
    println!("=== chained (own feedback) steps={steps} d={d} ===");
    for step in 0..steps {
        let ts = (step as i32).to_le_bytes().to_vec();
        let feed_cpu: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                let b = if i == 0 {
                    noise_cpu.as_slice()
                } else {
                    pre[i].0.as_slice()
                };
                let dt = if i == 0 { noise_cpu_dt } else { pre[i].1 };
                (names[i], b, dt)
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let feed_cuda: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                let b = if i == 0 {
                    noise_cuda.as_slice()
                } else {
                    cond_f32[i].1.as_slice()
                };
                (names[i], b, DType::F32)
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let oc = tf_cpu.run_typed(&feed_cpu);
        let om = tf_cuda.run_typed(&feed_cuda);
        let vc = to_f32(&oc[0].0, oc[0].1);
        let vm = to_f32(&om[0].0, om[0].1);
        println!(
            "step {step}: cos={:.6} peak_cpu={:.4} peak_cuda={:.4}",
            cos(&vc, &vm),
            peak(&vc),
            peak(&vm)
        );
        noise_cpu = oc[0].0.clone();
        noise_cpu_dt = oc[0].1;
        noise_cuda = as_f32_bytes(&om[0].0, om[0].1);
    }

    println!("=== teacher-forced (CPU noise → both) ===");
    let mut noise = pre[0].0.clone();
    let mut noise_dt = pre[0].1;
    for step in 0..steps {
        let ts = (step as i32).to_le_bytes().to_vec();
        let noise_f32 = as_f32_bytes(&noise, noise_dt);
        let feed_cpu: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                let b = if i == 0 {
                    noise.as_slice()
                } else {
                    pre[i].0.as_slice()
                };
                let dt = if i == 0 { noise_dt } else { pre[i].1 };
                (names[i], b, dt)
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let feed_cuda: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                let b = if i == 0 {
                    noise_f32.as_slice()
                } else {
                    cond_f32[i].1.as_slice()
                };
                (names[i], b, DType::F32)
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let oc = tf_cpu.run_typed(&feed_cpu);
        let om = tf_cuda.run_typed(&feed_cuda);
        let vc = to_f32(&oc[0].0, oc[0].1);
        let vm = to_f32(&om[0].0, om[0].1);
        println!(
            "step {step}: cos={:.6} peak_cpu={:.4} peak_cuda={:.4}",
            cos(&vc, &vm),
            peak(&vc),
            peak(&vm)
        );
        noise = oc[0].0.clone();
        noise_dt = oc[0].1;
    }
    Ok(())
}
