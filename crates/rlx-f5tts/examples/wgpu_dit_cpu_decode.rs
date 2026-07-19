//! DiT on wgpu, Decode on CPU — bisect e2e failure.
use half::f16;
use rlx_f5tts::config::{HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use rlx_f5tts::{DEFAULT_LOCAL_DIR, InferOpts, SAMPLE_RATE, peak_amplitude};
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
fn as_f32(b: &[u8], dt: DType) -> Vec<u8> {
    to_f32(b, dt).iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn as_f16(b: &[u8], dt: DType) -> Vec<u8> {
    to_f32(b, dt)
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
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
        sample_rate: SAMPLE_RATE,
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
    let names = [
        "noise",
        "rope_cos",
        "rope_sin",
        "cat_mel_text",
        "cat_mel_text_drop",
        "qk_rotated_empty",
    ];
    let cond: Vec<Vec<u8>> = (0..6).map(|i| as_f32(&pre[i].0, pre[i].1)).collect();
    let ref_signal_len = {
        let b = &pre[6].0;
        if b.len() >= 8 {
            i64::from_le_bytes(b[..8].try_into().unwrap())
        } else {
            i32::from_le_bytes(b[..4].try_into().unwrap()) as i64
        }
    };
    let mut noise = cond[0].clone();
    let mut tf = tiny.compile_named(
        "F5_Transformer",
        Device::Gpu,
        d,
        &[("max_duration", d), ("text_embed_len", 612)],
    )?;
    let ode = opts.nfe.saturating_sub(1).max(1);
    for step in 0..ode {
        let ts = (step as i32).to_le_bytes().to_vec();
        let feed: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                (
                    names[i],
                    if i == 0 {
                        noise.as_slice()
                    } else {
                        cond[i].as_slice()
                    },
                    DType::F32,
                )
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let out = tf.run_typed(&feed);
        noise = as_f32(&out[0].0, out[0].1);
        if step == 0 || step + 1 == ode {
            let v = to_f32(&noise, DType::F32);
            let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            eprintln!("dit step {step}/{ode} peak={pk:.4}");
        }
    }
    // CPU DiT for reference
    let mut noise_cpu = pre[0].0.clone();
    let mut noise_cpu_dt = pre[0].1;
    let mut tf_cpu = tiny.compile_named(
        "F5_Transformer",
        Device::Cpu,
        d,
        &[("max_duration", d), ("text_embed_len", 612)],
    )?;
    for step in 0..ode {
        let ts = (step as i32).to_le_bytes().to_vec();
        let feed: Vec<(&str, &[u8], DType)> = (0..6)
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
        let out = tf_cpu.run_typed(&feed);
        noise_cpu = out[0].0.clone();
        noise_cpu_dt = out[0].1;
    }
    let vc = to_f32(&noise_cpu, noise_cpu_dt);
    let vm = to_f32(&noise, DType::F32);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..vc.len().min(vm.len()) {
        let a = vc[i] as f64;
        let b = vm[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    let mean_abs = vc
        .iter()
        .zip(vm.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / vc.len() as f32;
    eprintln!(
        "final denoised cos={cos:.6} mean|c-w|={mean_abs:.5} peak_cpu={:.4} peak_wgpu={:.4}",
        vc.iter().fold(0.0f32, |m, &x| m.max(x.abs())),
        vm.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    );

    // Decode on CPU with f16 edge
    let denoised_f16 = as_f16(&noise, DType::F32);
    let rl = ref_signal_len.to_le_bytes().to_vec();
    let mut dec = tiny.compile_named("F5_Decode", Device::Cpu, d, &[("max_duration", d)])?;
    let out = dec.run_typed(&[
        ("denoised", &denoised_f16, DType::F16),
        ("ref_signal_len", &rl, DType::I64),
    ]);
    let mut wav = to_f32(&out[0].0, out[0].1);
    let skip = (ref_signal_len.max(0) as usize).saturating_mul(HOP_LENGTH);
    if wav.len() > skip + HOP_LENGTH {
        wav = wav[skip..].to_vec();
    }
    let peak = peak_amplitude(&wav);
    eprintln!(
        "hybrid dit=wgpu decode=cpu peak={peak:.4} samples={}",
        wav.len()
    );
    let out_path = PathBuf::from("tmp/f5tts_wavs/hybrid_wgpu_dit.wav");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    rlx_f5tts::write_wav(&wav, SAMPLE_RATE, &out_path)?;
    Ok(())
}
