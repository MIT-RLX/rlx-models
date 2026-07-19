use half::f16;
use rlx_f5tts::InferOpts;
use rlx_f5tts::config::{DEFAULT_LOCAL_DIR, HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
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
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}
fn as_f32_bytes(b: &[u8], dt: DType) -> Vec<u8> {
    to_f32(b, dt).iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn peak(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
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

fn main() -> anyhow::Result<()> {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31);
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
    // Optional shrink for arena/sharding experiments (`MAX_DURATION=128` etc.).
    let d = std::env::var("MAX_DURATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d);
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
        ("text_embed_len", 612usize),
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
    let cond_f32: Vec<Vec<u8>> = (0..6).map(|i| as_f32_bytes(&pre[i].0, pre[i].1)).collect();
    let tf_named = &[("max_duration", d), ("text_embed_len", 612usize)];
    let dev_a = match std::env::var("DEV_A")
        .unwrap_or_else(|_| "cpu".into())
        .to_lowercase()
        .as_str()
    {
        "metal" => Device::Metal,
        "gpu" => Device::Gpu,
        _ => Device::Cpu,
    };
    let mut tf_cpu = tiny.compile_named("F5_Transformer", dev_a, d, tf_named)?;
    eprintln!("DEV_A={dev_a:?}");
    let dev_b = match std::env::var("DEV_B")
        .unwrap_or_else(|_| "cuda".into())
        .to_lowercase()
        .as_str()
    {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        _ => Device::Cuda,
    };
    let mut tf_cuda = tiny.compile_named("F5_Transformer", dev_b, d, tf_named)?;
    eprintln!("DEV_B={dev_b:?}");

    let mut noise_cpu = pre[0].0.clone();
    let mut noise_cpu_dt = pre[0].1;
    let mut noise_cuda = cond_f32[0].clone();
    let teacher = std::env::var("TEACHER").is_ok();
    eprintln!("teacher={teacher}");
    println!("step,peak_cpu,peak_cuda,cos,mean_abs_diff");
    for step in 0..steps {
        let ts = (step as i32).to_le_bytes().to_vec();
        let use_f32_a = !matches!(dev_a, Device::Cpu);
        let noise_a = if use_f32_a {
            as_f32_bytes(&noise_cpu, noise_cpu_dt)
        } else {
            noise_cpu.clone()
        };
        let feed_cpu: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                if use_f32_a {
                    let b = if i == 0 {
                        noise_a.as_slice()
                    } else {
                        cond_f32[i].as_slice()
                    };
                    (names[i], b, DType::F32)
                } else {
                    let b = if i == 0 {
                        noise_cpu.as_slice()
                    } else {
                        pre[i].0.as_slice()
                    };
                    let dt = if i == 0 { noise_cpu_dt } else { pre[i].1 };
                    (names[i], b, dt)
                }
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let noise_b = if teacher {
            as_f32_bytes(&noise_cpu, noise_cpu_dt)
        } else {
            noise_cuda.clone()
        };
        let feed_cuda: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| {
                let b = if i == 0 {
                    noise_b.as_slice()
                } else {
                    cond_f32[i].as_slice()
                };
                (names[i], b, DType::F32)
            })
            .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
            .collect();
        let oc = tf_cpu.run_typed(&feed_cpu);
        let om = tf_cuda.run_typed(&feed_cuda);
        let vc = to_f32(&oc[0].0, oc[0].1);
        let vm = to_f32(&om[0].0, om[0].1);
        let mad: f64 = vc
            .iter()
            .zip(vm.iter())
            .map(|(a, b)| (*a - *b).abs() as f64)
            .sum::<f64>()
            / vc.len() as f64;
        println!(
            "{step},{:.4},{:.4},{:.6},{:.6e}",
            peak(&vc),
            peak(&vm),
            cos(&vc, &vm),
            mad
        );
        if step == 0 && std::env::var("DBG_STEP0").is_ok() {
            let n = vc.len().min(vm.len());
            let mut errs: Vec<f32> = vc
                .iter()
                .zip(vm.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50 = errs.get(errs.len() / 2).copied().unwrap_or(0.0);
            let p99 = errs.get(errs.len() * 99 / 100).copied().unwrap_or(0.0);
            eprintln!("step0 p50={p50:.3e} p99={p99:.3e} n={n}");
        }
        noise_cpu = oc[0].0.clone();
        noise_cpu_dt = oc[0].1;
        noise_cuda = as_f32_bytes(&om[0].0, om[0].1);
    }
    Ok(())
}
