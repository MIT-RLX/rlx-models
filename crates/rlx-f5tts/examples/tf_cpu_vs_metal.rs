//! Task 1: ONE F5_Transformer step, CPU preprocess reused, fed to CPU vs
//! Metal transformer. Reports cosine of the denoised output.
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
fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let (mut d, mut na, mut nb, mut mae, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
        let e = (x - y).abs();
        mae += e;
        mx = mx.max(e);
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12), mae / n as f64, mx)
}

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_F5TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let (reference, sr) = read_wav(&ref_path)?;
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
    let max_duration =
        (ref_audio_len + (ref_audio_len / ref_tl * gen_tl / opts.speed as f64)) as usize;
    let d = max_duration;
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
        ("text_embed_len", TEXT_EMBED_LEN),
    ];
    let audio_b: Vec<u8> = reference
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();
    let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_b = (max_duration as i64).to_le_bytes().to_vec();
    let pre_inputs: [(&str, &[u8], DType); 3] = [
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ];

    // preprocess on CPU only — this is the "reuse preprocess from CPU" step.
    let mut cpu_pre = tiny.compile_named("F5_Preprocess", Device::Cpu, d, &named)?;
    let pre = cpu_pre.run_typed(&pre_inputs);
    anyhow::ensure!(
        pre.len() >= 7,
        "preprocess expected 7 outputs, got {}",
        pre.len()
    );
    let names = [
        "noise",
        "rope_cos",
        "rope_sin",
        "cat_mel_text",
        "cat_mel_text_drop",
        "qk_rotated_empty",
        "ref_signal_len",
    ];
    let tf_named = &[("max_duration", d), ("text_embed_len", TEXT_EMBED_LEN)];

    for (dev, label) in [(Device::Cpu, "cpu"), (Device::Metal, "metal")] {
        let mut tf = tiny.compile_named("F5_Transformer", dev, d, tf_named)?;
        let ts_b = 0i32.to_le_bytes().to_vec();
        let feed: Vec<(&str, &[u8], DType)> = (0..6)
            .map(|i| (names[i], pre[i].0.as_slice(), pre[i].1))
            .chain(std::iter::once(("time_step", ts_b.as_slice(), DType::I32)))
            .collect();
        let out = tf.run_typed(&feed);
        let v = to_f32(&out[0].0, out[0].1);
        let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        println!(
            "{label}: denoised peak={pk:.4} n={} first8={:?}",
            v.len(),
            &v[..8.min(v.len())]
        );
        if dev == Device::Cpu {
            unsafe {
                std::env::set_var("TF_CPU_STASH", "1");
            }
        }
    }

    // Compare directly.
    let mut tf_cpu = tiny.compile_named("F5_Transformer", Device::Cpu, d, tf_named)?;
    let mut tf_metal = tiny.compile_named("F5_Transformer", Device::Metal, d, tf_named)?;
    let ts_b = 0i32.to_le_bytes().to_vec();
    let feed: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], pre[i].0.as_slice(), pre[i].1))
        .chain(std::iter::once(("time_step", ts_b.as_slice(), DType::I32)))
        .collect();
    let out_cpu = tf_cpu.run_typed(&feed);
    let out_metal = tf_metal.run_typed(&feed);
    let vc = to_f32(&out_cpu[0].0, out_cpu[0].1);
    let vm = to_f32(&out_metal[0].0, out_metal[0].1);
    let (cos, mae, mx) = stats(&vc, &vm);
    println!(
        "=== denoised[0] cos={cos:.6} mae={mae:.4e} max={mx:.4e} n={} ===",
        vc.len().min(vm.len())
    );
    for (i, (co, mo)) in out_cpu.iter().zip(out_metal.iter()).enumerate().skip(1) {
        let a = to_f32(&co.0, co.1);
        let b = to_f32(&mo.0, mo.1);
        let (cos, mae, mx) = stats(&a, &b);
        println!(
            "out[{i}] cos={cos:.6} mae={mae:.4e} max={mx:.4e} n={}",
            a.len()
        );
    }
    Ok(())
}

fn read_wav(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    Ok((samples, spec.sample_rate))
}
