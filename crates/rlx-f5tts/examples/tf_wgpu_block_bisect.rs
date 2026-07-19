//! Bisect CPU vs wgpu drift per-DiT-block using RLX_ONNX_TAP on each
//! `transformer_blocks.{i}/Add_3` (block residual output). Single forward
//! call (time_step=0, fresh conditioning) — no chaining/teacher-forcing.
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
const N_BLOCKS: usize = 22;

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
    let mut d = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mae = 0f64;
    let mut mx = 0f64;
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
fn as_f32_bytes(b: &[u8], dt: DType) -> Vec<u8> {
    to_f32(b, dt).iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    // TAP_NAMES=<comma-separated onnx tensor names> overrides the default
    // per-block Add_3 taps with arbitrary fine-grained sub-op taps.
    // TAP_BLOCK=<i> taps only block i (isolates readback/liveness effects
    // from having many simultaneous extra outputs); unset taps all blocks.
    let mut labels: Vec<String>;
    let taps: Vec<String> = if let Ok(spec) = std::env::var("TAP_NAMES") {
        let v: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).collect();
        labels = v.clone();
        v
    } else {
        let only: Option<usize> = std::env::var("TAP_BLOCK").ok().and_then(|s| s.parse().ok());
        let range: Vec<usize> = match only {
            Some(i) => vec![i],
            None => (0..N_BLOCKS).collect(),
        };
        labels = range.iter().map(|i| format!("block{i}")).collect();
        range
            .iter()
            .map(|i| format!("/f5_transformer/transformer_blocks.{i}/Add_3_output_0"))
            .collect()
    };
    unsafe {
        std::env::set_var("RLX_ONNX_TAP", taps.join(","));
    }
    let range = std::mem::take(&mut labels);

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

    let backend = std::env::var("BACKEND").unwrap_or_else(|_| "gpu".to_string());
    let dev2 = match backend.as_str() {
        "metal" => Device::Metal,
        "cuda" => Device::Cuda,
        "cpu" => Device::Cpu,
        _ => Device::Gpu,
    };
    let mut tf_cpu = tiny.compile_named("F5_Transformer", Device::Cpu, d, tf_named)?;
    let mut tf_wgpu = tiny.compile_named("F5_Transformer", dev2, d, tf_named)?;

    let step: i32 = std::env::var("TS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let ts = step.to_le_bytes().to_vec();
    let feed_cpu: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], pre[i].0.as_slice(), pre[i].1))
        .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
        .collect();
    let feed_wgpu: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], cond_f32[i].1.as_slice(), DType::F32))
        .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
        .collect();

    let oc = tf_cpu.run_typed(&feed_cpu);
    let om = tf_wgpu.run_typed(&feed_wgpu);
    println!(
        "outs cpu={} wgpu={} (main output + {N_BLOCKS} block taps)",
        oc.len(),
        om.len()
    );

    let vc0 = to_f32(&oc[0].0, oc[0].1);
    let vm0 = to_f32(&om[0].0, om[0].1);
    let (cos0, mae0, mx0) = stats(&vc0, &vm0);
    println!(
        "main[denoised]: cos={cos0:.6} mae={mae0:.4e} max={mx0:.4e} n={}",
        vc0.len()
    );

    for (k, label) in range.iter().enumerate() {
        let idx = 1 + k;
        if idx >= oc.len() || idx >= om.len() {
            break;
        }
        let c = to_f32(&oc[idx].0, oc[idx].1);
        let g = to_f32(&om[idx].0, om[idx].1);
        let (cos, mae, mx) = stats(&c, &g);
        let peak_c = c.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let peak_g = g.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        println!(
            "{label:>50}: cos={cos:.6} mae={mae:.4e} max={mx:.4e} peak_cpu={peak_c:.4} peak_wgpu={peak_g:.4} n={}",
            c.len()
        );
        if std::env::var("DBG_TAP_HEAD").is_ok() {
            let n = c.len().min(12);
            eprintln!("  cpu[:{n}]={:?}", &c[..n]);
            eprintln!("  gpu[:{n}]={:?}", &g[..n]);
            // per-chunk peaks for 6x1024 gemm
            if c.len() == 6144 {
                for ch in 0..6 {
                    let s = ch * 1024;
                    let e = s + 1024;
                    let pc = c[s..e].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                    let pg = g[s..e].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                    eprintln!("  chunk{ch}: peak_cpu={pc:.4} peak_gpu={pg:.4}");
                }
            }
        }
    }
    Ok(())
}
