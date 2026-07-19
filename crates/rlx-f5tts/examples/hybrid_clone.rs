//! Hybrid: F5_Preprocess on one device, F5_Transformer NFE loop on another,
//! F5_Decode back on the preprocess device. Isolates which stage produces
//! Whisper-garbage on Metal.
//! Env: PRE=cpu|metal|cuda (default cpu), TF=… (default metal), DEC=… (default cpu), NFE.
use half::f16;
use rlx_f5tts::DEFAULT_LOCAL_DIR;
use rlx_f5tts::config::{HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::{preprocess_ref_audio, soft_peak_limit};
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};
use std::path::PathBuf;

const TEXT_EMBED_LEN: usize = 612;
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const REF_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";
const FOX: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

fn dev_from(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "gpu" | "wgpu" => Device::Gpu,
        _ => Device::Cpu,
    }
}

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
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
fn f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}
fn i64_scalar(b: &[u8]) -> i64 {
    if b.len() >= 8 {
        i64::from_le_bytes(b[..8].try_into().unwrap())
    } else if b.len() >= 4 {
        let as_i = i32::from_le_bytes(b[..4].try_into().unwrap()) as i64;
        let as_f = f32::from_le_bytes(b[..4].try_into().unwrap());
        if as_f.is_finite() && as_f.abs() < 1.0e7 && (as_f - as_f.round()).abs() < 1e-3 {
            as_f.round() as i64
        } else {
            as_i
        }
    } else {
        0
    }
}
fn float_bytes_for(dev: Device, x: &[f32]) -> (Vec<u8>, DType) {
    match dev {
        Device::Metal
        | Device::Mlx
        | Device::Gpu
        | Device::Vulkan
        | Device::Cuda
        | Device::Rocm => (x.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32),
        _ => (f16_bytes(x), DType::F16),
    }
}
fn as_feed(bytes: &[u8], dt: DType, dev: Device) -> (Vec<u8>, DType) {
    float_bytes_for(dev, &to_f32(bytes, dt))
}

fn main() -> anyhow::Result<()> {
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let pre_dev = dev_from(&std::env::var("PRE").unwrap_or_else(|_| "cpu".into()));
    let tf_dev = dev_from(&std::env::var("TF").unwrap_or_else(|_| "metal".into()));
    let dec_dev = dev_from(&std::env::var("DEC").unwrap_or_else(|_| "cpu".into()));
    let nfe: usize = std::env::var("NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&ref_path)?;
    let sr = r.spec().sample_rate;
    let m = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let reference: Vec<f32> = r.samples::<i32>().map(|s| s.unwrap() as f32 / m).collect();
    let ref_text = normalize_ref_text(REF_TEXT);
    let reference = preprocess_ref_audio(&reference, sr);
    let vocab = Vocab::load(&model)?;
    let text_ids = encode(&ref_text, TEXT, &vocab);
    let n = reference.len();
    let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
    let ref_tl = text_len(&ref_text).max(1) as f64;
    let gen_tl = text_len(TEXT) as f64;
    let max_duration = (ref_audio_len + (ref_audio_len / ref_tl * gen_tl)) as usize;
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
    let audio_b = f16_bytes(&reference);
    let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_b = (max_duration as i64).to_le_bytes().to_vec();

    println!("PRE={pre_dev:?} TF={tf_dev:?} DEC={dec_dev:?} NFE={nfe}");
    let mut pre_g = tiny.compile_named("F5_Preprocess", pre_dev, d, &named)?;
    let pre = pre_g.run_typed(&[
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ]);
    let (mut noise, noise_dt) = as_feed(&pre[0].0, pre[0].1, tf_dev);
    let (rope_cos, rope_cos_dt) = as_feed(&pre[1].0, pre[1].1, tf_dev);
    let (rope_sin, rope_sin_dt) = as_feed(&pre[2].0, pre[2].1, tf_dev);
    let (cat_mel, cat_mel_dt) = as_feed(&pre[3].0, pre[3].1, tf_dev);
    let (cat_mel_drop, cat_mel_drop_dt) = as_feed(&pre[4].0, pre[4].1, tf_dev);
    let (qk, qk_dt) = as_feed(&pre[5].0, pre[5].1, tf_dev);
    let ref_signal_len = i64_scalar(&pre[6].0);

    let ode_steps = nfe.saturating_sub(1).max(1);
    let tf_named = &[("max_duration", d), ("text_embed_len", TEXT_EMBED_LEN)];
    let mut tf = tiny.compile_named("F5_Transformer", tf_dev, d, tf_named)?;
    for step in 0..ode_steps {
        let ts_b = (step as i32).to_le_bytes().to_vec();
        let out = tf.run_typed(&[
            ("noise", &noise, noise_dt),
            ("rope_cos", &rope_cos, rope_cos_dt),
            ("rope_sin", &rope_sin, rope_sin_dt),
            ("cat_mel_text", &cat_mel, cat_mel_dt),
            ("cat_mel_text_drop", &cat_mel_drop, cat_mel_drop_dt),
            ("qk_rotated_empty", &qk, qk_dt),
            ("time_step", &ts_b, DType::I32),
        ]);
        let (out_bytes, out_dt) = out.into_iter().next().unwrap();
        let v = to_f32(&out_bytes, out_dt);
        let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let nan_ct = v.iter().filter(|x| x.is_nan()).count();
        let inf_ct = v.iter().filter(|x| x.is_infinite()).count();
        if step == 0 || step + 1 == ode_steps || step % 8 == 0 {
            eprintln!(
                "[hybrid-dbg] denoised step {step}/{ode_steps} peak={pk:.4} nan={nan_ct} inf={inf_ct} n={}",
                v.len()
            );
        }
        let (n2, _) = as_feed(&out_bytes, out_dt, tf_dev);
        noise = n2;
    }

    // Decode edge: match decode device dtype.
    let (noise_dec, noise_dec_dt) = as_feed(&noise, noise_dt, dec_dev);
    let mut dec_g = tiny.compile_named("F5_Decode", dec_dev, d, &[("max_duration", d)])?;
    let rl_b = ref_signal_len.to_le_bytes().to_vec();
    let dec = dec_g.run_typed(&[
        ("denoised", &noise_dec, noise_dec_dt),
        ("ref_signal_len", &rl_b, DType::I64),
    ]);
    let (bytes, dt) = dec.into_iter().next().unwrap();
    let mut wav = to_f32(&bytes, dt);
    let skip = (ref_signal_len.max(0) as usize).saturating_mul(HOP_LENGTH);
    if wav.len() > skip + HOP_LENGTH {
        wav = wav[skip..].to_vec();
    }
    soft_peak_limit(&mut wav, 0.95);
    let peak = wav.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    println!("peak={peak:.4} len={}", wav.len());

    let wd = whisper_dir().ok_or_else(|| anyhow::anyhow!("Whisper weights required"))?;
    let pcm = resample_linear(&wav, rlx_f5tts::SAMPLE_RATE, WR as u32);
    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let transcript = w.transcribe_greedy(&pcm)?;
    let lower = transcript.to_lowercase();
    let hits = FOX.iter().filter(|word| lower.contains(*word)).count();
    println!("whisper: {transcript}");
    println!("fox: {hits}/6");
    rlx_f5tts::write_wav(
        &wav,
        rlx_f5tts::SAMPLE_RATE,
        &PathBuf::from(format!(
            "tmp/f5tts_wavs/hybrid_{:?}_{:?}_{:?}.wav",
            pre_dev, tf_dev, dec_dev
        )),
    )?;
    Ok(())
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
    for name in ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"] {
        let p = cache.join(name);
        if whisper_ready(&p) {
            return Some(p);
        }
    }
    None
}
fn whisper_ready(dir: &std::path::Path) -> bool {
    dir.join("model.safetensors").is_file()
        && dir.join("config.json").is_file()
        && dir.join("tokenizer.json").is_file()
}

fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}
