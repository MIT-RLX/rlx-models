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
const TAPS: &str = concat!(
    "/f5_text_embed/Cast_1_output_0,",
    "/f5_text_embed/Less_output_0,",
    "/f5_text_embed/Where_output_0,",
    "/f5_text_embed/Gather_output_0"
);

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
        DType::Bool => b.iter().map(|&x| x as f32).collect(),
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

fn main() -> anyhow::Result<()> {
    unsafe {
        std::env::set_var("RLX_ONNX_TAP", TAPS);
    }
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(&ref_path)?;
    let sr = r.spec().sample_rate;
    let m = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    let reference: Vec<f32> = r.samples::<i32>().map(|s| s.unwrap() as f32 / m).collect();
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
    let opts = InferOpts { nfe: 2, speed: 1.0 };
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
    let inputs: [(&str, &[u8], DType); 3] = [
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ];
    let mut cpu = tiny.compile_named("F5_Preprocess", Device::Cpu, d, &named)?;
    let mut metal = tiny.compile_named("F5_Preprocess", Device::Metal, d, &named)?;
    let cpu_out = cpu.run_typed(&inputs);
    let metal_out = metal.run_typed(&inputs);
    let tap_names: Vec<&str> = TAPS.split(',').collect();
    println!("outs cpu={} metal={}", cpu_out.len(), metal_out.len());
    for i in 0..cpu_out.len().min(metal_out.len()) {
        let c = to_f32(&cpu_out[i].0, cpu_out[i].1);
        let g = to_f32(&metal_out[i].0, metal_out[i].1);
        let (cos, mae, mx) = stats(&c, &g);
        let label = if i < 7 {
            format!("main[{i}] dt={:?}", cpu_out[i].1)
        } else {
            format!(
                "tap[{}]={} dt_c={:?} dt_m={:?}",
                i - 7,
                tap_names.get(i - 7).unwrap_or(&"?"),
                cpu_out[i].1,
                metal_out[i].1
            )
        };
        let preview_c: Vec<f32> = c.iter().take(8).copied().collect();
        let preview_m: Vec<f32> = g.iter().take(8).copied().collect();
        if i == 8 {
            println!(
                "Less nbytes cpu={} metal={} uniq_cpu={:?} uniq_metal={:?}",
                cpu_out[i].0.len(),
                metal_out[i].0.len(),
                {
                    let mut u = c.clone();
                    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    u.dedup();
                    u
                },
                {
                    let mut u = g.clone();
                    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    u.dedup();
                    u
                }
            );
        }
        if i == 8 || i == 9 {
            let lo = 285usize.min(c.len().saturating_sub(1));
            let hi = (295).min(c.len()).min(g.len());
            println!(
                "{label} slice[{lo}..{hi}] cpu={:?} metal={:?}",
                &c[lo..hi],
                &g[lo..hi]
            );
        }
        if i == 9 {
            let mut first = None;
            for j in 0..c.len().min(g.len()) {
                if (c[j] - g[j]).abs() > 1e-3 {
                    first = Some(j);
                    break;
                }
            }
            println!(
                "where first_diff={first:?} cpu_at={:?} metal_at={:?} cpu_tail={:?} metal_tail={:?}",
                first.map(|j| c[j]),
                first.map(|j| g[j]),
                &c[c.len().saturating_sub(8)..],
                &g[g.len().saturating_sub(8)..]
            );
        }
        println!(
            "{label}: cos={cos:.6} mae={mae:.3e} max={mx:.3e} n={} cpu={preview_c:?} metal={preview_m:?}",
            c.len().min(g.len())
        );
    }
    Ok(())
}
