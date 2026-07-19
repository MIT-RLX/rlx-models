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

fn main() -> anyhow::Result<()> {
    let model = PathBuf::from(DEFAULT_LOCAL_DIR);
    let ref_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let (reference, sr) = {
        let mut r = hound::WavReader::open(&ref_path)?;
        let samples: Vec<f32> = r
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect();
        (samples, r.spec().sample_rate)
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
    let mut tf_cpu = tiny.compile_named("F5_Transformer", Device::Cpu, d, tf_named)?;
    let mut tf_cuda = tiny.compile_named("F5_Transformer", Device::Cuda, d, tf_named)?;
    let ts = 0i32.to_le_bytes().to_vec();
    let feed_cpu: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], pre[i].0.as_slice(), pre[i].1))
        .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
        .collect();
    let feed_cuda: Vec<(&str, &[u8], DType)> = (0..6)
        .map(|i| (names[i], cond_f32[i].as_slice(), DType::F32))
        .chain(std::iter::once(("time_step", ts.as_slice(), DType::I32)))
        .collect();
    let oc = tf_cpu.run_typed(&feed_cpu);
    let om = tf_cuda.run_typed(&feed_cuda);
    let vc = to_f32(&oc[0].0, oc[0].1);
    let vm = to_f32(&om[0].0, om[0].1);
    // mel is [1, 100, d] = 100 * 580 = 58000
    let mel = 100usize;
    let frames = d;
    assert_eq!(vc.len(), mel * frames);
    let mut max_err = 0f32;
    let mut max_i = 0usize;
    let mut sum_err = 0f64;
    let mut row_err = vec![0f64; frames];
    for i in 0..vc.len() {
        let e = (vc[i] - vm[i]).abs();
        sum_err += e as f64;
        if e > max_err {
            max_err = e;
            max_i = i;
        }
        row_err[i % frames] += e as f64;
    }
    println!(
        "mad={:.6e} max_err={:.6} at i={} (mel={},frame={})",
        sum_err / vc.len() as f64,
        max_err,
        max_i,
        max_i / frames,
        max_i % frames
    );
    // top-10 frames by error
    let mut idx: Vec<usize> = (0..frames).collect();
    idx.sort_by(|&a, &b| row_err[b].partial_cmp(&row_err[a]).unwrap());
    print!("top frames by abs-err sum:");
    for &i in idx.iter().take(10) {
        print!(" {i}:{:.3}", row_err[i]);
    }
    println!();
    // hist of per-element abs err
    let mut buckets = [0u64; 6];
    for i in 0..vc.len() {
        let e = (vc[i] - vm[i]).abs();
        let b = if e < 1e-4 {
            0
        } else if e < 1e-3 {
            1
        } else if e < 1e-2 {
            2
        } else if e < 1e-1 {
            3
        } else if e < 1.0 {
            4
        } else {
            5
        };
        buckets[b] += 1;
    }
    println!("err hist [<1e-4,<1e-3,<1e-2,<1e-1,<1,>=1]: {:?}", buckets);

    // Also: feed Metal-style — CPU noise as f32 to BOTH and compare CUDA vs a second CPU run in... skip

    // Try CUDA with RLX_CUDA_FORCE_ATTENTION_ROW
    Ok(())
}
