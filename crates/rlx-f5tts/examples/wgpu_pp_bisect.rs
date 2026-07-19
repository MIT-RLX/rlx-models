//! Bisect F5_Preprocess CPU↔CUDA + micro-op probes (GRN reduce, depthwise conv).
//!
//! ```bash
//! RLX_F5TTS_DIR=weights/tts/f5tts cargo run -p rlx-f5tts --release \
//!   --example cuda_pp_bisect --features cuda
//! RLX_BISECT_GRN=1 …  # also run micro-op probes
//! ```

use std::path::PathBuf;

use half::f16;
use rlx_f5tts::config::{HOP_LENGTH, Layout, Vocab};
use rlx_f5tts::dsp::preprocess_ref_audio;
use rlx_f5tts::tokenize::{encode, normalize_ref_text, text_len};
use rlx_f5tts::{DEFAULT_LOCAL_DIR, InferOpts};
use rlx_runtime::{DType, Device, Graph, Op, Session, Shape};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

const TEXT_EMBED_LEN: usize = 612;

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

    // Optional opt bisect: RLX_BISECT_OPTS=dce,fold,fusion (comma flags to KEEP on;
    // unset = defaults; "none" = all off). Cache key now includes these flags.
    let mut copts = rlx_runtime::CompileOptions::default();
    if let Ok(spec) = std::env::var("RLX_BISECT_OPTS") {
        let keep: std::collections::HashSet<_> = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if spec.trim() == "none" || keep.contains(&"none") {
            copts.dce = false;
            copts.constant_folding = false;
            copts.fusion_opts.skip_fusion = true;
        } else {
            copts.dce = keep.contains(&"dce");
            copts.constant_folding = keep.contains(&"fold");
            copts.fusion_opts.skip_fusion = !keep.contains(&"fusion");
        }
        eprintln!(
            "[bisect-opts] dce={} fold={} skip_fusion={}",
            copts.dce, copts.constant_folding, copts.fusion_opts.skip_fusion
        );
    }
    let mut cpu_g =
        tiny.compile_named_with_options("F5_Preprocess", Device::Cpu, d, &named, copts.clone())?;
    let mut cuda_g =
        tiny.compile_named_with_options("F5_Preprocess", Device::Gpu, d, &named, copts)?;

    let audio_b = f32s_to_f16_bytes(&reference);
    let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_b = (max_duration as i64).to_le_bytes().to_vec();
    let inputs: [(&str, &[u8], DType); 3] = [
        ("audio", &audio_b, DType::F16),
        ("text_ids", &text_ids_b, DType::I32),
        ("max_duration", &md_b, DType::I64),
    ];

    let cpu_out = cpu_g.run_typed(&inputs);
    let cuda_out = cuda_g.run_typed(&inputs);
    println!(
        "preprocess outputs: cpu={} cuda={}",
        cpu_out.len(),
        cuda_out.len()
    );
    for (i, ((cb, cdt), (gb, gdt))) in cpu_out.iter().zip(cuda_out.iter()).enumerate() {
        let c = to_f32(cb, *cdt);
        let g = to_f32(gb, *gdt);
        let (cos, mae, mx) = stats(&c, &g);
        println!(
            "out[{i}] cos={cos:.6} mae={mae:.4e} maxabs={mx:.4e} n={}",
            c.len().min(g.len())
        );
    }

    if cpu_out.len() > 3 && cuda_out.len() > 3 {
        let c = to_f32(&cpu_out[3].0, cpu_out[3].1);
        let g = to_f32(&cuda_out[3].0, cuda_out[3].1);
        let (t_frames, width) = (d, TEXT_EMBED_LEN);
        assert_eq!(c.len(), t_frames * width);
        let mel_c: Vec<f32> = (0..t_frames)
            .flat_map(|i| c[i * width..i * width + 100].iter().copied())
            .collect();
        let mel_g: Vec<f32> = (0..t_frames)
            .flat_map(|i| g[i * width..i * width + 100].iter().copied())
            .collect();
        let txt_c: Vec<f32> = (0..t_frames)
            .flat_map(|i| c[i * width + 100..(i + 1) * width].iter().copied())
            .collect();
        let txt_g: Vec<f32> = (0..t_frames)
            .flat_map(|i| g[i * width + 100..(i + 1) * width].iter().copied())
            .collect();
        let (cos_m, _, _) = stats(&mel_c, &mel_g);
        let (cos_t, mae_t, mx_t) = stats(&txt_c, &txt_g);
        println!("cat_mel mel100 cos={cos_m:.6}");
        println!("cat_mel text512 cos={cos_t:.6} mae={mae_t:.4e} maxabs={mx_t:.4e}");
    }

    if std::env::var("RLX_BISECT_GRN").is_ok() {
        bisect_grn()?;
    }

    Ok(())
}

fn bisect_grn() -> anyhow::Result<()> {
    use rlx_runtime::op::{Activation, BinaryOp, ReduceOp};

    let b = 1usize;
    let l = 64usize;
    let c = 128usize;
    let mut rng = 0x1234_5678_u64;
    let mut x = vec![0f32; b * l * c];
    for v in &mut x {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((rng >> 33) as i32 as f32) / (1u32 << 24) as f32;
    }
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs = [("x", x_bytes.as_slice(), DType::F32)];

    // ReduceL2 over axis 1: sqrt(sum(x^2, axis=1, keepdims)) → [1,1,C]
    {
        let mut g = Graph::new("grn_l2");
        let xin = g.input("x", Shape::new(&[b, l, c], DType::F32));
        let sq = g.add_node(
            Op::Binary(BinaryOp::Mul),
            vec![xin, xin],
            Shape::new(&[b, l, c], DType::F32),
        );
        let summed = g.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![1],
                keep_dim: true,
            },
            vec![sq],
            Shape::new(&[b, 1, c], DType::F32),
        );
        let out = g.add_node(
            Op::Activation(Activation::Sqrt),
            vec![summed],
            Shape::new(&[b, 1, c], DType::F32),
        );
        g.set_outputs(vec![out]);
        let cpu = Session::new(Device::Cpu)
            .compile(g.clone())
            .run_typed(&inputs);
        let cuda = Session::new(Device::Gpu).compile(g).run_typed(&inputs);
        let (cos, mae, mx) = stats(&to_f32(&cpu[0].0, cpu[0].1), &to_f32(&cuda[0].0, cuda[0].1));
        println!("GRN ReduceL2(axis=1) cos={cos:.8} mae={mae:.4e} maxabs={mx:.4e}");
    }

    // ReduceMean last axis → [1,L,1]
    {
        let mut g = Graph::new("grn_mean");
        let xin = g.input("x", Shape::new(&[b, l, c], DType::F32));
        let out = g.add_node(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes: vec![2],
                keep_dim: true,
            },
            vec![xin],
            Shape::new(&[b, l, 1], DType::F32),
        );
        g.set_outputs(vec![out]);
        let cpu = Session::new(Device::Cpu)
            .compile(g.clone())
            .run_typed(&inputs);
        let cuda = Session::new(Device::Gpu).compile(g).run_typed(&inputs);
        let (cos, mae, mx) = stats(&to_f32(&cpu[0].0, cpu[0].1), &to_f32(&cuda[0].0, cuda[0].1));
        println!("GRN ReduceMean(axis=-1) cos={cos:.8} mae={mae:.4e} maxabs={mx:.4e}");
    }

    // Depthwise conv: [1,C,1,L] × [C,1,1,7] groups=C
    {
        let c_ch = 32usize;
        let len = 40usize;
        let k = 7usize;
        let mut act = vec![0f32; c_ch * len];
        let mut w = vec![0f32; c_ch * k];
        for (i, v) in act.iter_mut().enumerate() {
            *v = (i as f32 * 0.01).sin();
        }
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.1;
        }
        let mut g = Graph::new("dwconv");
        let x = g.input("x", Shape::new(&[1, c_ch, 1, len], DType::F32));
        let ww = g.param("w", Shape::new(&[c_ch, 1, 1, k], DType::F32));
        let y = g.add_node(
            Op::Conv {
                kernel_size: vec![1, k],
                stride: vec![1, 1],
                padding: vec![0, 3],
                dilation: vec![1, 1],
                groups: c_ch,
            },
            vec![x, ww],
            Shape::new(&[1, c_ch, 1, len], DType::F32),
        );
        g.set_outputs(vec![y]);
        let mut cpu = Session::new(Device::Cpu).compile(g.clone());
        cpu.set_param("w", &w);
        let mut cuda = Session::new(Device::Gpu).compile(g);
        cuda.set_param("w", &w);
        let xb: Vec<u8> = act.iter().flat_map(|v| v.to_le_bytes()).collect();
        let inputs = [("x", xb.as_slice(), DType::F32)];
        let (cos, mae, mx) = stats(
            &to_f32(&cpu.run_typed(&inputs)[0].0, DType::F32),
            &to_f32(&cuda.run_typed(&inputs)[0].0, DType::F32),
        );
        println!("depthwise Conv groups={c_ch} cos={cos:.8} mae={mae:.4e} maxabs={mx:.4e}");
    }

    // Gather embedding-style
    {
        let vocab = 100usize;
        let dim = 16usize;
        let n_idx = 8usize;
        let mut table = vec![0f32; vocab * dim];
        let mut idx = vec![0f32; n_idx];
        for (i, v) in table.iter_mut().enumerate() {
            *v = (i as f32) * 0.001;
        }
        for (i, v) in idx.iter_mut().enumerate() {
            *v = ((i * 7) % vocab) as f32;
        }
        let mut g = Graph::new("gather");
        let tab = g.param("t", Shape::new(&[vocab, dim], DType::F32));
        let ix = g.input("i", Shape::new(&[n_idx], DType::F32));
        let y = g.add_node(
            Op::Gather { axis: 0 },
            vec![tab, ix],
            Shape::new(&[n_idx, dim], DType::F32),
        );
        g.set_outputs(vec![y]);
        let mut cpu = Session::new(Device::Cpu).compile(g.clone());
        cpu.set_param("t", &table);
        let mut cuda = Session::new(Device::Gpu).compile(g);
        cuda.set_param("t", &table);
        let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let inputs = [("i", ib.as_slice(), DType::F32)];
        let (cos, mae, mx) = stats(
            &to_f32(&cpu.run_typed(&inputs)[0].0, DType::F32),
            &to_f32(&cuda.run_typed(&inputs)[0].0, DType::F32),
        );
        println!("Gather axis=0 cos={cos:.8} mae={mae:.4e} maxabs={mx:.4e}");
    }

    Ok(())
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut mae = 0.0f64;
    let mut mx = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        mae += d;
        mx = mx.max(d);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    };
    (cos, mae / n as f64, mx)
}

fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
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

fn f32s_to_f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
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
