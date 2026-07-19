//! Debug: compile the (tapped) F5_Transformer and run it in rlx on ort's saved
//! preprocess outputs, dumping chosen tap outputs. Compares against ort taps.
//! Env: TF_ONNX (tapped onnx path), SC (dir with pp_*.f16), TS (time step).
use std::path::PathBuf;

fn load_f16(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

fn main() -> anyhow::Result<()> {
    let sc = std::env::var("SC").expect("SC dir");
    let tf_onnx = std::env::var("TF_ONNX").expect("TF_ONNX path");
    let ts: i32 = std::env::var("TS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let d = 141usize;
    let named = &[("max_duration", d), ("text_embed_len", 612usize)];

    let onnx = PathBuf::from(&tf_onnx);
    let dir = onnx.parent().unwrap().to_path_buf();
    let cfg = rlx_tiny_tts::BundleConfig {
        model: String::new(),
        sample_rate: 24000,
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
    // TinyModel resolves `<comp>.onnx` in dir; the tapped file must be named so.
    let comp: &'static str = Box::leak(
        onnx.file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string()
            .into_boxed_str(),
    );
    let model = rlx_tiny_tts::model::TinyModel::new(dir, cfg);
    let mut g = model
        .compile_named(comp, rlx_runtime::Device::Cpu, d, named)
        .map_err(|e| anyhow::anyhow!("compile: {e:#}"))?;

    use rlx_runtime::DType::F16;
    let noise = load_f16(&format!("{sc}/pp_noise.f16"));
    let rc = load_f16(&format!("{sc}/pp_rope_cos.f16"));
    let rs = load_f16(&format!("{sc}/pp_rope_sin.f16"));
    let cm = load_f16(&format!("{sc}/pp_cat_mel_text.f16"));
    let cmd = load_f16(&format!("{sc}/pp_cat_mel_text_drop.f16"));
    let qk = load_f16(&format!("{sc}/pp_qk_rotated_empty.f16"));
    let tsb = ts.to_le_bytes().to_vec();
    let out = g.run_typed(&[
        ("noise", &noise, F16),
        ("rope_cos", &rc, F16),
        ("rope_sin", &rs, F16),
        ("cat_mel_text", &cm, F16),
        ("cat_mel_text_drop", &cmd, F16),
        ("qk_rotated_empty", &qk, F16),
        ("time_step", &tsb, rlx_runtime::DType::I32),
    ]);
    // Print each output's peak + first values (order = graph output decl order).
    for (i, (bytes, dt)) in out.iter().enumerate() {
        let v: Vec<f32> = match dt {
            rlx_runtime::DType::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            _ => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };
        let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        eprintln!(
            "ts={ts} out[{i}] elems={} peak={pk:.4} first={:?}",
            v.len(),
            &v[..5.min(v.len())]
        );
        if i == 0 {
            if let Ok(dir) = std::env::var("SC") {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/tf_rlx_denoised_ts{ts}.f32"), bytes);
            }
        }
    }
    Ok(())
}
