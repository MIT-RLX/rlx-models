//! Basic load + synthetic roundtrip.

use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE, default_mimi_dir};
use std::f32::consts::PI;

fn model_dir() -> Option<std::path::PathBuf> {
    let dir = default_mimi_dir();
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        std::env::var("RLX_MIMI_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.join("model.safetensors").is_file())
    }
}

#[test]
fn codec_loads_weights() {
    let Some(dir) = model_dir() else {
        eprintln!("skip codec_loads_weights: run `just fetch-mimi`");
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open mimi");
    assert_eq!(codec.config().sampling_rate, SAMPLE_RATE);
    assert_eq!(codec.config().num_quantizers, 32);
    assert_eq!(codec.config().samples_per_codec_frame(), 1920);
}

#[test]
fn encode_decode_synthetic() {
    let Some(dir) = model_dir() else {
        eprintln!("skip encode_decode_synthetic: missing weights");
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open mimi");
    let n = SAMPLE_RATE as usize / 2;
    let pcm: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.2)
        .collect();
    let (codes, recon, stats) = codec.roundtrip_pcm(&pcm, None).expect("roundtrip");
    assert!(codes.num_frames() >= 4);
    assert_eq!(codes.num_quantizers, 32);
    assert!(recon.len() > n / 4);
    let corr = pearson(&pcm, &recon);
    assert!(corr > 0.5, "roundtrip correlation {corr:.3}");
    assert!(stats.encode_ms > 0.0 && stats.decode_ms > 0.0);
}

#[test]
fn hf_layout_roundtrip() {
    let Some(dir) = model_dir() else {
        return;
    };
    let codec = MimiCodec::open(&dir).expect("open");
    let pcm = vec![0.1f32; SAMPLE_RATE as usize / 4];
    let codes = codec.encode_pcm(&pcm, None).expect("encode");
    let hf = codes.to_hf_layout();
    let back = MimiCodes::from_hf_layout(hf);
    assert_eq!(back.frames, codes.frames);
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let n = n as f32;
    let ma = a.iter().take(n as usize).sum::<f32>() / n;
    let mb = b.iter().take(n as usize).sum::<f32>() / n;
    let mut num = 0f32;
    let mut da = 0f32;
    let mut db = 0f32;
    for i in 0..n as usize {
        let dx = a[i] - ma;
        let dy = b[i] - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-8)
}
