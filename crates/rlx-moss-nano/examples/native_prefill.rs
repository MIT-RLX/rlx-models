//! De-risk: compile + RUN moss prefill natively (rlx, no ORT) and sanity-check the
//! outputs (global_hidden + KV) against onnxruntime. Env: RLX_MOSS_DIR.
//! `cargo run -p rlx-moss-nano --example native_prefill --release`
use std::path::PathBuf;

use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 48000,
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
    let model = TinyModel::new(dir, cfg);

    let seq = 8usize;
    let rw = 17usize;
    // Deterministic dummy prompt (text col + 16 audio pads). Values in-range.
    let mut ids = vec![0i32; seq * rw];
    for s in 0..seq {
        ids[s * rw] = (100 + s) as i32; // text/slot column
        for a in 1..rw {
            ids[s * rw + a] = 0; // audio pad
        }
    }
    let ids_b: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mask_b: Vec<u8> = (0..seq).flat_map(|_| 1i32.to_le_bytes()).collect();

    let comp: &'static str = Box::leak(
        std::env::var("COMP")
            .unwrap_or("moss_tts_prefill".into())
            .into_boxed_str(),
    );
    let named = &[("batch", 1usize), ("prefill_seq", seq)];
    let t0 = std::time::Instant::now();
    let mut g = model
        .compile_named(comp, Device::Cpu, seq, named)
        .map_err(|e| anyhow::anyhow!("compile {comp}: {e:#}"))?;
    eprintln!("compiled {comp} in {:.1}s", t0.elapsed().as_secs_f32());

    let out = g.run_typed(&[
        ("input_ids", &ids_b, DType::I32),
        ("attention_mask", &mask_b, DType::I32),
    ]);
    eprintln!("{comp} produced {} outputs", out.len());
    if let Ok(p) = std::env::var("DUMP0") {
        let _ = std::fs::write(&p, &out[0].0);
        eprintln!("dumped out[0] to {p}");
    }
    if let Ok(dir) = std::env::var("DUMP_ALL") {
        let _ = std::fs::create_dir_all(&dir);
        for (i, (b, _)) in out.iter().enumerate() {
            let _ = std::fs::write(format!("{dir}/o{i}.f32"), b);
        }
        eprintln!("dumped {} outputs to {dir}", out.len());
    }
    let show: Vec<usize> = std::env::var("SHOW")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| (0..out.len().min(3)).collect());
    for &i in &show {
        let Some((bytes, dt)) = out.get(i) else {
            continue;
        };
        match dt {
            DType::I64 => {
                let v: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                eprintln!(
                    "  out[{i}] i64 elems={} first={:?}",
                    v.len(),
                    &v[..8.min(v.len())]
                );
            }
            DType::I32 => {
                let v: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                eprintln!(
                    "  out[{i}] i32 elems={} first={:?}",
                    v.len(),
                    &v[..8.min(v.len())]
                );
            }
            DType::Bool => {
                eprintln!(
                    "  out[{i}] bool elems={} first={:?}",
                    bytes.len(),
                    &bytes[..8.min(bytes.len())]
                );
            }
            _ => {
                let v: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let nan = v.iter().any(|x| x.is_nan());
                eprintln!(
                    "  out[{i}] f32 elems={} peak={pk:.4} nan={nan} first={:?}",
                    v.len(),
                    &v[..4.min(v.len())]
                );
            }
        }
    }
    Ok(())
}
