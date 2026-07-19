//! Validate the fused sampling graph (`moss_tts_local_fixed_sampled_frame`) + the
//! codec run natively. Feeds a real prefill global_hidden. Env: RLX_MOSS_DIR.
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

fn cfg() -> BundleConfig {
    BundleConfig {
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
    }
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let model = TinyModel::new(dir, cfg());
    let n_vq = 16usize;

    // Dummy but plausible global_hidden (small values, like a real one).
    let hidden: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect();
    let h_b: Vec<u8> = hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rep_b: Vec<u8> = vec![0u8; n_vq * 1024 * 4]; // i32 zeros
    let aru_b: Vec<u8> = 0.3f32.to_le_bytes().to_vec();
    let aut_b: Vec<u8> = (0..n_vq)
        .flat_map(|i| (0.1 + 0.05 * i as f32).to_le_bytes())
        .collect();

    let comp: &'static str = Box::leak(
        std::env::var("COMP")
            .unwrap_or("moss_tts_local_fixed_sampled_frame".into())
            .into_boxed_str(),
    );
    let mut g = model
        .compile_named(comp, Device::Cpu, 1, &[("batch", 1)])
        .map_err(|e| anyhow::anyhow!("compile local: {e:#}"))?;
    let out = g.run_typed(&[
        ("global_hidden", &h_b, DType::F32),
        ("repetition_seen_mask", &rep_b, DType::I32),
        ("assistant_random_u", &aru_b, DType::F32),
        ("audio_random_u", &aut_b, DType::F32),
    ]);
    eprintln!("local produced {} outputs", out.len());
    for (i, (b, dt)) in out.iter().enumerate() {
        let (n, s) = match dt {
            DType::I64 => (
                b.len() / 8,
                format!(
                    "i64 {:?}",
                    b.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .take(26)
                        .collect::<Vec<_>>()
                ),
            ),
            DType::I32 => (
                b.len() / 4,
                format!(
                    "i32 {:?}",
                    b.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .take(26)
                        .collect::<Vec<_>>()
                ),
            ),
            DType::Bool => (
                b.len(),
                format!("bool {:?}", b.iter().take(26).collect::<Vec<_>>()),
            ),
            _ => (
                b.len() / 4,
                format!(
                    "f32 {:?}",
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .take(26)
                        .collect::<Vec<_>>()
                ),
            ),
        };
        eprintln!("  out[{i}] {dt:?} n={n} = {s}");
    }
    Ok(())
}
