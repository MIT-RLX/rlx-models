//! Validate `moss_tts_local_decoder` (475 nodes, single forward) runs natively —
//! the building block for host-side hierarchical sampling. Env: RLX_MOSS_DIR.
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

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
    let hidden: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect();
    let h_b: Vec<u8> = hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
    let text_tok: Vec<u8> = 5i32.to_le_bytes().to_vec();
    let prefix: Vec<u8> = (0..15).flat_map(|i| (10i32 + i).to_le_bytes()).collect();

    let comp: &'static str = Box::leak(
        std::env::var("COMP")
            .unwrap_or("moss_tts_local_decoder".into())
            .into_boxed_str(),
    );
    let mut g = model
        .compile_named(comp, Device::Cpu, 1, &[("batch", 1)])
        .map_err(|e| anyhow::anyhow!("compile {comp}: {e:#}"))?;
    let out = g.run_typed(&[
        ("global_hidden", &h_b, DType::F32),
        ("text_token_id", &text_tok, DType::I32),
        ("audio_prefix_token_ids", &prefix, DType::I32),
    ]);
    eprintln!("ldec produced {} outputs", out.len());
    for (i, (b, dt)) in out.iter().enumerate() {
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let nan = v.iter().any(|x| x.is_nan());
        eprintln!(
            "  out[{i}] {dt:?} elems={} peak={pk:.4} nan={nan} first={:?}",
            v.len(),
            &v[..4.min(v.len())]
        );
        if let Ok(d) = std::env::var("DUMP") {
            let _ = std::fs::write(format!("{d}/rlx_ldec_o{i}.f32"), b);
        }
    }
    Ok(())
}
