//! DepFormer parity at step 16 using Moshi-exported backbone hidden state.
//!
//! Export reference:
//! ```bash
//! .venv-kyutai-moshi/bin/python3 scripts/export_kyutai_depformer_step.py  # or inline export to /tmp/py_dep16.json
//! ```

use anyhow::{Result, bail};
use ndarray::Array1;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::model::KyutaiTtsModel;
use rlx_kyutai_tts::sampling::StreamSampler;
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Ref {
    hidden: Vec<f32>,
    depformer_text: u32,
    audio_all: Vec<u32>,
}

fn ref_path() -> PathBuf {
    std::env::var("RLX_KYUTAI_DEP16_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/py_dep16.json"))
}

#[test]
fn depformer_greedy_matches_moshi_hidden_export() -> Result<()> {
    let path = ref_path();
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return Ok(());
    }
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rlx_kyutai_tts::download::default_kyutai_tts_dir());
    if !dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        eprintln!("skip: missing weights");
        return Ok(());
    }

    let reference: Ref = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let mut model = KyutaiTtsModel::open(&dir, cfg, Device::Cpu)?;
    let hidden = Array1::from_vec(reference.hidden);
    let mut sampler = StreamSampler::new(42, 0.0, 0.0);
    let rust = model.depformer_step(&hidden, reference.depformer_text, &mut sampler)?;

    for (i, (&r, &p)) in rust.iter().zip(reference.audio_all.iter()).enumerate() {
        if r != p {
            eprintln!("mismatch cb{i}: rust={r} py={p}");
            bail!("depformer mismatch at cb {i}");
        }
    }
    Ok(())
}
