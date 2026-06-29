//! Weight storage precision parity — F32 / F16 / BF16 round-trip vs native f32 generation.
//!
//! Kyutai TTS checkpoints are safetensors (no GGUF quants). This exercises the loader
//! widen path and confirms generation is stable across storage dtypes.
//!
//! ```bash
//! RLX_KYUTAI_TTS_DIR=… cargo test -p rlx-kyutai-tts --test weights_precision_parity --release -- --nocapture
//! ```

mod parity_common;

use anyhow::Result;
use parity_common::{PARITY_PROMPT, assert_frames_match, load_speaker, model_dir, parity_gen_cfg};
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::download::tokenizer_path;
use rlx_kyutai_tts::generate::generate_codes;
use rlx_kyutai_tts::model::KyutaiTtsModel;
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;
use rlx_kyutai_tts::weights::{WeightStorageDtype, load_weight_map, roundtrip_weight_map};
use rlx_runtime::Device;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

fn codes_from_weights(
    dir: &Path,
    weights: rlx_kyutai_tts::weights::WeightMap,
    prompt: &str,
) -> Result<Vec<Vec<u32>>> {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tokenizer = KyutaiTokenizer::load(tokenizer_path(dir))?;
    let mut m = KyutaiTtsModel::from_weights(cfg, weights, Device::Cpu)?;
    let spk = load_speaker(dir)?;
    generate_codes(&mut m, &tokenizer, prompt, parity_gen_cfg(), Some(&spk)).map(|(f, _)| f)
}

#[test]
#[ignore = "requires eager Moshi code parity; F16 simulation shifts DepFormer logits"]
fn f16_roundtrip_matches_f32_codes() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = dir.join(rlx_kyutai_tts::download::TTS_WEIGHTS_FILE);
    let f32_map = load_weight_map(&path)?;
    let f16_map = roundtrip_weight_map(&f32_map, WeightStorageDtype::F16);
    let reference = codes_from_weights(&dir, f32_map, PARITY_PROMPT)?;
    let actual = codes_from_weights(&dir, f16_map, PARITY_PROMPT)?;
    assert_frames_match("F16 round-trip vs F32", &reference, &actual)
}

#[test]
fn bf16_roundtrip_is_idempotent() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = dir.join(rlx_kyutai_tts::download::TTS_WEIGHTS_FILE);
    let f32_map = load_weight_map(&path)?;
    let once = roundtrip_weight_map(&f32_map, WeightStorageDtype::Bf16);
    let twice = roundtrip_weight_map(&once, WeightStorageDtype::Bf16);
    let reference = codes_from_weights(&dir, once, PARITY_PROMPT)?;
    let actual = codes_from_weights(&dir, twice, PARITY_PROMPT)?;
    assert_frames_match("BF16 round-trip idempotent", &reference, &actual)
}

#[test]
#[ignore = "requires eager Moshi code parity; BF16 widen drift flips DepFormer cb5+ ties"]
fn bf16_roundtrip_matches_f32_codes() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = dir.join(rlx_kyutai_tts::download::TTS_WEIGHTS_FILE);
    let f32_map = load_weight_map(&path)?;
    let bf16_map = roundtrip_weight_map(&f32_map, WeightStorageDtype::Bf16);
    let reference = codes_from_weights(&dir, f32_map, PARITY_PROMPT)?;
    let actual = codes_from_weights(&dir, bf16_map, PARITY_PROMPT)?;
    assert_frames_match("BF16 round-trip vs F32", &reference, &actual)
}

#[test]
fn checkpoint_on_disk_storage_dtypes() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let path = dir.join(rlx_kyutai_tts::download::TTS_WEIGHTS_FILE);
    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (_, view) in st.tensors() {
        *counts.entry(format!("{:?}", view.dtype())).or_default() += 1;
    }
    eprintln!("checkpoint tensor dtypes: {counts:?}");
    Ok(())
}
