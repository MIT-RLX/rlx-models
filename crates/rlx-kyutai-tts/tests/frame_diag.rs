//! Codebook diversity in gathered Mimi frames (env-gated).

use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::download::{default_kyutai_tts_dir, tokenizer_path};
use rlx_kyutai_tts::generate::{GenerateConfig, generate_codes};
use rlx_kyutai_tts::model::KyutaiTtsModel;
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;
use rlx_runtime::Device;
use std::collections::HashSet;
use std::path::PathBuf;

#[test]
fn collected_frames_use_multiple_codebooks() {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if !dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        eprintln!("skip: missing weights");
        return;
    }
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tokenizer = KyutaiTokenizer::load(tokenizer_path(&dir)).expect("tok");
    let mut m = KyutaiTtsModel::open(&dir, cfg, Device::Cpu).expect("model");
    let gen_cfg = GenerateConfig {
        max_steps: 40,
        n_q: 8,
        cfg_alpha: 2.0,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        seed: 7,
    };
    let (frames, end, _) =
        generate_codes(&mut m, &tokenizer, "Hello world.", gen_cfg, None).expect("gen");
    eprintln!("frames: {} end_step: {end:?}", frames.len());
    let uniq: std::collections::HashSet<_> = frames
        .iter()
        .map(|f| format!("{:?}", &f[..4.min(f.len())]))
        .collect();
    eprintln!("unique frame prefixes: {}", uniq.len());
    for (i, fr) in frames.iter().enumerate().take(5) {
        let nz: Vec<_> = fr.iter().enumerate().filter(|(_, t)| **t != 0).collect();
        eprintln!("  [{i}] non-zero cbs: {:?}", nz);
    }
    let mid = frames.get(frames.len() / 2).expect("mid");
    let active: HashSet<_> = mid
        .iter()
        .enumerate()
        .filter(|(_, t)| **t != 0)
        .map(|(i, _)| i)
        .collect();
    assert!(
        active.len() >= 4,
        "mid frame only uses codebooks {active:?}: {mid:?}"
    );
}
