//! DepFormer multi-codebook check on real weights.

use ndarray::Array1;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::depformer_stream::DepformerStream;
use rlx_kyutai_tts::download::{TTS_WEIGHTS_FILE, default_kyutai_tts_dir};
use rlx_kyutai_tts::sampling::{LogitsProcessor, argmax};
use rlx_kyutai_tts::weights::load_weight_map;
use std::collections::HashSet;

#[test]
fn depformer_produces_diverse_codebooks() {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    let wpath = dir.join(TTS_WEIGHTS_FILE);
    if !wpath.is_file() {
        eprintln!("skip: missing weights");
        return;
    }
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let weights = load_weight_map(&wpath).expect("load");
    let mut df = DepformerStream::load(&cfg, &weights).expect("depformer");
    let hidden = Array1::from_elem(2048, 0.01f32);
    let mut lp = LogitsProcessor::new(0.0, 1, 42);
    df.reset();
    let mut prev = 100u32;
    let mut toks = Vec::new();
    for cb in 0..8 {
        let logits = df.forward_codebook(cb, &hidden, prev).expect("forward");
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let tok = lp.sample(&logits);
        eprintln!(
            "cb{cb}: argmax={} sampled={tok} max_logit={mx:.3}",
            argmax(&logits)
        );
        toks.push(tok);
        prev = tok;
    }
    let unique: HashSet<_> = toks.iter().copied().collect();
    assert!(unique.len() >= 4, "depformer collapsed to {toks:?}");
}
