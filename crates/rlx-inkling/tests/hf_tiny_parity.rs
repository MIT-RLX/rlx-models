// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Parity: Rust eager text forward vs transformers InklingForCausalLM dump
// (`scripts/dump_hf_tiny_parity.py` → `tests/fixtures/hf_tiny_parity`).

use rlx_inkling::config::InklingConfig;
use rlx_inkling::eager::forward_logits;
use rlx_inkling::fixture::{fixture_dir, load_logits, load_meta, load_text_weights, max_abs_diff};

#[test]
fn eager_matches_hf_tiny_fixture() {
    let dir = fixture_dir();
    assert!(
        dir.join("logits.bin").is_file(),
        "missing fixture at {} — run scripts/dump_hf_tiny_parity.py",
        dir.display()
    );
    let cfg = InklingConfig::from_json_path(dir.join("config.json"))
        .expect("fixture config")
        .text;
    let weights = load_text_weights(&dir).expect("weights");
    let meta = load_meta(&dir).expect("meta");
    let want = load_logits(&dir).expect("logits");
    let got = forward_logits(&cfg, &weights, &meta.input_ids).expect("forward");
    assert_eq!(got.len(), want.len());
    let mad = max_abs_diff(&got, &want);
    // Transformers eager f32 vs our f32 reference — should be near machine eps
    // for this tiny graph (observed ≪ 1e-5 when regenerating the fixture).
    assert!(
        mad < 1e-5,
        "max abs diff {mad} (got[:8]={:?} want[:8]={:?})",
        &got[..8.min(got.len())],
        &want[..8.min(want.len())]
    );
}
