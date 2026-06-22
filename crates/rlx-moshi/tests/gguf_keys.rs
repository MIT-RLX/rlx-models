//! Q8 GGUF tensor key coverage (env `RLX_MOSHI_Q8_GGUF`).

use rlx_moshi::{
    MoshiCheckpoint, MoshiVariant, expected_lm_keys, gguf_tensor_count, load_gguf_weight_map,
};
use std::path::PathBuf;

#[test]
fn gguf_tensor_count_matches_eager_keys() {
    let path = match std::env::var("RLX_MOSHI_Q8_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skip gguf_tensor_count_matches_eager_keys (set RLX_MOSHI_Q8_GGUF)");
            return;
        }
    };
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let cfg = MoshiVariant::MoshikoOneWay.lm_config();
    let keys = expected_lm_keys(&cfg);
    let n = gguf_tensor_count(&path).expect("gguf open");
    assert!(
        n >= keys.len(),
        "gguf has {n} tensors, expected at least {} LM keys",
        keys.len()
    );
}

#[test]
fn gguf_load_selected_keys() {
    let path = match std::env::var("RLX_MOSHI_Q8_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skip gguf_load_selected_keys (set RLX_MOSHI_Q8_GGUF)");
            return;
        }
    };
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let cfg = MoshiVariant::MoshikoOneWay.lm_config();
    let map = load_gguf_weight_map(&path, &cfg).expect("gguf load");
    assert_eq!(map.len(), expected_lm_keys(&cfg).len());
    let emb = map.get("text_emb.weight").expect("text_emb");
    assert_eq!(emb.1.len(), 2);
    assert!(emb.1[0] > 0 && emb.1[1] > 0);
    let _ = MoshiCheckpoint::Q8Gguf;
}
