use rlx_moshi::{MoshiVariant, expected_lm_keys, load_lm_weights, open_lm, resolve_moshi_dir};
use std::path::PathBuf;

fn moshi_dir() -> Option<PathBuf> {
    std::env::var("RLX_MOSHI_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            let d = PathBuf::from(".cache/moshiko");
            d.join("model.safetensors").is_file().then_some(d)
        })
}

#[test]
fn expected_keys_present_in_checkpoint() -> anyhow::Result<()> {
    let Some(dir) = moshi_dir() else {
        eprintln!("skip: run `just fetch-moshi`");
        return Ok(());
    };
    let cfg = MoshiVariant::Moshiko.lm_config();
    let want = expected_lm_keys(&cfg);
    let map = load_lm_weights(&dir, &cfg)?;
    let mut missing = Vec::new();
    for k in &want {
        if !map.contains_key(k) {
            missing.push(k.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "missing {} / {} keys, first few: {:?}",
        missing.len(),
        want.len(),
        &missing[..missing.len().min(8)]
    );
    eprintln!("checked {} tensor keys under {}", want.len(), dir.display());
    Ok(())
}

#[test]
fn can_open_lm() -> anyhow::Result<()> {
    let dir = resolve_moshi_dir(moshi_dir().as_deref());
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skip: run `just fetch-moshi`");
        return Ok(());
    }
    let cfg = MoshiVariant::MoshikoOneWay.lm_config();
    let lm = open_lm(&dir, cfg)?;
    assert_eq!(lm.config().transformer.num_layers, 32);
    Ok(())
}
