//! Moshi Q8 GGUF loading (Kyutai tensor names match safetensors keys).

use crate::config::LmConfig;
use crate::weights::expected_lm_keys;
use anyhow::{Context, Result};
use rlx_core::gguf_resolve::{PrefixStripGgufResolver, register_gguf_tensor_resolver};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

static GGUF_RESOLVER: OnceLock<()> = OnceLock::new();

fn ensure_moshi_gguf_resolver() {
    GGUF_RESOLVER.get_or_init(|| {
        register_gguf_tensor_resolver(Box::new(PrefixStripGgufResolver));
    });
}

/// Load Moshi LM tensors from a Kyutai `model.q8.gguf` into f32 maps (eager CPU).
pub fn load_gguf_weight_map(
    path: &Path,
    cfg: &LmConfig,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    ensure_moshi_gguf_resolver();
    let keys: HashSet<String> = expected_lm_keys(cfg).into_iter().collect();
    let path = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("gguf path not utf-8: {}", path.display()))?;
    let mut loader = GgufLoader::from_file(path)?;
    let mut map = HashMap::with_capacity(keys.len());
    for key in keys {
        let (data, shape) = loader
            .take(&key)
            .with_context(|| format!("missing GGUF tensor {key}"))?;
        map.insert(key, (data, shape));
    }
    Ok(map)
}

pub fn gguf_tensor_count(path: &Path) -> Result<usize> {
    ensure_moshi_gguf_resolver();
    let file = rlx_core::load_gguf_file(path)?;
    Ok(file.tensors.len())
}
