// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Safetensors **header** probing — validates tensor shapes without
//! downloading multi‑GB shard payloads.
//!
//! Each `.safetensors` file begins with an 8-byte little-endian header
//! length, then a JSON map of `{ name: { dtype, shape, data_offsets } }`.
//! That header is typically tens of KB even when the shard itself is ~17 GB.

use crate::config::{HF_MODEL_ID, InklingConfig};
use crate::shapes::{expected_hf_shapes, multimodal_presence_keys};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
#[cfg(any(test, feature = "hf-probe"))]
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    #[allow(dead_code)]
    data_offsets: Option<[usize; 2]>,
}

/// Parse the JSON header from the raw leading bytes of a safetensors file.
pub fn parse_safetensors_header(bytes: &[u8]) -> Result<HashMap<String, TensorMeta>> {
    if bytes.len() < 8 {
        bail!("safetensors header too short ({} bytes)", bytes.len());
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + n {
        bail!(
            "safetensors header claims {n} JSON bytes but only {} available",
            bytes.len().saturating_sub(8)
        );
    }
    let json = std::str::from_utf8(&bytes[8..8 + n]).context("header utf8")?;
    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_str(json).context("header json")?;
    let mut out = HashMap::new();
    for (name, val) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: HeaderEntry =
            serde_json::from_value(val).with_context(|| format!("entry {name}"))?;
        out.insert(
            name,
            TensorMeta {
                dtype: entry.dtype,
                shape: entry.shape,
            },
        );
    }
    Ok(out)
}

/// Read only the safetensors header from a local file (no full slurp).
pub fn read_local_header(path: impl AsRef<Path>) -> Result<HashMap<String, TensorMeta>> {
    let path = path.as_ref();
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).context("read header len")?;
    let n = u64::from_le_bytes(len_buf) as usize;
    let mut json = vec![0u8; n];
    f.read_exact(&mut json).context("read header json")?;
    let mut bytes = Vec::with_capacity(8 + n);
    bytes.extend_from_slice(&len_buf);
    bytes.extend_from_slice(&json);
    parse_safetensors_header(&bytes)
}

/// Compare `observed` against `expected` for keys present in both.
/// Returns `(ok_count, mismatches)`.
pub fn compare_shapes(
    expected: &HashMap<String, Vec<usize>>,
    observed: &HashMap<String, TensorMeta>,
) -> (usize, Vec<String>) {
    let mut ok = 0usize;
    let mut bad = Vec::new();
    for (name, want) in expected {
        let Some(got) = observed.get(name) else {
            continue;
        };
        if &got.shape == want {
            ok += 1;
        } else {
            bad.push(format!(
                "{name}: expected {want:?}, got {:?} ({})",
                got.shape, got.dtype
            ));
        }
    }
    (ok, bad)
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

pub fn load_weight_index(path: impl AsRef<Path>) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("read {}", path.as_ref().display()))?;
    let idx: WeightIndex = serde_json::from_str(&text).context("parse weight index")?;
    Ok(idx.weight_map)
}

/// Pick a small set of shards that cover stem + one dense + one MoE layer.
pub fn select_probe_shards(
    weight_map: &HashMap<String, String>,
    cfg: &InklingConfig,
) -> Vec<String> {
    let mut keys = vec![
        "model.llm.embed.weight".to_string(),
        "model.llm.layers.0.attn.wq_du.weight".to_string(),
        "model.llm.layers.0.attn.k_sconv.weight".to_string(),
        "model.llm.unembed.weight".to_string(),
    ];
    for k in multimodal_presence_keys() {
        keys.push((*k).to_string());
    }
    // First MoE layer (after dense_mlp_idx).
    let moe_layer = cfg
        .text
        .dense_mlp_idx
        .min(cfg.text.num_hidden_layers.saturating_sub(1));
    keys.push(format!("model.llm.layers.{moe_layer}.mlp.gate.weight"));
    keys.push(format!(
        "model.llm.layers.{moe_layer}.mlp.experts.w13_weight"
    ));

    let mut shards = Vec::new();
    for k in keys {
        if let Some(shard) = weight_map.get(&k) {
            if !shards.iter().any(|s| s == shard) {
                shards.push(shard.clone());
            }
        }
    }
    shards
}

fn check_multimodal_presence(observed: &HashMap<String, TensorMeta>) -> (usize, Vec<String>) {
    let mut seen = 0usize;
    let mut missing = Vec::new();
    for k in multimodal_presence_keys() {
        if observed.contains_key(*k) {
            seen += 1;
        } else {
            missing.push((*k).to_string());
        }
    }
    (seen, missing)
}

/// Validate local model dir: config + index + header-only probes of selected shards.
pub fn validate_model_dir(dir: impl AsRef<Path>) -> Result<ProbeReport> {
    let dir = dir.as_ref();
    let cfg = InklingConfig::from_model_dir(dir)?;
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        bail!("missing {}", index_path.display());
    }
    let weight_map = load_weight_index(&index_path)?;
    let expected = expected_hf_shapes(&cfg);
    let shards = select_probe_shards(&weight_map, &cfg);
    let mut observed = HashMap::new();
    let mut probed = Vec::new();
    for shard in &shards {
        let path = dir.join(shard);
        if !path.is_file() {
            // Shard not downloaded — skip (caller may only have config+index).
            continue;
        }
        let meta =
            read_local_header(&path).with_context(|| format!("header {}", path.display()))?;
        probed.push(shard.clone());
        observed.extend(meta);
    }
    if probed.is_empty() {
        bail!(
            "no probe shards present under {} (need at least one of {shards:?})",
            dir.display()
        );
    }
    let (ok, bad) = compare_shapes(&expected, &observed);
    let (mm_seen, mm_missing) = check_multimodal_presence(&observed);
    Ok(ProbeReport {
        model_id: HF_MODEL_ID.into(),
        probed_shards: probed,
        matched: ok,
        mismatches: bad,
        observed_tensors: observed.len(),
        expected_keys: expected.len(),
        multimodal_seen: mm_seen,
        multimodal_missing: mm_missing,
    })
}

#[derive(Debug)]
pub struct ProbeReport {
    pub model_id: String,
    pub probed_shards: Vec<String>,
    pub matched: usize,
    pub mismatches: Vec<String>,
    pub observed_tensors: usize,
    pub expected_keys: usize,
    pub multimodal_seen: usize,
    pub multimodal_missing: Vec<String>,
}

impl ProbeReport {
    pub fn assert_ok(&self) -> Result<()> {
        if !self.mismatches.is_empty() {
            bail!(
                "shape mismatches ({}):\n  {}",
                self.mismatches.len(),
                self.mismatches.join("\n  ")
            );
        }
        if self.matched == 0 {
            bail!("no expected keys found in probed shard headers");
        }
        Ok(())
    }

    pub fn print(&self) {
        println!("Inkling probe — {}", self.model_id);
        println!("  shards: {}", self.probed_shards.join(", "));
        println!(
            "  header tensors seen={}  text expected_keys={}  text shape matches={}",
            self.observed_tensors, self.expected_keys, self.matched
        );
        println!(
            "  multimodal presence: {}/{}{}",
            self.multimodal_seen,
            self.multimodal_seen + self.multimodal_missing.len(),
            if self.multimodal_missing.is_empty() {
                String::new()
            } else {
                format!(" (missing {})", self.multimodal_missing.join(", "))
            }
        );
        if self.mismatches.is_empty() {
            println!("  result: OK");
        } else {
            println!("  result: FAIL ({} mismatches)", self.mismatches.len());
            for m in &self.mismatches {
                println!("    {m}");
            }
        }
    }
}

/// Fetch config + index only (few hundred KB), then HTTP Range-probe shard headers.
#[cfg(feature = "hf-probe")]
pub fn probe_remote(cache_dir: Option<PathBuf>) -> Result<ProbeReport> {
    use std::io::Write;

    let cache = cache_dir.unwrap_or_else(default_cache_dir);
    std::fs::create_dir_all(&cache).ok();

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache.clone())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(HF_MODEL_ID.to_string());

    let config_path = repo.get("config.json").context("download config.json")?;
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("download model.safetensors.index.json")?;
    let cfg = InklingConfig::from_json_path(&config_path)?;
    let weight_map = load_weight_index(&index_path)?;
    let expected = expected_hf_shapes(&cfg);
    let shards = select_probe_shards(&weight_map, &cfg);

    let mut observed = HashMap::new();
    let mut probed = Vec::new();
    for shard in &shards {
        // Prefer a previously cached full shard (header read is cheap).
        let local = config_path
            .parent()
            .map(|p| p.join(shard))
            .filter(|p| p.is_file());
        let meta = if let Some(path) = local {
            read_local_header(&path)?
        } else {
            let url = format!("https://huggingface.co/{HF_MODEL_ID}/resolve/main/{shard}");
            let bytes = http_range_header(&url).with_context(|| format!("range-get {shard}"))?;
            // Cache the tiny header for offline re-runs.
            let hdr_cache = cache.join("inkling-headers").join(format!("{shard}.hdr"));
            if let Some(parent) = hdr_cache.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = File::create(&hdr_cache) {
                let _ = f.write_all(&bytes);
            }
            parse_safetensors_header(&bytes)?
        };
        probed.push(shard.clone());
        observed.extend(meta);
    }

    let (ok, bad) = compare_shapes(&expected, &observed);
    let (mm_seen, mm_missing) = check_multimodal_presence(&observed);
    Ok(ProbeReport {
        model_id: HF_MODEL_ID.into(),
        probed_shards: probed,
        matched: ok,
        mismatches: bad,
        observed_tensors: observed.len(),
        expected_keys: expected.len(),
        multimodal_seen: mm_seen,
        multimodal_missing: mm_missing,
    })
}

#[cfg(feature = "hf-probe")]
fn default_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".cache").join("huggingface")
        })
}

/// Download only the safetensors JSON header via HTTP Range.
#[cfg(feature = "hf-probe")]
fn http_range_header(url: &str) -> Result<Vec<u8>> {
    // First 8 bytes → header length.
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(60))
        .build();
    let len_resp = agent
        .get(url)
        .set("Range", "bytes=0-7")
        .call()
        .with_context(|| format!("HEAD/range len {url}"))?;
    let mut len_buf = Vec::new();
    len_resp
        .into_reader()
        .take(8)
        .read_to_end(&mut len_buf)
        .context("read len bytes")?;
    if len_buf.len() < 8 {
        bail!("short length prefix from {url}");
    }
    let n = u64::from_le_bytes(len_buf[0..8].try_into().unwrap()) as usize;
    if n > 64 * 1024 * 1024 {
        bail!("implausible safetensors header size {n}");
    }
    let end = 8 + n - 1;
    let body_resp = agent
        .get(url)
        .set("Range", &format!("bytes=0-{end}"))
        .call()
        .with_context(|| format!("range header {url}"))?;
    let mut bytes = Vec::with_capacity(8 + n);
    body_resp
        .into_reader()
        .take((8 + n) as u64)
        .read_to_end(&mut bytes)
        .context("read header body")?;
    if bytes.len() < 8 + n {
        bail!(
            "incomplete header download: got {} want {}",
            bytes.len(),
            8 + n
        );
    }
    Ok(bytes)
}

/// Write a synthetic safetensors **header-only** fixture (empty data offsets).
#[cfg(test)]
pub fn write_header_fixture(path: &Path, tensors: &HashMap<String, Vec<usize>>) -> Result<()> {
    use std::io::Write;
    let mut map = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape) in tensors {
        let n: usize = shape.iter().product();
        let nbytes = n * 4; // F32
        let entry = serde_json::json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        });
        map.insert(name.clone(), entry);
        offset += nbytes;
    }
    let json = serde_json::Value::Object(map).to_string();
    let json_bytes = json.as_bytes();
    let mut f = File::create(path)?;
    f.write_all(&(json_bytes.len() as u64).to_le_bytes())?;
    f.write_all(json_bytes)?;
    // No tensor payload — header probes never seek past the JSON.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::expected_hf_shapes;
    use crate::synth::tiny_mm_cfg;

    #[test]
    fn header_roundtrip_and_shape_match() {
        let cfg = tiny_mm_cfg();
        let expected = expected_hf_shapes(&cfg);
        let dir = tempfile_dir();
        let shard = dir.join("model-00001-of-00001.safetensors");
        // Subset of expected keys as if they lived on one shard.
        let mut subset = HashMap::new();
        for k in [
            "model.llm.embed.weight",
            "model.llm.layers.0.attn.wq_du.weight",
            "model.llm.layers.1.mlp.gate.weight",
        ] {
            subset.insert(k.to_string(), expected[k].clone());
        }
        write_header_fixture(&shard, &subset).unwrap();
        let observed = read_local_header(&shard).unwrap();
        let (ok, bad) = compare_shapes(&expected, &observed);
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(ok, 3);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rlx-inkling-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
