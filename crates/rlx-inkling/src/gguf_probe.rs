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

//! Unsloth Inkling GGUF **header** sniff — no weight payload download.
//!
//! Hub layout ([unsloth/inkling-GGUF](https://huggingface.co/unsloth/inkling-GGUF)):
//! - `UD-IQ1_S/…-00001-of-00007.gguf` — metadata + tokenizer (`n_tensors=0`, ~13 MB)
//! - later shards — tensor table + IQ packs (tens of GB each)
//!
//! Weight-shard headers are tens of KB; we HTTP Range-read only the prefix.
//! Shard 00001 is downloaded whole only because the tokenizer KV blobs inflate
//! the metadata section to ~13 MB (still not the ~270 GB weight payload).

use crate::config::{HF_GGUF_REPO, InklingTextConfig};
use crate::gguf_layout::{LAYOUT_NOTES, gguf_to_eager_key};
use anyhow::{Context, Result};
use rlx_gguf::{GgmlType, GgufFile, MetaValue};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

#[cfg(feature = "hf-probe")]
use anyhow::bail;
#[cfg(feature = "hf-probe")]
use std::io::Read;
#[cfg(any(test, feature = "hf-probe"))]
use std::path::PathBuf;

/// Default Unsloth quant folder under [`HF_GGUF_REPO`].
pub const DEFAULT_QUANT: &str = "UD-IQ1_S";

#[derive(Debug, Clone, Serialize)]
pub struct GgufTensorSniff {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
    pub shard: String,
    /// Eager key when 1:1; `None` for expert packs / unknown.
    pub eager_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufShardSniff {
    pub path: String,
    pub file_bytes: Option<u64>,
    pub prefix_bytes: usize,
    pub n_metadata: usize,
    pub n_tensors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufProbeReport {
    pub repo: String,
    pub quant: String,
    pub architecture: String,
    pub split_count: Option<u32>,
    pub split_tensors_count: Option<i32>,
    pub text: GgufTextSummary,
    pub shards: Vec<GgufShardSniff>,
    pub tensors: Vec<GgufTensorSniff>,
    pub dtype_histogram: BTreeMap<String, usize>,
    pub role_histogram: BTreeMap<String, usize>,
    pub unmapped_names: Vec<String>,
    pub layout_notes: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufTextSummary {
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub swa_kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub sliding_window: usize,
    pub dense_mlp_idx: usize,
    pub n_routed: usize,
    pub n_shared: usize,
    pub top_k: usize,
    pub dense_ff: usize,
    pub moe_ff: usize,
    pub sconv_k: usize,
}

impl From<&InklingTextConfig> for GgufTextSummary {
    fn from(c: &InklingTextConfig) -> Self {
        Self {
            layers: c.num_hidden_layers,
            hidden: c.hidden_size,
            vocab: c.vocab_size,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            swa_kv_heads: c.swa_num_key_value_heads,
            head_dim: c.head_dim,
            d_rel: c.d_rel,
            sliding_window: c.sliding_window_size,
            dense_mlp_idx: c.dense_mlp_idx,
            n_routed: c.n_routed_experts,
            n_shared: c.n_shared_experts,
            top_k: c.num_experts_per_tok,
            dense_ff: c.dense_intermediate_size,
            moe_ff: c.moe_intermediate_size,
            sconv_k: c.conv_kernel_size,
        }
    }
}

fn dtype_name(t: GgmlType) -> String {
    format!("{t:?}")
}

fn tensor_role(name: &str) -> &'static str {
    if name.starts_with("token_embd") {
        return "embed";
    }
    if name.starts_with("output") {
        return "output";
    }
    if name.contains("ffn_gate_exps")
        || name.contains("ffn_up_exps")
        || name.contains("ffn_down_exps")
    {
        return "moe_exps";
    }
    if name.contains("ffn_gate_shexp")
        || name.contains("ffn_up_shexp")
        || name.contains("ffn_down_shexp")
    {
        return "moe_shared";
    }
    if name.contains("ffn_gate_inp") || name.contains("exp_probs_b") {
        return "moe_router";
    }
    if name.contains("ffn_gate") || name.contains("ffn_up") || name.contains("ffn_down") {
        return "dense_ffn";
    }
    if name.contains("shortconv") {
        return "shortconv";
    }
    if name.contains("attn_rel_proj") {
        return "rel_bias";
    }
    if name.contains("attn_") {
        return "attn";
    }
    if name.contains("norm") || name.contains("gscale") {
        return "norm";
    }
    "other"
}

/// Parse a local GGUF header prefix (or full small meta shard).
pub fn sniff_local_prefix(
    path: impl AsRef<Path>,
    shard_label: &str,
) -> Result<(GgufFile, GgufShardSniff)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let file = GgufFile::header_from_bytes(&bytes)
        .with_context(|| format!("parse GGUF header {}", path.display()))?;
    let sniff = GgufShardSniff {
        path: shard_label.to_string(),
        file_bytes: std::fs::metadata(path).ok().map(|m| m.len()),
        prefix_bytes: bytes.len(),
        n_metadata: file.metadata.len(),
        n_tensors: file.tensors.len(),
    };
    Ok((file, sniff))
}

fn meta_u32(file: &GgufFile, key: &str) -> Option<u32> {
    file.metadata.get(key).and_then(MetaValue::as_u32)
}

fn meta_i32(file: &GgufFile, key: &str) -> Option<i32> {
    file.metadata.get(key).and_then(|v| match v {
        MetaValue::I32(x) => Some(*x),
        MetaValue::U32(x) => Some(*x as i32),
        MetaValue::I16(x) => Some(*x as i32),
        MetaValue::U16(x) => Some(*x as i32),
        _ => None,
    })
}

fn build_report(
    repo: &str,
    quant: &str,
    shards: Vec<(GgufFile, GgufShardSniff)>,
) -> Result<GgufProbeReport> {
    let mut architecture = String::new();
    let mut split_count = None;
    let mut split_tensors_count = None;
    let mut text_cfg: Option<InklingTextConfig> = None;
    let mut tensors = Vec::new();
    let mut dtype_histogram = BTreeMap::new();
    let mut role_histogram = BTreeMap::new();
    let mut unmapped = BTreeSet::new();
    let mut shard_sniffs = Vec::new();

    for (file, sniff) in shards {
        if architecture.is_empty() {
            if let Some(a) = file
                .metadata
                .get("general.architecture")
                .and_then(MetaValue::as_str)
            {
                architecture = a.to_string();
            }
        }
        if split_count.is_none() {
            split_count = meta_u32(&file, "split.count").or_else(|| {
                file.metadata.get("split.count").and_then(|v| match v {
                    MetaValue::U16(x) => Some(*x as u32),
                    _ => None,
                })
            });
        }
        if split_tensors_count.is_none() {
            split_tensors_count = meta_i32(&file, "split.tensors.count");
        }
        if text_cfg.is_none() && file.metadata.contains_key("inkling.block_count") {
            text_cfg = Some(InklingTextConfig::from_gguf(&file)?);
        }
        for (name, t) in &file.tensors {
            let dtype = dtype_name(t.dtype);
            *dtype_histogram.entry(dtype.clone()).or_insert(0) += 1;
            let role = tensor_role(name);
            *role_histogram.entry(role.to_string()).or_insert(0) += 1;
            let eager_key = gguf_to_eager_key(name);
            // Expert packs intentionally have no 1:1 eager key.
            if eager_key.is_none() && role == "other" {
                unmapped.insert(name.clone());
            }
            tensors.push(GgufTensorSniff {
                name: name.clone(),
                shape: t.shape.clone(),
                dtype,
                offset: t.offset,
                shard: sniff.path.clone(),
                eager_key,
            });
        }
        shard_sniffs.push(sniff);
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let text = text_cfg
        .as_ref()
        .map(GgufTextSummary::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no shard carried inkling.* metadata (expected …-00001-of-… meta shard)"
            )
        })?;

    Ok(GgufProbeReport {
        repo: repo.into(),
        quant: quant.into(),
        architecture: if architecture.is_empty() {
            "inkling".into()
        } else {
            architecture
        },
        split_count,
        split_tensors_count,
        text,
        shards: shard_sniffs,
        tensors,
        dtype_histogram,
        role_histogram,
        unmapped_names: unmapped.into_iter().collect(),
        layout_notes: LAYOUT_NOTES.into(),
    })
}

impl GgufProbeReport {
    pub fn print(&self) {
        println!(
            "Inkling GGUF sniff  {repo}/{quant}",
            repo = self.repo,
            quant = self.quant
        );
        println!(
            "  arch={}  split={:?}/{:?} tensors_in_split={:?}",
            self.architecture,
            self.split_count,
            self.shards.len(),
            self.split_tensors_count
        );
        let t = &self.text;
        println!(
            "  text: L={} H={} V={} heads={}/{} swa_kv={} hd={} d_rel={} win={}",
            t.layers,
            t.hidden,
            t.vocab,
            t.heads,
            t.kv_heads,
            t.swa_kv_heads,
            t.head_dim,
            t.d_rel,
            t.sliding_window
        );
        println!(
            "  moe: routed={} shared={} top_k={} dense_layers={} dense_ff={} moe_ff={} sconv={}",
            t.n_routed, t.n_shared, t.top_k, t.dense_mlp_idx, t.dense_ff, t.moe_ff, t.sconv_k
        );
        for s in &self.shards {
            println!(
                "  shard {}: prefix={}B file={:?} meta={} tensors={}",
                s.path, s.prefix_bytes, s.file_bytes, s.n_metadata, s.n_tensors
            );
        }
        println!("  tensors listed={}", self.tensors.len());
        println!("  dtypes: {:?}", self.dtype_histogram);
        println!("  roles:  {:?}", self.role_histogram);
        if !self.unmapped_names.is_empty() {
            println!(
                "  unmapped (need loader rules): {} — e.g. {:?}",
                self.unmapped_names.len(),
                &self.unmapped_names[..self.unmapped_names.len().min(8)]
            );
        }
        // Representative shapes for the loader.
        let samples = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_rel_proj.weight",
            "blk.0.shortconv_k.weight",
            "blk.0.ffn_gate.weight",
            "blk.2.ffn_gate_inp.weight",
            "blk.2.ffn_gate_exps.weight",
            "blk.2.ffn_up_exps.weight",
            "blk.2.ffn_down_exps.weight",
            "blk.2.ffn_gate_shexp.weight",
            "blk.5.attn_k.weight",
            "blk.5.attn_rel_proj.weight",
        ];
        println!("  shape samples:");
        let by_name: HashMap<&str, &GgufTensorSniff> =
            self.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        for name in samples {
            if let Some(t) = by_name.get(name) {
                println!(
                    "    {} {:?} {} eager={:?}",
                    t.name, t.shape, t.dtype, t.eager_key
                );
            }
        }
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let s = serde_json::to_string_pretty(self).context("serialize sniff")?;
        std::fs::write(path, s).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Compact fixture used by unit tests / docs (not the full tensor dump).
    pub fn write_compact_fixture(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut tensor_name_samples: Vec<String> = Vec::new();
        for want in [
            "token_embd.weight",
            "token_embd_norm.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_r.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_rel_proj.weight",
            "blk.0.shortconv_k.weight",
            "blk.0.shortconv_v.weight",
            "blk.0.shortconv_attn.weight",
            "blk.0.shortconv_mlp.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.2.ffn_gate_inp.weight",
            "blk.2.ffn_gate_exps.weight",
            "blk.2.ffn_up_exps.weight",
            "blk.2.ffn_down_exps.weight",
            "blk.2.ffn_gate_shexp.weight",
            "blk.2.ffn_up_shexp.weight",
            "blk.2.ffn_down_shexp.weight",
            "blk.2.exp_probs_b.bias",
        ] {
            if self.tensors.iter().any(|t| t.name == want) {
                tensor_name_samples.push(want.into());
            }
        }
        let by_name: HashMap<&str, &GgufTensorSniff> =
            self.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let shape_samples = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_rel_proj.weight",
            "blk.0.shortconv_k.weight",
            "blk.0.ffn_down.weight",
            "blk.2.ffn_gate_inp.weight",
            "blk.2.ffn_gate_exps.weight",
            "blk.2.ffn_down_exps.weight",
            "blk.2.ffn_gate_shexp.weight",
            "blk.5.attn_k.weight",
            "blk.5.attn_rel_proj.weight",
        ]
        .into_iter()
        .filter_map(|n| {
            by_name.get(n).map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "shape": t.shape,
                    "dtype": t.dtype,
                })
            })
        })
        .collect::<Vec<_>>();

        let compact = serde_json::json!({
            "general.architecture": self.architecture,
            "repo": self.repo,
            "quant": self.quant,
            "split.count": self.split_count,
            "split.tensors.count": self.split_tensors_count,
            "inkling.block_count": self.text.layers,
            "inkling.context_length": 1_048_576,
            "inkling.embedding_length": self.text.hidden,
            "inkling.feed_forward_length": self.text.dense_ff,
            "inkling.expert_feed_forward_length": self.text.moe_ff,
            "inkling.expert_count": self.text.n_routed,
            "inkling.expert_used_count": self.text.top_k,
            "inkling.expert_shared_count": self.text.n_shared,
            "inkling.dense_block_count": self.text.dense_mlp_idx,
            "inkling.attention.head_count": self.text.heads,
            "inkling.d_rel": self.text.d_rel,
            "inkling.attention.sliding_window": self.text.sliding_window,
            "inkling.shortconv_kernel": self.text.sconv_k,
            "inkling.vocab_size": self.text.vocab,
            "dtype_histogram": self.dtype_histogram,
            "role_histogram": self.role_histogram,
            "tensor_name_samples": tensor_name_samples,
            "shape_samples": shape_samples,
            "shards": self.shards,
            "layout_notes": {
                "source": format!("{}/{}", self.repo, self.quant),
                "global_kv_heads": self.text.kv_heads,
                "swa_kv_heads": self.text.swa_kv_heads,
                "shortconv_shape": "[kernel, channels]",
                "attn_rel_proj_shape": "[rel_extent, d_rel] (SWA window×d_rel, global rel_extent×d_rel)",
                "notes": self.layout_notes,
            }
        });
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, serde_json::to_string_pretty(&compact)?)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

/// Build a report from already-fetched local prefixes (tests / offline).
pub fn report_from_local_files(
    meta_path: impl AsRef<Path>,
    weight_path: impl AsRef<Path>,
    meta_label: &str,
    weight_label: &str,
) -> Result<GgufProbeReport> {
    let meta = sniff_local_prefix(meta_path, meta_label)?;
    let weight = sniff_local_prefix(weight_path, weight_label)?;
    build_report(HF_GGUF_REPO, DEFAULT_QUANT, vec![meta, weight])
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

#[cfg(feature = "hf-probe")]
fn hub_resolve_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{path}")
}

#[cfg(feature = "hf-probe")]
fn http_range(url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();
    let resp = agent
        .get(url)
        .set("Range", &format!("bytes={start}-{end_inclusive}"))
        .call()
        .with_context(|| format!("Range GET {url} bytes={start}-{end_inclusive}"))?;
    let status = resp.status();
    if status != 206 && status != 200 {
        bail!("unexpected HTTP {status} for Range GET {url}");
    }
    let mut bytes = Vec::new();
    resp.into_reader()
        .take(end_inclusive.saturating_sub(start).saturating_add(1))
        .read_to_end(&mut bytes)
        .context("read range body")?;
    if bytes.is_empty() {
        bail!("empty range body from {url}");
    }
    Ok(bytes)
}

/// Grow an HTTP Range window until [`GgufFile::header_from_bytes`] succeeds.
#[cfg(feature = "hf-probe")]
fn fetch_gguf_header_prefix(url: &str) -> Result<Vec<u8>> {
    const START: u64 = 256 * 1024;
    const MAX: u64 = 16 * 1024 * 1024; // meta shard KV+tokenizer is ~13MB
    let mut want = START;
    let mut last_err = anyhow::anyhow!("GGUF header fetch failed");
    loop {
        let bytes = http_range(url, 0, want - 1)?;
        match GgufFile::header_from_bytes(&bytes) {
            Ok(_) => return Ok(bytes),
            Err(e) => {
                let short = (bytes.len() as u64) < want;
                last_err = e;
                if short || want >= MAX {
                    break;
                }
                want = (want.saturating_mul(2)).min(MAX);
            }
        }
    }
    Err(last_err).with_context(|| format!("could not parse GGUF header from {url}"))
}

/// List `quant/` files via Hub API (tiny JSON).
#[cfg(feature = "hf-probe")]
fn list_quant_files(repo: &str, quant: &str) -> Result<Vec<(String, u64)>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main/{quant}");
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(60))
        .build();
    let resp = agent
        .get(&url)
        .call()
        .with_context(|| format!("list {url}"))?;
    let text = resp.into_string().context("list body")?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).context("parse tree json")?;
    let mut out = Vec::new();
    for e in entries {
        if e.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let path = e
            .get("path")
            .and_then(|p| p.as_str())
            .unwrap_or_default()
            .to_string();
        if !path.ends_with(".gguf") {
            continue;
        }
        let size = e.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        out.push((path, size));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    if out.is_empty() {
        bail!("no .gguf under {repo}/{quant}");
    }
    Ok(out)
}

/// Hub header sniff: meta shard + first weight shard (default `UD-IQ1_S`).
///
/// Downloads **only** GGUF prefixes (metadata / tensor tables). Never pulls
/// the multi‑GB IQ weight payloads.
#[cfg(feature = "hf-probe")]
pub fn probe_remote_gguf(
    repo: Option<&str>,
    quant: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<GgufProbeReport> {
    use std::io::Write;

    let repo = repo.unwrap_or(HF_GGUF_REPO);
    let quant = quant.unwrap_or(DEFAULT_QUANT);
    let cache = cache_dir.unwrap_or_else(default_cache_dir);
    let hdr_cache = cache.join("inkling-gguf-headers").join(quant);
    std::fs::create_dir_all(&hdr_cache).ok();

    let files = list_quant_files(repo, quant)?;
    let meta = files
        .iter()
        .find(|(p, _)| p.contains("-00001-of-"))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing 00001 meta shard in {repo}/{quant}"))?;
    let weight = files
        .iter()
        .find(|(p, _)| p.contains("-00002-of-"))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing 00002 weight shard in {repo}/{quant}"))?;

    let mut parts = Vec::new();
    for (path, file_bytes) in [meta, weight] {
        let cache_path = hdr_cache.join(path.replace('/', "__") + ".hdr");
        let bytes = if cache_path.is_file() {
            std::fs::read(&cache_path)
                .with_context(|| format!("read cache {}", cache_path.display()))?
        } else {
            let url = hub_resolve_url(repo, &path);
            eprintln!("rlx-inkling: Range-sniff {path} ({file_bytes} bytes on Hub)…");
            let bytes =
                fetch_gguf_header_prefix(&url).with_context(|| format!("header prefix {path}"))?;
            if let Ok(mut f) = std::fs::File::create(&cache_path) {
                let _ = f.write_all(&bytes);
            }
            bytes
        };
        let file =
            GgufFile::header_from_bytes(&bytes).with_context(|| format!("parse header {path}"))?;
        parts.push((
            file,
            GgufShardSniff {
                path: path.clone(),
                file_bytes: Some(file_bytes),
                prefix_bytes: bytes.len(),
                n_metadata: 0, // filled below
                n_tensors: 0,
            },
        ));
        let n = parts.len() - 1;
        parts[n].1.n_metadata = parts[n].0.metadata.len();
        parts[n].1.n_tensors = parts[n].0.tensors.len();
    }

    build_report(repo, quant, parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_prefixes_if_present() {
        let meta = PathBuf::from("/tmp/inkling-gguf/head.bin");
        let w = PathBuf::from("/tmp/inkling-gguf/s2head.bin");
        if !meta.is_file() || !w.is_file() {
            return;
        }
        let report = report_from_local_files(
            &meta,
            &w,
            "UD-IQ1_S/inkling-UD-IQ1_S-00001-of-00007.gguf",
            "UD-IQ1_S/inkling-UD-IQ1_S-00002-of-00007.gguf",
        )
        .expect("sniff");
        assert_eq!(report.architecture, "inkling");
        assert_eq!(report.text.layers, 66);
        assert_eq!(report.text.hidden, 6144);
        assert!(report.tensors.len() >= 300);
        assert!(
            report.dtype_histogram.contains_key("IQ1S")
                || report.dtype_histogram.keys().any(|k| k.contains("IQ"))
        );
        let emb = report
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .unwrap();
        assert_eq!(emb.shape, vec![6144, 201024]);
        let k0 = report
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_k.weight")
            .unwrap();
        assert_eq!(k0.shape, vec![6144, 2048]); // SWA kv
        let k5 = report
            .tensors
            .iter()
            .find(|t| t.name == "blk.5.attn_k.weight")
            .unwrap();
        assert_eq!(k5.shape, vec![6144, 1024]); // global kv
        let ex = report
            .tensors
            .iter()
            .find(|t| t.name == "blk.2.ffn_gate_exps.weight")
            .unwrap();
        assert_eq!(ex.shape.len(), 3);
        assert!(
            ex.shape.contains(&256) && ex.shape.contains(&6144),
            "gate_exps shape {:?} should include n_routed=256 and hidden=6144",
            ex.shape
        );
    }
}
