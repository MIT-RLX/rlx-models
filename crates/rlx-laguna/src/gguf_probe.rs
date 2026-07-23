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

//! Hub GGUF **header** probe for Unsloth Laguna quants (HTTP Range only — no full shard).

#![cfg(feature = "hf-probe")]

use crate::config::{GGUF_ARCH, HF_GGUF_REPO, LagunaConfig};
use crate::gguf_layout::{GGUF_ARCHES, LAYOUT_NOTES, gguf_to_eager_key};
use anyhow::{Context, Result, bail};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use std::fmt;
use std::io::Read;

pub const DEFAULT_QUANT: &str = "UD-Q4_K_XL";
/// Enough for GGUF header + tensor table on the first split shard.
const HEADER_PREFIX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub struct GgufProbeReport {
    pub repo: String,
    pub quant: String,
    pub shard: String,
    pub arch: String,
    pub cfg: LagunaConfig,
    pub n_tensors: usize,
    pub sample_tensors: Vec<(String, Option<String>)>,
}

impl fmt::Display for GgufProbeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Laguna GGUF probe  repo={}  quant={}  shard={}",
            self.repo, self.quant, self.shard
        )?;
        writeln!(
            f,
            "  arch={}  layers={}  hidden={}  experts={}  tensors_in_prefix={}",
            self.arch,
            self.cfg.num_hidden_layers,
            self.cfg.hidden_size,
            self.cfg.num_experts,
            self.n_tensors
        )?;
        for (n, mapped) in &self.sample_tensors {
            match mapped {
                Some(k) => writeln!(f, "  {n} -> {k}")?,
                None => writeln!(f, "  {n} (pack/other)")?,
            }
        }
        writeln!(f)?;
        write!(f, "{LAYOUT_NOTES}")
    }
}

pub fn probe_remote(quant: &str) -> Result<GgufProbeReport> {
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("hf-hub ApiBuilder")?;
    let repo = api.repo(Repo::new(HF_GGUF_REPO.to_string(), RepoType::Model));
    let info = repo.info().context("Hub repo info")?;
    let mut candidates: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|n| n.ends_with(".gguf") && n.contains(quant))
        .collect();
    candidates.sort();
    let path = candidates
        .into_iter()
        .find(|n| n.contains("00001-of-") || !n.contains("-of-"))
        .ok_or_else(|| anyhow::anyhow!("no .gguf matching quant={quant} in {HF_GGUF_REPO}"))?;

    let url = format!("https://huggingface.co/{HF_GGUF_REPO}/resolve/main/{path}");
    let resp = ureq::get(&url)
        .set("Range", &format!("bytes=0-{}", HEADER_PREFIX_BYTES - 1))
        .call()
        .with_context(|| format!("Range GET {url}"))?;
    let status = resp.status();
    if status != 200 && status != 206 {
        bail!("unexpected HTTP {status} for {url}");
    }
    let mut buf = Vec::new();
    resp.into_reader()
        .take(HEADER_PREFIX_BYTES)
        .read_to_end(&mut buf)
        .context("read Range body")?;

    let raw = rlx_gguf::GgufFile::header_from_bytes(&buf)
        .with_context(|| format!("GGUF header from Range prefix of {path}"))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(rlx_gguf::MetaValue::as_str)
        .unwrap_or("?")
        .to_string();
    if !GGUF_ARCHES.contains(&arch.as_str()) {
        bail!("expected arch {GGUF_ARCH}, got {arch}");
    }
    let cfg = LagunaConfig::from_gguf(&raw)?;
    let est = crate::memory::estimate_ram(&raw);
    let mut names: Vec<_> = raw.tensors.keys().cloned().collect();
    names.sort();
    let n_tensors = names.len();
    let sample_tensors = names
        .into_iter()
        .take(16)
        .map(|n| {
            let mapped = gguf_to_eager_key(&n);
            (n, mapped)
        })
        .collect();

    println!("{}", crate::memory::PACKED_ONLY_POLICY);
    if est.tensor_count > 0 {
        println!(
            "  probe estimate: packed≈{:.2} GB  F32-expand≈{:.2} GB (refused)",
            est.packed_gb(),
            est.f32_gb()
        );
    }

    Ok(GgufProbeReport {
        repo: HF_GGUF_REPO.into(),
        quant: quant.into(),
        shard: path,
        arch,
        cfg,
        n_tensors,
        sample_tensors,
    })
}
