use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::metrics::{NoiseMetrics, SpectralMetrics, WhisperMetrics};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRow {
    pub model: String,
    pub device: String,
    pub phrase: String,
    pub scenario: String,
    pub status: String,
    pub skip_reason: Option<String>,
    pub error: Option<String>,
    pub wall_ms: Option<f64>,
    pub rtf: Option<f64>,
    pub audio_sec: Option<f64>,
    pub sample_rate: Option<u32>,
    pub exec_label: Option<String>,
    pub cosine_vs_cpu: Option<f64>,
    pub whisper: Option<WhisperMetrics>,
    pub spectral: Option<SpectralMetrics>,
    pub noise: Option<NoiseMetrics>,
    pub wav_rel: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub n_rows: usize,
    pub n_ok: usize,
    pub n_skipped: usize,
    pub n_failed: usize,
    pub by_model: BTreeMap<String, ModelSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub n_ok: usize,
    pub median_rtf: Option<f64>,
    pub median_whisper_cov: Option<f64>,
}

pub fn write_results_jsonl(path: &Path, rows: &[BenchRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("{}", path.display()))?);
    for row in rows {
        serde_json::to_writer(&mut w, row)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

pub fn append_results_jsonl(path: &Path, rows: &[BenchRow]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("{}", path.display()))?,
    );
    for row in rows {
        serde_json::to_writer(&mut w, row)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_results_jsonl(path: &Path) -> Result<Vec<BenchRow>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let f = File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(line).with_context(|| format!("parse {}", path.display()))?);
    }
    Ok(rows)
}

pub fn write_summary_json(path: &Path, rows: &[BenchRow]) -> Result<Summary> {
    let mut by_model: BTreeMap<String, Vec<&BenchRow>> = BTreeMap::new();
    for r in rows {
        by_model.entry(r.model.clone()).or_default().push(r);
    }
    let mut model_sum = BTreeMap::new();
    for (model, rs) in &by_model {
        let ok: Vec<_> = rs.iter().filter(|r| r.status == "ok").collect();
        let mut rtfs: Vec<f64> = ok.iter().filter_map(|r| r.rtf).collect();
        rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut covs: Vec<f64> = ok
            .iter()
            .filter_map(|r| r.whisper.as_ref().map(|w| w.coverage))
            .collect();
        covs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        model_sum.insert(
            model.clone(),
            ModelSummary {
                n_ok: ok.len(),
                median_rtf: rtfs.get(rtfs.len() / 2).copied(),
                median_whisper_cov: covs.get(covs.len() / 2).copied(),
            },
        );
    }
    let summary = Summary {
        n_rows: rows.len(),
        n_ok: rows.iter().filter(|r| r.status == "ok").count(),
        n_skipped: rows.iter().filter(|r| r.status == "skipped").count(),
        n_failed: rows.iter().filter(|r| r.status == "failed").count(),
        by_model: model_sum,
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(path).with_context(|| format!("{}", path.display()))?;
    serde_json::to_writer_pretty(f, &summary)?;
    Ok(summary)
}
