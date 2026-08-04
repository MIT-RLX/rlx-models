// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Real dataset fetch for the quality tasks, via the HuggingFace
//! **datasets-server** JSON rows API — no auth, no parquet reader, no giant
//! tarball. Public datasets stream as paginated JSON:
//!
//! ```text
//! https://datasets-server.huggingface.co/rows?dataset=<d>&config=<c>&split=<s>&offset=<o>&length=<=100
//! ```
//!
//! Fetched rows are normalized into this crate's JSONL shapes and written to a
//! cache file, so scoring flows through the same [`load_mc_jsonl`] /
//! [`load_gen_jsonl`] loaders and re-runs are offline.
//!
//! Sources: MMLU = `cais/mmlu` (config `all`, 14 042 test rows); GSM8K =
//! `openai/gsm8k` (config `main`, 1 319 test rows).
//!
//! [`load_mc_jsonl`]: super::datasets::load_mc_jsonl
//! [`load_gen_jsonl`]: super::datasets::load_gen_jsonl

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

const DS_SERVER: &str = "https://datasets-server.huggingface.co/rows";
const PAGE: usize = 100; // datasets-server hard cap per request

/// Where and how much to fetch.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Directory for cached JSONL files.
    pub cache_dir: PathBuf,
    /// Cap on rows fetched (`None` = the full split).
    pub limit: Option<usize>,
    /// Re-download even if a cached file exists.
    pub force: bool,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            limit: None,
            force: false,
        }
    }
}

/// Default cache directory (`$RLX_LLM_BENCH_CACHE` or `.cache/llm-bench`).
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("RLX_LLM_BENCH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".cache/llm-bench"))
}

/// Percent-encode the one character (`/`) that appears in HF dataset ids.
fn enc(s: &str) -> String {
    s.replace('/', "%2F")
}

fn http_get_json(url: &str) -> Result<serde_json::Value> {
    let body = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP GET {url}: {e}"))?
        .into_string()
        .map_err(|e| anyhow!("reading body of {url}: {e}"))?;
    serde_json::from_str(&body).with_context(|| format!("parsing JSON rows from {url}"))
}

/// Paginate the rows API, calling `on_row` with each `row` object. Returns the
/// number of rows delivered.
fn download_rows(
    dataset: &str,
    config: &str,
    split: &str,
    limit: Option<usize>,
    mut on_row: impl FnMut(&serde_json::Value) -> Result<()>,
) -> Result<usize> {
    let mut offset = 0usize;
    let mut delivered = 0usize;
    loop {
        let url = format!(
            "{DS_SERVER}?dataset={}&config={config}&split={split}&offset={offset}&length={PAGE}",
            enc(dataset)
        );
        let v = http_get_json(&url)?;
        let rows = v["rows"]
            .as_array()
            .ok_or_else(|| anyhow!("unexpected response (no `rows`) from {url}"))?;
        if rows.is_empty() {
            break;
        }
        for r in rows {
            on_row(&r["row"])?;
            delivered += 1;
            if let Some(l) = limit {
                if delivered >= l {
                    return Ok(delivered);
                }
            }
        }
        offset += rows.len();
        let total = v["num_rows_total"].as_u64().unwrap_or(0) as usize;
        if total > 0 && offset >= total {
            break;
        }
    }
    Ok(delivered)
}

// ── Row → JSONL normalization (pure, unit-tested) ───────────────────────────

/// Map a `cais/mmlu` row to this crate's MC JSONL object, or `None` if the row
/// is malformed (missing fields / no choices).
pub fn mmlu_row_to_json(row: &serde_json::Value) -> Option<serde_json::Value> {
    let question = row.get("question")?.as_str()?;
    let subject = row.get("subject").and_then(|s| s.as_str()).unwrap_or("");
    let choices: Vec<&str> = row
        .get("choices")?
        .as_array()?
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    // `answer` is a ClassLabel index (0-based) on cais/mmlu.
    let answer = row.get("answer")?.as_i64()?;
    if question.is_empty() || choices.is_empty() || answer < 0 || answer as usize >= choices.len() {
        return None;
    }
    Some(serde_json::json!({
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": subject,
    }))
}

/// Map an `openai/gsm8k` row to this crate's generative JSONL object.
pub fn gsm8k_row_to_json(row: &serde_json::Value) -> Option<serde_json::Value> {
    let question = row.get("question")?.as_str()?;
    let answer = row.get("answer")?.as_str()?;
    if question.is_empty() || answer.is_empty() {
        return None;
    }
    Some(serde_json::json!({ "question": question, "answer": answer }))
}

// ── Public fetch entry points ───────────────────────────────────────────────

fn cache_path(cfg: &FetchConfig, stem: &str) -> PathBuf {
    let name = match cfg.limit {
        Some(l) => format!("{stem}-{l}.jsonl"),
        None => format!("{stem}.jsonl"),
    };
    cfg.cache_dir.join(name)
}

/// Stream `dataset/config/split` through `map`, writing normalized JSONL to a
/// cache file. Returns the cache path (reused when present unless `force`).
fn fetch_to_jsonl(
    cfg: &FetchConfig,
    stem: &str,
    dataset: &str,
    config: &str,
    split: &str,
    map: impl Fn(&serde_json::Value) -> Option<serde_json::Value>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(&cfg.cache_dir)
        .with_context(|| format!("creating cache dir {}", cfg.cache_dir.display()))?;
    let out = cache_path(cfg, stem);
    if out.is_file() && !cfg.force {
        eprintln!("[fetch] {stem}: using cached {}", out.display());
        return Ok(out);
    }

    // Write to a temp file then rename, so an interrupted download never leaves
    // a truncated cache that later reads as complete.
    let tmp = out.with_extension("jsonl.partial");
    let mut f =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut kept = 0usize;
    let delivered = download_rows(dataset, config, split, cfg.limit, |row| {
        if let Some(obj) = map(row) {
            writeln!(f, "{}", serde_json::to_string(&obj)?)?;
            kept += 1;
        }
        Ok(())
    })?;
    f.flush()?;
    drop(f);
    if kept == 0 {
        let _ = std::fs::remove_file(&tmp);
        bail!("fetched {delivered} rows from {dataset} but none were usable");
    }
    std::fs::rename(&tmp, &out).with_context(|| format!("finalizing {}", out.display()))?;
    eprintln!(
        "[fetch] {stem}: {kept}/{delivered} rows -> {}",
        out.display()
    );
    Ok(out)
}

/// Fetch MMLU (`cais/mmlu`, config `all`, `test`) into the cache; returns the
/// JSONL path for [`load_mc_jsonl`](super::datasets::load_mc_jsonl).
pub fn fetch_mmlu(cfg: &FetchConfig) -> Result<PathBuf> {
    fetch_to_jsonl(
        cfg,
        "mmlu-all-test",
        "cais/mmlu",
        "all",
        "test",
        mmlu_row_to_json,
    )
}

/// Fetch GSM8K (`openai/gsm8k`, config `main`, `test`) into the cache; returns
/// the JSONL path for [`load_gen_jsonl`](super::datasets::load_gen_jsonl).
pub fn fetch_gsm8k(cfg: &FetchConfig) -> Result<PathBuf> {
    fetch_to_jsonl(
        cfg,
        "gsm8k-main-test",
        "openai/gsm8k",
        "main",
        "test",
        gsm8k_row_to_json,
    )
}

// ── Multiple-choice bench registry ──────────────────────────────────────────

/// A registered multiple-choice benchmark: HF coordinates + its natural scoring
/// mode. Every entry normalizes into the MC JSONL shape and scores through
/// [`super::run_mmlu`], so it inherits the fast bucketed packed path for
/// single-token (`Letter`) modes.
pub struct McSource {
    pub name: &'static str,
    pub dataset: &'static str,
    pub config: &'static str,
    pub split: &'static str,
    pub default_mode: super::MmluMode,
}

/// All multiple-choice benchmarks the harness knows how to fetch.
pub const MC_SOURCES: &[McSource] = &[
    McSource {
        name: "mmlu",
        dataset: "cais/mmlu",
        config: "all",
        split: "test",
        default_mode: super::MmluMode::Letter,
    },
    McSource {
        name: "arc_challenge",
        dataset: "allenai/ai2_arc",
        config: "ARC-Challenge",
        split: "test",
        default_mode: super::MmluMode::Letter,
    },
    McSource {
        name: "arc_easy",
        dataset: "allenai/ai2_arc",
        config: "ARC-Easy",
        split: "test",
        default_mode: super::MmluMode::Letter,
    },
    McSource {
        name: "openbookqa",
        dataset: "allenai/openbookqa",
        config: "main",
        split: "test",
        default_mode: super::MmluMode::Letter,
    },
    McSource {
        name: "hellaswag",
        dataset: "Rowan/hellaswag",
        config: "default",
        split: "validation",
        default_mode: super::MmluMode::Raw,
    },
    McSource {
        name: "winogrande",
        dataset: "allenai/winogrande",
        config: "winogrande_xl",
        split: "validation",
        default_mode: super::MmluMode::Raw,
    },
];

/// Look up a registered MC bench by name.
pub fn mc_source(name: &str) -> Option<&'static McSource> {
    MC_SOURCES.iter().find(|s| s.name == name)
}

/// ARC / OpenBookQA: `{question|question_stem, choices:{text,label}, answerKey}`.
fn map_arc(row: &Value) -> Option<Value> {
    let question = row
        .get("question")
        .or_else(|| row.get("question_stem"))?
        .as_str()?;
    let ch = row.get("choices")?;
    let texts: Vec<&str> = ch
        .get("text")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    let labels: Vec<&str> = ch
        .get("label")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    let key = row.get("answerKey")?.as_str()?;
    let ans = labels.iter().position(|l| *l == key)?;
    if texts.is_empty() || ans >= texts.len() {
        return None;
    }
    Some(serde_json::json!({ "question": question, "choices": texts, "answer": ans }))
}

/// HellaSwag: `{ctx, endings, label}` (label is a stringified index).
fn map_hellaswag(row: &Value) -> Option<Value> {
    let ctx = row.get("ctx")?.as_str()?;
    let endings: Vec<&str> = row
        .get("endings")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    let lv = row.get("label")?;
    let ans = lv
        .as_str()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| lv.as_u64().map(|u| u as usize))?;
    if endings.is_empty() || ans >= endings.len() {
        return None;
    }
    Some(serde_json::json!({ "question": ctx, "choices": endings, "answer": ans }))
}

/// WinoGrande: `{sentence (with `_`), option1, option2, answer:"1"|"2"}`. The
/// stem is split at the blank; each choice is `option ++ suffix`, scored as a
/// continuation of the prefix.
fn map_winogrande(row: &Value) -> Option<Value> {
    let sent = row.get("sentence")?.as_str()?;
    let o1 = row.get("option1")?.as_str()?;
    let o2 = row.get("option2")?.as_str()?;
    let ans: usize = row.get("answer")?.as_str()?.parse().ok()?;
    if ans != 1 && ans != 2 {
        return None;
    }
    let (prefix, suffix) = sent.split_once('_')?;
    Some(serde_json::json!({
        "question": prefix.trim_end(),
        "choices": [format!("{o1}{suffix}"), format!("{o2}{suffix}")],
        "answer": ans - 1,
    }))
}

fn map_for(name: &str) -> fn(&Value) -> Option<Value> {
    match name {
        "arc_challenge" | "arc_easy" | "openbookqa" => map_arc,
        "hellaswag" => map_hellaswag,
        "winogrande" => map_winogrande,
        _ => mmlu_row_to_json,
    }
}

/// Fetch a registered MC bench into the cache; returns the JSONL path for
/// [`load_mc_jsonl`](super::datasets::load_mc_jsonl).
pub fn fetch_mc(name: &str, cfg: &FetchConfig) -> Result<PathBuf> {
    let src = mc_source(name).ok_or_else(|| {
        anyhow!(
            "unknown MC bench {name:?}; known: {}",
            MC_SOURCES
                .iter()
                .map(|s| s.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let stem = format!("{name}-{}", src.split);
    fetch_to_jsonl(
        cfg,
        &stem,
        src.dataset,
        src.config,
        src.split,
        map_for(name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmlu_row_maps_and_validates() {
        let row = serde_json::json!({
            "question": "2+2?",
            "subject": "math",
            "choices": ["3", "4", "5", "6"],
            "answer": 1
        });
        let obj = mmlu_row_to_json(&row).expect("valid row maps");
        assert_eq!(obj["answer"], 1);
        assert_eq!(obj["choices"].as_array().unwrap().len(), 4);
        assert_eq!(obj["subject"], "math");

        // Out-of-range answer is rejected.
        let bad = serde_json::json!({"question": "q", "choices": ["a", "b"], "answer": 5});
        assert!(mmlu_row_to_json(&bad).is_none());
        // Missing choices is rejected.
        let bad2 = serde_json::json!({"question": "q", "answer": 0});
        assert!(mmlu_row_to_json(&bad2).is_none());
    }

    #[test]
    fn gsm8k_row_maps() {
        let row = serde_json::json!({"question": "How many?", "answer": "work\n#### 7"});
        let obj = gsm8k_row_to_json(&row).expect("valid row maps");
        assert_eq!(obj["question"], "How many?");
        assert!(obj["answer"].as_str().unwrap().contains("#### 7"));
        // Empty answer rejected.
        let bad = serde_json::json!({"question": "q", "answer": ""});
        assert!(gsm8k_row_to_json(&bad).is_none());
    }

    #[test]
    fn cache_path_encodes_limit() {
        let cfg = FetchConfig {
            cache_dir: PathBuf::from("/tmp/x"),
            limit: Some(50),
            force: false,
        };
        assert_eq!(
            cache_path(&cfg, "mmlu-all-test"),
            PathBuf::from("/tmp/x/mmlu-all-test-50.jsonl")
        );
        let cfg2 = FetchConfig { limit: None, ..cfg };
        assert_eq!(
            cache_path(&cfg2, "mmlu-all-test"),
            PathBuf::from("/tmp/x/mmlu-all-test.jsonl")
        );
    }
}
