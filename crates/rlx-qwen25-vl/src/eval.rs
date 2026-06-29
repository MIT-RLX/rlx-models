// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// VLMEvalKit-style VQA eval helpers for AIF paper benchmarks.

mod vlmevalkit;

pub use vlmevalkit::{
    VlmevalkitDataset, VlmevalkitMetric, VlmevalkitRecord, VlmevalkitReport, VlmevalkitSample,
    extract_mcq_letter, infer_dataset, load_realworldqa_tsv, load_textvqa_tsv,
    load_vlmevalkit_dataset, normalized_levenshtein, realworldqa_question_with_choices,
    sample_question_text, score_prediction, textvqa_soft_match,
};

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// One open-ended VQA sample (RealWorldQA / TextVQA TSV/JSONL export).
#[derive(Debug, Clone)]
pub struct VqaSample {
    pub id: String,
    pub image_path: PathBuf,
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Deserialize)]
struct JsonlRow {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    question_id: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    image_path: Option<String>,
    question: String,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    gt: Option<String>,
    #[serde(default)]
    #[serde(rename = "ground_truth")]
    ground_truth: Option<String>,
}

/// Load JSONL exported from VLMEvalKit / RealWorldQA-style manifests.
pub fn load_vqa_jsonl(path: &Path, image_root: &Path) -> Result<Vec<VqaSample>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: JsonlRow =
            serde_json::from_str(&line).with_context(|| format!("parse line {}", line_no + 1))?;
        let id = row
            .id
            .or(row.question_id)
            .unwrap_or_else(|| format!("line_{}", line_no + 1));
        let rel = row
            .image_path
            .or(row.image)
            .ok_or_else(|| anyhow::anyhow!("line {} missing image path", line_no + 1))?;
        let answer = row
            .answer
            .or(row.gt)
            .or(row.ground_truth)
            .unwrap_or_default();
        out.push(VqaSample {
            id,
            image_path: image_root.join(rel),
            question: row.question,
            answer,
        });
    }
    Ok(out)
}

/// Normalized exact match (VLMEvalKit relaxed string match proxy).
pub fn normalized_exact_match(pred: &str, gt: &str) -> bool {
    normalize_answer(pred) == normalize_answer(gt)
}

pub fn normalize_answer(s: &str) -> String {
    s.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Clone, Default)]
pub struct EvalSummary {
    pub total: usize,
    pub correct: usize,
    pub correct_aif: usize,
}

impl EvalSummary {
    pub fn baseline_acc(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    pub fn aif_acc(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct_aif as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punct() {
        assert_eq!(normalize_answer("  Hello, World! "), "hello world");
    }

    #[test]
    fn exact_match_case_insensitive() {
        assert!(normalized_exact_match("Metrolink", "metrolink"));
        assert!(!normalized_exact_match("Burry", "Metrolink"));
    }
}
