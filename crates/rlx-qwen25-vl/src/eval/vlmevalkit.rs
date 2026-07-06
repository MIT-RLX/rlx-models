// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Native VLMEvalKit-style dataset loading and scoring (RealWorldQA, TextVQA, JSONL).

use crate::eval::{
    EvalSummary, VqaSample, load_vqa_jsonl, normalize_answer, normalized_exact_match,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Supported VLMEvalKit benchmark exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlmevalkitDataset {
    /// Open-ended JSONL (RealWorldQA export, custom manifests).
    Jsonl,
    /// RealWorldQA TSV (multiple choice A–D).
    RealWorldQa,
    /// TextVQA TSV (open-ended; soft VQA score).
    TextVqa,
}

impl VlmevalkitDataset {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "jsonl" | "custom" => Some(Self::Jsonl),
            "realworldqa" | "rwqa" => Some(Self::RealWorldQa),
            "textvqa" | "tvqa" => Some(Self::TextVqa),
            _ => None,
        }
    }
}

/// Scoring rule applied to predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VlmevalkitMetric {
    #[default]
    /// Normalized exact match (open VQA).
    ExactMatch,
    /// Multiple-choice letter match (RealWorldQA).
    McqLetter,
    /// TextVQA soft accuracy (min ANLS over reference answers).
    TextVqaSoft,
}

impl VlmevalkitMetric {
    pub fn for_dataset(ds: VlmevalkitDataset) -> Self {
        match ds {
            VlmevalkitDataset::Jsonl => Self::ExactMatch,
            VlmevalkitDataset::RealWorldQa => Self::McqLetter,
            VlmevalkitDataset::TextVqa => Self::TextVqaSoft,
        }
    }
}

/// One eval item with optional MCQ options or multiple reference answers.
#[derive(Debug, Clone)]
pub struct VlmevalkitSample {
    pub id: String,
    pub image_path: PathBuf,
    pub question: String,
    /// Primary ground truth (letter for MCQ, string for open VQA).
    pub answer: String,
    pub choices: Option<[String; 4]>,
    pub reference_answers: Vec<String>,
}

impl VlmevalkitSample {
    pub fn into_vqa(self) -> VqaSample {
        VqaSample {
            id: self.id,
            image_path: self.image_path,
            question: self.question,
            answer: self.answer.clone(),
        }
    }
}

/// Load a VLMEvalKit-exported manifest.
pub fn load_vlmevalkit_dataset(
    dataset: VlmevalkitDataset,
    path: &Path,
    image_root: &Path,
) -> Result<Vec<VlmevalkitSample>> {
    match dataset {
        VlmevalkitDataset::Jsonl => load_jsonl_as_vlmevalkit(path, image_root),
        VlmevalkitDataset::RealWorldQa => load_realworldqa_tsv(path, image_root),
        VlmevalkitDataset::TextVqa => load_textvqa_tsv(path, image_root),
    }
}

fn load_jsonl_as_vlmevalkit(path: &Path, image_root: &Path) -> Result<Vec<VlmevalkitSample>> {
    load_vqa_jsonl(path, image_root)?
        .into_iter()
        .map(|s| {
            Ok(VlmevalkitSample {
                id: s.id,
                image_path: s.image_path,
                question: s.question,
                answer: s.answer.clone(),
                choices: None,
                reference_answers: if s.answer.is_empty() {
                    Vec::new()
                } else {
                    vec![s.answer]
                },
            })
        })
        .collect()
}

/// RealWorldQA TSV: `index`, `question`, `A`–`D`, `answer`, `image`.
pub fn load_realworldqa_tsv(path: &Path, image_root: &Path) -> Result<Vec<VlmevalkitSample>> {
    let rows = read_tsv(path)?;
    let header = rows
        .first()
        .context("empty RealWorldQA TSV")?
        .iter()
        .map(|h| h.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let idx = |name: &str| header.iter().position(|h| h == name);
    let i_q = idx("question").context("RealWorldQA TSV missing question")?;
    let i_a = idx("answer").context("RealWorldQA TSV missing answer")?;
    let i_img = idx("image")
        .or_else(|| idx("image_path"))
        .context("RealWorldQA missing image")?;
    let choice_cols: [usize; 4] = ["a", "b", "c", "d"]
        .map(|c| idx(c).unwrap_or_else(|| panic!("RealWorldQA TSV missing column {c}")));
    let i_id = idx("index").or_else(|| idx("id"));

    let mut out = Vec::new();
    for (row_no, row) in rows.iter().skip(1).enumerate() {
        let get = |i: usize| row.get(i).map(|s| s.as_str()).unwrap_or("").trim();
        let id = i_id
            .and_then(|i| row.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("row_{row_no}"));
        let choices: [String; 4] = std::array::from_fn(|i| get(choice_cols[i]).to_string());
        let answer = get(i_a).to_string();
        out.push(VlmevalkitSample {
            id,
            image_path: image_root.join(get(i_img)),
            question: get(i_q).to_string(),
            answer: answer.clone(),
            choices: Some(choices),
            reference_answers: vec![answer],
        });
    }
    Ok(out)
}

/// TextVQA TSV: `question_id`, `image`, `question`, `answers` (JSON list or pipe-separated).
pub fn load_textvqa_tsv(path: &Path, image_root: &Path) -> Result<Vec<VlmevalkitSample>> {
    let rows = read_tsv(path)?;
    let header = rows
        .first()
        .context("empty TextVQA TSV")?
        .iter()
        .map(|h| h.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let idx = |name: &str| header.iter().position(|h| h == name);
    let i_q = idx("question").context("TextVQA TSV missing question")?;
    let i_img = idx("image")
        .or_else(|| idx("image_path"))
        .context("TextVQA missing image")?;
    let i_ans = idx("answers").or_else(|| idx("answer"));
    let i_id = idx("question_id")
        .or_else(|| idx("index"))
        .or_else(|| idx("id"));

    let mut out = Vec::new();
    for (row_no, row) in rows.iter().skip(1).enumerate() {
        let get = |i: usize| row.get(i).map(|s| s.as_str()).unwrap_or("").trim();
        let id = i_id
            .and_then(|i| row.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("row_{row_no}"));
        let refs = i_ans
            .and_then(|i| row.get(i))
            .map(|s| parse_reference_answers(s.as_str()))
            .transpose()?
            .unwrap_or_default();
        let answer = refs.first().cloned().unwrap_or_default();
        out.push(VlmevalkitSample {
            id,
            image_path: image_root.join(get(i_img)),
            question: get(i_q).to_string(),
            answer,
            choices: None,
            reference_answers: refs,
        });
    }
    Ok(out)
}

fn parse_reference_answers(raw: &str) -> Result<Vec<String>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        let v: Vec<String> = serde_json::from_str(trimmed)
            .with_context(|| format!("parse answers JSON: {trimmed}"))?;
        return Ok(v);
    }
    Ok(trimmed
        .split('|')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn read_tsv(path: &Path) -> Result<Vec<Vec<String>>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(split_tsv_line(&line));
    }
    Ok(rows)
}

fn split_tsv_line(line: &str) -> Vec<String> {
    line.split('\t').map(|s| s.trim().to_string()).collect()
}

/// Extract MCQ option letter from a free-form model response (VLMEvalKit heuristic).
pub fn extract_mcq_letter(pred: &str) -> Option<char> {
    let p = pred.trim();
    for ch in p.chars().rev() {
        if matches!(ch, 'A' | 'B' | 'C' | 'D' | 'a' | 'b' | 'c' | 'd') {
            return Some(ch.to_ascii_uppercase());
        }
    }
    let lower = p.to_ascii_lowercase();
    for (letter, pat) in [
        ('A', "option a"),
        ('B', "option b"),
        ('C', "option c"),
        ('D', "option d"),
    ] {
        if lower.contains(pat) {
            return Some(letter);
        }
    }
    None
}

/// Normalized Levenshtein similarity in `[0, 1]`.
pub fn normalized_levenshtein(a: &str, b: &str) -> f32 {
    let a = normalize_answer(a);
    let b = normalize_answer(b);
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dist = levenshtein(&a, &b);
    let max_len = a.len().max(b.len()) as f32;
    1.0 - (dist as f32 / max_len)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (cur[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// TextVQA soft score: max normalized similarity vs any reference (threshold 0.5).
pub fn textvqa_soft_match(pred: &str, references: &[String]) -> bool {
    if references.is_empty() {
        return false;
    }
    references
        .iter()
        .any(|r| normalized_levenshtein(pred, r) >= 0.5)
}

/// Score one prediction under the chosen metric.
pub fn score_prediction(pred: &str, sample: &VlmevalkitSample, metric: VlmevalkitMetric) -> bool {
    match metric {
        VlmevalkitMetric::ExactMatch => normalized_exact_match(pred, &sample.answer),
        VlmevalkitMetric::McqLetter => {
            let gt = sample
                .answer
                .trim()
                .chars()
                .next()
                .unwrap_or(' ')
                .to_ascii_uppercase();
            extract_mcq_letter(pred).is_some_and(|p| p == gt)
        }
        VlmevalkitMetric::TextVqaSoft => {
            let refs = if sample.reference_answers.is_empty() {
                vec![sample.answer.clone()]
            } else {
                sample.reference_answers.clone()
            };
            textvqa_soft_match(pred, &refs)
        }
    }
}

/// Per-sample eval record (JSON-serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmevalkitRecord {
    pub id: String,
    pub question: String,
    pub ground_truth: String,
    pub baseline_pred: String,
    pub aif_pred: String,
    pub baseline_correct: bool,
    pub aif_correct: bool,
}

/// Aggregate eval report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmevalkitReport {
    pub dataset: String,
    pub metric: String,
    pub total: usize,
    pub baseline_acc: f64,
    pub aif_acc: f64,
    pub records: Vec<VlmevalkitRecord>,
}

impl VlmevalkitReport {
    pub fn from_records(
        dataset: VlmevalkitDataset,
        metric: VlmevalkitMetric,
        records: Vec<VlmevalkitRecord>,
    ) -> Self {
        let total = records.len();
        let base_ok = records.iter().filter(|r| r.baseline_correct).count();
        let aif_ok = records.iter().filter(|r| r.aif_correct).count();
        Self {
            dataset: format!("{dataset:?}"),
            metric: format!("{metric:?}"),
            total,
            baseline_acc: if total == 0 {
                0.0
            } else {
                base_ok as f64 / total as f64
            },
            aif_acc: if total == 0 {
                0.0
            } else {
                aif_ok as f64 / total as f64
            },
            records,
        }
    }

    pub fn summary(&self) -> EvalSummary {
        EvalSummary {
            total: self.total,
            correct: self.records.iter().filter(|r| r.baseline_correct).count(),
            correct_aif: self.records.iter().filter(|r| r.aif_correct).count(),
        }
    }
}

/// Build RealWorldQA MCQ prompt suffix (options appended to question).
pub fn realworldqa_question_with_choices(question: &str, choices: &[String; 4]) -> String {
    format!(
        "{question}\nA. {}\nB. {}\nC. {}\nD. {}",
        choices[0], choices[1], choices[2], choices[3]
    )
}

/// Resolve prompt text for a sample (MCQ options inlined when present).
pub fn sample_question_text(sample: &VlmevalkitSample) -> String {
    if let Some(ref c) = sample.choices {
        realworldqa_question_with_choices(&sample.question, c)
    } else {
        sample.question.clone()
    }
}

/// Infer dataset type from file extension / header sniffing.
pub fn infer_dataset(path: &Path) -> Result<VlmevalkitDataset> {
    if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
        return Ok(VlmevalkitDataset::Jsonl);
    }
    let rows = read_tsv(path)?;
    let header = rows
        .first()
        .context("empty TSV")?
        .iter()
        .map(|h| h.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if header.iter().any(|h| h == "a") && header.iter().any(|h| h == "answer") {
        return Ok(VlmevalkitDataset::RealWorldQa);
    }
    if header.iter().any(|h| h == "answers" || h == "question_id") {
        return Ok(VlmevalkitDataset::TextVqa);
    }
    bail!(
        "could not infer VLMEvalKit dataset from TSV header: {:?}",
        rows[0]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcq_letter_extraction() {
        assert_eq!(extract_mcq_letter("The answer is (B)."), Some('B'));
        assert_eq!(extract_mcq_letter("D"), Some('D'));
    }

    #[test]
    fn textvqa_soft_accepts_close_match() {
        assert!(textvqa_soft_match("metrolink", &["Metrolink".into()]));
        assert!(!textvqa_soft_match("bus", &["Metrolink".into()]));
    }

    #[test]
    fn load_realworldqa_tsv_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rlx_vlmevalkit_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tsv = dir.join("rwqa.tsv");
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&tsv).unwrap();
            writeln!(f, "index\tquestion\tA\tB\tC\tD\tanswer\timage").unwrap();
            writeln!(f, "1\tWhat color?\tred\tblue\tgreen\tyellow\tB\timg.jpg").unwrap();
        }
        std::fs::write(dir.join("img.jpg"), b"x").unwrap();
        let samples = load_realworldqa_tsv(&tsv, &dir).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].answer, "B");
        assert!(score_prediction(
            "Answer: B",
            &samples[0],
            VlmevalkitMetric::McqLetter
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
