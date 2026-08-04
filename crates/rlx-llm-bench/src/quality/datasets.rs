// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dataset documents + JSONL loaders for the quality tasks, plus tiny synthetic
//! sets so the harness (and its tests) run with no files or weights on disk.
//!
//! JSONL shapes accepted (one object per line):
//! - **MMLU / MC**: `{"question": str, "choices": [str, …], "answer": int|str}`
//!   where `answer` is a 0-based index or a letter (`"A"`, `"b"`, …). An
//!   optional `"subject"` is used for the prompt preamble.
//! - **GSM8K**: `{"question": str, "answer": str}` where `answer` is the full
//!   solution ending in `#### <number>` (the canonical GSM8K format).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

/// A multiple-choice document (MMLU, ARC, HellaSwag, …).
#[derive(Debug, Clone)]
pub struct McDoc {
    pub question: String,
    pub choices: Vec<String>,
    /// 0-based index of the correct choice.
    pub answer: usize,
    pub subject: Option<String>,
}

/// A generative document (GSM8K-style): a question and its gold answer string.
#[derive(Debug, Clone)]
pub struct GenDoc {
    pub question: String,
    /// Full gold answer text; the numeric target is parsed out at scoring time.
    pub answer: String,
}

// ── JSONL row shapes ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct McRow {
    question: String,
    choices: Vec<String>,
    answer: serde_json::Value,
    #[serde(default)]
    subject: Option<String>,
}

#[derive(Deserialize)]
struct GenRow {
    question: String,
    answer: String,
}

/// Parse an `answer` field that may be a 0-based integer index or a letter
/// (`"A".."Z"`, case-insensitive) into a choice index.
fn parse_answer_index(v: &serde_json::Value, n_choices: usize) -> Result<usize> {
    let idx = match v {
        serde_json::Value::Number(n) => n
            .as_u64()
            .ok_or_else(|| anyhow!("answer number is not a non-negative integer: {n}"))?
            as usize,
        serde_json::Value::String(s) => {
            let t = s.trim();
            if let Ok(i) = t.parse::<usize>() {
                i
            } else {
                let c = t
                    .chars()
                    .next()
                    .ok_or_else(|| anyhow!("empty answer string"))?
                    .to_ascii_uppercase();
                if !c.is_ascii_uppercase() {
                    bail!("answer letter not A-Z: {s:?}");
                }
                (c as u8 - b'A') as usize
            }
        }
        other => bail!("unsupported answer type: {other}"),
    };
    if idx >= n_choices {
        bail!("answer index {idx} out of range for {n_choices} choices");
    }
    Ok(idx)
}

/// Load multiple-choice docs from a JSONL file.
pub fn load_mc_jsonl(path: &Path) -> Result<Vec<McDoc>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading MC dataset {}", path.display()))?;
    let mut docs = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: McRow = serde_json::from_str(line)
            .with_context(|| format!("{}: line {}", path.display(), lineno + 1))?;
        if row.choices.is_empty() {
            bail!("{}: line {} has no choices", path.display(), lineno + 1);
        }
        let answer = parse_answer_index(&row.answer, row.choices.len())
            .with_context(|| format!("{}: line {}", path.display(), lineno + 1))?;
        docs.push(McDoc {
            question: row.question,
            choices: row.choices,
            answer,
            subject: row.subject,
        });
    }
    if docs.is_empty() {
        bail!("{} contained no MC documents", path.display());
    }
    Ok(docs)
}

/// Load generative (GSM8K-style) docs from a JSONL file.
pub fn load_gen_jsonl(path: &Path) -> Result<Vec<GenDoc>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading generative dataset {}", path.display()))?;
    let mut docs = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: GenRow = serde_json::from_str(line)
            .with_context(|| format!("{}: line {}", path.display(), lineno + 1))?;
        docs.push(GenDoc {
            question: row.question,
            answer: row.answer,
        });
    }
    if docs.is_empty() {
        bail!("{} contained no generative documents", path.display());
    }
    Ok(docs)
}

/// A handful of trivial MC items for smoke tests / `--dry-run` (no files).
pub fn synthetic_mc() -> Vec<McDoc> {
    let mk = |q: &str, choices: &[&str], answer: usize| McDoc {
        question: q.to_string(),
        choices: choices.iter().map(|s| s.to_string()).collect(),
        answer,
        subject: Some("general".into()),
    };
    vec![
        mk(
            "The capital of France is",
            &["Paris", "Berlin", "Rome", "Madrid"],
            0,
        ),
        mk("Two plus two equals", &["three", "four", "five", "six"], 1),
        mk(
            "Water is made of hydrogen and",
            &["helium", "carbon", "oxygen", "nitrogen"],
            2,
        ),
    ]
}

/// A couple of trivial generative items for smoke tests (no files).
pub fn synthetic_gen() -> Vec<GenDoc> {
    vec![
        GenDoc {
            question: "Natalia has 3 apples and buys 2 more. How many apples does she have?".into(),
            answer: "She has 3 + 2 = 5 apples.\n#### 5".into(),
        },
        GenDoc {
            question: "A pack has 6 pens. How many pens are in 4 packs?".into(),
            answer: "6 * 4 = 24 pens.\n#### 24".into(),
        },
    ]
}
