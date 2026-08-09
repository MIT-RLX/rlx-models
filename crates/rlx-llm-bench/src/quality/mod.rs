// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Quality dimension: accuracy on standard tasks.
//!
//! - [`run_mmlu`] — multiple-choice log-likelihood scoring (MMLU/ARC/…). Two
//!   conventions: [`MmluMode::Cloze`] scores each answer *text* as a
//!   continuation (length-normalized, robust for base models), [`MmluMode::Letter`]
//!   enumerates `A./B./…` and scores the single answer letter.
//! - [`run_gsm8k`] — generative exact-match: prompt (optionally few-shot),
//!   greedy-decode, take the last number, compare to the gold `#### N`.
//! - [`run_perplexity`] — reuses [`fn@rlx_eval::perplexity`] via the [`BenchModel`]
//!   `LmLogprobs` impl.

pub mod datasets;
pub mod fetch;
pub mod gsm8k;

use anyhow::Result;

use crate::model::BenchModel;
use datasets::{GenDoc, McDoc};
use rlx_eval::{McItem, PerplexityConfig};

/// One flattened quality metric for the leaderboard.
#[derive(Debug, Clone)]
pub struct QualityRow {
    pub task: String,
    pub n: usize,
    pub metric: String,
    pub value: f64,
}

// ── MMLU / multiple-choice ──────────────────────────────────────────────────

/// How a multiple-choice item is rendered and scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MmluMode {
    /// Question + `\nAnswer:`, then score each answer *text* as a continuation;
    /// pick the highest length-normalized log-prob. MMLU cloze default.
    #[default]
    Cloze,
    /// Enumerate the choices as `A. … B. …` and score the single answer letter.
    /// Single-token → runs on the fast bucketed packed path.
    Letter,
    /// Score each choice as a *direct* continuation of the question with no
    /// `Answer:` scaffold — for sentence-completion tasks (HellaSwag) and
    /// fill-in-the-blank (WinoGrande) where the choice simply follows the stem.
    Raw,
}

/// MMLU run options.
#[derive(Debug, Clone, Default)]
pub struct MmluOptions {
    pub mode: MmluMode,
    /// Cap the number of documents scored (`None` = all).
    pub max_docs: Option<usize>,
}

/// Per-document MMLU prediction (for cross-harness agreement checks).
#[derive(Debug, Clone, Copy)]
pub struct McPred {
    pub gold: usize,
    pub best: usize,
    pub best_norm: usize,
}

/// MMLU result: raw accuracy and length-normalized accuracy.
#[derive(Debug, Clone)]
pub struct MmluResult {
    pub n: usize,
    /// Fraction correct by raw summed log-prob (`best`).
    pub acc: f64,
    /// Fraction correct by length-normalized log-prob (`best_norm`).
    pub acc_norm: f64,
    pub mode: MmluMode,
    /// Per-document predictions in scored order.
    pub preds: Vec<McPred>,
}

impl MmluResult {
    /// The headline accuracy for this mode (`acc_norm` for Cloze, `acc` for
    /// Letter, whose single-token choices make normalization a no-op).
    pub fn headline(&self) -> f64 {
        match self.mode {
            MmluMode::Cloze | MmluMode::Raw => self.acc_norm,
            MmluMode::Letter => self.acc,
        }
    }

    pub fn bench_line(&self, name: &str, device: &str) -> String {
        format!(
            "LLMBENCH kind=mmlu model={name} device={device} n={} acc={:.4} acc_norm={:.4} mode={:?}",
            self.n, self.acc, self.acc_norm, self.mode
        )
    }

    pub fn rows(&self) -> Vec<QualityRow> {
        vec![
            QualityRow {
                task: "mmlu".into(),
                n: self.n,
                metric: "acc".into(),
                value: self.acc,
            },
            QualityRow {
                task: "mmlu".into(),
                n: self.n,
                metric: "acc_norm".into(),
                value: self.acc_norm,
            },
        ]
    }
}

const LETTERS: &[&str] = &[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

/// Render the shared context string for one MC doc.
fn render_context(doc: &McDoc, mode: MmluMode) -> String {
    let mut ctx = String::new();
    // `Raw` is a bare sentence stem — no subject preamble, no `Answer:` scaffold.
    if mode != MmluMode::Raw {
        if let Some(subj) = &doc.subject {
            ctx.push_str(&format!(
                "The following is a multiple choice question about {subj}.\n\n"
            ));
        }
    }
    ctx.push_str(&doc.question);
    match mode {
        MmluMode::Raw => {}
        MmluMode::Cloze => {
            ctx.push_str("\nAnswer:");
        }
        MmluMode::Letter => {
            ctx.push('\n');
            for (i, c) in doc.choices.iter().enumerate() {
                let letter = LETTERS.get(i).copied().unwrap_or("?");
                ctx.push_str(&format!("{letter}. {c}\n"));
            }
            ctx.push_str("Answer:");
        }
    }
    ctx
}

/// Continuation strings scored against the context (one per choice).
fn choice_continuations(doc: &McDoc, mode: MmluMode) -> Vec<String> {
    match mode {
        MmluMode::Cloze | MmluMode::Raw => doc.choices.iter().map(|c| format!(" {c}")).collect(),
        MmluMode::Letter => (0..doc.choices.len())
            .map(|i| format!(" {}", LETTERS.get(i).copied().unwrap_or("?")))
            .collect(),
    }
}

/// Score MMLU-style multiple-choice accuracy over `docs`.
pub fn run_mmlu(model: &mut BenchModel, docs: &[McDoc], opts: &MmluOptions) -> Result<MmluResult> {
    let take = opts.max_docs.unwrap_or(docs.len()).min(docs.len());
    let mut n = 0usize;
    let mut correct = 0usize;
    let mut correct_norm = 0usize;
    let mut preds = Vec::with_capacity(take);

    for doc in &docs[..take] {
        let ctx_text = render_context(doc, opts.mode);
        let context = model.encode(&ctx_text)?;
        let mut choices = Vec::with_capacity(doc.choices.len());
        for cont in choice_continuations(doc, opts.mode) {
            choices.push(model.encode(&cont)?);
        }
        let item = McItem { context, choices };
        let res = model.score_mc(&item)?;
        n += 1;
        if res.best == doc.answer {
            correct += 1;
        }
        if res.best_norm == doc.answer {
            correct_norm += 1;
        }
        preds.push(McPred {
            gold: doc.answer,
            best: res.best,
            best_norm: res.best_norm,
        });
    }

    let denom = n.max(1) as f64;
    Ok(MmluResult {
        n,
        acc: correct as f64 / denom,
        acc_norm: correct_norm as f64 / denom,
        mode: opts.mode,
        preds,
    })
}

// ── GSM8K / generative ──────────────────────────────────────────────────────

/// A compact 4-shot chain-of-thought preamble in the canonical GSM8K style,
/// each example ending in a final number so the last-number extractor lands on
/// the answer. Override via [`Gsm8kOptions::fewshot`] (or `None` for zero-shot).
pub const DEFAULT_GSM8K_FEWSHOT: &str = "\
Question: There are 15 trees in the grove. Grove workers will plant trees today. After they are done there will be 21 trees. How many trees did they plant?
Answer: There were 15 trees, then 21 trees, so they planted 21 - 15 = 6 trees. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more arrive, how many cars are in the parking lot?
Answer: There are 3 cars and 2 more arrive, so 3 + 2 = 5 cars. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Together they had 32 + 42 = 74. After eating 35, they have 74 - 35 = 39. The answer is 39.

Question: Shawn has five toys. For Christmas he got two toys each from his mom and dad. How many toys does he have now?
Answer: He starts with 5. He gets 2 from mom and 2 from dad, so 2 + 2 = 4 more. 5 + 4 = 9. The answer is 9.

";

/// GSM8K run options.
#[derive(Debug, Clone)]
pub struct Gsm8kOptions {
    pub max_new_tokens: usize,
    /// Few-shot preamble prepended to every question. `None` = zero-shot.
    pub fewshot: Option<String>,
    pub max_docs: Option<usize>,
}

impl Default for Gsm8kOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            fewshot: Some(DEFAULT_GSM8K_FEWSHOT.to_string()),
            max_docs: None,
        }
    }
}

/// Per-document GSM8K prediction (for cross-harness agreement checks).
#[derive(Debug, Clone)]
pub struct GenPred {
    pub gold: Option<String>,
    pub pred: Option<String>,
    pub correct: bool,
}

/// GSM8K result: exact-match accuracy.
#[derive(Debug, Clone)]
pub struct Gsm8kResult {
    pub n: usize,
    pub acc: f64,
    /// Per-document predictions in scored order.
    pub preds: Vec<GenPred>,
}

impl Gsm8kResult {
    pub fn bench_line(&self, name: &str, device: &str) -> String {
        format!(
            "LLMBENCH kind=gsm8k model={name} device={device} n={} acc={:.4}",
            self.n, self.acc
        )
    }

    pub fn row(&self) -> QualityRow {
        QualityRow {
            task: "gsm8k".into(),
            n: self.n,
            metric: "acc".into(),
            value: self.acc,
        }
    }
}

/// Score GSM8K exact-match accuracy over `docs`.
pub fn run_gsm8k(
    model: &mut BenchModel,
    docs: &[GenDoc],
    opts: &Gsm8kOptions,
) -> Result<Gsm8kResult> {
    let take = opts.max_docs.unwrap_or(docs.len()).min(docs.len());
    let preamble = opts.fewshot.as_deref().unwrap_or("");
    let mut n = 0usize;
    let mut correct = 0usize;
    let mut preds = Vec::with_capacity(take);

    for doc in &docs[..take] {
        let prompt = format!("{preamble}Question: {}\nAnswer:", doc.question);
        let ids = model.encode(&prompt)?;
        let (_ids, text) = model.generate_text(&ids, opts.max_new_tokens)?;
        // Stop at a hallucinated next question so we score only this answer.
        let answer_span = text.split("Question:").next().unwrap_or(&text);
        let pred = gsm8k::extract_pred(answer_span);
        let gold = gsm8k::extract_gold(&doc.answer);
        let is_correct = matches!(
            (&pred, &gold),
            (Some(p), Some(g)) if gsm8k::answers_match(p, g)
        );
        n += 1;
        if is_correct {
            correct += 1;
        }
        preds.push(GenPred {
            gold,
            pred,
            correct: is_correct,
        });
    }

    Ok(Gsm8kResult {
        n,
        acc: correct as f64 / n.max(1) as f64,
        preds,
    })
}

// ── Perplexity ──────────────────────────────────────────────────────────────

/// Sliding-window perplexity over a tokenized corpus, via [`rlx_eval`].
pub fn run_perplexity(
    model: &mut BenchModel,
    token_ids: &[u32],
    cfg: PerplexityConfig,
) -> Result<f64> {
    rlx_eval::perplexity(model, token_ids, cfg)
}
