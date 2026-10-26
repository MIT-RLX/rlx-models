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

//! The System One prompt and readout — a port of d1's `prompt.py`.
//!
//! A decision is one forward pass that stops at the answer slot: everything
//! the model sees is built here, and the answer is a softmax over its options'
//! tokens. Every string below is part of the contract with the checkpoint;
//! `tests/prompt_reference.rs` pins the rendered prompts against the
//! reference byte for byte.

use std::collections::HashSet;

use anyhow::{Result, anyhow, bail};
use rlx_core::ordered_json::Json;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";

/// Token ids for a piece of text, encoded without special-token wrapping
/// (`tokenizer.encode(text, add_special_tokens=False)`).
pub trait TextTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

/// One typed question in the Decision Index schema.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// Yes or no. `criteria` optionally defines each side (`true` / `false`).
    Noul {
        instructions: String,
        criteria: Option<Json>,
    },
    /// One of named options: `(name, description)` in the order given.
    Choice {
        instructions: String,
        criteria: Vec<(String, Json)>,
    },
    /// An ordered scale of 2 to 10 levels, lowest first.
    Score {
        instructions: String,
        criteria: Vec<Json>,
    },
}

impl Question {
    /// A question in the Decision Index's JSON (`{"type": "noul" | "choice" |
    /// "score", "instructions", "criteria"}`); `type` defaults to `choice`.
    pub fn from_json(q: &Json) -> Result<Self> {
        let instructions = match q.get("instructions") {
            Some(v) => v.python_str(),
            None => bail!("question has no `instructions`"),
        };
        let kind = q.get("type").and_then(Json::as_str).unwrap_or("choice");
        let criteria = q.get("criteria");
        Ok(match kind {
            "noul" => Question::Noul {
                instructions,
                criteria: criteria.filter(|c| !c.is_null()).cloned(),
            },
            "score" => Question::Score {
                instructions,
                criteria: criteria
                    .and_then(Json::as_array)
                    .ok_or_else(|| anyhow!("a score question needs a `criteria` list"))?
                    .to_vec(),
            },
            _ => Question::Choice {
                instructions,
                criteria: criteria
                    .and_then(Json::as_object)
                    .ok_or_else(|| anyhow!("a choice question needs a `criteria` object"))?
                    .to_vec(),
            },
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }
}

// --------------------------------------------------------------------------
// verbalizer
// --------------------------------------------------------------------------

/// Native letters when the labels already are letters, else `A..Z`, else
/// `00..`; one rule for every cardinality.
pub fn option_codes(labels: &[String]) -> Vec<String> {
    let labs: Vec<&str> = labels.iter().map(|s| s.trim()).collect();
    let letters = !labs.is_empty()
        && labs.iter().all(|k| {
            let mut c = k.chars();
            matches!((c.next(), c.next()), (Some(ch), None) if ch.is_alphabetic())
        });
    if letters {
        return labs.iter().map(|s| s.to_string()).collect();
    }
    if labs.len() <= 26 {
        return (0..labs.len())
            .map(|i| char::from(b'A' + i as u8).to_string())
            .collect();
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

/// Codes tried, in order, when a code is not one token or is taken.
fn fallback_pool() -> Vec<String> {
    let mut pool: Vec<String> = (b'A'..=b'Z').map(|c| char::from(c).to_string()).collect();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend((b'a'..=b'z').map(|c| char::from(c).to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    // Where a tokenizer splits digits, "00".."99" and "#i" are two tokens, and
    // two capital letters are often one. Last in the pool, so a tokenizer with
    // digit pairs never reaches it.
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            pool.push(format!("{}{}", char::from(a), char::from(b)));
        }
    }
    pool
}

/// Every label's distinct single-token code: `[(code, token_id)]`.
pub fn aliases(tok: &dyn TextTokenizer, labels: &[String]) -> Result<Vec<(String, u32)>> {
    let mut used: HashSet<u32> = HashSet::new();
    let mut out = Vec::with_capacity(labels.len());
    let mut pool: Option<Vec<String>> = None;
    for code in option_codes(labels) {
        let mut take = |raw: &str| -> Result<bool> {
            let enc = tok.encode(raw)?;
            if enc.len() != 1 || used.contains(&enc[0]) {
                return Ok(false);
            }
            used.insert(enc[0]);
            out.push((raw.to_string(), enc[0]));
            Ok(true)
        };
        if take(&code)? {
            continue;
        }
        let pool = pool.get_or_insert_with(fallback_pool);
        let mut found = false;
        for raw in pool.iter() {
            if take(raw)? {
                found = true;
                break;
            }
        }
        if !found {
            bail!("no single-token alias left for {} options", labels.len());
        }
    }
    Ok(out)
}

/// The single-token encodings among `texts`, deduplicated, in order.
fn single_ids(tok: &dyn TextTokenizer, texts: &[&str]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for t in texts {
        let enc = tok.encode(t)?;
        if enc.len() == 1 && !out.contains(&enc[0]) {
            out.push(enc[0]);
        }
    }
    Ok(out)
}

pub const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
pub const NO_FORMS: [&str; 3] = ["no", "No", "NO"];

/// Token ids to score, one group per option, max-pooled: the answer is a
/// softmax over these and nothing else.
pub fn readout_ids(tok: &dyn TextTokenizer, q: &Question) -> Result<Vec<Vec<u32>>> {
    match q {
        Question::Noul { .. } => {
            let yes = single_ids(tok, &YES_FORMS)?;
            let no = single_ids(tok, &NO_FORMS)?;
            if yes.is_empty() || no.is_empty() {
                bail!("tokenizer has no single-token yes/no");
            }
            Ok(vec![yes, no])
        }
        Question::Score { criteria, .. } => {
            let mut groups = Vec::with_capacity(criteria.len());
            for i in 0..criteria.len() {
                let g = single_ids(tok, &[&i.to_string()])?;
                if g.is_empty() {
                    bail!(
                        "score with {} levels needs single-token digits; the primitive is \
                         defined for 2 to 10",
                        criteria.len()
                    );
                }
                groups.push(g);
            }
            Ok(groups)
        }
        Question::Choice { criteria, .. } => {
            let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
            let mut groups = Vec::with_capacity(labels.len());
            for (code, tid) in aliases(tok, &labels)? {
                let mut g = vec![tid];
                g.extend(
                    single_ids(tok, &[&format!(" {code}")])?
                        .into_iter()
                        .filter(|&i| i != tid),
                );
                groups.push(g);
            }
            if groups.is_empty() {
                bail!("choice with no options");
            }
            Ok(groups)
        }
    }
}

/// Option probabilities from the scores at the answer slot.
///
/// `logit(id)` is the model's logit (or log-probability — the softmax over
/// options does not see the difference) for a vocabulary id. Each option
/// scores its best form.
pub fn readout(groups: &[Vec<u32>], logit: impl Fn(u32) -> f64) -> Vec<f64> {
    let scores: Vec<f64> = groups
        .iter()
        .map(|g| {
            g.iter()
                .map(|&i| logit(i))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

// --------------------------------------------------------------------------
// state and question rendering
// --------------------------------------------------------------------------

/// How a state is rendered. `JsonOnly`, the default, writes every non-string
/// state as the object it is; `Json` keeps three shortcuts (`Message:`,
/// `Passage:` / `Asked:`, a lone question's text); `Sections` writes nested
/// states as labelled blocks. `SectionsOnly` is `Sections` without the
/// shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StateStyle {
    #[default]
    JsonOnly,
    Json,
    Sections,
    SectionsOnly,
}

impl StateStyle {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "json_only" => Self::JsonOnly,
            "json" => Self::Json,
            "sections" => Self::Sections,
            "sections_only" => Self::SectionsOnly,
            _ => bail!("unknown state style `{s}` (json_only, json, sections, sections_only)"),
        })
    }

    fn only(self) -> bool {
        matches!(self, Self::JsonOnly | Self::SectionsOnly)
    }

    fn json(self) -> bool {
        matches!(self, Self::JsonOnly | Self::Json)
    }
}

/// The system turn: none by default, or the `isolated` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SystemTurn {
    #[default]
    None,
    Isolated,
}

impl SystemTurn {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "none" => Self::None,
            "isolated" => Self::Isolated,
            _ => bail!("unknown system turn `{s}` (none, isolated)"),
        })
    }

    fn text(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Isolated => Some(
                "You are a System One decision model. Answer only the current isolated \
                 question from the shared state. Other questions do not exist.",
            ),
        }
    }
}

/// How an option line reads: its description (`Desc`, the default), or
/// `name: description` when the name says more than a placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptionStyle {
    #[default]
    Desc,
    NameDesc,
}

/// Every rendering choice, with d1's defaults.
#[derive(Debug, Clone, Default)]
pub struct PromptStyle {
    /// Written first; `<|startoftext|>` for d1 (the tokenizer's BOS).
    pub bos: String,
    pub state: StateStyle,
    pub system: SystemTurn,
    pub options: OptionStyle,
    /// What sits between the assistant header and the answer slot: nothing
    /// on LFM2-VL, whose template opens no reasoning block.
    pub lead: String,
}

fn is_scalar(v: &Json) -> bool {
    match v {
        Json::Null => true,
        Json::Array(_) | Json::Object(_) => false,
        other => !other.python_str().contains('\n'),
    }
}

fn none_or_str(v: &Json) -> String {
    if v.is_null() {
        String::new()
    } else {
        v.python_str()
    }
}

/// Flatten a nested state into labelled blocks, keeping real newlines
/// (`_sections`). `json.dumps` escapes every newline inside a record; this
/// keeps a multi-line record readable.
fn sections(obj: &Json, path: &str, out: &mut Vec<String>) {
    let head = if path.is_empty() {
        String::new()
    } else {
        format!("[{path}]\n")
    };
    match obj {
        Json::Object(fields) => {
            let scalars: Vec<String> = fields
                .iter()
                .filter(|(_, v)| is_scalar(v))
                .map(|(k, v)| format!("{k}: {}", none_or_str(v)))
                .collect();
            if !scalars.is_empty() {
                out.push(format!("{head}{}", scalars.join("\n")));
            }
            for (k, v) in fields.iter().filter(|(_, v)| !is_scalar(v)) {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                sections(v, &p, out);
            }
        }
        Json::Array(items) => {
            if !items.is_empty() && items.iter().all(is_scalar) {
                let body: Vec<String> = items
                    .iter()
                    .map(|v| format!("- {}", none_or_str(v)))
                    .collect();
                out.push(format!("{head}{}", body.join("\n")));
                return;
            }
            let n = items.len();
            for (i, v) in items.iter().enumerate() {
                let p = if path.is_empty() {
                    format!("{}/{n}", i + 1)
                } else {
                    format!("{path} {}/{n}", i + 1)
                };
                sections(v, &p, out);
            }
        }
        scalar => out.push(format!("{head}{}", none_or_str(scalar))),
    }
}

fn render_state(state: &Json) -> String {
    if let Json::Str(s) = state {
        return s.clone();
    }
    let mut out = Vec::new();
    sections(state, "", &mut out);
    out.join("\n\n")
}

/// The block that precedes `QUESTION:`.
pub fn state_block(state: &Json, style: StateStyle) -> String {
    if let Json::Object(fields) = state
        && !style.only()
    {
        let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        let get = |k: &str| state.get(k).map(Json::python_str).unwrap_or_default();
        let has = |k: &str| keys.contains(&k);
        if keys == ["text"] {
            return format!("Message: {}\n\n", get("text"));
        }
        if keys.len() == 2 && has("passage") && has("question") {
            return format!(
                "Passage: {}\n\nAsked: {}\n\n",
                get("passage"),
                get("question")
            );
        }
        if keys == ["question"] {
            return format!("{}\n\n", get("question"));
        }
    }
    if style.json() {
        if let Json::Str(s) = state {
            return format!("{s}\n\n");
        }
        return format!("{}\n\n", state.to_python_json(Some(2)));
    }
    format!("{}\n\n", render_state(state))
}

fn is_placeholder(label: &str) -> bool {
    label
        .strip_prefix("opt")
        .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

fn option_line(code: &str, label: &str, desc: &Json, style: OptionStyle) -> String {
    let text = if desc.is_truthy() {
        desc.python_str()
    } else {
        label.replace('_', " ")
    };
    if style == OptionStyle::NameDesc && !is_placeholder(label) && label != text {
        return format!("{code} {label}: {text}");
    }
    format!("{code} {text}")
}

/// The question as the model reads it, ending in its reply instruction.
pub fn question_block(tok: &dyn TextTokenizer, q: &Question, style: OptionStyle) -> Result<String> {
    Ok(match q {
        Question::Choice {
            instructions,
            criteria,
        } => {
            let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
            let codes = aliases(tok, &labels)?;
            let lines: Vec<String> = criteria
                .iter()
                .zip(&codes)
                .map(|((label, desc), (code, _))| option_line(code, label, desc, style))
                .collect();
            format!(
                "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                lines.join("\n")
            )
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            let extra = match criteria {
                Some(c) if c.is_truthy() => {
                    let side = |k: &str| c.get(k).cloned().unwrap_or(Json::Null).python_str();
                    format!("\nYes: {}\nNo: {}", side("true"), side("false"))
                }
                _ => String::new(),
            };
            format!("{instructions}{extra}\n\nReply with yes or no only.")
        }
        Question::Score {
            instructions,
            criteria,
        } => {
            let legend: Vec<String> = criteria
                .iter()
                .enumerate()
                .map(|(i, name)| format!("{i} {}", name.python_str()))
                .collect();
            format!(
                "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                legend.join("\n"),
                criteria.len().saturating_sub(1)
            )
        }
    })
}

/// Everything before the question, shared by all questions on one state: the
/// pictures' markup at the head of the user turn, then the state. With no
/// state (`null`) the question follows the pictures directly.
pub fn prefix_text(state: &Json, style: &PromptStyle, images: &str) -> String {
    let turn = match style.system.text() {
        Some(t) => format!("{IM_START}system\n{t}{IM_END}\n"),
        None => String::new(),
    };
    let body = if state.is_null() {
        String::new()
    } else {
        format!("{}\nQUESTION:\n", state_block(state, style.state))
    };
    format!("{}{turn}{IM_START}user\n{images}{body}", style.bos)
}

/// The question and the assistant header, up to the answer slot.
pub fn suffix_text(tok: &dyn TextTokenizer, q: &Question, style: &PromptStyle) -> Result<String> {
    let body = question_block(tok, q, style.options)?;
    Ok(format!(
        "{body}{IM_END}\n{IM_START}assistant\n{}",
        style.lead
    ))
}

/// The whole prompt for one question over one state.
pub fn render(
    tok: &dyn TextTokenizer,
    state: &Json,
    q: &Question,
    style: &PromptStyle,
    images: &str,
) -> Result<String> {
    Ok(prefix_text(state, style, images) + &suffix_text(tok, q, style)?)
}

/// A question's typed answer from its option probabilities (d1's
/// `api.answer`): a noul's P(yes); a choice's pick, confidence and
/// probabilities; a score's expected level, confidence, probabilities and
/// legend.
pub fn answer(q: &Question, probs: &[f64]) -> Json {
    // Python's `max(range(n), key=...)` keeps the first maximum.
    let best = probs
        .iter()
        .enumerate()
        .fold(0, |b, (i, &p)| if p > probs[b] { i } else { b });
    match q {
        Question::Noul { .. } => Json::Object(vec![
            ("type".into(), "noul".into()),
            ("noul".into(), Json::Float(probs[0])),
        ]),
        Question::Choice { criteria, .. } => Json::Object(vec![
            ("type".into(), "choice".into()),
            ("choice".into(), Json::Str(criteria[best].0.clone())),
            ("confidence".into(), Json::Float(probs[best])),
            (
                "probabilities".into(),
                Json::Object(
                    criteria
                        .iter()
                        .zip(probs)
                        .map(|((n, _), &p)| (n.clone(), Json::Float(p)))
                        .collect(),
                ),
            ),
        ]),
        Question::Score { criteria, .. } => Json::Object(vec![
            ("type".into(), "score".into()),
            (
                "score".into(),
                Json::Float(probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()),
            ),
            ("confidence".into(), Json::Float(probs[best])),
            (
                "probabilities".into(),
                Json::Object(
                    probs
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (i.to_string(), Json::Float(p)))
                        .collect(),
                ),
            ),
            (
                "legend".into(),
                Json::Object(
                    criteria
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (i.to_string(), t.clone()))
                        .collect(),
                ),
            ),
        ]),
    }
}
