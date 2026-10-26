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

//! TypeSafe-compatible `POST /v1/systemone` request / response shapes, mapped
//! onto the single pointer primitive.
//!
//! Mirrors `kev/api.py`:
//!
//! | type     | options fed to the model          | answer |
//! |----------|-----------------------------------|---|
//! | `noul`   | `[false-text, true-text]`         | `p(true)` |
//! | `choice` | `name` or `name: desc`, in order  | argmax + per-name probabilities |
//! | `score`  | ordered level descriptions        | expected level index + legend |
//!
//! Everything prompt-bearing flows through [`crate::json::Json`] rather than
//! `serde_json::Value`, because option order and object field order are part
//! of the prompt. See that module for why.

use anyhow::{Result, bail};
use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};

use crate::json::{Json, obj, render};

/// Upper bound on options per question (`kev/api.py::MAX_OPTIONS`).
pub const MAX_OPTIONS: usize = 255;

/// An insertion-ordered JSON object.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderedMap(pub Vec<(String, Json)>);

impl OrderedMap {
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

impl<'de> Deserialize<'de> for OrderedMap {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<OrderedMap, M::Error> {
                let mut out = Vec::with_capacity(m.size_hint().unwrap_or(0));
                while let Some((k, v)) = m.next_entry::<String, Json>()? {
                    out.push((k, v));
                }
                Ok(OrderedMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// One typed question.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no. `criteria` optionally describes the `true` / `false` sides.
    Noul {
        instructions: Json,
        #[serde(default)]
        criteria: Option<OrderedMap>,
    },
    /// Named options, each with a description or `null`.
    Choice {
        instructions: Json,
        criteria: OrderedMap,
    },
    /// Ordered levels, lowest first.
    Score {
        instructions: Json,
        criteria: Vec<Json>,
    },
}

impl Question {
    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }

    fn validate(&self, id: &str) -> Result<()> {
        match self {
            Question::Noul { .. } => Ok(()),
            Question::Choice { criteria, .. } => {
                if !(1..=MAX_OPTIONS).contains(&criteria.len()) {
                    bail!("question {id:?}: criteria must have 1..{MAX_OPTIONS} options");
                }
                Ok(())
            }
            Question::Score { criteria, .. } => {
                if !(2..=MAX_OPTIONS).contains(&criteria.len()) {
                    bail!("question {id:?}: score criteria must have 2..{MAX_OPTIONS} levels");
                }
                Ok(())
            }
        }
    }
}

/// Question ids in request order.
///
/// Deserialized straight from the document — never via an intermediate
/// `serde_json::Value`, which would sort the keys and silently reorder both
/// the questions and every option list inside them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Questions(pub Vec<(String, Question)>);

impl<'de> Deserialize<'de> for Questions {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Questions;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of question id to question")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Questions, M::Error> {
                let mut out = Vec::with_capacity(m.size_hint().unwrap_or(0));
                while let Some((k, v)) = m.next_entry::<String, Question>()? {
                    out.push((k, v));
                }
                Ok(Questions(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// `POST /v1/systemone` body.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneRequest {
    /// The content to evaluate: string, object or array.
    pub state: Json,
    #[serde(default = "default_model")]
    pub model: String,
    /// Question id → question. Ids are yours; the model never sees them.
    pub questions: Questions,
}

fn default_model() -> String {
    "kev-latest".into()
}

impl SystemOneRequest {
    /// Parse from JSON text.
    pub fn parse(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    fn checked(&self) -> Result<&[(String, Question)]> {
        if self.questions.0.is_empty() {
            bail!("questions must not be empty");
        }
        for (id, q) in &self.questions.0 {
            q.validate(id)?;
        }
        Ok(&self.questions.0)
    }
}

/// `name` alone, or `name: description` when a description is present.
///
/// An empty-string description counts as absent, exactly as in the Python.
pub fn option_text(name: &str, desc: Option<&Json>) -> String {
    match desc {
        None => name.to_string(),
        Some(d) if d.is_blank() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

/// The keys a question's probabilities are reported under, in option order.
pub fn question_keys(q: &Question) -> Vec<String> {
    match q {
        Question::Choice { criteria, .. } => criteria.keys().map(str::to_string).collect(),
        Question::Noul { .. } => vec!["false".into(), "true".into()],
        Question::Score { criteria, .. } => (0..criteria.len()).map(|i| i.to_string()).collect(),
    }
}

/// One question as the encoder sees it: instructions plus rendered options.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordQuestion {
    pub instr: String,
    pub options: Vec<String>,
    /// Ground-truth option index. `0` for inference; set for training.
    pub label: usize,
}

/// One request flattened into the internal record the encoder packs.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub state: String,
    pub questions: Vec<RecordQuestion>,
}

/// Per-question bookkeeping needed to map probabilities back to the response.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionMeta {
    pub id: String,
    pub qtype: String,
    pub keys: Vec<String>,
    /// `score` only: key → level description.
    pub legend: Option<Vec<(String, String)>>,
}

/// Flatten a request into `(record, per-question metadata)`.
pub fn to_record(req: &SystemOneRequest) -> Result<(Record, Vec<QuestionMeta>)> {
    let typed = req.checked()?;
    let mut qs = Vec::with_capacity(typed.len());
    let mut meta = Vec::with_capacity(typed.len());
    for (id, q) in typed {
        let keys = question_keys(q);
        let (instr, options, legend) = match q {
            Question::Noul {
                instructions,
                criteria,
            } => {
                let c = criteria.clone().unwrap_or_default();
                let opts = vec![
                    option_text("no", c.get("false")),
                    option_text("yes", c.get("true")),
                ];
                (render(instructions, 0), opts, None)
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                let opts = criteria
                    .0
                    .iter()
                    .map(|(k, v)| option_text(k, Some(v)))
                    .collect();
                (render(instructions, 0), opts, None)
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                let opts: Vec<String> = criteria.iter().map(|x| render(x, 0)).collect();
                let legend = keys.iter().cloned().zip(opts.iter().cloned()).collect();
                (render(instructions, 0), opts, Some(legend))
            }
        };
        qs.push(RecordQuestion {
            instr,
            options,
            label: 0,
        });
        meta.push(QuestionMeta {
            id: id.clone(),
            qtype: q.type_name().to_string(),
            keys,
            legend,
        });
    }
    Ok((
        Record {
            state: render(&req.state, 0),
            questions: qs,
        },
        meta,
    ))
}

/// `(p_max − 1/K) / (1 − 1/K)`; a single option is trivially confident.
pub fn choice_confidence(p: &[f32]) -> f32 {
    let k = p.len() as f32;
    if p.len() == 1 {
        return 1.0;
    }
    let pmax = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (pmax - 1.0 / k) / (1.0 - 1.0 / k)
}

/// `1 − E|level − mode| / (L − 1)`.
///
/// An approximation of TypeSafe's distance-from-the-modal-level statistic,
/// whose exact formula is unpublished — so this is the number kev reports,
/// not a claim about Jev.
pub fn score_confidence(p: &[f32]) -> f32 {
    let l = p.len();
    if l < 2 {
        return 1.0;
    }
    let mode = argmax(p);
    let e: f32 = p
        .iter()
        .enumerate()
        .map(|(i, pi)| pi * (i as f32 - mode as f32).abs())
        .sum();
    1.0 - e / (l - 1) as f32
}

/// First index of the maximum — ties go to the lowest index, as `max()` does
/// in the Python. Tie-breaking direction matters: an MLX `TopK` that broke
/// ties toward the *largest* index has bitten this codebase before.
pub fn argmax(p: &[f32]) -> usize {
    rlx_runtime::search::argmax(p)
}

/// Round to 2 decimals the way Python's `round(x, 2)` does, and return `f64`
/// so the value survives into JSON as `0.12` rather than `0.11999999731779099`.
///
/// Python rounds the *exact* binary value of the double, ties to even. The
/// obvious `(x * 100).round() / 100` disagrees on both halves of that:
/// `round(0.125, 2)` is `0.12`, not `0.13`, while for a double slightly above
/// a tie it rounds up. Rust's `{:.2}` formatting has exactly these semantics,
/// so we round through the decimal string.
///
/// The `f32 -> f64` widening matches the reference, which sees the same
/// widened values via `tensor.tolist()`.
pub fn r2(x: f32) -> f64 {
    let wide = x as f64;
    format!("{wide:.2}").parse::<f64>().unwrap_or(wide)
}

/// One answered question.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: Vec<(String, f64)>,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: Vec<(String, String)>,
        probabilities: Vec<(String, f64)>,
    },
}

impl Answer {
    /// The response-body JSON for this answer.
    pub fn to_json(&self) -> Json {
        let pairs = |p: &Vec<(String, f64)>| {
            Json::Object(
                p.iter()
                    .map(|(k, v)| (k.clone(), Json::Float(*v)))
                    .collect(),
            )
        };
        match self {
            Answer::Noul { noul } => {
                obj([("type", Json::from("noul")), ("noul", Json::Float(*noul))])
            }
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => obj([
                ("type", Json::from("choice")),
                ("choice", Json::from(choice.clone())),
                ("confidence", Json::Float(*confidence)),
                ("probabilities", pairs(probabilities)),
            ]),
            Answer::Score {
                score,
                confidence,
                legend,
                probabilities,
            } => obj([
                ("type", Json::from("score")),
                ("score", Json::Float(*score)),
                (
                    "legend",
                    Json::Object(
                        legend
                            .iter()
                            .map(|(k, v)| (k.clone(), Json::from(v.clone())))
                            .collect(),
                    ),
                ),
                ("probabilities", pairs(probabilities)),
                ("confidence", Json::Float(*confidence)),
            ]),
        }
    }
}

/// Turn per-question probability vectors into typed answers.
pub fn to_answers(probs: &[Vec<f32>], meta: &[QuestionMeta]) -> Result<Vec<(String, Answer)>> {
    if probs.len() != meta.len() {
        bail!(
            "got {} probability vectors for {} questions",
            probs.len(),
            meta.len()
        );
    }
    let mut out = Vec::with_capacity(meta.len());
    for (p, m) in probs.iter().zip(meta) {
        if p.len() != m.keys.len() {
            bail!(
                "question {:?}: {} probabilities for {} options",
                m.id,
                p.len(),
                m.keys.len()
            );
        }
        let answer = match m.qtype.as_str() {
            "noul" => Answer::Noul { noul: r2(p[1]) },
            "choice" => Answer::Choice {
                choice: m.keys[argmax(p)].clone(),
                confidence: r2(choice_confidence(p)),
                probabilities: m
                    .keys
                    .iter()
                    .cloned()
                    .zip(p.iter().map(|v| r2(*v)))
                    .collect(),
            },
            _ => {
                let score: f32 = p.iter().enumerate().map(|(i, pi)| i as f32 * pi).sum();
                Answer::Score {
                    score: r2(score),
                    confidence: r2(score_confidence(p)),
                    legend: m.legend.clone().unwrap_or_default(),
                    probabilities: m
                        .keys
                        .iter()
                        .cloned()
                        .zip(p.iter().map(|v| r2(*v)))
                        .collect(),
                }
            }
        };
        out.push((m.id.clone(), answer));
    }
    Ok(out)
}

/// The `answers` object of the response body, in request order.
pub fn answers_json(answers: &[(String, Answer)]) -> Json {
    Json::Object(
        answers
            .iter()
            .map(|(id, a)| (id.clone(), a.to_json()))
            .collect(),
    )
}
