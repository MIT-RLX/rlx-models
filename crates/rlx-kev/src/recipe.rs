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

//! The released training recipe: frozen suites, and the per-epoch
//! augmentation that runs over them.
//!
//! `jaredpalmer/kev-4b` trained on `evals/v7/decision-v7` from
//! [`jaredpalmer/kev-suites`], with
//! `p_none = 0.1`, `p_none_distract = 0.12`, `p_distract = 0.15`,
//! `p_none_pair = 0.25`. Those are not incidental: a tenth of the
//! multiple-choice questions have their correct option *removed* and replaced
//! by a "none of the above", which is most of what teaches the model to say
//! "none of these" rather than picking the closest wrong option.
//!
//! Every draw goes through [`crate::pyrandom`], so the augmented dataset is
//! byte-identical to the reference's. `tests/recipe_parity.rs` checks that
//! against kev's own `augment()` over real suite records.
//!
//! [`jaredpalmer/kev-suites`]: https://huggingface.co/datasets/jaredpalmer/kev-suites

use anyhow::{Context, Result, bail};

use crate::api::{Question, Record, question_keys};
use crate::json::Json;
use crate::pyrandom::PyRandom;

/// `kev.data.NONE_OPTIONS`, in order — `choice` consumes an index into the
/// list *after* filtering by key, so the order is part of the stream.
pub const NONE_OPTIONS: [(&str, Option<&str>); 13] = [
    ("other", Some("None of the above")),
    ("other", Some("A reason that fits none of the above")),
    ("none", Some("None of these")),
    ("other", Some("Something else")),
    ("not_listed", Some("Not listed here")),
    ("none_of_the_above", None),
    ("other", Some("A category that fits none of the above")),
    ("other", Some("None of the listed options apply")),
    (
        "unknown",
        Some("Cannot be determined from the options given"),
    ),
    ("other", Some("Other")),
    ("none", None),
    ("other", Some("An answer not covered by the other options")),
    ("no_match", Some("No option matches")),
];

/// `kev.data.DISTRACTORS`. Python dict order = insertion order.
pub const DISTRACTORS: [(&str, &str); 4] = [
    ("weather", "Bad weather caused it"),
    ("purple", "The colour purple"),
    ("pancakes", "A recipe for pancakes"),
    ("taxes", "Unrelated: quarterly tax filing"),
];

/// Sources the v7 manifest marks eval-only; training on them is an error.
pub const EVAL_ONLY: [&str; 9] = [
    "mmlu",
    "emotion",
    "tweet_offensive",
    "qnli",
    "paws",
    "sciq",
    "contrastive",
    "legacy_holdout",
    "composition_holdout",
];

/// `kev.suite.SYNTHETIC_SOURCES`.
pub const SYNTHETIC_SOURCES: [&str; 3] = ["legacy_policy", "compositional", "contrastive"];

/// How a suite encodes the correct answer, which differs per question type.
#[derive(Debug, Clone, PartialEq)]
pub enum SuiteLabel {
    /// `choice`: the option **name**.
    Name(String),
    /// `noul`: a bool, mapping onto the `[no, yes]` option order.
    Bool(bool),
    /// `score`: the level index.
    Level(usize),
}

/// One question of a labelled suite record.
#[derive(Debug, Clone)]
pub struct LabelledQuestion {
    pub id: String,
    pub question: Question,
    pub label: SuiteLabel,
    pub src: String,
    /// Soft target keyed by option name — used by the "unknowable" records,
    /// which are trained toward a uniform answer rather than a single option.
    pub target: Option<Vec<(String, f64)>>,
}

/// One labelled suite record.
#[derive(Debug, Clone)]
pub struct LabelledRecord {
    pub state: Json,
    pub questions: Vec<LabelledQuestion>,
    pub source: String,
    pub id: String,
}

fn as_object(v: &Json, what: &str) -> Result<Vec<(String, Json)>> {
    v.as_object()
        .map(<[(String, Json)]>::to_vec)
        .ok_or_else(|| anyhow::anyhow!("{what} must be an object"))
}

fn get<'a>(fields: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Parse one line of a frozen suite's JSONL.
pub fn parse_suite_record(line: &str) -> Result<LabelledRecord> {
    let doc = Json::parse(line).context("parsing a suite record")?;
    let fields = as_object(&doc, "record")?;
    let state = get(&fields, "state")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("record has no `state`"))?;
    let meta = get(&fields, "_meta")
        .and_then(|m| m.as_object())
        .unwrap_or(&[]);
    let source = meta
        .iter()
        .find(|(k, _)| k == "source")
        .and_then(|(_, v)| v.as_str())
        .unwrap_or("custom")
        .to_string();
    let id = meta
        .iter()
        .find(|(k, _)| k == "id")
        .and_then(|(_, v)| v.as_str())
        .unwrap_or("")
        .to_string();

    let qs =
        get(&fields, "questions").ok_or_else(|| anyhow::anyhow!("record has no `questions`"))?;
    let qs = as_object(qs, "questions")?;
    let mut questions = Vec::with_capacity(qs.len());
    for (qid, qv) in &qs {
        let qf = as_object(qv, "question")?;
        // The request half is `type`/`instructions`/`criteria` only — the same
        // projection `kev.data.api_request` makes before validating.
        let req_only: Vec<(String, Json)> = qf
            .iter()
            .filter(|(k, _)| matches!(k.as_str(), "type" | "instructions" | "criteria"))
            .cloned()
            .collect();
        let question: Question = serde_json::from_str(
            &serde_json::to_string(&Json::Object(req_only)).expect("re-serialize"),
        )
        .with_context(|| format!("question {qid:?}"))?;

        let raw =
            get(&qf, "label").ok_or_else(|| anyhow::anyhow!("question {qid:?} has no `label`"))?;
        let label = match (&question, raw) {
            (Question::Choice { .. }, Json::Str(s)) => SuiteLabel::Name(s.clone()),
            (Question::Noul { .. }, Json::Bool(b)) => SuiteLabel::Bool(*b),
            (Question::Noul { .. }, Json::Int(i)) => SuiteLabel::Bool(*i != 0),
            (Question::Score { .. }, Json::Int(i)) => SuiteLabel::Level(*i as usize),
            (Question::Score { .. }, Json::UInt(u)) => SuiteLabel::Level(*u as usize),
            (q, other) => bail!(
                "question {qid:?} is `{}` but its label is {other:?}",
                q.type_name()
            ),
        };
        let target = get(&qf, "target")
            .filter(|t| !t.is_null())
            .and_then(|t| t.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| match v {
                        Json::Float(f) => Some((k.clone(), *f)),
                        Json::Int(i) => Some((k.clone(), *i as f64)),
                        Json::UInt(u) => Some((k.clone(), *u as f64)),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            });
        let src = get(&qf, "src")
            .and_then(|v| v.as_str())
            .unwrap_or(&source)
            .to_string();
        questions.push(LabelledQuestion {
            id: qid.clone(),
            question,
            label,
            src,
            target,
        });
    }
    Ok(LabelledRecord {
        state,
        questions,
        source,
        id,
    })
}

/// Read a suite split (`train.jsonl`, `development.jsonl`, …).
pub fn load_split(path: &std::path::Path) -> Result<Vec<LabelledRecord>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            parse_suite_record(line).with_context(|| format!("{}:{}", path.display(), i + 1))?,
        );
    }
    if out.is_empty() {
        bail!("{}: no records", path.display());
    }
    Ok(out)
}

/// `kev.data.augment` — permute option order always; sometimes swap the
/// correct option for a "none of the above", sometimes add one as a wrong
/// alternative, sometimes add an irrelevant distractor.
///
/// Draw order per `choice` question, which is what has to match: `random()`,
/// then at most one `choice(...)`, then `shuffle(keys)`. A soft-target
/// question is permuted only — inserting or removing options would change what
/// the target means.
pub fn augment(
    rec: &LabelledRecord,
    rng: &mut PyRandom,
    p_none: f64,
    p_none_distract: f64,
    p_distract: f64,
) -> Result<LabelledRecord> {
    if p_none < 0.0
        || p_none_distract < 0.0
        || p_distract < 0.0
        || p_none + p_none_distract + p_distract > 1.0
    {
        bail!("augmentation probabilities must be nonnegative and sum to at most one");
    }
    let mut out = rec.clone();
    for q in out.questions.iter_mut() {
        let Question::Choice {
            instructions,
            criteria,
        } = &q.question
        else {
            continue;
        };
        let mut crit = criteria.0.clone();

        if q.target.is_some() {
            let mut keys: Vec<String> = crit.iter().map(|(k, _)| k.clone()).collect();
            rng.shuffle(&mut keys);
            q.question = rebuild_choice(instructions.clone(), &crit, &keys);
            continue;
        }
        let SuiteLabel::Name(y) = q.label.clone() else {
            bail!("choice question {:?} has a non-name label", q.id);
        };

        let r = rng.random();
        let none_options: Vec<(&str, Option<&str>)> = NONE_OPTIONS
            .iter()
            .filter(|(k, _)| !crit.iter().any(|(ck, _)| ck == k))
            .copied()
            .collect();
        let distractors: Vec<(&str, &str)> = DISTRACTORS
            .iter()
            .filter(|(k, _)| !crit.iter().any(|(ck, _)| ck == k))
            .copied()
            .collect();

        let mut label = y.clone();
        if crit.len() > 2 && r < p_none && !none_options.is_empty() {
            let (nk, nd) = none_options[rng.choice_index(none_options.len())];
            crit.retain(|(k, _)| k != &y); // dict.pop(y)
            crit.push((nk.to_string(), none_value(nd)));
            label = nk.to_string();
        } else if p_none <= r
            && r < p_none + p_none_distract
            && crit.len() < 255
            && !none_options.is_empty()
        {
            let (nk, nd) = none_options[rng.choice_index(none_options.len())];
            crit.push((nk.to_string(), none_value(nd)));
        } else if p_none + p_none_distract <= r
            && r < p_none + p_none_distract + p_distract
            && crit.len() < 255
            && !distractors.is_empty()
        {
            let (k, v) = distractors[rng.choice_index(distractors.len())];
            crit.push((k.to_string(), Json::Str(v.to_string())));
        }

        let mut keys: Vec<String> = crit.iter().map(|(k, _)| k.clone()).collect();
        rng.shuffle(&mut keys);
        q.question = rebuild_choice(instructions.clone(), &crit, &keys);
        q.label = SuiteLabel::Name(label);
    }
    Ok(out)
}

fn none_value(v: Option<&str>) -> Json {
    v.map_or(Json::Null, |s| Json::Str(s.to_string()))
}

fn rebuild_choice(instructions: Json, crit: &[(String, Json)], keys: &[String]) -> Question {
    let ordered: Vec<(String, Json)> = keys
        .iter()
        .map(|k| {
            let v = crit
                .iter()
                .find(|(ck, _)| ck == k)
                .map(|(_, v)| v.clone())
                .unwrap_or(Json::Null);
            (k.clone(), v)
        })
        .collect();
    Question::Choice {
        instructions,
        criteria: crate::api::OrderedMap(ordered),
    }
}

/// `kev.data.none_pair` — the minimal pair for the none-of-the-above
/// shortcut: the same question twice, once with the correct option present (a
/// none option is then wrong) and once with it removed (the same none option
/// is then right). Option order is identical between the two, so the only
/// usable difference is whether the evidence matches an option.
///
/// Returns an empty vector when nothing is eligible (needs a `choice` with
/// ≥3 options and no soft target).
pub fn none_pair(rec: &LabelledRecord, rng: &mut PyRandom) -> Result<Vec<LabelledRecord>> {
    let eligible: Vec<usize> = rec
        .questions
        .iter()
        .enumerate()
        .filter(|(_, q)| {
            matches!(&q.question, Question::Choice { criteria, .. } if criteria.len() >= 3)
                && q.target.is_none()
        })
        .map(|(i, _)| i)
        .collect();
    if eligible.is_empty() {
        return Ok(Vec::new());
    }
    let pick = eligible[rng.choice_index(eligible.len())];
    let q = &rec.questions[pick];
    let Question::Choice {
        instructions,
        criteria,
    } = &q.question
    else {
        unreachable!("filtered to choice");
    };
    let candidates: Vec<(&str, Option<&str>)> = NONE_OPTIONS
        .iter()
        .filter(|(k, _)| !criteria.0.iter().any(|(ck, _)| ck == k))
        .copied()
        .collect();
    let (nk, nd) = if candidates.is_empty() {
        ("none_of_these", None)
    } else {
        candidates[rng.choice_index(candidates.len())]
    };

    let mut keys: Vec<String> = criteria.0.iter().map(|(k, _)| k.clone()).collect();
    keys.push(nk.to_string());
    rng.shuffle(&mut keys);

    let mut with_none = criteria.0.clone();
    with_none.push((nk.to_string(), none_value(nd)));
    let present = rebuild_choice(instructions.clone(), &with_none, &keys);

    let SuiteLabel::Name(y) = q.label.clone() else {
        bail!("choice question {:?} has a non-name label", q.id);
    };
    let absent_keys: Vec<String> = keys.iter().filter(|k| **k != y).cloned().collect();
    let absent = rebuild_choice(instructions.clone(), &with_none, &absent_keys);

    let one = |question: Question, label: SuiteLabel| LabelledRecord {
        state: rec.state.clone(),
        questions: vec![LabelledQuestion {
            id: q.id.clone(),
            question,
            label,
            src: q.src.clone(),
            target: None,
        }],
        source: rec.source.clone(),
        id: rec.id.clone(),
    };
    Ok(vec![
        one(present, SuiteLabel::Name(y)),
        one(absent, SuiteLabel::Name(nk.to_string())),
    ])
}

/// Per-question soft target, aligned to the option order.
#[derive(Debug, Clone)]
pub struct Materialized {
    pub record: Record,
    /// `Some(dist)` for a soft-target question, normalized over its options.
    pub targets: Vec<Option<Vec<f32>>>,
}

/// `kev.data.materialize` — a labelled request becomes an internal record with
/// integer labels, via the *serving* path so training and inference render
/// identically.
pub fn materialize(rec: &LabelledRecord) -> Result<Materialized> {
    let req = crate::api::SystemOneRequest {
        state: rec.state.clone(),
        model: "kev-latest".into(),
        questions: crate::api::Questions(
            rec.questions
                .iter()
                .map(|q| (q.id.clone(), q.question.clone()))
                .collect(),
        ),
    };
    let (mut record, meta) = crate::api::to_record(&req)?;
    let mut targets = Vec::with_capacity(rec.questions.len());
    for (i, q) in rec.questions.iter().enumerate() {
        let keys = question_keys(&q.question);
        let label = match &q.label {
            SuiteLabel::Name(n) => keys
                .iter()
                .position(|k| k == n)
                .ok_or_else(|| anyhow::anyhow!("label {n:?} is not an option of {:?}", q.id))?,
            SuiteLabel::Bool(b) => usize::from(*b),
            SuiteLabel::Level(l) => *l,
        };
        if label >= keys.len() {
            bail!(
                "label {label} out of range for {:?} ({} options)",
                q.id,
                keys.len()
            );
        }
        record.questions[i].label = label;
        let _ = &meta[i];
        targets.push(match &q.target {
            None => None,
            Some(t) => {
                let raw: Vec<f64> = keys
                    .iter()
                    .map(|k| t.iter().find(|(tk, _)| tk == k).map_or(0.0, |(_, v)| *v))
                    .collect();
                let sum: f64 = raw.iter().sum();
                if sum <= 0.0 {
                    bail!("target for {:?} puts no mass on any option", q.id);
                }
                Some(raw.iter().map(|x| (x / sum) as f32).collect())
            }
        });
    }
    Ok(Materialized { record, targets })
}

/// Refuse a training set that contains an eval-only source.
pub fn check_trainable(records: &[LabelledRecord], extra_eval_only: &[String]) -> Result<()> {
    let mut bad: Vec<&str> = records
        .iter()
        .map(|r| r.source.as_str())
        .filter(|s| EVAL_ONLY.contains(s) || extra_eval_only.iter().any(|e| e == s))
        .collect();
    bad.sort_unstable();
    bad.dedup();
    if !bad.is_empty() {
        bail!("training data contains eval-only sources: {bad:?}");
    }
    Ok(())
}

/// Whether a source is one of the synthetic policy pools (`--synthetic_repeat`).
pub fn is_synthetic(source: &str) -> bool {
    SYNTHETIC_SOURCES.contains(&source)
}
