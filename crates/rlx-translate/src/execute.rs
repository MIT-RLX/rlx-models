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

//! Runs a pair's stage graph instead of hand-wiring the stages.
//!
//! [`crate::pipeline::TranslationPlan`] already resolves the config's DAG in
//! dependency order; this walks it. That matters because the order and the
//! fan-in are the OS's, read from the shipped JSON, rather than a guess encoded
//! in a `match` somewhere — and because new blocks now have an obvious home.
//!
//! # What each block does here
//!
//! A stage evaluates to `Option<String>`: the text it produced, or `None` for a
//! branch that did not fire. That is enough to express the graph's control
//! flow, because [`BlockKind::Merger`] is a **coalesce** — every one of the 752
//! mergers in the shipped configs carries `merge-style: "any"`, so it means
//! "whichever input has a value". `Select` and `Null` are the same shape.
//!
//! | block | here |
//! | --- | --- |
//! | `PhraseBook` | exact lookup; `None` on a miss |
//! | `PDecTranslator` | the NMT, via the caller's closure |
//! | `Merger` / `Select` | first input that produced text |
//! | `Null` | `None` |
//! | `CaseMap` | sentence-cases its input |
//! | `DoNotTranslate` | passes through, recording spans that went missing |
//! | `QualityEstimator` | passes through, recording flags |
//! | everything else | pass-through |
//!
//! # What this is not
//!
//! the shipped graph passes *token arrays* between the SentencePiece, alignment and
//! merge stages; this passes text. So `SentencePiece` encode/decode is a
//! no-op here rather than a real round trip, and `AlignmentProcessor`,
//! `PDecForceAlign` and `LinkAlternatives` pass through. Those stages exist to
//! project spans between source and target, which needs the token-level graph;
//! running them at text level would be pretending. The stages that decide *what
//! the output says* — phrasebook, NMT, merge, casing — are real.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::pipeline::{StageKind, TranslationPlan};
use crate::quasar::{BlockKind, SpmAction};
use crate::spm::Vocab;

/// What flows along an edge.
///
/// the shipped graph carries *token arrays* between the SentencePiece, alignment
/// and merge stages, not text — which is why `spm_encode` and `spm_decode` are
/// separate stages at all. Modelling that lets those stages do real work
/// instead of being no-ops, and is the precondition for the alignment blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Val {
    Text(String),
    Tokens(Vec<u32>),
}

impl Val {
    /// Surface text, decoding tokens when needed.
    pub fn text(&self, vocab: Option<&Vocab>) -> String {
        match self {
            Self::Text(t) => t.clone(),
            Self::Tokens(ids) => match vocab {
                // `Vocab::decode`, not a textual join of pieces: byte
                // fallback has to be reassembled or the `<0xNN>` markers reach
                // the translator as literal text.
                Some(v) => v.decode(ids),
                // Without a vocabulary the ids cannot be rendered; an empty
                // string is honest, and the stage that needed text will simply
                // produce nothing rather than something invented.
                None => String::new(),
            },
        }
    }
}

/// Everything the graph needs from outside.
pub struct Context<'a> {
    /// Exact-match lookup, already normalized. `None` disables the stage.
    pub phrasebook: Option<&'a dyn Fn(&str) -> Option<String>>,
    /// The NMT, given *the stage's own* decode parameters.
    ///
    /// A pivot pair has several translator stages across two models — the
    /// config for `ar_AE-de_DE` carries eight, routing ar->en->de — so the
    /// parameters cannot be hoisted out of the loop. `None` leaves
    /// `PDecTranslator` producing nothing, which the mergers coalesce past.
    pub nmt: Option<&'a dyn Fn(&crate::pdec::PDecParams, &str) -> Result<String>>,
    /// Locale for `CaseMap`.
    pub case_locale: Option<String>,
    /// Target locale, for output punctuation.
    pub target: String,
    /// Source locale, for rules that depend on the writing system either side
    /// uses. `None` disables them.
    pub source: Option<String>,
    /// Quality checks for the `QualityEstimator` stage.
    pub quality: Option<&'a crate::quality::QualityEstimator>,
    /// Vocabulary for the `SentencePiece` stages. Without it those stages pass
    /// their input through, which is what this executor used to do everywhere.
    pub vocab: Option<&'a Vocab>,
}

/// What running the graph produced.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Final text, if the graph reached one.
    pub text: Option<String>,
    /// Name of the stage the output came from.
    pub source_stage: Option<String>,
    /// Quality flags raised along the way.
    pub flags: Vec<crate::quality::Flag>,
    /// Do-not-translate spans that did not survive.
    pub lost_spans: Vec<String>,
    /// Every stage and what it produced, in order.
    pub trace: Vec<(String, Option<String>)>,
}

/// Whether this module runs a block, as opposed to passing its input through.
///
/// Lives here, beside the match it describes, because it did not: the same
/// classification in `pipeline` went stale as the executor grew, and
/// `rlx-translate plan` spent a while reporting implemented stages as `todo`.
/// A new arm below wants a line here.
pub fn handles(kind: &BlockKind) -> bool {
    match kind {
        BlockKind::PhraseBook
        | BlockKind::PDecTranslator
        | BlockKind::Null
        | BlockKind::CaseMap
        | BlockKind::DoNotTranslate
        | BlockKind::QualityEstimator
        | BlockKind::SentencePiece
        // Merger is a coalesce and Select/Tokenizer are plumbing; all three are
        // handled by the fall-through, which is their implementation.
        | BlockKind::Merger
        | BlockKind::Select
        | BlockKind::Tokenizer => true,
        BlockKind::PDecForceAlign
        | BlockKind::SimpleTokenizer
        | BlockKind::Segmentation
        | BlockKind::StructuredPrediction
        | BlockKind::AlignmentProcessor
        | BlockKind::AmbiguityAnnotator
        | BlockKind::Romanizer
        | BlockKind::Filter
        | BlockKind::LinkAlternatives
        | BlockKind::Other(_) => false,
    }
}

/// Runs `plan` over `input`.
///
/// `input` should already be normalized — the sed-script normalizer is a
/// pipeline *input*, not one of the graph's stages.
/// Whether there is anything here a translator could act on.
///
/// The shipped framework refuses empty input with `nothingToTranslate` and
/// returns anything else that is not language — `"   "`, `"🎉🎉🎉"`, `"123"`,
/// `"!!!"` — unchanged. This port instead fed them to the NMT, which
/// hallucinated: empty input came back as `'` and `🎉🎉🎉` as a single `🎉`.
pub fn has_translatable_content(s: &str) -> bool {
    s.chars().any(char::is_alphabetic)
}

/// Splits `input` into the paragraphs the framework would translate.
///
/// Read off `TREdge`, which asks the shipped framework rather than guessing:
///
/// - A run of line breaks — `\n`, `\r`, `\r\n`, in any number — is **one**
///   paragraph boundary. `"a\nb"` and `"a\n\n\nb"` both come back as
///   `"A\n\nB"`, and empty parts are dropped, so `"line\n"` is just `"Ligne"`.
/// - Every other whitespace is a space *within* a line, including tab,
///   zero-width space and U+2028: `"a\tb"` and `"a\u{200b}b"` both give
///   `"A b"`. Left alone they reached the model intact and it answered `"a\tb"`
///   with `"A A"`.
fn paragraphs(input: &str) -> Vec<String> {
    input
        .split(['\n', '\r'])
        .map(|line| {
            // Anything else that is whitespace becomes one, and runs collapse.
            let spaced: String = line
                .chars()
                .map(|c| {
                    if c.is_whitespace() || c == '\u{200b}' {
                        ' '
                    } else {
                        c
                    }
                })
                .collect();
            spaced.split_whitespace().collect::<Vec<_>>().join(" ")
        })
        .filter(|p| !p.is_empty())
        .collect()
}

/// Runs `plan` over `input`, one paragraph at a time.
///
/// The framework treats a line break as a paragraph boundary and joins the
/// results with a blank line. Passing the whole string through as one gave
/// `"Une"` for `"a\nb"` — every line after the first silently lost.
pub fn run(plan: &TranslationPlan, ctx: &Context<'_>, input: &str) -> Result<Outcome> {
    let parts = paragraphs(input);
    // One paragraph and nothing rewritten: the common path, unchanged.
    if parts.len() == 1 && parts[0] == input {
        return run_one(plan, ctx, input);
    }
    if parts.is_empty() {
        // Whitespace only. The framework returns it as given.
        return Ok(Outcome {
            text: Some(input.to_string()),
            ..Outcome::default()
        });
    }
    let mut done: Vec<String> = Vec::with_capacity(parts.len());
    let mut out = Outcome::default();
    for part in &parts {
        let o = run_one(plan, ctx, part)?;
        done.push(o.text.clone().unwrap_or_else(|| part.clone()));
        out.flags.extend(o.flags);
        out.lost_spans.extend(o.lost_spans);
        out.trace.extend(o.trace);
        if out.source_stage.is_none() {
            out.source_stage = o.source_stage;
        }
    }
    out.text = Some(done.join("\n\n"));
    Ok(out)
}

fn run_one(plan: &TranslationPlan, ctx: &Context<'_>, input: &str) -> Result<Outcome> {
    if !has_translatable_content(input) {
        // Unchanged, as the framework does. An empty input is the caller's to
        // reject — the framework raises `nothingToTranslate` — but returning it
        // unchanged is the same string either way.
        return Ok(Outcome {
            text: Some(input.to_string()),
            ..Outcome::default()
        });
    }
    // A stage can expose several results. `None` is its default; a named port
    // is what a downstream `stage:port` reference reads. The phrasebook needs
    // this: `pb:out` is its hit and `pb:final` is the text to carry on with, so
    // the NMT sitting downstream is not starved when the lookup misses.
    let mut values: BTreeMap<(String, Option<String>), Option<Val>> = BTreeMap::new();
    values.insert(
        ("graph-input".to_string(), None),
        Some(Val::Text(input.to_string())),
    );
    let mut out = Outcome::default();

    for stage in &plan.stages {
        let read =
            |vals: &BTreeMap<(String, Option<String>), Option<Val>>, i: usize| -> Option<Val> {
                let name = stage.inputs.get(i)?;
                let port = stage.ports.get(i).cloned().flatten();
                match vals.get(&(name.clone(), port)) {
                    // The port exists: `None` means that branch genuinely produced
                    // nothing, and must not silently fall back to the stage's
                    // pass-through — that is how an untranslated source sneaks past
                    // a merge that was supposed to prefer the NMT.
                    Some(v) => v.clone(),
                    // Not published separately: read the stage's default result.
                    None => vals.get(&(name.clone(), None)).cloned().flatten(),
                }
            };
        // First upstream input that produced text. This is `merge-style: any`,
        // and it is also the sensible reading for every single-input stage.
        let first = |vals: &BTreeMap<(String, Option<String>), Option<Val>>| -> Option<Val> {
            (0..stage.inputs.len()).find_map(|i| read(vals, i))
        };
        // Most blocks want surface text regardless of what the edge carries.
        let first_text =
            |vals: &BTreeMap<(String, Option<String>), Option<Val>>| -> Option<String> {
                first(vals).map(|v| v.text(ctx.vocab))
            };

        let produced: Option<Val> = match &stage.kind {
            StageKind::Output => {
                let v = first(&values);
                out.source_stage = stage.inputs.first().cloned();
                v
            }

            StageKind::Block(kind) => match kind {
                BlockKind::PhraseBook => {
                    let src = first_text(&values);
                    let hit = match (ctx.phrasebook, &src) {
                        (Some(f), Some(s)) => f(s).map(Val::Text),
                        _ => None,
                    };
                    let src = src.map(Val::Text);
                    // The *default* result is the text carried onwards — the
                    // NMT chain hangs off it (`pb_feature <- ["pb"]`), so a
                    // miss must not starve it. `:final` and `:out` are the hit
                    // alone, which is what the merges read; if they carried the
                    // pass-through too, a merge would prefer the untranslated
                    // source over the NMT's answer.
                    values.insert((stage.name.clone(), Some("out".to_string())), hit.clone());
                    values.insert((stage.name.clone(), Some("final".to_string())), hit.clone());
                    src
                }
                BlockKind::PDecTranslator => {
                    match (&ctx.nmt, stage.pdec.as_ref(), first_text(&values)) {
                        (Some(f), Some(params), Some(src)) => Some(Val::Text(f(params, &src)?)),
                        _ => None,
                    }
                }
                BlockKind::Null => None,
                BlockKind::CaseMap => first_text(&values).map(|t| {
                    Val::Text(match &ctx.case_locale {
                        Some(loc) => crate::casemap::sentence_case(&t, loc),
                        None => t,
                    })
                }),
                BlockKind::DoNotTranslate => {
                    // This block names its inputs: `{"target": ..., "source":
                    // ...}`. Taking the first would read the *source* back out
                    // — the roles are sorted, and "source" precedes "target" —
                    // which returns the input untranslated.
                    let target = stage
                        .roles
                        .iter()
                        .position(|r| r == "target")
                        .and_then(|i| read(&values, i))
                        .or_else(|| first(&values));
                    match target.as_ref().map(|v| v.text(ctx.vocab)) {
                        Some(t) => {
                            // Put back any run of a script neither language
                            // writes. The model alters rather than drops such
                            // text, so this is repairing corruption, not
                            // declining to translate.
                            let repaired = crate::dnt::restore_foreign_scripts(
                                input,
                                &t,
                                ctx.source.as_deref().unwrap_or(""),
                                &ctx.target,
                            );
                            out.lost_spans.extend(
                                crate::dnt::lost(input, &repaired)
                                    .into_iter()
                                    .map(|s| s.text.to_string()),
                            );
                            Some(Val::Text(repaired))
                        }
                        None => target,
                    }
                }
                BlockKind::QualityEstimator => {
                    let t = first(&values);
                    if let (Some(qe), Some(v)) = (ctx.quality, &t) {
                        out.flags.extend(qe.check(input, &v.text(ctx.vocab)));
                    }
                    t
                }
                // The one block whose direction comes from an attribute rather
                // than its type. With no vocabulary it passes through, which is
                // what every stage here used to do.
                BlockKind::SentencePiece => match (stage.spm, ctx.vocab) {
                    (Some(SpmAction::Encode), Some(v)) => {
                        first_text(&values).map(|t| Val::Tokens(v.encode(&t)))
                    }
                    (Some(SpmAction::Decode), Some(_)) => first_text(&values).map(Val::Text),
                    _ => first(&values),
                },
                // Merger, Select and the rest are pass-through / coalesce.
                _ => first(&values),
            },
        };

        values.insert((stage.name.clone(), None), produced.clone());
        out.trace
            .push((stage.name.clone(), produced.map(|v| v.text(ctx.vocab))));
    }

    // Take the graph's own output sentinel, not whatever sorted last. Some
    // graphs place another stage after it in dependency order — the zh/ja/ko
    // and id/th/vi bundles have a `romanizer` there — and reading the last
    // stage then returns `None` for a translation the graph did produce.
    out.text = plan
        .stages
        .iter()
        .rev()
        .find(|s| s.kind == StageKind::Output)
        .or_else(|| plan.stages.last())
        .and_then(|s| values.get(&(s.name.clone(), None)).cloned().flatten())
        .map(|v| crate::postproc::normalize_punctuation_for(&v.text(ctx.vocab), &ctx.target));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Stage, Support};

    fn stage(name: &str, kind: BlockKind, inputs: &[&str]) -> Stage {
        // A translator stage carries its own parameters; without them it cannot
        // know which direction it is translating, which is the whole point of
        // per-stage params on a pivot graph.
        let pdec = (kind == BlockKind::PDecTranslator).then(crate::pdec::PDecParams::default);
        Stage {
            name: name.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            roles: inputs.iter().map(|_| "in".to_string()).collect(),
            ports: inputs.iter().map(|_| None).collect(),
            kind: StageKind::Block(kind),
            block: None,
            files: Vec::new(),
            spm: None,
            pdec,
            support: Support::Ready,
        }
    }

    fn plan(stages: Vec<Stage>) -> TranslationPlan {
        TranslationPlan {
            task: "mt_app".into(),
            pair: crate::quasar::LangPair::parse("en_US-fr_FR").expect("pair"),
            stages,
            pdec: None,
        }
    }

    /// A phrasebook hit short-circuits the NMT, and the merger coalesces to it.
    /// The framework's own answers, from `TREdge` against `en_US-fr_FR`.
    #[test]
    fn a_line_break_is_a_paragraph_boundary() {
        let p = plan(vec![
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "out".into(),
                inputs: vec!["pdec".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let pb = |_: &str| None;
        // Uppercase stands in for translating: it shows which lines were seen.
        let nmt = |_: &crate::pdec::PDecParams, t: &str| Ok(t.to_uppercase());
        let ctx = Context {
            phrasebook: Some(&pb),
            nmt: Some(&nmt),
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        // The shipped framework renders "a\nb" as "A\n\nB": each line is its
        // own translation and they are joined by a blank line. Passing the whole
        // string through instead lost every line after the first.
        assert_eq!(
            run(&p, &ctx, "a\nb").expect("run").text.as_deref(),
            Some("A\n\nB")
        );
        assert_eq!(
            run(&p, &ctx, "one\ntwo\nthree")
                .expect("run")
                .text
                .as_deref(),
            Some("ONE\n\nTWO\n\nTHREE")
        );
        // Blank lines collapse: the framework renders "a\n\n\nb" as "A\n\nB",
        // one boundary however many breaks. This asserted the opposite, having
        // been written before the framework was asked.
        assert_eq!(
            run(&p, &ctx, "a\n\nb").expect("run").text.as_deref(),
            Some("A\n\nB")
        );
        assert_eq!(
            run(&p, &ctx, "a\n\n\nb").expect("run").text.as_deref(),
            Some("A\n\nB")
        );
        // Tab and zero-width space are spaces within a line, not boundaries;
        // a carriage return is a boundary.
        assert_eq!(
            run(&p, &ctx, "a\tb").expect("run").text.as_deref(),
            Some("A B")
        );
        assert_eq!(
            run(&p, &ctx, "a\u{200b}b").expect("run").text.as_deref(),
            Some("A B")
        );
        assert_eq!(
            run(&p, &ctx, "a\rb").expect("run").text.as_deref(),
            Some("A\n\nB")
        );
    }

    #[test]
    fn input_with_nothing_to_translate_is_returned_unchanged() {
        let p = plan(vec![
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "out".into(),
                inputs: vec!["pdec".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let pb = |_: &str| None;
        // Anything reaching the model here is a failure of the guard.
        let nmt = |_: &crate::pdec::PDecParams, _: &str| Ok("TRANSLATED".to_string());
        let ctx = Context {
            phrasebook: Some(&pb),
            nmt: Some(&nmt),
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        // The framework returns each of these unchanged; this port used to hand
        // them to the NMT, which answered "" with `'` and "🎉🎉🎉" with one `🎉`.
        for t in ["", "   ", "🎉🎉🎉", "123", "!!!", "...", "42 + 1 = 43"] {
            assert_eq!(run(&p, &ctx, t).expect("run").text.as_deref(), Some(t));
        }
        // And still translates anything with a letter in it.
        assert_eq!(
            run(&p, &ctx, "a1").expect("run").text.as_deref(),
            Some("TRANSLATED")
        );
    }

    #[test]
    fn a_phrasebook_hit_wins_the_merge() {
        let p = plan(vec![
            stage("pb", BlockKind::PhraseBook, &["graph-input"]),
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "merge".into(),
                inputs: vec!["pb".into(), "pdec".into()],
                roles: vec!["in1".into(), "in2".into()],
                // The merge reads the phrasebook's *hit* port, as the shipped
                // graph does; its default port is the pass-through.
                ports: vec![Some("final".into()), None],
                kind: StageKind::Block(BlockKind::Merger),
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
            Stage {
                name: "out".into(),
                inputs: vec!["merge".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let pb = |s: &str| (s == "platypus").then(|| "ornithorynque".to_string());
        let nmt = |_: &crate::pdec::PDecParams, _: &str| Ok("ornitharyque".to_string());
        let ctx = Context {
            phrasebook: Some(&pb),
            nmt: Some(&nmt),
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        let o = run(&p, &ctx, "platypus").expect("run");
        assert_eq!(o.text.as_deref(), Some("ornithorynque"));
    }

    /// On a miss the same graph falls through to the NMT — no special casing.
    #[test]
    fn a_phrasebook_miss_falls_through_to_the_nmt() {
        let p = plan(vec![
            stage("pb", BlockKind::PhraseBook, &["graph-input"]),
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "merge".into(),
                inputs: vec!["pb".into(), "pdec".into()],
                roles: vec!["in1".into(), "in2".into()],
                // The merge reads the phrasebook's *hit* port, as the shipped
                // graph does; its default port is the pass-through.
                ports: vec![Some("final".into()), None],
                kind: StageKind::Block(BlockKind::Merger),
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
            Stage {
                name: "out".into(),
                inputs: vec!["merge".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let pb = |_: &str| None;
        let nmt = |_: &crate::pdec::PDecParams, _: &str| Ok("un ornithorynque nage".to_string());
        let ctx = Context {
            phrasebook: Some(&pb),
            nmt: Some(&nmt),
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        let o = run(&p, &ctx, "a platypus swims").expect("run");
        assert_eq!(o.text.as_deref(), Some("un ornithorynque nage"));
    }

    /// The do-not-translate stage sees the original source, not its input.
    #[test]
    fn a_lost_identifier_is_reported_by_the_graph() {
        let p = plan(vec![
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            stage("dnt", BlockKind::DoNotTranslate, &["pdec"]),
            Stage {
                name: "out".into(),
                inputs: vec!["dnt".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let nmt =
            |_: &crate::pdec::PDecParams, _: &str| Ok("Le fichier est LISEZMOI.md".to_string());
        let ctx = Context {
            phrasebook: None,
            nmt: Some(&nmt),
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        let o = run(&p, &ctx, "the file is README.md").expect("run");
        assert_eq!(o.lost_spans, vec!["README.md".to_string()]);
    }

    /// The result comes from the graph's output sentinel, not from whatever
    /// sorted last.
    ///
    /// The `zh/ja/ko` and `id/th/vi` bundles place a `romanizer` after
    /// `graph-output` in dependency order. Reading the last stage returned
    /// `None` for translations the graph had produced, and silently failed 21
    /// of 397 directions.
    #[test]
    fn a_stage_after_the_output_does_not_hide_the_result() {
        let p = plan(vec![
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "out".into(),
                inputs: vec!["pdec".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
            // Runs after the sentinel and produces nothing.
            stage("romanizer", BlockKind::Romanizer, &["missing"]),
        ]);
        let nmt = |_: &crate::pdec::PDecParams, _: &str| Ok("translated".to_string());
        let ctx = Context {
            phrasebook: None,
            nmt: Some(&nmt),
            case_locale: None,
            target: "ja_JP".into(),
            quality: None,
            vocab: None,
            source: Some("en_US".into()),
        };
        let o = run(&p, &ctx, "anything").expect("run");
        assert_eq!(o.text.as_deref(), Some("translated"));
    }

    /// With no NMT the graph still runs; the branch simply produces nothing.
    #[test]
    fn a_missing_executor_leaves_an_empty_branch_rather_than_failing() {
        let p = plan(vec![
            stage("pdec", BlockKind::PDecTranslator, &["graph-input"]),
            Stage {
                name: "out".into(),
                inputs: vec!["pdec".into()],
                roles: vec!["in".into()],
                ports: vec![None],
                kind: StageKind::Output,
                block: None,
                files: vec![],
                spm: None,
                pdec: None,
                support: Support::Ready,
            },
        ]);
        let ctx = Context {
            phrasebook: None,
            nmt: None,
            case_locale: None,
            target: "fr_FR".into(),
            source: Some("en_US".into()),
            quality: None,
            vocab: None,
        };
        let o = run(&p, &ctx, "anything").expect("run");
        assert_eq!(o.text, None);
        assert_eq!(o.trace.len(), 2);
    }
}
