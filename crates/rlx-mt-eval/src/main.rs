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

//! CLI for [`rlx_mt_eval`].

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rlx_mt_eval::{CorpusScore, CueTimingInput, motor_clip_entities, score_corpus, score_timing};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "rlx-mt-eval",
    about = "chrF / BLEU / TER / entity F1 / timing fit for RLX MT bake-offs"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Score parallel hyp/ref line files (one segment per line).
    Score {
        #[arg(long)]
        hyp: PathBuf,
        #[arg(long)]
        r#ref: PathBuf,
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long, value_enum, default_value_t = GoldPack::None)]
        gold: GoldPack,
        #[arg(long)]
        json: bool,
    },
    /// Score a translator-core `result.json` against gold references.
    ScoreResult {
        #[arg(long)]
        result: PathBuf,
        #[arg(long)]
        lang: String,
        #[arg(long, value_enum, default_value_t = GoldPack::Motor)]
        gold: GoldPack,
        #[arg(long)]
        gold_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Score `dir/{lang}/result.json` for each language.
    ScoreBench {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value = "fr,de,uk")]
        langs: String,
        #[arg(long, value_enum, default_value_t = GoldPack::Motor)]
        gold: GoldPack,
        #[arg(long)]
        gold_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Score every run subdir under a bake-off folder that has `result.json`.
    ///
    /// Language is inferred per run from `target_lang` in JSON, path (`…/de/…`),
    /// or run-name suffix (`mt_nllb_de`). `--lang` is the fallback.
    ScoreBakeoff {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value = "fr")]
        lang: String,
        #[arg(long, value_enum, default_value_t = GoldPack::Motor)]
        gold: GoldPack,
        #[arg(long)]
        gold_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum GoldPack {
    None,
    Motor,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Score {
            hyp,
            r#ref,
            lang,
            gold,
            json,
        } => {
            let hyps = read_lines(&hyp)?;
            let refs = read_lines(&r#ref)?;
            if hyps.len() != refs.len() {
                bail!("line count mismatch: hyp={} ref={}", hyps.len(), refs.len());
            }
            let ents = entities_for(gold, &lang);
            let ent_vecs: Vec<Vec<&str>> = (0..hyps.len()).map(|_| ents.clone()).collect();
            let pairs: Vec<(&str, &str)> = hyps
                .iter()
                .zip(refs.iter())
                .map(|(h, r)| (h.as_str(), r.as_str()))
                .collect();
            let score = score_corpus(pairs, &ent_vecs);
            emit_corpus(&lang, &score, json);
        }
        Cmd::ScoreResult {
            result,
            lang,
            gold,
            gold_file,
            json,
        } => {
            let score = score_one_result(&result, &lang, gold, gold_file.as_deref())?;
            emit_corpus(&lang, &score, json);
        }
        Cmd::ScoreBench {
            dir,
            langs,
            gold,
            gold_file,
            json,
        } => {
            let mut rows = Vec::new();
            for lang in langs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let path = dir.join(lang).join("result.json");
                let score = score_one_result(&path, lang, gold, gold_file.as_deref())?;
                rows.push((lang.to_string(), score));
            }
            emit_bench_rows(&dir, &rows, json)?;
        }
        Cmd::ScoreBakeoff {
            dir,
            lang,
            gold,
            gold_file,
            json,
        } => {
            let mut runs = Vec::new();
            let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
                .with_context(|| format!("read {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            entries.sort();
            for run in entries {
                let run_name = run
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string();
                let inferred = infer_run_lang(&run, &run_name, &lang);
                let path = find_result_json(&run, &inferred);
                let Some(path) = path else { continue };
                let use_lang = infer_lang_from_result(&path).unwrap_or(inferred);
                match score_one_result(&path, &use_lang, gold, gold_file.as_deref()) {
                    Ok(score) => runs.push((format!("{run_name}[{use_lang}]"), score)),
                    Err(e) => eprintln!("skip {} ({use_lang}): {e:#}", run.display()),
                }
            }
            if json {
                let obj = json!({
                    "dir": dir,
                    "runs": runs.iter().map(|(name, s)| corpus_json(name, s)).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
                println!(
                    "{:<32} {:>5} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}",
                    "run", "n", "exact", "chrF", "BLEU", "TER↓", "entF1", "over↑"
                );
                for (name, s) in &runs {
                    let over = s
                        .timing
                        .as_ref()
                        .map(|t| t.mean_overrun_ratio)
                        .unwrap_or(0.0);
                    println!(
                        "{:<32} {:>5} {:>7} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>7.3}",
                        trunc(name, 32),
                        s.n,
                        s.exact,
                        s.mean_chrf,
                        s.corpus_bleu,
                        s.mean_ter,
                        s.mean_entity_f1,
                        over
                    );
                }
            }
        }
    }
    Ok(())
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

const KNOWN_LANGS: &[&str] = &[
    "fr", "de", "uk", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ko", "ar", "tr", "sv", "cs",
];

fn lang_from_token(tok: &str) -> Option<String> {
    let t = tok.trim().to_ascii_lowercase();
    KNOWN_LANGS
        .iter()
        .find(|l| **l == t)
        .map(|l| (*l).to_string())
}

fn infer_run_lang(run: &Path, run_name: &str, fallback: &str) -> String {
    // Prefer lang-named subdirectory with result.json.
    for lang in KNOWN_LANGS {
        if run.join(lang).join("result.json").is_file() {
            return (*lang).to_string();
        }
    }
    // Run name suffix: mt_nllb_de, mt_hy-mt_uk, …
    if let Some(tok) = run_name.rsplit(['_', '-']).next()
        && let Some(l) = lang_from_token(tok)
    {
        return l;
    }
    fallback.to_string()
}

fn infer_lang_from_result(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    if let Some(l) = v.get("target_lang").and_then(|x| x.as_str())
        && !l.is_empty()
    {
        return Some(l.to_string());
    }
    // Parent dir name (…/de/result.json)
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .and_then(lang_from_token)
}

fn find_result_json(run: &Path, lang: &str) -> Option<PathBuf> {
    let candidates = [
        run.join(lang).join("result.json"),
        run.join("result.json"),
        run.join("result.translate.json"),
    ];
    candidates.into_iter().find(|p| p.is_file()).or_else(|| {
        fs::read_dir(run)
            .ok()?
            .filter_map(|e| e.ok())
            .find_map(|e| {
                let p = e.path().join("result.json");
                p.is_file().then_some(p)
            })
    })
}

fn corpus_json(label: &str, s: &CorpusScore) -> serde_json::Value {
    json!({
        "name": label,
        "n": s.n,
        "exact": s.exact,
        "mean_chrf": s.mean_chrf,
        "corpus_bleu": s.corpus_bleu,
        "mean_ter": s.mean_ter,
        "mean_entity_f1": s.mean_entity_f1,
        "mean_entity_recall": s.mean_entity_recall,
        "timing": s.timing,
        "pairs": s.pairs,
    })
}

fn emit_bench_rows(dir: &Path, rows: &[(String, CorpusScore)], as_json: bool) -> Result<()> {
    if as_json {
        let obj = json!({
            "dir": dir,
            "langs": rows.iter().map(|(l, s)| corpus_json(l, s)).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!(
            "{:<6} {:>5} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}",
            "lang", "n", "exact", "chrF", "BLEU", "TER↓", "entF1", "over↑"
        );
        for (lang, s) in rows {
            let over = s
                .timing
                .as_ref()
                .map(|t| t.mean_overrun_ratio)
                .unwrap_or(0.0);
            println!(
                "{:<6} {:>5} {:>7} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>7.3}",
                lang, s.n, s.exact, s.mean_chrf, s.corpus_bleu, s.mean_ter, s.mean_entity_f1, over
            );
        }
        if !rows.is_empty() {
            let n = rows.len() as f64;
            let mean_chrf = rows.iter().map(|(_, s)| s.mean_chrf).sum::<f64>() / n;
            let mean_bleu = rows.iter().map(|(_, s)| s.corpus_bleu).sum::<f64>() / n;
            let mean_ter = rows.iter().map(|(_, s)| s.mean_ter).sum::<f64>() / n;
            let mean_ef1 = rows.iter().map(|(_, s)| s.mean_entity_f1).sum::<f64>() / n;
            println!(
                "{:<6} {:>5} {:>7} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>7}",
                "AVG", "-", "-", mean_chrf, mean_bleu, mean_ter, mean_ef1, "-"
            );
        }
    }
    Ok(())
}

fn emit_corpus(lang: &str, score: &CorpusScore, as_json: bool) {
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&corpus_json(lang, score)).unwrap()
        );
    } else {
        let timing = score
            .timing
            .as_ref()
            .map(|t| {
                format!(
                    " fill={:.3} over={:.3} max_over={:.3}",
                    t.mean_fill_ratio, t.mean_overrun_ratio, t.max_overrun_ratio
                )
            })
            .unwrap_or_default();
        println!(
            "lang={lang} n={} exact={} chrF={:.4} BLEU={:.4} TER={:.4} entF1={:.4}{timing}",
            score.n,
            score.exact,
            score.mean_chrf,
            score.corpus_bleu,
            score.mean_ter,
            score.mean_entity_f1,
        );
        for (i, p) in score.pairs.iter().enumerate() {
            let miss = if p.missing_entities.is_empty() {
                String::new()
            } else {
                format!(" missing={}", p.missing_entities.join(","))
            };
            println!(
                "  [{i}] chrF={:.4} BLEU={:.4} TER={:.4} entF1={:.4} exact={}{miss}",
                p.chrf, p.bleu, p.ter, p.entity_f1, p.exact
            );
        }
    }
}

fn entities_for(gold: GoldPack, lang: &str) -> Vec<&'static str> {
    match gold {
        GoldPack::None => Vec::new(),
        GoldPack::Motor => motor_clip_entities(lang),
    }
}

fn read_lines(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw
        .lines()
        .map(|l| l.trim_end().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

#[derive(Debug, Deserialize)]
struct ResultFile {
    cues: Vec<ResultCue>,
}

#[derive(Debug, Deserialize)]
struct ResultCue {
    id: Option<usize>,
    #[serde(flatten)]
    texts: serde_json::Map<String, serde_json::Value>,
}

impl ResultCue {
    fn hyp_text(&self, lang: &str) -> Option<String> {
        let key = format!("text_{}", lang);
        if let Some(v) = self.texts.get(&key).and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
        self.texts
            .get("text_fr")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn f64_field(&self, key: &str) -> Option<f64> {
        self.texts.get(key).and_then(|v| v.as_f64()).or_else(|| {
            self.texts
                .get(key)
                .and_then(|v| v.as_i64())
                .map(|i| i as f64)
        })
    }

    fn timing_input(&self, idx: usize) -> Option<CueTimingInput> {
        let start = self.f64_field("start_sec")?;
        let end = self.f64_field("end_sec")?;
        Some(CueTimingInput {
            id: self.id.or(Some(idx)),
            start_sec: start,
            end_sec: end,
            placed_duration_sec: self.f64_field("placed_duration_sec"),
        })
    }
}

#[derive(Debug, Deserialize)]
struct GoldFile {
    cues: Vec<GoldCue>,
}

#[derive(Debug, Deserialize)]
struct GoldCue {
    id: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    text_en: Option<String>,
    #[serde(flatten)]
    langs: serde_json::Map<String, serde_json::Value>,
}

impl GoldCue {
    fn ref_for(&self, lang: &str) -> Option<String> {
        self.langs
            .get(lang)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

fn embedded_motor_gold() -> GoldFile {
    serde_json::from_str(include_str!("../gold/motor_helix.json")).expect("motor gold JSON")
}

fn load_gold(gold: GoldPack, gold_file: Option<&Path>) -> Result<Option<GoldFile>> {
    if let Some(path) = gold_file {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        return Ok(Some(serde_json::from_str(&raw)?));
    }
    Ok(match gold {
        GoldPack::None => None,
        GoldPack::Motor => Some(embedded_motor_gold()),
    })
}

fn score_one_result(
    path: &Path,
    lang: &str,
    gold: GoldPack,
    gold_file: Option<&Path>,
) -> Result<CorpusScore> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let result: ResultFile = serde_json::from_str(&raw).context("parse result.json")?;
    let gold = load_gold(gold, gold_file)?.context("need --gold motor or --gold-file")?;

    let ents = motor_clip_entities(lang);
    let mut hyps = Vec::new();
    let mut refs = Vec::new();
    let mut ent_vecs: Vec<Vec<&str>> = Vec::new();
    let mut timing_in = Vec::new();

    for (i, cue) in result.cues.iter().enumerate() {
        let Some(hyp) = cue.hyp_text(lang) else {
            bail!("{}: cue {i} missing target text", path.display());
        };
        let reference = gold
            .cues
            .iter()
            .find(|g| g.id == cue.id || (cue.id.is_none() && g.id == Some(i)))
            .and_then(|g| g.ref_for(lang))
            .or_else(|| gold.cues.get(i).and_then(|g| g.ref_for(lang)))
            .with_context(|| format!("no gold ref for cue {i} lang={lang}"))?;
        if let Some(t) = cue.timing_input(i) {
            timing_in.push(t);
        }
        hyps.push(hyp);
        refs.push(reference);
        ent_vecs.push(ents.clone());
    }

    let pairs: Vec<(&str, &str)> = hyps
        .iter()
        .zip(refs.iter())
        .map(|(h, r)| (h.as_str(), r.as_str()))
        .collect();
    let mut score = score_corpus(pairs, &ent_vecs);
    if !timing_in.is_empty() {
        score.timing = Some(score_timing(&timing_in));
    }
    Ok(score)
}
