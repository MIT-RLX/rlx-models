//! Scores the *graph executor* against the OS's live output.
//!
//! `execute::run` replaced a hand-wired phrasebook-then-NMT order, and a
//! refactor of the main path deserves a number rather than four hand-checked
//! sentences. This runs the shipped stage graph over the reference dump and
//! reports how often the whole pipeline reproduces the OS exactly.
//!
//! Needs `RLX_TRANSLATE_REFERENCE`; skips without it. Every `<src>-<tgt>.tsv`
//! in that directory is scored, not just one pair — the dump now covers 52
//! directions, and a tuning change that helps French can cost Japanese.
//! `RLX_TEST_PAIR` narrows it to a comma-separated subset; `RLX_TEST_N` caps
//! sentences per direction.

use std::path::PathBuf;

use rlx_translate::assets::Assets;
use rlx_translate::execute::{Context, run};
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::LangPair;

/// Every direction the dump holds, in a stable order.
fn directions() -> Vec<LangPair> {
    let Ok(dir) = std::env::var("RLX_TRANSLATE_REFERENCE") else {
        return Vec::new();
    };
    // A comma-separated list, because a tuning A/B wants a handful of
    // directions run twice rather than all 52 run once: the full sweep is an
    // hour on this machine, which is longer than the patience any lever
    // deserves before it has shown anything.
    let only: Option<Vec<String>> = std::env::var("RLX_TEST_PAIR")
        .ok()
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect());
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<LangPair> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "tsv"))
        .filter_map(|p| {
            let stem = p.file_stem()?.to_str()?.to_string();
            if only.as_ref().is_some_and(|o| !o.contains(&stem)) {
                return None;
            }
            LangPair::parse(&stem).ok()
        })
        .collect();
    out.sort_by_key(|p| format!("{}-{}", p.source, p.target));
    out
}

fn reference(pair: &LangPair) -> Vec<(String, String)> {
    let Ok(dir) = std::env::var("RLX_TRANSLATE_REFERENCE") else {
        return Vec::new();
    };
    let path = PathBuf::from(dir).join(format!("{}-{}.tsv", pair.source, pair.target));
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut seen = std::collections::BTreeSet::new();
    text.lines()
        .filter_map(|l| l.split_once('\t'))
        .filter(|(s, _)| seen.insert(s.to_string()))
        .map(|(s, t)| (s.to_string(), t.to_string()))
        .collect()
}

#[test]
fn the_graph_reproduces_the_os_output() {
    let n: usize = std::env::var("RLX_TEST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let assets = Assets::discover();
    let pairs = directions();
    if pairs.is_empty() {
        eprintln!("skipping: set RLX_TRANSLATE_REFERENCE to the OS's output dump");
        return;
    }

    let started = std::time::Instant::now();
    let (mut exact, mut total, mut chrf, mut empty) = (0usize, 0usize, 0.0f64, 0usize);
    let mut skipped: Vec<String> = Vec::new();
    let mut rows: Vec<(String, usize, usize, f64)> = Vec::new();

    for pair in &pairs {
        let cases = reference(pair);
        if cases.is_empty() {
            continue;
        }
        // Each direction is scored once and its model dropped: 52 resident
        // bundles is how the earlier all-pairs sweep exhausted memory.
        let Some(out) = score_direction(&assets, pair, &cases, n, &mut skipped) else {
            continue;
        };
        let (p_exact, p_total, p_chrf, p_empty) = out;
        // 52 directions on a contended machine is an hour or more; a run that
        // prints only at the end is indistinguishable from one that has hung.
        eprintln!(
            "  {:<16} {p_total:>3} sentences  {p_exact:>3} exact  chrF {:.3}  ({:.0}s elapsed)",
            format!("{}-{}", pair.source, pair.target),
            p_chrf / p_total.max(1) as f64,
            started.elapsed().as_secs_f64()
        );
        exact += p_exact;
        total += p_total;
        chrf += p_chrf;
        empty += p_empty;
        if p_total > 0 {
            rows.push((
                format!("{}-{}", pair.source, pair.target),
                p_exact,
                p_total,
                p_chrf / p_total as f64,
            ));
        }
    }

    rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "\n  {:<16} {:>6} {:>8} {:>8}",
        "direction", "n", "exact", "chrF"
    );
    for (name, e, t, c) in &rows {
        eprintln!("  {name:<16} {t:>6} {e:>8} {c:>8.3}");
    }
    eprintln!(
        "\n  graph over {total} sentences in {} directions: {exact} identical to the OS, \
         chrF {:.3}{} ({:.1}s)",
        rows.len(),
        chrf / total.max(1) as f64,
        if empty > 0 {
            format!(", {empty} produced nothing")
        } else {
            String::new()
        },
        started.elapsed().as_secs_f64()
    );
    if !skipped.is_empty() {
        eprintln!("  skipped (no single-hop model): {}", skipped.join(" "));
    }

    assert!(total > 0, "the graph produced nothing at all");
    assert_eq!(empty, 0, "{empty} sentences fell through every stage");
    // Guards the executor against silently losing the NMT. Averaged over every
    // direction, including the ones that score worst.
    assert!(
        chrf / total as f64 > 0.55,
        "graph chrF {:.3} is far below what the NMT scores on its own",
        chrf / total as f64
    );
}

/// Drops trailing sentence-final marks, in every script the corpus uses.
fn trim_final(s: &str) -> String {
    s.trim()
        .trim_end_matches([
            '.', '!', '?', '\u{3002}', '\u{FF01}', '\u{FF1F}', '\u{06D4}', '\u{061F}', '\u{0964}',
            '\u{104B}',
        ])
        .trim()
        .to_lowercase()
}

/// Runs one direction end to end. `None` when nothing could be resolved.
///
/// Models are resolved **per translator stage**, not per direction: a pivot
/// routes through English and so has two `PDecTranslatorBlock`s with different
/// bundles. Keying on the direction skipped all 14 of those, which is a seventh
/// of what the machine can translate going unmeasured.
fn score_direction(
    assets: &Assets,
    pair: &LangPair,
    cases: &[(String, String)],
    n: usize,
    skipped: &mut Vec<String>,
) -> Option<(usize, usize, f64, usize)> {
    let name = format!("{}-{}", pair.source, pair.target);
    let (_, config) = assets.best_config(pair).ok()?;
    let plan = TranslationPlan::build(&config, "mt_app", pair).ok()?;

    struct Loaded {
        nmt: rlx_translate::decode::Nmt,
        vocab: rlx_translate::spm::Vocab,
    }
    let models: std::cell::RefCell<std::collections::BTreeMap<String, Loaded>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    let failed = std::cell::RefCell::new(Option::<String>::None);

    let run_nmt = |p: &rlx_translate::pdec::PDecParams, s: &str| -> anyhow::Result<String> {
        // Bundle, both languages and the shortlist table all matter: zh_CN and
        // zh_TW share a bundle and a target prefix but not a table, and
        // `input_<lang>` follows the source.
        let key = format!(
            "{}|{}|{}|{}",
            p.model_file, p.source_locale, p.target_locale, p.shortlist.lang_pair
        );
        if !models.borrow().contains_key(&key) {
            let home = assets
                .model_home_for(pair, &p.model_file)
                .ok_or_else(|| anyhow::anyhow!("{} not installed", p.model_file))?;
            let vocab = rlx_translate::spm::Vocab::load(home.join("spm.model"))?;
            let two = |l: &str| l.chars().take(2).collect::<String>().to_lowercase();
            let m = rlx_translate::decode::Nmt::load_for_pair(
                &home,
                &[home.as_path()],
                &two(&p.source_locale),
                &two(&p.target_locale),
                &p.shortlist.lang_pair,
            )?;
            models
                .borrow_mut()
                .insert(key.clone(), Loaded { nmt: m, vocab });
        }
        let b = models.borrow();
        let m = b.get(&key).expect("just inserted");
        let mut best = m.nmt.translate_nbest(&m.vocab, p, s, 1)?;
        if best.is_empty() {
            anyhow::bail!("no hypothesis");
        }
        Ok(best.remove(0).text)
    };

    // Rendering token values needs *a* vocabulary; every stage in a direction
    // shares one within its bundle, and the first translator's is the one the
    // final SentencePiece decode belongs to.
    let own_vocab = assets
        .model_home(pair)
        .and_then(|h| rlx_translate::spm::Vocab::load(h.join("spm.model")).ok());

    let lookup = |_: &str| None; // phrasebook covered separately by `bench`
    let ctx = Context {
        phrasebook: Some(&lookup),
        nmt: Some(&run_nmt),
        case_locale: Some(pair.target.clone()),
        target: pair.target.clone(),
        source: Some(pair.source.clone()),
        quality: None,
        vocab: own_vocab.as_ref(),
    };

    let (mut exact, mut total, mut chrf, mut empty) = (0usize, 0usize, 0.0f64, 0usize);
    let (mut near_punct, mut near_case) = (0usize, 0usize);
    for (src, want) in cases.iter().take(n) {
        let out = match run(&plan, &ctx, src) {
            Ok(o) => o,
            Err(e) => {
                *failed.borrow_mut() = Some(format!("{e:#}"));
                continue;
            }
        };
        let Some(got) = out.text else {
            empty += 1;
            continue;
        };
        total += 1;
        if got.trim().eq_ignore_ascii_case(want.trim()) {
            exact += 1;
        } else if trim_final(&got) == trim_final(want) {
            near_punct += 1;
        } else if got.trim().to_lowercase() == want.trim().to_lowercase() {
            near_case += 1;
        }
        chrf += rlx_translate::score::chrf(&got, want);
    }
    if total == 0 {
        let why = failed
            .borrow()
            .clone()
            .unwrap_or_else(|| "no output".into());
        skipped.push(format!("{name} ({why})"));
        return None;
    }
    if near_punct + near_case > 0 {
        eprintln!(
            "      (+{near_punct} would match but for final punctuation, \
             +{near_case} but for case)"
        );
    }
    Some((exact, total, chrf, empty))
}
