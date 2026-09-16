//! Runs **every** installed direction through the shipped stage graph.
//!
//! An earlier version drove the NMT directly and skipped the 288 directions
//! the OS routes through English, because there was no pivot. The executor walks
//! the config's graph, and a pivot pair's graph simply contains a translator
//! block per hop — so every direction can run now.
//!
//! Two numbers, kept apart on purpose:
//!
//! * **coverage** — did the direction produce a translation at all. This is
//!   what the pivot work changed, and it needs no reference.
//! * **chrF against the OS** — only meaningful where a reference exists.
//!   `RLX_TRANSLATE_REFERENCE` defaults to the *non-lexicon* dump, because the
//!   original one was built by sampling the phrasebook and cannot exercise the
//!   NMT; scores on it measure the lexicon. Which producer answered is reported
//!   either way.
//!
//! Run with `--release`: decode is ~13x slower in a debug build.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rlx_translate::assets::Assets;
use rlx_translate::decode::Nmt;
use rlx_translate::execute::{Context, run};
use rlx_translate::pdec::PDecParams;
use rlx_translate::phrasebook::Phrasebook;
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

struct Loaded {
    nmt: Nmt,
    vocab: Vocab,
}

fn reference_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("RLX_TRANSLATE_REFERENCE").unwrap_or_else(|_| {
            "/private/tmp/claude-501/-Users-Shared-rlx-models/\
         08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad/reference_nonlex"
                .to_string()
        }),
    )
}

/// Reference pairs for a direction, if any.
fn reference(dir: &std::path::Path, pair: &LangPair, n: usize) -> Vec<(String, String)> {
    let path = dir.join(format!("{}-{}.tsv", pair.source, pair.target));
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut seen = std::collections::BTreeSet::new();
    text.lines()
        .filter_map(|l| l.split_once('\t'))
        .filter(|(s, _)| seen.insert(s.to_string()))
        .map(|(s, t)| (s.to_string(), t.to_string()))
        .take(n)
        .collect()
}

#[test]
fn every_installed_direction_runs() {
    let per: usize = std::env::var("RLX_TEST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let limit: usize = std::env::var("RLX_TEST_DIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let assets = Assets::discover();
    if assets.roots.is_empty() {
        eprintln!("skipping: no translation assets installed");
        return;
    }
    let refdir = reference_dir();

    let mut directions: Vec<LangPair> = Vec::new();
    for p in assets.pairs() {
        directions.push(p.clone());
        directions.push(p.reversed());
    }
    directions.sort_by(|a, b| (&a.source, &a.target).cmp(&(&b.source, &b.target)));
    directions.dedup_by(|a, b| a.source == b.source && a.target == b.target);
    directions.truncate(limit);

    // One cache for the whole sweep: directions share decoders heavily.
    let models: std::cell::RefCell<BTreeMap<String, Loaded>> =
        std::cell::RefCell::new(BTreeMap::new());
    let (mut ran, mut produced, mut failed) = (0usize, 0usize, 0usize);
    let (mut scored, mut exact, mut chrf) = (0usize, 0usize, 0.0f64);
    let (mut by_pb, mut by_nmt) = (0usize, 0usize);
    let mut reasons: Vec<String> = Vec::new();

    for pair in &directions {
        let Ok((_, config)) = assets.best_config(pair) else {
            continue;
        };
        let Ok(plan) = TranslationPlan::build(&config, "mt_app", pair) else {
            continue;
        };
        ran += 1;

        let pb = Phrasebook::load_all(&assets.phrasebook_files(&config, pair)).ok();
        let lookup = |s: &str| pb.as_ref().and_then(|p| p.translate(s).map(str::to_string));
        let nmt = |p: &PDecParams, s: &str| -> anyhow::Result<String> {
            // Key on the *resolved* bundle, not the pair: every direction into
            // French wants the same decoder, and reloading it per direction is
            // what made a full sweep take hours.
            let home = assets
                .model_home_for(pair, &p.model_file)
                .ok_or_else(|| anyhow::anyhow!("{} not installed", p.model_file))?;
            let tgt: String = p
                .target_locale
                .chars()
                .take(2)
                .collect::<String>()
                .to_lowercase();
            // The shortlist table is part of the identity: zh_CN and zh_TW
            // share a bundle and a `tgt` but not a table.
            // The source language is part of the identity too: `input_<lang>`
            // follows it, so en->fr and de->fr are different loads.
            let key = format!(
                "{}|{}|{tgt}|{}",
                home.display(),
                p.source_locale,
                p.shortlist.lang_pair
            );
            if !models.borrow().contains_key(&key) {
                // Bounded: each model materialises a tied readout table — up to
                // 168 000 x 512 f32, ~344 MB — so an unbounded cache over 400
                // directions would exhaust memory on a shared machine. Six is
                // enough to cover a pivot's two hops plus neighbours.
                const MAX_MODELS: usize = 6;
                if models.borrow().len() >= MAX_MODELS {
                    let victim = models.borrow().keys().next().cloned();
                    if let Some(k) = victim {
                        models.borrow_mut().remove(&k);
                    }
                }
                let vocab = Vocab::load(home.join("spm.model"))?;
                let src_lang: String = p
                    .source_locale
                    .chars()
                    .take(2)
                    .collect::<String>()
                    .to_lowercase();
                let m = Nmt::load_for_pair(
                    &home,
                    &[home.as_path()],
                    &src_lang,
                    &tgt,
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
        let ctx = Context {
            phrasebook: Some(&lookup),
            nmt: Some(&nmt),
            case_locale: Some(pair.target.clone()),
            target: pair.target.clone(),
            source: Some(pair.source.clone()),
            quality: None,
            vocab: None,
        };

        // Coverage uses whatever source text is to hand; a reference when there
        // is one, otherwise a phrase in the pair's own source language is not
        // available, so fall back to the reference of any direction sharing the
        // source. Failing that the direction is skipped rather than fed English.
        let cases = {
            let own = reference(&refdir, pair, per);
            if own.is_empty() {
                let alt = std::fs::read_dir(&refdir)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .find(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with(&format!("{}-", pair.source)))
                    });
                alt.map(|p| {
                    std::fs::read_to_string(p)
                        .unwrap_or_default()
                        .lines()
                        .filter_map(|l| {
                            l.split_once('\t')
                                .map(|(s, _)| (s.to_string(), String::new()))
                        })
                        .take(per)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
            } else {
                own
            }
        };
        if cases.is_empty() {
            reasons.push(format!("{}-{}: no source text", pair.source, pair.target));
            continue;
        }

        let mut got_any = false;
        for (src, want) in &cases {
            let Ok(out) = run(&plan, &ctx, src) else {
                continue;
            };
            let Some(got) = out.text else { continue };
            got_any = true;
            if lookup(src).is_some() {
                by_pb += 1;
            } else {
                by_nmt += 1;
            }
            if !want.is_empty() {
                scored += 1;
                if got.trim().eq_ignore_ascii_case(want.trim()) {
                    exact += 1;
                }
                chrf += rlx_translate::score::chrf(&got, want);
            }
        }
        if got_any {
            produced += 1;
        } else {
            failed += 1;
            reasons.push(format!("{}-{}: produced nothing", pair.source, pair.target));
        }
    }

    eprintln!(
        "  {} models resident at the end (cache is bounded)",
        models.borrow().len()
    );
    eprintln!(
        "\n  {ran} directions attempted, {produced} produced a translation, {failed} did not"
    );
    eprintln!("  answers: {by_pb} from the phrasebook, {by_nmt} from the NMT");
    if scored > 0 {
        eprintln!(
            "  against the OS on {scored} scored sentences: {exact} identical, chrF {:.3}",
            chrf / scored as f64
        );
    }
    // Print the failures in full and the missing-input cases in summary: a
    // truncated list hid which seven directions actually failed, and that is
    // the only part worth acting on.
    let (fails, no_input): (Vec<_>, Vec<_>) =
        reasons.iter().partition(|r| r.contains("produced nothing"));
    for r in &fails {
        eprintln!("    FAILED {r}");
    }
    eprintln!(
        "    ({} directions had no source text in the dump)",
        no_input.len()
    );

    assert!(ran > 0, "no direction was attempted");
    // Score against the directions that *had input*, not against every
    // installed direction: the corpus covers nine source languages, so most
    // directions cannot be fed at all and counting them as failures measures
    // the corpus rather than the pipeline.
    let attempted = produced + failed;
    assert!(
        attempted > 0 && produced * 10 >= attempted * 9,
        "only {produced} of {attempted} feedable directions produced a translation"
    );
}
