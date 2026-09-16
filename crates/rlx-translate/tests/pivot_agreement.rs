//! Scores the *pivot* directions against the OS, with real source text.
//!
//! the OS routes anything that is not X<->English through English, and the graph
//! carries a translator block per hop across two models. Those directions used
//! to produce nothing at all; this measures what they produce now.
//!
//! Source sentences come from the reference dump, so they are genuinely in the
//! source language — feeding one English string to every direction measures the
//! wrong thing and makes a working pivot look broken.
//!
//! Needs `RLX_TRANSLATE_REFERENCE`. Slow: a pivot loads two models per
//! direction.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rlx_translate::assets::Assets;
use rlx_translate::decode::Nmt;
use rlx_translate::execute::{Context, run};
use rlx_translate::pdec::PDecParams;
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

struct Loaded {
    nmt: Nmt,
    vocab: Vocab,
}

fn cases(pair: &LangPair, n: usize) -> Vec<(String, String)> {
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
        .filter(|(s, _)| s.split_whitespace().count() <= 6 && seen.insert(s.to_string()))
        .map(|(s, t)| (s.to_string(), t.to_string()))
        .take(n)
        .collect()
}

#[test]
fn pivot_directions_reproduce_the_os() {
    let per: usize = std::env::var("RLX_TEST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let assets = Assets::discover();
    if assets.roots.is_empty() {
        eprintln!("skipping: no assets installed");
        return;
    }
    // A spread of pivots: different source scripts, different bundle pairings.
    let specs = [
        "ar_AE-de_DE",
        "ar_AE-fr_FR",
        "ja_JP-fr_FR",
        "ru_RU-it_IT",
        "tr_TR-fr_FR",
        "hi_IN-fr_FR",
    ];
    let (mut dirs, mut exact, mut scored, mut chrf) = (0usize, 0usize, 0usize, 0.0f64);
    let (mut t_pb, mut t_mt, mut t_mt_c) = (0usize, 0usize, 0.0f64);
    let mut skipped = Vec::new();

    for spec in specs {
        let pair = LangPair::parse(spec).expect("pair");
        let Ok((_, config)) = assets.best_config(&pair) else {
            skipped.push(format!("{spec} (no config)"));
            continue;
        };
        // Pick sentences the phrasebook *misses*. The first short sentences in
        // every dump happen to be entities the lexicon maps to itself, so an
        // unfiltered sample scores 1.000 and exercises no NMT at all.
        let pb0 = rlx_translate::phrasebook::Phrasebook::load_all(
            &assets.phrasebook_files(&config, &pair),
        )
        .ok();
        let cases: Vec<(String, String)> = cases(&pair, 400)
            .into_iter()
            .filter(|(s, _)| pb0.as_ref().is_none_or(|p| p.translate(s).is_none()))
            .take(per)
            .collect();
        if cases.is_empty() {
            skipped.push(format!("{spec} (no reference the phrasebook misses)"));
            continue;
        }
        let Ok(plan) = TranslationPlan::build(&config, "mt_app", &pair) else {
            skipped.push(format!("{spec} (no plan)"));
            continue;
        };

        // One model per translator stage, keyed as the CLI keys them.
        let models: std::cell::RefCell<BTreeMap<String, Loaded>> =
            std::cell::RefCell::new(BTreeMap::new());
        let nmt = |p: &PDecParams, s: &str| -> anyhow::Result<String> {
            // The table is part of the identity: zh_CN and zh_TW share a model
            // file and a target-language prefix but not a shortlist.
            let key = format!(
                "{}|{}|{}|{}",
                p.model_file, p.source_locale, p.target_locale, p.shortlist.lang_pair
            );
            if !models.borrow().contains_key(&key) {
                let home = assets
                    .model_home_for(&pair, &p.model_file)
                    .ok_or_else(|| anyhow::anyhow!("{} not installed", p.model_file))?;
                let vocab = Vocab::load(home.join("spm.model"))?;
                let tgt: String = p
                    .target_locale
                    .chars()
                    .take(2)
                    .collect::<String>()
                    .to_lowercase();
                let src_lang: String = p
                    .source_locale
                    .chars()
                    .take(2)
                    .collect::<String>()
                    .to_lowercase();
                let mut m = Nmt::load_for_pair(
                    &home,
                    &[home.as_path()],
                    &src_lang,
                    &tgt,
                    &p.shortlist.lang_pair,
                )?;
                // The repetition levers were rejected earlier on lexicon data,
                // where repetition never happens. This corpus is where it does.
                if let Ok(v) = std::env::var("RLX_TRANSLATE_NO_REPEAT")
                    && let Ok(k) = v.parse::<usize>()
                {
                    m.set_no_repeat_ngram(k);
                }
                if let Ok(v) = std::env::var("RLX_TRANSLATE_NO_REPEAT_CHARS")
                    && let Ok(k) = v.parse::<usize>()
                {
                    m.set_no_repeat_char_ngram(k);
                }
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
        // The phrasebook is part of the pipeline, not a confound: an earlier
        // version of this test stubbed it out and measured the NMT-only path,
        // which understates the real thing badly. `10.5" iPad Pro` is a
        // phrasebook entity that maps to itself; without the lookup the NMT
        // duplicates it and the direction looks broken when it is not.
        let pb = rlx_translate::phrasebook::Phrasebook::load_all(
            &assets.phrasebook_files(&config, &pair),
        )
        .ok();
        let lookup = |s: &str| pb.as_ref().and_then(|p| p.translate(s).map(str::to_string));
        let ctx = Context {
            phrasebook: Some(&lookup),
            nmt: Some(&nmt),
            case_locale: Some(pair.target.clone()),
            target: pair.target.clone(),
            source: Some(pair.source.clone()),
            quality: None,
            vocab: None,
        };

        let (mut de, mut dc, mut dn) = (0usize, 0.0f64, 0usize);
        // Split the score by who produced it. A sample that happens to be all
        // phrasebook entities scores 1.000 and says nothing about the pivot;
        // reporting the two together is how a measurement flatters itself.
        let (mut pb_n, mut pb_e) = (0usize, 0usize);
        let (mut mt_n, mut mt_e, mut mt_c) = (0usize, 0usize, 0.0f64);
        let mut sample = String::new();
        for (src, want) in &cases {
            let Ok(out) = run(&plan, &ctx, src) else {
                continue;
            };
            let Some(got) = out.text else { continue };
            dn += 1;
            let hit = lookup(src).is_some();
            let ok = got.trim().eq_ignore_ascii_case(want.trim());
            let c = rlx_translate::score::chrf(&got, want);
            if ok {
                de += 1;
            }
            dc += c;
            if hit {
                pb_n += 1;
                if ok {
                    pb_e += 1;
                }
            } else {
                mt_n += 1;
                mt_c += c;
                if ok {
                    mt_e += 1;
                }
            }
            if sample.is_empty() {
                sample = format!("{src:?} -> {got:?} (os {want:?})");
            }
        }
        if dn == 0 {
            skipped.push(format!("{spec} (produced nothing)"));
            continue;
        }
        eprintln!(
            "  {spec}  {de}/{dn} identical, chrF {:.3}  [phrasebook {pb_e}/{pb_n}, nmt {mt_e}/{mt_n} chrF {:.3}]   {sample}",
            dc / dn as f64,
            if mt_n > 0 { mt_c / mt_n as f64 } else { 0.0 }
        );
        dirs += 1;
        t_pb += pb_n;
        t_mt += mt_n;
        t_mt_c += mt_c;
        exact += de;
        scored += dn;
        chrf += dc;
    }

    for s in &skipped {
        eprintln!("  skipped {s}");
    }
    eprintln!(
        "\n  {dirs} pivot directions, {scored} sentences: {exact} identical, chrF {:.3}",
        chrf / scored.max(1) as f64
    );
    eprintln!(
        "  of those, {t_pb} were phrasebook hits and {t_mt} went through the pivot NMT{}",
        if t_mt > 0 {
            format!(" (chrF {:.3})", t_mt_c / t_mt as f64)
        } else {
            " — so this sample says nothing about the NMT".to_string()
        }
    );
    if dirs == 0 {
        // Not a failure of the pivot: the reference dump was generated by
        // sampling the phrasebook files themselves, so every sentence in it is
        // a lexicon entry and none reaches the NMT. Scoring the pivot NMT
        // against the OS needs a dump of non-lexicon sentences, which means
        // another run of the Swift harness.
        eprintln!(
            "  no direction had reference material the phrasebook misses — the dump is\n             \x20 phrasebook-derived, so it cannot exercise the NMT. Regenerate it with\n             \x20 non-lexicon sentences to score this."
        );
        return;
    }
    // These used to produce nothing at all; anything semantically related beats
    // that, and chrF this low would mean the second hop is mistranslating.
    assert!(
        scored > 0 && chrf / scored as f64 > 0.35,
        "pivot chrF {:.3} suggests a hop is translating the wrong direction",
        chrf / scored as f64
    );
}
