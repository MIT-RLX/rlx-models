//! How good is the translation, not just how faithfully do we reproduce the OS?
//!
//! Every measurement in this port so far scores our output against *the OS's*,
//! which says whether the port is right and nothing about whether the model is
//! any good. FLORES-200 devtest is human-translated and line-aligned across all
//! 200 languages, so the same sentence set gives a human reference in every
//! direction.
//!
//! Reports three chrF figures per direction: ours against the human reference,
//! the OS's against it, and ours against the OS's. The first two together say
//! whether a gap to the OS costs anything a reader would notice.
//!
//! Needs `RLX_TRANSLATE_FLORES` (the aligned source/reference directory) and
//! `RLX_TRANSLATE_REFERENCE` (the OS's output on the same sentences).

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rlx_translate::assets::Assets;
use rlx_translate::execute::{Context, run};
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::LangPair;
use rlx_translate::score::chrf;

fn lines(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

fn main() -> Result<()> {
    // Two shapes of reference set. FLORES is aligned across *languages*, so the
    // human reference is line `i` of the target locale's file. Tatoeba is
    // aligned per *pair*, so it comes in a per-direction TSV beside the source.
    let flores = std::env::var("RLX_TRANSLATE_FLORES").ok();
    let pools = std::env::var("RLX_TRANSLATE_HUMAN").ok();
    if flores.is_none() && pools.is_none() {
        anyhow::bail!(
            "set RLX_TRANSLATE_FLORES (aligned per locale) or RLX_TRANSLATE_HUMAN (per direction)"
        );
    }
    let refs = std::env::var("RLX_TRANSLATE_REFERENCE")
        .map_err(|_| anyhow!("set RLX_TRANSLATE_REFERENCE to the OS's output"))?;
    let per: usize = std::env::var("RLX_TEST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let assets = Assets::discover();

    let mut names: Vec<String> = std::fs::read_dir(&refs)?
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension()? == "tsv").then(|| p.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    names.sort();
    if let Ok(only) = std::env::var("RLX_TEST_PAIR") {
        let keep: Vec<&str> = only.split(',').map(str::trim).collect();
        names.retain(|n| keep.contains(&n.as_str()));
    }

    println!(
        "{:<16}{:>6}{:>11}{:>11}{:>12}",
        "direction", "n", "ours~human", "os~human", "ours~os"
    );
    let (mut so, mut sa, mut soa, mut sn) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for name in &names {
        let Ok(pair) = LangPair::parse(name) else {
            continue;
        };
        // Human reference keyed by source text, not by position: the batch is
        // not promised to come back in order.
        let mut human: HashMap<String, String> = HashMap::new();
        if let Some(dir) = &pools {
            for l in lines(&std::path::Path::new(dir).join(format!("{name}.tsv"))) {
                if let Some((s, h)) = l.split_once('\t') {
                    human.insert(s.to_string(), h.to_string());
                }
            }
        }
        if human.is_empty()
            && let Some(dir) = &flores
        {
            let dir = std::path::Path::new(dir);
            let src_lines = lines(&dir.join(format!("{}.txt", pair.source)));
            let tgt_lines = lines(&dir.join(format!("{}.txt", pair.target)));
            if src_lines.len() == tgt_lines.len() {
                human = src_lines.into_iter().zip(tgt_lines).collect();
            }
        }
        if human.is_empty() {
            continue;
        }

        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let Ok(plan) = TranslationPlan::build(&config, "mt_app", &pair) else {
            continue;
        };
        let models: std::cell::RefCell<
            std::collections::BTreeMap<
                String,
                (rlx_translate::decode::Nmt, rlx_translate::spm::Vocab),
            >,
        > = std::cell::RefCell::new(std::collections::BTreeMap::new());
        let run_nmt = |p: &rlx_translate::pdec::PDecParams, s: &str| -> Result<String> {
            let key = format!(
                "{}|{}|{}|{}",
                p.model_file, p.source_locale, p.target_locale, p.shortlist.lang_pair
            );
            if !models.borrow().contains_key(&key) {
                let home = assets
                    .model_home_for(&pair, &p.model_file)
                    .ok_or_else(|| anyhow!("{} not installed", p.model_file))?;
                let vocab = rlx_translate::spm::Vocab::load(home.join("spm.model"))?;
                let two = |l: &str| l.chars().take(2).collect::<String>().to_lowercase();
                let m = rlx_translate::decode::Nmt::load_for_pair(
                    &home,
                    &[home.as_path()],
                    &two(&p.source_locale),
                    &two(&p.target_locale),
                    &p.shortlist.lang_pair,
                )?;
                models.borrow_mut().insert(key.clone(), (m, vocab));
            }
            let b = models.borrow();
            let (m, v) = b.get(&key).expect("just inserted");
            let mut best = m.translate_nbest(v, p, s, 1)?;
            if best.is_empty() {
                anyhow::bail!("no hypothesis");
            }
            Ok(best.remove(0).text)
        };
        let lookup = |_: &str| None;
        let own = assets
            .model_home(&pair)
            .and_then(|h| rlx_translate::spm::Vocab::load(h.join("spm.model")).ok());
        let ctx = Context {
            phrasebook: Some(&lookup),
            nmt: Some(&run_nmt),
            case_locale: Some(pair.target.clone()),
            target: pair.target.clone(),
            source: Some(pair.source.clone()),
            quality: None,
            vocab: own.as_ref(),
        };

        let (mut o, mut a, mut oa, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for line in lines(&std::path::Path::new(&refs).join(format!("{name}.tsv")))
            .iter()
            .take(per)
        {
            let Some((src, os_ref)) = line.split_once('\t') else {
                continue;
            };
            let Some(human) = human.get(src) else {
                continue; // no human reference for this sentence
            };
            let Ok(out) = run(&plan, &ctx, src) else {
                continue;
            };
            let Some(got) = out.text else { continue };
            if std::env::var("RLX_BENCH_SHOW").is_ok() {
                println!("  src   {src}");
                println!("  human {human}");
                println!("  os    {os_ref}");
                println!("  ours  {got}");
                println!(
                    "  chrf ours~human {:.3} os~human {:.3}\n",
                    chrf(&got, human),
                    chrf(os_ref, human)
                );
            }
            o += chrf(&got, human);
            a += chrf(os_ref, human);
            oa += chrf(&got, os_ref);
            n += 1;
        }
        if n == 0 {
            continue;
        }
        println!(
            "{name:<16}{n:>6}{:>11.3}{:>11.3}{:>12.3}",
            o / n as f64,
            a / n as f64,
            oa / n as f64
        );
        so += o;
        sa += a;
        soa += oa;
        sn += n;
    }
    if sn > 0 {
        println!(
            "\n  {sn} sentences: ours vs human {:.3}, os vs human {:.3}, ours vs os {:.3}",
            so / sn as f64,
            sa / sn as f64,
            soa / sn as f64
        );
    }
    Ok(())
}
