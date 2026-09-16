//! Every direction this machine can translate, scored against human references.
//!
//! The agreement harness covers the 57 directions the reference corpus was
//! collected for, but the machine supports around 400 — and this port has twice
//! found a defect that hit exactly one direction (`en_US-zh_TW`'s shortlist,
//! `en_US-tr_TR`'s input graph). Those were found because a direction happened
//! to be in the corpus. The rest have never been looked at.
//!
//! FLORES is line-aligned across every locale, so a human reference exists for
//! every ordered pair without needing the OS's output — which means this needs
//! no harness run and can sweep the lot. It is a smoke test, not a measurement:
//! a handful of sentences per direction cannot resolve 0.01, but it will find
//! the next direction that is broken rather than merely different.

use anyhow::{Result, anyhow};
use rlx_translate::assets::Assets;
use rlx_translate::execute::{Context, run};
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::LangPair;
use rlx_translate::score::chrf;

fn lines(p: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(p)
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

fn main() -> Result<()> {
    let dir = std::env::var("RLX_TRANSLATE_FLORES")
        .map_err(|_| anyhow!("set RLX_TRANSLATE_FLORES to the aligned corpus"))?;
    let per: usize = std::env::var("RLX_TEST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let dir = std::path::Path::new(&dir);
    let assets = Assets::discover();

    let mut locales: Vec<String> = std::fs::read_dir(dir)?
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension()? != "txt" {
                return None;
            }
            Some(p.file_stem()?.to_str()?.to_string())
        })
        .collect();
    locales.sort();
    let text: std::collections::BTreeMap<String, Vec<String>> = locales
        .iter()
        .map(|l| (l.clone(), lines(&dir.join(format!("{l}.txt")))))
        .collect();

    // One cache for the whole sweep. It used to be per direction, so the same
    // ten bundles were loaded 378 times; they are shared across dozens of
    // directions each.
    type Loaded = (rlx_translate::decode::Nmt, rlx_translate::spm::Vocab);
    let models: std::cell::RefCell<std::collections::BTreeMap<String, Loaded>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    let mut rows: Vec<(String, usize, f64)> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let started = std::time::Instant::now();
    for src in &locales {
        for tgt in &locales {
            if src == tgt {
                continue;
            }
            let name = format!("{src}-{tgt}");
            let Ok(pair) = LangPair::parse(&name) else {
                continue;
            };
            let Ok((_, config)) = assets.best_config(&pair) else {
                continue;
            };
            let Ok(plan) = TranslationPlan::build(&config, "mt_app", &pair) else {
                continue;
            };
            let (Some(s), Some(t)) = (text.get(src), text.get(tgt)) else {
                continue;
            };

            let run_nmt = |p: &rlx_translate::pdec::PDecParams, q: &str| -> Result<String> {
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
                let mut best = m.translate_nbest(v, p, q, 1)?;
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

            let (mut acc, mut n) = (0.0f64, 0usize);
            for (q, human) in s.iter().zip(t).take(per) {
                let Ok(out) = run(&plan, &ctx, q) else {
                    continue;
                };
                let Some(got) = out.text else { continue };
                acc += chrf(&got, human);
                n += 1;
            }
            if n == 0 {
                failed.push(name);
                continue;
            }
            let score = acc / n as f64;
            println!(
                "{name:<16}{n:>4}{score:>9.3}   ({:.0}s)",
                started.elapsed().as_secs_f64()
            );
            rows.push((name, n, score));
        }
    }
    rows.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n  {} directions scored\n\n  weakest:", rows.len());
    for (d, n, v) in rows.iter().take(20) {
        println!("    {d:<16}{n:>4}{v:>9.3}");
    }
    let mean = rows.iter().map(|r| r.2).sum::<f64>() / rows.len().max(1) as f64;
    println!("\n  mean chrF vs human {mean:.3}");
    if !failed.is_empty() {
        println!("  produced nothing: {}", failed.join(" "));
    }
    Ok(())
}
