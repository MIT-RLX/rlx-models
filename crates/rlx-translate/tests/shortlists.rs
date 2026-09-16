//! Every shortlist table this machine has must parse.
//!
//! The reader probed a hard-coded length taken from the French bundle, so it
//! failed on every other vocabulary — and `Nmt::load` swallowed the failure
//! with `.ok()`, falling back to scoring the whole vocabulary. The output stayed
//! plausible, which is why it survived: `en_US-zh_TW` came out in Simplified
//! Chinese and scored chrF 0.348, the worst of 43 directions, without anything
//! erroring.
//!
//! One assertion per installed table, so a format assumption that holds for one
//! bundle and not another fails here rather than in the translations.

use std::collections::BTreeSet;

use rlx_translate::shortlist::Shortlist;

#[test]
fn every_installed_shortlist_table_parses() {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    let assets = rlx_translate::assets::Assets::discover();
    // Search the asset directories rather than the roots: on this machine the
    // bundles live under opaque `<sha1>.asset/AssetData/` paths.
    let mut search: Vec<std::path::PathBuf> = assets.roots.clone();
    search.extend(assets.asset_dirs.values().cloned());
    for dir in search {
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if p.file_name().is_some_and(|n| n == "shortlists") {
                        roots.push(p);
                    } else {
                        stack.push(p);
                    }
                }
            }
        }
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut checked = 0usize;
    for dir in roots {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "shortlist") {
                continue;
            }
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            let s = Shortlist::load(&p).unwrap_or_else(|e| panic!("{name} did not parse: {e:#}"));
            // The table covers the whole vocabulary, so it is never trivially
            // short — a parser that lands on a wrong length usually lands low.
            assert!(
                s.len() >= 48_000,
                "{name} covers only {} source tokens",
                s.len()
            );
            checked += 1;
            eprintln!("  {name:<22} {} source tokens", s.len());
        }
    }
    if checked == 0 {
        eprintln!("skipping: no shortlist tables installed");
        return;
    }
    eprintln!("  {checked} tables parsed");
}
