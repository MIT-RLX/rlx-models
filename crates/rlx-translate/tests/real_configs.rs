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

//! Validates the reader against the configs macOS actually installed.
//!
//! These skip (rather than fail) when no translation config asset is present,
//! so the suite still passes on Linux and on a Mac that has never opened the
//! Translation Languages pane.

use rlx_translate::assets::Assets;
use rlx_translate::pdec::PDecParams;
use rlx_translate::pipeline::TranslationPlan;
use rlx_translate::quasar::{BlockKind, QuasarConfig, TASK_MT_APP};

fn assets_or_skip() -> Option<Assets> {
    let a = Assets::discover();
    if a.configs.is_empty() {
        eprintln!("skipping: no translation configs installed");
        return None;
    }
    Some(a)
}

#[test]
fn every_installed_config_parses() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut parsed = 0usize;
    for cfg in &assets.configs {
        let c = QuasarConfig::load(&cfg.path)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e:#}", cfg.path.display()));
        assert!(
            !c.decoders.is_empty(),
            "{} has no decoders",
            cfg.path.display()
        );
        assert!(
            c.model_info.version.starts_with("MT-"),
            "{} has an unexpected model version {:?}",
            cfg.path.display(),
            c.model_info.version
        );
        parsed += 1;
    }
    assert!(parsed > 0);
    eprintln!("parsed {parsed} shipped configs");
}

#[test]
fn every_pair_graph_is_acyclic_and_fully_wired() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut graphs = 0usize;
    for cfg in &assets.configs {
        let c = QuasarConfig::load(&cfg.path).expect("config parses");
        for (task, decoder) in &c.decoders {
            for (pair, graph) in &decoder.graphs {
                graph
                    .topo_order()
                    .unwrap_or_else(|e| panic!("{} · {task} · {pair}: {e:#}", cfg.path.display()));
                graphs += 1;
            }
        }
    }
    eprintln!("topologically ordered {graphs} pair graphs");
}

#[test]
fn every_graph_stage_resolves_to_a_defined_block_or_inline_type() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    for cfg in &assets.configs {
        let c = QuasarConfig::load(&cfg.path).expect("config parses");
        for (task, decoder) in &c.decoders {
            for (pair, graph) in &decoder.graphs {
                for node in graph.nodes.values() {
                    if let Some(name) = &node.block {
                        assert!(
                            decoder.blocks.contains_key(name),
                            "{} · {task} · {pair}: stage {:?} names undefined block {name:?}",
                            cfg.path.display(),
                            node.name
                        );
                    } else {
                        // `graph-output` is the terminal sink: it names the
                        // stage holding the result and runs nothing itself.
                        assert!(
                            node.block_type.is_some() || node.is_output(),
                            "{} · {task} · {pair}: stage {:?} has neither block nor block-type",
                            cfg.path.display(),
                            node.name
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn no_block_type_falls_through_to_other() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut unknown = std::collections::BTreeSet::new();
    for cfg in &assets.configs {
        let c = QuasarConfig::load(&cfg.path).expect("config parses");
        for decoder in c.decoders.values() {
            for b in decoder.blocks.values() {
                if let BlockKind::Other(name) = &b.kind {
                    unknown.insert(name.clone());
                }
            }
            // Graph stages may declare a block type inline instead of naming a
            // definition — MergerBlock, NullBlock and friends only ever appear
            // this way, so scanning `block-definitions` alone misses them.
            for graph in decoder.graphs.values() {
                for node in graph.nodes.values() {
                    if let Some(BlockKind::Other(name)) = &node.block_type {
                        unknown.insert(name.clone());
                    }
                }
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "unmodelled block types shipped by macOS: {unknown:?}"
    );
}

#[test]
fn every_translator_block_yields_usable_decode_params() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut seen = 0usize;
    for cfg in &assets.configs {
        let c = QuasarConfig::load(&cfg.path).expect("config parses");
        for decoder in c.decoders.values() {
            for b in decoder.blocks.values() {
                if b.kind != BlockKind::PDecTranslator {
                    continue;
                }
                let p = PDecParams::from_block(b)
                    .unwrap_or_else(|e| panic!("{}: {e:#}", cfg.path.display()));
                assert!(p.beam >= 1, "{}: beam {} < 1", cfg.path.display(), p.beam);
                assert!(
                    !p.model_file.is_empty(),
                    "{}: translator has no model-file",
                    cfg.path.display()
                );
                assert_eq!(
                    p.model_type,
                    "espresso",
                    "{}: unexpected model-type",
                    cfg.path.display()
                );
                assert!(
                    !p.target_tokens().is_empty(),
                    "{}: translator has no target control token",
                    cfg.path.display()
                );
                // The budget must always land inside the configured bounds.
                for src_len in [0usize, 1, 17, 500] {
                    let b = p.length_budget(src_len);
                    assert!(
                        b <= p.max_seq_length && b >= p.max_seq_length_floor.min(p.max_seq_length),
                        "{}: budget {b} out of range for src_len {src_len}",
                        cfg.path.display()
                    );
                }
                seen += 1;
            }
        }
    }
    assert!(seen > 0, "no PDecTranslatorBlock found in any config");
    eprintln!("checked {seen} translator blocks");
}

#[test]
fn plans_build_for_every_installed_pair() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut built = 0usize;
    for pair in assets.pairs() {
        let config = rlx_translate::assets::load_config(&assets, &pair).expect("config loads");
        // A config file is named for one direction but defines both.
        for p in [pair.clone(), pair.reversed()] {
            if !config
                .decoders
                .get(TASK_MT_APP)
                .is_some_and(|d| d.graphs.contains_key(&p))
            {
                continue;
            }
            let plan = TranslationPlan::build(&config, TASK_MT_APP, &p)
                .unwrap_or_else(|e| panic!("plan for {p}: {e:#}"))
                .with_asset_status(&assets);
            assert!(!plan.stages.is_empty(), "{p}: empty plan");
            built += 1;
        }
    }
    assert!(built > 0);
    eprintln!("built {built} pipeline plans");
}

#[test]
fn model_files_are_reported_missing_until_a_pair_is_installed() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let pair = assets
        .pairs()
        .into_iter()
        .next()
        .expect("at least one pair");
    let config = rlx_translate::assets::load_config(&assets, &pair).expect("config loads");
    let avail = assets
        .availability(&config, TASK_MT_APP, &pair)
        .expect("availability computes");
    // Whatever the machine's state, the two views must agree.
    assert_eq!(
        avail.is_complete(),
        avail.missing.is_empty(),
        "is_complete disagrees with the missing set"
    );
    for rel in &avail.missing {
        assert!(
            assets.resolve(rel).is_none(),
            "{rel} is reported missing but resolves"
        );
    }
    for (rel, path) in &avail.present {
        assert!(
            path.exists(),
            "{rel} is reported present but {path:?} is gone"
        );
    }
    eprintln!(
        "{pair}: {} present, {} missing, missing assets {:?}",
        avail.present.len(),
        avail.missing.len(),
        avail.missing_assets()
    );
}

#[test]
fn installed_pairs_resolve_every_file_they_reference() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut complete = Vec::new();
    for pair in assets.pairs() {
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let Ok(avail) = assets.availability(&config, TASK_MT_APP, &pair) else {
            continue;
        };
        if avail.is_complete() && !avail.present.is_empty() {
            complete.push(pair.clone());
            // A complete pair must select a config whose model asset exists —
            // the variant number alone is not enough.
            for path in avail.present.values() {
                assert!(path.exists(), "{pair}: {path:?} vanished");
            }
        }
    }
    if complete.is_empty() {
        eprintln!("no language pair fully installed; skipping deeper checks");
        return;
    }
    eprintln!("fully installed pairs: {complete:?}");
}

#[test]
fn shipped_manifest_parses_and_describes_the_nmt() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut checked = 0usize;
    for pair in assets.pairs() {
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let Ok(plan) = TranslationPlan::build(&config, TASK_MT_APP, &pair) else {
            continue;
        };
        let Some(pdec) = &plan.pdec else { continue };
        let Some(path) = assets.resolve(&pdec.model_file) else {
            continue;
        };
        let m = rlx_translate::espresso::Manifest::load(&path)
            .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));

        assert_eq!(m.str("BEspressoEngine"), Some("CPU"));
        assert!(
            m.str("EncoderGraph")
                .is_some_and(|g| g.ends_with(".espresso.net")),
            "encoder graph should be an espresso net"
        );
        assert!(
            m.decoder_layers() > 0,
            "decoder layer count must be inferable"
        );
        assert!(
            !m.languages().is_empty(),
            "manifest must name target languages"
        );
        // The target language of this pair must have a decoder graph.
        let lang = pdec
            .target_locale
            .split('_')
            .next()
            .expect("locale has a language part");
        assert!(
            m.languages().contains(&lang),
            "{}: no decoder graph for {lang} (has {:?})",
            path.display(),
            m.languages()
        );
        // Handover and state tensors must agree on the layer count.
        assert_eq!(
            m.csv("HandoverStrings").len(),
            m.decoder_layers() * 2,
            "expected a key and a value transpose per decoder layer"
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("no pyespresso.mdl.bin installed; skipping");
        return;
    }
    eprintln!("validated {checked} shipped manifests");
}

#[test]
fn shipped_sed_scripts_parse_and_normalize() {
    let Some(assets) = assets_or_skip() else {
        return;
    };
    let mut checked = 0usize;
    for name in ["normalizer.pat", "tokenizer.pat"] {
        for root in &assets.roots {
            let p = root.join("MT").join(name);
            if !p.exists() {
                continue;
            }
            let n = rlx_translate::Normalizer::load(&p)
                .unwrap_or_else(|e| panic!("{}: {e:#}", p.display()));
            assert!(!n.is_empty(), "{}: no substitutions", p.display());
            // Normalizing must be total — no panics on awkward input.
            for probe in ["", "  ", "Hello, world!", "Ça va ? — Oui…", "a\tb"] {
                let _ = n.normalize(probe);
            }
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("no shipped .pat files installed; skipping");
        return;
    }
    eprintln!("validated {checked} shipped sed scripts");
}
