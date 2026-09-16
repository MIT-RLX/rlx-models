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

//! Loads every Espresso graph macOS installed and checks the reader against
//! them. Skips when no language pack is installed.

use rlx_translate::assets::Assets;
use rlx_translate::espresso::Manifest;
use rlx_translate::net::Graph;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Every layer type this port has to implement to run the NMT.
const KNOWN_KINDS: &[&str] = &[
    "quantized_gather",
    "elementwise",
    "dynamic_quantize",
    "inner_product",
    "dynamic_dequantize",
    "reshape",
    "transpose",
    "batch_matmul",
    "softmax",
    "instancenorm_1d",
    "copy",
];

/// Locates an installed manifest and the directories holding its graphs.
fn installed() -> Option<(Manifest, Vec<PathBuf>)> {
    let assets = Assets::discover();
    let manifest_path = assets
        .roots
        .iter()
        .map(|r| r.join("MT").join("pyespresso.mdl.bin"))
        .find(|p| p.exists())?;
    let m = Manifest::load(&manifest_path).expect("shipped manifest parses");
    let dirs: Vec<PathBuf> = assets
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.is_dir())
        .collect();
    Some((m, dirs))
}

fn find_graph(dirs: &[PathBuf], net_file: &str) -> Option<Graph> {
    for d in dirs {
        if d.join(net_file).exists() {
            return Some(
                Graph::load(d, net_file).unwrap_or_else(|e| panic!("loading {net_file}: {e:#}")),
            );
        }
    }
    None
}

#[test]
fn every_installed_graph_loads_with_matching_weights() {
    let Some((m, dirs)) = installed() else {
        eprintln!("skipping: no MT assets installed");
        return;
    };
    let mut loaded = 0usize;
    let mut kinds: BTreeSet<String> = BTreeSet::new();
    for net_file in m.graph_files() {
        let Some(g) = find_graph(&dirs, &net_file) else {
            continue;
        };
        assert!(!g.layers.is_empty(), "{net_file}: no layers");
        assert!(!g.weights.is_empty(), "{net_file}: no weight blobs");
        // Every weight a layer names must exist in the container.
        for l in &g.layers {
            if let Some(w) = l.attrs.get("weights").and_then(|v| v.as_object()) {
                for (key, idx) in w {
                    let idx = idx.as_u64().unwrap_or_else(|| {
                        panic!("{net_file}: layer {} weight {key} is not an index", l.name)
                    });
                    g.weights.raw(idx).unwrap_or_else(|e| {
                        panic!("{net_file}: layer {} weight {key}: {e:#}", l.name)
                    });
                }
            }
        }
        kinds.extend(g.layer_kinds().into_keys());
        loaded += 1;
    }
    assert!(loaded > 0, "no graphs found beside the manifest");
    eprintln!("loaded {loaded} graphs; layer kinds: {kinds:?}");
}

#[test]
fn no_graph_uses_a_layer_type_outside_the_known_set() {
    let Some((m, dirs)) = installed() else {
        eprintln!("skipping: no MT assets installed");
        return;
    };
    let mut unknown: BTreeSet<String> = BTreeSet::new();
    for net_file in m.graph_files() {
        let Some(g) = find_graph(&dirs, &net_file) else {
            continue;
        };
        for l in g.unsupported(KNOWN_KINDS) {
            unknown.insert(l.kind.clone());
        }
    }
    assert!(
        unknown.is_empty(),
        "unhandled Espresso layer types: {unknown:?}"
    );
}

#[test]
fn declared_shapes_cover_the_graph_and_are_self_consistent() {
    let Some((m, dirs)) = installed() else {
        eprintln!("skipping: no MT assets installed");
        return;
    };
    let mut checked = 0usize;
    for net_file in m.graph_files() {
        let Some(g) = find_graph(&dirs, &net_file) else {
            continue;
        };
        if g.shapes.is_empty() {
            continue;
        }
        for (blob, shape) in &g.shapes {
            assert!(
                shape.rank >= 1 && shape.rank <= 4,
                "{net_file}/{blob}: rank {}",
                shape.rank
            );
            assert_eq!(
                shape.dims().iter().product::<usize>(),
                shape.len(),
                "{net_file}/{blob}: dims disagree with element count"
            );
        }
        // The graph's terminal blob should have a declared shape.
        let out = g.output().expect("graph has an output");
        assert!(
            g.declared_shape(out).is_some(),
            "{net_file}: no declared shape for output {out:?}"
        );
        checked += 1;
    }
    assert!(checked > 0, "no shape files found");
    eprintln!("shape-checked {checked} graphs");
}

#[test]
fn graph_wiring_matches_the_manifest() {
    let Some((m, dirs)) = installed() else {
        eprintln!("skipping: no MT assets installed");
        return;
    };
    // The encoder consumes what the input net produces, and the handover net
    // consumes the encoder's output. Those names come from the manifest.
    let Some(enc) = m.str("EncoderGraph").and_then(|f| find_graph(&dirs, f)) else {
        eprintln!("skipping: encoder graph not installed");
        return;
    };
    let enc_in = enc.inputs();
    assert_eq!(
        enc_in.len(),
        1,
        "encoder should take exactly one input, got {enc_in:?}"
    );
    assert_eq!(
        Some(enc_in[0].as_str()),
        m.str("InputNetValuesStr"),
        "encoder input must be the manifest's InputNetValuesStr"
    );
    assert_eq!(
        Some(enc.output().expect("output")),
        m.str("EncoderValuesStr"),
        "encoder output must be the manifest's EncoderValuesStr"
    );

    // The embedding graph takes the source tokens named by the manifest.
    if let Some(emb) = m.str("EmbeddingGraph").and_then(|f| find_graph(&dirs, f)) {
        let ins = emb.inputs();
        assert!(
            ins.iter()
                .any(|i| Some(i.as_str()) == m.str("SourceInputStr")),
            "embedding inputs {ins:?} should include SourceInputStr"
        );
    }
    eprintln!(
        "encoder: {:?} -> {:?}",
        enc_in,
        enc.output().expect("output")
    );
}

#[test]
fn decoder_state_and_handover_tensors_exist_in_the_graphs() {
    let Some((m, dirs)) = installed() else {
        eprintln!("skipping: no MT assets installed");
        return;
    };
    let states = m.csv("StateStrings");
    let handovers = m.csv("HandoverStrings");
    assert!(!states.is_empty() && !handovers.is_empty());

    let mut checked = 0usize;
    for (lang, net_file) in m.lang_graphs.get("DecoderLangGraph").into_iter().flatten() {
        let Some(g) = find_graph(&dirs, net_file) else {
            continue;
        };
        let ins: BTreeSet<String> = g.inputs().into_iter().collect();
        for s in &states {
            assert!(ins.contains(s), "decoder {lang}: missing state input {s:?}");
        }
        for h in &handovers {
            assert!(
                ins.contains(h),
                "decoder {lang}: missing handover input {h:?}"
            );
        }
        // Each state must also be produced, as `<state>.next`.
        let marked: BTreeSet<String> = g.marked_outputs().into_iter().collect();
        for s in &states {
            let next = format!("{s}.next");
            assert!(
                marked.contains(&next),
                "decoder {lang}: {next:?} is not a marked output"
            );
        }
        assert_eq!(
            g.output().expect("output"),
            m.str("ScoresStr").expect("ScoresStr"),
            "decoder {lang}: final blob must be ScoresStr"
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipping: no decoder graphs installed");
        return;
    }
    eprintln!("checked {checked} decoder graphs against the manifest");
}
