//! Verifies that the three exported formats actually agree.
//!
//! Writing a file is not evidence that it holds the right numbers. These read
//! the exports back: GGUF against the safetensors it was staged from, and the
//! `.rlxp` against the directory it packed. Set `RLX_TRANSLATE_EXPORT` to the
//! output of `rlx-translate convert`; without it the tests skip.

use std::collections::HashMap;
use std::path::PathBuf;

fn export_dir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("RLX_TRANSLATE_EXPORT")
            .unwrap_or_else(|_| "/Volumes/FOUR/rlx-translate-export".to_string()),
    );
    d.join("index.json").is_file().then_some(d)
}

/// Every bundle in the index, as `(directory, gguf path)`.
fn bundles(root: &std::path::Path) -> Vec<(PathBuf, PathBuf)> {
    let Ok(raw) = std::fs::read_to_string(root.join("index.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    v["bundles"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|b| b["name"].as_str())
                .map(|n| (root.join(n), root.join(n).join(format!("{n}.gguf"))))
                .collect()
        })
        .unwrap_or_default()
}

/// Reads every f32 tensor of a safetensors file.
fn safetensors_of(path: &std::path::Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let Ok(raw) = std::fs::read(path) else {
        return HashMap::new();
    };
    let Ok(st) = safetensors::SafeTensors::deserialize(&raw) else {
        return HashMap::new();
    };
    st.tensors()
        .into_iter()
        .map(|(n, t)| {
            let v: Vec<f32> = t
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect();
            (n, (t.shape().to_vec(), v))
        })
        .collect()
}

#[test]
fn gguf_matches_the_safetensors_it_was_staged_from() {
    let Some(root) = export_dir() else {
        eprintln!("skipping: no export; run `rlx-translate convert <dir>` first");
        return;
    };
    let mut checked = 0usize;
    let mut worst = 0.0f32;
    for (dir, gguf_path) in bundles(&root) {
        let Ok(g) = rlx_gguf::GgufFile::from_path(&gguf_path) else {
            panic!("could not read {}", gguf_path.display());
        };
        assert!(
            g.metadata.contains_key("quasar.model_json"),
            "{} carries no graph description, so its tensors are unusable alone",
            gguf_path.display()
        );
        // One safetensors file per graph; the GGUF namespaces them `graph.tensor`.
        for ent in std::fs::read_dir(&dir).expect("read bundle dir") {
            let p = ent.expect("entry").path();
            if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let graph = p.file_stem().expect("stem").to_string_lossy().to_string();
            for (name, (shape, want)) in safetensors_of(&p) {
                let key = format!("{graph}.{name}");
                let t = g
                    .tensors
                    .get(&key)
                    .unwrap_or_else(|| panic!("{key} missing from {}", gguf_path.display()));
                assert_eq!(t.shape, shape, "{key}: shape differs");
                let bytes = g.tensor_bytes(t).expect("tensor bytes");
                // F16 storage: compare against the f32 source.
                assert_eq!(t.dtype, rlx_gguf::GgmlType::F16, "{key}: unexpected dtype");
                let got: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                assert_eq!(got.len(), want.len(), "{key}: element count differs");
                // F16 has ~11 significant bits; the weights it encodes came from
                // int8, whose step is ~1/127, so this bound is far looser than
                // the encoding error yet tight enough to catch a wrong tensor.
                for (a, b) in got.iter().zip(&want) {
                    let d = (a - b).abs() / b.abs().max(1e-3);
                    worst = worst.max(d);
                    assert!(d < 1e-2, "{key}: {a} vs {b}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no tensors were compared");
    eprintln!("  compared {checked} tensors; worst relative difference {worst:.2e}");
}

#[test]
fn the_rlxp_pack_holds_what_the_directory_held() {
    let Some(root) = export_dir() else {
        eprintln!("skipping: no export");
        return;
    };
    let all = bundles(&root);
    assert!(!all.is_empty(), "index.json names no bundles");
    // Every pack, not the first: a release ships all of them, and one of them
    // being short a file is exactly the kind of thing a sample would miss.
    let mut packed = 0usize;
    for (dir, _) in &all {
        let name = dir.file_name().expect("name").to_string_lossy().to_string();
        let pack = root.join(format!("{name}.rlxp"));
        assert!(pack.is_file(), "{} was not written", pack.display());

        let tmp = std::env::temp_dir().join(format!("rlx-tr-rlxp-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        rlx_assets::pack::materialize_rlxp(&pack, &tmp).expect("materialize");

        // Everything needed to run: the graph description, the tokenizer, and
        // at least one weight file in each of the two tensor formats.
        for want in ["model.json", "spm.model"] {
            assert!(
                tmp.join(want).is_file(),
                "{want} is missing from {}",
                pack.display()
            );
        }
        let names: Vec<String> = std::fs::read_dir(&tmp)
            .expect("read")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        for (ext, what) in [(".safetensors", "safetensors"), (".gguf", "gguf")] {
            assert!(
                names.iter().any(|n| n.ends_with(ext)),
                "{} has no {what}: {names:?}",
                pack.display()
            );
        }
        // The packed files are the same bytes as the loose ones, so a consumer
        // reading either gets the same weights.
        let st = names
            .iter()
            .find(|n| n.ends_with(".safetensors"))
            .expect("a safetensors entry");
        let from_pack = safetensors_of(&tmp.join(st));
        let from_dir = safetensors_of(&dir.join(st));
        assert!(!from_pack.is_empty(), "{st} in {name}.rlxp did not parse");
        assert_eq!(
            from_pack.len(),
            from_dir.len(),
            "{st}: {} tensors packed against {} loose",
            from_pack.len(),
            from_dir.len()
        );
        for (k, (shape, want)) in &from_dir {
            let (got_shape, got) = from_pack.get(k).unwrap_or_else(|| panic!("{k} missing"));
            assert_eq!(got_shape, shape, "{k}: shape differs inside the pack");
            assert_eq!(got, want, "{k}: bytes differ inside the pack");
        }
        eprintln!("  {name}.rlxp: {} entries, {st} verified", names.len());
        packed += 1;
        std::fs::remove_dir_all(&tmp).ok();
    }
    assert_eq!(packed, all.len(), "not every bundle was packed");
}

#[test]
fn every_shipped_config_was_converted() {
    let Some(root) = export_dir() else {
        eprintln!("skipping: no export");
        return;
    };
    let installed = rlx_translate::assets::Assets::discover().configs.len();
    let converted = std::fs::read_dir(root.join("configs"))
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    eprintln!("  {converted} configs exported, {installed} installed");
    assert_eq!(
        converted, installed,
        "the export dropped configurations silently"
    );
    assert!(root.join("quasar-configs.rlxp").is_file());
}

/// The descriptor has to be enough to rebuild the model.
///
/// `decoder_layers` was `null` in every bundle this crate ever exported: the
/// count is *inferred* from `StateStrings`, and the converter asked the
/// manifest for a `DecoderLayers` key that does not exist. A consumer reading
/// the descriptor could not have built the decoder.
#[test]
fn the_exported_descriptor_is_complete() {
    let Some(root) = export_dir() else {
        eprintln!("skipping: no export");
        return;
    };
    let mut checked = 0usize;
    for (dir, _) in bundles(&root) {
        let raw = std::fs::read_to_string(dir.join("model.json")).expect("model.json");
        let d: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        let name = d["bundle"].as_str().unwrap_or("?").to_string();
        for key in ["format", "bundle"] {
            assert!(d[key].as_str().is_some(), "{name}: {key} is not a string");
        }
        for key in ["decoder_layers", "state_width"] {
            let v = d[key].as_u64();
            assert!(v.is_some_and(|n| n > 0), "{name}: {key} is {:?}", d[key]);
        }
        assert!(
            d["languages"].as_array().is_some_and(|a| !a.is_empty()),
            "{name}: no languages"
        );
        assert!(
            d["graphs"].as_object().is_some_and(|g| !g.is_empty()),
            "{name}: no graphs"
        );
        checked += 1;
    }
    assert!(checked > 0);
    eprintln!("  {checked} descriptors complete");
}

/// Every shortlist beside the graphs must reach the export.
///
/// The converter built the filename from the language — `all-<lang>` — which is
/// right for nineteen of the twenty tables and wrong for `all-zh_TW`, the one
/// direction whose shortlist this port has already been caught mishandling
/// once. Its absence does not fail: it translates Traditional Chinese into
/// Simplified.
#[test]
fn every_installed_shortlist_was_exported() {
    let Some(root) = export_dir() else {
        eprintln!("skipping: no export");
        return;
    };
    let assets = rlx_translate::assets::Assets::discover();
    // What the machine has, by bundle directory name.
    let mut installed: HashMap<String, std::collections::BTreeSet<String>> = HashMap::new();
    for dir in assets.asset_dirs.values() {
        let Ok(rd) = std::fs::read_dir(dir.join("MT/shortlists")) else {
            continue;
        };
        let key = dir.join("MT").to_string_lossy().to_string();
        let e = installed.entry(key).or_default();
        for ent in rd.flatten() {
            if let Some(n) = ent.path().file_name().and_then(|n| n.to_str())
                && n.ends_with(".shortlist")
            {
                e.insert(n.to_string());
            }
        }
    }
    let all: std::collections::BTreeSet<String> = installed.values().flatten().cloned().collect();
    let mut exported: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (dir, _) in bundles(&root) {
        if let Ok(rd) = std::fs::read_dir(dir.join("shortlists")) {
            for ent in rd.flatten() {
                if let Some(n) = ent.path().file_name().and_then(|n| n.to_str()) {
                    exported.insert(n.to_string());
                }
            }
        }
    }
    if all.is_empty() {
        eprintln!("skipping: no shortlists installed");
        return;
    }
    let missing: Vec<&String> = all.difference(&exported).collect();
    assert!(
        missing.is_empty(),
        "installed but not exported: {missing:?}"
    );
    eprintln!("  {} shortlist tables, all exported", all.len());
}
