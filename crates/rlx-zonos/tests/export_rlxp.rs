// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! An exported Zonos v0.1 (`weights/tts/zonos`, DAC from
//! `weights/tts/parler-dac`), opened by rlx alone: the delay-pattern codes
//! `generate_codes_compiled` makes for the same phonemes (greedy and
//! sampled, until the runner's cached decode and the package's recomputed
//! context tip a near tie), and DAC's speech on the runner's codes.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_zonos::config::ZonosFileConfig;
use rlx_zonos::engine::{BackboneEngine, default_max_seq};
use rlx_zonos::export::export_rlxp;
use rlx_zonos::generate::{GenerateOpts, generate_codes_compiled};
use rlx_zonos::native::NativeZonos;
use rlx_zonos::phonemes::tokenize_phoneme_string;
use rlx_zonos::weights::WeightMap;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_zonos_speaks_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let (dir, dac) = (root.join("zonos"), root.join("parler-dac"));
    if !dir.join("model.safetensors").is_file() || !dac.is_dir() {
        eprintln!("skip: no Zonos checkpoint or DAC");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("zonos.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    let ids = tokenize_phoneme_string("ɡʊd mˈɔːɹnɪŋ ðə tˈiː ɪz ɹˈɛdi.");
    let cfg = ZonosFileConfig::load(dir.join("config.json")).unwrap();
    let w = WeightMap::load(dir.join("model.safetensors")).unwrap();
    let new = 24;
    let mut engine =
        BackboneEngine::open(&cfg, &w, Device::Cpu, default_max_seq(new, ids.len() + 8)).unwrap();
    let native = NativeZonos::open(&dir, &dac, Device::Cpu).unwrap();
    for greedy in [true, false] {
        let opts = GenerateOpts {
            max_new_tokens: new,
            greedy,
            ..GenerateOpts::default()
        };
        let want = generate_codes_compiled(&cfg, &w, &mut engine, &ids, &opts).unwrap();
        let got = m
            .run(
                &[],
                &json!({"entry": "speak", "phonemes": ids, "max_new_tokens": new,
                        "greedy": greedy, "seed": opts.seed}),
            )
            .unwrap();
        let t = got[0].shape[1];
        let flat: Vec<i64> = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect();
        let got: Vec<Vec<i64>> = flat.chunks(t).map(<[i64]>::to_vec).collect();
        let frames_same = (0..want[0].len().min(t))
            .take_while(|&f| (0..9).all(|q| got[q][f] == want[q][f]))
            .count();
        eprintln!(
            "greedy {greedy}: {frames_same} of {} frames the runner's ({t} from the package)",
            want[0].len()
        );
        assert!(frames_same >= 4, "{frames_same} frames the same");

        // DAC on the runner's codes.
        let want_audio = native.decode_codes(&want).unwrap();
        let audio = m.synthesize("-", &json!({"codes": want})).unwrap();
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.samples.len(), want_audio.len());
        let cos = cosine(&audio.samples, &want_audio);
        eprintln!(
            "greedy {greedy}: {} samples, cosine {cos}",
            want_audio.len()
        );
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
