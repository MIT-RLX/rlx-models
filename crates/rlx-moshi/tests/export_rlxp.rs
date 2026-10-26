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

//! An exported Moshiko one-way (`weights/tts/moshiko-bf16`, Mimi from
//! `weights/tts/kyutai-mimi`), opened by rlx alone, speaks as the crate's
//! runner does: greedy, its Mimi frames are the runner's; sampled with the
//! runner's seeds, its frames until a near tie. The runner is dropped before
//! the package opens: each holds the 7B model.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_moshi::config::MoshiVariant;
use rlx_moshi::export::export_with;
use rlx_moshi::session::{GenerationConfig, MoshiSession};
use serde_json::json;

#[test]
fn an_exported_moshi_speaks_as_the_runner_does() {
    let w = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let (dir, mimi) = (w.join("moshiko-bf16"), w.join("kyutai-mimi"));
    if !dir.join("model.safetensors").is_file() || !mimi.join("model.safetensors").is_file() {
        eprintln!("skip: no Moshiko checkpoint and Mimi");
        return;
    }
    rlx_text::install_model_tokenizer();
    let text = "Hello, how are you today?";
    let steps = 12;
    let sampled_cfg = GenerationConfig {
        max_steps: steps,
        ..GenerationConfig::default()
    };
    let greedy_cfg = GenerationConfig {
        text_temperature: 0.0,
        audio_temperature: 0.0,
        ..sampled_cfg.clone()
    };
    let (greedy, sampled) = {
        let t = std::time::Instant::now();
        let mut s = MoshiSession::open(&dir, &mimi, MoshiVariant::MoshikoOneWay).unwrap();
        eprintln!("runner open: {:.1} s", t.elapsed().as_secs_f64());
        let g = s.generate_one_way(text, &greedy_cfg).unwrap().audio_frames;
        let r = s.generate_one_way(text, &sampled_cfg).unwrap().audio_frames;
        (g, r)
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("moshi.rlxp");
    eprintln!(
        "{}",
        export_with(&dir, &mimi, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut run = |extra: serde_json::Value| -> Vec<Vec<u32>> {
        let mut body = json!({"entry": "speak", "text": text, "max_steps": steps});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let t = std::time::Instant::now();
        let got = m.run(&[], &body).unwrap();
        eprintln!("package run: {:.1} s", t.elapsed().as_secs_f64());
        let books = got[0].shape[1];
        got[0]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect::<Vec<_>>()
            .chunks(books)
            .map(|c| c.to_vec())
            .collect()
    };
    let same = |a: &[Vec<u32>], b: &[Vec<u32>]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let frames = run(json!({"text_temperature": 0.0, "audio_temperature": 0.0}));
    eprintln!(
        "greedy: {} of {} frames the runner's ({} from the package)",
        same(&frames, &greedy),
        greedy.len(),
        frames.len()
    );
    assert_eq!(frames, greedy);
    let frames = run(json!({}));
    let n = same(&frames, &sampled);
    eprintln!(
        "sampled: {n} of {} frames the runner's ({} from the package)",
        sampled.len(),
        frames.len()
    );
    assert!(n >= 4, "only {n} frames the runner's");
}
