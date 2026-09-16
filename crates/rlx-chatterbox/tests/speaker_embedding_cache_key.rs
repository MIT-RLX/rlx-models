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

//! A reference clip's speaker embedding must not depend on what was embedded
//! before it.
//!
//! `speech_encoder` is always compiled at `seq = 100` and takes its real extent
//! from the reference clip's sample count, but the on-disk AOT key named only
//! `seq` — so clips of different durations built different graphs and then
//! shared one cache entry. The first voice cloned in a session served its
//! compiled graph to every later one. Against onnxruntime the first clip scored
//! cosine 1.00000000 and the next two 0.992 and 0.569, which is the kind of
//! wrong that looks like a mediocre model rather than a bug.
//!
//! Order-independence catches it without needing a reference runtime: embed two
//! clips of different lengths in one order, wipe the cache, embed them in the
//! other, and the results must agree.

use rlx_chatterbox::native::NativeChatterBox;
use std::path::Path;

const SR: u32 = 24_000;

fn clip(samples: usize, hz: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| 0.3 * (i as f32 * hz * std::f32::consts::TAU / SR as f32).sin())
        .collect()
}

fn wipe_aot_cache() {
    let td = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&td) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        if name.to_string_lossy().starts_with("rlx_chatterbox_aot_") {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb).max(1e-9)
}

#[test]
fn embedding_does_not_depend_on_what_was_compiled_first() {
    // Tests run with the crate dir as cwd; the weights live at the workspace root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let dir = root.join("weights/tts/chatterbox");
    if !dir.join("onnx/speech_encoder.onnx").exists() {
        eprintln!("no ChatterBox weights at {} — skipping", dir.display());
        return;
    }
    let dir = dir.as_path();
    // Two different durations, both above the 2 s floor the encoder pads to.
    let short = clip(48_000, 180.0);
    let long = clip(62_400, 180.0);

    wipe_aot_cache();
    let cb = NativeChatterBox::load(dir).expect("load");
    let short_first = cb.speaker_embedding(&short, SR).expect("short");
    let long_second = cb.speaker_embedding(&long, SR).expect("long");
    drop(cb);

    wipe_aot_cache();
    let cb = NativeChatterBox::load(dir).expect("reload");
    let long_first = cb.speaker_embedding(&long, SR).expect("long");
    let short_second = cb.speaker_embedding(&short, SR).expect("short");

    let cs = cosine(&short_first, &short_second);
    let cl = cosine(&long_first, &long_second);
    assert!(
        cs > 0.9999,
        "the short clip's embedding changed with compile order (cosine {cs:.6}) \
         — a graph compiled for a different reference length was reused"
    );
    assert!(
        cl > 0.9999,
        "the long clip's embedding changed with compile order (cosine {cl:.6})"
    );
}
