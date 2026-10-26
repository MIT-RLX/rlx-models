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

//! A GGUF handed over as bytes (`Qwen35RunnerBuilder::gguf_loader`, the
//! browser's only option) must generate exactly what the same file does when
//! opened by path.
//!
//! ```sh
//! QWEN35_GGUF_PATH=weights/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p rlx-qwen35 --release --test gguf_in_memory -- --nocapture
//! ```

use rlx_core::weight_loader::GgufLoader;
use rlx_qwen35::{Qwen35Runner, StageInput, StageOutput};
use rlx_runtime::Device;

fn test_device() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

#[test]
fn in_memory_gguf_matches_path() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    let prompt = [760u32, 3712, 314, 279, 17933, 3379, 11690, 303];
    let n = 12;

    let by_path = Qwen35Runner::builder()
        .weights(&path)
        .device(test_device())
        .max_seq(prompt.len() + n)
        .skip_auto_mmproj(true)
        .build()
        .expect("build from path")
        .generate(&prompt, n, |_| true)
        .expect("generate from path");

    let bytes = std::fs::read(&path).expect("read gguf");
    let file = rlx_gguf::GgufFile::from_bytes(bytes).expect("parse");
    let from_memory = Qwen35Runner::builder()
        .gguf_loader(GgufLoader::from_gguf_file(file))
        .device(test_device())
        .max_seq(prompt.len() + n)
        .skip_auto_mmproj(true)
        .build()
        .expect("build from memory")
        .generate(&prompt, n, |_| true)
        .expect("generate from memory");

    eprintln!("path:   {by_path:?}\nmemory: {from_memory:?}");
    assert_eq!(by_path, from_memory);
}

/// The step-at-a-time loop the browser awaits (`generate_async`) has a
/// blocking twin, `generate_stepwise`; it must agree with `generate`.
#[test]
fn stepwise_matches_generate() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    let prompt = [760u32, 3712, 314, 279, 17933, 3379, 11690, 303];
    let n = 12;
    let build = || {
        Qwen35Runner::builder()
            .weights(&path)
            .device(test_device())
            .max_seq(prompt.len() + n)
            .skip_auto_mmproj(true)
            .build()
            .expect("build")
    };
    let want = build().generate(&prompt, n, |_| true).expect("generate");
    let got = build()
        .generate_stepwise(&prompt, n, |_| true)
        .expect("generate_stepwise");
    eprintln!("generate: {want:?}\nstepwise: {got:?}");
    assert_eq!(want, got);
}

/// The browser's configuration (`rlx-models-web`'s `Qwen35Generator`): an
/// in-memory GGUF, host-gathered embeddings, a 128-wide static prefill inside
/// a 256 context, no warm-up, step-at-a-time decode.
#[test]
fn browser_configuration_matches_generate() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
    let prompt = [
        248045u32, 846, 198, 7734, 264, 2716, 13901, 883, 279, 9117, 13, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ];
    let n = 12;
    let want = Qwen35Runner::builder()
        .weights(&path)
        .device(test_device())
        .max_seq(256)
        .skip_auto_mmproj(true)
        .build()
        .expect("build")
        .generate(&prompt, n, |_| true)
        .expect("generate");
    let bytes = std::fs::read(&path).expect("read gguf");
    let file = rlx_gguf::GgufFile::from_bytes(bytes).expect("parse");
    let got = Qwen35Runner::builder()
        .gguf_loader(GgufLoader::from_gguf_file(file))
        .device(test_device())
        .max_seq(256)
        .prefill_seq(128)
        .skip_auto_mmproj(true)
        .skip_warm(true)
        .build()
        .expect("build browser config")
        .generate_stepwise(&prompt, n, |_| true)
        .expect("generate_stepwise");
    eprintln!("generate: {want:?}\nbrowser:  {got:?}");
    assert_eq!(want, got);
}

/// Run `prompt` through a `count`-stage pipeline of runners, each holding only
/// its own layers (`pipeline_stage`), handing hidden rows from one to the next.
/// `ranged`: each stage gets the header plus only the byte ranges it reads
/// (what a browser worker fetches with HTTP `Range`), not the whole file.
fn pipeline_generate(path: &str, prompt: &[u32], n: usize, count: usize, ranged: bool) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read gguf");
    let mut stages: Vec<Qwen35Runner> = (0..count)
        .map(|i| {
            let file = if ranged {
                let header = rlx_gguf::GgufFile::header_from_bytes(&bytes).expect("header");
                let keep = rlx_qwen35::pipeline_stage_keeps(&header, i, count).expect("keeps");
                let chunks: Vec<(u64, Vec<u8>)> = header
                    .tensor_ranges(&keep, 1 << 20)
                    .expect("ranges")
                    .into_iter()
                    .map(|r| (r.start, bytes[r.start as usize..r.end as usize].to_vec()))
                    .collect();
                let fetched: usize = chunks.iter().map(|(_, b)| b.len()).sum();
                eprintln!(
                    "stage {i}/{count}: {} ranges, {fetched} bytes",
                    chunks.len()
                );
                header.with_tensor_chunks(&keep, &chunks).expect("assemble")
            } else {
                rlx_gguf::GgufFile::from_bytes(bytes.clone()).expect("parse")
            };
            Qwen35Runner::builder()
                .gguf_loader(GgufLoader::from_gguf_file(file))
                .device(test_device())
                .max_seq(256)
                .prefill_seq(128)
                .skip_auto_mmproj(true)
                .skip_warm(true)
                .pipeline_stage(i, count)
                .build()
                .expect("build stage")
        })
        .collect();
    let mut step = |prefill: bool, first: StageInput<'_>| -> u32 {
        let mut hidden: Vec<f32> = Vec::new();
        for (i, stage) in stages.iter_mut().enumerate() {
            let input = if i == 0 {
                first
            } else {
                StageInput::Hidden(&hidden)
            };
            let out = if prefill {
                stage.stage_prefill(input)
            } else {
                stage.stage_decode(input)
            }
            .expect("stage run");
            match out {
                StageOutput::Hidden(h) => hidden = h,
                StageOutput::Token(t) => {
                    assert_eq!(i + 1, count, "only the last stage yields a token");
                    return t;
                }
                StageOutput::Tokens(_) => panic!("a one-row step yields one token"),
            }
        }
        panic!("the last stage yielded no token");
    };
    let mut out = vec![step(true, StageInput::Tokens(prompt))];
    while out.len() < n {
        let t = *out.last().unwrap();
        out.push(step(false, StageInput::Tokens(&[t])));
    }
    out
}

/// A model split into 2, 3 and 6 pipeline stages generates exactly what the
/// whole model does — the browser's multi-worker split.
#[test]
fn pipeline_stages_match_generate() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
    let prompt = [
        248045u32, 846, 198, 7734, 264, 2716, 13901, 883, 279, 9117, 13, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ];
    let n = 12;
    let want = Qwen35Runner::builder()
        .weights(&path)
        .device(test_device())
        .max_seq(256)
        .skip_auto_mmproj(true)
        .build()
        .expect("build")
        .generate(&prompt, n, |_| true)
        .expect("generate");
    for (count, ranged) in [(1, false), (2, false), (3, false), (6, false), (3, true)] {
        let got = pipeline_generate(&path, &prompt, n, count, ranged);
        eprintln!(
            "{count} stage(s){}: {got:?}",
            if ranged { ", ranged" } else { "" }
        );
        assert_eq!(want, got, "{count}-stage pipeline (ranged: {ranged})");
    }
}

/// Multi-row decode steps (`decode_rows`, the browser's "Speculate") generate
/// exactly what one-token decode does: one row per step, the prompt in 4-row
/// chunks, and a speculative check of guessed tokens — guesses that all hold,
/// and guesses that fail part-way and are undone (`stage_rows_undo`). On
/// wgpu the run crosses a context bucket, where the device-held state moves.
#[test]
fn decode_rows_match_generate() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
    let device = test_device();
    let n = if device == Device::Cpu { 40 } else { 300 };
    let prompt = [
        248045u32, 846, 198, 7734, 264, 2716, 13901, 883, 279, 9117, 13, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ];
    let build = |rows: usize| {
        let file = rlx_gguf::GgufFile::from_bytes(std::fs::read(&path).expect("read gguf"))
            .expect("parse");
        Qwen35Runner::builder()
            .gguf_loader(GgufLoader::from_gguf_file(file))
            .device(device)
            .max_seq(2048)
            .prefill_seq(128)
            .skip_auto_mmproj(true)
            .skip_warm(true)
            .decode_rows(rows)
            .build()
            .expect("build")
    };
    let tokens = |o: StageOutput| match o {
        StageOutput::Tokens(t) => t,
        StageOutput::Token(t) => vec![t],
        o => panic!("unexpected stage output {o:?}"),
    };
    let want = build(1)
        .generate_stepwise(&prompt, n, |_| true)
        .expect("generate_stepwise");
    let mut r = build(4);

    let mut got = tokens(r.stage_prefill(StageInput::Tokens(&prompt)).unwrap());
    while got.len() < n {
        let last = *got.last().unwrap();
        got.extend(tokens(
            r.stage_rows(StageInput::Tokens(&[last]), false, None)
                .unwrap(),
        ));
    }
    assert_eq!(want, got, "one row per step");

    let mut tok = tokens(r.stage_prefill(StageInput::Tokens(&prompt[..7])).unwrap());
    for chunk in prompt[7..].chunks(4) {
        tok = tokens(
            r.stage_rows(StageInput::Tokens(chunk), false, None)
                .unwrap(),
        );
    }
    assert_eq!(tok.last(), want.first(), "prompt in 4-row chunks");

    for corrupt in [false, true] {
        let mut got = tokens(r.stage_prefill(StageInput::Tokens(&prompt)).unwrap());
        let mut undos = 0;
        while got.len() < n {
            let at = got.len();
            let mut guesses: Vec<u32> = want[at..(at + 3).min(n)].to_vec();
            if corrupt && at % 3 == 0 && guesses.len() > 1 {
                guesses[1] ^= 1;
            }
            let mut rows = vec![*got.last().unwrap()];
            rows.extend(&guesses);
            let next = tokens(
                r.stage_rows(StageInput::Tokens(&rows), true, Some(&guesses))
                    .unwrap(),
            );
            let mut j = 0;
            while j < guesses.len() && guesses[j] == next[j] {
                j += 1;
            }
            if j < guesses.len() {
                r.stage_rows_undo(j + 1).unwrap();
                undos += 1;
            }
            got.extend(&guesses[..j]);
            got.push(next[j]);
        }
        got.truncate(n);
        assert_eq!(corrupt, undos > 0);
        assert_eq!(want, got, "speculative check (failing guesses: {corrupt})");
    }
}

/// The last pipeline stage's `stage_scores` after a prefill equal the whole
/// model's `predict_logits` at those ids — what the browser's classifier reads.
#[test]
fn pipeline_stage_scores_match_predict_logits() {
    let Ok(path) = std::env::var("QWEN35_GGUF_PATH") else {
        eprintln!("skip: set QWEN35_GGUF_PATH");
        return;
    };
    rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
    let prompt = [
        248045u32, 846, 198, 7734, 264, 2716, 13901, 883, 279, 9117, 13, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ];
    let ids = [15u32, 1000, 9117, 13901, 248045];
    let want = Qwen35Runner::builder()
        .weights(&path)
        .device(test_device())
        .max_seq(256)
        .skip_auto_mmproj(true)
        .build()
        .expect("build")
        .predict_logits(&prompt)
        .expect("predict_logits")
        .logits;
    let bytes = std::fs::read(&path).expect("read gguf");
    let count = 3;
    let mut hidden: Vec<f32> = Vec::new();
    let mut got = Vec::new();
    for i in 0..count {
        let file = rlx_gguf::GgufFile::from_bytes(bytes.clone()).expect("parse");
        let mut stage = Qwen35Runner::builder()
            .gguf_loader(GgufLoader::from_gguf_file(file))
            .device(test_device())
            .max_seq(256)
            .prefill_seq(128)
            .skip_auto_mmproj(true)
            .skip_warm(true)
            .pipeline_stage(i, count)
            .build()
            .expect("build stage");
        let input = if i == 0 {
            StageInput::Tokens(&prompt)
        } else {
            StageInput::Hidden(&hidden)
        };
        match stage.stage_prefill(input).expect("prefill") {
            StageOutput::Hidden(h) => hidden = h,
            StageOutput::Token(_) => got = stage.stage_scores(&ids).expect("scores"),
            StageOutput::Tokens(_) => panic!("a prefill yields one token"),
        }
    }
    let spread = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (k, &id) in ids.iter().enumerate() {
        let (w, g) = (want[id as usize], got[k]);
        eprintln!("token {id}: whole {w:.5} stage {g:.5}");
        assert!(
            (w - g).abs() <= 1e-3 * spread.max(1.0),
            "token {id}: whole-model {w} vs stage_scores {g}"
        );
    }
}
