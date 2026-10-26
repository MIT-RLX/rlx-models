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

//! The paged runner against the resident whole-model graph.
//!
//! The two share their sublayer emitters but nothing else: the runner splits
//! every sparse layer around a host-side expert gather, keeps the key/value
//! cache on the host (a ring for sliding layers), and right-pads chunks. Every
//! one of those is a place to be silently wrong, and this is where it shows.
//! The prompt is longer than the tiny model's window (4), so sliding layers
//! evict, and the chunk sizes straddle it.

use rlx_core::flow_util::compile_built;
use rlx_flint::synth::{SynthDtype, tiny_config, tiny_config_json, write_checkpoint};
use rlx_flint::{Checkpoint, CheckpointF32, FlintRunner, RunnerOptions, build_flint_text_flow};
use rlx_runtime::Device;
use std::sync::Arc;

const IDS: [u32; 13] = [3, 17, 42, 5, 88, 13, 21, 60, 7, 33, 2, 71, 50];

fn resident_logits(dir: &std::path::Path) -> Vec<f32> {
    let cfg = tiny_config();
    let ckpt = Checkpoint::open(dir).unwrap();
    // Always f32: on a bf16 checkpoint the runner goes packed, and this is
    // what it is compared against.
    let built = build_flint_text_flow(&cfg, &mut CheckpointF32(&ckpt), IDS.len()).unwrap();
    let mut g = compile_built(built, Device::Cpu).unwrap();
    let ids: Vec<f32> = IDS.iter().map(|&t| t as f32).collect();
    g.run(&[("input_ids", &ids)]).pop().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn runner(dir: &std::path::Path, chunk: usize) -> FlintRunner {
    let ckpt = Arc::new(Checkpoint::open(dir).unwrap());
    let opts = RunnerOptions::new(Device::Cpu)
        .max_context(32)
        .prefill_chunk(chunk);
    FlintRunner::from_checkpoint(tiny_config(), ckpt, opts).unwrap()
}

#[test]
fn paged_runner_matches_the_resident_graph_at_every_chunk_size() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path(), &tiny_config_json(), SynthDtype::F32, 3, 7).unwrap();
    let want = resident_logits(tmp.path());
    assert!(want.iter().all(|x| x.is_finite()));

    // 1 = token by token through the decode graphs; 3 < window; 5 > window
    // with a padded last chunk; 16 = the whole prompt in one padded chunk.
    for chunk in [1, 3, 5, 16] {
        let mut r = runner(tmp.path(), chunk);
        let got = r.all_logits(&IDS).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-4, "chunk {chunk}: max |paged - resident| = {d}");
    }
}

#[test]
fn decode_continues_a_chunked_prefill_exactly() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path(), &tiny_config_json(), SynthDtype::F32, 2, 11).unwrap();
    let want = resident_logits(tmp.path());
    let v = tiny_config().vocab_size;

    let mut r = runner(tmp.path(), 4);
    // Prefill the first 6 in chunks, then decode the rest one at a time.
    let mut got = r.extend(&IDS[..6]).unwrap();
    assert!(max_abs_diff(&got, &want[5 * v..6 * v]) < 1e-4);
    for (i, &t) in IDS.iter().enumerate().skip(6) {
        got = r.extend(&[t]).unwrap();
        let d = max_abs_diff(&got, &want[i * v..(i + 1) * v]);
        assert!(d < 1e-4, "position {i}: {d}");
    }
    assert_eq!(r.position(), IDS.len());

    // A reset forgets the sequence: the same prompt gives the same answer.
    r.reset();
    let again = r.extend(&IDS[..6]).unwrap();
    assert!(max_abs_diff(&again, &want[5 * v..6 * v]) < 1e-4);
}

#[test]
fn a_bf16_checkpoint_runs_packed_and_matches_its_f32_upcast() {
    // On bf16 weights the runner keeps every projection and every paged expert
    // packed (`DequantMatMul` / `DequantGroupedMatMul { GgufBF16 }`). The
    // resident graph here reads the same bytes upcast to f32, so any
    // difference is the packed path's, not rounding.
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path(), &tiny_config_json(), SynthDtype::BF16, 4, 7).unwrap();
    let want = resident_logits(tmp.path());
    let mut r = runner(tmp.path(), 5);
    let got = r.all_logits(&IDS).unwrap();
    assert!(max_abs_diff(&got, &want) < 1e-4);
}

#[test]
fn context_overflow_is_refused_not_wrapped() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path(), &tiny_config_json(), SynthDtype::F32, 1, 3).unwrap();
    let ckpt = Arc::new(Checkpoint::open(tmp.path()).unwrap());
    let opts = RunnerOptions::new(Device::Cpu)
        .max_context(8)
        .prefill_chunk(1);
    let mut r = FlintRunner::from_checkpoint(tiny_config(), ckpt, opts).unwrap();
    r.extend(&IDS[..8]).unwrap();
    assert!(r.extend(&[1]).is_err());
}
