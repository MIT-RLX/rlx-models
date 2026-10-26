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

//! Parity vs OpenBMB's `modeling_minicpm_sala.py` (run by
//! `scripts/minicpm_sala_reference.py`).
//!
//! ```sh
//! .venv-sala/bin/python -I scripts/minicpm_sala_reference.py tiny .cache/sala/tiny
//! RLX_SALA_TINY_DIR=$PWD/.cache/sala/tiny cargo test -p rlx-minicpm-sala --release --test reference_parity -- --nocapture
//!
//! .venv-sala/bin/python -I scripts/minicpm_sala_reference.py real weights/lm/minicpm-sala \
//!     .cache/sala/real4 --layers 4 --dtype bfloat16 --widen   # --layers 0: all 32
//! RLX_SALA_DIR=$PWD/weights/lm/minicpm-sala RLX_SALA_REF=$PWD/.cache/sala/real4 \
//!     cargo test -p rlx-minicpm-sala --release --test reference_parity real -- --nocapture
//! ```

use rlx_minicpm_sala::SalaRunner;
use rlx_qwen3::sampling::SampleOpts;
use rlx_runtime::Device;

fn test_device() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

struct Reference {
    prompt: Vec<u32>,
    last_logits: Vec<f32>,
    greedy: Vec<u32>,
    margins: Vec<f64>,
    layers: Option<usize>,
}

fn load(path: &str) -> Reference {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read reference")).unwrap();
    let u = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    Reference {
        prompt: u("prompt_ids"),
        last_logits: v["last_logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect(),
        greedy: u("greedy"),
        margins: v["margins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
        layers: v.get("layers").and_then(|x| x.as_u64()).map(|n| n as usize),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Prefill logits within `tol`, then greedy tokens equal up to a near-tie.
fn check(runner: &mut SalaRunner, r: &Reference, tol: f32, margin: f64) {
    let logits = runner.feed(&r.prompt).unwrap();
    let d = max_abs(&logits, &r.last_logits);
    let scale = r.last_logits.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!("prefill logits: max |rlx - ref| = {d:.3e} (max |ref| {scale:.2})");
    assert!(d < tol, "prefill logits differ by {d}");
    runner.reset();
    let got = runner
        .generate(&r.prompt, r.greedy.len(), SampleOpts::greedy(), |_| true)
        .unwrap();
    eprintln!("rlx greedy: {got:?}\nref greedy: {:?}", r.greedy);
    if let Some(i) = got.iter().zip(&r.greedy).position(|(a, b)| a != b) {
        eprintln!(
            "first divergence at step {i}: reference margin {:.4}",
            r.margins[i]
        );
        assert!(
            r.margins[i] < margin,
            "greedy diverged at step {i}, margin {}",
            r.margins[i]
        );
    } else if got.len() < r.greedy.len() {
        // `generate` stops at an end-of-turn id; the reference loop does not.
        let next = r.greedy[got.len()];
        assert!(
            runner.config().eos_token_ids.contains(&next),
            "stopped early at {} tokens, but the reference continues with {next}",
            got.len()
        );
    }
}

#[test]
fn tiny_random_model() {
    let Ok(dir) = rlx_ir::env::var_result("RLX_SALA_TINY_DIR") else {
        eprintln!("skip: set RLX_SALA_TINY_DIR (scripts/minicpm_sala_reference.py tiny)");
        return;
    };
    let r = load(&format!("{dir}/reference.json"));
    // Chunk sizes that split the 21-token prompt differently: one token at a
    // time, chunks with a padded tail, and the whole prompt in one run.
    let chunks: Vec<usize> = rlx_ir::env::var_result("RLX_SALA_TEST_CHUNKS")
        .map(|s| {
            s.split(',')
                .map(|c| c.trim().parse().expect("chunk"))
                .collect()
        })
        .unwrap_or_else(|_| vec![1, 5, 32]);
    for chunk in chunks {
        eprintln!("--- chunk {chunk}");
        let mut runner = SalaRunner::builder()
            .model_dir(&dir)
            .device(test_device())
            .chunk(chunk)
            .max_seq(64)
            .f32_weights(true)
            .build()
            .expect("build tiny runner");
        check(&mut runner, &r, 1e-3, 1e-3);
    }
}

#[test]
fn real_weights() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_SALA_DIR"),
        rlx_ir::env::var_result("RLX_SALA_REF"),
    ) else {
        eprintln!(
            "skip: set RLX_SALA_DIR and RLX_SALA_REF (scripts/minicpm_sala_reference.py real)"
        );
        return;
    };
    let r = load(&format!("{prefix}.json"));
    // Our chat template + tokenizer give the reference's prompt ids.
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.json")).unwrap()).unwrap();
    if let Some(q) = meta["question"].as_str() {
        let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
        let ids = tok
            .encode(rlx_minicpm_sala::chat_prompt(q).as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(ids, r.prompt, "chat template + tokenizer");
        eprintln!("prompt ids match ({} tokens)", ids.len());
    }
    let mut b = SalaRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .chunk(16)
        .max_seq(256);
    if let Some(n) = r.layers {
        b = b.layers(n);
    }
    let mut runner = b.build().expect("build runner");
    // bf16 weights in an f32 graph against an f32 reference of the same
    // bf16 values: the products agree to f32 rounding.
    check(&mut runner, &r, 5e-2, 0.05);
}

/// InfLLM-v2 block-sparse attention against OpenBMB's CUDA kernels
/// (`scripts/minicpm_sala_reference.py sparse`, run where `infllm_v2` builds —
/// `scripts/infllm_reference_env.sh`): a prompt past `dense_len`, the first
/// layers. Every query's chosen blocks in the first attention layer, and the
/// prefill logits. The reference runs in bf16 (the kernels have no f32 path),
/// so blocks whose pooled scores sit at the top-k edge can swap; the test
/// reports them and bounds their share.
///
/// ```sh
/// RLX_SALA_DIR=$PWD/weights/lm/minicpm-sala RLX_SALA_SPARSE_REF=$PWD/.cache/sala/sparse2 \
///     cargo test -p rlx-minicpm-sala --release --test reference_parity sparse -- --nocapture
/// ```
#[test]
fn sparse_attention_vs_cuda_kernels() {
    let (Ok(dir), Ok(prefix)) = (
        rlx_ir::env::var_result("RLX_SALA_DIR"),
        rlx_ir::env::var_result("RLX_SALA_SPARSE_REF"),
    ) else {
        eprintln!("skip: set RLX_SALA_DIR and RLX_SALA_SPARSE_REF");
        return;
    };
    let r = load(&format!("{prefix}.json"));
    let bytes = std::fs::read(format!("{prefix}.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let topk = st.tensor("topk.0").unwrap();
    let [nkv, nq, k] = [topk.shape()[0], topk.shape()[1], topk.shape()[2]];
    let ref_idx: Vec<i32> = topk
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    assert_eq!(nq, r.prompt.len(), "selection rows");

    let mut runner = SalaRunner::builder()
        .model_dir(&dir)
        .device(test_device())
        .layers(r.layers.expect("layer count"))
        .max_seq(r.prompt.len() + r.greedy.len())
        .chunk(16)
        .build()
        .expect("build runner");
    // Rows per run (the runner's chunk).
    let chunk = 16usize;
    let mut dump: Vec<u8> = Vec::new();
    let mut dump_nb = 0usize;
    let mut rows_differing = 0usize;
    let mut blocks_swapped = 0usize;
    let mut blocks_total = 0usize;
    let logits = runner
        .feed_observed(&r.prompt, |start, n, sel| {
            let t = chunk;
            let nb = sel.len() / (nkv * t);
            if dump.is_empty() {
                dump_nb = nb;
                dump = vec![0u8; nkv * nq * nb];
            }
            for h in 0..nkv {
                for u in 0..n {
                    for b in 0..nb {
                        dump[(h * nq + start + u) * dump_nb + b] =
                            u8::from(sel[(h * t + u) * nb + b] > 0.5);
                    }
                }
            }
            for h in 0..nkv {
                for u in 0..n {
                    let i = start + u;
                    let ours: Vec<i32> = (0..nb)
                        .filter(|&b| sel[(h * t + u) * nb + b] > 0.5)
                        .map(|b| b as i32)
                        .collect();
                    let mut theirs: Vec<i32> = ref_idx[(h * nq + i) * k..(h * nq + i + 1) * k]
                        .iter()
                        .copied()
                        .filter(|&b| b >= 0)
                        .collect();
                    theirs.sort_unstable();
                    theirs.dedup();
                    blocks_total += theirs.len();
                    if ours != theirs {
                        rows_differing += 1;
                        blocks_swapped += theirs.iter().filter(|b| !ours.contains(b)).count();
                    }
                }
            }
        })
        .unwrap();
    if let Ok(path) = rlx_ir::env::var_result("RLX_SALA_SPARSE_DUMP") {
        // Our choice, `[nkv, N, nb]` bytes, for scripts/infllm_selection_check.py.
        std::fs::write(&path, &dump).unwrap();
        eprintln!("wrote our block choice to {path}");
    }
    let rows = nkv * nq;
    eprintln!(
        "selection: {rows_differing} of {rows} (head, query) rows differ; {blocks_swapped} of \
         {blocks_total} chosen blocks swapped"
    );
    let d = logits
        .iter()
        .zip(&r.last_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = r.last_logits.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!("prefill logits: max |rlx - ref| = {d:.3e} (max |ref| {scale:.2}, ref in bf16)");
    assert!(
        (blocks_swapped as f64) < 0.01 * blocks_total as f64,
        "{blocks_swapped} of {blocks_total} blocks swapped"
    );
    assert!(d < 0.05 * scale.max(1.0), "prefill logits differ by {d}");
}
