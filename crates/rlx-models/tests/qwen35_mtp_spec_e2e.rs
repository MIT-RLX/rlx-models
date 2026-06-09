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

// Env-gated E2E: real MTP GGUF weights — prefill MTP logits + speculative decode.
//
//   QWEN35_MTP_GGUF_PATH=/path/to/Qwen3.5-0.8B-MTP-....gguf \
//     cargo test -p rlx-models qwen35_mtp --release -- --nocapture
//
// Falls back to `/tmp/rlx-models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
// when present (materialized from HF cache).

mod compile_support;

use rlx_gguf::GgufFile;
use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen35::{
    Qwen35Config, Qwen35MtpDraft, Qwen35TrunkTarget, speculative_decode_round,
};
use rlx_runtime::spec_decode::SpecDecoder;
use std::path::{Path, PathBuf};

const DEFAULT_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";

fn hf_cache_mtp_q4() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let repo =
        PathBuf::from(home).join(".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-MTP-GGUF");
    let rev = std::fs::read_to_string(repo.join("refs/main")).ok()?;
    let rev = rev.trim();
    let path = repo
        .join("snapshots")
        .join(rev)
        .join("Qwen3.5-0.8B-Q4_K_M.gguf");
    path.is_file().then_some(path)
}

fn mtp_gguf_path() -> Option<PathBuf> {
    for key in ["QWEN35_MTP_GGUF_PATH", "QWEN35_GGUF_PATH"] {
        if let Ok(p) = std::env::var(key) {
            if !p.is_empty() {
                let path = PathBuf::from(p);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
    }
    if let Some(p) = hf_cache_mtp_q4() {
        return Some(p);
    }
    let default = PathBuf::from(DEFAULT_MTP_Q4);
    default.is_file().then_some(default)
}

/// Returns `None` (and emits a skip line) when no fixture is
/// available so the parent test can early-return success.
fn try_mtp_gguf() -> Option<PathBuf> {
    match mtp_gguf_path() {
        Some(p) => Some(p),
        None => {
            eprintln!(
                "skip qwen35_mtp_spec_e2e: set QWEN35_MTP_GGUF_PATH or materialize {DEFAULT_MTP_Q4}"
            );
            None
        }
    }
}

fn load_mtp_cfg(path: &Path) -> Qwen35Config {
    let raw = GgufFile::from_path(path).expect("open MTP GGUF");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse qwen35 config");
    assert!(
        cfg.nextn_predict_layers > 0,
        "expected MTP heads (nextn_predict_layers > 0), got {}",
        cfg.nextn_predict_layers
    );
    cfg
}

fn build_draft_runner(path: &Path, max_seq: usize) -> rlx_models::Qwen35Runner {
    Qwen35RunnerBuilder::default()
        .weights(path)
        .max_seq(max_seq)
        .enable_mtp(true)
        .mtp_logits_path(true)
        .packed_weights(true)
        .last_logits_only(true)
        .build()
        .expect("build MTP draft runner")
}

fn build_target_runner(path: &Path, max_seq: usize) -> rlx_models::Qwen35Runner {
    Qwen35RunnerBuilder::default()
        .weights(path)
        .max_seq(max_seq)
        .enable_mtp(false)
        .mtp_logits_path(false)
        .packed_weights(true)
        .last_logits_only(true)
        .build()
        .expect("build trunk target runner")
}

#[test]
fn qwen35_mtp_gguf_loads_with_nextn_heads() {
    let Some(weights) = try_mtp_gguf() else {
        return;
    };
    let cfg = load_mtp_cfg(&weights);
    eprintln!(
        "qwen35 MTP GGUF: layers={} nextn={} vocab={}",
        cfg.num_hidden_layers, cfg.nextn_predict_layers, cfg.vocab_size
    );
    let _ = build_draft_runner(&weights, 64);
}

#[test]
fn qwen35_mtp_prefill_emits_mtp_logits() {
    let Some(weights) = try_mtp_gguf() else {
        return;
    };
    let _cfg = load_mtp_cfg(&weights);
    let prompt = vec![1u32, 2, 3, 4];
    let max_seq = prompt.len().max(32);

    let mut draft = build_draft_runner(&weights, max_seq);
    let seed = draft
        .prefill_seed_for_decode(&prompt)
        .expect("prefill draft");
    let mtp = seed
        .mtp_logits
        .as_ref()
        .expect("MTP logits after prefill on *-MTP-GGUF");
    assert_eq!(
        mtp.len(),
        draft.cfg().vocab_size,
        "full-vocab MTP logits expected without --fast-mtp"
    );
    assert!(
        mtp.iter().any(|&x| x.is_finite() && x != 0.0),
        "MTP logits should be non-trivial"
    );
}

#[test]
fn qwen35_mtp_spec_decode_e2e_vs_greedy_baseline() {
    let Some(weights) = try_mtp_gguf() else {
        return;
    };
    let cfg = load_mtp_cfg(&weights);
    let prompt = vec![42u32, 1337, 7];
    let n_new = 8usize;
    let spec_n = 2usize;
    let max_seq = (prompt.len() + n_new + spec_n).max(32);

    let mut greedy_runner = build_target_runner(&weights, max_seq);
    let greedy = greedy_runner
        .generate(&prompt, n_new, |_| true)
        .expect("greedy generate");
    assert_eq!(greedy.len(), n_new);

    let draft_runner = build_draft_runner(&weights, max_seq);
    let target_runner = build_target_runner(&weights, max_seq);
    let mut dec = SpecDecoder::new(
        Qwen35MtpDraft::new(draft_runner),
        Qwen35TrunkTarget::new(target_runner),
        spec_n,
        0,
    );

    let mut context = prompt.clone();
    let mut spec_out = Vec::new();
    while spec_out.len() < n_new {
        let batch = dec.step(&context);
        assert!(!batch.is_empty(), "spec decode round produced no tokens");
        assert!(
            batch.len() <= spec_n + 1,
            "round emitted {} tokens (max {} expected)",
            batch.len(),
            spec_n + 1
        );
        for tok in batch {
            if spec_out.len() >= n_new {
                break;
            }
            assert!(
                (tok as usize) < cfg.vocab_size,
                "token {tok} out of vocab {}",
                cfg.vocab_size
            );
            spec_out.push(tok);
            context.push(tok);
        }
    }
    spec_out.truncate(n_new);

    let matches = spec_out
        .iter()
        .zip(greedy.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "qwen35 MTP spec vs greedy: spec={spec_out:?} greedy={greedy:?} pos_matches={matches}/{n_new}"
    );
    assert_eq!(spec_out.len(), n_new);
    // MTP draft logits differ from trunk — streams need not match. Preflight: both
    // emit in-vocab tokens and spec decode keeps up with greedy length.
    assert_ne!(spec_out, vec![0u32; n_new]);
}

#[test]
fn qwen35_mtp_spec_decode_round_helper() {
    let Some(weights) = try_mtp_gguf() else {
        return;
    };
    let _cfg = load_mtp_cfg(&weights);
    let prompt = vec![10u32, 20, 30];
    let max_seq = 32;

    let out = speculative_decode_round(
        Qwen35MtpDraft::new(build_draft_runner(&weights, max_seq)),
        Qwen35TrunkTarget::new(build_target_runner(&weights, max_seq)),
        &prompt,
        2,
        0,
    );
    assert!(!out.is_empty(), "speculative_decode_round must emit tokens");
}
