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

//! Live HTTPS quick-check test for `rlx_cli::check_hf_repo`.
//!
//! Gated on `RLX_NET_TESTS=1` because it hits the public HuggingFace
//! Hub. Builds the test binary with the `compat-net` feature so the
//! `check_hf_repo` symbol resolves to the real implementation rather
//! than the cfg-stub.
//!
//! Run:
//!   ```sh
//!   RLX_NET_TESTS=1 cargo test -p rlx-cli --features compat-net --test hf_repo_check_live -- --nocapture
//!   ```

#![cfg(feature = "compat-net")]

use rlx_cli::{CompatibilityStatus, check_hf_repo};

fn net_enabled() -> bool {
    std::env::var("RLX_NET_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn check_real_gguf_repo_returns_supported() {
    if !net_enabled() {
        eprintln!("skip: set RLX_NET_TESTS=1 to run live HTTP tests");
        return;
    }
    let report = check_hf_repo("bartowski/SmolLM2-135M-Instruct-GGUF")
        .expect("HF tree API + GGUF header fetch should succeed");

    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert_eq!(*runner, "llama32", "SmolLM2 is llama-arch");
        }
        other => panic!("expected Supported, got {other:?}\nreport:\n{report}"),
    }
    let fields = report
        .gguf_fields
        .as_ref()
        .expect("GGUF fields populated when sourced from GGUF header");
    assert!(fields.is_complete(), "missing: {:?}", fields.missing());
    assert_eq!(fields.tokenizer_model.as_deref(), Some("gpt2"));
    eprintln!("HF live check {report}");
}

#[test]
fn check_safetensors_only_repo_uses_config_json() {
    if !net_enabled() {
        eprintln!("skip: set RLX_NET_TESTS=1");
        return;
    }
    // Pure safetensors repo — no .gguf — exercises the config.json path.
    let report = check_hf_repo("HuggingFaceTB/SmolLM2-135M-Instruct")
        .expect("config.json fetch should succeed");
    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert_eq!(*runner, "llama32");
        }
        other => panic!("expected Supported, got {other:?}\nreport:\n{report}"),
    }
    eprintln!("HF live safetensors check {report}");
}

#[test]
fn check_known_unimplemented_arch_reports_milestone() {
    if !net_enabled() {
        eprintln!("skip: set RLX_NET_TESTS=1");
        return;
    }
    // unsloth/MiniMax-M2.5-GGUF would be ideal but is 100+ GB.
    // Anything tagged minimax-m2 / glm4 / nemotron_h will do — the
    // header parser only fetches the first 4 MB. Skip gracefully if
    // the chosen repo isn't reachable.
    // Range fetch only pulls 4 MB so multi-GB candidates are fine —
    // we never download the full file.
    let candidates = [
        "bartowski/Phi-3-mini-4k-instruct-GGUF",     // phi3 → M4
        "bartowski/Phi-3.1-mini-128k-instruct-GGUF", // phi3 → M4
        "bartowski/MiniMax-M2-GGUF",
        "bartowski/zai-org_GLM-4.6-GGUF",
    ];
    for repo in &candidates {
        match check_hf_repo(repo) {
            Ok(report) => {
                if let CompatibilityStatus::KnownUnimplemented(u) = &report.status {
                    eprintln!(
                        "HF live unimplemented check `{repo}` → {} ({})",
                        u.family, u.milestone
                    );
                    assert!(["M4", "M5", "M6", "M7"].contains(&u.milestone));
                    return;
                } else {
                    eprintln!(
                        "WARN: `{repo}` returned unexpected status {:?} — likely implemented now or arch tag changed",
                        report.status
                    );
                }
            }
            Err(e) => {
                eprintln!("WARN: `{repo}` fetch failed: {e:#}");
            }
        }
    }
    eprintln!("skip: no candidate KnownUnimplemented repo was reachable");
}
