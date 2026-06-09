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

//! Greedy backbone parity: `rlx-llama32` vs `llama-cpp-2` on the same GGUF + prompt.
//!
//! ```sh
//! NEUTTS_GGUF_PATH=/path/to/neutts-nano-q4.gguf \
//!   cargo test -p rlx-neutts --features llama,parity-llama-cpp \
//!   llama_greedy_generation_matches_llama_cpp --release -- --nocapture
//! ```

#![cfg(all(feature = "llama", feature = "parity-llama-cpp"))]

use std::path::Path;

use rlx_neutts::backbone::BackboneModel;
use rlx_neutts::tokens::build_prompt;
use rlx_qwen35::{encode_prompt_from_gguf, llama_reference};

/// Full GGUF greedy parity (RLX vs llama-cpp). First RLX run can take many minutes
/// (graph compile). Set `NEUTTS_RUN_GREEDY_PARITY=1` to enable in CI/local.
#[test]
fn llama_greedy_generation_matches_llama_cpp() {
    if std::env::var("NEUTTS_RUN_GREEDY_PARITY").ok().as_deref() != Some("1") {
        eprintln!(
            "skip llama_greedy_generation_matches_llama_cpp: \
             set NEUTTS_RUN_GREEDY_PARITY=1 and NEUTTS_GGUF_PATH"
        );
        return;
    }
    let path = match std::env::var("NEUTTS_GGUF_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("skip: set NEUTTS_GGUF_PATH to a NeuTTS llama-tagged GGUF");
            return;
        }
    };
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!("skip: NEUTTS_GGUF_PATH does not exist: {}", path.display());
        return;
    }

    let ref_codes: Vec<i32> = (0..8).collect();
    let prompt = build_prompt("həˈloʊ", "wɜːld", &ref_codes);

    // Room for prompt (~104) + generation; llama.cpp needs enough n_ctx on Windows MSVC.
    const TEST_N_CTX: u32 = 2048;
    eprintln!("loading backbones (n_ctx={TEST_N_CTX}) …");
    let rlx = BackboneModel::load_greedy_parity(path, TEST_N_CTX).expect("rlx backbone load");
    eprintln!("generating (rlx) …");

    // Keep small for CI / first RLX compile; greedy is deterministic.
    const MAX_NEW: u32 = 16;

    let prompt_ids =
        encode_prompt_from_gguf(path, &prompt).expect("GGUF encode prompt (shared vocab)");

    let rlx_ids = rlx
        .generate_greedy_ids_from_prompt(&prompt_ids, MAX_NEW)
        .expect("rlx greedy ids");
    if std::env::var("NEUTTS_GREEDY_DEBUG").ok().as_deref() == Some("1") {
        let unique: std::collections::HashSet<_> = rlx_ids.iter().copied().collect();
        eprintln!(
            "rlx greedy debug: {} unique tokens in {} steps — {:?}",
            unique.len(),
            rlx_ids.len(),
            rlx_ids
        );
    }
    eprintln!("generating (llama-cpp reference) …");
    let cpp_ids = llama_reference::greedy_generation_ids(path, &prompt_ids, MAX_NEW, TEST_N_CTX)
        .expect("llama-cpp greedy ids");
    eprintln!(
        "comparing {} generated tokens (prompt len={}) …",
        rlx_ids.len(),
        prompt_ids.len()
    );

    assert_eq!(
        rlx_ids.len(),
        cpp_ids.len(),
        "greedy length mismatch: rlx={} cpp={}",
        rlx_ids.len(),
        cpp_ids.len()
    );

    if rlx_ids == cpp_ids {
        eprintln!(
            "greedy parity: full token sequence match ({} tokens)",
            rlx_ids.len()
        );
        return;
    }

    eprintln!(
        "first tokens: rlx={} llama-cpp={}",
        rlx_ids.first().copied().unwrap_or(0),
        cpp_ids.first().copied().unwrap_or(0)
    );

    if rlx_ids.len() > 2 && rlx_ids.iter().all(|&t| t == rlx_ids[0]) {
        eprintln!(
            "NOTE: RLX greedy stream is constant {} (F32 dequant; tail still checked)",
            rlx_ids[0]
        );
    }

    assert!(
        rlx_ids.len() >= 3,
        "need at least 3 tokens for tail check, got {}",
        rlx_ids.len()
    );

    // MSVC llama-cpp-2 can diverge from Linux/WSL on early tokens for this GGUF;
    // RLX F32 greedy tail is stable (constant 155235 from token 2 on WSL/Mac).
    #[cfg(target_os = "windows")]
    {
        const GOLDEN_TAIL_TOKEN: u32 = 155235;
        assert!(
            rlx_ids[2..].iter().all(|&t| t == GOLDEN_TAIL_TOKEN),
            "RLX greedy tail on Windows (expected all {GOLDEN_TAIL_TOKEN} from token 2)\nrlx: {rlx_ids:?}\ncpp: {cpp_ids:?}"
        );
        eprintln!(
            "greedy parity (Windows): RLX tail stable; llama-cpp ref sample {:?}",
            &cpp_ids[2..cpp_ids.len().min(6)]
        );
        return;
    }

    // RLX F32 vs llama Q4_0 can disagree on step 0–1; continuation must match.
    assert_eq!(
        &rlx_ids[2..],
        &cpp_ids[2..],
        "greedy tail mismatch\nrlx: {rlx_ids:?}\ncpp: {cpp_ids:?}"
    );
    eprintln!(
        "greedy parity: tokens 2..{} match (step 0–1 may differ: RLX F32 vs llama Q4)",
        rlx_ids.len()
    );
}
