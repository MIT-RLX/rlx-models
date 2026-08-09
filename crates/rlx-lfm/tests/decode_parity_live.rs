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

//! The incremental (KV + conv-state) decode must be token-identical to the
//! reference re-prefill path. Gated on `RLX_LFM_WEIGHTS=<file.gguf|dir>`.
//!
//! ```sh
//! RLX_LFM_WEIGHTS=/path/LFM2.5-2.6B-Q4_K_M.gguf \
//!   cargo test -p rlx-lfm --test decode_parity_live -- --nocapture
//! ```

use rlx_lfm::{Lfm2GgufRunner, resolve_gguf};
use rlx_runtime::Device;
use std::path::PathBuf;

fn weights() -> Option<PathBuf> {
    std::env::var("RLX_LFM_WEIGHTS").ok().map(PathBuf::from)
}

#[test]
fn incremental_decode_matches_reference_prefill() {
    let Some(w) = weights() else {
        eprintln!("skip: set RLX_LFM_WEIGHTS=<LFM2 .gguf>");
        return;
    };
    let gguf = resolve_gguf(&w).expect("resolve gguf");
    let runner = Lfm2GgufRunner::open(&gguf, Device::Cpu).expect("open runner");

    // Arbitrary in-vocab prompt ids — parity must hold regardless of meaning.
    let prompt: Vec<u32> = vec![100, 2000, 500, 42, 7, 99];
    let n_new = 24;

    let t0 = std::time::Instant::now();
    let decode = runner.generate(&prompt, n_new, |_| true).expect("decode");
    let dt_decode = t0.elapsed();

    let t1 = std::time::Instant::now();
    let prefill = runner
        .generate_prefill(&prompt, n_new, |_| true)
        .expect("prefill");
    let dt_prefill = t1.elapsed();

    eprintln!("decode  ({:?}): {decode:?}", dt_decode);
    eprintln!("prefill ({:?}): {prefill:?}", dt_prefill);
    assert_eq!(
        decode, prefill,
        "incremental decode diverged from reference prefill"
    );
    eprintln!(
        "✅ incremental decode == reference prefill ({} tokens); decode {:.2?} vs prefill {:.2?}",
        decode.len(),
        dt_decode,
        dt_prefill
    );
}
