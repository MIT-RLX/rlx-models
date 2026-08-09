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

//! Live HybridDNATokenizer round-trip against a real Carbon model directory.
//!
//! Gated on `RLX_CARBON_MODEL=<dir>` (the dir must contain `tokenizer.json` +
//! `dna_config.json`), since it reads the bundled Qwen3 `tokenizer.json`.
//!
//! ```sh
//! RLX_CARBON_MODEL=/path/to/Carbon-500M \
//!   cargo test -p rlx-carbon --features tokenizer --test dna_roundtrip_live -- --nocapture
//! ```

#![cfg(feature = "tokenizer")]

use rlx_carbon::HybridDnaTokenizer;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_CARBON_MODEL").ok().map(PathBuf::from)
}

#[test]
fn dna_and_text_round_trip_through_real_tokenizer() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_CARBON_MODEL=<Carbon model dir>");
        return;
    };
    let tok = HybridDnaTokenizer::from_dir(&dir).expect("load tokenizer");
    let dna = tok.dna_config();

    // 1. A clean, 6-mer-aligned DNA region round-trips exactly.
    let seq = "ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACCGATCTG"; // 45 bases → not aligned
    let aligned = "ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACCGATCTGA"; // 46 → still not; use 48
    let clean = "ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACCGATCTGGCA"; // 48 = 8 × 6-mer
    assert_eq!(clean.len() % dna.k, 0);

    // Closed region: <dna>…</dna>. Ids: begin, 8 k-mers, end.
    let ids = tok.encode(&format!("<dna>{clean}</dna>")).expect("encode");
    assert_eq!(ids.first().copied(), Some(dna.begin_id()));
    assert_eq!(ids.last().copied(), Some(dna.end_id()));
    let kmer_ids = &ids[1..ids.len() - 1];
    assert_eq!(kmer_ids.len(), clean.len() / dna.k);
    for &id in kmer_ids {
        assert!(dna.is_dna_id(id), "k-mer id {id} not in DNA range");
        assert_ne!(id, dna.oov_id(), "clean sequence should not produce <oov>");
    }
    // skip_special decode recovers the nucleotides.
    let decoded = tok.decode(&ids, true).expect("decode");
    assert_eq!(decoded, clean, "closed-region DNA round-trip");

    // 2. Open region (generation form) has begin but no end, still recovers seq.
    let open = tok.encode(&format!("<dna>{clean}")).expect("encode open");
    assert_eq!(open.first().copied(), Some(dna.begin_id()));
    assert!(!open.contains(&dna.end_id()));
    assert_eq!(tok.decode(&open, true).unwrap(), clean);

    // 3. Bare k-mer ids (the generation case: no <dna> wrapper) decode to bases.
    let bare = &ids[1..ids.len() - 1];
    assert_eq!(tok.decode(bare, true).unwrap(), clean);

    // 4. Plain text goes through the base Qwen3 BPE and round-trips.
    let text = "The quick brown fox";
    let tids = tok.encode(text).expect("encode text");
    assert!(
        tids.iter().all(|&t| !dna.is_dna_id(t)),
        "text ids stay in BPE range"
    );
    assert_eq!(tok.decode(&tids, true).unwrap(), text);

    // 5. Non-ATCG bases inside a region map to <oov>.
    let with_n = tok.encode("<dna>ATCGATNNNNNN").unwrap();
    assert!(
        with_n.contains(&dna.oov_id()),
        "expected an <oov> for NNNNNN"
    );

    let _ = (seq, aligned);
    eprintln!(
        "carbon tokenizer round-trip OK ({} k-mers, vocab {})",
        kmer_ids.len(),
        tok.vocab_size()
    );
}
