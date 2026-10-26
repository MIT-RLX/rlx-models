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

//! Parity vs sentence-transformers.
//!
//! ```sh
//! python3 scripts/embedding_hf_reference.py weights/embed/jina-embeddings-v5-text-nano \
//!     /tmp/jina-v5-nano-retrieval.ref.json --task retrieval --prompt-name query
//! RLX_JINA_V5_DIR=$PWD/weights/embed/jina-embeddings-v5-text-nano \
//! RLX_JINA_V5_REF=/tmp/jina-v5-nano-retrieval.ref.json \
//!     cargo test -p rlx-jina-v5 --release --test hf_parity_live -- --nocapture
//! ```
//! `RLX_TEST_DEVICE=metal` (with the matching feature) runs it on a GPU.

use rlx_jina_v5::JinaV5;

#[test]
fn matches_sentence_transformers() {
    let (Ok(dir), Ok(path)) = (
        rlx_ir::env::var_result("RLX_JINA_V5_DIR"),
        rlx_ir::env::var_result("RLX_JINA_V5_REF"),
    ) else {
        eprintln!("skip: set RLX_JINA_V5_DIR and RLX_JINA_V5_REF");
        return;
    };
    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(rlx_runtime::Device::Cpu);
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read reference")).unwrap();
    let prompt = r["prompt"].as_str().unwrap();
    let task = r["task"].as_str().unwrap_or("retrieval");
    let mut model = JinaV5::open(&dir, task, device).expect("open");

    for row in r["rows"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let want_ids: Vec<u32> = row["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want: Vec<f32> = row["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let ids = model
            .tokenize(&format!("{prompt}{text}"))
            .expect("tokenize");
        assert_eq!(ids, want_ids, "token ids for {text:?}");
        let got = model.embed_ids(&ids).expect("embed");
        let cos: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{text:?}: cosine {cos:.7}, max |d| {max_abs:.2e}");
        // Measured ≤ 9.6e-7 on CPU (retrieval and text-matching adapters);
        // skipping the LoRA merge gives cosine 0.94 and max |d| 0.18.
        assert!(cos > 0.99999, "{text:?}: cosine {cos}");
        assert!(max_abs < 1e-5, "{text:?}: max |d| {max_abs}");
    }
}
