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

//! Prompt assembly against reference token ids.
//!
//! `tests/fixtures/jina_ocr_prompt_ids.json` was produced by the checkpoint's
//! own `tokenizer.json`, split exactly the way
//! `processing_deepseek_ocr.py::preprocess_prompt_and_image` splits it.
//!
//! The template checks run everywhere. The id checks need `tokenizer.json`
//! (~10 MB) but *not* the 6.7 GB weights, so `RLX_JINA_OCR_DIR` pointed at a
//! tokenizer-only directory is enough to exercise them.

use rlx_jina_ocr::config::IMAGE_TOKEN_ID;
use rlx_jina_ocr::prompt::{self, DEFAULT_OCR_PROMPT};
use std::path::PathBuf;

const FIXTURE: &str = include_str!("fixtures/jina_ocr_prompt_ids.json");

struct Case {
    name: String,
    rendered: String,
    pre_ids: Vec<u32>,
    post_ids: Vec<u32>,
}

fn cases() -> Vec<Case> {
    let doc: serde_json::Value = serde_json::from_str(FIXTURE).expect("parse prompt fixture");
    assert_eq!(
        doc["image_token_id"].as_u64().unwrap() as u32,
        IMAGE_TOKEN_ID
    );
    doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            rendered: c["rendered"].as_str().unwrap().to_string(),
            pre_ids: ids(&c["pre_ids"]),
            post_ids: ids(&c["post_ids"]),
        })
        .collect()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// A directory holding `tokenizer.json`, weights or not.
fn tokenizer_dir() -> Option<PathBuf> {
    if let Ok(raw) = rlx_ir::env::var_result("RLX_JINA_OCR_DIR") {
        let p = PathBuf::from(raw);
        if p.join("tokenizer.json").is_file() {
            return Some(p);
        }
    }
    let dir = rlx_jina_ocr::hub::default_model_dir().ok()?;
    dir.join("tokenizer.json").is_file().then_some(dir)
}

/// The chat template must reproduce the reference rendering byte for byte —
/// a stray newline or a missing role prefix shifts every downstream id.
#[test]
fn chat_template_matches_the_reference_rendering() {
    let by_name: Vec<Case> = cases();
    let find = |n: &str| by_name.iter().find(|c| c.name == n).expect("case");

    assert_eq!(
        prompt::ocr_prompt_text(DEFAULT_OCR_PROMPT),
        find("default_ocr_prompt").rendered
    );
    assert_eq!(
        prompt::ocr_prompt_text("Extract every table as HTML."),
        find("custom_prompt").rendered
    );

    let with_system = vec![
        prompt::Message::system("Be terse."),
        prompt::Message::user(vec![
            prompt::Content::Image,
            prompt::Content::text("OCR this."),
        ]),
    ];
    assert_eq!(
        prompt::apply_chat_template(&with_system, true),
        find("with_system").rendered
    );
}

/// Ids straight from the checkpoint's tokenizer: no BOS, placeholder run
/// spliced at the marker, everything else byte-identical to the reference.
#[test]
fn prompt_ids_match_the_reference_tokenizer() {
    let Some(dir) = tokenizer_dir() else {
        eprintln!(
            "[prompt_ids] skipped — set RLX_JINA_OCR_DIR to a directory holding tokenizer.json"
        );
        return;
    };
    let tok = rlx_jina_ocr::JinaTokenizer::open(&dir).expect("load tokenizer");
    assert_eq!(tok.token_to_id("<image>"), Some(IMAGE_TOKEN_ID));

    for case in cases() {
        let n_image = 7usize;
        let ids = prompt::build_prompt_ids(&case.rendered, n_image, IMAGE_TOKEN_ID, |chunk| {
            tok.encode(chunk)
        })
        .unwrap_or_else(|e| panic!("{}: {e:#}", case.name));

        let mut want = case.pre_ids.clone();
        want.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_image));
        want.extend_from_slice(&case.post_ids);
        assert_eq!(ids, want, "{}", case.name);

        assert_ne!(
            ids[0],
            rlx_jina_ocr::config::BOS_TOKEN_ID,
            "{}: the processor calls text_encode(bos=False)",
            case.name
        );
    }
}

/// The n-gram whitelist ids must really be `<td>` / `</td>`; a stale pair would
/// silently truncate long tables instead of protecting them.
#[test]
fn ngram_whitelist_ids_are_the_table_cell_tokens() {
    let Some(dir) = tokenizer_dir() else {
        eprintln!("[prompt_ids] skipped — no tokenizer.json");
        return;
    };
    let tok = rlx_jina_ocr::JinaTokenizer::open(&dir).expect("load tokenizer");
    assert_eq!(
        tok.token_to_id("<td>"),
        Some(rlx_jina_ocr::config::NGRAM_WHITELIST[0])
    );
    assert_eq!(
        tok.token_to_id("</td>"),
        Some(rlx_jina_ocr::config::NGRAM_WHITELIST[1])
    );
    // The tokenizer defines 128 827 ids; `config.vocab_size` (and so the
    // embedding table and LM head) is padded above that. Every id the
    // tokenizer can emit must still index into the head.
    assert!(
        tok.vocab_size() <= 129_280,
        "tokenizer vocab {} exceeds the LM head width",
        tok.vocab_size()
    );
    assert!(
        rlx_jina_ocr::config::NGRAM_WHITELIST
            .iter()
            .all(|&id| (id as usize) < 129_280)
    );
}
