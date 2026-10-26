//! The tokenizer rebuilt from a Qwen3.5 GGUF's metadata must tokenize like
//! the model's `tokenizer.json`.
//!
//! It used to byte-level split with GPT-2's regex and skip NFC, which turns
//! `"\n\n"` into `198 198` instead of `271` and leaves decomposed accents
//! unnormalized — every chat template has blank lines, so every GGUF-only
//! prompt was off-distribution.
//!
//! Needs a Qwen3.5 GGUF with its `tokenizer.json` beside it:
//! `QWEN35_TOKENIZER_GGUF` (default `weights/lm/qwen3.5-0.8b-q4km/model.gguf`);
//! skipped when absent.
#![cfg(feature = "qwen35-tokenizer")]

use std::path::PathBuf;

const CASES: &[&str] = &[
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi\n\n\nthere<|im_end|>\n",
    "<think>\n\n</think>\n\n",
    "It's we'LL they'RE I'd you've",
    "1234567 3.14159 0x7ffe",
    "café cafe\u{0301} Zoe\u{0308}",
    "नमस्ते दुनिया",
    "สวัสดีครับ",
    "日本語と한국어",
    "tabs\tand\r\nwindows\r\n\r\nlines   \n  ",
];

#[test]
fn gguf_tokenizer_matches_tokenizer_json() {
    let gguf = rlx_ir::env::read("QWEN35_TOKENIZER_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/lm/qwen3.5-0.8b-q4km/model.gguf")
        });
    let json = gguf.with_file_name("tokenizer.json");
    if !gguf.exists() || !json.exists() {
        eprintln!("skipping: need {} and {}", gguf.display(), json.display());
        return;
    }
    let ours =
        tokenizers::Tokenizer::from_bytes(rlx_qwen35::gguf_tokenizer_json(&gguf).unwrap()).unwrap();
    let theirs = tokenizers::Tokenizer::from_file(&json).unwrap();
    for case in CASES {
        let a = ours.encode(*case, false).unwrap();
        let b = theirs.encode(*case, false).unwrap();
        assert_eq!(a.get_ids(), b.get_ids(), "{case:?}");
    }
}
