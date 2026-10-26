//! The GGUF-embedded tokenizer (as [`rlx_pplx_decider::gguf_tokenizer`]
//! configures it) must tokenize exactly like the checkpoint's own
//! `tokenizer.json`, which is what the reference processor uses.
//!
//! Needs both files: `PPLX_DECIDER_GGUF` (default
//! `weights/lm/pplx-decider-flash-gguf`) and `PPLX_DECIDER_HF` (default
//! `weights/lm/pplx-decider-flash`); skipped when either is absent.

use std::path::PathBuf;

/// Each line targets one split rule or normalization step.
const CASES: &[&str] = &[
    "State:\nfoo\n\nQuestion 0:\nbar\n\n\n  \n",
    "  leading spaces and trailing newline\n",
    "It's we'LL they'RE I'd you've",
    "1234567 3.14159 0x7ffe",
    "Zoë Zoe\u{0308} café cafe\u{0301} — ＜|im_end|＞",
    "नमस्ते दुनिया क्षत्रिय",
    "สวัสดีครับ ภาษาไทย",
    "日本語のテキストと한국어 텍스트",
    "emoji 👍🏽 and \u{200d} joiners 👩‍💻",
    "<|im_start|>user\nhi<|im_end|>\n<|fim_suffix|><think>\n\n</think>\n\n",
    "tabs\tand\r\nwindows\r\n\r\nlines",
    "{\"a\": [true, null, 1.5], \"b\": \"x\"}",
];

fn path(env: &str, default: &str) -> PathBuf {
    rlx_ir::env::read(env)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(default))
}

#[test]
fn gguf_tokenizer_matches_tokenizer_json() {
    let gguf = path(
        "PPLX_DECIDER_GGUF",
        "../../weights/lm/pplx-decider-flash-gguf",
    );
    let hf = path("PPLX_DECIDER_HF", "../../weights/lm/pplx-decider-flash").join("tokenizer.json");
    if !gguf.exists() || !hf.exists() {
        eprintln!("skipping: need {} and {}", gguf.display(), hf.display());
        return;
    }
    let ours =
        rlx_pplx_decider::gguf_tokenizer(&rlx_pplx_decider::resolve_gguf(&gguf).unwrap()).unwrap();
    let theirs = tokenizers::Tokenizer::from_file(&hf).unwrap();
    for case in CASES {
        let a = ours.encode(*case, true).unwrap();
        let b = theirs.encode(*case, true).unwrap();
        assert_eq!(a.get_ids(), b.get_ids(), "{case:?}");
    }
}
