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

//! Optional HuggingFace `tokenizer.json` bridge for Qwen3.5 CLI use.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Resolve a tokenizer path: explicit `--tokenizer`, sibling of the
/// GGUF weights, or `tokenizer.json` in the weights directory.
pub fn resolve_tokenizer_path(weights: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    if weights.is_dir() {
        let in_dir = weights.join("tokenizer.json");
        if in_dir.is_file() {
            return Some(in_dir);
        }
    }
    let sibling = weights.with_extension("tokenizer.json");
    if sibling.is_file() {
        return Some(sibling);
    }
    weights
        .parent()
        .map(|d| d.join("tokenizer.json"))
        .filter(|p| p.is_file())
}

/// Path-keyed cache of parsed tokenizers. Building a `tokenizers::Tokenizer`
/// means reading + parsing a multi-MB `tokenizer.json` (or rebuilding a BPE from
/// GGUF metadata). Streaming decode calls the detokenizer **once per generated
/// token**, so doing that per call dominated decode latency (device-independent,
/// ~10-50×). Cache it so the parse happens once per file.
#[cfg(feature = "qwen35-tokenizer")]
pub(crate) fn cached_tokenizer(
    key: &Path,
    build: impl FnOnce() -> Result<tokenizers::Tokenizer>,
) -> Result<std::sync::Arc<tokenizers::Tokenizer>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<tokenizers::Tokenizer>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(tok) = cache.lock().unwrap().get(key) {
        return Ok(tok.clone());
    }
    let tok = Arc::new(build()?);
    cache.lock().unwrap().insert(key.to_path_buf(), tok.clone());
    Ok(tok)
}

/// Encode `text` to token ids using a HuggingFace tokenizer file.
#[cfg(feature = "qwen35-tokenizer")]
pub fn encode_prompt(tokenizer_path: &Path, text: &str) -> Result<Vec<u32>> {
    let tok = cached_tokenizer(tokenizer_path, || {
        let data = std::fs::read_to_string(tokenizer_path)
            .with_context(|| format!("read tokenizer {}", tokenizer_path.display()))?;
        tokenizers::Tokenizer::from_bytes(data.as_bytes())
            .map_err(|e| anyhow::anyhow!("parse tokenizer.json: {e}"))
    })?;
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[cfg(not(feature = "qwen35-tokenizer"))]
pub fn encode_prompt(_tokenizer_path: &Path, _text: &str) -> Result<Vec<u32>> {
    anyhow::bail!("tokenizer support not compiled in — rebuild with feature `qwen35-tokenizer`")
}

/// Encode with an optional explicit path; falls back to GGUF
/// embedded vocab via [`encode_prompt_from_gguf`] when no
/// `tokenizer.json` is found.
pub fn encode_prompt_auto(weights: &Path, explicit: Option<&Path>, text: &str) -> Result<Vec<u32>> {
    if let Some(path) = resolve_tokenizer_path(weights, explicit) {
        return encode_prompt(&path, text);
    }
    // PLAN.md M8 — GGUF vocab-only fallback.
    let is_gguf = weights
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if is_gguf {
        return encode_prompt_from_gguf(weights, text);
    }
    anyhow::bail!(
        "no tokenizer found for {:?}. Pass --tokenizer <path> or place \
         tokenizer.json next to the file",
        weights
    )
}

/// PLAN.md M8 — tokenize using the BPE vocab + merges embedded in the
/// GGUF metadata under `tokenizer.ggml.{tokens, merges}`. Suitable for
/// GPT-2 / Qwen / Llama / Mistral byte-level-BPE tokenizers; rejects
/// SPM tokenizers (the GGUF `tokenizer.ggml.model = "llama"` family)
/// with a clear error since those need a sentencepiece reconstruction
/// the `tokenizers` crate doesn't support from raw vocab arrays.
#[cfg(feature = "qwen35-tokenizer")]
pub fn encode_prompt_from_gguf(weights: &Path, text: &str) -> Result<Vec<u32>> {
    use rlx_gguf::{GgufFile, MetaValue};
    use tokenizers::AddedToken;
    use tokenizers::Tokenizer;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let raw = GgufFile::from_path(weights).with_context(|| format!("open GGUF {weights:?}"))?;
    let model_kind = raw
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(MetaValue::as_str)
        .unwrap_or("gpt2");
    if model_kind == "llama" || model_kind == "no_vocab" {
        anyhow::bail!(
            "GGUF tokenizer.ggml.model = `{model_kind}` (SentencePiece family) — \
             not yet supported in the vocab-only fallback; provide a tokenizer.json"
        );
    }

    let tokens_meta = raw
        .metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.tokens"))?;
    let tokens: Vec<String> = match tokens_meta {
        MetaValue::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => anyhow::bail!("tokenizer.ggml.tokens not an array"),
    };
    if tokens.is_empty() {
        anyhow::bail!("tokenizer.ggml.tokens is empty");
    }
    let merges_meta = raw
        .metadata
        .get("tokenizer.ggml.merges")
        .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.merges"))?;
    let merges_raw: Vec<String> = match merges_meta {
        MetaValue::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => anyhow::bail!("tokenizer.ggml.merges not an array"),
    };

    let vocab: tokenizers::models::bpe::Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, tok)| (tok.clone(), i as u32))
        .collect();
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .filter_map(|line| {
            let mut it = line.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("build BPE from GGUF vocab: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));

    // Register CONTROL tokens (chat-template markers like <|im_start|>, <|im_end|>,
    // <|endoftext|>) as added/special tokens so the pre-tokenizer doesn't byte-level
    // split them into single characters. Without this, the model never sees the
    // chat-template signals and produces gibberish for any instruct prompt.
    //
    // GGUF stores per-token category in `tokenizer.ggml.token_type` (parallel to
    // `tokenizer.ggml.tokens`). The GGUF spec uses:
    //   1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL, 4 = USER_DEFINED,
    //   5 = UNUSED, 6 = BYTE
    // Both CONTROL (3) and USER_DEFINED (4) should bypass BPE splitting.
    if let Some(MetaValue::Array(arr)) = raw.metadata.get("tokenizer.ggml.token_type") {
        let mut added: Vec<AddedToken> = Vec::new();
        for (idx, meta) in arr.iter().enumerate() {
            let Some(kind) = meta.as_u32() else { continue };
            if kind != 3 && kind != 4 {
                continue;
            }
            let Some(text) = tokens.get(idx) else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            added.push(AddedToken::from(text.clone(), kind == 3).normalized(false));
        }
        if !added.is_empty() {
            tok.add_special_tokens(&added);
        }
    }

    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[cfg(not(feature = "qwen35-tokenizer"))]
pub fn encode_prompt_from_gguf(_weights: &Path, _text: &str) -> Result<Vec<u32>> {
    anyhow::bail!("GGUF vocab-only tokenization needs feature `qwen35-tokenizer`")
}

/// Decode `ids` back to text via the HF tokenizer at `tokenizer_path`.
/// `skip_special_tokens=true` drops `<|im_end|>` / `<|endoftext|>` etc.
#[cfg(feature = "qwen35-tokenizer")]
pub fn decode_ids(tokenizer_path: &Path, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
    let tok = cached_tokenizer(tokenizer_path, || {
        let data = std::fs::read_to_string(tokenizer_path)
            .with_context(|| format!("read tokenizer {}", tokenizer_path.display()))?;
        tokenizers::Tokenizer::from_bytes(data.as_bytes())
            .map_err(|e| anyhow::anyhow!("parse tokenizer.json: {e}"))
    })?;
    tok.decode(ids, skip_special_tokens)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
}

#[cfg(not(feature = "qwen35-tokenizer"))]
pub fn decode_ids(
    _tokenizer_path: &Path,
    _ids: &[u32],
    _skip_special_tokens: bool,
) -> Result<String> {
    anyhow::bail!("tokenizer support not compiled in — rebuild with feature `qwen35-tokenizer`")
}

/// Mirror of [`encode_prompt_from_gguf`] for detokenization — uses the
/// GGUF-embedded byte-level BPE vocab to reconstruct text from ids.
#[cfg(feature = "qwen35-tokenizer")]
pub fn decode_ids_from_gguf(
    weights: &Path,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    let tok = cached_tokenizer(weights, || {
        use rlx_gguf::{GgufFile, MetaValue};
        use tokenizers::Tokenizer;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let raw = GgufFile::from_path(weights).with_context(|| format!("open GGUF {weights:?}"))?;
        let model_kind = raw
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(MetaValue::as_str)
            .unwrap_or("gpt2");
        if model_kind == "llama" || model_kind == "no_vocab" {
            anyhow::bail!(
                "GGUF tokenizer.ggml.model = `{model_kind}` (SentencePiece family) — \
             not supported in the vocab-only decode fallback; provide a tokenizer.json"
            );
        }

        let tokens_meta = raw
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.tokens"))?;
        let tokens: Vec<String> = match tokens_meta {
            MetaValue::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => anyhow::bail!("tokenizer.ggml.tokens not an array"),
        };
        let merges_meta = raw
            .metadata
            .get("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.merges"))?;
        let merges_raw: Vec<String> = match merges_meta {
            MetaValue::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => anyhow::bail!("tokenizer.ggml.merges not an array"),
        };

        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .filter_map(|line| {
                let mut it = line.splitn(2, ' ');
                Some((it.next()?.to_string(), it.next()?.to_string()))
            })
            .collect();

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| anyhow::anyhow!("build BPE from GGUF vocab: {e}"))?;
        let mut tok = Tokenizer::new(bpe);
        tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(
            false, true, true,
        )));

        // Register control / user-defined tokens so `skip_special_tokens` actually
        // strips them and the decoder doesn't byte-level-reverse `<|im_end|>` etc.
        // Mirror of the encoder change above.
        if let Some(MetaValue::Array(arr)) = raw.metadata.get("tokenizer.ggml.token_type") {
            use tokenizers::AddedToken;
            let mut added: Vec<AddedToken> = Vec::new();
            for (idx, meta) in arr.iter().enumerate() {
                let Some(kind) = meta.as_u32() else { continue };
                if kind != 3 && kind != 4 {
                    continue;
                }
                let Some(text) = tokens.get(idx) else {
                    continue;
                };
                if text.is_empty() {
                    continue;
                }
                added.push(AddedToken::from(text.clone(), kind == 3).normalized(false));
            }
            if !added.is_empty() {
                tok.add_special_tokens(&added);
            }
        }

        Ok(tok)
    })?;
    tok.decode(ids, skip_special_tokens)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
}

#[cfg(not(feature = "qwen35-tokenizer"))]
pub fn decode_ids_from_gguf(
    _weights: &Path,
    _ids: &[u32],
    _skip_special_tokens: bool,
) -> Result<String> {
    anyhow::bail!("GGUF vocab-only detokenization needs feature `qwen35-tokenizer`")
}

/// Mirror of [`encode_prompt_auto`] for detokenization.
pub fn decode_ids_auto(
    weights: &Path,
    explicit: Option<&Path>,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    if let Some(path) = resolve_tokenizer_path(weights, explicit) {
        return decode_ids(&path, ids, skip_special_tokens);
    }
    let is_gguf = weights
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if is_gguf {
        return decode_ids_from_gguf(weights, ids, skip_special_tokens);
    }
    anyhow::bail!(
        "no tokenizer found for {:?}. Pass an explicit path or place \
         tokenizer.json next to the file",
        weights
    )
}
