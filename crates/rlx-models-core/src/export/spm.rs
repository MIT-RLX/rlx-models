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

//! A SentencePiece unigram `.model` as a Hugging Face `tokenizer.json`, the
//! one tokenizer format a package's program reads: each piece and its score
//! (`ModelProto.pieces`, read from the protobuf directly), `▁` prepended and
//! digits split one by one (`add_dummy_prefix`, `split_digits`, the identity
//! normalizer: Kyutai's and Moshi's models), bytes past the vocabulary as
//! `<0x..>` pieces.

use anyhow::{Result, bail};
use serde_json::json;

/// One varint at `at`, and the offset past it.
fn varint(b: &[u8], mut at: usize) -> Result<(u64, usize)> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&byte) = b.get(at) else {
            bail!("a varint runs past the end");
        };
        at += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((v, at));
        }
    }
    bail!("a varint longer than 64 bits")
}

/// Each `(field, wire type, payload)` of a protobuf message.
fn fields(b: &[u8]) -> Result<Vec<(u64, u64, &[u8], u64)>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < b.len() {
        let (key, next) = varint(b, at)?;
        at = next;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                let (v, next) = varint(b, at)?;
                at = next;
                out.push((field, wire, &b[0..0], v));
            }
            1 => {
                out.push((field, wire, &b[at..at + 8], 0));
                at += 8;
            }
            2 => {
                let (len, next) = varint(b, at)?;
                let end = next + len as usize;
                if end > b.len() {
                    bail!("a field runs past the end");
                }
                out.push((field, wire, &b[next..end], 0));
                at = end;
            }
            5 => {
                out.push((field, wire, &b[at..at + 4], 0));
                at += 4;
            }
            w => bail!("wire type {w} in a SentencePiece model"),
        }
    }
    Ok(out)
}

/// The `tokenizer.json` for the SentencePiece unigram model `model`.
pub fn tokenizer_json(model: &[u8]) -> Result<String> {
    let mut vocab = Vec::new();
    let mut unk = 0usize;
    for (field, wire, payload, _) in fields(model)? {
        if field != 1 || wire != 2 {
            continue;
        }
        let (mut piece, mut score, mut kind) = (String::new(), 0f32, 1u64);
        for (f, w, p, v) in fields(payload)? {
            match (f, w) {
                (1, 2) => piece = String::from_utf8_lossy(p).into_owned(),
                (2, 5) => score = f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                (3, 0) => kind = v,
                _ => {}
            }
        }
        // Type 2 is the unknown piece.
        if kind == 2 {
            unk = vocab.len();
        }
        vocab.push(json!([piece, score]));
    }
    if vocab.is_empty() {
        bail!("no pieces: not a SentencePiece model");
    }
    let tok = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always",
             "split": true},
            {"type": "Digits", "individual_digits": true},
        ]},
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ]},
        "model": {"type": "Unigram", "unk_id": unk, "vocab": vocab, "byte_fallback": true},
    });
    Ok(serde_json::to_string(&tok)?)
}
