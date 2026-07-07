#!/usr/bin/env python3
"""Convert MOSS-TTS-Nano's `tokenizer.model` (a BPE SentencePiece model) to a
pure-Rust-loadable `tokenizer.json`.

We deliberately avoid the C++ `sentencepiece` crate at runtime: it statically
links its own protobuf, which clashes (ODR) with ONNX Runtime's bundled protobuf
and corrupts ORT model loading ("Protobuf parsing failed"). Converting to a
`tokenizer.json` lets the Rust `tokenizers` crate (pure Rust) handle text.

  pip install sentencepiece tokenizers          # in a venv
  python convert_tokenizer.py weights/tts/moss-nano

Verified bit-exact against the manifest's pre-encoded `text_samples`.
"""
import json
import sys

from sentencepiece import sentencepiece_model_pb2 as spb
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers


def main(model_dir: str) -> None:
    proto = spb.ModelProto()
    proto.ParseFromString(open(f"{model_dir}/tokenizer.model", "rb").read())
    assert proto.trainer_spec.model_type == 2, "expected a BPE SentencePiece model"

    vocab_scores = [(p.piece, p.score) for p in proto.pieces]
    vocab = {tok: i for i, (tok, _) in enumerate(vocab_scores)}

    # Reconstruct BPE merges from the SPM pieces (transformers' SpmExtractor algorithm).
    merges = []
    for piece, score in vocab_scores:
        local = []
        for i in range(1, len(piece)):
            left, right = piece[:i], piece[i:]
            if left in vocab and right in vocab:
                local.append((left, right, score))
        local.sort(key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)
    merges = [(a, b) for a, b, _ in sorted(merges, key=lambda v: v[2], reverse=True)]

    tok = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tok.normalizer = normalizers.Sequence([
        normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap),  # nmt_nfkc
        normalizers.Replace(Regex(r" {2,}"), " "),
    ])
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    # Verify bit-exact against the manifest.
    man = json.load(open(f"{model_dir}/browser_poc_manifest.json"))
    for s in man["text_samples"]:
        got = tok.encode(s["text"]).ids
        assert got == s["text_token_ids"], f"mismatch on {s['id']}"

    tok.save(f"{model_dir}/tokenizer.json")
    print(f"saved {model_dir}/tokenizer.json (verified exact)")


if __name__ == "__main__":
    main(sys.argv[1])
