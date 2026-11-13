#!/usr/bin/env python3
"""Reference ids / decoded text from the C++ SentencePiece, for
`crates/rlx-models-core/tests/sentencepiece_parity.rs`.

    python scripts/sentencepiece_reference.py OUT_DIR [MODEL.model ...]

Writes `OUT_DIR/<name>.json` per model: `{"model": path, "cases": [{"text",
"ids", "decoded"}]}`. With `--train`, also trains small models next to the
references to cover what Pocket TTS's identity-normalized unigram models do
not: an `nmt_nfkc` unigram (precompiled charsmap), a BPE and a user-defined
symbol, then

    RLX_SPM_REFERENCE_DIR=OUT_DIR cargo test -p rlx-models-core --test sentencepiece_parity

Needs `pip install sentencepiece`.
"""

import argparse
import json
import os

import sentencepiece as spm

TEXTS = [
    "",
    " ",
    "Bonjour, je m'appelle Marie et j'habite à Paris.",
    "        Où est la bibliothèque ?",
    "L'été dernier, nous sommes allés à la plage — c'était magnifique !",
    "Ça coûte 3,14 € ou 1 234,56 $… vraiment ?",
    "Œuvre, cœur, naïve, Noël, façade, déjà-vu, garçon.",
    "«Bien sûr», répondit-il. « Pourquoi pas ? »",
    "Hello world. This is a test, with numbers 12345 and 3.14.",
    "Multiple   spaces\tand\ttabs\nand newlines\r\n here.",
    "ＦＵＬＬＷＩＤＴＨ ｔｅｘｔ ① ﬁ ligature ㎏",
    "Emoji 🎉🙂 and CJK 漢字かな 한국어 and Ελληνικά, русский.",
    "Zero​width and combining é vs é.",
    "<sep> user<sep>defined <sep><sep> symbols",
    "trailing spaces   ",
    "a" * 300,
    "!!!???...,,,;;;:::",
    " non breaking spaces　ideographic",
]

CORPUS = """\
Le chat est sur le tapis. Les enfants jouent dans le jardin.
Où est la gare ? Je voudrais un café, s'il vous plaît.
The quick brown fox jumps over the lazy dog. 1234567890.
ＦＵＬＬＷＩＤＴＨ text ① ﬁ ㎏ appears here too, «guillemets» et — tirets.
Naïve cœur Noël façade garçon déjà vu été hiver printemps automne.
"""


def dump(model_path, out_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    cases = []
    for t in TEXTS:
        ids = sp.encode(t)
        cases.append({"text": t, "ids": ids, "decoded": sp.decode(ids)})
    # Decode-only: what precedes the first piece changes whether its
    # dummy-prefix marker is stripped.
    word = sp.encode("bonjour le monde")
    space = sp.piece_to_id("\u2581")
    prefixes = [[sp.bos_id()], [sp.unk_id()], [space, space], [sp.bos_id(), space]]
    if sp.piece_to_id("<0x41>") != sp.unk_id():
        prefixes.append([sp.piece_to_id("<0x41>")])
        prefixes.append([sp.piece_to_id("<0xC3>")])
    for pre in prefixes:
        ids = [i for i in pre if i >= 0] + word
        cases.append({"text": None, "ids": ids, "decoded": sp.decode(ids)})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": os.path.abspath(model_path), "cases": cases}, f, ensure_ascii=False)
    print(f"{out_path}: {len(cases)} cases")


def train(out_dir):
    corpus = os.path.join(out_dir, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(20):
            f.write(CORPUS)
    models = []
    for name, extra in [
        ("unigram_nfkc", dict(model_type="unigram", normalization_rule_name="nmt_nfkc")),
        ("unigram_nfkc_cf", dict(model_type="unigram", normalization_rule_name="nmt_nfkc_cf",
                                 user_defined_symbols=["<sep>"])),
        ("bpe_bytes", dict(model_type="bpe", byte_fallback=True, user_defined_symbols=["<sep>"])),
        ("unigram_keep_ws", dict(model_type="unigram", remove_extra_whitespaces=False,
                                 byte_fallback=True)),
        ("unigram_suffix", dict(model_type="unigram", treat_whitespace_as_suffix=True)),
        ("char", dict(model_type="char")),
        ("word", dict(model_type="word")),
    ]:
        prefix = os.path.join(out_dir, name)
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=prefix, vocab_size=400 if "byte" in name or extra.get("byte_fallback") else 150,
            hard_vocab_limit=False, minloglevel=2, **extra)
        models.append(prefix + ".model")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="*")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    models = list(a.models) + (train(a.out_dir) if a.train else [])
    for m in models:
        stem = os.path.splitext(os.path.basename(m))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(m)))
        dump(m, os.path.join(a.out_dir, f"{parent}-{stem}.json"))


if __name__ == "__main__":
    main()
