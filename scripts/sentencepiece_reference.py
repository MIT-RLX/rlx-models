#!/usr/bin/env python3
"""Reference outputs of the C++ SentencePiece, for
`crates/rlx-models-core/tests/sentencepiece_parity.rs`.

    python scripts/sentencepiece_reference.py OUT_DIR [MODEL.model ...] [--train]

Writes `OUT_DIR/<name>.json` per model with, for a fixed set of texts:

- `cases`: ids, pieces (piece, id, surface, begin, end) and decoded text;
- `extra`: ids and piece strings under add_bos / add_eos / reverse /
  emit_unk_piece;
- `normalize`: normalized text and its offsets (in characters, as the
  binding reports them);
- `nbest`: the 8 best segmentations with scores (unigram);
- `decode`: decoded text and per-piece spans for id lists that start with
  control, unknown, byte and whitespace pieces;
- `sample`: segmentation distributions of `enable_sampling` encoding, exact
  for unigram (softmax of alpha * score over every segmentation) and from
  20000 draws of the C++ for BPE-dropout.

`--train` also trains small models covering what the given ones may not: an
`nmt_nfkc` unigram (precompiled charsmap), `nmt_nfkc_cf` with a user-defined
symbol, user-defined symbols competing with learned pieces, BPE with and
without byte fallback, unigram keeping whitespace,
suffix whitespace, a denormalization rule, char and word. Then

    RLX_SPM_REFERENCE_DIR=OUT_DIR cargo test -p rlx-models-core --test sentencepiece_parity

Needs `pip install sentencepiece` (0.2.x).
"""

import argparse
import json
import math
import os
from collections import Counter

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
    "le chat chante",
    "jouent dans le jardin, garçon, printemps <é> automne",
]

# Short enough that n-best lists every segmentation.
SAMPLE_TEXTS = ["bonjour", "le chat", "hello"]
SAMPLE_DRAWS = 20000

CORPUS = """\
Le chat est sur le tapis. Les enfants jouent dans le jardin.
Où est la gare ? Je voudrais un café, s'il vous plaît.
The quick brown fox jumps over the lazy dog. 1234567890.
ＦＵＬＬＷＩＤＴＨ text ① ﬁ ㎏ appears here too, «guillemets» et — tirets.
Naïve cœur Noël façade garçon déjà vu été hiver printemps automne.
"""


def spans(pieces):
    return [[p.piece, p.id, p.surface, p.begin, p.end] for p in pieces]


def dump(model_path, out_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    try:
        sp.nbest_encode("a", nbest_size=2)
        unigram = True
    except Exception:
        unigram = False
    ref = {"model": os.path.abspath(model_path), "cases": [], "extra": [], "normalize": [],
           "nbest": [], "decode": [], "sample": []}
    for t in TEXTS:
        p = sp.encode_as_proto(t)
        ids = [x.id for x in p.pieces]
        ref["cases"].append({"text": t, "ids": ids, "pieces": spans(p.pieces), "decoded": sp.decode(ids)})
        norm, offs = sp.normalize(t, with_offsets=True)
        ref["normalize"].append({"text": t, "normalized": norm, "offsets": list(offs)})
        for opts in [dict(add_bos=True), dict(add_eos=True), dict(add_bos=True, add_eos=True, reverse=True),
                     dict(reverse=True), dict(emit_unk_piece=True)]:
            try:
                e_ids = sp.encode(t, **opts)
                e_str = sp.encode(t, out_type=str, **opts)
            except Exception as err:  # e.g. no <s> piece
                ref["extra"].append({"text": t, "opts": opts, "error": str(err)})
                continue
            ref["extra"].append({"text": t, "opts": opts, "ids": e_ids, "pieces": e_str})
        if unigram:
            nb = sp.nbest_encode_as_proto(t, nbest_size=8)
            ref["nbest"].append({"text": t, "n": 8, "results": [
                {"ids": [x.id for x in r.pieces], "score": r.score} for r in nb.nbests]})
        if ids:
            space = sp.piece_to_id("▁")
            prefixes = [[], [sp.unk_id()], [space, space]]
            if sp.bos_id() >= 0:
                prefixes += [[sp.bos_id()], [sp.bos_id(), space]]
            if sp.is_byte(sp.piece_to_id("<0x41>")):
                prefixes += [[sp.piece_to_id("<0x41>")], [sp.piece_to_id("<0xC3>")]]
            suffix = [sp.eos_id()] if sp.eos_id() >= 0 else []
            for pre in prefixes:
                d_ids = pre + ids[:12] + suffix
                d = sp.decode_ids_as_proto(d_ids)
                ref["decode"].append({"ids": d_ids, "text": d.text, "pieces": spans(d.pieces)})
    for t in SAMPLE_TEXTS:
        settings = [(-1, 0.5), (-1, 0.1), (4, 0.5)] if unigram else [(-1, 0.1), (-1, 0.5)]
        for nbest_size, alpha in settings:
            if unigram:
                k = 512 if nbest_size < 0 else nbest_size
                nb = sp.nbest_encode_as_proto(t, nbest_size=k).nbests
                if nbest_size < 0 and len(nb) >= 512:
                    continue
                top = max(alpha * r.score for r in nb)
                w = {json.dumps([x.id for x in r.pieces]): math.exp(alpha * r.score - top) for r in nb}
                z = sum(w.values())
                dist = {key: v / z for key, v in w.items()}
                exact = True
            else:
                try:
                    c = Counter(json.dumps(sp.encode(t, enable_sampling=True, nbest_size=nbest_size,
                                                     alpha=alpha)) for _ in range(SAMPLE_DRAWS))
                except RuntimeError as err:  # char / word models
                    ref["sample"].append({"text": t, "nbest_size": nbest_size, "alpha": alpha,
                                          "error": str(err)})
                    continue
                dist = {key: v / SAMPLE_DRAWS for key, v in c.items()}
                exact = False
            ref["sample"].append({"text": t, "nbest_size": nbest_size, "alpha": alpha,
                                  "exact": exact, "dist": dist})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False)
    print(f"{out_path}: {len(ref['cases'])} texts, unigram={unigram}")


def train(out_dir):
    corpus = os.path.join(out_dir, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(20):
            f.write(CORPUS)
    denorm = os.path.join(out_dir, "denorm.tsv")
    with open(denorm, "w") as f:
        f.write("61\t41\n")  # a -> A when decoding
        f.write("63 68\t43 48\n")  # ch -> CH
    models = []
    for name, extra in [
        ("unigram_nfkc", dict(model_type="unigram", normalization_rule_name="nmt_nfkc")),
        ("unigram_nfkc_cf", dict(model_type="unigram", normalization_rule_name="nmt_nfkc_cf",
                                 user_defined_symbols=["<sep>"])),
        ("bpe_bytes", dict(model_type="bpe", byte_fallback=True, user_defined_symbols=["<sep>"])),
        ("bpe", dict(model_type="bpe")),
        ("unigram_keep_ws", dict(model_type="unigram", remove_extra_whitespaces=False, byte_fallback=True)),
        ("unigram_suffix", dict(model_type="unigram", treat_whitespace_as_suffix=True)),
        ("unigram_denorm", dict(model_type="unigram", denormalization_rule_tsv=denorm)),
        # User-defined symbols that compete with learned pieces, one non-ASCII.
        ("unigram_user_defined", dict(model_type="unigram",
                                      user_defined_symbols=["<é>", "an", "ou", "le", "in", "ar"])),
        ("char", dict(model_type="char")),
        ("word", dict(model_type="word")),
    ]:
        prefix = os.path.join(out_dir, name)
        vocab = 400 if extra.get("byte_fallback") else 150
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=vocab,
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
