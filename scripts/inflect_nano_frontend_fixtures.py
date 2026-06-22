#!/usr/bin/env python3
"""Frontend parity fixtures: dump every text-pipeline stage from the reference
so the Rust port can be validated stage-by-stage.

  - normalize_corpus.json : [input -> normalized] over a large numeric corpus
                            (fast; exercises numbers/ordinals/currency/time)
  - pipeline_corpus.json  : full clean→normalize→g2p→ids over natural sentences

    .venv-inflect/bin/python scripts/inflect_nano_frontend_fixtures.py \
        --repo /tmp/inflect-nano --out weights/inflect-nano-rlx/fixtures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_numeric_corpus() -> list[str]:
    items: list[str] = []
    items += [str(n) for n in range(0, 2101)]          # cardinals incl. year rules
    items += [str(n) for n in range(0, 130, 7)]
    items += [f"{n}st" for n in (1, 21, 31, 101, 121)]
    items += [f"{n}nd" for n in (2, 22, 42, 102)]
    items += [f"{n}rd" for n in (3, 23, 43, 103)]
    items += [f"{n}th" for n in (4, 5, 11, 12, 13, 20, 100, 111, 1000)]
    items += ["$42.50", "$1,234.56", "$0.01", "$0.99", "$1.00", "£5.50", "¥100", "$7"]
    items += ["2.5", "3.14159", "0.5", "100,000", "1,000,000", "12,345"]
    items += ["3:15 pm", "12:00 am", "9:05", "23:59", "1:30 PM", "00:00", "11:11 am"]
    items += ["-5", "-100", "minus 3"]
    return items


PIPELINE_CORPUS = [
    "The weather is nice today, and I feel very relaxed.",
    "Hello, world!",
    "Dr. Smith paid $42.50 at 3:15 pm.",
    "In 1999 we found 1024 reasons.",
    "Zephyrization glorptastic woogle.",
    "It costs 7 dollars, maybe 8.",
    "Mr. and Mrs. Johnson live on St. Paul Street.",
    "The 23rd of December, 2024.",
    "She owns 3 cats, 12 dogs, and 100 fish.",
    "Call me at 9:45 am or 5:30 pm.",
    "Lt. Col. Brown served for 21 years.",
    "Quizzically, the xylophone hummed.",
    "$1,234.56 was the total cost.",
    "Wow... that's amazing!!!",
    "The quick brown fox jumps over the lazy dog.",
    "antidisestablishmentarianism",
    "Pneumonoultramicroscopicsilicovolcanoconiosis is long.",
    "I'll see you at 12:00 pm sharp.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))

    from tiny_tts.nn import commons
    from tiny_tts.text import phonemes_to_ids
    from tiny_tts.text.english import grapheme_to_phoneme, normalize_text
    from tiny_tts.utils import ADD_BLANK
    from tinytts_text_cleaning import clean_tinytts_text

    # normalize-only corpus (no g2p — fast)
    norm = []
    for s in build_numeric_corpus():
        cleaned = clean_tinytts_text(s)
        norm.append({"input": s, "cleaned": cleaned, "normalized": normalize_text(cleaned)})
    (args.out / "normalize_corpus.json").write_text(json.dumps(norm, indent=0), encoding="utf-8")
    print(f"normalize_corpus: {len(norm)} cases")

    # full pipeline corpus
    pipe = []
    for s in PIPELINE_CORPUS:
        cleaned = clean_tinytts_text(s)
        normalized = normalize_text(cleaned)
        phones, tones, _ = grapheme_to_phoneme(normalized)
        phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
        if ADD_BLANK:
            phone_ids = commons.insert_blanks(phone_ids, 0)
            tone_ids = commons.insert_blanks(tone_ids, 0)
            lang_ids = commons.insert_blanks(lang_ids, 0)
        pipe.append({
            "input": s,
            "cleaned": cleaned,
            "normalized": normalized,
            "phones": phones,
            "tones": tones,
            "phone_ids": [int(x) for x in phone_ids],
            "tone_ids": [int(x) for x in tone_ids],
            "lang_ids": [int(x) for x in lang_ids],
        })
    (args.out / "pipeline_corpus.json").write_text(json.dumps(pipe, indent=0), encoding="utf-8")
    print(f"pipeline_corpus: {len(pipe)} cases")


if __name__ == "__main__":
    main()
