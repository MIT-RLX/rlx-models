#!/usr/bin/env python3
"""Fixture for `rlx-kittentts2`'s text normalization parity test.

Runs `kittenml.kittentts2.text.normalize_text` (which wraps
`kitten_text_processing` 0.1.0) over a corpus of written-form English and
writes `{input, normalized, raw}` triples, `raw` being the `normalize=False`
path:

    .venv-kitten2/bin/python scripts/kittentts2_normalize_fixture.py \
        crates/rlx-kittentts2/tests/fixtures/normalize_cases.json
"""

import json
import random
import sys

from kittenml.kittentts2 import text as t

FIXED = [
    "I paid $12.50 for 3 books.",
    "The meeting is on January 5th, 2024 at 3:30 PM.",
    "Call me at 555-123-4567 tomorrow.",
    "It costs €5 or £3.99, depending.",
    "We ran 10km in 45 minutes.",
    "The temperature dropped to -5°C overnight.",
    "Only 25% of people agreed.",
    "Chapter IV begins on page 112.",
    "He finished 1st, she came 2nd and I was 23rd.",
    "Add 1/2 cup of sugar and 3/4 tsp of salt.",
    "Dr. Smith lives at 221B Baker St.",
    "The year 1999 was wild; 2000 was calmer.",
    "Visit https://example.com or email me at john.doe@example.org.",
    "The ratio is 3:2 and pi is about 3.14159.",
    "It weighs 2.5 kg and is 30 cm long.",
    "Version 2.1.3 ships on 12/25/2025.",
    "I have 1,234,567 reasons.",
    "[joyful] We won the grant <laugh> I can (((hardly))) believe it!",
    "The 90s were great, the 1980s even better.",
    "Mr. and Mrs. Jones arrived at 7am.",
    "Flight AA123 departs at 06:45.",
    "That's 100 mph!",
    "Room 404 — not found.",
    "He said \"12 test\" then left.",
    "It was 5' 11\" tall.",
    "Prices rose 3.5x in 2 years.",
    "We need 3 x 4 meters.",
    "Is it 2? Or 3!",
    "I'm 21 years old.",
    "Nothing to normalize here at all.",
    "e.g. this, i.e. that, etc.",
    "A $1.5 million deal and a $2B fund.",
    "On 2023-10-05 we shipped.",
    "Set the timer for 1h 30m.",
    "Section [3] says x < 5.",
    "We got 0.5 points.",
    "Order #42 is ready.",
    "It is -12 degrees.",
    "1 apple, 2 apples.",
    "The 4th of July.",
]

WORDS = ["the", "price", "is", "we", "saw", "about", "on", "at", "costs", "only", "and", "for"]


def random_case(rng):
    pieces = []
    for _ in range(rng.randint(3, 9)):
        r = rng.random()
        if r < 0.45:
            pieces.append(rng.choice(WORDS))
        elif r < 0.6:
            pieces.append(str(rng.randint(0, 10 ** rng.randint(1, 7))))
        elif r < 0.68:
            pieces.append(f"${rng.randint(0, 999)}.{rng.randint(0, 99):02d}")
        elif r < 0.75:
            pieces.append(f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}")
        elif r < 0.82:
            pieces.append(f"{rng.randint(1, 99)}%")
        elif r < 0.88:
            pieces.append(f"{rng.randint(1, 31)}/{rng.randint(1, 12)}/{rng.randint(1900, 2099)}")
        elif r < 0.94:
            pieces.append(f"{rng.uniform(0, 100):.{rng.randint(1, 3)}f}")
        else:
            pieces.append(rng.choice(["1st", "2nd", "3rd", "11th", "22nd", "101st"]))
    s = " ".join(pieces)
    return s[0].upper() + s[1:] + rng.choice([".", "!", "?", ""])


def main():
    rng = random.Random(7)
    cases = FIXED + [random_case(rng) for _ in range(160)]
    out = [{"input": c,
            "normalized": t.normalize_text(c, normalize=True),
            "raw": t.normalize_text(c, normalize=False)} for c in cases]
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print(f"{len(out)} cases")


if __name__ == "__main__":
    main()
