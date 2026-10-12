"""Regenerate crates/rlx-kev/tests/fixtures/pyrandom_golden.json.

The kev recipe seeds a `random.Random` per record and consumes its stream in a
fixed order, so reproducing the training data means reproducing CPython's
PRNG exactly. These vectors pin `crates/rlx-kev/src/pyrandom.rs` against the
real thing.

    python3 scripts/kev_pyrandom_golden.py
"""
import json, random, hashlib, struct


def source_seed(seed, tag):
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


out = {
    "_provenance": "CPython random.Random golden vectors. Regenerate with scripts/kev_pyrandom_golden.py.",
    "cases": [],
}
for seed in [0, 1, 7, 12345, 2**31, 2**40 + 9,
             source_seed(0, "0:yelp/train/1"), source_seed(1, "3:agnews/train/77")]:
    r = random.Random(seed)
    # Stored as IEEE-754 bit patterns: a decimal round-trip through JSON can
    # land 1 ULP away, and this test is about bit-identity.
    randoms = [struct.unpack("<Q", struct.pack("<d", r.random()))[0] for _ in range(6)]
    bits = [r.getrandbits(k) for k in (1, 3, 8, 17, 32, 33, 53, 64)]
    below = [r._randbelow(n) for n in (1, 2, 3, 5, 13, 255, 1000)]
    lst = list(range(12))
    r.shuffle(lst)
    choices = [r._randbelow(13) for _ in range(5)]
    out["cases"].append({"seed": seed, "random_bits": randoms, "getrandbits": bits,
                         "randbelow": below, "shuffled": lst, "choice_index": choices})
out["source_seed"] = [{"seed": s, "tag": t, "value": source_seed(s, t)}
                      for s, t in [(0, "0:a"), (1, "3:agnews/train/77"), (7, "public_frac"), (0, "replay:0")]]
json.dump(out, open("crates/rlx-kev/tests/fixtures/pyrandom_golden.json", "w"), indent=1)
print("wrote", len(out["cases"]), "cases")
