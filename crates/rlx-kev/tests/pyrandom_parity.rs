//! `pyrandom` must be **bit-identical** to CPython's `random.Random`.
//!
//! The recipe seeds a generator per record and consumes it in a fixed order,
//! so a different stream yields a different (equally valid) dataset — and then
//! any metric difference against the published numbers is unattributable.
//! Vectors come from CPython itself (`scripts/kev_pyrandom_golden.py`).

use rlx_kev::pyrandom::{PyRandom, source_seed};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/pyrandom_golden.json")).expect("fixture parses")
}

#[test]
fn source_seed_matches_cpython() {
    for c in fixture()["source_seed"].as_array().expect("array") {
        let seed = c["seed"].as_u64().expect("seed");
        let tag = c["tag"].as_str().expect("tag");
        let want = c["value"].as_u64().expect("value");
        assert_eq!(
            source_seed(seed, tag),
            want,
            "source_seed({seed}, {tag:?}) — sha256 prefix read big-endian"
        );
    }
}

#[test]
fn the_whole_stream_matches_cpython() {
    for c in fixture()["cases"].as_array().expect("cases") {
        let seed = c["seed"].as_u64().expect("seed");
        let mut r = PyRandom::seed_u64(seed);

        // Compared as bit patterns: a decimal round-trip through JSON can
        // land 1 ULP away, and the claim here is bit-identity.
        for (i, v) in c["random_bits"]
            .as_array()
            .expect("random_bits")
            .iter()
            .enumerate()
        {
            let want = f64::from_bits(v.as_u64().expect("u64"));
            let got = r.random();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "seed {seed}: random() #{i} = {got:?}, want {want:?} \
                 — a 53-bit double from two words"
            );
        }
        for (k, v) in [1u32, 3, 8, 17, 32, 33, 53, 64]
            .iter()
            .zip(c["getrandbits"].as_array().expect("getrandbits"))
        {
            assert_eq!(
                r.getrandbits(*k),
                v.as_u64().expect("u64"),
                "seed {seed}: getrandbits({k})"
            );
        }
        for (n, v) in [1u64, 2, 3, 5, 13, 255, 1000]
            .iter()
            .zip(c["randbelow"].as_array().expect("randbelow"))
        {
            assert_eq!(
                r.randbelow(*n),
                v.as_u64().expect("u64"),
                "seed {seed}: _randbelow({n}) — the rejection loop must consume \
                 the same number of words"
            );
        }
        let mut lst: Vec<u64> = (0..12).collect();
        r.shuffle(&mut lst);
        let want: Vec<u64> = c["shuffled"]
            .as_array()
            .expect("shuffled")
            .iter()
            .map(|x| x.as_u64().expect("u64"))
            .collect();
        assert_eq!(lst, want, "seed {seed}: shuffle()");

        for (i, v) in c["choice_index"]
            .as_array()
            .expect("choice_index")
            .iter()
            .enumerate()
        {
            assert_eq!(
                r.choice_index(13) as u64,
                v.as_u64().expect("u64"),
                "seed {seed}: choice #{i}"
            );
        }
    }
}

/// The recipe's per-record seed is `source_seed(seed, "{epoch}:{id}")`.
#[test]
fn per_record_seeding_is_reproducible_and_distinct() {
    let a = PyRandom::from_source_seed(0, "0:yelp/train/1").random();
    let b = PyRandom::from_source_seed(0, "0:yelp/train/1").random();
    let c = PyRandom::from_source_seed(0, "1:yelp/train/1").random();
    let d = PyRandom::from_source_seed(1, "0:yelp/train/1").random();
    assert_eq!(a, b, "the same (seed, tag) must replay identically");
    assert_ne!(a, c, "a different epoch must give a different stream");
    assert_ne!(a, d, "a different run seed must give a different stream");
}
