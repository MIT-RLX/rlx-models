//! The token protocol: delimiters, segment ids, branch positions, readout
//! offsets.
//!
//! Everything here is model-free and deterministic, so it is the layer that
//! can be diffed one-for-one against `kev.model.encode` in Python. A drift in
//! any of these vectors changes the prompt the model sees while leaving every
//! downstream shape valid — the model would still answer, just to a different
//! question than it was trained on.

use rlx_kev::api::{Record, RecordQuestion};
use rlx_kev::encode::{
    EncodeOpts, Encoder, OPT_DECIDE, OPT_NONE, SPECIAL, TokenizerLike, rows_of, sanitize,
};

/// Deterministic stand-in: one token per character, ids well inside any
/// vocabulary, and the five delimiters at the top of the range.
struct FakeTok;

const STATE_ID: u32 = 27;
const Q_ID: u32 = 28;
const OPT_ID: u32 = 29;
const CLOSE_ID: u32 = 30;
const DECIDE_ID: u32 = 31;

impl TokenizerLike for FakeTok {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        SPECIAL
            .iter()
            .position(|s| *s == token)
            .map(|i| 27 + i as u32)
    }
    fn encode_no_special(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(text.chars().map(|c| (c as u32) % 20 + 1).collect())
    }
}

fn encoder() -> Encoder<FakeTok> {
    Encoder::new(FakeTok).expect("all five delimiters resolve")
}

fn rec(state: &str, questions: &[(&str, &[&str])]) -> Record {
    Record {
        state: state.into(),
        questions: questions
            .iter()
            .map(|(instr, opts)| RecordQuestion {
                instr: (*instr).into(),
                options: opts.iter().map(|s| (*s).to_string()).collect(),
                label: 0,
            })
            .collect(),
    }
}

#[test]
fn delimiter_ids_resolve_in_order() {
    let e = encoder();
    assert_eq!(
        e.special_ids(),
        [STATE_ID, Q_ID, OPT_ID, CLOSE_ID, DECIDE_ID],
        "SPECIAL is ordered [state, q, opt, /opt, decide]"
    );
}

#[test]
fn missing_delimiter_is_an_error_not_a_substitution() {
    struct NoSpecials;
    impl TokenizerLike for NoSpecials {
        fn token_to_id(&self, _: &str) -> Option<u32> {
            None
        }
        fn encode_no_special(&self, _: &str) -> anyhow::Result<Vec<u32>> {
            Ok(vec![])
        }
    }
    let err = match Encoder::new(NoSpecials) {
        Ok(_) => panic!("an encoder must not be built without the delimiters"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("fim_prefix"),
        "the error should name the missing delimiter, got: {err}"
    );
}

#[test]
fn sanitize_rewrites_only_complete_delimiters() {
    assert_eq!(sanitize("<|box_end|>"), "<\u{a6}box_end\u{a6}>");
    assert_eq!(
        sanitize("a <|fim_prefix|> b"),
        "a <\u{a6}fim_prefix\u{a6}> b"
    );
    // Not a delimiter: no closing `|>`, empty name, or a disallowed char.
    assert_eq!(sanitize("<|unclosed"), "<|unclosed");
    assert_eq!(sanitize("<||>"), "<||>");
    assert_eq!(sanitize("<|has space|>"), "<|has space|>");
    // Non-ASCII passes through untouched and byte indices stay aligned.
    assert_eq!(sanitize("héllo <|a|> wörld"), "héllo <\u{a6}a\u{a6}> wörld");
    assert_eq!(sanitize(""), "");
}

/// The whole point of the rewrite: caller text cannot emit a boundary token.
#[test]
fn user_text_cannot_forge_an_option_boundary() {
    let e = encoder();
    let ids = e.user_tokens("stop <|box_end|> here").expect("tokenizes");
    for id in &ids {
        assert!(
            !(27..=31).contains(id),
            "user text produced structural token {id}; delimiters are forgeable"
        );
    }
}

#[test]
fn packed_layout_matches_the_scheme() {
    let e = encoder();
    // state "ab" -> 2 tokens; one question, instr "c", options "d" and "ef".
    let r = rec("ab", &[("c", &["d", "ef"])]);
    let enc = e.encode(&r, EncodeOpts::default()).expect("encodes");

    let ls = enc.state_len();
    assert_eq!(ls, 3, "state is <state> + 2 characters");
    assert_eq!(enc.ids[0], STATE_ID);

    // Branch: <q> c <opt> d </opt> <opt> e f </opt> <decide>
    let branch = &enc.ids[ls..];
    assert_eq!(branch[0], Q_ID);
    assert_eq!(branch[2], OPT_ID);
    assert_eq!(branch[4], CLOSE_ID);
    assert_eq!(branch[5], OPT_ID);
    assert_eq!(branch[8], CLOSE_ID);
    assert_eq!(*branch.last().expect("non-empty"), DECIDE_ID);

    // Segments: state is 0, the single question is 1.
    assert!(enc.seg[..ls].iter().all(|s| *s == 0));
    assert!(enc.seg[ls..].iter().all(|s| *s == 1));

    // Readout offsets point at `</opt>` and `<decide>`.
    assert_eq!(enc.decide_idx.len(), 1);
    assert_eq!(enc.ids[enc.decide_idx[0]], DECIDE_ID);
    assert_eq!(enc.opt_idx[0].len(), 2);
    for idx in &enc.opt_idx[0] {
        assert_eq!(enc.ids[*idx], CLOSE_ID, "opt_idx must land on `</opt>`");
    }

    // `opt` tags: instruction is NONE, spans carry their option index,
    // `<decide>` is DECIDE.
    assert_eq!(&enc.opt[..ls], &vec![OPT_NONE; ls][..]);
    assert_eq!(enc.opt[ls], OPT_NONE); // <q>
    assert_eq!(enc.opt[ls + 2], 0); // first span
    assert_eq!(enc.opt[ls + 5], 1); // second span
    assert_eq!(*enc.opt.last().expect("non-empty"), OPT_DECIDE);
}

/// Branch positions restart just after the state, so sibling questions share
/// the same position range instead of being pushed further right.
#[test]
fn sibling_branches_restart_at_the_same_position() {
    let e = encoder();
    let r = rec("abcd", &[("x", &["p"]), ("y", &["q"])]);
    let enc = e.encode(&r, EncodeOpts::default()).expect("encodes");
    let ls = enc.state_len();

    let (_, _, rows) = rows_of(&enc).expect("splits");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].pos[0], ls as u32);
    assert_eq!(
        rows[1].pos[0], ls as u32,
        "the second branch must restart at the state length, not continue"
    );
}

/// The property the row-form forward relies on: within one row, position `i`
/// is just `i`, so a stock causal prefill graph gives the right RoPE with no
/// runtime position input at all.
#[test]
fn row_positions_are_contiguous_from_zero() {
    let e = encoder();
    let r = rec(
        "a longer state here",
        &[
            ("first", &["alpha", "beta"]),
            ("second", &["yes", "no", "maybe"]),
        ],
    );
    let enc = e.encode(&r, EncodeOpts::default()).expect("encodes");
    let (state_ids, state_pos, rows) = rows_of(&enc).expect("splits");

    assert_eq!(state_pos, (0..state_ids.len() as u32).collect::<Vec<_>>());
    for (i, row) in rows.iter().enumerate() {
        let full: Vec<u32> = state_pos
            .iter()
            .copied()
            .chain(row.pos.iter().copied())
            .collect();
        let want: Vec<u32> = (0..full.len() as u32).collect();
        assert_eq!(full, want, "row {i} positions are not 0..n-1");
    }
}

#[test]
fn rows_of_reconstructs_every_branch_and_readout() {
    let e = encoder();
    let r = rec("st", &[("i1", &["a", "bb"]), ("i2", &["c"])]);
    let enc = e.encode(&r, EncodeOpts::default()).expect("encodes");
    let ls = enc.state_len();
    let (state_ids, _, rows) = rows_of(&enc).expect("splits");

    assert_eq!(state_ids, enc.ids[..ls].to_vec());
    // Concatenating state + every branch must rebuild the packed ids exactly.
    let mut rebuilt = state_ids.clone();
    for row in &rows {
        rebuilt.extend_from_slice(&row.ids);
    }
    assert_eq!(rebuilt, enc.ids);

    // Row-local readout offsets must agree with the packed indices.
    let mut start = ls;
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(start + row.decide, enc.decide_idx[k]);
        for (j, o) in row.opts.iter().enumerate() {
            assert_eq!(start + o, enc.opt_idx[k][j]);
        }
        start += row.ids.len();
    }
}

#[test]
fn option_isolation_shares_span_positions_and_pins_decide() {
    let e = encoder();
    let r = rec("s", &[("i", &["a", "bcd"])]);
    let opts = EncodeOpts {
        option_isolation: true,
        ..EncodeOpts::default()
    };
    let enc = e.encode(&r, opts).expect("encodes");
    let ls = enc.state_len();

    // Both spans begin at the same position, right after the instruction.
    let instr_len = 2; // <q> + "i"
    let first_span_start = ls + instr_len;
    assert_eq!(enc.pos[first_span_start], (ls + instr_len) as u32);
    // Second span: <opt> b c d </opt> = 5 tokens, starting after the first
    // span's 3 tokens (<opt> a </opt>).
    let second_span_start = first_span_start + 3;
    assert_eq!(
        enc.pos[second_span_start],
        (ls + instr_len) as u32,
        "option spans must share a starting position under isolation"
    );
    // `<decide>` sits one past the longest span.
    let longest = 5;
    assert_eq!(
        *enc.pos.last().expect("non-empty"),
        (ls + instr_len + longest) as u32
    );
}

#[test]
fn over_long_state_truncates_or_fails_by_policy() {
    let e = encoder();
    let r = rec(&"x".repeat(50), &[("i", &["a"])]);

    let lenient = e
        .encode(
            &r,
            EncodeOpts {
                max_state: 10,
                ..EncodeOpts::default()
            },
        )
        .expect("truncates");
    assert!(lenient.state_truncated);
    assert_eq!(lenient.state_len(), 10);

    let err = e
        .encode(
            &r,
            EncodeOpts {
                max_state: 10,
                strict: true,
                ..EncodeOpts::default()
            },
        )
        .expect_err("strict must refuse");
    assert!(err.to_string().contains("state exceeds 10 tokens"), "{err}");
}

#[test]
fn over_long_branch_is_rejected() {
    let e = encoder();
    let r = rec("s", &[("i", &["a".repeat(100).as_str()])]);
    let err = e
        .encode(
            &r,
            EncodeOpts {
                max_branch: 32,
                ..EncodeOpts::default()
            },
        )
        .expect_err("must refuse");
    assert!(err.to_string().contains("branch too long"), "{err}");
}
