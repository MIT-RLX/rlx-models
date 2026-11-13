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

use super::*;

fn model(pieces: &[(&str, f32, PieceType)], ty: ModelType, byte_fallback: bool) -> SentencePiece {
    let mut all: Vec<Piece> = pieces
        .iter()
        .map(|&(p, s, k)| Piece {
            piece: p.into(),
            score: s,
            kind: k,
        })
        .collect();
    if byte_fallback {
        all.extend((0..=255u8).map(|b| Piece {
            piece: byte_piece(b),
            score: 0.0,
            kind: PieceType::Byte,
        }));
    }
    let trainer = TrainerSpec {
        model_type: ty,
        byte_fallback,
        ..TrainerSpec::default()
    };
    let normalizer = NormalizerSpec {
        name: "identity".into(),
        ..NormalizerSpec::default()
    };
    SentencePiece::new(all, trainer, normalizer, None).unwrap()
}

fn base() -> Vec<(&'static str, f32, PieceType)> {
    vec![
        ("<unk>", 0.0, PieceType::Unknown),
        ("<s>", 0.0, PieceType::Control),
        ("</s>", 0.0, PieceType::Control),
        ("▁", -2.0, PieceType::Normal),
        ("a", -3.0, PieceType::Normal),
        ("b", -3.0, PieceType::Normal),
        ("ab", -1.0, PieceType::Normal),
        ("▁ab", -1.5, PieceType::Normal),
        ("c", -4.0, PieceType::Normal),
        ("▁a", -2.5, PieceType::Normal),
    ]
}

#[test]
fn normalizer_collapses_and_escapes_whitespace() {
    let sp = model(&base(), ModelType::Unigram, false);
    assert_eq!(sp.normalize("  ab   c  "), "▁ab▁c");
    assert_eq!(sp.normalize(""), "");
    assert_eq!(sp.normalize("   "), "");
    let (n, offs) = sp.normalize_with_offsets(" ab c ");
    assert_eq!(n, "▁ab▁c");
    // ▁ (dummy, from 1) a b ▁ (the space at 3) c, then the end before the
    // trailing space.
    assert_eq!(offs, vec![1, 1, 1, 1, 2, 3, 3, 3, 4, 5]);
}

#[test]
fn unigram_picks_the_best_path() {
    let sp = model(&base(), ModelType::Unigram, false);
    // "▁ab" (-1.5) beats "▁"+"ab" (-3.0).
    assert_eq!(sp.encode("ab"), vec![7]);
    assert_eq!(sp.encode("ab c"), vec![7, 3, 8]);
    assert_eq!(sp.decode(&[7, 3, 8]).unwrap(), "ab c");
}

#[test]
fn spans_point_into_the_input() {
    let sp = model(&base(), ModelType::Unigram, true);
    let e = sp.encode_full(" ab é");
    let spans: Vec<_> = e
        .pieces
        .iter()
        .map(|p| (p.piece.as_str(), p.surface.as_str(), p.begin, p.end))
        .collect();
    assert_eq!(
        spans,
        vec![
            ("▁ab", "ab", 1, 3),
            ("▁", " ", 3, 4),
            ("<0xC3>", "", 4, 4),
            ("<0xA9>", "é", 4, 6),
        ]
    );
}

#[test]
fn unknowns_merge_or_fall_back_to_bytes() {
    let sp = model(&base(), ModelType::Unigram, false);
    let e = sp.encode_full("aé€");
    assert_eq!(e.ids(), vec![9, 0]);
    assert_eq!(e.pieces[1].piece, "é€");
    assert_eq!((e.pieces[1].begin, e.pieces[1].end), (1, 6));
    assert_eq!(sp.decode(&[9, 0]).unwrap(), "a \u{2047} ");

    let sp = model(&base(), ModelType::Unigram, true);
    let ids = sp.encode("aé");
    let bytes: Vec<u32> = "é"
        .bytes()
        .map(|b| sp.piece_to_id(&byte_piece(b)))
        .collect();
    assert_eq!(ids, [vec![9], bytes].concat());
    assert_eq!(sp.decode(&ids).unwrap(), "aé");
    // A lone continuation byte decodes to U+FFFD.
    assert_eq!(sp.decode(&[sp.piece_to_id("<0xA9>")]).unwrap(), "\u{FFFD}");
}

#[test]
fn decode_drops_control_pieces_and_keeps_unknown_strings() {
    let sp = model(&base(), ModelType::Unigram, false);
    assert_eq!(sp.decode(&[1, 7, 2]).unwrap(), "ab");
    assert_eq!(sp.decode_pieces(&["▁ab", "▁", "zz"]), "ab zz");
    let d = sp.decode_full(&[1, 7, 3, 8]).unwrap();
    let spans: Vec<_> = d
        .pieces
        .iter()
        .map(|p| (p.surface.as_str(), p.begin, p.end))
        .collect();
    assert_eq!(
        spans,
        vec![("", 0, 0), ("ab", 0, 2), (" ", 2, 3), ("c", 3, 4)]
    );
}

#[test]
fn extra_options_apply_in_order() {
    let sp = model(&base(), ModelType::Unigram, false);
    let mut e = sp.encode_full("ab €");
    sp.apply_extra_options(&mut e, &parse_extra_options("bos:eos:reverse:unk").unwrap())
        .unwrap();
    assert_eq!(e.piece_strs(), vec!["</s>", "<unk>", "▁", "▁ab", "<s>"]);
    let e = sp
        .encode_with(
            "ab",
            &EncodeOptions {
                add_bos: true,
                add_eos: true,
                reverse: true,
                ..EncodeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(e.ids(), vec![1, 7, 2]);
    assert!(parse_extra_options("bos:nope").is_err());
}

/// Every segmentation of `s` and its score, by brute force.
fn all_segmentations(sp: &SentencePiece, s: &str) -> Vec<(Vec<u32>, f32)> {
    if s.is_empty() {
        return vec![(Vec::new(), 0.0)];
    }
    let mut out = Vec::new();
    for (len, _) in s.char_indices().skip(1).chain([(s.len(), ' ')]) {
        let head = &s[..len];
        let Some(&id) = sp.vocab.get(head) else {
            continue;
        };
        let score = sp.pieces[id as usize].score;
        for (mut rest, rs) in all_segmentations(sp, &s[len..]) {
            rest.insert(0, id);
            out.push((rest, score + rs));
        }
    }
    out
}

#[test]
fn nbest_enumerates_segmentations_best_first() {
    let sp = model(&base(), ModelType::Unigram, false);
    let mut want = all_segmentations(&sp, "▁ab▁ab");
    want.sort_by(|a, b| b.1.total_cmp(&a.1));
    let got = sp.nbest_encode("ab ab", 100).unwrap();
    assert_eq!(got.len(), want.len());
    for (g, (_, score)) in got.iter().zip(&want) {
        // Scores come out in the brute-force order; ties may swap paths, so
        // check each path's own score separately.
        assert!((g.score - score).abs() < 1e-5, "{} vs {score}", g.score);
        let own = want
            .iter()
            .find(|(i, _)| *i == g.ids())
            .expect("a real segmentation");
        assert!((own.1 - g.score).abs() < 1e-5);
    }
    assert_eq!(got[0].ids(), sp.encode("ab ab"));
    assert_eq!(sp.nbest_encode("ab", 2).unwrap().len(), 2);
    assert!(
        model(&base(), ModelType::Bpe, false)
            .nbest_encode("ab", 2)
            .is_err()
    );
}

#[test]
fn sampling_follows_the_segmentation_distribution() {
    let sp = model(&base(), ModelType::Unigram, false);
    let alpha = 0.5f32;
    let segs = all_segmentations(&sp, "▁ab");
    let z: f64 = segs.iter().map(|(_, s)| f64::from(alpha * s).exp()).sum();
    let mut rng = SplitMix64(7);
    let n = 40_000;
    let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
    for _ in 0..n {
        let e = sp
            .sample_encode("ab", -1, alpha, &mut || rng.uniform())
            .unwrap();
        *counts.entry(e.ids()).or_default() += 1;
    }
    for (ids, s) in &segs {
        let p = f64::from(alpha * s).exp() / z;
        let f = counts.get(ids).copied().unwrap_or(0) as f64 / n as f64;
        assert!((p - f).abs() < 0.01, "{ids:?}: p {p:.4} vs sampled {f:.4}");
    }
    // nbest_size 1 is plain encoding; nbest_size 2 only ever picks the two best.
    assert_eq!(
        sp.sample_encode("ab", 1, alpha, &mut || rng.uniform())
            .unwrap()
            .ids(),
        vec![7]
    );
    let top2: Vec<Vec<u32>> = sp
        .nbest_encode("ab", 2)
        .unwrap()
        .iter()
        .map(Encoding::ids)
        .collect();
    for _ in 0..200 {
        let e = sp
            .sample_encode("ab", 2, alpha, &mut || rng.uniform())
            .unwrap();
        assert!(top2.contains(&e.ids()));
    }
}

#[test]
fn bpe_merges_by_score_and_drops_out() {
    let sp = model(&base(), ModelType::Bpe, false);
    // ▁ a b → (ab: -1.0) → ▁ ab → (▁ab: -1.5) → ▁ab.
    assert_eq!(sp.encode("ab"), vec![7]);
    assert_eq!(sp.encode("abc"), vec![7, 8]);
    let mut rng = SplitMix64(1);
    assert_eq!(
        sp.sample_encode("ab", 0, 0.0, &mut || rng.uniform())
            .unwrap()
            .ids(),
        vec![7]
    );
    // Every merge dropped: single characters.
    assert_eq!(
        sp.sample_encode("ab", 0, 1.0, &mut || rng.uniform())
            .unwrap()
            .ids(),
        vec![3, 4, 5]
    );
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        seen.insert(
            sp.sample_encode("ab", 0, 0.5, &mut || rng.uniform())
                .unwrap()
                .ids(),
        );
    }
    assert!(seen.len() > 1);
}

#[test]
fn user_defined_symbols_stay_whole() {
    let mut p = base();
    p.push(("<sep>", 0.0, PieceType::UserDefined));
    let sp = model(&p, ModelType::Unigram, false);
    let sep = sp.piece_to_id("<sep>");
    assert_eq!(sp.encode("ab<sep>c"), vec![7, sep, 8]);
    assert!(sp.is_user_defined(sep));
}

#[test]
fn special_ids() {
    let sp = model(&base(), ModelType::Unigram, false);
    assert_eq!(
        (sp.bos_id(), sp.eos_id(), sp.pad_id(), sp.unk_id()),
        (Some(1), Some(2), None, 0)
    );
    assert!(sp.is_control(1) && sp.is_unknown(0) && !sp.is_byte(4));
}

#[test]
fn proto_round_trip() {
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    fn bytes_field(f: u64, b: &[u8], out: &mut Vec<u8>) {
        varint(f << 3 | 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }
    let mut m = Vec::new();
    for (p, s, k) in [("<unk>", 0.0f32, 2u64), ("▁", -1.0, 1), ("x", -2.0, 1)] {
        let mut e = Vec::new();
        bytes_field(1, p.as_bytes(), &mut e);
        varint(2 << 3 | 5, &mut e);
        e.extend_from_slice(&s.to_le_bytes());
        varint(3 << 3, &mut e);
        varint(k, &mut e);
        bytes_field(1, &e, &mut m);
    }
    let mut norm = Vec::new();
    bytes_field(1, b"identity", &mut norm);
    varint(4 << 3, &mut norm);
    varint(0, &mut norm);
    bytes_field(3, &norm, &mut m);
    let sp = SentencePiece::from_bytes(&m).unwrap();
    assert_eq!(sp.len(), 3);
    assert!(!sp.normalizer_spec().remove_extra_whitespaces);
    assert_eq!(sp.encode("x  x"), vec![1, 2, 1, 1, 2]);
}
