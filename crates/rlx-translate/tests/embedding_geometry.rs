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

//! Validates the *inferred* companded dequantization of the embedding table by
//! measuring the cosine geometry of the resulting space.
//!
//! A wrong dequantization still yields vectors, so shape checks cannot catch it.
//! Semantics can: in a real embedding space random token pairs are close to
//! orthogonal (mean cosine ≈ 0) while related words are markedly closer. If the
//! curve were wrong — say every value landed in one sign — random pairs would
//! crowd towards cosine 1 and the signal would vanish.
//!
//! The companded curve is also compared against the obvious alternative, a
//! plain affine `min + (max−min)·byte/255` using only the outer two control
//! points, so the choice is measured rather than asserted.

use rlx_translate::assets::Assets;
use rlx_translate::exec::dequant_gather_row;
use rlx_translate::net::Graph;
use rlx_translate::spm::Vocab;
use std::path::PathBuf;

fn mt_dirs() -> Vec<PathBuf> {
    Assets::discover()
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.join("spm.model").exists() && p.join("embedding.espresso.net").exists())
        .collect()
}

/// Which interpretation of `Q_meta` to dequantize with.
#[derive(Clone, Copy, PartialEq)]
enum Curve {
    /// Whatever curve the port currently applies, via `dequant_gather_row`.
    Port,
    /// Per column, but only the outer two control points: a plain linear ramp.
    Affine,
    /// **Deliberately wrong control**: read `Q_meta` as four planes of `cols`
    /// rather than `cols` groups of four. Same numbers, wrong grouping.
    WrongLayout,
}

/// Dequantized row `id` of the vocabulary table.
fn row(table: &[u8], meta: &[f32], cols: usize, id: u32, curve: Curve) -> Vec<f32> {
    let base = id as usize * cols;
    (0..cols)
        .map(|c| {
            let byte = table[base + c];
            match curve {
                Curve::Port => dequant_gather_row(byte, &meta[c * 4..c * 4 + 4]),
                Curve::Affine => {
                    let q = &meta[c * 4..c * 4 + 4];
                    q[0] + (q[3] - q[0]) * (byte as f32 / 255.0)
                }
                Curve::WrongLayout => {
                    let q = [
                        meta[c],
                        meta[cols + c],
                        meta[2 * cols + c],
                        meta[3 * cols + c],
                    ];
                    dequant_gather_row(byte, &q)
                }
            }
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Word pairs that should sit closer than chance in any sane embedding.
const RELATED: &[(&str, &str)] = &[
    ("dog", "cat"),
    ("summer", "winter"),
    ("man", "woman"),
    ("water", "milk"),
    ("car", "truck"),
    ("city", "town"),
    ("morning", "evening"),
    ("book", "books"),
    ("big", "small"),
    ("red", "blue"),
];

struct Table {
    data: Vec<u8>,
    meta: Vec<f32>,
    cols: usize,
    vocab: Vocab,
}

fn load() -> Option<Table> {
    let dir = mt_dirs().into_iter().find(|d| {
        // The 168k-piece bundle is the one with the rich English vocabulary.
        Vocab::load(d.join("spm.model"))
            .map(|v| v.len() > 100_000)
            .unwrap_or(false)
    })?;
    let g = Graph::load(&dir, "embedding.espresso.net").expect("embedding graph loads");
    let l = g
        .layers
        .iter()
        .filter(|l| l.kind == "quantized_gather")
        .max_by_key(|l| l.int("nRow").unwrap_or(0))
        .expect("a vocabulary gather");
    let cols = l.int("nCol").expect("nCol") as usize;
    Some(Table {
        data: g
            .weights
            .raw(l.blob("weights_u8").expect("weights"))
            .expect("table")
            .to_vec(),
        meta: g
            .weights
            .f32s(l.blob("Q_meta").expect("meta"))
            .expect("meta"),
        cols,
        vocab: Vocab::load(dir.join("spm.model")).expect("vocab loads"),
    })
}

/// Mean cosine over related pairs and over a deterministic random sample.
fn geometry(t: &Table, curve: Curve) -> (f32, f32, usize) {
    let mut rel = Vec::new();
    for (a, b) in RELATED {
        let (Some(ia), Some(ib)) = (t.vocab.word_id(a), t.vocab.word_id(b)) else {
            continue;
        };
        rel.push(cosine(
            &row(&t.data, &t.meta, t.cols, ia, curve),
            &row(&t.data, &t.meta, t.cols, ib, curve),
        ));
    }
    // Deterministic pseudo-random ids, avoiding the control-token block.
    let mut rnd = Vec::new();
    let n = t.vocab.len() as u64;
    let (mut x, mut y) = (12_345u64, 67_891u64);
    for _ in 0..400 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        y = y.wrapping_mul(6364136223846793005).wrapping_add(1013904223);
        let ia = (1000 + x % (n - 1000)) as u32;
        let ib = (1000 + y % (n - 1000)) as u32;
        if ia == ib {
            continue;
        }
        rnd.push(cosine(
            &row(&t.data, &t.meta, t.cols, ia, curve),
            &row(&t.data, &t.meta, t.cols, ib, curve),
        ));
    }
    (mean(&rel), mean(&rnd), rel.len())
}

#[test]
fn embedding_space_has_the_geometry_of_a_real_embedding() {
    let Some(t) = load() else {
        eprintln!("skipping: no 168k-piece bundle installed");
        return;
    };
    let (rel, rnd, n) = geometry(&t, Curve::Port);
    eprintln!(
        "companded: related cos {rel:.4} (dist {:.4}) over {n} pairs · \
         random cos {rnd:.4} (dist {:.4}) · separation {:.4}",
        1.0 - rel,
        1.0 - rnd,
        rel - rnd
    );
    assert!(n >= 5, "only {n} related pairs resolved in the vocabulary");
    assert!(
        rnd.abs() < 0.5,
        "random pairs average cosine {rnd:.3}; a real embedding is near-orthogonal, \
         so the dequantization is collapsing the space"
    );
    assert!(
        rel > rnd,
        "related pairs ({rel:.3}) are no closer than random ({rnd:.3}) — the \
         dequantization is destroying semantic structure"
    );
}

#[test]
fn a_token_is_identical_to_itself_and_distinct_from_others() {
    let Some(t) = load() else {
        eprintln!("skipping: no 168k-piece bundle installed");
        return;
    };
    let Some(id) = t.vocab.word_id("dog") else {
        eprintln!("skipping: no dog in the vocabulary");
        return;
    };
    let v = row(&t.data, &t.meta, t.cols, id, Curve::Port);
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-4, "self-cosine must be 1");
    let other = row(&t.data, &t.meta, t.cols, id + 1, Curve::Port);
    assert!(
        cosine(&v, &other) < 0.999,
        "adjacent ids must not be the same vector"
    );
    assert!(
        v.iter().any(|x| *x < 0.0),
        "an embedding row should span both signs"
    );
}

#[test]
fn the_ports_curve_beats_a_deliberately_wrong_layout() {
    let Some(t) = load() else {
        eprintln!("skipping: no 168k-piece bundle installed");
        return;
    };
    let (p_rel, p_rnd, _) = geometry(&t, Curve::Port);
    let (a_rel, a_rnd, _) = geometry(&t, Curve::Affine);
    let (w_rel, w_rnd, _) = geometry(&t, Curve::WrongLayout);
    eprintln!(
        "port   separation {:.4} (rel {p_rel:.4} / rnd {p_rnd:.4})\n\
         affine separation {:.4} (rel {a_rel:.4} / rnd {a_rnd:.4})\n\
         wrong  separation {:.4} (rel {w_rel:.4} / rnd {w_rnd:.4})",
        p_rel - p_rnd,
        a_rel - a_rnd,
        w_rel - w_rnd
    );
    // Neighbour separation cannot choose *between monotone curves*: measured on
    // the shipped table, companded/quartile/seg/affine all give the same nearest
    // neighbours, because each is a monotone reparameterisation of the others.
    // So the affine number is printed, not asserted. What separation does catch
    // is the wrong `Q_meta` grouping, which is not monotone in the byte at all.
    assert!(
        (p_rel - p_rnd) > (w_rel - w_rnd) + 0.05,
        "reading Q_meta as four planes separates as well as four-per-column \
         ({:.4} vs {:.4}); the layout inference is suspect",
        w_rel - w_rnd,
        p_rel - p_rnd
    );
}

/// The quartile knot placement is what makes a column Gaussian.
///
/// This is the evidence the curve was actually chosen on: the byte codes are
/// uniform over 0..255, so the knot placement alone sets the value
/// distribution. Reading the four `Q_meta` values as the 0/25/75/100th
/// percentiles puts the knots at bytes 0, 64, 192, 255 and yields kurtosis ~3;
/// the equal-thirds reading yields ~2, far too flat.
#[test]
fn quartile_knots_give_a_gaussian_column() {
    let Some(t) = load() else {
        eprintln!("skipping: no 168k-piece bundle installed");
        return;
    };
    let kurtosis = |knots: [f32; 4]| -> f64 {
        let rows = 4000usize;
        let mut k = 0.0f64;
        for c in (0..t.cols).step_by(8) {
            let q = &t.meta[c * 4..c * 4 + 4];
            let v: Vec<f64> = (0..rows)
                .map(|r| {
                    let x = f32::from(t.data[r * t.cols + c]);
                    let seg = if x >= knots[2] {
                        2
                    } else if x >= knots[1] {
                        1
                    } else {
                        0
                    };
                    let f = ((x - knots[seg]) / (knots[seg + 1] - knots[seg])).clamp(0.0, 1.0);
                    f64::from(q[seg] + (q[seg + 1] - q[seg]) * f)
                })
                .collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
            if var <= 0.0 {
                continue;
            }
            k += v
                .iter()
                .map(|x| ((x - m) / var.sqrt()).powi(4))
                .sum::<f64>()
                / v.len() as f64;
        }
        k / (0..t.cols).step_by(8).count() as f64
    };
    let quartile = kurtosis([0.0, 64.0, 192.0, 255.0]);
    let thirds = kurtosis([0.0, 85.0, 170.0, 255.0]);
    eprintln!("kurtosis: quartile knots {quartile:.3}, equal thirds {thirds:.3} (Gaussian is 3.0)");
    assert!(
        (quartile - 3.0).abs() < (thirds - 3.0).abs(),
        "equal-thirds knots are closer to Gaussian than quartile knots \
         ({thirds:.3} vs {quartile:.3}); the curve inference is suspect"
    );
}

#[test]
fn a_wrong_q_meta_layout_collapses_the_geometry() {
    let Some(t) = load() else {
        eprintln!("skipping: no 168k-piece bundle installed");
        return;
    };
    let (good_rel, good_rnd, _) = geometry(&t, Curve::Port);
    let (bad_rel, bad_rnd, _) = geometry(&t, Curve::WrongLayout);
    eprintln!(
        "per-column layout : related {good_rel:.4} random {good_rnd:.4} separation {:.4}\n\
         plane-major layout: related {bad_rel:.4} random {bad_rnd:.4} separation {:.4}",
        good_rel - good_rnd,
        bad_rel - bad_rnd
    );
    // Same 2048 floats, only regrouped. If the wrong grouping scored as well,
    // the cosine test would prove nothing about the layout.
    assert!(
        (good_rel - good_rnd) > (bad_rel - bad_rnd) + 0.05,
        "the wrong Q_meta grouping separates related from random about as well \
         ({:.4}) as the per-column one ({:.4}); this test is not discriminating",
        bad_rel - bad_rnd,
        good_rel - good_rnd
    );
}
