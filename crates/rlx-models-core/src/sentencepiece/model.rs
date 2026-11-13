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

//! The segmentation models: `unigram_model.cc` (the optimized Viterbi, and
//! the lattice behind n-best and sampling), `bpe_model.cc`, `word_model.cc`,
//! `char_model.cc`. Each returns `(piece, id)` over the normalized text, in
//! order, covering it.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::{PieceType, SPACE_SYMBOL, SentencePiece, char_len};

/// Score penalty of an unknown character below the lowest piece score.
const UNK_PENALTY: f32 = 10.0;
/// `Lattice::NBest` agenda bounds.
const MAX_AGENDA: usize = 100_000;
const MIN_AGENDA: usize = 512;

pub(super) type Segments<'a> = Vec<(&'a str, u32)>;

impl SentencePiece {
    /// Every piece of `vocab` starting at byte `start` of `s`, shortest first,
    /// with its lattice score (unused pieces skipped).
    fn matches_at<'s>(
        &'s self,
        s: &'s str,
        start: usize,
    ) -> impl Iterator<Item = (usize, u32, f32)> + 's {
        let limit = self.max_piece_bytes.min(s.len() - start);
        (1..=limit).filter_map(move |len| {
            let end = start + len;
            if !s.is_char_boundary(end) {
                return None;
            }
            let &id = self.vocab.get(&s[start..end])?;
            let p = &self.pieces[id as usize];
            let score = match p.kind {
                PieceType::Unused => return None,
                // 0.1 per character beyond the first: measured on the C++,
                // which the lattice and the fast path share.
                PieceType::UserDefined => s[start..end].chars().count() as f32 * 0.1 - 0.1,
                _ => p.score,
            };
            Some((len, id, score))
        })
    }

    fn unk_score(&self) -> f32 {
        self.min_score - UNK_PENALTY
    }

    /// `unigram::Model::EncodeOptimized`.
    pub(super) fn encode_unigram<'a>(&self, s: &'a str) -> Segments<'a> {
        #[derive(Clone, Copy)]
        struct Best {
            id: u32,
            score: f32,
            start: Option<usize>,
        }
        let n = s.len();
        let unk_score = self.unk_score();
        let mut best = vec![
            Best {
                id: 0,
                score: 0.0,
                start: None,
            };
            n + 1
        ];
        let mut start = 0;
        while start < n {
            let here = best[start].score;
            let mblen = char_len(&s[start..]);
            let mut has_single = false;
            let mut relax = |end: usize, id: u32, score: f32| {
                let cand = score + here;
                let t = &mut best[end];
                if t.start.is_none() || cand > t.score {
                    *t = Best {
                        id,
                        score: cand,
                        start: Some(start),
                    };
                }
            };
            for (len, id, score) in self.matches_at(s, start) {
                relax(start + len, id, score);
                has_single |= len == mblen;
            }
            if !has_single {
                relax(start + mblen, self.unk_id, unk_score);
            }
            start += mblen;
        }
        let mut out = Vec::new();
        let mut end = n;
        while end > 0 {
            let b = best[end];
            let start = b.start.expect("every character boundary is reachable");
            out.push((&s[start..end], b.id));
            end = start;
        }
        out.reverse();
        out
    }

    /// `Lattice` with `PopulateNodes`: every piece and unknown character.
    fn lattice(&self, s: &str) -> Lattice {
        let n = s.len();
        let mut l = Lattice {
            nodes: Vec::new(),
            begin: vec![Vec::new(); n + 1],
            end: vec![Vec::new(); n + 1],
        };
        l.insert(0, 0, 0, 0.0); // BOS: only in end[0]
        l.begin[0].clear();
        l.end[0] = vec![BOS];
        l.nodes.push(Node {
            pos: n,
            len: 0,
            id: 0,
            score: 0.0,
            backtrace: 0.0,
        }); // EOS: only in begin[n]
        l.begin[n].push(EOS);
        let unk_score = self.unk_score();
        let mut start = 0;
        while start < n {
            let mblen = char_len(&s[start..]);
            let mut has_single = false;
            for (len, id, score) in self.matches_at(s, start) {
                l.insert(start, len, id, score);
                has_single |= len == mblen;
            }
            if !has_single {
                l.insert(start, mblen, self.unk_id, unk_score);
            }
            start += mblen;
        }
        l
    }

    /// `Lattice::NBest` (exact A* over the Viterbi scores): up to `n`
    /// segmentations, best first, each with its total score.
    pub(super) fn nbest_unigram<'a>(&self, s: &'a str, n: usize) -> Vec<(Segments<'a>, f32)> {
        if s.is_empty() {
            return vec![(Vec::new(), 0.0)];
        }
        let mut l = self.lattice(s);
        l.viterbi();
        struct Hyp {
            node: usize,
            next: Option<usize>,
            gx: f32,
        }
        #[derive(PartialEq)]
        struct Entry(f32, usize);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, o: &Self) -> Ordering {
                self.0.total_cmp(&o.0).then_with(|| o.1.cmp(&self.1))
            }
        }
        let mut hyps = vec![Hyp {
            node: EOS,
            next: None,
            gx: 0.0,
        }];
        let mut agenda = BinaryHeap::from([Entry(0.0, 0)]);
        let mut out = Vec::new();
        while let Some(Entry(fx, top)) = agenda.pop() {
            let node = hyps[top].node;
            if node == BOS {
                let mut seg = Vec::new();
                let mut h = hyps[top].next;
                while let Some(i) = h {
                    let Some(next) = hyps[i].next else { break };
                    let nd = &l.nodes[hyps[i].node];
                    seg.push((&s[nd.pos..nd.pos + nd.len], nd.id));
                    h = Some(next);
                }
                out.push((seg, fx));
                if out.len() == n {
                    break;
                }
                continue;
            }
            let gx = hyps[top].gx;
            for &ln in &l.end[l.nodes[node].pos] {
                let lnode = &l.nodes[ln];
                hyps.push(Hyp {
                    node: ln,
                    next: Some(top),
                    gx: lnode.score + gx,
                });
                agenda.push(Entry(lnode.backtrace + gx, hyps.len() - 1));
            }
            if agenda.len() >= MAX_AGENDA {
                let mut keep = Vec::with_capacity(MIN_AGENDA);
                while keep.len() < MIN_AGENDA
                    && let Some(e) = agenda.pop()
                {
                    keep.push(e);
                }
                agenda = keep.into_iter().collect();
            }
        }
        out
    }

    /// `Lattice::Sample`: one segmentation drawn with probability
    /// `∝ exp(theta · score)` (forward filtering, backward sampling).
    pub(super) fn sample_unigram<'a>(
        &self,
        s: &'a str,
        theta: f32,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Segments<'a> {
        if s.is_empty() {
            return Vec::new();
        }
        let l = self.lattice(s);
        let alpha = l.forward(theta);
        let mut out = Vec::new();
        let mut node = EOS;
        let mut probs = Vec::new();
        loop {
            let a = alpha[node];
            let ends = &l.end[l.nodes[node].pos];
            probs.clear();
            probs.extend(
                ends.iter()
                    .map(|&ln| f64::from(alpha[ln] + theta * l.nodes[ln].score - a).exp()),
            );
            node = ends[pick(&probs, uniform)];
            if node == BOS {
                break;
            }
            let nd = &l.nodes[node];
            out.push((&s[nd.pos..nd.pos + nd.len], nd.id));
        }
        out.reverse();
        out
    }

    /// `bpe::Model::SampleEncode` / `Encode`: merges by score, ties to the
    /// leftmost pair; `skip_merge` drops a merge (BPE-dropout).
    pub(super) fn encode_bpe<'a>(
        &self,
        s: &'a str,
        skip_merge: &mut dyn FnMut() -> bool,
    ) -> Segments<'a> {
        struct Symbol {
            start: usize,
            end: usize,
            prev: Option<usize>,
            next: Option<usize>,
            freeze: bool,
        }
        struct Pair {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl PartialEq for Pair {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == Ordering::Equal
            }
        }
        impl Eq for Pair {}
        impl PartialOrd for Pair {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Pair {
            // Highest score first, then the leftmost pair.
            fn cmp(&self, o: &Self) -> Ordering {
                self.score
                    .total_cmp(&o.score)
                    .then_with(|| o.left.cmp(&self.left))
            }
        }

        let mut symbols: Vec<Symbol> = Vec::new();
        let mut at = 0;
        while at < s.len() {
            let (len, freeze) = match self.user_defined.longest(&s[at..]) {
                Some(n) => (n, true),
                None => (char_len(&s[at..]), false),
            };
            let i = symbols.len();
            symbols.push(Symbol {
                start: at,
                end: at + len,
                prev: i.checked_sub(1),
                next: (at + len < s.len()).then_some(i + 1),
                freeze,
            });
            at += len;
        }

        let mut agenda = BinaryHeap::new();
        let mut rev_merge: HashMap<&'a str, (&'a str, &'a str)> = HashMap::new();
        let mut add_pair = |symbols: &[Symbol],
                            agenda: &mut BinaryHeap<Pair>,
                            left: Option<usize>,
                            right: Option<usize>| {
            let (Some(l), Some(r)) = (left, right) else {
                return;
            };
            if symbols[l].freeze || symbols[r].freeze {
                return;
            }
            let piece = &s[symbols[l].start..symbols[r].end];
            let Some(&id) = self.vocab.get(piece) else {
                return;
            };
            let p = &self.pieces[id as usize];
            agenda.push(Pair {
                score: p.score,
                left: l,
                right: r,
                size: piece.len(),
            });
            if p.kind == PieceType::Unused {
                rev_merge.insert(
                    piece,
                    (
                        &s[symbols[l].start..symbols[l].end],
                        &s[symbols[r].start..symbols[r].end],
                    ),
                );
            }
        };
        for i in 1..symbols.len() {
            add_pair(&symbols, &mut agenda, Some(i - 1), Some(i));
        }
        while let Some(top) = agenda.pop() {
            let (l, r) = (top.left, top.right);
            let (ls, rs) = (&symbols[l], &symbols[r]);
            if ls.start == ls.end
                || rs.start == rs.end
                || (ls.end - ls.start) + (rs.end - rs.start) != top.size
            {
                continue;
            }
            if skip_merge() {
                continue;
            }
            let (r_end, r_next) = (rs.end, rs.next);
            symbols[l].end = r_end;
            symbols[l].next = r_next;
            if let Some(nn) = r_next {
                symbols[nn].prev = Some(l);
            }
            symbols[r].start = symbols[r].end;
            let prev = symbols[l].prev;
            add_pair(&symbols, &mut agenda, prev, Some(l));
            let next = symbols[l].next;
            add_pair(&symbols, &mut agenda, Some(l), next);
        }

        let mut out = Vec::new();
        let mut cur = (!symbols.is_empty()).then_some(0);
        while let Some(i) = cur {
            self.bpe_resegment(&s[symbols[i].start..symbols[i].end], &rev_merge, &mut out);
            cur = symbols[i].next;
        }
        out
    }

    /// An `UNUSED` piece goes back to the two pieces it was merged from.
    fn bpe_resegment<'a>(
        &self,
        w: &'a str,
        rev_merge: &HashMap<&'a str, (&'a str, &'a str)>,
        out: &mut Segments<'a>,
    ) {
        let id = self.piece_to_id(w);
        if self.kind(id) != Some(PieceType::Unused) {
            out.push((w, id));
            return;
        }
        match rev_merge.get(w) {
            Some(&(a, b)) => {
                self.bpe_resegment(a, rev_merge, out);
                self.bpe_resegment(b, rev_merge, out);
            }
            None => out.push((w, id)),
        }
    }

    /// `word::Model::Encode`: one piece per whitespace-delimited word.
    pub(super) fn encode_word<'a>(&self, s: &'a str) -> Segments<'a> {
        let mut words = Vec::new();
        let mut begin = 0;
        for (i, _) in s.match_indices(SPACE_SYMBOL) {
            if self.normalizer.treat_whitespace_as_suffix {
                let end = i + SPACE_SYMBOL.len();
                words.push(&s[begin..end]);
                begin = end;
            } else if i > begin {
                words.push(&s[begin..i]);
                begin = i;
            }
        }
        if begin < s.len() {
            words.push(&s[begin..]);
        }
        words
            .into_iter()
            .map(|w| (w, self.piece_to_id(w)))
            .collect()
    }

    /// `character::Model::Encode`.
    pub(super) fn encode_char<'a>(&self, s: &'a str) -> Segments<'a> {
        s.char_indices()
            .map(|(i, c)| {
                let w = &s[i..i + c.len_utf8()];
                (w, self.piece_to_id(w))
            })
            .collect()
    }
}

const BOS: usize = 0;
const EOS: usize = 1;

struct Node {
    pos: usize,
    len: usize,
    id: u32,
    score: f32,
    /// Best score from BOS through this node (`backtrace_score`).
    backtrace: f32,
}

/// Nodes by start (`begin_nodes`) and end (`end_nodes`) byte, in insertion
/// order, which the tie-breaking follows.
struct Lattice {
    nodes: Vec<Node>,
    begin: Vec<Vec<usize>>,
    end: Vec<Vec<usize>>,
}

impl Lattice {
    fn insert(&mut self, pos: usize, len: usize, id: u32, score: f32) {
        let i = self.nodes.len();
        self.nodes.push(Node {
            pos,
            len,
            id,
            score,
            backtrace: 0.0,
        });
        self.begin[pos].push(i);
        self.end[pos + len].push(i);
    }

    /// `Lattice::Viterbi`: each node's best prefix score.
    fn viterbi(&mut self) {
        for pos in 0..self.begin.len() {
            for bi in 0..self.begin[pos].len() {
                let r = self.begin[pos][bi];
                let mut best: Option<f32> = None;
                for &ln in &self.end[pos] {
                    let score = self.nodes[ln].backtrace + self.nodes[r].score;
                    if best.is_none_or(|b| score > b) {
                        best = Some(score);
                    }
                }
                self.nodes[r].backtrace = best.unwrap_or(0.0);
            }
        }
    }

    /// `Lattice::ForwardAlgorithm`: `log Σ exp(theta · score)` over every
    /// path from BOS to each node, excluding the node's own score.
    fn forward(&self, theta: f32) -> Vec<f32> {
        let mut alpha = vec![0.0f32; self.nodes.len()];
        for pos in 0..self.begin.len() {
            for &r in &self.begin[pos] {
                for (k, &ln) in self.end[pos].iter().enumerate() {
                    let v = theta * self.nodes[ln].score + alpha[ln];
                    alpha[r] = log_sum_exp(alpha[r], v, k == 0);
                }
            }
        }
        alpha
    }
}

/// `LogSumExp` of `unigram_model.cc`, including its 50-nat cut-off.
fn log_sum_exp(x: f32, y: f32, init: bool) -> f32 {
    if init {
        return y;
    }
    let (lo, hi) = (x.min(y), x.max(y));
    if hi > lo + 50.0 {
        hi
    } else {
        hi + (f64::from(lo - hi).exp() + 1.0).ln() as f32
    }
}

/// An index drawn from unnormalized weights `w`.
pub(super) fn pick(w: &[f64], uniform: &mut dyn FnMut() -> f64) -> usize {
    let total: f64 = w.iter().sum();
    let mut u = uniform() * total;
    for (i, &x) in w.iter().enumerate() {
        if u < x {
            return i;
        }
        u -= x;
    }
    w.iter().rposition(|&x| x > 0.0).unwrap_or(0)
}
