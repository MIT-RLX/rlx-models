//! spaCy 3.8's transition-based dependency parser (`parser` of
//! `en_core_web_lg`, greedy, `beam_width = 1`): arc-eager with BREAK
//! (`_parser_internals/arc_eager.pyx`, `_state.pxd`), the precomputed
//! lower layer (`_precomputable_affine.py`, `parser_model.pyx`
//! `predict_states`), `set_annotations`, `set_children_from_heads` and
//! `nonproj.deprojectivize` — so heads, labels and sentence starts match.
//!
//! Every product is the BLIS-order [`dot`] of `tok2vec`; feature rows are
//! summed in feature order like `saxpy`, and the maxout pick is thinc's
//! two-class `Vec.arg_max` (`0 if a > b else 1`, ties go to piece 1).

use crate::tok2vec::{Tok2Vec, dot_rows, thinc_params};
use crate::tokenizer::{Token, py_isspace};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Move {
    Shift,
    Reduce,
    Left,
    Right,
    Break,
}

const MOVE_NAMES: [&str; 5] = ["S", "D", "L", "R", "B"];

#[derive(Clone, Copy)]
struct Arc {
    head: i32,
    child: i32,
    label: u32,
}

/// `StateC` (only what the greedy parser touches).
struct State {
    length: i32,
    heads: Vec<i32>,
    stack: Vec<i32>,
    rebuffer: Vec<i32>,
    left_arcs: HashMap<i32, Vec<Arc>>,
    right_arcs: HashMap<i32, Vec<Arc>>,
    unshiftable: Vec<bool>,
    sent_starts: HashSet<i32>,
    b_i: i32,
}

impl State {
    fn new(length: usize) -> Self {
        State {
            length: length as i32,
            heads: vec![-1; length],
            stack: Vec::new(),
            rebuffer: Vec::new(),
            left_arcs: HashMap::new(),
            right_arcs: HashMap::new(),
            unshiftable: vec![false; length],
            sent_starts: HashSet::new(),
            b_i: 0,
        }
    }
    fn s(&self, i: i32) -> i32 {
        if i < 0 || i as usize >= self.stack.len() {
            -1
        } else {
            self.stack[self.stack.len() - (i as usize + 1)]
        }
    }
    fn b(&self, i: i32) -> i32 {
        if i < 0 {
            return -1;
        }
        let r = self.rebuffer.len() as i32;
        if i < r {
            self.rebuffer[(r - (i + 1)) as usize]
        } else {
            let b_i = self.b_i + (i - r);
            if b_i >= self.length { -1 } else { b_i }
        }
    }
    fn nth_child(arcs: &HashMap<i32, Vec<Arc>>, head: i32, idx: usize) -> i32 {
        if idx < 1 {
            return -1;
        }
        let Some(v) = arcs.get(&head) else { return -1 };
        let mut n = 0;
        for a in v.iter().rev() {
            if a.child != -1 {
                n += 1;
                if n == idx {
                    return a.child;
                }
            }
        }
        -1
    }
    fn l(&self, head: i32, idx: usize) -> i32 {
        Self::nth_child(&self.left_arcs, head, idx)
    }
    fn r(&self, head: i32, idx: usize) -> i32 {
        Self::nth_child(&self.right_arcs, head, idx)
    }
    fn buffer_length(&self) -> i32 {
        (self.length - self.b_i) + self.rebuffer.len() as i32
    }
    fn stack_depth(&self) -> i32 {
        self.stack.len() as i32
    }
    fn is_final(&self) -> bool {
        self.stack_depth() <= 0 && self.buffer_length() == 0
    }
    /// The doc's own `sent_start` is 1 for token 0 (`Doc.push_back`) and 0
    /// elsewhere; nothing is preset to -1, so `cannot_sent_start` is 0.
    fn is_sent_start(&self, w: i32) -> bool {
        w >= 0 && w < self.length && (w == 0 || self.sent_starts.contains(&w))
    }
    fn has_head(&self, c: i32) -> bool {
        self.heads[c as usize] >= 0
    }
    fn is_unshiftable(&self, i: i32) -> bool {
        // `B(0)` can be -1 here, which `unshiftable.size()` (unsigned)
        // compares as huge: not unshiftable.
        i >= 0 && (i as usize) < self.unshiftable.len() && self.unshiftable[i as usize]
    }
    fn push(&mut self) {
        let b0 = if let Some(b) = self.rebuffer.pop() {
            b
        } else {
            self.b_i += 1;
            self.b_i - 1
        };
        self.stack.push(b0);
    }
    fn unshift(&mut self) {
        let s0 = self.stack.pop().unwrap();
        self.unshiftable[s0 as usize] = true;
        self.rebuffer.push(s0);
    }
    fn del_arc(&mut self, h: i32, c: i32) {
        let map = if h > c {
            &mut self.left_arcs
        } else {
            &mut self.right_arcs
        };
        let Some(arcs) = map.get_mut(&h) else { return };
        let Some(last) = arcs.last() else { return };
        if last.head == h && last.child == c {
            arcs.pop();
        } else {
            let n = arcs.len() - 1;
            if let Some(a) = arcs[..n].iter_mut().find(|a| a.head == h && a.child == c) {
                *a = Arc {
                    head: -1,
                    child: -1,
                    label: 0,
                };
            }
        }
    }
    fn add_arc(&mut self, head: i32, child: i32, label: u32) {
        if self.has_head(child) {
            self.del_arc(self.heads[child as usize], child);
        }
        let arc = Arc { head, child, label };
        if head > child {
            self.left_arcs.entry(head).or_default().push(arc);
        } else {
            self.right_arcs.entry(head).or_default().push(arc);
        }
        self.heads[child as usize] = head;
    }
    /// `set_context_tokens(ids, 8)`.
    fn context(&self) -> [i32; 8] {
        let (b0, s0) = (self.b(0), self.s(0));
        [
            b0,
            self.b(1),
            s0,
            self.s(1),
            self.s(2),
            self.l(b0, 1),
            self.l(s0, 1),
            self.r(s0, 1),
        ]
    }

    fn valid(&self, m: Move) -> bool {
        match m {
            Move::Shift => {
                if self.stack_depth() == 0 {
                    true
                } else if self.buffer_length() < 2 {
                    false
                } else {
                    !self.is_sent_start(self.b(0)) && !self.is_unshiftable(self.b(0))
                }
            }
            Move::Reduce => self.stack_depth() != 0,
            Move::Left | Move::Right => {
                self.stack_depth() != 0
                    && self.buffer_length() != 0
                    && !self.is_sent_start(self.b(0))
            }
            Move::Break => {
                self.buffer_length() >= 2
                    && self.b(1) == self.b(0) + 1
                    && !self.is_sent_start(self.b(1))
            }
        }
    }

    fn apply(&mut self, m: Move, label: u32) {
        match m {
            Move::Shift => self.push(),
            Move::Reduce => {
                if self.has_head(self.s(0)) || self.stack_depth() == 1 {
                    self.stack.pop();
                } else {
                    self.unshift();
                }
            }
            Move::Left => {
                let (b0, s0) = (self.b(0), self.s(0));
                self.add_arc(b0, s0, label);
                if b0 >= 0 && (b0 as usize) < self.unshiftable.len() {
                    self.unshiftable[b0 as usize] = false;
                }
                self.stack.pop();
            }
            Move::Right => {
                let (s0, b0) = (self.s(0), self.b(0));
                self.add_arc(s0, b0, label);
                self.push();
            }
            Move::Break => {
                let b1 = self.b(1);
                self.sent_starts.insert(b1);
            }
        }
    }
}

/// A parsed doc: absolute `heads`, `deps` labels and `sent_starts`
/// (`bool(token.is_sent_start)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Parse {
    pub heads: Vec<usize>,
    pub deps: Vec<String>,
    pub sent_starts: Vec<bool>,
}

impl Parse {
    /// `doc.sents` as token ranges.
    pub fn sents(&self) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        let mut start = 0;
        for i in 1..self.heads.len() {
            if self.sent_starts[i] {
                out.push(start..i);
                start = i;
            }
        }
        if start != self.heads.len() {
            out.push(start..self.heads.len());
        }
        out
    }
}

pub struct Parser {
    actions: Vec<(Move, u32)>,
    labels: Vec<String>, // label id -> string; id 0 = ""
    lin_w: Vec<f32>,     // [64, 96]
    lin_b: Vec<f32>,
    aff_w: Vec<f32>, // [nF * nO * nP, 64]
    aff_b: Vec<f32>, // [nO * nP]
    pad: Vec<f32>,   // [nF * nO * nP]
    up_w: Vec<f32>,  // [classes, nO]
    up_b: Vec<f32>,
    n_f: usize,
    n_o: usize,
    n_p: usize,
    n_h: usize, // tokvec width into the lower layer
    n_i: usize, // tok2vec width
}

impl Parser {
    /// Load `<pipeline>/parser/{moves,model}`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let (actions, labels) = load_moves(&model_dir.join("parser/moves"))?;
        let params = thinc_params(&model_dir.join("parser/model"))?;
        let mut linears = Vec::new();
        let mut affine = None;
        for (name, p) in &params {
            if name == "precomputable_affine" {
                affine = Some(p);
            } else if name == "linear" {
                linears.push(p);
            }
        }
        let aff = affine.context("parser: no precomputable_affine")?;
        let (ws, aff_w) = aff.get("W").context("affine W")?.clone();
        ensure!(ws.len() == 4, "affine W shape {ws:?}");
        let (n_f, n_o, n_p, n_h) = (ws[0], ws[1], ws[2], ws[3]);
        let mut lower = None;
        let mut upper = None;
        for p in linears {
            let (s, w) = p.get("W").context("linear W")?.clone();
            let b = p.get("b").context("linear b")?.1.clone();
            if s[0] == n_h {
                lower = Some((s, w, b));
            } else if s[0] == actions.len() && s[1] == n_o {
                upper = Some((w, b));
            }
        }
        let (ls, lin_w, lin_b) = lower.context("parser: no tok2vec linear")?;
        let (up_w, up_b) = upper.context("parser: no upper linear")?;
        Ok(Parser {
            labels,
            lin_w,
            lin_b,
            aff_w,
            aff_b: aff.get("b").context("affine b")?.1.clone(),
            pad: aff.get("pad").context("affine pad")?.1.clone(),
            up_w,
            up_b,
            n_f,
            n_o,
            n_p,
            n_h,
            n_i: ls[1],
            actions,
        })
    }

    /// `parser.moves.get_class_name(i)` for every class.
    pub fn class_names(&self) -> Vec<String> {
        self.actions
            .iter()
            .map(|&(m, l)| {
                let name = MOVE_NAMES[m as usize];
                if l == 0 {
                    name.to_string()
                } else {
                    format!("{name}-{}", self.labels[l as usize])
                }
            })
            .collect()
    }

    /// Parse a doc from its `tok2vec` output (`[n, n_i]`).
    pub fn parse_tokvecs(&self, toks: &[Token], t2v: &[f32]) -> Parse {
        self.run(toks, t2v, None)
    }

    /// [`Parser::parse_tokvecs`] that also returns the lower layer's
    /// tokvecs (`[n, 64]`) and every step's class scores, for parity checks.
    pub fn parse_traced(&self, toks: &[Token], t2v: &[f32]) -> (Parse, Vec<f32>, Vec<f32>) {
        let mut tr = (Vec::new(), Vec::new());
        let p = self.run(toks, t2v, Some(&mut tr));
        (p, tr.0, tr.1)
    }

    fn run(
        &self,
        toks: &[Token],
        t2v: &[f32],
        mut trace: Option<&mut (Vec<f32>, Vec<f32>)>,
    ) -> Parse {
        let n = toks.len();
        if n == 0 {
            return Parse {
                heads: vec![],
                deps: vec![],
                sent_starts: vec![],
            };
        }
        let (fo, op) = (self.n_f * self.n_o * self.n_p, self.n_o * self.n_p);
        // list2array >> Linear: tokvecs = t2v @ W.T + b
        let mut tv = vec![0f32; n * self.n_h];
        for t in 0..n {
            let x = &t2v[t * self.n_i..(t + 1) * self.n_i];
            let y = &mut tv[t * self.n_h..(t + 1) * self.n_h];
            dot_rows(x, &self.lin_w, y);
            for (v, b) in y.iter_mut().zip(&self.lin_b) {
                *v += b;
            }
        }
        if let Some(t) = trace.as_deref_mut() {
            t.0 = tv.clone();
        }
        // precomputable_affine: Yf[0] = pad, Yf[1 + t] = tokvecs[t] @ W.T
        let mut yf = vec![0f32; (n + 1) * fo];
        yf[..fo].copy_from_slice(&self.pad);
        for t in 0..n {
            let x = &tv[t * self.n_h..(t + 1) * self.n_h];
            dot_rows(x, &self.aff_w, &mut yf[(t + 1) * fo..(t + 2) * fo]);
        }
        let mut st = State::new(n);
        let mut unmaxed = vec![0f32; op];
        let mut hidden = vec![0f32; self.n_o];
        let mut scores = vec![0f32; self.actions.len()];
        let mut valid_move = [false; 5];
        while !st.is_final() {
            let ids = st.context();
            unmaxed.iter_mut().for_each(|v| *v = 0.0);
            for (f, &id) in ids.iter().enumerate() {
                let feat = if id < 0 {
                    &yf[f * op..(f + 1) * op]
                } else {
                    let at = (id as usize + 1) * fo + f * op;
                    &yf[at..at + op]
                };
                for (u, v) in unmaxed.iter_mut().zip(feat) {
                    *u += v;
                }
            }
            for (u, b) in unmaxed.iter_mut().zip(&self.aff_b) {
                *u += b;
            }
            for j in 0..self.n_o {
                let p = &unmaxed[j * self.n_p..(j + 1) * self.n_p];
                hidden[j] = if self.n_p == 2 {
                    if p[0] > p[1] { p[0] } else { p[1] }
                } else {
                    let mut best = 0;
                    for i in 1..self.n_p {
                        if p[i] > p[best] {
                            best = i;
                        }
                    }
                    p[best]
                };
            }
            dot_rows(&hidden, &self.up_w, &mut scores);
            for (s, b) in scores.iter_mut().zip(&self.up_b) {
                *s += b;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.1.extend_from_slice(&scores);
            }
            // no unseen classes in en_core_web_lg: the class mask is all ones
            for (k, m) in [
                Move::Shift,
                Move::Reduce,
                Move::Left,
                Move::Right,
                Move::Break,
            ]
            .iter()
            .enumerate()
            {
                valid_move[k] = st.valid(*m);
            }
            let mut best: Option<usize> = None;
            for (i, &(m, _)) in self.actions.iter().enumerate() {
                if valid_move[m as usize] && best.is_none_or(|b| scores[i] > scores[b]) {
                    best = Some(i);
                }
            }
            match best {
                Some(i) => {
                    let (m, l) = self.actions[i];
                    st.apply(m, l);
                }
                None => {
                    st.stack.clear();
                    st.b_i = st.length;
                }
            }
        }
        self.annotate(toks, &st)
    }

    /// Tokenized doc -> parse (runs `tok2vec` first).
    pub fn parse(&self, t2v: &Tok2Vec, toks: &[Token]) -> Parse {
        self.parse_tokvecs(toks, &t2v.forward(toks))
    }

    /// `ArcEager.set_annotations` + `DependencyParser` postprocessing.
    fn annotate(&self, toks: &[Token], st: &State) -> Parse {
        let n = toks.len();
        let mut heads: Vec<usize> = (0..n).collect();
        let mut deps: Vec<String> = vec![String::new(); n];
        for arcs in st.left_arcs.values().chain(st.right_arcs.values()) {
            for a in arcs {
                if a.head != -1 && a.child != -1 {
                    heads[a.child as usize] = a.head as usize;
                    deps[a.child as usize] = self.labels[a.label as usize].clone();
                }
            }
        }
        for i in 0..n {
            if heads[i] == i {
                deps[i] = "ROOT".to_string();
            }
        }
        let mut edges = children_from_heads(&heads);
        // nonproj.deprojectivize: children come from the edges of the last
        // `set_children_from_heads` and the CURRENT heads.
        let is_space: Vec<bool> = toks
            .iter()
            .map(|t| !t.text.is_empty() && t.text.chars().all(py_isspace))
            .collect();
        for i in 0..n {
            if let Some((new_label, head_label)) = deps[i].split_once("||") {
                let (new_label, head_label) = (new_label.to_string(), head_label.to_string());
                let h = find_new_head(i, &head_label, &heads, &deps, &edges.0, &is_space);
                heads[i] = h;
                deps[i] = new_label;
            }
        }
        edges = children_from_heads(&heads);
        Parse {
            heads,
            deps,
            sent_starts: edges.1,
        }
    }
}

/// `token.children` (lefts in `[l_edge, i)`, rights in `(i, r_edge]`).
fn children(i: usize, heads: &[usize], edges: &[(usize, usize)]) -> Vec<usize> {
    let (l, r) = edges[i];
    (l..i).chain(i + 1..=r).filter(|&c| heads[c] == i).collect()
}

fn find_new_head(
    tok: usize,
    head_label: &str,
    heads: &[usize],
    deps: &[String],
    edges: &[(usize, usize)],
    is_space: &[bool],
) -> usize {
    let mut queue = vec![heads[tok]];
    while !queue.is_empty() {
        let mut next = Vec::new();
        for &q in &queue {
            for c in children(q, heads, edges) {
                if is_space[c] || c == tok {
                    continue;
                }
                if deps[c] == head_label {
                    return c;
                }
                next.push(c);
            }
        }
        queue = next;
    }
    heads[tok]
}

/// `set_children_from_heads`: (l_edge, r_edge) per token and the sentence
/// starts (`sent_start == 1`).
fn children_from_heads(heads: &[usize]) -> (Vec<(usize, usize)>, Vec<bool>) {
    let n = heads.len();
    let mut e: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    let mut loop_count = 0;
    loop {
        // _set_lr_kids_and_edges
        for i in 0..n {
            let h = heads[i];
            if e[i].0 < e[h].0 {
                e[h].0 = e[i].0;
            }
            if e[i].1 > e[h].1 {
                e[h].1 = e[i].1;
            }
        }
        for i in (0..n).rev() {
            let h = heads[i];
            if e[i].1 > e[h].1 {
                e[h].1 = e[i].1;
            }
            if e[i].0 < e[h].0 {
                e[h].0 = e[i].0;
            }
        }
        let starts: HashSet<usize> = (0..n).filter(|&i| heads[i] == i).map(|i| e[i].0).collect();
        let mut within = true;
        let mut cur = 0;
        'check: for i in 0..n {
            if (i > 0 && starts.contains(&i)) || i == n - 1 {
                for j in cur..i {
                    if heads[j] < cur || heads[j] >= i + 1 {
                        within = false;
                        break 'check;
                    }
                }
                cur = i;
            }
        }
        if within || loop_count > 10 {
            break;
        }
        loop_count += 1;
    }
    let mut ss = vec![false; n];
    for i in 0..n {
        if heads[i] == i {
            ss[e[i].0] = true;
        }
    }
    (e, ss)
}

/// `TransitionSystem.from_bytes` + `initialize_actions`: actions by move id,
/// labels by descending (frequency, label).
fn load_moves(path: &Path) -> Result<(Vec<(Move, u32)>, Vec<String>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let v = rmpv::decode::read_value(&mut bytes.as_slice())?;
    let js = v
        .as_map()
        .and_then(|m| m.iter().find(|(k, _)| k.as_str() == Some("moves")))
        .and_then(|(_, x)| x.as_str())
        .context("moves: no 'moves' entry")?;
    let j: serde_json::Value = serde_json::from_str(js)?;
    let mut by_action: Vec<(String, &serde_json::Map<String, serde_json::Value>)> = j
        .as_object()
        .context("moves object")?
        .iter()
        .map(|(k, v)| (k.clone(), v.as_object().unwrap()))
        .collect();
    by_action.sort_by(|a, b| a.0.cmp(&b.0));
    let mut labels = vec![String::new()];
    let mut label_ids: HashMap<String, u32> = HashMap::from([(String::new(), 0)]);
    let mut actions: Vec<(Move, u32)> = Vec::new();
    for (k, freqs) in by_action {
        let m = match k.parse::<usize>()? {
            0 => Move::Shift,
            1 => Move::Reduce,
            2 => Move::Left,
            3 => Move::Right,
            4 => Move::Break,
            x => bail!("unknown move {x}"),
        };
        let mut sorted: Vec<(i64, &String)> = freqs
            .iter()
            .map(|(l, f)| (f.as_i64().unwrap(), l))
            .collect();
        sorted.sort();
        sorted.reverse();
        for (f, l) in sorted {
            ensure!(
                f >= 0,
                "negative label frequency (added labels) not supported"
            );
            let id = *label_ids.entry(l.clone()).or_insert_with(|| {
                labels.push(l.clone());
                labels.len() as u32 - 1
            });
            if !actions.contains(&(m, id)) {
                actions.push((m, id));
            }
        }
    }
    Ok((actions, labels))
}
