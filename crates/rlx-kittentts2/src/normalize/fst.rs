// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Byte-level tropical weighted transducers in the `KITTEN1` format —
//! a port of `kitten_text_processing._fst`.
//!
//! An arc label packs the input byte in its low 8 bits (0 = ε) and the output
//! byte above them (0 = none). Rewrites take the lowest-cost accepting path;
//! weights are rounded to f32 after every addition as the source grammars
//! were. Ties are where all the care goes: a frontier decoder handles the
//! common unambiguous case, and when two equal-cost paths emit different
//! strings the whole lattice is built and walked in OpenFst's topological
//! order, which is what decides equal-weight readings upstream. Every map
//! whose iteration order matters is insertion-ordered, like Python's dict.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Read;
use std::rc::Rc;

use anyhow::{Context, Result, bail, ensure};

/// No accepting path for this input.
#[derive(Debug, Clone)]
pub struct NoPath;

impl std::fmt::Display for NoPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no accepting path")
    }
}

impl std::error::Error for NoPath {}

/// `unpack(pack('<f', a + b))` — the f64 sum rounded to f32 — or `a`
/// unchanged when `b` is zero.
fn add(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        a
    } else {
        (a as f64 + b as f64) as f32
    }
}

/// One contracted arc: target, weight, and the output bytes it emits.
#[derive(Debug, Clone)]
struct Arc {
    target: u32,
    weight: f32,
    suffix: Rc<[u8]>,
}

/// Arcs of one state grouped by input byte, in first-occurrence order.
type ArcIndex = Vec<(u8, Vec<Arc>)>;

fn lookup(index: &ArcIndex, byte: u8) -> &[Arc] {
    index
        .iter()
        .find(|(b, _)| *b == byte)
        .map_or(&[], |(_, v)| v.as_slice())
}

/// Insertion-ordered map (Python dict semantics for iteration).
#[derive(Debug, Clone)]
struct OrderedMap<K: std::hash::Hash + Eq + Copy, V> {
    keys: Vec<K>,
    map: HashMap<K, V>,
}

impl<K: std::hash::Hash + Eq + Copy, V> OrderedMap<K, V> {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            map: HashMap::new(),
        }
    }
    fn get(&self, k: &K) -> Option<&V> {
        self.map.get(k)
    }
    fn insert(&mut self, k: K, v: V) {
        if self.map.insert(k, v).is_none() {
            self.keys.push(k);
        }
    }
    fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.keys.iter().map(|k| (k, &self.map[k]))
    }
    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// A frontier entry: cost, output so far, and whether an equal-cost path
/// with different output reached the same state.
#[derive(Debug, Clone)]
struct Entry {
    cost: f32,
    out: Vec<u8>,
    ambiguous: bool,
}

/// A loaded grammar.
pub struct Graph {
    start: u32,
    offsets: Vec<u32>,
    finals: Vec<f32>,
    labels: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<f32>,
    project_output: bool,
    indegree: Vec<u8>,
    cache: RefCell<HashMap<u32, Rc<ArcIndex>>>,
}

type Node = (usize, u32);

impl Graph {
    /// Parse a gzipped `KITTEN1` grammar.
    pub fn from_gz(bytes: &[u8], project_output: bool) -> Result<Self> {
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut raw)
            .context("gunzip grammar")?;
        ensure!(
            raw.len() >= 20 && &raw[..8] == b"KITTEN1\0",
            "not a KITTEN1 grammar"
        );
        let u = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        let (start, states, arcs) = (u(8), u(12) as usize, u(16) as usize);
        let mut off = 20;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = raw.get(off..off + 4 * n).context("truncated grammar")?;
            off += 4 * n;
            Ok(s)
        };
        let words = |b: &[u8]| -> Vec<u32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect()
        };
        let floats = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let offsets = words(take(states + 1)?);
        let finals = floats(take(states)?);
        let labels = words(take(arcs)?);
        let targets = words(take(arcs)?);
        let weights = floats(take(arcs)?);
        let mut indegree = vec![0u8; states];
        for &t in &targets {
            let d = &mut indegree[t as usize];
            if *d < 2 {
                *d += 1;
            }
        }
        Ok(Self {
            start,
            offsets,
            finals,
            labels,
            targets,
            weights,
            project_output,
            indegree,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Arcs of `state`, with deterministic zero-cost output-only chains
    /// contracted into their predecessor (never across a join, whose arc
    /// order resolves ties).
    fn arcs(&self, state: u32) -> Rc<ArcIndex> {
        if let Some(i) = self.cache.borrow().get(&state) {
            return i.clone();
        }
        let mut index: ArcIndex = Vec::new();
        let (lo, hi) = (
            self.offsets[state as usize] as usize,
            self.offsets[state as usize + 1] as usize,
        );
        for i in lo..hi {
            let label = self.labels[i];
            let mut target = self.targets[i];
            let mut suffix = Vec::new();
            if label >> 8 != 0 {
                suffix.push((label >> 8) as u8);
            }
            let mut visited = HashSet::new();
            while !self.project_output
                && self.indegree[target as usize] == 1
                && !self.finals[target as usize].is_finite()
                && !visited.contains(&target)
            {
                let (s, e) = (
                    self.offsets[target as usize] as usize,
                    self.offsets[target as usize + 1] as usize,
                );
                if e != s + 1 || self.labels[s] & 255 != 0 || self.weights[s] != 0.0 {
                    break;
                }
                if self.indegree[self.targets[s] as usize] != 1 {
                    break;
                }
                visited.insert(target);
                let o = self.labels[s] >> 8;
                if o != 0 {
                    suffix.push(o as u8);
                }
                target = self.targets[s];
            }
            let arc = Arc {
                target,
                weight: self.weights[i],
                suffix: suffix.into(),
            };
            let byte = (label & 255) as u8;
            match index.iter_mut().find(|(b, _)| *b == byte) {
                Some((_, v)) => v.push(arc),
                None => index.push((byte, vec![arc])),
            }
        }
        let rc = Rc::new(index);
        let mut cache = self.cache.borrow_mut();
        if cache.len() >= 50_000 {
            cache.clear();
        }
        cache.insert(state, rc.clone());
        rc
    }

    fn relax(
        map: &mut OrderedMap<u32, Entry>,
        target: u32,
        cand: f32,
        value: Vec<u8>,
        ambiguous: bool,
    ) -> bool {
        match map.get(&target) {
            None => {
                map.insert(
                    target,
                    Entry {
                        cost: cand,
                        out: value,
                        ambiguous,
                    },
                );
                true
            }
            Some(old) if cand < old.cost => {
                map.insert(
                    target,
                    Entry {
                        cost: cand,
                        out: value,
                        ambiguous,
                    },
                );
                true
            }
            Some(old) if cand == old.cost && !old.ambiguous && (ambiguous || value != old.out) => {
                let (c, o) = (old.cost, old.out.clone());
                map.insert(
                    target,
                    Entry {
                        cost: c,
                        out: o,
                        ambiguous: true,
                    },
                );
                true
            }
            _ => false,
        }
    }

    /// Best output for `text`.
    pub fn rewrite(&self, text: &str) -> std::result::Result<String, NoPath> {
        let text: String = text.chars().filter(|&c| c != '\0').collect();
        if self.project_output {
            return self.rewrite_ordered(&text);
        }
        let mut active: OrderedMap<u32, Entry> = OrderedMap::new();
        active.insert(
            self.start,
            Entry {
                cost: 0.0,
                out: Vec::new(),
                ambiguous: false,
            },
        );
        let bytes = text.as_bytes();
        for pos in 0..=bytes.len() {
            // ε-closure.
            let mut queue: VecDeque<u32> = active.keys.iter().copied().collect();
            let mut queued: HashSet<u32> = queue.iter().copied().collect();
            while let Some(state) = queue.pop_front() {
                queued.remove(&state);
                let e = active.get(&state).unwrap().clone();
                for a in lookup(&self.arcs(state), 0) {
                    let cand = add(e.cost, a.weight);
                    let mut value = e.out.clone();
                    value.extend_from_slice(&a.suffix);
                    if Self::relax(&mut active, a.target, cand, value, e.ambiguous)
                        && queued.insert(a.target)
                    {
                        queue.push_back(a.target);
                    }
                }
            }
            if pos == bytes.len() {
                let mut best: Option<Entry> = None;
                for (state, e) in active.iter() {
                    let f = self.finals[*state as usize];
                    if !f.is_finite() {
                        continue;
                    }
                    let cand = add(e.cost, f);
                    match &mut best {
                        None => {
                            best = Some(Entry {
                                cost: cand,
                                out: e.out.clone(),
                                ambiguous: e.ambiguous,
                            })
                        }
                        Some(b) if cand < b.cost => {
                            *b = Entry {
                                cost: cand,
                                out: e.out.clone(),
                                ambiguous: e.ambiguous,
                            }
                        }
                        Some(b) if cand == b.cost && (e.ambiguous || e.out != b.out) => {
                            b.ambiguous = true
                        }
                        _ => {}
                    }
                }
                let best = best.ok_or(NoPath)?;
                if best.ambiguous {
                    return self.rewrite_ordered(&text);
                }
                return Ok(String::from_utf8_lossy(&best.out).into_owned());
            }
            let ch = bytes[pos];
            let mut following: OrderedMap<u32, Entry> = OrderedMap::new();
            for (state, e) in active.iter() {
                for a in lookup(&self.arcs(*state), ch) {
                    let cand = add(e.cost, a.weight);
                    let mut value = e.out.clone();
                    value.extend_from_slice(&a.suffix);
                    Self::relax(&mut following, a.target, cand, value, e.ambiguous);
                }
            }
            if following.is_empty() {
                return Err(NoPath);
            }
            active = following;
        }
        unreachable!()
    }

    fn rewrite_ordered(&self, text: &str) -> std::result::Result<String, NoPath> {
        let data = text.as_bytes();
        let len = data.len();
        let start: Node = (0, self.start);
        let mut adjacency: HashMap<Node, Vec<(Node, f32, Rc<[u8]>)>> = HashMap::new();
        let edges = |node: Node, adjacency: &mut HashMap<Node, Vec<(Node, f32, Rc<[u8]>)>>| {
            let (pos, state) = node;
            let idx = self.arcs(state);
            let mut out: Vec<(Node, f32, Rc<[u8]>)> = lookup(&idx, 0)
                .iter()
                .map(|a| ((pos, a.target), a.weight, a.suffix.clone()))
                .collect();
            if pos < len {
                out.extend(
                    lookup(&idx, data[pos])
                        .iter()
                        .map(|a| ((pos + 1, a.target), a.weight, a.suffix.clone())),
                );
            }
            adjacency.insert(node, out);
        };
        // Reverse-DFS finishing order = OpenFst's topological queue.
        let mut seen: HashSet<Node> = HashSet::from([start]);
        let mut visiting: HashSet<Node> = HashSet::from([start]);
        edges(start, &mut adjacency);
        let mut stack: Vec<(Node, usize)> = vec![(start, 0)];
        let mut ordered: Vec<Node> = Vec::new();
        let mut live: HashSet<Node> = HashSet::new();
        let mut cyclic = false;
        while let Some(&mut (node, ref mut i)) = stack.last_mut() {
            let edge = adjacency[&node].get(*i).map(|e| e.0);
            *i += 1;
            match edge {
                None => {
                    ordered.push(node);
                    let is_final = node.0 == len && self.finals[node.1 as usize].is_finite();
                    if is_final || adjacency[&node].iter().any(|e| live.contains(&e.0)) {
                        live.insert(node);
                    }
                    visiting.remove(&node);
                    stack.pop();
                }
                Some(target) => {
                    if visiting.contains(&target) {
                        cyclic = true;
                    }
                    if seen.insert(target) {
                        visiting.insert(target);
                        edges(target, &mut adjacency);
                        stack.push((target, 0));
                    }
                }
            }
        }
        if cyclic {
            let mut reverse: HashMap<Node, Vec<Node>> = HashMap::new();
            for (node, out) in &adjacency {
                for e in out {
                    reverse.entry(e.0).or_default().push(*node);
                }
            }
            let mut pending: Vec<Node> = live.iter().copied().collect();
            while let Some(n) = pending.pop() {
                for p in reverse.get(&n).map_or(&[][..], |v| v.as_slice()) {
                    if live.insert(*p) {
                        pending.push(*p);
                    }
                }
            }
        }
        if !live.contains(&start) {
            return Err(NoPath);
        }
        let mut finals: HashMap<Node, f32> = live
            .iter()
            .filter(|n| n.0 == len)
            .map(|n| (*n, self.finals[n.1 as usize]))
            .collect();
        if self.project_output {
            ordered = Self::project_lattice(&mut adjacency, &mut finals, &ordered, &live, start);
            live = ordered.iter().copied().collect();
        }
        // Shortest path in topological order (requeue on ε-cycle improvement).
        let mut costs: HashMap<Node, f32> = HashMap::from([(start, 0.0)]);
        let mut parents: HashMap<Node, (Node, Rc<[u8]>)> = HashMap::new();
        let (mut best, mut fin) = (f32::INFINITY, None);
        let mut pending: VecDeque<Node> = ordered
            .iter()
            .rev()
            .filter(|n| live.contains(n))
            .copied()
            .collect();
        let mut queued: HashSet<Node> = pending.iter().copied().collect();
        while let Some(node) = pending.pop_front() {
            queued.remove(&node);
            let cost = costs.get(&node).copied().unwrap_or(f32::INFINITY);
            if !cost.is_finite() {
                continue;
            }
            let fw = finals.get(&node).copied().unwrap_or(f32::INFINITY);
            if fw.is_finite() {
                let cand = add(cost, fw);
                if cand < best {
                    best = cand;
                    fin = Some(node);
                }
            }
            for (target, w, suffix) in adjacency[&node].clone() {
                if !live.contains(&target) {
                    continue;
                }
                let cand = add(cost, w);
                if cand < costs.get(&target).copied().unwrap_or(f32::INFINITY) {
                    costs.insert(target, cand);
                    parents.insert(target, (node, suffix));
                    if queued.insert(target) {
                        pending.push_back(target);
                    }
                }
            }
        }
        let mut node = fin.ok_or(NoPath)?;
        let mut parts: Vec<Rc<[u8]>> = Vec::new();
        while node != start {
            let (p, s) = parents[&node].clone();
            parts.push(s);
            node = p;
        }
        let out: Vec<u8> = parts.iter().rev().flat_map(|s| s.iter().copied()).collect();
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Output projection + ε-removal (`pynini` `top_rewrite` on a projected
    /// lattice), in OpenFst `RmEpsilon`'s arc order. Returns the new
    /// finishing order from `start`.
    fn project_lattice(
        adjacency: &mut HashMap<Node, Vec<(Node, f32, Rc<[u8]>)>>,
        finals: &mut HashMap<Node, f32>,
        ordered: &[Node],
        live: &HashSet<Node>,
        start: Node,
    ) -> Vec<Node> {
        let mut incoming: HashSet<Node> = HashSet::from([start]);
        for node in live {
            for (t, _, o) in &adjacency[node] {
                if !o.is_empty() && live.contains(t) {
                    incoming.insert(*t);
                }
            }
        }
        for &source in ordered {
            if !live.contains(&source) || !incoming.contains(&source) {
                continue;
            }
            let mut dist: HashMap<Node, f32> = HashMap::from([(source, 0.0)]);
            let mut pending: VecDeque<Node> = VecDeque::from([source]);
            let mut queued: HashSet<Node> = HashSet::from([source]);
            while let Some(node) = pending.pop_front() {
                queued.remove(&node);
                for (t, w, o) in adjacency[&node].clone() {
                    if !o.is_empty() || !live.contains(&t) {
                        continue;
                    }
                    let d = add(dist[&node], w);
                    if d < dist.get(&t).copied().unwrap_or(f32::INFINITY) {
                        dist.insert(t, d);
                        if queued.insert(t) {
                            pending.push_back(t);
                        }
                    }
                }
            }
            let mut expanded: OrderedMap<(Node, usize), (f32, Rc<[u8]>)> = OrderedMap::new();
            let mut out_keys: Vec<Rc<[u8]>> = Vec::new();
            let mut final_w = f32::INFINITY;
            let mut stack = vec![source];
            let mut seen: HashSet<Node> = HashSet::new();
            while let Some(node) = stack.pop() {
                if !seen.insert(node) {
                    continue;
                }
                let d = dist[&node];
                final_w = final_w.min(add(d, finals.get(&node).copied().unwrap_or(f32::INFINITY)));
                for (t, w, o) in adjacency[&node].clone() {
                    if !live.contains(&t) {
                        continue;
                    }
                    if o.is_empty() {
                        if !seen.contains(&t) {
                            stack.push(t);
                        }
                    } else {
                        // Key on (target, output); outputs interned by value.
                        let oi = match out_keys.iter().position(|k| **k == *o) {
                            Some(i) => i,
                            None => {
                                out_keys.push(o.clone());
                                out_keys.len() - 1
                            }
                        };
                        let cand = add(d, w);
                        let key = (t, oi);
                        let cur = expanded.get(&key).map_or(f32::INFINITY, |v| v.0);
                        expanded.insert(key, (cur.min(cand), o.clone()));
                    }
                }
            }
            let new: Vec<(Node, f32, Rc<[u8]>)> = expanded
                .keys
                .iter()
                .rev()
                .map(|k| {
                    let (w, o) = &expanded.map[k];
                    (k.0, *w, o.clone())
                })
                .collect();
            adjacency.insert(source, new);
            finals.insert(source, final_w);
        }
        let mut seen: HashSet<Node> = HashSet::from([start]);
        let mut order = Vec::new();
        let mut stack: Vec<(Node, usize)> = vec![(start, 0)];
        while let Some(&mut (node, ref mut i)) = stack.last_mut() {
            let next = adjacency.get(&node).and_then(|v| v.get(*i)).map(|e| e.0);
            *i += 1;
            match next {
                None => {
                    order.push(node);
                    stack.pop();
                }
                Some(t) => {
                    if seen.insert(t) {
                        stack.push((t, 0));
                    }
                }
            }
        }
        order
    }
}

/// Load-time check that a grammar parses (used by tests).
pub fn check(bytes: &[u8]) -> Result<()> {
    let g = Graph::from_gz(bytes, false)?;
    if g.offsets.len() != g.finals.len() + 1 {
        bail!("inconsistent grammar");
    }
    Ok(())
}
