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

//! `spacy.Tok2Vec.v2` = `MultiHashEmbed.v2` (NORM, PREFIX, SUFFIX, SHAPE,
//! SPACY, IS_SPACE hash embeddings + ORTH static vectors, maxout + layer
//! norm) followed by `MaxoutWindowEncoder.v2` (residual window-1 maxout
//! blocks), as thinc 8.3 computes it on the CPU.
//!
//! Lexeme attributes follow spaCy: `NORM` is the special case's norm, else
//! the `lexeme_norm` lookup, else `BASE_NORMS`, else `lower()`; `PREFIX` the
//! first character, `SUFFIX` the last three, `SHAPE` `word_shape`. Strings
//! hash with MurmurHash64A (seed 1, spaCy's `hash_string`); embedding rows
//! with thinc's `MurmurHash3_x86_128_uint64`; rows are summed in order
//! (`gather_add`).

use crate::tokenizer::{Token, py_isspace};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// spaCy's string id (`StringStore.add`): the symbol id for spaCy's fixed
/// symbols (`spacy.symbols.IDS`), else `hash_string` — MurmurHash64A of the
/// UTF-8 bytes, seed 1.
pub fn hash_string(s: &str) -> u64 {
    static SYMBOLS: std::sync::OnceLock<HashMap<String, u64>> = std::sync::OnceLock::new();
    let sym =
        SYMBOLS.get_or_init(|| serde_json::from_str(include_str!("../data/symbols.json")).unwrap());
    match sym.get(s) {
        Some(&id) => id,
        None => murmur64a(s.as_bytes(), 1),
    }
}

fn murmur64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let len = key.len();
    let mut h = seed ^ (len as u64).wrapping_mul(M);
    let mut chunks = key.chunks_exact(8);
    for c in &mut chunks {
        let mut k = u64::from_le_bytes(c.try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let t = chunks.remainder();
    if !t.is_empty() {
        for (i, &b) in t.iter().enumerate() {
            h ^= (b as u64) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^= k >> 33;
    k
}

/// thinc `MurmurHash3_x86_128_uint64(val, seed)`: four 32-bit keys.
pub fn thinc_hash(val: u64, seed: u32) -> [u32; 4] {
    let mut h1 = val.wrapping_mul(0x87c3_7b91_1142_53d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5_ad43_2745_937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2 = seed as u64;
    h2 ^= 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// `spacy.lang.lex_attrs.word_shape`.
pub fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".into();
    }
    let mut shape = String::new();
    let mut last = '\0';
    let mut seq = 0;
    for c in text.chars() {
        let sc = if c.is_alphabetic() {
            if c.is_uppercase() { 'X' } else { 'x' }
        } else if c.to_digit(10).is_some()
            || (c.is_numeric() && !c.is_alphabetic() && unicode_digit(c))
        {
            'd'
        } else {
            c
        };
        if sc == last {
            seq += 1;
        } else {
            seq = 0;
            last = sc;
        }
        if seq < 4 {
            shape.push(sc);
        }
    }
    shape
}

/// Python `str.isdigit` beyond ASCII: Unicode decimal digits and the
/// superscript/subscript digits (`Numeric_Type=Digit`).
fn unicode_digit(c: char) -> bool {
    matches!(c, '\u{b2}' | '\u{b3}' | '\u{b9}' | '\u{2070}'..='\u{2079}' | '\u{2080}'..='\u{2089}' | '\u{2460}'..='\u{2468}')
        || c.to_digit(10).is_some()
}

/// A dense float array from a thinc / msgpack-numpy map.
fn ndarray(v: &rmpv::Value) -> Result<(Vec<usize>, Vec<f32>)> {
    let m = v.as_map().context("ndarray map")?;
    let field = |name: &[u8]| {
        m.iter()
            .find(|(k, _)| {
                k.as_slice() == Some(name) || k.as_str().map(str::as_bytes) == Some(name)
            })
            .map(|(_, v)| v)
    };
    let ty = field(b"type").and_then(|t| t.as_str()).unwrap_or("");
    if ty != "<f4" {
        bail!("ndarray dtype {ty:?}");
    }
    let shape: Vec<usize> = field(b"shape")
        .and_then(|s| s.as_array())
        .context("shape")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let data = field(b"data").and_then(|d| d.as_slice()).context("data")?;
    Ok((
        shape,
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    ))
}

/// The parameters of every node of a serialized thinc model, by node name.
pub(crate) fn thinc_params(
    path: &Path,
) -> Result<Vec<(String, HashMap<String, (Vec<usize>, Vec<f32>)>)>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let v = rmpv::decode::read_value(&mut bytes.as_slice())?;
    let get = |k: &str| {
        v.as_map()
            .and_then(|m| m.iter().find(|(kk, _)| kk.as_str() == Some(k)))
            .map(|(_, x)| x)
    };
    let nodes = get("nodes").and_then(|n| n.as_array()).context("nodes")?;
    let params = get("params").and_then(|n| n.as_array()).context("params")?;
    let mut out = Vec::new();
    for (n, p) in nodes.iter().zip(params) {
        let name = n
            .as_map()
            .and_then(|m| m.iter().find(|(k, _)| k.as_str() == Some("name")))
            .and_then(|(_, x)| x.as_str())
            .unwrap_or("")
            .to_string();
        let mut ps = HashMap::new();
        for (k, a) in p.as_map().context("params entry")? {
            ps.insert(k.as_str().context("param name")?.to_string(), ndarray(a)?);
        }
        if !ps.is_empty() {
            out.push((name, ps));
        }
    }
    Ok(out)
}

/// numpy's pairwise float32 `add.reduce` over a contiguous row.
pub(crate) fn np_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut s = 0f32;
        for &v in a {
            s += v;
        }
        s
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let n2 = (n / 2) - ((n / 2) % 8);
        np_sum_f32(&a[..n2]) + np_sum_f32(&a[n2..])
    }
}

/// One output of thinc's float32 `gemm` (`x @ w.T`) as BLIS 1.3's generic
/// `sgemmsup` reference kernel computes it on aarch64 (read from the
/// shipped machine code): k in blocks of `KC = 256`; inside a block 16
/// fused accumulators (four 4-lane vectors over consecutive k), reduced as
/// `((v1 + v0) + (v2 + v3))` lane-wise then pairwise, the `k % 16` tail
/// fused onto the sum; blocks added into `c` in order.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    const KC: usize = 256;
    let mut c = 0f32;
    let mut at = 0;
    while at < a.len() {
        let e = (at + KC).min(a.len());
        let part = blis_block(&a[at..e], &b[at..e]);
        c = if at == 0 { part } else { c + part };
        at = e;
    }
    c
}

#[inline(always)]
fn blis_block(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let main = n - n % 16;
    let mut v = [[0f32; 4]; 4];
    let mut k = 0;
    while k < main {
        for r in 0..4 {
            for l in 0..4 {
                v[r][l] = a[k + 4 * r + l].mul_add(b[k + 4 * r + l], v[r][l]);
            }
        }
        k += 16;
    }
    let t = [0, 1, 2, 3].map(|l| (v[1][l] + v[0][l]) + (v[2][l] + v[3][l]));
    let mut s = (t[0] + t[1]) + (t[2] + t[3]);
    for i in main..n {
        s = a[i].mul_add(b[i], s);
    }
    s
}

/// `out[r] = dot(x, w[r * k..(r + 1) * k])` for every row of `w`, with
/// the same per-output arithmetic as [`dot`] (bit-identical): on aarch64
/// four rows share each load of `x`, one NEON register per accumulator
/// vector of the BLIS kernel.
pub(crate) fn dot_rows(x: &[f32], w: &[f32], out: &mut [f32]) {
    let k = x.len();
    assert_eq!(w.len(), k * out.len());
    let mut r = 0;
    #[cfg(target_arch = "aarch64")]
    while r + 4 <= out.len() {
        // SAFETY: the four rows are in bounds (asserted above); NEON is
        // baseline on aarch64.
        let v = unsafe { neon::dot4(x, &w[r * k..(r + 4) * k]) };
        out[r..r + 4].copy_from_slice(&v);
        r += 4;
    }
    while r < out.len() {
        out[r] = dot(x, &w[r * k..(r + 1) * k]);
        r += 1;
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// [`super::dot`] of `x` against four consecutive rows of `w`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot4(x: &[f32], w: &[f32]) -> [f32; 4] {
        const KC: usize = 256;
        let k = x.len();
        debug_assert_eq!(w.len(), 4 * k);
        let mut c = [0f32; 4];
        let mut at = 0;
        while at < k {
            let e = (at + KC).min(k);
            // SAFETY: `at..e` lies inside `x` and inside each row of `w`.
            let p = unsafe { block4(x, w, k, at, e) };
            for i in 0..4 {
                c[i] = if at == 0 { p[i] } else { c[i] + p[i] };
            }
            at = e;
        }
        c
    }

    #[inline(always)]
    unsafe fn block4(x: &[f32], w: &[f32], k: usize, s: usize, e: usize) -> [f32; 4] {
        // SAFETY: the caller keeps `s..e` within `x` and every row of `w`.
        unsafe {
            let main = s + (e - s) / 16 * 16;
            let xp = x.as_ptr();
            let wp = w.as_ptr();
            let z = vdupq_n_f32(0.0);
            let mut acc = [[z; 4]; 4]; // [row][accumulator vector]
            let mut i = s;
            while i < main {
                let xv = [
                    vld1q_f32(xp.add(i)),
                    vld1q_f32(xp.add(i + 4)),
                    vld1q_f32(xp.add(i + 8)),
                    vld1q_f32(xp.add(i + 12)),
                ];
                for r in 0..4 {
                    let row = wp.add(r * k + i);
                    for v in 0..4 {
                        acc[r][v] = vfmaq_f32(acc[r][v], xv[v], vld1q_f32(row.add(4 * v)));
                    }
                }
                i += 16;
            }
            let mut out = [0f32; 4];
            for r in 0..4 {
                let a = &acc[r];
                let t = vaddq_f32(vaddq_f32(a[1], a[0]), vaddq_f32(a[2], a[3]));
                let mut sum = (vgetq_lane_f32::<0>(t) + vgetq_lane_f32::<1>(t))
                    + (vgetq_lane_f32::<2>(t) + vgetq_lane_f32::<3>(t));
                for j in main..e {
                    sum = (*xp.add(j)).mul_add(*wp.add(r * k + j), sum);
                }
                out[r] = sum;
            }
            out
        }
    }
}

pub(crate) struct Maxout {
    w: Vec<f32>, // [nO * nP, nI]
    b: Vec<f32>, // [nO * nP]
    n_o: usize,
    n_p: usize,
}

impl Maxout {
    fn from(p: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<Self> {
        let (ws, w) = p.get("W").context("maxout W")?.clone();
        let (_, b) = p.get("b").context("maxout b")?.clone();
        Ok(Maxout {
            w,
            b,
            n_o: ws[0],
            n_p: ws[1],
        })
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut ys = vec![0f32; self.n_o * self.n_p];
        dot_rows(x, &self.w, &mut ys);
        let mut out = vec![0f32; self.n_o];
        for o in 0..self.n_o {
            let mut best = f32::NEG_INFINITY;
            for p in 0..self.n_p {
                let r = o * self.n_p + p;
                let y = ys[r] + self.b[r];
                if p == 0 || y > best {
                    best = y;
                }
            }
            out[o] = best;
        }
        out
    }
}

pub(crate) struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn from(p: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<Self> {
        Ok(LayerNorm {
            g: p.get("G").context("G")?.1.clone(),
            b: p.get("b").context("b")?.1.clone(),
        })
    }
    /// thinc `LayerNorm`: numpy float32 mean/var (+1e-8), `var ** -0.5`.
    fn forward(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let mu = np_sum_f32(x) / n;
        let d: Vec<f32> = x.iter().map(|v| v - mu).collect();
        let sq: Vec<f32> = d.iter().map(|v| v * v).collect();
        let var = np_sum_f32(&sq) / n + 1e-8f32;
        let inv = var.powf(-0.5);
        for i in 0..x.len() {
            x[i] = d[i] * inv * self.g[i] + self.b[i];
        }
    }
}

pub struct Tok2Vec {
    embeds: Vec<(Vec<f32>, usize, u32)>, // (E [nV, 96], nV, seed)
    static_w: Vec<f32>,                  // [96, 300]
    vectors: Vec<f32>,                   // [rows, 300]
    key2row: HashMap<u64, usize>,
    reduce: (Maxout, LayerNorm),
    blocks: Vec<(Maxout, LayerNorm)>,
    lexeme_norm: HashMap<u64, String>,
    base_norms: HashMap<String, String>,
    pub width: usize,
}

/// `spacy.lang.norm_exceptions.BASE_NORMS` (spaCy 3.8.14), dumped verbatim.
const BASE_NORMS_JSON: &str = include_str!("../data/base_norms.json");

impl Tok2Vec {
    /// Load `<pipeline>/tok2vec/model` and the vocab (`vectors`, `key2row`,
    /// `lookups.bin`) of a spaCy pipeline directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let params = thinc_params(&model_dir.join("tok2vec/model"))?;
        let mut embeds = Vec::new();
        let (mut maxouts, mut lns) = (Vec::new(), Vec::new());
        let mut static_w = Vec::new();
        for (name, p) in &params {
            match name.as_str() {
                "hashembed" => {
                    let (s, e) = p.get("E").context("E")?.clone();
                    let seed = 8 + embeds.len() as u32;
                    embeds.push((e, s[0], seed));
                }
                "static_vectors" => static_w = p.get("W").context("W")?.1.clone(),
                "maxout" => maxouts.push(Maxout::from(p)?),
                "layernorm" => lns.push(LayerNorm::from(p)?),
                _ => {}
            }
        }
        if embeds.len() != 6 || maxouts.len() != 5 || lns.len() != 5 || static_w.is_empty() {
            bail!("unexpected tok2vec layout");
        }
        let mut ml: Vec<(Maxout, LayerNorm)> = maxouts.into_iter().zip(lns).collect();
        let reduce = ml.remove(0);
        // vectors.npy
        let raw = std::fs::read(model_dir.join("vocab/vectors"))?;
        let hl = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        let header = std::str::from_utf8(&raw[10..10 + hl])?;
        if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
            bail!("vectors: unexpected npy header {header}");
        }
        let vectors: Vec<f32> = raw[10 + hl..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let k2r = std::fs::read(model_dir.join("vocab/key2row"))?;
        let kv = rmpv::decode::read_value(&mut k2r.as_slice())?;
        let key2row = kv
            .as_map()
            .context("key2row")?
            .iter()
            .map(|(k, v)| (k.as_u64().unwrap(), v.as_u64().unwrap() as usize))
            .collect();
        let lk = std::fs::read(model_dir.join("vocab/lookups.bin"))?;
        let lv = rmpv::decode::read_value(&mut lk.as_slice())?;
        let mut lexeme_norm = HashMap::new();
        if let Some(t) = lv
            .as_map()
            .and_then(|m| m.iter().find(|(k, _)| k.as_str() == Some("lexeme_norm")))
            .and_then(|(_, t)| t.as_map())
        {
            for (k, v) in t {
                lexeme_norm.insert(
                    k.as_u64().context("norm key")?,
                    v.as_str().context("norm value")?.to_string(),
                );
            }
        }
        let base_norms = serde_json::from_str(BASE_NORMS_JSON)?;
        Ok(Tok2Vec {
            embeds,
            static_w,
            vectors,
            key2row,
            reduce,
            blocks: ml,
            lexeme_norm,
            base_norms,
            width: 96,
        })
    }

    /// The lexeme `NORM` of a string (no special case).
    pub fn lex_norm(&self, s: &str) -> String {
        if let Some(n) = self.lexeme_norm.get(&hash_string(s)) {
            return n.clone();
        }
        if let Some(n) = self.base_norms.get(s) {
            return n.clone();
        }
        s.to_lowercase()
    }

    /// `doc.to_array([NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE])`.
    pub fn features(&self, toks: &[Token]) -> Vec<[u64; 6]> {
        toks.iter()
            .map(|t| {
                let norm = t.norm.clone().unwrap_or_else(|| self.lex_norm(&t.text));
                let cs: Vec<char> = t.text.chars().collect();
                let prefix: String = cs.iter().take(1).collect();
                let suffix: String = cs[cs.len().saturating_sub(3)..].iter().collect();
                let is_space = !cs.is_empty() && cs.iter().all(|&c| py_isspace(c));
                [
                    hash_string(&norm),
                    hash_string(&prefix),
                    hash_string(&suffix),
                    hash_string(&word_shape(&t.text)),
                    t.whitespace as u64,
                    is_space as u64,
                ]
            })
            .collect()
    }

    /// The `MultiHashEmbed` output (`[n, 96]`, before the encoder).
    pub fn embed(&self, toks: &[Token]) -> Vec<f32> {
        let feats = self.features(toks);
        let w = self.width;
        let mut out = Vec::with_capacity(toks.len() * w);
        let mut cat = vec![0f32; 7 * w];
        for (t, f) in toks.iter().zip(&feats) {
            for (c, (e, nv, seed)) in self.embeds.iter().enumerate() {
                let keys = thinc_hash(f[c], *seed);
                let dst = &mut cat[c * w..(c + 1) * w];
                dst.iter_mut().for_each(|x| *x = 0.0);
                for k in keys {
                    let row = &e[(k as usize % nv) * w..(k as usize % nv + 1) * w];
                    for i in 0..w {
                        dst[i] += row[i];
                    }
                }
            }
            let dst = &mut cat[6 * w..7 * w];
            match self.key2row.get(&hash_string(&t.text)) {
                Some(&r) => {
                    let v = &self.vectors[r * 300..(r + 1) * 300];
                    dot_rows(v, &self.static_w, dst);
                }
                None => dst.iter_mut().for_each(|x| *x = 0.0),
            }
            let mut y = self.reduce.0.forward(&cat);
            self.reduce.1.forward(&mut y);
            out.extend(y);
        }
        out
    }

    /// The full tok2vec output (`[n, 96]`). The encoder runs inside
    /// `with_array(pad=4)`: the doc is flattened between 4 zero rows on each
    /// side, and those rows go through every residual block too (they stop
    /// being zero after the first), so the edge windows see them.
    pub fn forward(&self, toks: &[Token]) -> Vec<f32> {
        const PAD: usize = 4;
        let w = self.width;
        let n = toks.len();
        let m = n + 2 * PAD;
        let mut x = vec![0f32; m * w];
        x[PAD * w..(PAD + n) * w].copy_from_slice(&self.embed(toks));
        for (mo, ln) in &self.blocks {
            let mut y = vec![0f32; m * w];
            let mut win = vec![0f32; 3 * w];
            for t in 0..m {
                win.iter_mut().for_each(|v| *v = 0.0);
                if t > 0 {
                    win[..w].copy_from_slice(&x[(t - 1) * w..t * w]);
                }
                win[w..2 * w].copy_from_slice(&x[t * w..(t + 1) * w]);
                if t + 1 < m {
                    win[2 * w..].copy_from_slice(&x[(t + 1) * w..(t + 2) * w]);
                }
                let mut r = mo.forward(&win);
                ln.forward(&mut r);
                for i in 0..w {
                    y[t * w + i] = x[t * w + i] + r[i];
                }
            }
            x = y;
        }
        x[PAD * w..(PAD + n) * w].to_vec()
    }
}
