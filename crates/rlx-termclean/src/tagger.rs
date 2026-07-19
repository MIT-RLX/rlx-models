//! Loadable-weights ML tagger — the INFERENCE path for the per-char
//! content/chrome model trained by `bin/train_multi.rs`.
//!
//! This mirrors the training forward pass exactly (rezero-gain residuals,
//! tanh-bounded Q/K, the vertical-consistency extra feature) but runs
//! forward-only (`with_loss=false` → raw logits), loads its weights from the
//! self-describing bundle (`manifest.json` + `vocab.txt` + `weights.f32`), and
//! produces cleaned text that mirrors [`crate::fastclean`]'s output contract.
//!
//! Only compiled with the `infer` feature (pulls `rlx-tensor`'s `eval-metal`
//! forward stack + `serde_json`); the default build stays pure-std zero-dep.

use std::collections::HashMap;
use std::path::Path;

use rlx_tensor::{Device, Func, GraphScope, MaskKind, Tensor, is_available, shape};

use crate::fastclean::{is_pager_status, strip_ansi};

// Architecture constants — must match `train_multi.rs` (and the shipped
// bundle's `arch`). `layers` is read from the bundle, not hard-coded.
const L: usize = 200; // seq_len
const NH: usize = 4; // n_heads
const DH: usize = 16; // head_dim
const D: usize = NH * DH; // d_model = 64
const FF: usize = 4 * D; // 256

/// A loaded tagger: bound params + vocab, ready to tag/clean frames.
pub struct Tagger {
    params: HashMap<String, Vec<f32>>,
    vocab: HashMap<char, usize>,
    v: usize,
    layers: usize,
    dev: Device,
}

/// One entry of the manifest's `params` table (offset/len are f32 ELEMENT
/// counts into `weights.f32`).
#[derive(serde::Deserialize)]
struct ParamEntry {
    name: String,
    offset: usize,
    len: usize,
}

#[derive(serde::Deserialize)]
struct Arch {
    layers: usize,
    vocab: usize,
}

#[derive(serde::Deserialize)]
struct Manifest {
    arch: Arch,
    params: Vec<ParamEntry>,
}

impl Tagger {
    /// Load the tagger bundle from `dir` (`manifest.json`, `vocab.txt`,
    /// `weights.f32`). Picks Metal if available, else CPU.
    pub fn load(dir: impl AsRef<Path>) -> std::io::Result<Tagger> {
        let dir = dir.as_ref();
        let manifest_raw = std::fs::read_to_string(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&manifest_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // vocab.txt: one decimal Unicode codepoint per line; line index i → char
        // id i+1 (id 0 is UNK/pad, so it maps no char).
        let vocab_raw = std::fs::read_to_string(dir.join("vocab.txt"))?;
        let mut vocab: HashMap<char, usize> = HashMap::new();
        for (i, line) in vocab_raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(cp) = line.parse::<u32>() {
                if let Some(c) = char::from_u32(cp) {
                    vocab.insert(c, i + 1);
                }
            }
        }

        // weights.f32: raw little-endian f32, sliced per param by offset/len.
        let bytes = std::fs::read(dir.join("weights.f32"))?;
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        for p in &manifest.params {
            let start = p.offset * 4;
            let end = (p.offset + p.len) * 4;
            if end > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "param {:?} slice [{start}..{end}] exceeds weights.f32 ({} bytes)",
                        p.name,
                        bytes.len()
                    ),
                ));
            }
            let data: Vec<f32> = bytes[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            params.insert(p.name.clone(), data);
        }

        let dev = if is_available(Device::Metal) {
            Device::Metal
        } else {
            Device::Cpu
        };

        Ok(Tagger {
            params,
            vocab,
            v: manifest.arch.vocab,
            layers: manifest.arch.layers,
            dev,
        })
    }

    /// Build the forward-only `Func` (logits) for the loaded arch, binding all
    /// params. B is fixed to 1 (single-line inference).
    fn build(&self) -> Func {
        let v = self.v;
        let layers = self.layers;
        let mut f = Func::new("tagger", move |s| forward(s, v, layers));
        for (name, data) in &self.params {
            f = f.with_param(name.clone(), data.clone());
        }
        f
    }

    /// Tag one stripped line (chars) given its per-column vertical-consistency
    /// feature. Returns one bool per input char (`true` = content), truncated to
    /// L; positions past the truncation are simply absent from the result.
    pub fn tag_line(&self, stripped: &[char], vdiv: &[f32]) -> Vec<bool> {
        let n = stripped.len().min(L);
        // one-hot x: [1, L, v]; pos identity: [L, L]; vd: [L, 1].
        let mut onehot = vec![0f32; L * self.v];
        let mut pos = vec![0f32; L * L];
        let mut vd = vec![0f32; L];
        for i in 0..L {
            let id = if i < n {
                *self.vocab.get(&stripped[i]).unwrap_or(&0)
            } else {
                0 // pad
            };
            onehot[i * self.v + id] = 1.0;
            pos[i * L + i] = 1.0;
            vd[i] = if i < n {
                vdiv.get(i).copied().unwrap_or(0.0)
            } else {
                0.0
            };
        }
        let model = self.build();
        let logits = model
            .run_on(self.dev, &[("x", &onehot), ("pos", &pos), ("vd", &vd)])
            .remove(0);
        // Decision: logit > 0.0 (matches training's pre-sigmoid threshold).
        (0..n).map(|i| logits[i] > 0.0).collect()
    }

    /// Clean one raw frame → content text, mirroring
    /// [`crate::fastclean::clean_frame_into`]'s output contract: per line strip
    /// ANSI, drop bare pager status, keep the tagger's content chars, then join
    /// the non-empty trimmed lines with '\n'.
    pub fn clean_frame(&self, frame: &str) -> String {
        // Strip ANSI once for every line and drop bare pager prompts up front,
        // so both vdiv and tagging see the same stripped char rows.
        let mut sbuf = String::new();
        let mut stripped_rows: Vec<Vec<char>> = Vec::new();
        for line in frame.lines() {
            strip_ansi(line, &mut sbuf);
            if is_pager_status(&sbuf) {
                // Dropped line: keep a placeholder row so it contributes nothing
                // and produces empty output (mirrors fastclean returning "").
                stripped_rows.push(Vec::new());
            } else {
                stripped_rows.push(sbuf.chars().collect());
            }
        }

        let vdiv = compute_vdiv(&stripped_rows);

        let mut out_lines: Vec<String> = Vec::new();
        for (li, row) in stripped_rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            let tags = self.tag_line(row, &vdiv[li]);
            let mut out = String::with_capacity(row.len());
            for (i, &keep) in tags.iter().enumerate() {
                if keep {
                    out.push(row[i]);
                }
            }
            if !out.trim().is_empty() {
                out_lines.push(out);
            }
        }
        out_lines.join("\n")
    }

    /// Batched clean (sequential; see the serve binary for the threading note).
    pub fn clean_batch(&self, frames: &[&str]) -> Vec<String> {
        frames.iter().map(|f| self.clean_frame(f)).collect()
    }
}

/// Forward pass, ported verbatim from `train_multi.rs::forward` with `B=1` and
/// `with_loss=false` — returns the per-position logits `[L]` (flattened `[L,1]`).
fn forward(s: &mut GraphScope, v: usize, layers: usize) -> Tensor {
    const B: usize = 1;
    let bl = (B * L) as i64;
    let heads = |t: Tensor| t.reshape(vec![B as i64, L as i64, NH as i64, DH as i64]);
    let x = s.input("x", shape![B, L, v]);
    let pos = s.input("pos", shape![B * L, L]);
    let we = s.param("we", shape![v, D]);
    let pe = s.param("pe", shape![L, D]);
    let emb = x.reshape(vec![bl, v as i64]).matmul(&we);
    // vertical-consistency feature (per column) projected into the hidden.
    let vd = s.input("vd", shape![B * L, 1]);
    let wvd = s.param("wvd", shape![1, D]);
    let mut h = &(&emb + &pos.matmul(&pe)) + &vd.matmul(&wvd);
    for i in 0..layers {
        let wq = s.param(format!("wq{i}"), shape![D, D]);
        let wk = s.param(format!("wk{i}"), shape![D, D]);
        let wv = s.param(format!("wv{i}"), shape![D, D]);
        // tanh-bounded Q/K (matches training; NOT ablated at inference time).
        let q = heads(h.matmul(&wq).tanh());
        let k = heads(h.matmul(&wk).tanh());
        let vv = heads(h.matmul(&wv));
        let attn = q
            .attention(&k, &vv, NH, DH, MaskKind::None)
            .reshape(vec![bl, D as i64]);
        // ReZero gate on the attention branch.
        let ga = s.param(format!("ga{i}"), shape![D]);
        h = &h + &(&attn * &ga);
        let w1 = s.param(format!("w1{i}"), shape![D, FF]);
        let b1 = s.param(format!("b1{i}"), shape![FF]);
        let w2 = s.param(format!("w2{i}"), shape![FF, D]);
        let b2 = s.param(format!("b2{i}"), shape![D]);
        let ff = &(&h.matmul(&w1) + &b1).gelu().matmul(&w2) + &b2;
        let gf = s.param(format!("gf{i}"), shape![D]);
        h = &h + &(&ff * &gf);
    }
    let wo = s.param("wo", shape![D, 1]);
    let bo = s.param("bo", shape![1]);
    &h.matmul(&wo) + &bo
}

/// Single-frame vertical-consistency feature: all lines are treated as one
/// group (one screen = one app). For each column, the fraction of non-space
/// chars equal to the modal char (dividers ≈ 1.0). Ported from
/// `train_multi.rs::compute_vdiv` specialized to a single group.
fn compute_vdiv(rows: &[Vec<char>]) -> Vec<Vec<f32>> {
    let n = rows.len();
    let mut out = vec![vec![0f32; L]; n];
    // truncated views (take L)
    let trows: Vec<&[char]> = rows.iter().map(|r| &r[..r.len().min(L)]).collect();
    let mut cons = vec![0f32; L];
    for c in 0..L {
        let mut counts: HashMap<char, u32> = HashMap::new();
        let mut nonspace = 0u32;
        for r in &trows {
            if let Some(&ch) = r.get(c) {
                if ch != ' ' {
                    *counts.entry(ch).or_insert(0) += 1;
                    nonspace += 1;
                }
            }
        }
        if nonspace >= 3 {
            cons[c] = counts.values().copied().max().unwrap_or(0) as f32 / nonspace as f32;
        }
    }
    for (li, r) in trows.iter().enumerate() {
        for c in 0..L.min(r.len()) {
            if r[c] != ' ' {
                out[li][c] = cons[c];
            }
        }
    }
    out
}
