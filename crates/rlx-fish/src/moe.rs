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

//! Drama 3's routed experts, run on the host straight from the checkpoint.
//!
//! The backbone holds 40 × 256 SwiGLU experts — 32 B parameters, 32 GB as
//! block-FP8 and four times that as f32 — of which a token fires 8 per layer.
//! So the experts never enter a graph: the layer graph hands back the router
//! logits and the normalized input, and this module reads the 8 chosen
//! experts out of the mapped shards, dequantizes them (`e4m3 · scale` per
//! 128 × 128 block) and applies them with BLAS. Dequantized experts are kept
//! in an LRU under a byte budget; consecutive frames reuse most of them.
//!
//! The router *logits* come from the graph; only the softmax / top-k runs
//! here, on exactly those values.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use rayon::prelude::*;
use safetensors::Dtype;

use crate::store::{ShardStore, e4m3, to_f32};

const FP8_BLOCK: usize = 128;
/// Up to this many rows per expert multiply straight from FP8 (decode);
/// more rows amortize a dequantized f32 copy and BLAS (prefill).
const DIRECT_MAX_ROWS: usize = 4;

/// `out[r, o] = Σ_i x[r, i] · w[o, i]` for an FP8 `w: [n_out, k]` with
/// `128 × 128` block scales, `x: [rows, k]` — parallel over output rows.
fn fp8_matmul_bt(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    rows: usize,
    k: usize,
    n_out: usize,
    out: &mut [f32],
) {
    let kb = k.div_ceil(FP8_BLOCK);
    let lut = crate::store::e4m3_table();
    let cols: Vec<f32> = (0..n_out)
        .into_par_iter()
        .with_min_len(16)
        .flat_map_iter(|o| {
            let wrow = &w[o * k..(o + 1) * k];
            let srow = &scales[(o / FP8_BLOCK) * kb..(o / FP8_BLOCK + 1) * kb];
            (0..rows).map(move |r| {
                let xr = &x[r * k..(r + 1) * k];
                let mut acc = 0f32;
                for (b, (wb, xb)) in wrow.chunks(FP8_BLOCK).zip(xr.chunks(FP8_BLOCK)).enumerate() {
                    let mut s = [0f32; 8];
                    for (wc, xc) in wb
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .zip(xb.as_chunks::<8>().0.iter())
                    {
                        for j in 0..8 {
                            s[j] += lut[wc[j] as usize] * xc[j];
                        }
                    }
                    let mut part: f32 = s.iter().sum();
                    let tail = wb.len() - wb.len() % 8;
                    for (wv, xv) in wb[tail..].iter().zip(&xb[tail..]) {
                        part += lut[*wv as usize] * xv;
                    }
                    acc += part * srow[b];
                }
                acc
            })
        })
        .collect();
    for o in 0..n_out {
        for r in 0..rows {
            out[r * n_out + o] = cols[o * rows + r];
        }
    }
}

/// One expert in f32: `gate`/`up` are `[ffn, hidden]`, `down` `[hidden, ffn]`.
struct Expert {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl Expert {
    fn bytes(&self) -> usize {
        4 * (self.gate.len() + self.up.len() + self.down.len())
    }
}

/// LRU of dequantized experts.
struct Cache {
    map: HashMap<(usize, usize), (Arc<Expert>, u64)>,
    bytes: usize,
    budget: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

/// The routed experts of every MoE layer.
pub struct ExpertBank {
    store: Arc<ShardStore>,
    prefix: String,
    hidden: usize,
    ffn: usize,
    top_k: usize,
    norm_topk: bool,
    cache: Mutex<Cache>,
}

impl ExpertBank {
    /// `prefix` is the layer stack (`model.language_model.layers.`);
    /// `budget` bytes of dequantized experts are cached.
    pub fn new(
        store: Arc<ShardStore>,
        prefix: &str,
        hidden: usize,
        ffn: usize,
        top_k: usize,
        norm_topk: bool,
        budget: usize,
    ) -> Self {
        Self {
            store,
            prefix: prefix.to_string(),
            hidden,
            ffn,
            top_k,
            norm_topk,
            cache: Mutex::new(Cache {
                map: HashMap::new(),
                bytes: 0,
                budget,
                clock: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// `(hits, misses)` of the expert cache so far.
    pub fn cache_stats(&self) -> (u64, u64) {
        let c = self.cache.lock().expect("expert cache");
        (c.hits, c.misses)
    }

    fn matrix(
        &self,
        layer: usize,
        expert: usize,
        proj: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let key = format!("{}{layer}.mlp.experts.{expert}.{proj}.weight", self.prefix);
        let info = self.store.info(&key)?;
        ensure!(
            info.shape == [rows, cols],
            "{key}: {:?}, want [{rows}, {cols}]",
            info.shape
        );
        let raw = self.store.bytes(&key)?;
        if info.dtype != Dtype::F8_E4M3 {
            return to_f32(&key, info.dtype, raw);
        }
        let scales = self.store.bytes(&format!("{key}_scale_inv"))?;
        let scales = to_f32(&key, Dtype::F32, scales)?;
        let (rb, cb) = (rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK));
        ensure!(
            scales.len() == rb * cb,
            "{key}: {} block scales for [{rows}, {cols}]",
            scales.len()
        );
        let mut out = vec![0f32; rows * cols];
        for (r, (dst, src)) in out.chunks_mut(cols).zip(raw.chunks(cols)).enumerate() {
            let srow = &scales[(r / FP8_BLOCK) * cb..(r / FP8_BLOCK + 1) * cb];
            for (c, (d, &b)) in dst.iter_mut().zip(src).enumerate() {
                *d = e4m3(b) * srow[c / FP8_BLOCK];
            }
        }
        Ok(out)
    }

    fn expert(&self, layer: usize, expert: usize) -> Result<Arc<Expert>> {
        {
            let mut c = self.cache.lock().expect("expert cache");
            c.clock += 1;
            let now = c.clock;
            if let Some((e, used)) = c.map.get_mut(&(layer, expert)) {
                *used = now;
                let e = e.clone();
                c.hits += 1;
                return Ok(e);
            }
            c.misses += 1;
        }
        let (h, f) = (self.hidden, self.ffn);
        let e = Arc::new(Expert {
            gate: self.matrix(layer, expert, "gate_proj", f, h)?,
            up: self.matrix(layer, expert, "up_proj", f, h)?,
            down: self.matrix(layer, expert, "down_proj", h, f)?,
        });
        let mut c = self.cache.lock().expect("expert cache");
        let size = e.bytes();
        while c.bytes + size > c.budget && !c.map.is_empty() {
            let oldest = *c
                .map
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .expect("non-empty")
                .0;
            if let Some((old, _)) = c.map.remove(&oldest) {
                c.bytes -= old.bytes();
            }
        }
        if size <= c.budget {
            let now = c.clock;
            c.map.insert((layer, expert), (e.clone(), now));
            c.bytes += size;
        }
        Ok(e)
    }

    fn cached(&self, layer: usize, expert: usize) -> bool {
        self.cache
            .lock()
            .expect("expert cache")
            .map
            .contains_key(&(layer, expert))
    }

    /// The stored bytes and block scales of an FP8 matrix (`None` if the
    /// checkpoint stores it in another dtype).
    fn fp8(
        &self,
        layer: usize,
        expert: usize,
        proj: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<(&[u8], Vec<f32>)>> {
        let key = format!("{}{layer}.mlp.experts.{expert}.{proj}.weight", self.prefix);
        let info = self.store.info(&key)?;
        if info.dtype != Dtype::F8_E4M3 {
            return Ok(None);
        }
        ensure!(
            info.shape == [rows, cols],
            "{key}: {:?}, want [{rows}, {cols}]",
            info.shape
        );
        let scales = to_f32(
            &key,
            Dtype::F32,
            self.store.bytes(&format!("{key}_scale_inv"))?,
        )?;
        ensure!(
            scales.len() == rows.div_ceil(FP8_BLOCK) * cols.div_ceil(FP8_BLOCK),
            "{key}: block scales"
        );
        Ok(Some((self.store.bytes(&key)?, scales)))
    }

    /// One expert on `n` rows straight from the mapped FP8 weights — no f32
    /// copy of the expert is made. `None` when the expert is not FP8.
    fn direct(
        &self,
        layer: usize,
        expert: usize,
        xs: &[f32],
        n: usize,
    ) -> Result<Option<Vec<f32>>> {
        let (h, f) = (self.hidden, self.ffn);
        let Some(gate) = self.fp8(layer, expert, "gate_proj", f, h)? else {
            return Ok(None);
        };
        let Some(up) = self.fp8(layer, expert, "up_proj", f, h)? else {
            return Ok(None);
        };
        let Some(down) = self.fp8(layer, expert, "down_proj", h, f)? else {
            return Ok(None);
        };
        let mut g = vec![0f32; n * f];
        let mut u = vec![0f32; n * f];
        fp8_matmul_bt(xs, gate.0, &gate.1, n, h, f, &mut g);
        fp8_matmul_bt(xs, up.0, &up.1, n, h, f, &mut u);
        for (gv, uv) in g.iter_mut().zip(&u) {
            *gv = *gv / (1.0 + (-*gv).exp()) * uv;
        }
        let mut y = vec![0f32; n * h];
        fp8_matmul_bt(&g, down.0, &down.1, n, f, h, &mut y);
        Ok(Some(y))
    }

    /// Top-k routing of one row: `(expert, weight)` pairs, most likely first.
    pub fn route(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
        let mut picks: Vec<(usize, f32)> =
            order[..self.top_k].iter().map(|&i| (i, e[i] / s)).collect();
        if self.norm_topk {
            let t: f32 = picks.iter().map(|p| p.1).sum();
            for p in &mut picks {
                p.1 /= t;
            }
        }
        picks
    }

    /// Routed-expert output `[rows, hidden]` for `x: [rows, hidden]` given the
    /// router logits `[rows, num_experts]`.
    pub fn forward(
        &self,
        layer: usize,
        x: &[f32],
        router: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let (h, f) = (self.hidden, self.ffn);
        ensure!(
            x.len() == rows * h,
            "moe input is {} values for {rows} rows",
            x.len()
        );
        let n_exp = router.len() / rows;
        // expert → [(row, weight)], in row order.
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp];
        for r in 0..rows {
            for (e, w) in self.route(&router[r * n_exp..(r + 1) * n_exp]) {
                by_expert[e].push((r, w));
            }
        }
        let active: Vec<usize> = (0..n_exp).filter(|&e| !by_expert[e].is_empty()).collect();
        let partial: Vec<Result<(usize, Vec<f32>)>> = active
            .par_iter()
            .map(|&e| {
                let picks = &by_expert[e];
                let n = picks.len();
                let mut xs = Vec::with_capacity(n * h);
                for &(r, _) in picks {
                    xs.extend_from_slice(&x[r * h..(r + 1) * h]);
                }
                if n <= DIRECT_MAX_ROWS
                    && !self.cached(layer, e)
                    && let Some(y) = self.direct(layer, e, &xs, n)?
                {
                    return Ok((e, y));
                }
                let ex = self.expert(layer, e)?;
                let mut g = vec![0f32; n * f];
                let mut u = vec![0f32; n * f];
                rlx_core::host_kernels::matmul_bt(&xs, &ex.gate, &mut g, n, h, f, 1.0);
                rlx_core::host_kernels::matmul_bt(&xs, &ex.up, &mut u, n, h, f, 1.0);
                for (gv, uv) in g.iter_mut().zip(&u) {
                    *gv = *gv / (1.0 + (-*gv).exp()) * uv;
                }
                let mut y = vec![0f32; n * h];
                rlx_core::host_kernels::matmul_bt(&g, &ex.down, &mut y, n, f, h, 1.0);
                Ok((e, y))
            })
            .collect();
        let mut out = vec![0f32; rows * h];
        for res in partial {
            let (e, y) = res?;
            for (i, &(r, w)) in by_expert[e].iter().enumerate() {
                for (o, v) in out[r * h..(r + 1) * h]
                    .iter_mut()
                    .zip(&y[i * h..(i + 1) * h])
                {
                    *o += w * v;
                }
            }
        }
        Ok(out)
    }
}
