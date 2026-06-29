//! Streaming depth decoder for Kyutai TTS (`depformer.layers` + `depformer_emb`).
//!
//! At each temporal frame the DepFormer walks codebooks `0..dep_q-1`, maintaining
//! a KV cache along the codebook axis (not time). Per-codebook input projections
//! (`depformer_in`) and output heads (`linears`) follow the published schedule.

use crate::config::KyutaiTtsConfig;
use crate::low_rank_embedding::LowRankEmbedding;
use crate::nn::{linear, rms_norm, swiglu_mlp};
use crate::util::{take_mat2, take_rms_alpha};
use crate::weights::WeightMap;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Axis};

/// Demuxed DepFormer text embedding (`depformer_text_emb.out{1,2}` + low-rank table).
#[derive(Debug)]
struct DemuxDepformerTextEmb {
    table: Array2<f32>,
    out1: Array2<f32>,
    out2: Array2<f32>,
    card: usize,
    dim: usize,
}

impl DemuxDepformerTextEmb {
    fn load(weights: &WeightMap, card: usize) -> Result<Self> {
        let table = take_mat2(weights, "depformer_text_emb.weight")?;
        let out1 = take_mat2(weights, "depformer_text_emb.out1.weight")?;
        let out2 = take_mat2(weights, "depformer_text_emb.out2.weight")?;
        let dim = out1.nrows();
        Ok(Self {
            table,
            out1,
            out2,
            card,
            dim,
        })
    }

    fn rank_vec(&self, id: u32) -> Array1<f32> {
        self.table.row(id as usize).to_owned()
    }

    fn forward_one(&self, token: u32) -> Array1<f32> {
        if token == u32::MAX {
            return Array1::<f32>::zeros(self.dim);
        }
        let card = self.card as u32;
        let left_id = token % card;
        let right = (token / card) as i32 - 1;
        let left = linear(
            self.rank_vec(left_id).view().insert_axis(Axis(0)),
            &self.out1,
        )
        .row(0)
        .to_owned();
        if right < 0 {
            return left;
        }
        let right_proj = linear(
            self.rank_vec(right as u32).view().insert_axis(Axis(0)),
            &self.out2,
        );
        let right_row = right_proj.row(0);
        let mut y = left;
        for (a, &b) in y.iter_mut().zip(right_row.iter()) {
            *a += b;
        }
        y
    }
}

#[derive(Debug, Clone)]
struct DepLayerWeights {
    norm1_alpha: Array1<f32>,
    norm2_alpha: Array1<f32>,
    /// Moshi stacks `mult` per-step QKV projections in one tensor → split to `[3072, dim]` each.
    in_projs: Vec<Array2<f32>>,
    out_projs: Vec<Array2<f32>>,
    /// `head_id -> (gate_in, gate_out)` SwiGLU weights for this layer.
    gating: Vec<(Array2<f32>, Array2<f32>)>,
}

#[derive(Debug)]
struct DepKvCache {
    k: Array3<f32>,
    v: Array3<f32>,
    len: usize,
    capacity: usize,
}

impl DepKvCache {
    fn new(num_heads: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            k: Array3::zeros((num_heads, capacity, head_dim)),
            v: Array3::zeros((num_heads, capacity, head_dim)),
            len: 0,
            capacity,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn append(&mut self, k_new: &Array3<f32>, v_new: &Array3<f32>) {
        let slot = self.len;
        if slot >= self.capacity {
            return;
        }
        for h in 0..self.k.dim().0 {
            for d in 0..self.k.dim().2 {
                self.k[[h, slot, d]] = k_new[[h, 0, d]];
                self.v[[h, slot, d]] = v_new[[h, 0, d]];
            }
        }
        self.len += 1;
    }

    fn effective_len(&self) -> usize {
        self.len
    }
}

#[derive(Debug)]
struct DepLayer {
    w: DepLayerWeights,
    kv: DepKvCache,
    num_heads: usize,
    head_dim: usize,
}

/// Per-frame streaming DepFormer (codebook axis).
pub struct DepformerStream {
    layers: Vec<DepLayer>,
    depformer_in: Vec<Array2<f32>>,
    linears: Vec<Array2<f32>>,
    depformer_text_emb: DemuxDepformerTextEmb,
    depformer_emb: Vec<LowRankEmbedding>,
    schedule: Vec<usize>,
    dep_q: usize,
}

impl DepformerStream {
    pub fn load(cfg: &KyutaiTtsConfig, weights: &WeightMap) -> Result<Self> {
        let df = &cfg.depformer;
        let dep_q = cfg.dep_q;
        let d_dep = df.dim;
        let num_heads = df.num_heads;
        let head_dim = d_dep / num_heads;
        let num_weight_sets = df
            .weights_per_step_schedule
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(11);

        let mut depformer_in = Vec::with_capacity(num_weight_sets);
        for h in 0..num_weight_sets {
            depformer_in.push(take_mat2(weights, &format!("depformer_in.{h}.weight"))?);
        }

        let mut linears = Vec::with_capacity(dep_q);
        for cb in 0..dep_q {
            linears.push(take_mat2(weights, &format!("linears.{cb}.weight"))?);
        }

        let depformer_text_emb = DemuxDepformerTextEmb::load(weights, cfg.text_card + 1)?;
        let mut depformer_emb = Vec::with_capacity(dep_q.saturating_sub(1));
        for cb in 0..dep_q.saturating_sub(1) {
            depformer_emb.push(load_low_rank(weights, &format!("depformer_emb.{cb}"))?);
        }

        let mut layers = Vec::with_capacity(df.num_layers);
        for li in 0..df.num_layers {
            let p = format!("depformer.layers.{li}.");
            let mut gating = Vec::with_capacity(num_weight_sets);
            for h in 0..num_weight_sets {
                gating.push((
                    take_mat2(weights, &format!("{p}gating.{h}.linear_in.weight"))?,
                    take_mat2(weights, &format!("{p}gating.{h}.linear_out.weight"))?,
                ));
            }
            let in_full = take_mat2(weights, &format!("{p}self_attn.in_proj_weight"))?;
            let out_full = take_mat2(weights, &format!("{p}self_attn.out_proj.weight"))?;
            layers.push(DepLayer {
                w: DepLayerWeights {
                    norm1_alpha: take_rms_alpha(weights, &format!("{p}norm1.alpha"))?,
                    norm2_alpha: take_rms_alpha(weights, &format!("{p}norm2.alpha"))?,
                    in_projs: split_weight_rows(in_full, num_weight_sets)?,
                    out_projs: split_weight_rows(out_full, num_weight_sets)?,
                    gating,
                },
                kv: DepKvCache::new(num_heads, head_dim, dep_q.max(1)),
                num_heads,
                head_dim,
            });
        }

        Ok(Self {
            layers,
            depformer_in,
            linears,
            depformer_text_emb,
            depformer_emb,
            schedule: df.weights_per_step_schedule.clone(),
            dep_q,
        })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.kv.reset();
        }
    }

    fn head_for(&self, codebook: usize) -> Result<usize> {
        self.schedule
            .get(codebook)
            .copied()
            .with_context(|| format!("no schedule entry for codebook {codebook}"))
    }

    fn embed_token(&self, codebook: usize, token: u32) -> Result<Array1<f32>> {
        if codebook == 0 {
            Ok(self.depformer_text_emb.forward_one(token))
        } else {
            Ok(self
                .depformer_emb
                .get(codebook - 1)
                .with_context(|| format!("missing depformer_emb.{}", codebook - 1))?
                .forward_one(token))
        }
    }

    /// Sample logits for one codebook slot at the current temporal frame.
    pub fn forward_codebook(
        &mut self,
        codebook: usize,
        temporal_hidden: &Array1<f32>,
        token: u32,
    ) -> Result<Array1<f32>> {
        let head = self.head_for(codebook)?;
        let in_w = self
            .depformer_in
            .get(head)
            .with_context(|| format!("missing depformer_in.{head}"))?;
        let h_view = temporal_hidden.view().insert_axis(ndarray::Axis(0));
        let mut x = linear(h_view, in_w);
        let emb = self.embed_token(codebook, token)?;
        for (i, v) in emb.iter().enumerate() {
            x[[0, i]] += v;
        }

        for layer in &mut self.layers {
            let nh = layer.num_heads;
            let hd = layer.head_dim;
            let n1 = rms_norm(x.view(), &layer.w.norm1_alpha);
            let in_proj = &layer.w.in_projs[head];
            let out_proj = &layer.w.out_projs[head];
            let attn = dep_self_attention(in_proj, out_proj, &mut layer.kv, &n1, nh, hd);
            x = &x + &attn;
            let n2 = rms_norm(x.view(), &layer.w.norm2_alpha);
            let (gate_in, gate_out) = &layer.w.gating[head];
            let mlp = swiglu_mlp(n2.view(), gate_in, gate_out);
            x = &x + &mlp;
        }

        let logits = linear(x.view(), &self.linears[codebook]);
        Ok(logits.row(0).to_owned())
    }

    pub fn dep_q(&self) -> usize {
        self.dep_q
    }
}

fn split_weight_rows(w: Array2<f32>, n_sets: usize) -> Result<Vec<Array2<f32>>> {
    let rows = w.nrows();
    anyhow::ensure!(
        rows.is_multiple_of(n_sets),
        "weight rows {rows} not divisible by {n_sets} sets"
    );
    let chunk = rows / n_sets;
    Ok((0..n_sets)
        .map(|i| {
            w.slice(ndarray::s![i * chunk..(i + 1) * chunk, ..])
                .to_owned()
        })
        .collect())
}

fn dep_self_attention(
    in_proj: &Array2<f32>,
    out_proj: &Array2<f32>,
    kv: &mut DepKvCache,
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let nh = num_heads;
    let hd = head_dim;
    let d_model = nh * hd;
    let qkv = linear(x.view(), in_proj);
    let mut q = Array3::<f32>::zeros((nh, 1, hd));
    let mut k = Array3::<f32>::zeros((nh, 1, hd));
    let mut v = Array3::<f32>::zeros((nh, 1, hd));
    for hi in 0..nh {
        for di in 0..hd {
            let base = hi * hd + di;
            q[[hi, 0, di]] = qkv[[0, base]];
            k[[hi, 0, di]] = qkv[[0, nh * hd + base]];
            v[[hi, 0, di]] = qkv[[0, 2 * nh * hd + base]];
        }
    }
    kv.append(&k, &v);
    let k_len = kv.effective_len();
    let q_pos = k_len.saturating_sub(1);
    let mut attn_out = Array2::<f32>::zeros((1, d_model));
    let scale = 1.0 / (hd as f32).sqrt();
    for hi in 0..nh {
        let mut weights = Vec::with_capacity(k_len);
        let mut slots = Vec::with_capacity(k_len);
        for ki in 0..k_len {
            if ki > q_pos {
                continue;
            }
            let mut dot = 0.0f32;
            for di in 0..hd {
                dot += q[[hi, 0, di]] * kv.k[[hi, ki, di]];
            }
            weights.push((dot * scale).exp());
            slots.push(ki);
        }
        let sum: f32 = weights.iter().sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (wi, &slot) in slots.iter().enumerate() {
            let w = weights[wi] * inv;
            for di in 0..hd {
                attn_out[[0, hi * hd + di]] += w * kv.v[[hi, slot, di]];
            }
        }
    }
    linear(attn_out.view(), out_proj)
}

fn load_low_rank(weights: &WeightMap, prefix: &str) -> Result<LowRankEmbedding> {
    let a = take_mat2(weights, &format!("{prefix}.weight"))?;
    let b_stored = take_mat2(weights, &format!("{prefix}.low_rank.weight"))?;
    // Checkpoint layout: `weight` is `[card, rank]`, `low_rank.weight` is `[dim, rank]`
    // (row-major). Transpose the factor to `[rank, dim]` for `LowRankEmbedding`.
    let b = if a.ncols() == b_stored.nrows() {
        b_stored
    } else if a.ncols() == b_stored.ncols() {
        b_stored.t().to_owned()
    } else {
        anyhow::bail!(
            "{prefix}: expected rank {} in low_rank.weight, got {:?}",
            a.ncols(),
            b_stored.dim()
        );
    };
    Ok(LowRankEmbedding::new(a, b))
}
