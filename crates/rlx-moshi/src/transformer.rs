use crate::config::{PositionalEmbedding, TransformerConfig};
use crate::nn::{apply_rope_vec, linear, rms_norm, rope_tables, swiglu_mlp};
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct AttnWeights {
    in_proj: Array2<f32>,
    out_proj: Array2<f32>,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Debug, Clone)]
struct LayerWeights {
    norm1_alpha: Array1<f32>,
    norm2_alpha: Array1<f32>,
    attn: AttnWeights,
    gate_in: Array2<f32>,
    gate_out: Array2<f32>,
}

#[derive(Debug, Clone)]
struct KvCache {
    k: Array3<f32>,
    v: Array3<f32>,
    len: usize,
    context: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct KvSnapshot {
    layers: Vec<KvSnapshotLayer>,
    seq_len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct KvSnapshotLayer {
    k: Array3<f32>,
    v: Array3<f32>,
    len: usize,
}

impl KvCache {
    fn new(num_heads: usize, head_dim: usize, context: usize) -> Self {
        Self {
            k: Array3::<f32>::zeros((num_heads, context, head_dim)),
            v: Array3::<f32>::zeros((num_heads, context, head_dim)),
            len: 0,
            context,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn append(&mut self, k_new: &Array3<f32>, v_new: &Array3<f32>) {
        let t = k_new.dim().1;
        for step in 0..t {
            let slot = self.len % self.context;
            for h in 0..self.k.dim().0 {
                for d in 0..self.k.dim().2 {
                    self.k[[h, slot, d]] = k_new[[h, step, d]];
                    self.v[[h, slot, d]] = v_new[[h, step, d]];
                }
            }
            self.len += 1;
        }
    }

    fn effective_len(&self) -> usize {
        self.len.min(self.context)
    }

    fn snapshot(&self) -> KvSnapshotLayer {
        KvSnapshotLayer {
            k: self.k.clone(),
            v: self.v.clone(),
            len: self.len,
        }
    }

    fn restore(&mut self, snap: &KvSnapshotLayer) {
        self.k = snap.k.clone();
        self.v = snap.v.clone();
        self.len = snap.len;
    }
}

#[derive(Debug, Clone)]
struct AttnState {
    kv: KvCache,
}

impl AttnState {
    fn attention(
        &mut self,
        w: &AttnWeights,
        x: &Array2<f32>,
        rope_cos: Option<&Array2<f32>>,
        rope_sin: Option<&Array2<f32>>,
        causal: bool,
    ) -> Array2<f32> {
        let (t, d_model) = x.dim();
        let h = w.num_heads;
        let hd = w.head_dim;
        let qkv = linear(x.view(), &w.in_proj);
        let mut q_heads = Array3::<f32>::zeros((h, t, hd));
        let mut k_heads = Array3::<f32>::zeros((h, t, hd));
        let mut v_heads = Array3::<f32>::zeros((h, t, hd));
        for ti in 0..t {
            for hi in 0..h {
                for di in 0..hd {
                    let base = hi * hd + di;
                    q_heads[[hi, ti, di]] = qkv[[ti, base]];
                    k_heads[[hi, ti, di]] = qkv[[ti, h * hd + base]];
                    v_heads[[hi, ti, di]] = qkv[[ti, 2 * h * hd + base]];
                }
            }
        }
        if let (Some(cos), Some(sin)) = (rope_cos, rope_sin) {
            for ti in 0..t {
                for hi in 0..h {
                    let mut q = q_heads.slice_mut(ndarray::s![hi, ti, ..]).to_vec();
                    let mut k = k_heads.slice_mut(ndarray::s![hi, ti, ..]).to_vec();
                    apply_rope_vec(&mut q, &mut k, cos.row(ti), sin.row(ti));
                    for di in 0..hd {
                        q_heads[[hi, ti, di]] = q[di];
                        k_heads[[hi, ti, di]] = k[di];
                    }
                }
            }
        }
        self.kv.append(&k_heads, &v_heads);
        let k_len = self.kv.effective_len();
        let slots: Vec<usize> = if self.kv.len <= self.kv.context {
            (0..self.kv.len).collect()
        } else {
            (0..self.kv.context).collect()
        };
        let mut out = Array2::<f32>::zeros((t, d_model));
        let scale = 1.0 / (hd as f32).sqrt();
        for ti in 0..t {
            let q_pos = self.kv.len - t + ti;
            for hi in 0..h {
                let mut attn_weights = Vec::new();
                let mut attn_vals = Vec::new();
                for (ki, &slot) in slots.iter().enumerate() {
                    let k_pos = if self.kv.len <= self.kv.context {
                        ki
                    } else {
                        self.kv.len - k_len + ki
                    };
                    if causal && k_pos > q_pos {
                        continue;
                    }
                    if q_pos + 1 > k_pos + self.kv.context {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for di in 0..hd {
                        dot += q_heads[[hi, ti, di]] * self.kv.k[[hi, slot, di]];
                    }
                    attn_weights.push((dot * scale).exp());
                    attn_vals.push((hi, slot));
                }
                let sum: f32 = attn_weights.iter().sum();
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for (wi, &(hi2, slot)) in attn_vals.iter().enumerate() {
                    let wgt = attn_weights[wi] * inv;
                    for di in 0..hd {
                        out[[ti, hi2 * hd + di]] += wgt * self.kv.v[[hi2, slot, di]];
                    }
                }
            }
        }
        linear(out.view(), &w.out_proj)
    }

    fn copy_from(&mut self, other: &Self) {
        self.kv = other.kv.clone();
    }
}

#[derive(Debug, Clone)]
struct Layer {
    w: LayerWeights,
    attn: AttnState,
}

impl Layer {
    fn forward(
        &mut self,
        x: &Array2<f32>,
        rope_cos: Option<&Array2<f32>>,
        rope_sin: Option<&Array2<f32>>,
        causal: bool,
    ) -> Array2<f32> {
        let n1 = rms_norm(x.view(), &self.w.norm1_alpha);
        let attn_out = self
            .attn
            .attention(&self.w.attn, &n1, rope_cos, rope_sin, causal);
        let mut h = x + attn_out;
        let n2 = rms_norm(h.view(), &self.w.norm2_alpha);
        let mlp_out = swiglu_mlp(n2.view(), &self.w.gate_in, &self.w.gate_out);
        h = &h + &mlp_out;
        h
    }

    fn reset(&mut self) {
        self.attn.kv.reset();
    }

    fn copy_state_from(&mut self, other: &Self) {
        self.attn.copy_from(&other.attn);
    }
}

/// Streaming causal transformer (Helium temporal or DepFormer slice).
#[derive(Debug)]
pub struct StreamingTransformer {
    layers: Vec<Layer>,
    cfg: TransformerConfig,
    seq_len: usize,
}

impl StreamingTransformer {
    pub fn build(
        cfg: &TransformerConfig,
        weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        ensure!(cfg.kv_repeat == 1, "only kv_repeat=1 supported");
        let head_dim = cfg.d_model / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            let p = format!("transformer.layers.{li}.");
            let norm1_alpha = take_vec1(weights, &format!("{p}norm1.alpha"))?;
            let norm2_alpha = take_vec1(weights, &format!("{p}norm2.alpha"))?;
            let in_proj = take_mat(weights, &format!("{p}self_attn.in_proj_weight"))?;
            let out_proj = take_mat(weights, &format!("{p}self_attn.out_proj.weight"))?;
            let gate_in = take_mat(weights, &format!("{p}gating.linear_in.weight"))?;
            let gate_out = take_mat(weights, &format!("{p}gating.linear_out.weight"))?;
            let w = LayerWeights {
                norm1_alpha,
                norm2_alpha,
                attn: AttnWeights {
                    in_proj,
                    out_proj,
                    num_heads: cfg.num_heads,
                    head_dim,
                },
                gate_in,
                gate_out,
            };
            layers.push(Layer {
                w,
                attn: AttnState {
                    kv: KvCache::new(cfg.num_heads, head_dim, cfg.context.max(1)),
                },
            });
        }
        Ok(Self {
            layers,
            cfg: cfg.clone(),
            seq_len: 0,
        })
    }

    pub fn build_prefixed(
        cfg: &TransformerConfig,
        prefix: &str,
        weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        let mut remapped = HashMap::new();
        for (k, v) in weights {
            if let Some(rest) = k.strip_prefix(prefix) {
                remapped.insert(rest.to_string(), v.clone());
            }
        }
        Self::build(cfg, &remapped)
    }

    pub fn reset_state(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn copy_state_from(&mut self, other: &Self) -> Result<()> {
        ensure!(
            self.layers.len() == other.layers.len(),
            "layer count mismatch"
        );
        for (a, b) in self.layers.iter_mut().zip(other.layers.iter()) {
            a.copy_state_from(b);
        }
        self.seq_len = other.seq_len;
        Ok(())
    }

    pub(crate) fn clone_kv_snapshot(&self) -> KvSnapshot {
        KvSnapshot {
            layers: self.layers.iter().map(|l| l.attn.kv.snapshot()).collect(),
            seq_len: self.seq_len,
        }
    }

    pub(crate) fn restore_kv_snapshot(&mut self, snap: &KvSnapshot) {
        for (layer, ls) in self.layers.iter_mut().zip(snap.layers.iter()) {
            layer.attn.kv.restore(ls);
        }
        self.seq_len = snap.seq_len;
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let t = x.dim().0;
        let positions: Vec<usize> = (self.seq_len..self.seq_len + t).collect();
        let (rope_cos, rope_sin) = match self.cfg.positional_embedding {
            PositionalEmbedding::Rope => {
                let head_dim = self.cfg.d_model / self.cfg.num_heads;
                let (cos, sin) = rope_tables(head_dim, self.cfg.max_period, &positions);
                (Some(cos), Some(sin))
            }
            PositionalEmbedding::None | PositionalEmbedding::Sin => (None, None),
        };
        let mut h = x.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, rope_cos.as_ref(), rope_sin.as_ref(), self.cfg.causal);
        }
        self.seq_len += t;
        h
    }
}

fn take_mat(weights: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    let rows = shape[0];
    let cols = shape[1];
    Ok(Array2::from_shape_vec((rows, cols), data.clone())?)
}

fn take_vec1(weights: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    let n: usize = shape.iter().product();
    ensure!(n == data.len(), "{key}: shape/product mismatch");
    Ok(Array1::from_vec(data.clone()))
}
