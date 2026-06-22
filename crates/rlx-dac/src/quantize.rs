use crate::layers::WnConv1d;
use crate::ops::weight_norm;
use crate::weights::WeightStore;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, ArrayView2};

#[derive(Debug, Clone)]
struct VectorQuantizer {
    in_proj: WnConv1d,
    out_proj: WnConv1d,
    codebook: Array2<f32>,
    /// Per-entry L2 norm of the codebook rows, clamped to 1e-12. Precomputed
    /// once so the nearest-code search below does not recompute `sqrt(Σ c²)` for
    /// all 1024 entries at every time step. Bit-identical to the inline form
    /// (same summation order, same `sqrt`, same `max`).
    cb_norm: Vec<f32>,
}

impl VectorQuantizer {
    fn new(in_proj: WnConv1d, out_proj: WnConv1d, codebook: Array2<f32>) -> Self {
        let (n, d) = codebook.dim();
        let mut cb_norm = vec![0f32; n];
        for i in 0..n {
            let mut cb_sq = 0.0f32;
            for j in 0..d {
                let c = codebook[[i, j]];
                cb_sq += c * c;
            }
            cb_norm[i] = cb_sq.sqrt().max(1e-12);
        }
        Self {
            in_proj,
            out_proj,
            codebook,
            cb_norm,
        }
    }

    fn decode_code(&self, indices: &[u32]) -> Array2<f32> {
        let d = self.codebook.dim().1;
        let mut out = Array2::<f32>::zeros((d, indices.len()));
        for (ti, &idx) in indices.iter().enumerate() {
            for j in 0..d {
                out[[j, ti]] = self.codebook[[idx as usize, j]];
            }
        }
        out
    }

    fn decode_latents(&self, latents: ArrayView2<f32>) -> (Array2<f32>, Vec<u32>) {
        let (d, t) = latents.dim();
        let n = self.codebook.dim().0;
        // Row-major `[n, d]` slice so the inner dot product walks contiguous
        // memory (autovectorizable) instead of `[[i, j]]` bounds-checked indexing.
        let cb = self
            .codebook
            .as_slice()
            .expect("codebook is standard layout");
        let mut indices = vec![0u32; t];
        let mut z_q = Array2::<f32>::zeros((d, t));

        let mut enc = vec![0f32; d];
        for ti in 0..t {
            let mut enc_sq = 0.0f32;
            for j in 0..d {
                enc[j] = latents[[j, ti]];
                enc_sq += enc[j] * enc[j];
            }
            enc_sq = enc_sq.sqrt().max(1e-12);
            for e in enc.iter_mut() {
                *e /= enc_sq;
            }

            let mut best_idx = 0usize;
            let mut best_dist = f32::INFINITY;
            for i in 0..n {
                let row = &cb[i * d..i * d + d];
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += enc[j] * row[j];
                }
                let dist = 2.0 - 2.0 * dot / self.cb_norm[i];
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            indices[ti] = best_idx as u32;
            let best = &cb[best_idx * d..best_idx * d + d];
            for j in 0..d {
                z_q[[j, ti]] = best[j];
            }
        }
        (z_q, indices)
    }

    fn forward(&self, residual: ArrayView2<f32>) -> (Array2<f32>, Vec<u32>, Array2<f32>) {
        let z_e = self.in_proj.forward(residual);
        let (z_q, indices) = self.decode_latents(z_e.view());
        let z_q = self.out_proj.forward(z_q.view());
        (z_q, indices, z_e)
    }

    // Decodes codes back to embeddings — `from_codes` reads naturally despite `&self`.
    #[allow(clippy::wrong_self_convention)]
    fn from_codes(&self, frame: &[u32]) -> Array2<f32> {
        let z_p = self.decode_code(frame);
        self.out_proj.forward(z_p.view())
    }
}

pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn encode(&self, z: ArrayView2<f32>, n_quantizers: usize) -> (Array2<f32>, Vec<Vec<u32>>) {
        let n = n_quantizers.min(self.quantizers.len());
        let mut residual = z.to_owned();
        let mut z_q_sum = Array2::<f32>::zeros(z.dim());
        let mut all_codes = Vec::with_capacity(n);

        for q in self.quantizers.iter().take(n) {
            let (z_q_i, indices, _z_e) = q.forward(residual.view());
            for c in 0..residual.dim().0 {
                for t in 0..residual.dim().1 {
                    residual[[c, t]] -= z_q_i[[c, t]];
                    z_q_sum[[c, t]] += z_q_i[[c, t]];
                }
            }
            all_codes.push(indices);
        }
        (z_q_sum, all_codes)
    }

    // Decodes codes back to embeddings — `from_codes` reads naturally despite `&self`.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &[Vec<u32>]) -> Array2<f32> {
        let t = codes.first().map(|c| c.len()).unwrap_or(0);
        let d = self
            .quantizers
            .first()
            .map(|q| q.out_proj.weight.dim().0)
            .unwrap_or(0);
        let mut z_q = Array2::<f32>::zeros((d, t));
        for (q, frame) in self.quantizers.iter().zip(codes.iter()) {
            let z_q_i = q.from_codes(frame);
            for c in 0..d {
                for ti in 0..t {
                    z_q[[c, ti]] += z_q_i[[c, ti]];
                }
            }
        }
        z_q
    }
}

fn load_wn_conv1d(store: &WeightStore, prefix: &str) -> Result<WnConv1d> {
    let (g_data, g_shape) = store.get(&format!("{prefix}.weight_g"))?;
    let (v_data, v_shape) = store.get(&format!("{prefix}.weight_v"))?;
    ensure!(
        g_shape.len() == 3 && v_shape.len() == 3 && g_shape[0] == v_shape[0],
        "{prefix} weight_g/v out-channel mismatch: g={g_shape:?} v={v_shape:?}"
    );
    let weight = weight_norm(
        ndarray::ArrayView3::from_shape((g_shape[0], g_shape[1], g_shape[2]), g_data)?,
        ndarray::ArrayView3::from_shape((v_shape[0], v_shape[1], v_shape[2]), v_data)?,
    );
    let bias = store
        .get_optional(&format!("{prefix}.bias"))?
        .map(|(d, s)| Array1::from_shape_vec(s[0], d).expect("bias shape"));
    Ok(WnConv1d {
        weight,
        bias,
        stride: 1,
        pad: 0,
        dilation: 1,
        same_length: true,
    })
}

pub fn build_quantizer(
    cfg: &crate::config::DacConfig,
    store: &WeightStore,
) -> Result<ResidualVectorQuantizer> {
    let mut quantizers = Vec::with_capacity(cfg.n_codebooks);
    for i in 0..cfg.n_codebooks {
        let prefix = format!("quantizer.quantizers.{i}");
        let (cb_data, cb_shape) = store
            .get(&format!("{prefix}.codebook.weight"))
            .with_context(|| format!("missing {prefix}.codebook.weight"))?;
        ensure!(
            cb_shape == vec![cfg.codebook_size, cfg.codebook_dim],
            "codebook shape {cb_shape:?}"
        );
        let codebook =
            Array2::from_shape_vec((cfg.codebook_size, cfg.codebook_dim), cb_data.to_vec())?;
        quantizers.push(VectorQuantizer::new(
            load_wn_conv1d(store, &format!("{prefix}.in_proj"))?,
            load_wn_conv1d(store, &format!("{prefix}.out_proj"))?,
            codebook,
        ));
    }
    Ok(ResidualVectorQuantizer { quantizers })
}
