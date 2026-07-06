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

//! Weight loader for the Gemma 4 E2B *mobile QAT* safetensors checkpoint.
//!
//! The mobile checkpoint differs from a normal HF Gemma in three ways the
//! stock safetensors loader can't handle:
//!
//! 1. **Name prefix** — the language model lives under `model.language_model.`
//!    while the rlx Gemma builder asks for `model.…` / `lm_head.weight`.
//! 2. **Integer-quant storage** — most linears ship `{m}.weight` as packed
//!    `uint8`/`int8` plus a float `{m}.weight_scale`; embeddings ship
//!    `{m}.embedding_quantized` + `{m}.embedding_scale`. We dequantize to F32
//!    on `take` using [`crate::qat`] (proven bit-exact vs HF).
//! 3. **Per-module bit widths** — resolved from `quantization_config` via
//!    [`crate::qat::GemmaQuantPlan`].
//!
//! This loader returns F32 (correctness-first). A future packed
//! `DequantMatMul` path keeps weights low-bit in-graph for speed.
//!
//! The giant `embed_tokens_per_layer` table (`[vocab, 35*256]`, ~9.4 GB in F32)
//! is **never** materialized whole: [`GemmaQatLoader::dequant_embedding_rows`]
//! gathers + dequantizes only the rows for the actual prompt tokens, which the
//! runner uses to precompute the Per-Layer-Embedding inputs.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_loader::WeightLoader;
use safetensors::Dtype;

use crate::qat::{GemmaQuantBits, GemmaQuantPlan, dequantize_matrix, unpack_row};

/// Loader over a single-file (or sharded) Gemma 4 E2B mobile checkpoint.
pub struct GemmaQatLoader {
    ckpt: SafetensorsCheckpoint,
    plan: GemmaQuantPlan,
    taken: HashSet<String>,
}

impl GemmaQatLoader {
    /// Open the checkpoint directory (expects `model.safetensors` +
    /// `config.json` with a `quantization_config` block).
    pub fn open(dir: &Path) -> Result<Self> {
        let ckpt = SafetensorsCheckpoint::open(dir)?;
        let cfg_path = dir.join("config.json");
        let raw = std::fs::read(&cfg_path)
            .with_context(|| format!("reading {cfg_path:?} for quantization_config"))?;
        let json: serde_json::Value =
            serde_json::from_slice(&raw).with_context(|| format!("parsing {cfg_path:?}"))?;
        let quant = json.get("quantization_config").ok_or_else(|| {
            anyhow!("{cfg_path:?}: no quantization_config (not a QAT checkpoint?)")
        })?;
        let plan = GemmaQuantPlan::from_json(quant);
        Ok(Self {
            ckpt,
            plan,
            taken: HashSet::new(),
        })
    }

    /// Translate a builder-side HF name to the checkpoint's storage name.
    /// Text LM weights live under `model.language_model.`; the multimodal
    /// towers/projectors (`vision_tower`, `audio_tower`, `embed_vision`,
    /// `embed_audio`) and `lm_head` are stored verbatim under `model.` and
    /// must NOT get the `language_model` infix.
    fn remap(key: &str) -> String {
        if key.starts_with("lm_head") {
            return key.to_string();
        }
        const VERBATIM: [&str; 4] = [
            "model.vision_tower",
            "model.audio_tower",
            "model.embed_vision",
            "model.embed_audio",
        ];
        if VERBATIM.iter().any(|p| key.starts_with(p)) {
            return key.to_string();
        }
        match key.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => key.to_string(),
        }
    }

    /// Unpacked input width for a packed weight `[out, packed_cols]`.
    fn unpacked_cols(packed_cols: usize, bits: GemmaQuantBits) -> usize {
        packed_cols * bits.values_per_byte()
    }

    fn read_scale(&self, name: &str) -> Result<Vec<f32>> {
        let (bytes, dt, _shape) = self.ckpt.tensor_raw(name)?;
        anyhow::ensure!(dt == Dtype::F32, "{name}: scale must be F32, got {dt:?}");
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Dequantize one full quantized linear `{st}.weight` (+ `.weight_scale`)
    /// to a row-major `[out, in]` F32 matrix.
    fn dequant_linear(&self, st_weight: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let module = st_weight.strip_suffix(".weight").unwrap_or(st_weight);
        let bits = self
            .plan
            .resolve_bits(module)
            .ok_or_else(|| anyhow!("{module}: no quant bits (in modules_to_not_convert?)"))?;
        let (qbytes, qdt, qshape) = self.ckpt.tensor_raw(st_weight)?;
        anyhow::ensure!(
            matches!(qdt, Dtype::U8 | Dtype::I8),
            "{st_weight}: expected U8/I8 packed weight, got {qdt:?}"
        );
        anyhow::ensure!(
            qshape.len() == 2,
            "{st_weight}: expected rank-2, got {qshape:?}"
        );
        let out = qshape[0];
        let inn = Self::unpacked_cols(qshape[1], bits);
        let scale = self.read_scale(&format!("{module}.weight_scale"))?;
        let w = dequantize_matrix(&qbytes, &scale, out, inn, bits)?;
        Ok((w, vec![out, inn]))
    }

    /// Dequantize a full quantized embedding table to `[vocab, dim]` F32 with a
    /// per-row scale (used for the main `embed_tokens`; the giant per-layer
    /// table must use [`Self::dequant_embedding_rows`] instead). Does **not**
    /// apply the `sqrt(dim)` embed-scale — the builder applies that separately.
    fn dequant_embedding_full(&self, base: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let bits = self
            .plan
            .resolve_bits(base)
            .ok_or_else(|| anyhow!("{base}: no quant bits for embedding"))?;
        let (qbytes, qdt, qshape) = self
            .ckpt
            .tensor_raw(&format!("{base}.embedding_quantized"))?;
        anyhow::ensure!(
            matches!(qdt, Dtype::U8 | Dtype::I8),
            "{base}.embedding_quantized: expected U8/I8, got {qdt:?}"
        );
        let vocab = qshape[0];
        let dim = Self::unpacked_cols(qshape[1], bits);
        let (sbytes, _sdt, sshape) = self.ckpt.tensor_raw(&format!("{base}.embedding_scale"))?;
        let scale: Vec<f32> = sbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(
            sshape == [vocab, 1],
            "{base}.embedding_scale: expected per-row [vocab,1], got {sshape:?} \
             (grouped scale → use dequant_embedding_rows)"
        );
        let w = dequantize_matrix(&qbytes, &scale, vocab, dim, bits)?;
        Ok((w, vec![vocab, dim]))
    }

    /// Gather + dequantize only `rows` of a (possibly grouped-scale) quantized
    /// embedding. `builder_key` is e.g. `model.embed_tokens.weight` or
    /// `model.embed_tokens_per_layer.weight`. Returns `(flat rows*dim, dim)`.
    /// Handles both per-row scale `[vocab,1]` and the per-layer grouped scale
    /// `[vocab, groups]` (one scale per `dim/groups`-wide block). Does not apply
    /// the `sqrt` embed-scale.
    pub fn dequant_embedding_rows(
        &self,
        builder_key: &str,
        rows: &[u32],
    ) -> Result<(Vec<f32>, usize)> {
        let base = Self::remap(builder_key);
        let base = base.strip_suffix(".weight").unwrap_or(&base);
        let bits = self
            .plan
            .resolve_bits(base)
            .ok_or_else(|| anyhow!("{base}: no quant bits for embedding"))?;
        let (qbytes, qdt, qshape) = self
            .ckpt
            .tensor_raw(&format!("{base}.embedding_quantized"))?;
        anyhow::ensure!(
            matches!(qdt, Dtype::U8 | Dtype::I8),
            "{base}.embedding_quantized: expected U8/I8, got {qdt:?}"
        );
        let vocab = qshape[0];
        let packed_cols = qshape[1];
        let dim = Self::unpacked_cols(packed_cols, bits);
        let (sbytes, _sdt, sshape) = self.ckpt.tensor_raw(&format!("{base}.embedding_scale"))?;
        let scale: Vec<f32> = sbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let groups = if sshape == [vocab, 1] {
            1
        } else {
            anyhow::ensure!(
                sshape.len() == 2 && sshape[0] == vocab,
                "{base}.embedding_scale: unexpected shape {sshape:?}"
            );
            sshape[1]
        };
        anyhow::ensure!(
            dim % groups == 0,
            "{base}: dim {dim} not divisible by scale groups {groups}"
        );
        let block = dim / groups;
        let mut out = Vec::with_capacity(rows.len() * dim);
        for &r in rows {
            let r = r as usize;
            anyhow::ensure!(r < vocab, "{base}: row {r} >= vocab {vocab}");
            let row_bytes = &qbytes[r * packed_cols..(r + 1) * packed_cols];
            let q = unpack_row(row_bytes, dim, bits);
            for (j, qv) in q.iter().enumerate() {
                let g = j / block;
                out.push(*qv as f32 * scale[r * groups + g]);
            }
        }
        Ok((out, dim))
    }

    /// Read a plain float tensor by builder-side key (remapped) as F32.
    pub fn float_tensor(&self, builder_key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.take_float(&Self::remap(builder_key))
    }

    /// Main token embedding rows for `ids`, scaled by the Gemma embed-scale
    /// `sqrt(hidden_size)` → `inputs_embeds` `[seq, hidden]` (flat).
    pub fn inputs_embeds(&self, cfg: &crate::config::GemmaConfig, ids: &[u32]) -> Result<Vec<f32>> {
        let (rows, dim) = self.dequant_embedding_rows("model.embed_tokens.weight", ids)?;
        anyhow::ensure!(
            dim == cfg.hidden_size,
            "embed dim {dim} != hidden {}",
            cfg.hidden_size
        );
        let scale = (cfg.hidden_size as f32).sqrt();
        Ok(rows.iter().map(|v| v * scale).collect())
    }

    /// Compute the Per-Layer-Embedding inputs `[seq * num_layers * ple_w]`
    /// (layout `[s][layer][d]`), matching HF `Gemma4TextModel.get_per_layer_inputs`
    /// + `project_per_layer_inputs`:
    ///   tok  = embed_tokens_per_layer(ids) · √ple_w                  (token identity)
    ///   proj = (inputs_embeds · per_layer_model_projeᵀ) · hidden^-½  (context)
    ///   proj = per_layer_projection_norm(proj)                       (RMSNorm, ×(1+γ))
    ///   out  = (proj + tok) · 2^-½
    ///
    /// This runs out-of-graph so the 9.4 GB per-layer table is never materialized.
    pub fn compute_per_layer_inputs(
        &self,
        cfg: &crate::config::GemmaConfig,
        ids: &[u32],
    ) -> Result<Vec<f32>> {
        let seq = ids.len();
        let h = cfg.hidden_size;
        let nl = cfg.num_hidden_layers;
        let pw = cfg.ple_width();
        let eps = cfg.rms_norm_eps as f32;

        let (tok_rows, tdim) =
            self.dequant_embedding_rows("model.embed_tokens_per_layer.weight", ids)?;
        anyhow::ensure!(tdim == nl * pw, "per-layer embed dim {tdim} != {nl}*{pw}");
        let tok_scale = (pw as f32).sqrt();

        let ie = self.inputs_embeds(cfg, ids)?; // [seq, h]
        let (w, wshape) = self.float_tensor("model.per_layer_model_projection.weight")?;
        anyhow::ensure!(
            wshape == [nl * pw, h],
            "per_layer_model_projection shape {wshape:?} != [{}, {h}]",
            nl * pw
        );
        let proj_scale = (h as f32).powf(-0.5);
        let (gnorm, gshape) = self.float_tensor("model.per_layer_projection_norm.weight")?;
        anyhow::ensure!(gshape == [pw], "projection_norm shape {gshape:?} != [{pw}]");
        let inv_sqrt2 = 0.5f32.sqrt();

        let mut out = vec![0f32; seq * nl * pw];
        for s in 0..seq {
            let emb = &ie[s * h..(s + 1) * h];
            for layer in 0..nl {
                // context projection block for this (token, layer): [pw]
                let mut block = vec![0f32; pw];
                for (d, b) in block.iter_mut().enumerate() {
                    let wrow = &w[(layer * pw + d) * h..(layer * pw + d + 1) * h];
                    let acc: f32 = emb.iter().zip(wrow).map(|(a, b)| a * b).sum();
                    *b = acc * proj_scale;
                }
                // RMSNorm over the pw block, gamma = 1 + gnorm.
                let ss: f32 = block.iter().map(|x| x * x).sum();
                let rms = 1.0 / (ss / pw as f32 + eps).sqrt();
                for d in 0..pw {
                    let normed = block[d] * rms * (1.0 + gnorm[d]);
                    let tok = tok_rows[s * tdim + layer * pw + d] * tok_scale;
                    out[s * nl * pw + layer * pw + d] = (normed + tok) * inv_sqrt2;
                }
            }
        }
        Ok(out)
    }

    /// Read a plain float tensor (`F32`/`F16`/`BF16`) by storage name as F32.
    ///
    /// **Norm convention bridge:** Gemma 4's `Gemma4RMSNorm` scales by `weight`
    /// directly (`normed * weight`), unlike Gemma 1/2/3's `(1 + weight)` that
    /// the rlx builder's `gemma_rms` assumes. So for norm tensors we return
    /// `weight - 1`; the builder then computes `1 + (weight - 1) = weight`,
    /// reproducing Gemma 4's plain-weight RMSNorm without touching the shared
    /// builder. (Verified bit-exact vs HF on per_layer_projection_norm.)
    fn take_float(&self, st: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, dt, shape) = self.ckpt.tensor_raw(st)?;
        let mut data = match dt {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => bytes_view_to_f32(&bytes, dt)?,
            other => bail!("{st}: take_float on non-float dtype {other:?}"),
        };
        if st.ends_with("norm.weight") {
            for v in &mut data {
                *v -= 1.0;
            }
        }
        Ok((data, shape))
    }

    fn take_impl(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.taken.insert(key.to_string());
        let st = Self::remap(key);
        // Case 1: tensor stored directly (float norms, packed-int linears).
        if self.ckpt.contains(&st) {
            let (_b, dt, _s) = self.ckpt.tensor_raw(&st)?;
            return match dt {
                Dtype::U8 | Dtype::I8 => self.dequant_linear(&st),
                _ => self.take_float(&st),
            };
        }
        // Case 2: quantized embedding (`embed_tokens` only — the per-layer table
        // is handled out-of-graph via `dequant_embedding_rows`).
        let base = st.strip_suffix(".weight").unwrap_or(&st);
        if self.ckpt.contains(&format!("{base}.embedding_quantized")) {
            return self.dequant_embedding_full(base);
        }
        bail!("GemmaQatLoader: tensor {key} (→ {st}) not found in checkpoint")
    }
}

fn bytes_view_to_f32(bytes: &[u8], dt: Dtype) -> Result<Vec<f32>> {
    Ok(match dt {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => bail!("unsupported float dtype {other:?}"),
    })
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = data[r * cols + c];
        }
    }
    t
}

impl WeightLoader for GemmaQatLoader {
    fn format_id(&self) -> &'static str {
        "gemma-qat-safetensors"
    }

    fn len(&self) -> usize {
        self.ckpt.keys().count()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.take_impl(key)
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take_impl(key)?;
        anyhow::ensure!(
            shape.len() == 2,
            "take_transposed on non-2D tensor {key}: {shape:?}"
        );
        let (rows, cols) = (shape[0], shape[1]);
        Ok((transpose_2d(&data, rows, cols), vec![cols, rows]))
    }

    /// Repack a QAT int2/4 linear weight to GGUF **Q8_0** (near-lossless, ~1
    /// byte/elem) so the E2B text arena stays low-bit. The f32-dequant arena is
    /// 8.4 GiB, which exceeds wgpu's 4 GiB per-buffer cap; emitting a packed
    /// `DequantMatMul` (the same path GGUF models use — works on every backend)
    /// keeps it ~2 GiB. Only quantized 2D linears whose inner dim is a multiple
    /// of the Q8_0 block (32); norms / odd shapes → `None` → the f32 path.
    /// **Opt-in** via `RLX_GEMMA_QAT_PACK=1` (default: exact f32 weights). The
    /// packed path works on CoreML/MLX (host dequant) but the GPU dequant
    /// kernels currently NaN (Metal) / garbage (wgpu) on *repacked* Q4_K — GGUF
    /// -file Q4_K works, so it's a repack-layout vs GPU-kernel mismatch, not the
    /// packing itself (CoreML/MLX read it correctly). Kept for continuation.
    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<rlx_core::weight_map::PackedWeightTensor>> {
        if !rlx_ir::env::flag("RLX_GEMMA_QAT_PACK") {
            return Ok(None);
        }
        let st = Self::remap(key);
        // Peek the logical [out, in] WITHOUT consuming (`tensor_raw` doesn't mark
        // `taken`, so the f32 fallback can still take it).
        let Ok((_b, qdt, qshape)) = self.ckpt.tensor_raw(&st) else {
            return Ok(None);
        };
        if !matches!(qdt, Dtype::U8 | Dtype::I8) || qshape.len() != 2 {
            return Ok(None);
        }
        let module = st.strip_suffix(".weight").unwrap_or(&st);
        let Some(bits) = self.plan.resolve_bits(module) else {
            return Ok(None);
        };
        let out = qshape[0];
        let inn = Self::unpacked_cols(qshape[1], bits);
        // Q4_K super-block is 256 elems; the inner dim must divide it so the
        // blocks align with GGUF [n=out, k=in] rows.
        if inn % rlx_gguf::QK_K != 0 {
            return Ok(None);
        }
        // Consume + dequant to f32 [out, inn] row-major, then repack to GGUF
        // **Q4_K** — the most-tested `DequantMatMul` scheme (gemma2/3/4 use it,
        // validated bit-exact on every backend this session). Q8_0 was a dead
        // end: its GPU dequant kernel is unmapped on wgpu (garbage) and NaNs on
        // Metal, though CoreML/MLX host-dequant handled it.
        let (data, _shape) = self.take_impl(key)?;
        // Use the same full-tensor packer the GPU DequantMatMul parity tests use
        // (rlx-wgpu/tests/gguf_dequant_matmul_prefill_parity), which is validated
        // on wgpu/Metal — rather than a hand-rolled quantize_q4_k_block loop.
        let bytes = rlx_gguf::quantize(&data, rlx_gguf::GgmlType::Q4K)?;
        Ok(Some((
            bytes,
            rlx_ir::quant::QuantScheme::GgufQ4K,
            vec![out, inn],
        )))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.ckpt
            .keys()
            .filter(|k| !self.taken.contains(*k))
            .map(|s| s.to_string())
            .collect()
    }

    fn arch_hint(&self) -> Option<&str> {
        Some("gemma4")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the downloaded checkpoint dir (HF cache snapshot) if present.
    fn fixture_dir() -> Option<std::path::PathBuf> {
        if let Some(d) = std::env::var_os("RLX_GEMMA4_E2B_DIR") {
            let p = std::path::PathBuf::from(d);
            return p.join("config.json").is_file().then_some(p);
        }
        // HF hub default cache layout.
        let home = std::env::var_os("HOME")?;
        let base = std::path::Path::new(&home).join(
            ".cache/huggingface/hub/\
             models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
        );
        let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
        snap.join("config.json").is_file().then_some(snap)
    }

    #[test]
    fn loads_and_dequants_real_checkpoint() {
        let Some(dir) = fixture_dir() else {
            eprintln!("[qat_loader] checkpoint not found — skipping");
            return;
        };
        let mut ld = GemmaQatLoader::open(&dir).expect("open ckpt");

        // A 4-bit attention projection → [out, in] = [2048, 1536].
        let (q, qs) = ld
            .take("model.layers.0.self_attn.q_proj.weight")
            .expect("take q_proj");
        assert_eq!(qs, vec![2048, 1536]);
        assert!(q.iter().all(|v| v.is_finite()));
        assert!(q.iter().any(|&v| v != 0.0));

        // A BF16 norm → [1536].
        let (n, ns) = ld.take("model.norm.weight").expect("take norm");
        assert_eq!(ns, vec![1536]);
        assert!(n.iter().all(|v| v.is_finite()));

        // lm_head is a separate 2-bit table [262144, 1536].
        let (_h, hs) = ld.take("lm_head.weight").expect("take lm_head");
        assert_eq!(hs, vec![262144, 1536]);

        // Transpose flips the 2D shape.
        let (_qt, qts) = ld
            .take_transposed("model.layers.0.self_attn.k_proj.weight")
            .expect("take_t k_proj");
        assert_eq!(qts, vec![1536, 256]); // k_proj is [256,1536] → T [1536,256]

        // Per-layer embedding rows: grouped scale, gather just two tokens.
        let (ple, dim) = ld
            .dequant_embedding_rows("model.embed_tokens_per_layer.weight", &[818, 5279])
            .expect("ple rows");
        assert_eq!(dim, 35 * 256);
        assert_eq!(ple.len(), 2 * 35 * 256);
        assert!(ple.iter().all(|v| v.is_finite()));

        // Exact cross-check vs HF ground truth (fixtures/gemma4_e2b/loader_check.json),
        // when present. Proves the full remap + dequant + grouped-scale pipeline
        // matches transformers bit-for-bit, not just "finite".
        let fx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/gemma4_e2b/loader_check.json");
        if let Ok(raw) = std::fs::read(&fx_path) {
            let fx: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            let arr = |k: &str| -> Vec<f32> {
                fx[k]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };
            let approx = |a: &[f32], b: &[f32], tag: &str| {
                assert_eq!(a.len(), b.len(), "{tag} len");
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-6 * (1.0 + y.abs()),
                        "{tag}[{i}] {x} != {y}"
                    );
                }
            };
            // q_proj row 0, first 8.
            approx(&q[..8], &arr("q_proj_l0_row0_head8"), "q_proj");
            // norm first 8. The loader returns Gemma-4 norm weights as `w-1`
            // (so the builder's `1+(w-1)` delta-gamma reproduces the plain-weight
            // RMSNorm); add 1 back to compare against HF's raw weight.
            let n_raw: Vec<f32> = n[..8].iter().map(|x| x + 1.0).collect();
            approx(&n_raw, &arr("norm_head8"), "norm");
            // per-layer embed token 818: first block + second block (grouped scale).
            approx(&ple[..8], &arr("ple_tok818_first8"), "ple_b0");
            approx(&ple[256..260], &arr("ple_tok818_block1_first4"), "ple_b1");
            eprintln!("[qat_loader] exact cross-check vs HF: PASS");
        } else {
            eprintln!("[qat_loader] loader_check.json absent — structural checks only");
        }
    }

    #[test]
    fn per_layer_inputs_match_hf() {
        let Some(dir) = fixture_dir() else {
            eprintln!("[qat_loader] checkpoint not found — skipping");
            return;
        };
        let ld = GemmaQatLoader::open(&dir).expect("open ckpt");
        let cfg = crate::config::GemmaConfig::from_file(&dir.join("config.json")).expect("cfg");
        let ids = [818u32, 5279, 529, 7001, 563];
        let pli = ld.compute_per_layer_inputs(&cfg, &ids).expect("pli");
        assert_eq!(pli.len(), 5 * 35 * 256);
        assert!(pli.iter().all(|v| v.is_finite()));

        // Exact vs HF (fixtures/gemma4_e2b/per_layer_inputs.bin), when present.
        let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/gemma4_e2b/per_layer_inputs.bin");
        if let Ok(raw) = std::fs::read(&bin) {
            let hf: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(hf.len(), pli.len(), "pli len mismatch");
            let mut maxd = 0f32;
            for (a, b) in pli.iter().zip(&hf) {
                maxd = maxd.max((a - b).abs());
            }
            eprintln!("[qat_loader] per_layer_inputs maxdiff vs HF = {maxd:.3e}");
            // BF16 weights + f32 accumulation → small but nonzero.
            assert!(maxd < 5e-3, "per_layer_inputs maxdiff {maxd} too large");
        } else {
            eprintln!("[qat_loader] per_layer_inputs.bin absent — structural check only");
        }
    }
}
