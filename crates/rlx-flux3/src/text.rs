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

//! The Qwen3-VL-4B text tower as FLUX 3's text encoder.
//!
//! BFL `flux3/text_encoder.py`: the prompt goes through the chat template
//! as one user turn (generation prompt added), is right-padded with
//! `<|endoftext|>` to a multiple of 80 tokens, and the hidden states after
//! layers 4, 8, …, 32 are stacked per token into a `8 · 2560 = 20480`-wide
//! text token. Only the first 32 of the 36 layers ever run.
//!
//! The padding tokens are fed to the DiT, so their states must be exact.
//! Under the reference's causal + padding mask a pad query sees every real
//! key and no pad key — not even itself — and every pad shares one token
//! id and one position id (1). All pad rows are therefore identical at
//! every layer, and one extra query row attending to the real keys, with
//! no mask, reproduces them. Text-only M-RoPE (all three sections at the
//! same position) is the ordinary 1-D RoPE.

use anyhow::{Context, Result, ensure};
use rlx_flow::{BuiltModel, Emit, ModelFlow, WeightStorage};
use rlx_flux1::weights::{TensorFile, WeightQuant};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::path::Path;

use crate::config::TextEncoderConfig;
use crate::embed::round_bf16;
use crate::weights::PrefixSource;

/// Name prefix of the text tower inside a `Qwen3VLForConditionalGeneration`.
pub const QWEN3_VL_PREFIX: &str = "model.language_model.";

/// One user turn with the generation prompt, as the Qwen3-VL chat template
/// renders a plain string message.
pub fn chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// `(cos, sin)` `[positions.len(), head_dim / 2]`, half-split pairs.
pub fn rope_tables(positions: &[usize], head_dim: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let half = head_dim / 2;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &w in &inv {
            let a = p as f64 * w;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// Layers `layers` over `real` real rows plus, when `pad`, one pad row.
pub fn build_text_chunk(
    cfg: &TextEncoderConfig,
    real: usize,
    pad: bool,
    layers: std::ops::Range<usize>,
    weights: &mut PrefixSource<'_>,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let n = real + usize::from(pad);
    let d = cfg.hidden_size;
    let cfg_c = cfg.clone();
    ModelFlow::new("flux3_text_chunk")
        .with_weight_storage(WeightStorage::Auto)
        .input("x", Shape::new(&[1, n, d], f))
        .plugin_named("flux3.text_layers", move |emit, _| {
            let mut x = emit.flow_input("x")?.hir_id();
            let mut positions: Vec<usize> = (0..real).collect();
            if pad {
                positions.push(1);
            }
            let (cos, sin) = rope_tables(&positions, cfg_c.head_dim, cfg_c.rope_theta);
            let half = Shape::new(&[n, cfg_c.head_dim / 2], f);
            let f32s = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
            let cos = emit.synth_param("te.rope_cos", f32s(cos), half.clone());
            let sin = emit.synth_param("te.rope_sin", f32s(sin), half);
            let zero_d = emit.synth_param("te.zero_d", vec![0.0; d], Shape::new(&[d], f));
            let zero_h = emit.synth_param(
                "te.zero_h",
                vec![0.0; cfg_c.head_dim],
                Shape::new(&[cfg_c.head_dim], f),
            );
            let mut l = TextLayer {
                emit,
                cfg: &cfg_c,
                real,
                pad,
                rope: (cos, sin),
                zero_d,
                zero_h,
            };
            for li in layers.clone() {
                x = l.layer(li, x)?;
            }
            Ok(Some(l.emit.wrap(x, Shape::new(&[1, n, d], f))))
        })
        .output("hidden")
        .build(weights)
}

struct TextLayer<'e, 'a> {
    emit: &'e mut Emit<'a>,
    cfg: &'e TextEncoderConfig,
    real: usize,
    pad: bool,
    rope: (HirNodeId, HirNodeId),
    zero_d: HirNodeId,
    zero_h: HirNodeId,
}

impl TextLayer<'_, '_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.emit.hir())
    }

    fn linear(&mut self, x: HirNodeId, key: &str) -> Result<HirNodeId> {
        if let Some(p) = self.emit.weights.take_packed(key)? {
            let qkey = format!("{key}\0q");
            let id = self
                .emit
                .hir()
                .param(&qkey, Shape::new(&[p.w_q.len()], DType::U8));
            self.emit.state.typed_params.push((qkey, p.w_q, DType::U8));
            let s = self.g().shape(x).clone();
            let last = s.rank() - 1;
            let shape = s.with_dim(last, rlx_ir::Dim::Static(p.out_dim));
            return Ok(self.emit.hir().dequant_matmul(
                x,
                id,
                None,
                None,
                p.scheme,
                shape.with_dtype(DType::F32),
            ));
        }
        let w = self.emit.load_matrix(key, true)?;
        Ok(self.g().mm(x, w))
    }

    fn norm(&mut self, x: HirNodeId, key: &str) -> Result<HirNodeId> {
        let w = self.emit.load_param(key, false)?;
        let (zero, eps) = (self.zero_d, self.cfg.rms_norm_eps as f32);
        Ok(self.g().rms_norm(x, w, zero, eps))
    }

    fn head_norm(&mut self, x: HirNodeId, heads: usize, key: &str) -> Result<HirNodeId> {
        let w = self.emit.load_param(key, false)?;
        let (zero, eps, hd) = (self.zero_h, self.cfg.rms_norm_eps as f32, self.cfg.head_dim);
        let n = (self.real + usize::from(self.pad)) as i64;
        let mut g = self.g();
        let x4 = g.reshape_(x, vec![1, n, heads as i64, hd as i64]);
        let y = g.rms_norm(x4, w, zero, eps);
        Ok(g.reshape_(y, vec![1, n, (heads * hd) as i64]))
    }

    /// `[1, n, nkv·hd]` → `[1, n, nh·hd]`, each kv head repeated for its group.
    fn repeat_kv(&mut self, x: HirNodeId) -> HirNodeId {
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let group = nh / nkv;
        if group == 1 {
            return x;
        }
        let mut g = self.g();
        let mut pieces = Vec::with_capacity(nh);
        for h in 0..nkv {
            let s = g.narrow_(x, 2, h * hd, hd);
            pieces.extend(std::iter::repeat_n(s, group));
        }
        g.concat_(pieces, 2)
    }

    fn layer(&mut self, li: usize, x: HirNodeId) -> Result<HirNodeId> {
        let p = format!("layers.{li}");
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let (real, pad) = (self.real, self.pad);
        let h = self.norm(x, &format!("{p}.input_layernorm.weight"))?;
        let q = self.linear(h, &format!("{p}.self_attn.q_proj.weight"))?;
        let k = self.linear(h, &format!("{p}.self_attn.k_proj.weight"))?;
        let v = self.linear(h, &format!("{p}.self_attn.v_proj.weight"))?;
        let q = self.head_norm(q, nh, &format!("{p}.self_attn.q_norm.weight"))?;
        let k = self.head_norm(k, nkv, &format!("{p}.self_attn.k_norm.weight"))?;
        let (cos, sin) = self.rope;
        let (q, k) = {
            let mut g = self.g();
            (g.rope(q, cos, sin, hd), g.rope(k, cos, sin, hd))
        };
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);
        let attn = {
            let width = nh * hd;
            let mut g = self.g();
            if pad {
                let qr = g.narrow_(q, 1, 0, real);
                let kr = g.narrow_(k, 1, 0, real);
                let vr = g.narrow_(v, 1, 0, real);
                let qp = g.narrow_(q, 1, real, 1);
                let ar = g.attention_kind(
                    qr,
                    kr,
                    vr,
                    nh,
                    hd,
                    MaskKind::Causal,
                    Shape::new(&[1, real, width], DType::F32),
                );
                let ap = g.attention_kind(
                    qp,
                    kr,
                    vr,
                    nh,
                    hd,
                    MaskKind::None,
                    Shape::new(&[1, 1, width], DType::F32),
                );
                g.concat_(vec![ar, ap], 1)
            } else {
                g.attention_kind(
                    q,
                    k,
                    v,
                    nh,
                    hd,
                    MaskKind::Causal,
                    Shape::new(&[1, real, width], DType::F32),
                )
            }
        };
        let o = self.linear(attn, &format!("{p}.self_attn.o_proj.weight"))?;
        let x = self.g().add(x, o);
        let h = self.norm(x, &format!("{p}.post_attention_layernorm.weight"))?;
        let gate = self.linear(h, &format!("{p}.mlp.gate_proj.weight"))?;
        let up = self.linear(h, &format!("{p}.mlp.up_proj.weight"))?;
        let act = {
            let mut g = self.g();
            let s = g.silu(gate);
            g.mul(s, up)
        };
        let down = self.linear(act, &format!("{p}.mlp.down_proj.weight"))?;
        Ok(self.g().add(x, down))
    }
}

/// Text tokens for the DiT.
#[derive(Debug, Clone)]
pub struct TextTokens {
    /// `[len, context_dim]`, bf16-rounded like the reference's `.to(bfloat16)`.
    pub tokens: Vec<f32>,
    pub len: usize,
    /// Real (non-padding) tokens.
    pub real: usize,
}

/// The text tower: tokenizer, weights, and the graphs compiled for the last
/// prompt length.
pub struct TextEncoder {
    pub cfg: TextEncoderConfig,
    file: TensorFile,
    tokenizer: tokenizers::Tokenizer,
    pad_id: u32,
    /// Hidden-state indices to stack (sorted, ≥ 1).
    layers: Vec<usize>,
    pad_multiple: usize,
    max_tokens: usize,
    device: Device,
    quant: Option<WeightQuant>,
    compiled: Option<((usize, bool), Vec<CompiledGraph>)>,
}

impl TextEncoder {
    /// Open the `text_encoder/` directory of a FLUX 3 package.
    pub fn open(
        dir: &Path,
        layers: Vec<usize>,
        pad_multiple: usize,
        max_tokens: usize,
        device: Device,
        quant: Option<WeightQuant>,
    ) -> Result<Self> {
        let cfg = TextEncoderConfig::load(dir)?;
        let file = TensorFile::open(dir)?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", tok_path.display()))?;
        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .context("tokenizer has no <|endoftext|>")?;
        ensure!(
            layers.windows(2).all(|w| w[0] < w[1]) && layers.first().is_some_and(|&l| l >= 1),
            "text layers {layers:?} must be increasing and ≥ 1"
        );
        ensure!(
            *layers.last().expect("non-empty") <= cfg.num_hidden_layers,
            "text layer {} > {} layers",
            layers.last().expect("non-empty"),
            cfg.num_hidden_layers
        );
        Ok(Self {
            cfg,
            file,
            tokenizer,
            pad_id,
            layers,
            pad_multiple: pad_multiple.max(1),
            max_tokens,
            device,
            quant,
            compiled: None,
        })
    }

    /// Width of one text token (`layers · hidden`).
    pub fn context_dim(&self) -> usize {
        self.layers.len() * self.cfg.hidden_size
    }

    /// Token ids of `prompt` in its chat turn, truncated to the limit.
    pub fn token_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(chat_prompt(prompt), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(self.max_tokens);
        Ok(ids)
    }

    /// Padded length for `real` tokens.
    pub fn padded_len(&self, real: usize) -> usize {
        real.div_ceil(self.pad_multiple)
            .saturating_mul(self.pad_multiple)
            .min(self.max_tokens)
            .max(real)
    }

    fn ensure_compiled(&mut self, real: usize, pad: bool) -> Result<()> {
        if self
            .compiled
            .as_ref()
            .is_some_and(|(k, _)| *k == (real, pad))
        {
            return Ok(());
        }
        self.compiled = None;
        let mut graphs = Vec::with_capacity(self.layers.len());
        let mut start = 0;
        for &end in &self.layers {
            let mut src = PrefixSource {
                file: &self.file,
                prefix: QWEN3_VL_PREFIX,
                quant: self.quant,
            };
            let built = build_text_chunk(&self.cfg, real, pad, start..end, &mut src)?;
            graphs.push(rlx_core::flow_util::compile_built(built, self.device)?);
            start = end;
        }
        self.compiled = Some(((real, pad), graphs));
        Ok(())
    }

    /// Encode `prompt` → `[len, context_dim]` with `len` a multiple of the
    /// pad width.
    pub fn encode(&mut self, prompt: &str) -> Result<TextTokens> {
        let ids = self.token_ids(prompt)?;
        self.encode_ids(&ids)
    }

    /// [`Self::encode`] from token ids.
    pub fn encode_ids(&mut self, ids: &[u32]) -> Result<TextTokens> {
        let real = ids.len();
        ensure!(real > 0, "empty token sequence");
        let len = self.padded_len(real);
        let pad = len > real;
        self.ensure_compiled(real, pad)?;
        let d = self.cfg.hidden_size;
        let mut rows: Vec<u32> = ids.to_vec();
        if pad {
            rows.push(self.pad_id);
        }
        let mut x = self
            .file
            .load_rows(&format!("{QWEN3_VL_PREFIX}embed_tokens.weight"), &rows)?;
        let n = rows.len();
        let k = self.layers.len();
        let width = k * d;
        let mut tokens = vec![0.0f32; len * width];
        let (_, graphs) = self.compiled.as_mut().expect("compiled above");
        for (j, graph) in graphs.iter_mut().enumerate() {
            let out = graph.run(&[("x", &x)]);
            x = out
                .into_iter()
                .next()
                .context("text chunk produced no output")?;
            ensure!(x.len() == n * d, "text chunk output {}", x.len());
            for t in 0..len {
                let src = t.min(n - 1);
                let dst = &mut tokens[t * width + j * d..t * width + (j + 1) * d];
                for (o, &v) in dst.iter_mut().zip(&x[src * d..(src + 1) * d]) {
                    *o = round_bf16(v);
                }
            }
        }
        Ok(TextTokens { tokens, len, real })
    }

    /// Drop the compiled graphs (device memory) but keep the tokenizer.
    pub fn release(&mut self) {
        self.compiled = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_turn_matches_the_template() {
        assert_eq!(
            chat_prompt("a fox"),
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn rope_is_half_split_angles() {
        let (cos, sin) = rope_tables(&[0, 3], 8, 10_000.0);
        assert_eq!(cos.len(), 8);
        assert!(sin[..4].iter().all(|&s| s == 0.0));
        assert!((sin[4] - 3f64.sin()).abs() < 1e-15);
        assert!((cos[5] - (3.0 / 10f64).cos()).abs() < 1e-15);
    }
}
