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

//! CLIP ViT-L/14 text tower (FLUX.1's `text_encoder`): only its pooled
//! output conditions the denoiser (through `vector_in`).
//!
//! HF `CLIPTextModel`: learned token + position embeddings (added on host),
//! pre-LN causal blocks with quick-GELU, `final_layer_norm`; the pooled vector
//! is the final hidden state at the first end-of-text token.

use anyhow::{Context, Result, ensure};
use rlx_flow::{BuiltModel, ModelFlow, WeightStorage};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::ClipTextConfig;
use crate::weights::{PlainSource, TensorFile};

pub(crate) const PREFIX: &str = "text_model.";

/// Build the CLIP text graph: input `hidden` `[1, ctx, width]`, output the
/// post-`final_layer_norm` sequence.
pub fn build_clip_text(
    cfg: &ClipTextConfig,
    weights: &mut dyn rlx_flow::WeightSource,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let (ctx, w, heads) = (
        cfg.max_position_embeddings,
        cfg.hidden_size,
        cfg.num_attention_heads,
    );
    let eps = cfg.layer_norm_eps as f32;
    let cfg_c = cfg.clone();
    let quick = cfg.hidden_act == "quick_gelu";
    ModelFlow::new("flux1_clip_text")
        .with_weight_storage(WeightStorage::Auto)
        .input("hidden", Shape::new(&[1, ctx, w], f))
        .plugin_named("clip.text", move |emit, _| {
            let mut x = emit.flow_input("hidden")?.hir_id();
            let inter = cfg_c.intermediate_size;
            let k1702 = emit.synth_param(
                "clip.quick_gelu_k",
                vec![1.702; inter],
                Shape::new(&[inter], f),
            );
            for i in 0..cfg_c.num_hidden_layers {
                let p = format!("encoder.layers.{i}");
                let mut lp = |n: &str, t: bool| -> Result<_> {
                    if t {
                        emit.load_matrix(&format!("{p}.{n}"), true)
                    } else {
                        emit.load_param(&format!("{p}.{n}"), false)
                    }
                };
                let (ln1g, ln1b) = (
                    lp("layer_norm1.weight", false)?,
                    lp("layer_norm1.bias", false)?,
                );
                let (qw, qb) = (
                    lp("self_attn.q_proj.weight", true)?,
                    lp("self_attn.q_proj.bias", false)?,
                );
                let (kw, kb) = (
                    lp("self_attn.k_proj.weight", true)?,
                    lp("self_attn.k_proj.bias", false)?,
                );
                let (vw, vb) = (
                    lp("self_attn.v_proj.weight", true)?,
                    lp("self_attn.v_proj.bias", false)?,
                );
                let (ow, ob) = (
                    lp("self_attn.out_proj.weight", true)?,
                    lp("self_attn.out_proj.bias", false)?,
                );
                let (ln2g, ln2b) = (
                    lp("layer_norm2.weight", false)?,
                    lp("layer_norm2.bias", false)?,
                );
                let (f1w, f1b) = (lp("mlp.fc1.weight", true)?, lp("mlp.fc1.bias", false)?);
                let (f2w, f2b) = (lp("mlp.fc2.weight", true)?, lp("mlp.fc2.bias", false)?);
                let mut g = HirMut::new(emit.hir());
                let h = g.ln(x, ln1g, ln1b, eps);
                let lin = |g: &mut HirMut<'_>, x, w, b| {
                    let y = g.mm(x, w);
                    g.add(y, b)
                };
                let q = lin(&mut g, h, qw, qb);
                let k = lin(&mut g, h, kw, kb);
                let v = lin(&mut g, h, vw, vb);
                let shape = g.shape(v).clone();
                let a = g.attention_kind(q, k, v, heads, w / heads, MaskKind::Causal, shape);
                let o = lin(&mut g, a, ow, ob);
                x = g.add(x, o);
                let h = g.ln(x, ln2g, ln2b, eps);
                let h = lin(&mut g, h, f1w, f1b);
                let h = if quick {
                    let s = g.mul(h, k1702);
                    let shape = g.shape(s).clone();
                    let sig = g.activation(Activation::Sigmoid, s, shape);
                    g.mul(h, sig)
                } else {
                    g.gelu(h)
                };
                let h = lin(&mut g, h, f2w, f2b);
                x = g.add(x, h);
            }
            let fg = emit.load_param("final_layer_norm.weight", false)?;
            let fb = emit.load_param("final_layer_norm.bias", false)?;
            let x = HirMut::new(emit.hir()).ln(x, fg, fb, eps);
            Ok(Some(emit.wrap(x, Shape::new(&[1, ctx, w], f))))
        })
        .output("hidden")
        .build(weights)
}

/// A compiled CLIP text tower.
pub struct ClipTextEncoder {
    pub cfg: ClipTextConfig,
    file: TensorFile,
    graph: CompiledGraph,
}

impl ClipTextEncoder {
    /// Load from an HF CLIP (text or full) checkpoint directory / file.
    pub fn compile(path: &std::path::Path, cfg: ClipTextConfig, device: Device) -> Result<Self> {
        let file = TensorFile::open(path)?;
        let mut src = PlainSource {
            file: &file,
            prefix: PREFIX,
            quant: None,
        };
        let built = build_clip_text(&cfg, &mut src)?;
        let graph = rlx_core::flow_util::compile_built(built, device)?;
        Ok(Self { cfg, file, graph })
    }

    /// Pooled embedding `[hidden_size]` for exactly `max_position_embeddings` ids.
    pub fn pooled(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let (ctx, w) = (self.cfg.max_position_embeddings, self.cfg.hidden_size);
        ensure!(
            ids.len() == ctx,
            "CLIP expects {ctx} ids, got {}",
            ids.len()
        );
        let mut hidden = self
            .file
            .load_rows(&format!("{PREFIX}embeddings.token_embedding.weight"), ids)?;
        let (pos, _) = self
            .file
            .load(&format!("{PREFIX}embeddings.position_embedding.weight"))?;
        hidden
            .iter_mut()
            .zip(&pos[..ctx * w])
            .for_each(|(h, p)| *h += p);
        let out = self.graph.run(&[("hidden", &hidden)]);
        let seq = out.into_iter().next().context("CLIP produced no output")?;
        let eos = ids
            .iter()
            .position(|&t| t == self.cfg.eos_token_id)
            .unwrap_or(ctx - 1);
        Ok(seq[eos * w..(eos + 1) * w].to_vec())
    }
}

/// CLIP ids for FLUX: `<|startoftext|> … <|endoftext|>`, truncated to `ctx`
/// (keeping the end token) and padded with `<|endoftext|>`.
pub fn clip_ids(tok: &tokenizers::Tokenizer, text: &str, ctx: usize, eos: u32) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("CLIP tokenize: {e}"))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.len() > ctx {
        ids.truncate(ctx - 1);
        ids.push(eos);
    }
    ids.resize(ctx, eos);
    Ok(ids)
}
