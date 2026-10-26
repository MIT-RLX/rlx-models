// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Where layer weights come from when a model is handled **one layer at a time**
//! (streamed training, activation dumps, pipeline workers).
//!
//! [`WeightSource`] hands out one layer's backbone, a compact bank of just the
//! routed experts a batch picked, the embedding rows and the head. Two
//! implementations: [`CheckpointSource`] reads an HF checkpoint lazily (the real
//! 1.6 TB model or a [`crate::synth`] one), [`MemorySource`] serves a
//! [`FlowWeights`] already in RAM (tests, small configs).

use crate::config::{KimiK3Config, KimiLinearConfig};
use crate::flow::{AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights};
use crate::loader::CheckpointLoader;
use crate::moe::MoeWeights;
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

pub const EMBED: &str = "language_model.model.embed_tokens.weight";
pub const LM_HEAD: &str = "language_model.lm_head.weight";
pub const FINAL_NORM: &str = "language_model.model.norm.weight";
pub const OUT_RES_NORM: &str = "language_model.model.output_attn_res_norm.weight";
pub const OUT_RES_PROJ: &str = "language_model.model.output_attn_res_proj.weight";

/// Checkpoint prefix of layer `i`.
pub fn layer_prefix(i: usize) -> String {
    format!("language_model.model.layers.{i}")
}

/// Check the graph dims [`KimiLinearConfig::flow_config`] derives from the config
/// against the checkpoint's own tensor shapes (headers only): the embedding, the
/// first KDA, MLA, dense and MoE layers, and one routed expert. A config that
/// disagrees with its weights (a missing field falling back to a default) fails
/// here instead of loading and computing something else. Scalars no shape
/// carries — routed scaling, situ betas, gate bound — are reported by
/// [`numerics_summary`].
pub fn verify_flow_dims(tc: &KimiLinearConfig, ck: &mut CheckpointLoader) -> Result<()> {
    let cfg = tc.flow_config(1, 1);
    let h = cfg.hidden;
    let mut want: Vec<(String, Vec<usize>)> = vec![(EMBED.into(), vec![cfg.vocab, h])];
    let n = tc.num_hidden_layers;
    if let Some(i) = (0..n).find(|&i| tc.is_kda_layer(i)) {
        let (p, k) = (layer_prefix(i), &cfg.kda);
        let proj = k.num_heads * k.head_dim;
        want.push((format!("{p}.self_attn.q_proj.weight"), vec![proj, h]));
        want.push((
            format!("{p}.self_attn.f_a_proj.weight"),
            vec![k.head_dim, h],
        ));
        want.push((format!("{p}.self_attn.b_proj.weight"), vec![k.num_heads, h]));
        want.push((
            format!("{p}.self_attn.q_conv1d.weight"),
            vec![proj, 1, k.conv_kernel],
        ));
    }
    if let Some(i) = (0..n).find(|&i| !tc.is_kda_layer(i)) {
        let (p, m) = (layer_prefix(i), &cfg.mla);
        want.push((
            format!("{p}.self_attn.q_a_proj.weight"),
            vec![m.q_lora_rank, h],
        ));
        want.push((
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            vec![m.kv_lora_rank + m.qk_rope_head_dim, h],
        ));
        want.push((
            format!("{p}.self_attn.kv_b_proj.weight"),
            vec![
                m.num_heads * (m.qk_nope_head_dim + m.v_head_dim),
                m.kv_lora_rank,
            ],
        ));
        want.push((
            format!("{p}.self_attn.o_proj.weight"),
            vec![h, m.num_heads * m.v_head_dim],
        ));
    }
    if let Some(i) = (0..n).find(|&i| !tc.is_moe_layer(i)) {
        let p = layer_prefix(i);
        want.push((
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.dense_inter, h],
        ));
    }
    if let Some(i) = (0..n).find(|&i| tc.is_moe_layer(i)) {
        let (p, m) = (layer_prefix(i), &cfg.moe);
        let b = format!("{p}.block_sparse_moe");
        want.push((format!("{b}.gate.weight"), vec![m.num_experts, h]));
        want.push((
            format!("{b}.routed_expert_down_proj.weight"),
            vec![m.latent, h],
        ));
        want.push((
            format!("{b}.shared_experts.gate_proj.weight"),
            vec![m.num_shared * m.moe_inter, h],
        ));
        want.push((
            format!("{b}.experts.0.w1.weight_packed"),
            vec![m.moe_inter, m.latent / 2],
        ));
    }
    let mut bad = Vec::new();
    for (name, shape) in want {
        match ck.tensor_meta(&name) {
            Ok((_, got, _, _)) if got == shape => {}
            Ok((_, got, _, _)) => bad.push(format!(
                "{name}: checkpoint {got:?}, config implies {shape:?}"
            )),
            Err(e) => bad.push(format!("{name}: {e}")),
        }
    }
    ensure!(
        bad.is_empty(),
        "config.json disagrees with the checkpoint's tensors:\n  {}",
        bad.join("\n  ")
    );
    Ok(())
}

/// The numerics a config sets that no tensor shape can confirm, for a log line:
/// a field missing from `config.json` silently falls back to a default.
pub fn numerics_summary(tc: &KimiLinearConfig) -> String {
    let c = tc.flow_config(1, 1);
    format!(
        "routed_scaling {} · situ beta {} / linear {:?} · KDA gate bound {:?} · AttnRes block {} · eps {:e}",
        c.moe.routed_scaling,
        c.situ_beta,
        c.situ_linear_beta,
        c.kda.gate_lower_bound,
        c.attn_res_block_size,
        c.eps
    )
}

/// The decoder head: output Attention-Residual mix, final norm, untied LM head.
#[derive(Debug, Clone)]
pub struct HeadWeights {
    pub final_norm: Vec<f32>,
    pub out_res_norm: Vec<f32>,
    pub out_res_proj: Vec<f32>,
    /// `[hidden, vocab]`.
    pub lm_head: Vec<f32>,
}

/// A layer-at-a-time view of a Kimi-K3 model's weights.
pub trait WeightSource {
    fn num_layers(&self) -> usize;
    fn is_moe_layer(&self, i: usize) -> bool;
    /// Layer `i`'s backbone. For a MoE layer, `FfnWeights::Moe` carries the router,
    /// latent projections, norm and shared experts with **empty** expert banks —
    /// the routed experts come from [`Self::experts`].
    fn layer(&mut self, i: usize, cfg: &FlowConfig) -> Result<LayerWeights>;
    /// The routed experts `ids` of layer `i` as a compact bank, in order:
    /// `(gate_up [n, L, 2·mi], down [n, mi, L])`.
    fn experts(
        &mut self,
        i: usize,
        ids: &[usize],
        cfg: &FlowConfig,
    ) -> Result<(Vec<f32>, Vec<f32>)>;
    /// Embedding rows for `tokens`, `[tokens.len() · hidden]`.
    fn embed(&mut self, tokens: &[u32], hidden: usize) -> Result<Vec<f32>>;
    /// The full `[vocab, hidden]` embedding table (training it, dumping it).
    fn embed_table(&mut self) -> Result<Vec<f32>>;
    fn head(&mut self, cfg: &FlowConfig) -> Result<HeadWeights>;
}

/// [`WeightSource`] over an HF checkpoint directory, read lazily.
pub struct CheckpointSource {
    pub ck: CheckpointLoader,
    pub config: KimiK3Config,
}

impl CheckpointSource {
    /// Open a checkpoint and check its config against its tensor shapes
    /// ([`verify_flow_dims`]).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = KimiK3Config::load(dir.join("config.json"))?;
        let mut ck = CheckpointLoader::open(dir)?;
        verify_flow_dims(&config.text_config, &mut ck)?;
        Ok(Self { ck, config })
    }

    pub fn text_config(&self) -> &KimiLinearConfig {
        &self.config.text_config
    }
}

impl WeightSource for CheckpointSource {
    fn num_layers(&self) -> usize {
        self.config.text_config.num_hidden_layers
    }

    fn is_moe_layer(&self, i: usize) -> bool {
        self.config.text_config.is_moe_layer(i)
    }

    fn layer(&mut self, i: usize, cfg: &FlowConfig) -> Result<LayerWeights> {
        let tc = &self.config.text_config;
        let lp = layer_prefix(i);
        let ck = &mut self.ck;
        let attn = if tc.is_kda_layer(i) {
            AttnWeights::Kda(Box::new(ck.load_kda(&lp, cfg.kda)?))
        } else {
            AttnWeights::Mla(Box::new(ck.load_mla(&lp, cfg.mla)?))
        };
        let ffn = if tc.is_moe_layer(i) {
            FfnWeights::Moe(Box::new(ck.load_moe_dense(&lp, cfg.moe)?))
        } else {
            FfnWeights::Dense(Box::new(ck.load_dense_mlp(
                &lp,
                cfg.hidden,
                cfg.dense_inter,
            )?))
        };
        Ok(LayerWeights {
            input_ln: ck.tensor_f32(&format!("{lp}.input_layernorm.weight"))?,
            post_ln: ck.tensor_f32(&format!("{lp}.post_attention_layernorm.weight"))?,
            sa_res_norm: ck.tensor_f32(&format!("{lp}.self_attention_res_norm.weight"))?,
            sa_res_proj: ck.tensor_f32(&format!("{lp}.self_attention_res_proj.weight"))?,
            mlp_res_norm: ck.tensor_f32(&format!("{lp}.mlp_res_norm.weight"))?,
            mlp_res_proj: ck.tensor_f32(&format!("{lp}.mlp_res_proj.weight"))?,
            attn,
            ffn,
        })
    }

    fn experts(
        &mut self,
        i: usize,
        ids: &[usize],
        cfg: &FlowConfig,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use rayon::prelude::*;
        let (l, mi) = (cfg.moe.latent, cfg.moe.moe_inter);
        let lp = layer_prefix(i);
        // Resolve byte ranges here, read + dequant + transpose concurrently.
        let ranges: Vec<_> = ids
            .iter()
            .map(|&e| self.ck.expert_ranges(&lp, e))
            .collect::<Result<_>>()?;
        let loaded: Vec<(Vec<f32>, Vec<f32>)> = ranges
            .par_iter()
            .map(|r| crate::loader::load_expert_ranges(r, l, mi))
            .collect::<Result<_>>()?;
        let mut gate_up = Vec::with_capacity(ids.len() * l * 2 * mi);
        let mut down = Vec::with_capacity(ids.len() * mi * l);
        for (gu, dn) in loaded {
            gate_up.extend(gu);
            down.extend(dn);
        }
        Ok((gate_up, down))
    }

    fn embed(&mut self, tokens: &[u32], hidden: usize) -> Result<Vec<f32>> {
        self.ck.gather_embed(EMBED, tokens, hidden)
    }

    fn embed_table(&mut self) -> Result<Vec<f32>> {
        self.ck.tensor_f32(EMBED)
    }

    fn head(&mut self, cfg: &FlowConfig) -> Result<HeadWeights> {
        Ok(HeadWeights {
            final_norm: self.ck.tensor_f32(FINAL_NORM)?,
            out_res_norm: self.ck.tensor_f32(OUT_RES_NORM)?,
            out_res_proj: self.ck.tensor_f32(OUT_RES_PROJ)?,
            lm_head: self.ck.linear_t(LM_HEAD, cfg.vocab, cfg.hidden)?,
        })
    }
}

/// [`WeightSource`] over a model held in RAM (full expert banks included).
pub struct MemorySource {
    pub weights: FlowWeights,
    /// `[vocab, hidden]`.
    pub embed: Vec<f32>,
    pub hidden: usize,
}

impl MemorySource {
    fn moe(&self, i: usize) -> Result<&MoeWeights> {
        match &self.weights.layers[i].ffn {
            FfnWeights::Moe(m) => Ok(m),
            FfnWeights::Dense(_) => bail!("layer {i} has a dense FFN"),
        }
    }
}

impl WeightSource for MemorySource {
    fn num_layers(&self) -> usize {
        self.weights.layers.len()
    }

    fn is_moe_layer(&self, i: usize) -> bool {
        matches!(self.weights.layers[i].ffn, FfnWeights::Moe(_))
    }

    fn layer(&mut self, i: usize, _cfg: &FlowConfig) -> Result<LayerWeights> {
        let lw = self
            .weights
            .layers
            .get(i)
            .with_context(|| format!("no layer {i}"))?;
        Ok(LayerWeights {
            input_ln: lw.input_ln.clone(),
            post_ln: lw.post_ln.clone(),
            sa_res_norm: lw.sa_res_norm.clone(),
            sa_res_proj: lw.sa_res_proj.clone(),
            mlp_res_norm: lw.mlp_res_norm.clone(),
            mlp_res_proj: lw.mlp_res_proj.clone(),
            attn: match &lw.attn {
                AttnWeights::Kda(k) => AttnWeights::Kda(k.clone()),
                AttnWeights::Mla(m) => AttnWeights::Mla(m.clone()),
            },
            ffn: match &lw.ffn {
                FfnWeights::Dense(d) => FfnWeights::Dense(d.clone()),
                FfnWeights::Moe(m) => FfnWeights::Moe(Box::new(MoeWeights {
                    experts_gate_up: Vec::new(),
                    experts_down: Vec::new(),
                    ..(**m).clone()
                })),
            },
        })
    }

    fn experts(
        &mut self,
        i: usize,
        ids: &[usize],
        cfg: &FlowConfig,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (l, mi, e) = (cfg.moe.latent, cfg.moe.moe_inter, cfg.moe.num_experts);
        let m = self.moe(i)?;
        let (gu_n, dn_n) = (l * 2 * mi, mi * l);
        ensure!(
            m.experts_gate_up.len() == e * gu_n && m.experts_down.len() == e * dn_n,
            "layer {i}: expert banks do not match {e} experts"
        );
        let mut gate_up = Vec::with_capacity(ids.len() * gu_n);
        let mut down = Vec::with_capacity(ids.len() * dn_n);
        for &x in ids {
            ensure!(x < e, "layer {i}: expert {x} out of range");
            gate_up.extend_from_slice(&m.experts_gate_up[x * gu_n..(x + 1) * gu_n]);
            down.extend_from_slice(&m.experts_down[x * dn_n..(x + 1) * dn_n]);
        }
        Ok((gate_up, down))
    }

    fn embed(&mut self, tokens: &[u32], hidden: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        for &t in tokens {
            let r = t as usize * hidden;
            ensure!(r + hidden <= self.embed.len(), "token {t} out of range");
            out.extend_from_slice(&self.embed[r..r + hidden]);
        }
        Ok(out)
    }

    fn embed_table(&mut self) -> Result<Vec<f32>> {
        Ok(self.embed.clone())
    }

    fn head(&mut self, _cfg: &FlowConfig) -> Result<HeadWeights> {
        let w = &self.weights;
        Ok(HeadWeights {
            final_norm: w.final_norm.clone(),
            out_res_norm: w.out_res_norm.clone(),
            out_res_proj: w.out_res_proj.clone(),
            lm_head: w.lm_head.clone(),
        })
    }
}

/// Load a whole model — every layer **with full expert banks** — plus the
/// embedding table into RAM. Only for checkpoints that fit (synthetic or
/// truncated ones); the real model's experts alone are 1.45 TB.
pub fn load_flow_weights(
    src: &mut CheckpointSource,
    cfg: &FlowConfig,
) -> Result<(FlowWeights, Vec<f32>)> {
    let n = src.num_layers();
    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        let mut lw = src.layer(i, cfg)?;
        if let FfnWeights::Moe(m) = &mut lw.ffn {
            let all: Vec<usize> = (0..cfg.moe.num_experts).collect();
            let (gu, dn) = src.experts(i, &all, cfg)?;
            m.experts_gate_up = gu;
            m.experts_down = dn;
        }
        layers.push(lw);
    }
    let head = src.head(cfg)?;
    let embed = src.embed_table()?;
    Ok((
        FlowWeights {
            layers,
            final_norm: head.final_norm,
            out_res_norm: head.out_res_norm,
            out_res_proj: head.out_res_proj,
            lm_head: head.lm_head,
        },
        embed,
    ))
}
