//! Load CSM-1B transformers-layout safetensors (skip embedded `codec_model.*`).

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::config::{DepthDecoderConfig, SesameConfig};
use crate::nn::{apply_rope, gqa_attend, llama3_inv_freq, matmul_in_out, matvec, rms_norm, silu};

pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub q_w: Vec<f32>,
    pub k_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub o_w: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub up_w: Vec<f32>,
    pub down_w: Vec<f32>,
}

pub struct TransformerWeights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub inv_freq: Vec<f64>,
}

pub struct CsmWeights {
    pub cfg: SesameConfig,
    pub text_embed: Vec<f32>,
    /// Shared audio codebook embeddings `[num_codebooks * vocab, hidden]`.
    pub audio_embed: Vec<f32>,
    pub backbone: TransformerWeights,
    pub lm_head: Vec<f32>,
    pub depth: TransformerWeights,
    /// `[1024, 2048]` — projects backbone dim → depth dim.
    pub depth_projector: Vec<f32>,
    /// `[31, depth_hidden, vocab]` flattened as `[31 * depth_hidden * vocab]`.
    pub codebooks_head: Vec<f32>,
}

pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

pub struct KvCache {
    pub layers: Vec<LayerKv>,
    pub num_tokens: usize,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKv {
                    k: Vec::new(),
                    v: Vec::new(),
                })
                .collect(),
            num_tokens: 0,
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.k.clear();
            l.v.clear();
        }
        self.num_tokens = 0;
    }
}

fn f32_from_bytes_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn half_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let exp = ((h as u32 >> 10) & 0x1f) as i32 - 15 + 127;
    let mant = (h as u32 & 0x3ff) << 13;
    if exp <= 0 {
        return f32::from_bits(sign);
    }
    if exp >= 255 {
        return f32::from_bits(sign | 0x7f80_0000 | mant);
    }
    f32::from_bits(sign | ((exp as u32) << 23) | mant)
}

fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half_bits_to_f32(bits)
        })
        .collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn read_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("tensor '{name}' not found"))?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(f32_from_bytes_le(bytes)),
        Dtype::F16 => Ok(f16_to_f32(bytes)),
        Dtype::BF16 => Ok(bf16_to_f32(bytes)),
        other => anyhow::bail!("unsupported dtype {other:?} for '{name}'"),
    }
}

fn load_transformer(
    st: &SafeTensors<'_>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_theta: f64,
    rope_scaling: Option<&crate::config::RopeScaling>,
) -> Result<TransformerWeights> {
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let p = format!("{prefix}.layers.{i}");
        layers.push(LayerWeights {
            input_norm: read_f32(st, &format!("{p}.input_layernorm.weight"))?,
            post_attn_norm: read_f32(st, &format!("{p}.post_attention_layernorm.weight"))?,
            q_w: read_f32(st, &format!("{p}.self_attn.q_proj.weight"))?,
            k_w: read_f32(st, &format!("{p}.self_attn.k_proj.weight"))?,
            v_w: read_f32(st, &format!("{p}.self_attn.v_proj.weight"))?,
            o_w: read_f32(st, &format!("{p}.self_attn.o_proj.weight"))?,
            gate_w: read_f32(st, &format!("{p}.mlp.gate_proj.weight"))?,
            up_w: read_f32(st, &format!("{p}.mlp.up_proj.weight"))?,
            down_w: read_f32(st, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    let final_norm = read_f32(st, &format!("{prefix}.norm.weight"))?;
    let inv_freq = llama3_inv_freq(rope_theta, head_dim, rope_scaling);
    Ok(TransformerWeights {
        layers,
        final_norm,
        hidden,
        intermediate,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_eps,
        inv_freq,
    })
}

fn resolve_weights_path(model_dir: &Path) -> Result<PathBuf> {
    let candidates = [
        model_dir.join("model.safetensors"),
        model_dir.join("transformers-00001-of-00002.safetensors"),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    anyhow::bail!(
        "no CSM safetensors in {} (expected model.safetensors)",
        model_dir.display()
    )
}

impl CsmWeights {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let cfg = SesameConfig::from_file(model_dir.join("config.json"))?;
        let path = resolve_weights_path(model_dir)?;
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.context("mmap safetensors")?;
        let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;

        let text_embed = read_f32(&st, "embed_text_tokens.weight")?;
        let audio_embed = read_f32(&st, "depth_decoder.model.embed_tokens.weight")?;
        let lm_head = read_f32(&st, "lm_head.weight")?;
        let depth_projector = read_f32(&st, "depth_decoder.model.inputs_embeds_projector.weight")?;
        let codebooks_head = read_f32(&st, "depth_decoder.codebooks_head.weight")?;

        let backbone = load_transformer(
            &st,
            "backbone_model",
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rms_norm_eps as f32,
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        )?;

        let dcfg: &DepthDecoderConfig = &cfg.depth_decoder_config;
        let depth = load_transformer(
            &st,
            "depth_decoder.model",
            dcfg.hidden_size,
            dcfg.intermediate_size,
            dcfg.num_hidden_layers,
            dcfg.num_attention_heads,
            dcfg.num_key_value_heads,
            dcfg.head_dim,
            dcfg.rms_norm_eps as f32,
            dcfg.rope_theta,
            dcfg.rope_scaling.as_ref(),
        )?;

        Ok(Self {
            cfg,
            text_embed,
            audio_embed,
            backbone,
            lm_head,
            depth,
            depth_projector,
            codebooks_head,
        })
    }

    pub fn embed_text(&self, token: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let id = token as usize;
        self.text_embed[id * h..(id + 1) * h].to_vec()
    }

    pub fn embed_audio(&self, codebook: usize, token: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let id = codebook * vocab + token as usize;
        self.audio_embed[id * h..(id + 1) * h].to_vec()
    }

    /// Sum masked frame embeds: audio codebooks 0..K-1 + optional text column.
    pub fn embed_frame(&self, tokens: &[u32], mask: &[bool]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let k = self.cfg.num_codebooks;
        debug_assert_eq!(tokens.len(), k + 1);
        debug_assert_eq!(mask.len(), k + 1);
        let mut out = vec![0.0f32; h];
        for cb in 0..k {
            if mask[cb] {
                let e = self.embed_audio(cb, tokens[cb]);
                for (o, v) in out.iter_mut().zip(&e) {
                    *o += *v;
                }
            }
        }
        if mask[k] {
            let e = self.embed_text(tokens[k]);
            for (o, v) in out.iter_mut().zip(&e) {
                *o += *v;
            }
        }
        out
    }

    pub fn c0_logits(&self, hidden: &[f32]) -> Vec<f32> {
        // lm_head: [vocab, hidden]
        matvec(
            &self.lm_head,
            hidden,
            self.cfg.hidden_size,
            self.cfg.vocab_size,
        )
    }

    pub fn codebook_logits(&self, depth_h: &[f32], codebook_idx: usize) -> Vec<f32> {
        // codebooks_head: [31, depth_hidden, vocab]; codebook_idx is 1..31 → head 0..30
        // PyTorch: `h @ audio_head[i-1]` with W shape [depth_hidden, vocab].
        let dh = self.depth.hidden;
        let vocab = self.cfg.vocab_size;
        let head = codebook_idx - 1;
        let offset = head * dh * vocab;
        let w = &self.codebooks_head[offset..offset + dh * vocab];
        matmul_in_out(w, depth_h, dh, vocab)
    }

    pub fn project_to_depth(&self, backbone_h: &[f32]) -> Vec<f32> {
        // projector: [depth_hidden, backbone_hidden]
        matvec(
            &self.depth_projector,
            backbone_h,
            self.cfg.hidden_size,
            self.depth.hidden,
        )
    }
}

fn layer_step(
    h: &[f32],
    layer: &LayerWeights,
    kv: &mut LayerKv,
    pos: usize,
    tw: &TransformerWeights,
) -> Vec<f32> {
    let hidden = tw.hidden;
    let q_dim = tw.num_heads * tw.head_dim;
    let kv_dim = tw.num_kv_heads * tw.head_dim;

    let h_norm = rms_norm(h, &layer.input_norm, tw.rms_eps);
    let mut q = matvec(&layer.q_w, &h_norm, hidden, q_dim);
    let mut k = matvec(&layer.k_w, &h_norm, hidden, kv_dim);
    let v = matvec(&layer.v_w, &h_norm, hidden, kv_dim);

    apply_rope(&mut q, pos, tw.head_dim, &tw.inv_freq);
    apply_rope(&mut k, pos, tw.head_dim, &tw.inv_freq);

    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    let n_kv = pos + 1;

    let attn_out = gqa_attend(
        &q,
        &kv.k,
        &kv.v,
        n_kv,
        tw.num_heads,
        tw.num_kv_heads,
        tw.head_dim,
    );
    let attn_proj = matvec(&layer.o_w, &attn_out, q_dim, hidden);
    let mut h_new: Vec<f32> = h.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();

    let h_norm2 = rms_norm(&h_new, &layer.post_attn_norm, tw.rms_eps);
    let gate = matvec(&layer.gate_w, &h_norm2, hidden, tw.intermediate);
    let up = matvec(&layer.up_w, &h_norm2, hidden, tw.intermediate);
    let ffn_hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    let ffn_out = matvec(&layer.down_w, &ffn_hidden, tw.intermediate, hidden);
    for (hi, fi) in h_new.iter_mut().zip(&ffn_out) {
        *hi += fi;
    }
    h_new
}

/// Run one position through a transformer stack; append to KV; return final-normed hidden.
pub fn transformer_step(embed: &[f32], tw: &TransformerWeights, kv: &mut KvCache) -> Vec<f32> {
    let pos = kv.num_tokens;
    let mut h = embed.to_vec();
    for (li, layer) in tw.layers.iter().enumerate() {
        h = layer_step(&h, layer, &mut kv.layers[li], pos, tw);
    }
    let out = rms_norm(&h, &tw.final_norm, tw.rms_eps);
    kv.num_tokens += 1;
    out
}

/// Prefill a sequence of embeds through the backbone.
pub fn backbone_prefill(embeds: &[Vec<f32>], weights: &CsmWeights, kv: &mut KvCache) -> Vec<f32> {
    let mut last = Vec::new();
    for e in embeds {
        last = transformer_step(e, &weights.backbone, kv);
    }
    last
}
