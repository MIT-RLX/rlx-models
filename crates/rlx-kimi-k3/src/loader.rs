// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Real-weight loader for a Kimi-K3 checkpoint (HF sharded safetensors). mmaps
//! shards on demand via `model.safetensors.index.json`, reads BF16 / F32 tensors,
//! and dequantizes the **MXFP4** routed experts (`weight_packed` = 2× E2M1
//! nibbles/byte, `weight_scale` = E8M0 block exponent per 32 elements). Fills the
//! dense weight structs the graph builders consume (`KdaWeights`, `MlaWeights`,
//! `DenseMlpWeights`, `MoeWeights`).
//!
//! The 114 GB backbone (attention/dense/embed/norm/head) loads densely and fits;
//! the 1.45 TB of routed experts cannot be materialized in full — `load_expert`
//! dequantizes ONE expert at a time (for paging / spot checks), never a whole
//! 896-expert layer.

use crate::kda::{KdaDims, KdaWeights};
use crate::mla::{MlaDims, MlaWeights};
use crate::moe::{DenseMlpWeights, MoeDims, MoeWeights};
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// One shard's parsed header: name → (dtype, shape, byte range after the header).
struct Shard {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

/// One MoE expert's MXFP4 weights kept PACKED (raw bytes) for the GPU
/// dequant-matmul path — see [`CheckpointLoader::load_expert_packed`]. `w1`/`w3`
/// are `[mi, L]`, `w2` is `[L, mi]` (Kimi HF `[out, in]` = `MlxMxfp4` convention).
pub struct ExpertPacked {
    pub w1_q: Vec<u8>,
    pub w1_s: Vec<u8>,
    pub w3_q: Vec<u8>,
    pub w3_s: Vec<u8>,
    pub w2_q: Vec<u8>,
    pub w2_s: Vec<u8>,
}

/// Lazily-mmapped sharded safetensors checkpoint.
pub struct CheckpointLoader {
    dir: PathBuf,
    weight_map: HashMap<String, String>, // tensor name → shard file
    shards: HashMap<String, Shard>,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    // Parallel: also lets multiple cores fault-in the mmap concurrently.
    b.par_chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn f32_of(b: &[u8]) -> Vec<f32> {
    b.par_chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Core MXFP4 dequant on raw bytes (no `self`) — shared by [`CheckpointLoader::
/// dequant_mxfp4`] and the parallel expert loader. `packed` = 2× E2M1 nibbles/byte
/// (low nibble first); `scale` = E8M0 bytes, one per `block` elements. Row-major
/// `[rows*cols]`. Scale hoisted per block; whole-nibble value LUT.
fn dequant_mxfp4_bytes(
    packed: &[u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
    block: usize,
) -> Vec<f32> {
    const VAL: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let scale_lut: [f32; 256] = std::array::from_fn(|e| 2f32.powi(e as i32 - 127));
    let (pcols, scols) = (cols / 2, cols / block);
    let mut out = vec![0f32; rows * cols];
    out.par_chunks_mut(cols).enumerate().for_each(|(r, orow)| {
        let prow = &packed[r * pcols..(r + 1) * pcols];
        let srow = &scale[r * scols..(r + 1) * scols];
        for (bi, run) in orow.chunks_mut(block).enumerate() {
            let s = scale_lut[srow[bi] as usize];
            let c0 = bi * block;
            for (o, cell) in run.iter_mut().enumerate() {
                let c = c0 + o;
                let byte = prow[c / 2];
                let nib = if c.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                *cell = VAL[nib as usize] * s;
            }
        }
    });
    out
}

impl CheckpointLoader {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let idx: serde_json::Value = serde_json::from_reader(
            File::open(dir.join("model.safetensors.index.json")).context("open index")?,
        )?;
        let weight_map = idx
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("no weight_map")?
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        Ok(Self {
            dir,
            weight_map,
            shards: HashMap::new(),
        })
    }

    fn shard(&mut self, file: &str) -> Result<&Shard> {
        if !self.shards.contains_key(file) {
            let f = File::open(self.dir.join(file)).with_context(|| format!("open {file}"))?;
            let mmap = unsafe { Mmap::map(&f)? };
            let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])?;
            let mut tensors = HashMap::new();
            for (name, meta) in header.as_object().context("header")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|u| u as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                let off = meta["data_offsets"].as_array().context("offsets")?;
                let (a, b) = (
                    off[0].as_u64().unwrap() as usize,
                    off[1].as_u64().unwrap() as usize,
                );
                tensors.insert(name.clone(), (dtype, shape, a, b));
            }
            self.shards.insert(
                file.to_string(),
                Shard {
                    mmap,
                    data_start: 8 + hlen,
                    tensors,
                },
            );
        }
        Ok(&self.shards[file])
    }

    /// Raw bytes + dtype + shape for a tensor (a slice into the shard mmap).
    fn raw(&mut self, name: &str) -> Result<(&'static str, Vec<usize>, &[u8])> {
        let file = self
            .weight_map
            .get(name)
            .with_context(|| format!("{name} not in index"))?
            .clone();
        let sh = self.shard(&file)?;
        let (dt, shape, a, b) = sh
            .tensors
            .get(name)
            .with_context(|| format!("{name} in shard"))?;
        let bytes = &sh.mmap[sh.data_start + a..sh.data_start + b];
        let dt: &'static str = match dt.as_str() {
            "BF16" => "BF16",
            "F32" => "F32",
            "U8" => "U8",
            other => bail!("{name}: unsupported dtype {other}"),
        };
        Ok((dt, shape.clone(), bytes))
    }

    /// F32 tensor (F32 as-is or BF16 upcast), flat.
    pub fn tensor_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, b) = self.raw(name)?;
        Ok(match dt {
            "F32" => f32_of(b),
            "BF16" => bf16_to_f32(b),
            _ => bail!("{name}: not float"),
        })
    }

    /// [`linear_t`], or an **empty Vec when a prequantized backbone is being loaded**
    /// (`RLX_KIMI_PREQUANT_LOAD`): the int8-resident graph builder mmaps this weight's
    /// codes by name in `emit_int8_resident`, so the bf16 read + f32 upcast + transpose
    /// are skipped entirely. Only used for weights that are always int8-resident.
    pub(crate) fn lin(&mut self, name: &str, out_dim: usize, in_dim: usize) -> Result<Vec<f32>> {
        if crate::common::prequant_load_active() {
            Ok(Vec::new())
        } else {
            self.linear_t(name, out_dim, in_dim)
        }
    }

    /// Just the checkpoint shape of `name` (no data materialized).
    pub fn tensor_shape(&mut self, name: &str) -> Result<Vec<usize>> {
        let (_, shape, _) = self.raw(name)?;
        Ok(shape)
    }

    /// F32 tensor **with its checkpoint shape** (F32 as-is or BF16 upcast), flat in
    /// native `[out, in]` order. Used by the backbone quantizer to iterate + encode
    /// every weight without transposing.
    pub fn tensor_f32_shaped(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (dt, shape, b) = self.raw(name)?;
        let f = match dt {
            "F32" => f32_of(b),
            "BF16" => bf16_to_f32(b),
            _ => bail!("{name}: not float"),
        };
        Ok((shape, f))
    }

    /// An `nn.Linear` weight (HF stores `[out, in]`) transposed to `[in, out]`,
    /// matching the crate's `x[.,in] @ w[in,out]` convention.
    pub fn linear_t(&mut self, name: &str, out_dim: usize, in_dim: usize) -> Result<Vec<f32>> {
        let (_, shape, b) = self.raw(name)?;
        if shape != [out_dim, in_dim] {
            bail!("{name}: shape {shape:?} != [{out_dim},{in_dim}]");
        }
        let flat = if b.len() == out_dim * in_dim * 2 {
            bf16_to_f32(b)
        } else {
            f32_of(b)
        };
        // Parallel transpose `[out,in] → [in,out]`: each output row `i` gathers
        // column `i` of `flat` (strided by in_dim). Rows are independent.
        let mut out = vec![0f32; in_dim * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(i, row)| {
                for (o, cell) in row.iter_mut().enumerate() {
                    *cell = flat[o * in_dim + i];
                }
            });
        Ok(out)
    }

    /// Load `name` (`[out, in]` = `[N, K]` **BF16** in the checkpoint, HF
    /// `[vocab, hidden]`) TRANSPOSED to a genuine `[K, N]` = `[in, out]` BF16 tensor
    /// — the standard `g.mm(x[M,K], w[K,N])` weight layout, so EVERY backend's native
    /// BF16 matmul consumes it correctly (not a CPU-only B-transposed reinterpret).
    /// Half the bytes of the f32 `linear_t`. One pass, parallel over the `K` output
    /// rows (each gathers its strided source column), directly into the byte buffer.
    pub fn lm_head_bf16_kn(
        &mut self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<u8>> {
        let (dt, shape, b) = self.raw(name)?;
        if shape != [out_dim, in_dim] {
            bail!("{name}: shape {shape:?} != [{out_dim},{in_dim}]");
        }
        if b.len() != out_dim * in_dim * 2 {
            bail!(
                "{name}: expected BF16 ({} bytes), got {} (dtype {dt})",
                out_dim * in_dim * 2,
                b.len()
            );
        }
        let (n, k) = (out_dim, in_dim); // src [N,K] → dst [K,N]
        let mut out = vec![0u8; n * k * 2];
        // dst row `kk` (0..K): dst[kk, nn] = src[nn, kk]  for nn in 0..N
        out.par_chunks_mut(n * 2).enumerate().for_each(|(kk, row)| {
            for nn in 0..n {
                let s = 2 * (nn * k + kk);
                row[nn * 2] = b[s];
                row[nn * 2 + 1] = b[s + 1];
            }
        });
        Ok(out)
    }

    /// Load the **MoonViT vision tower + patchmergerv2 projector** real weights and
    /// derive its `VisionDims` from the config + weight shapes. `grid_h`/`grid_w`
    /// (the image's patch grid) default to `init_pos_emb_{height,width}`; the caller
    /// overrides them per image. All matmul weights are `linear_t`-transposed to the
    /// `[in, out]` layout [`crate::vision::build_vision`] expects.
    pub fn load_vision(
        &mut self,
        cfg: &crate::config::KimiVisionConfig,
    ) -> Result<(crate::vision::VisionWeights, crate::vision::VisionDims)> {
        use crate::vision::{VisionBlockWeights, VisionDims, VisionWeights};
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let inter = cfg.intermediate_size;
        let merge = cfg.merge_kernel_size.first().copied().unwrap_or(2);
        let eps = cfg.rms_norm_eps;
        // derive attention + projector widths from the real HF weight shapes.
        let qkv_hidden = {
            let (_, s, _) = self.raw("vision_tower.encoder.blocks.0.wqkv.weight")?;
            s[0] / 3 // HF wqkv is [3*qkv_hidden, hidden]
        };
        let head_dim = qkv_hidden / num_heads.max(1);
        let proj_mid = {
            let (_, s, _) = self.raw("mm_projector.proj.0.weight")?;
            s[0] // HF [proj_mid, merge_in]
        };
        let text_hidden = {
            let (_, s, _) = self.raw("mm_projector.proj.2.weight")?;
            s[0] // HF [text_hidden, proj_mid]
        };
        let d = VisionDims {
            hidden,
            qkv_hidden,
            num_heads,
            head_dim,
            inter,
            merge,
            text_hidden,
            proj_mid,
            eps,
            grid_h: cfg.init_pos_emb_height,
            grid_w: cfg.init_pos_emb_width,
        };
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for b in 0..cfg.num_hidden_layers {
            let p = format!("vision_tower.encoder.blocks.{b}");
            blocks.push(VisionBlockWeights {
                norm0: self.tensor_f32(&format!("{p}.norm0.weight"))?,
                wqkv: self.linear_t(&format!("{p}.wqkv.weight"), 3 * qkv_hidden, hidden)?,
                wo: self.linear_t(&format!("{p}.wo.weight"), hidden, qkv_hidden)?,
                norm1: self.tensor_f32(&format!("{p}.norm1.weight"))?,
                fc0: self.linear_t(&format!("{p}.mlp.fc0.weight"), inter, hidden)?,
                fc1: self.linear_t(&format!("{p}.mlp.fc1.weight"), hidden, inter)?,
            });
        }
        let merge_in = merge * merge * hidden;
        let w = VisionWeights {
            blocks,
            final_norm: self.tensor_f32("vision_tower.encoder.final_layernorm.weight")?,
            proj0: self.linear_t("mm_projector.proj.0.weight", proj_mid, merge_in)?,
            proj2: self.linear_t("mm_projector.proj.2.weight", text_hidden, proj_mid)?,
            post_norm: self.tensor_f32("mm_projector.post_norm.weight")?,
        };
        Ok((w, d))
    }

    /// Load the vision **patch embedding**: the patchify conv weight
    /// `[hidden, 3, patch, patch]` flattened to `[hidden, 3*patch*patch]` (row =
    /// output channel — a patch flattened `[c,kh,kw]` dotted against each row gives
    /// that channel), and the learnable position embedding `[pos_h, pos_w, hidden]`
    /// (bilinear-interpolated to the image's patch grid). Returns
    /// `(conv, pos_emb, pos_h, pos_w)`. No bias in this checkpoint.
    pub fn load_patch_embed(&mut self) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let (pos_h, pos_w) = {
            let (_, s, _) = self.raw("vision_tower.patch_embed.pos_emb.weight")?;
            (s[0], s[1]) // [pos_h, pos_w, hidden]
        };
        let conv = self.tensor_f32("vision_tower.patch_embed.proj.weight")?; // [hid,3,p,p] row-major
        let pos_emb = self.tensor_f32("vision_tower.patch_embed.pos_emb.weight")?;
        Ok((conv, pos_emb, pos_h, pos_w))
    }

    /// Gather embedding rows for `tokens` from `embed_tokens.weight`
    /// (`[vocab, hidden]` BF16) → `[tokens.len() * hidden]` f32.
    pub fn gather_embed(&mut self, name: &str, tokens: &[u32], hidden: usize) -> Result<Vec<f32>> {
        let (dt, shape, b) = self.raw(name)?;
        if shape.len() != 2 || shape[1] != hidden {
            bail!("{name}: embed shape {shape:?}");
        }
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        for &tk in tokens {
            let r = tk as usize;
            match dt {
                "BF16" => out.extend(bf16_to_f32(&b[r * hidden * 2..(r + 1) * hidden * 2])),
                "F32" => out.extend(f32_of(&b[r * hidden * 4..(r + 1) * hidden * 4])),
                _ => bail!("embed dtype"),
            }
        }
        Ok(out)
    }

    // ── MXFP4 ────────────────────────────────────────────────────────────────

    /// Dequantize an MXFP4 `[rows, cols]` tensor: `packed [rows, cols/2]` U8 (two
    /// E2M1 nibbles/byte, low nibble first) × `scale [rows, cols/32]` U8 (E8M0
    /// exponent, value `2^(e-127)`). Returns row-major `[rows*cols]` f32.
    pub fn dequant_mxfp4(
        &mut self,
        packed: &str,
        scale: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let (_, ps, pb) = self.raw(packed)?;
        if ps != [rows, cols / 2] {
            bail!("{packed}: shape {ps:?} != [{rows},{}]", cols / 2);
        }
        let pb = pb.to_vec();
        let (_, ss, sb) = self.raw(scale)?;
        let block = cols / ss[1].max(1);
        if ss != [rows, cols / block] {
            bail!("{scale}: shape {ss:?}");
        }
        let sb = sb.to_vec();
        Ok(dequant_mxfp4_bytes(&pb, &sb, rows, cols, block))
    }

    // ── layer loaders ─────────────────────────────────────────────────────────

    /// Load a KDA `self_attn` block into `KdaWeights` (prefix e.g.
    /// `language_model.model.layers.0`).
    pub fn load_kda(&mut self, layer_prefix: &str, d: KdaDims) -> Result<KdaWeights> {
        let (hidden, h, hd, proj) = (d.hidden, d.num_heads, d.head_dim, d.proj());
        let a = format!("{layer_prefix}.self_attn");
        Ok(KdaWeights {
            q_proj: self.lin(&format!("{a}.q_proj.weight"), proj, hidden)?,
            k_proj: self.lin(&format!("{a}.k_proj.weight"), proj, hidden)?,
            v_proj: self.lin(&format!("{a}.v_proj.weight"), proj, hidden)?,
            q_conv: self.tensor_f32(&format!("{a}.q_conv1d.weight"))?,
            k_conv: self.tensor_f32(&format!("{a}.k_conv1d.weight"))?,
            v_conv: self.tensor_f32(&format!("{a}.v_conv1d.weight"))?,
            f_a: self.lin(&format!("{a}.f_a_proj.weight"), hd, hidden)?,
            f_b: self.lin(&format!("{a}.f_b_proj.weight"), proj, hd)?,
            dt_bias: self.tensor_f32(&format!("{a}.dt_bias"))?,
            a_log: self.tensor_f32(&format!("{a}.A_log"))?,
            b_proj: self.lin(&format!("{a}.b_proj.weight"), h, hidden)?,
            g_proj: self.lin(&format!("{a}.g_proj.weight"), proj, hidden)?,
            o_norm: self.tensor_f32(&format!("{a}.o_norm.weight"))?,
            o_proj: self.lin(&format!("{a}.o_proj.weight"), hidden, proj)?,
        })
    }

    /// Load the dense (`first_k_dense_replace`) MLP into `DenseMlpWeights`.
    pub fn load_dense_mlp(
        &mut self,
        layer_prefix: &str,
        hidden: usize,
        inter: usize,
    ) -> Result<DenseMlpWeights> {
        let m = format!("{layer_prefix}.mlp");
        Ok(DenseMlpWeights {
            gate: self.linear_t(&format!("{m}.gate_proj.weight"), inter, hidden)?,
            up: self.linear_t(&format!("{m}.up_proj.weight"), inter, hidden)?,
            down: self.linear_t(&format!("{m}.down_proj.weight"), hidden, inter)?,
        })
    }

    /// Load an MLA `self_attn` block into `MlaWeights`.
    pub fn load_mla(&mut self, layer_prefix: &str, d: MlaDims) -> Result<MlaWeights> {
        let (hidden, h, ql, kvl, nope, rope, vd) = (
            d.hidden,
            d.num_heads,
            d.q_lora_rank,
            d.kv_lora_rank,
            d.qk_nope_head_dim,
            d.qk_rope_head_dim,
            d.v_head_dim,
        );
        let qk = nope + rope;
        let a = format!("{layer_prefix}.self_attn");
        Ok(MlaWeights {
            q_a_proj: self.lin(&format!("{a}.q_a_proj.weight"), ql, hidden)?,
            q_a_layernorm: self.tensor_f32(&format!("{a}.q_a_layernorm.weight"))?,
            q_b_proj: self.lin(&format!("{a}.q_b_proj.weight"), h * qk, ql)?,
            kv_a_proj_with_mqa: self.linear_t(
                &format!("{a}.kv_a_proj_with_mqa.weight"),
                kvl + rope,
                hidden,
            )?,
            kv_a_layernorm: self.tensor_f32(&format!("{a}.kv_a_layernorm.weight"))?,
            kv_b_proj: self.lin(&format!("{a}.kv_b_proj.weight"), h * (nope + vd), kvl)?,
            g_proj: self.lin(&format!("{a}.g_proj.weight"), h * vd, hidden)?,
            o_proj: self.lin(&format!("{a}.o_proj.weight"), hidden, h * vd)?,
        })
    }

    /// Load the LatentMoE **non-expert** weights (router, latent down/up, norm,
    /// 2 shared experts) — small & dense. The 896 routed experts are NOT loaded
    /// here (MXFP4, 16 GB/layer); use `load_expert` to page them.
    pub fn load_moe_dense(&mut self, layer_prefix: &str, d: MoeDims) -> Result<MoeWeights> {
        let (hidden, l, e, si) = (
            d.hidden,
            d.latent,
            d.num_experts,
            d.num_shared * d.moe_inter,
        );
        let m = format!("{layer_prefix}.block_sparse_moe");
        Ok(MoeWeights {
            router: self.linear_t(&format!("{m}.gate.weight"), e, hidden)?,
            e_score_bias: self.tensor_f32(&format!("{m}.gate.e_score_correction_bias"))?,
            down_latent: self.linear_t(
                &format!("{m}.routed_expert_down_proj.weight"),
                l,
                hidden,
            )?,
            up_latent: self.linear_t(&format!("{m}.routed_expert_up_proj.weight"), hidden, l)?,
            routed_norm: self.tensor_f32(&format!("{m}.routed_expert_norm.weight"))?,
            experts_gate_up: Vec::new(), // paged per token, not resident
            experts_down: Vec::new(),
            shared_gate: self.linear_t(
                &format!("{m}.shared_experts.gate_proj.weight"),
                si,
                hidden,
            )?,
            shared_up: self.linear_t(&format!("{m}.shared_experts.up_proj.weight"), si, hidden)?,
            shared_down: self.linear_t(
                &format!("{m}.shared_experts.down_proj.weight"),
                hidden,
                si,
            )?,
        })
    }

    /// Dequantize ONE routed expert's `w1`(gate)/`w3`(up)/`w2`(down) → the
    /// GroupedMatMul layouts: `(gate_up [L, 2*mi], down [mi, L])`. `w1/w3` are HF
    /// `[mi, L]`, `w2` is `[L, mi]`; all transposed to `[K, N]`.
    pub fn load_expert(
        &mut self,
        layer_prefix: &str,
        e: usize,
        l: usize,
        mi: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let p = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
        let w1 = self.dequant_mxfp4(
            &format!("{p}.w1.weight_packed"),
            &format!("{p}.w1.weight_scale"),
            mi,
            l,
        )?; // [mi,L]
        let w3 = self.dequant_mxfp4(
            &format!("{p}.w3.weight_packed"),
            &format!("{p}.w3.weight_scale"),
            mi,
            l,
        )?; // [mi,L]
        let w2 = self.dequant_mxfp4(
            &format!("{p}.w2.weight_packed"),
            &format!("{p}.w2.weight_scale"),
            l,
            mi,
        )?; // [L,mi]

        // gate_up [L, 2*mi]: row `r` = [w1ᵀ[r], w3ᵀ[r]] — fuse transpose into the
        // assembly, in parallel over the L rows (was 2 single-threaded transposes).
        let mut gate_up = vec![0f32; l * 2 * mi];
        gate_up
            .par_chunks_mut(2 * mi)
            .enumerate()
            .for_each(|(r, row)| {
                for o in 0..mi {
                    row[o] = w1[o * l + r];
                    row[mi + o] = w3[o * l + r];
                }
            });
        // down [mi, L] = w2ᵀ, parallel over the mi rows.
        let mut down = vec![0f32; mi * l];
        down.par_chunks_mut(l).enumerate().for_each(|(m, row)| {
            for (r, cell) in row.iter_mut().enumerate() {
                *cell = w2[r * mi + m];
            }
        });
        Ok((gate_up, down))
    }

    /// Load ONE expert's MXFP4 weights kept **PACKED** (raw `weight_packed` +
    /// `weight_scale` bytes for w1/w3/w2) — NO dequant, NO transpose. Kimi's format
    /// (E2M1 nibbles low-first + E8M0 group-32 scale) is bit-identical to rlx's
    /// [`rlx_ir::quant::QuantScheme::MlxMxfp4`] and `[out, in]` layout matches its
    /// convention, so these feed `DequantGroupedMatMul{MlxMxfp4}` directly — the GPU
    /// does the dequant+matmul, eliminating the CPU dequant that dominates paging.
    pub fn load_expert_packed(&mut self, layer_prefix: &str, e: usize) -> Result<ExpertPacked> {
        let p = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
        let raw_u8 = |ck: &mut Self, t: &str, which: &str| -> Result<Vec<u8>> {
            let name = format!("{p}.{t}.{which}");
            let (dt, _, b) = ck.raw(&name)?;
            if dt != "U8" {
                bail!("{name}: expected U8 (packed MXFP4), got {dt}");
            }
            Ok(b.to_vec())
        };
        Ok(ExpertPacked {
            w1_q: raw_u8(self, "w1", "weight_packed")?, // gate [mi, L]
            w1_s: raw_u8(self, "w1", "weight_scale")?,
            w3_q: raw_u8(self, "w3", "weight_packed")?, // up   [mi, L]
            w3_s: raw_u8(self, "w3", "weight_scale")?,
            w2_q: raw_u8(self, "w2", "weight_packed")?, // down [L, mi]
            w2_s: raw_u8(self, "w2", "weight_scale")?,
        })
    }

    /// Resolve the on-disk `(path,start,end)` ranges for one expert's 6 MXFP4
    /// tensors (w1/w3/w2 × packed/scale) on the main thread (needs shard headers),
    /// so [`load_expert_ranges`] can read+dequant+transpose them CONCURRENTLY off a
    /// rayon worker (different disk offsets → queue depth for the SSD; MoE paging is
    /// the dominant, disk-bound cost).
    pub fn expert_ranges(
        &mut self,
        layer_prefix: &str,
        e: usize,
    ) -> Result<[(PathBuf, usize, usize); 6]> {
        let p = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
        Ok([
            self.byte_range(&format!("{p}.w1.weight_packed"))?,
            self.byte_range(&format!("{p}.w1.weight_scale"))?,
            self.byte_range(&format!("{p}.w3.weight_packed"))?,
            self.byte_range(&format!("{p}.w3.weight_scale"))?,
            self.byte_range(&format!("{p}.w2.weight_packed"))?,
            self.byte_range(&format!("{p}.w2.weight_scale"))?,
        ])
    }

    // ── prefetch (page-cache warming) ──────────────────────────────────────────

    /// Absolute `(path, start, end)` on-disk byte range of a tensor's data, for
    /// out-of-band prefetch. Parses the shard header if needed (cheap — header
    /// only) but does NOT touch the tensor data itself.
    pub fn byte_range(&mut self, name: &str) -> Result<(PathBuf, usize, usize)> {
        let file = self
            .weight_map
            .get(name)
            .with_context(|| format!("{name} not in index"))?
            .clone();
        let path = self.dir.join(&file);
        let sh = self.shard(&file)?;
        let (_, _, a, b) = sh
            .tensors
            .get(name)
            .with_context(|| format!("{name} in shard"))?;
        Ok((path, sh.data_start + *a, sh.data_start + *b))
    }

    /// Checkpoint directory (for spawning a second loader on a producer thread).
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Names of a layer's **resident** (non-expert) tensors — attention, norms,
    /// AttnRes projections, and the MoE router/shared/latent (everything under
    /// `layers.{layer}.` except the paged `block_sparse_moe.experts.*`). This is
    /// exactly the per-layer disk read to prefetch while the previous layer runs.
    pub fn layer_backbone_names(&self, layer: usize) -> Vec<String> {
        let pfx = format!("language_model.model.layers.{layer}.");
        self.weight_map
            .keys()
            .filter(|k| k.starts_with(&pfx) && !k.contains(".block_sparse_moe.experts."))
            .cloned()
            .collect()
    }
}

/// Read (touch) the given `(path, start, end)` byte ranges to pull them into the
/// OS page cache — the file-backed cache is shared with [`CheckpointLoader`]'s
/// mmap, so a subsequent load hits warm pages instead of faulting from disk. Run
/// on a background thread to overlap the next layer's disk read with compute.
/// Read + dequant + transpose one expert from its [`CheckpointLoader::expert_ranges`]
/// (no `self` → runs on a rayon worker; each mmaps its own file offsets so N
/// experts fault-in concurrently). Returns `(gate_up [L,2·mi], down [mi,L])`,
/// bit-identical to [`CheckpointLoader::load_expert`]. `block` = Kimi MXFP4 group 32.
pub fn load_expert_ranges(
    r: &[(PathBuf, usize, usize); 6],
    l: usize,
    mi: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Same pread-not-mmap fix as `load_expert_ranges_packed`: open each DISTINCT shard
    // once and `read_exact_at` the contiguous range (one sequential I/O with kernel
    // readahead), instead of 6× File::open + Mmap of the whole ~16 GB shard (cold 4 KB
    // page-faults through a huge mapping — the f32 path's real cost). Bit-identical bytes.
    use std::os::unix::fs::FileExt;
    let mut files: HashMap<&std::path::Path, File> = HashMap::new();
    for (path, _, _) in r {
        if !files.contains_key(path.as_path()) {
            files.insert(
                path.as_path(),
                File::open(path).with_context(|| format!("open {path:?}"))?,
            );
        }
    }
    let rd = |i: usize| -> Result<Vec<u8>> {
        let (path, a, b) = &r[i];
        let mut buf = vec![0u8; b - a];
        files[path.as_path()].read_exact_at(&mut buf, *a as u64)?;
        Ok(buf)
    };
    let w1 = dequant_mxfp4_bytes(&rd(0)?, &rd(1)?, mi, l, 32); // [mi,L]
    let w3 = dequant_mxfp4_bytes(&rd(2)?, &rd(3)?, mi, l, 32); // [mi,L]
    let w2 = dequant_mxfp4_bytes(&rd(4)?, &rd(5)?, l, mi, 32); // [L,mi]
    // gate_up [L,2mi]: row r = [w1ᵀ[r], w3ᵀ[r]]; down [mi,L] = w2ᵀ (same as load_expert)
    let mut gate_up = vec![0f32; l * 2 * mi];
    gate_up
        .par_chunks_mut(2 * mi)
        .enumerate()
        .for_each(|(row, cells)| {
            for o in 0..mi {
                cells[o] = w1[o * l + row];
                cells[mi + o] = w3[o * l + row];
            }
        });
    let mut down = vec![0f32; mi * l];
    down.par_chunks_mut(l).enumerate().for_each(|(m, cells)| {
        for (r_i, cell) in cells.iter_mut().enumerate() {
            *cell = w2[r_i * mi + m];
        }
    });
    Ok((gate_up, down))
}

/// Read one expert's 6 RAW MXFP4 byte ranges (packed codes + E8M0 scales) with NO
/// dequant and NO transpose — the fused `DequantGroupedMatMulMlx` consumes them
/// packed. No `self` → runs on a rayon worker (concurrent fault-in) exactly like
/// [`load_expert_ranges`], but moves ~4× less data and does zero CPU dequant.
/// Range order matches [`CheckpointLoader::expert_ranges`]:
/// `[w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale]`.
pub fn load_expert_ranges_packed(r: &[(PathBuf, usize, usize); 6]) -> Result<ExpertPacked> {
    use std::os::unix::fs::FileExt;
    // The 6 tensors sit in 1–2 shards; open each DISTINCT shard once (was: 6×
    // File::open + Mmap of the whole ~16 GB shard) and `pread` each CONTIGUOUS range
    // (one sequential I/O with kernel readahead, vs cold 4 KB page-faults through a
    // 16 GB mapping). Same zero-materialize idea that gave the backbone ~5.7×.
    let mut files: HashMap<&std::path::Path, File> = HashMap::new();
    for (path, _, _) in r {
        if !files.contains_key(path.as_path()) {
            files.insert(
                path.as_path(),
                File::open(path).with_context(|| format!("open {path:?}"))?,
            );
        }
    }
    let rd = |i: usize| -> Result<Vec<u8>> {
        let (path, a, b) = &r[i];
        let mut buf = vec![0u8; b - a];
        files[path.as_path()].read_exact_at(&mut buf, *a as u64)?;
        Ok(buf)
    };
    Ok(ExpertPacked {
        w1_q: rd(0)?,
        w1_s: rd(1)?,
        w3_q: rd(2)?,
        w3_s: rd(3)?,
        w2_q: rd(4)?,
        w2_s: rd(5)?,
    })
}

pub fn warm_ranges(ranges: &[(PathBuf, usize, usize)]) {
    use std::io::{Read, Seek, SeekFrom};
    let mut scratch = vec![0u8; 8 << 20]; // 8 MiB streaming buffer
    for (path, a, b) in ranges {
        let Ok(mut f) = File::open(path) else {
            continue;
        };
        if f.seek(SeekFrom::Start(*a as u64)).is_err() {
            continue;
        }
        let mut left = b.saturating_sub(*a);
        while left > 0 {
            let n = left.min(scratch.len());
            match f.read(&mut scratch[..n]) {
                Ok(0) | Err(_) => break,
                Ok(k) => left -= k,
            }
        }
    }
}
