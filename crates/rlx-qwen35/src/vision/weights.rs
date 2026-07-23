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

//! Qwen3.5 VLM mmproj weight loader — llama.cpp tensor names from
//! `tools/mtmd/clip.cpp` / `clip-impl.h`.

use super::config::MmProjConfig;
use anyhow::{anyhow, ensure, Context, Result};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_gguf::GgufFile;

/// One transformer block in the Qwen3-VL vision tower.
#[derive(Debug, Clone)]
pub struct VisionBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub qkv_w: Vec<f32>,
    pub qkv_b: Vec<f32>,
    pub attn_out_w: Vec<f32>,
    pub attn_out_b: Vec<f32>,
    pub ffn_gate_w: Vec<f32>,
    pub ffn_gate_b: Vec<f32>,
    pub ffn_up_w: Vec<f32>,
    pub ffn_up_b: Vec<f32>,
    pub ffn_down_w: Vec<f32>,
    pub ffn_down_b: Vec<f32>,
    /// When false, MLP is `down(act(fc1(x)))` (HF Qwen3.5). When true,
    /// SwiGLU-style `down(act(gate(x)) * up(x))` (GGUF / llama.cpp).
    pub ffn_gated: bool,
    pub deepstack: Option<DeepstackWeights>,
}

#[derive(Debug, Clone)]
pub struct DeepstackWeights {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

/// Full mmproj weight bundle for [`super::builder::build_qwen35_vision_hir`].
#[derive(Debug, Clone)]
pub struct MmProjWeights {
    pub patch_embd_0: Vec<f32>,
    pub patch_embd_1: Vec<f32>,
    pub patch_bias: Vec<f32>,
    pub position_embd: Vec<f32>,
    pub pre_ln_w: Vec<f32>,
    pub pre_ln_b: Vec<f32>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    /// When false, skip pre-LN (HF Qwen3.5 vision has none — ones/zeros
    /// would still normalize and break features).
    pub has_pre_ln: bool,
    /// When false, skip post-LN (same as [`Self::has_pre_ln`]).
    pub has_post_ln: bool,
    pub mm_norm_w: Vec<f32>,
    pub mm_norm_b: Vec<f32>,
    pub mm_0_w: Vec<f32>,
    pub mm_0_b: Vec<f32>,
    pub mm_1_w: Vec<f32>,
    pub mm_1_b: Vec<f32>,
    pub blocks: Vec<VisionBlockWeights>,
}

impl MmProjWeights {
    /// Load from an on-disk mmproj GGUF via [`GgufLoader`].
    pub fn from_gguf_path(path: &str) -> Result<(MmProjConfig, Self)> {
        let mut loader = GgufLoader::from_file(path)?;
        let cfg = MmProjConfig::from_gguf(loader.file())?;
        let w = Self::from_loader(&cfg, &mut loader)?;
        Ok((cfg, w))
    }

    pub fn from_gguf_loader(
        raw: &GgufFile,
        loader: &mut GgufLoader,
    ) -> Result<(MmProjConfig, Self)> {
        let cfg = MmProjConfig::from_gguf(raw)?;
        let w = Self::from_loader(&cfg, loader)?;
        Ok((cfg, w))
    }

    pub fn from_loader(cfg: &MmProjConfig, loader: &mut GgufLoader) -> Result<Self> {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let ps = cfg.patch_size;
        let patch_out = n;
        let patch_in = 3;
        let patch_elems = patch_out * patch_in * ps * ps;

        let patch_embd_0 = take_tensor(loader, "v.patch_embd.weight")?;
        let patch_embd_1 = take_tensor(loader, "v.patch_embd.weight.1")?;
        let patch_bias = take_tensor(loader, "v.patch_embd.bias")?;
        let position_embd = take_tensor(loader, "vision.position_embd.weight")?;

        check_len("v.patch_embd.weight", &patch_embd_0, patch_elems)?;
        check_len("v.patch_embd.weight.1", &patch_embd_1, patch_elems)?;
        check_len("v.patch_embd.bias", &patch_bias, n)?;
        check_len("vision.position_embd.weight", &position_embd, n * n)?;

        let pre_ln_w = take_tensor(loader, "vision.pre_ln.weight")?;
        let pre_ln_b = take_tensor(loader, "vision.pre_ln.bias")?;
        let post_ln_w = take_tensor(loader, "vision.post_ln.weight")?;
        let post_ln_b = take_tensor(loader, "vision.post_ln.bias")?;
        check_len("vision.pre_ln.weight", &pre_ln_w, n)?;
        check_len("vision.pre_ln.bias", &pre_ln_b, n)?;
        check_len("vision.post_ln.weight", &post_ln_w, n)?;
        check_len("vision.post_ln.bias", &post_ln_b, n)?;
        // Older mmproj exports omit the merger norm. It is identity in that
        // layout, while HF Fara checkpoints provide it explicitly.
        let mm_norm_w = vec![1.0; n];
        let mm_norm_b = vec![0.0; n];

        let merge = cfg.n_merge;
        let merge_sq = merge * merge;
        let mm_in = n * merge_sq;
        let mm_0_w = take_tensor(loader, "mm.0.weight")?;
        let mm_0_b = take_tensor(loader, "mm.0.bias")?;
        let mm_1_w = take_tensor(loader, "mm.1.weight")?;
        let mm_1_b = take_tensor(loader, "mm.1.bias")?;
        check_len("mm.0.weight", &mm_0_w, n_ff * mm_in)?;
        check_len("mm.0.bias", &mm_0_b, n_ff)?;
        check_len("mm.1.weight", &mm_1_w, cfg.llm_hidden_size * n_ff)?;
        check_len("mm.1.bias", &mm_1_b, cfg.llm_hidden_size)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("vision.blk.{il}");
            let qkv_w = take_tensor(loader, &format!("{p}.attn_qkv.weight"))?;
            let qkv_b = take_tensor(loader, &format!("{p}.attn_qkv.bias"))?;
            let attn_out_w = take_tensor(loader, &format!("{p}.attn_out.weight"))?;
            let attn_out_b = take_tensor(loader, &format!("{p}.attn_out.bias"))?;
            check_len(&format!("{p}.attn_qkv.weight"), &qkv_w, 3 * n * n)?;
            check_len(&format!("{p}.attn_qkv.bias"), &qkv_b, 3 * n)?;
            check_len(&format!("{p}.attn_out.weight"), &attn_out_w, n * n)?;
            check_len(&format!("{p}.attn_out.bias"), &attn_out_b, n)?;

            let ffn_gate_w = take_tensor(loader, &format!("{p}.ffn_gate.weight"))?;
            let ffn_gate_b = take_tensor(loader, &format!("{p}.ffn_gate.bias"))?;
            let ffn_up_w = take_tensor(loader, &format!("{p}.ffn_up.weight"))?;
            let ffn_up_b = take_tensor(loader, &format!("{p}.ffn_up.bias"))?;
            let ffn_down_w = take_tensor(loader, &format!("{p}.ffn_down.weight"))?;
            let ffn_down_b = take_tensor(loader, &format!("{p}.ffn_down.bias"))?;
            check_len(&format!("{p}.ffn_gate.weight"), &ffn_gate_w, n_ff * n)?;
            check_len(&format!("{p}.ffn_gate.bias"), &ffn_gate_b, n_ff)?;
            check_len(&format!("{p}.ffn_up.weight"), &ffn_up_w, n_ff * n)?;
            check_len(&format!("{p}.ffn_up.bias"), &ffn_up_b, n_ff)?;
            check_len(&format!("{p}.ffn_down.weight"), &ffn_down_w, n * n_ff)?;
            check_len(&format!("{p}.ffn_down.bias"), &ffn_down_b, n)?;

            let deepstack = if cfg.deepstack_layers.contains(&il) {
                let ds = format!("v.deepstack.{il}");
                Some(DeepstackWeights {
                    norm_w: take_tensor(loader, &format!("{ds}.norm.weight"))?,
                    norm_b: take_tensor(loader, &format!("{ds}.norm.bias"))?,
                    fc1_w: take_tensor(loader, &format!("{ds}.fc1.weight"))?,
                    fc1_b: take_tensor(loader, &format!("{ds}.fc1.bias"))?,
                    fc2_w: take_tensor(loader, &format!("{ds}.fc2.weight"))?,
                    fc2_b: take_tensor(loader, &format!("{ds}.fc2.bias"))?,
                })
            } else {
                None
            };

            blocks.push(VisionBlockWeights {
                ln1_w: take_tensor(loader, &format!("{p}.ln1.weight"))?,
                ln1_b: take_tensor(loader, &format!("{p}.ln1.bias"))?,
                ln2_w: take_tensor(loader, &format!("{p}.ln2.weight"))?,
                ln2_b: take_tensor(loader, &format!("{p}.ln2.bias"))?,
                qkv_w,
                qkv_b,
                attn_out_w,
                attn_out_b,
                ffn_gate_w,
                ffn_gate_b,
                ffn_up_w,
                ffn_up_b,
                ffn_down_w,
                ffn_down_b,
                ffn_gated: true,
                deepstack,
            });
        }

        Ok(Self {
            patch_embd_0,
            patch_embd_1,
            patch_bias,
            position_embd,
            pre_ln_w,
            pre_ln_b,
            post_ln_w,
            post_ln_b,
            has_pre_ln: true,
            has_post_ln: true,

            mm_norm_w,
            mm_norm_b,
            mm_0_w,
            mm_0_b,
            mm_1_w,
            mm_1_b,
            blocks,
        })
    }

    /// Drain `model.visual.*` tensors from a HuggingFace safetensors
    /// loader into the mmproj layout expected by the vision HIR.
    ///
    /// Packing notes vs HF Qwen3.5 / Fara:
    /// - Temporal patch embed (`[C_out, 3, T=2, P, P]`) is split into the
    ///   dual `v.patch_embd` kernels.
    /// - Block MLP is non-gated `fc2(gelu_tanh(fc1(x)))` (`ffn_gated = false`).
    /// - Missing pre/post LayerNorm are skipped (`has_pre_ln` / `has_post_ln`).
    pub fn from_hf_visual(
        cfg: &MmProjConfig,
        loader: &mut dyn rlx_core::WeightLoader,
    ) -> Result<Self> {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let ps = cfg.patch_size;
        let merge_sq = cfg.n_merge * cfg.n_merge;

        let (patch_raw, patch_shape) = take_wl(loader, "model.visual.patch_embed.proj.weight")?;
        let (patch_embd_0, patch_embd_1) =
            split_temporal_patch_embed(&patch_raw, &patch_shape, n, ps)?;
        let patch_bias = take_wl(loader, "model.visual.patch_embed.proj.bias")
            .map(|(d, _)| d)
            .unwrap_or_else(|_| vec![0.0; n]);
        check_len("patch_bias", &patch_bias, n)?;

        let (position_embd, _) = take_wl(loader, "model.visual.pos_embed.weight")?;
        // HF pos_embed is often `[1, N, C]` or `[N, C]`; flatten to `[N*C]`.
        if position_embd.len() < n {
            return Err(anyhow!(
                "model.visual.pos_embed.weight: len {} too small for n_embd={n}",
                position_embd.len()
            ));
        }

        let ones = |len: usize| vec![1.0f32; len];
        let zeros = |len: usize| vec![0.0f32; len];
        let (pre_ln_w, pre_ln_b, has_pre_ln) = take_ln_pair(
            loader,
            "model.visual.pre_layernorm",
            "model.visual.pre_ln",
            n,
            &ones,
            &zeros,
        )?;
        let (post_ln_w, post_ln_b, has_post_ln) = take_ln_pair(
            loader,
            "model.visual.post_layernorm",
            "model.visual.post_ln",
            n,
            &ones,
            &zeros,
        )?;

        // HF applies this LayerNorm before spatial reshape and merger MLP.
        let (mm_norm_w, mm_norm_b, _) = take_ln_pair(
            loader,
            "model.visual.merger.norm",
            "model.visual.merger.norm",
            n,
            &ones,
            &zeros,
        )?;
        let (mm_0_w, _) = take_wl(loader, "model.visual.merger.linear_fc1.weight")?;
        let (mm_0_b, _) = take_wl(loader, "model.visual.merger.linear_fc1.bias")?;
        let (mm_1_w, _) = take_wl(loader, "model.visual.merger.linear_fc2.weight")?;
        let (mm_1_b, _) = take_wl(loader, "model.visual.merger.linear_fc2.bias")?;
        check_len("mm.0.weight", &mm_0_w, n_ff * n * merge_sq)?;
        check_len("mm.0.bias", &mm_0_b, n_ff)?;
        check_len("mm.1.weight", &mm_1_w, cfg.llm_hidden_size * n_ff)?;
        check_len("mm.1.bias", &mm_1_b, cfg.llm_hidden_size)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("model.visual.blocks.{il}");
            let (qkv_w, _) = take_wl(loader, &format!("{p}.attn.qkv.weight"))?;
            let (qkv_b, _) = take_wl(loader, &format!("{p}.attn.qkv.bias"))?;
            let (attn_out_w, _) = take_wl(loader, &format!("{p}.attn.proj.weight"))?;
            let (attn_out_b, _) = take_wl(loader, &format!("{p}.attn.proj.bias"))?;
            check_len(&format!("{p}.attn.qkv.weight"), &qkv_w, 3 * n * n)?;
            check_len(&format!("{p}.attn.qkv.bias"), &qkv_b, 3 * n)?;
            check_len(&format!("{p}.attn.proj.weight"), &attn_out_w, n * n)?;
            check_len(&format!("{p}.attn.proj.bias"), &attn_out_b, n)?;

            let (ln1_w, ln1_b, _) = take_ln_pair(
                loader,
                &format!("{p}.norm1"),
                &format!("{p}.ln1"),
                n,
                &ones,
                &zeros,
            )?;
            let (ln2_w, ln2_b, _) = take_ln_pair(
                loader,
                &format!("{p}.norm2"),
                &format!("{p}.ln2"),
                n,
                &ones,
                &zeros,
            )?;

            let (ffn_gate_w, _) = take_wl(loader, &format!("{p}.mlp.linear_fc1.weight"))?;
            let (ffn_gate_b, _) = take_wl(loader, &format!("{p}.mlp.linear_fc1.bias"))?;
            let (ffn_down_w, _) = take_wl(loader, &format!("{p}.mlp.linear_fc2.weight"))?;
            let (ffn_down_b, _) = take_wl(loader, &format!("{p}.mlp.linear_fc2.bias"))?;
            check_len(&format!("{p}.mlp.fc1.weight"), &ffn_gate_w, n_ff * n)?;
            check_len(&format!("{p}.mlp.fc1.bias"), &ffn_gate_b, n_ff)?;
            check_len(&format!("{p}.mlp.fc2.weight"), &ffn_down_w, n * n_ff)?;
            check_len(&format!("{p}.mlp.fc2.bias"), &ffn_down_b, n)?;
            // HF Vision MLP is `fc2(act(fc1(x)))`, not gated SwiGLU.
            // Leave up empty; builder skips the mul when `ffn_gated` is false.
            let ffn_up_w = Vec::new();
            let ffn_up_b = Vec::new();

            blocks.push(VisionBlockWeights {
                ln1_w,
                ln1_b,
                ln2_w,
                ln2_b,
                qkv_w,
                qkv_b,
                attn_out_w,
                attn_out_b,
                ffn_gate_w,
                ffn_gate_b,
                ffn_up_w,
                ffn_up_b,
                ffn_down_w,
                ffn_down_b,
                ffn_gated: false,
                deepstack: None,
            });
        }

        Ok(Self {
            patch_embd_0,
            patch_embd_1,
            patch_bias,
            position_embd,
            pre_ln_w,
            pre_ln_b,
            post_ln_w,
            post_ln_b,
            has_pre_ln,
            has_post_ln,
            mm_norm_w,
            mm_norm_b,
            mm_0_w,
            mm_0_b,
            mm_1_w,
            mm_1_b,
            blocks,
        })
    }

    /// Synthetic weights for unit tests (tiny 1-layer tower).
    pub fn synthetic(cfg: &MmProjConfig) -> Self {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let ps = cfg.patch_size;
        let merge_sq = cfg.n_merge * cfg.n_merge;
        let ramp = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| 0.001 + scale * (i as f32) * 0.01)
                .collect()
        };
        let ones = |len: usize| vec![1.0f32; len];
        Self {
            patch_embd_0: ramp(3 * n * ps * ps, 0.001),
            patch_embd_1: ramp(3 * n * ps * ps, 0.002),
            patch_bias: ramp(n, 0.003),
            position_embd: ramp(n * n, 0.004),
            pre_ln_w: ones(n),
            pre_ln_b: vec![0.0; n],
            post_ln_w: ones(n),
            post_ln_b: vec![0.0; n],
            has_pre_ln: true,
            has_post_ln: true,
            mm_norm_w: ones(n),
            mm_norm_b: vec![0.0; n],
            mm_0_w: ramp(n_ff * n * merge_sq, 0.01),
            mm_0_b: ramp(n_ff, 0.02),
            mm_1_w: ramp(cfg.llm_hidden_size * n_ff, 0.03),
            mm_1_b: ramp(cfg.llm_hidden_size, 0.04),
            blocks: (0..cfg.n_layer)
                .map(|_| VisionBlockWeights {
                    ln1_w: ones(n),
                    ln1_b: vec![0.0; n],
                    ln2_w: ones(n),
                    ln2_b: vec![0.0; n],
                    qkv_w: ramp(3 * n * n, 0.005),
                    qkv_b: ramp(3 * n, 0.006),
                    attn_out_w: ramp(n * n, 0.007),
                    attn_out_b: ramp(n, 0.008),
                    ffn_gate_w: ramp(n_ff * n, 0.009),
                    ffn_gate_b: ramp(n_ff, 0.01),
                    ffn_up_w: ramp(n_ff * n, 0.011),
                    ffn_up_b: ramp(n_ff, 0.012),
                    ffn_down_w: ramp(n * n_ff, 0.013),
                    ffn_down_b: ramp(n, 0.014),
                    ffn_gated: true,
                    deepstack: None,
                })
                .collect(),
        }
    }
}

fn take_tensor(loader: &mut GgufLoader, key: &str) -> Result<Vec<f32>> {
    loader
        .take(key)
        .with_context(|| format!("mmproj weight `{key}`"))
        .map(|(data, _shape)| data)
}

fn take_wl(loader: &mut dyn rlx_core::WeightLoader, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    loader
        .take(key)
        .with_context(|| format!("HF vision weight `{key}`"))
}

fn take_ln_pair(
    loader: &mut dyn rlx_core::WeightLoader,
    primary: &str,
    alt: &str,
    n: usize,
    ones: &dyn Fn(usize) -> Vec<f32>,
    zeros: &dyn Fn(usize) -> Vec<f32>,
) -> Result<(Vec<f32>, Vec<f32>, bool)> {
    let found_w = take_wl(loader, &format!("{primary}.weight"))
        .or_else(|_| take_wl(loader, &format!("{alt}.weight")));
    let found = found_w.is_ok();
    let w = found_w.map(|(d, _)| d).unwrap_or_else(|_| ones(n));
    let b = take_wl(loader, &format!("{primary}.bias"))
        .or_else(|_| take_wl(loader, &format!("{alt}.bias")))
        .map(|(d, _)| d)
        .unwrap_or_else(|_| zeros(n));
    check_len(&format!("{primary}.weight"), &w, n)?;
    check_len(&format!("{primary}.bias"), &b, n)?;
    Ok((w, b, found))
}

/// Split HF 3-D temporal patch embed `[out, in, T, P, P]` (or already-flat
/// dual-kernel packing) into the two spatial kernels expected by mmproj.
fn split_temporal_patch_embed(
    data: &[f32],
    shape: &[usize],
    n_out: usize,
    patch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let spatial = 3 * patch_size * patch_size;
    let one = n_out * spatial;
    // Common HF layout: [out, 3, temporal=2, P, P].
    if shape.len() == 5 && shape[0] == n_out && shape[1] == 3 && shape[2] == 2 {
        let p = shape[3];
        ensure!(
            p == patch_size && shape[4] == patch_size,
            "patch size mismatch: shape {shape:?} vs patch_size={patch_size}"
        );
        let mut a = Vec::with_capacity(one);
        let mut b = Vec::with_capacity(one);
        for o in 0..n_out {
            for c in 0..3 {
                for t in 0..2 {
                    let base = (((o * 3 + c) * 2 + t) * patch_size) * patch_size;
                    let slice = &data[base..base + patch_size * patch_size];
                    if t == 0 {
                        a.extend_from_slice(slice);
                    } else {
                        b.extend_from_slice(slice);
                    }
                }
            }
        }
        return Ok((a, b));
    }
    if data.len() == 2 * one {
        // Already packed as concat(kernel0, kernel1) in [out,3,P,P] order.
        return Ok((data[..one].to_vec(), data[one..].to_vec()));
    }
    if data.len() == one {
        return Ok((data.to_vec(), data.to_vec()));
    }
    Err(anyhow!(
        "model.visual.patch_embed.proj.weight: unexpected len {} shape {shape:?} \
         (want 2*{one} or 5-D temporal)",
        data.len()
    ))
}

fn check_len(name: &str, data: &[f32], expected: usize) -> Result<()> {
    if data.len() != expected {
        return Err(anyhow!("{name}: len {} != expected {expected}", data.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn tiny_vision_cfg() -> MmProjConfig {
        MmProjConfig {
            patch_size: 2,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            image_size: 4,
            image_min_pixels: 16,
            image_max_pixels: 256,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen3vl".into(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            spatial_merge_size: 2,
            llm_hidden_size: 16,
            n_ff: 16,
            deepstack_layers: vec![],
        }
    }

    fn put(
        map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
        key: &str,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) {
        map.insert(key.to_string(), (data, shape));
    }

    #[test]
    fn split_temporal_5d_patch_embed() {
        let n = 2;
        let ps = 2;
        let mut data = Vec::new();
        // [out=2, in=3, T=2, P=2, P=2]
        for o in 0..n {
            for c in 0..3 {
                for t in 0..2 {
                    for _ in 0..(ps * ps) {
                        data.push((o * 100 + c * 10 + t) as f32);
                    }
                }
            }
        }
        let shape = vec![2, 3, 2, 2, 2];
        let (a, b) = split_temporal_patch_embed(&data, &shape, n, ps).unwrap();
        assert_eq!(a.len(), n * 3 * ps * ps);
        assert_eq!(b.len(), a.len());
        // First spatial of out0/ch0 comes from t=0 → value 0; t=1 → value 1.
        assert_eq!(a[0], 0.0);
        assert_eq!(b[0], 1.0);
    }

    #[test]
    fn from_hf_visual_drains_tiny_map() {
        let cfg = tiny_vision_cfg();
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let ps = cfg.patch_size;
        let merge_sq = cfg.n_merge * cfg.n_merge;
        let mut tensors = HashMap::new();
        // Temporal patch: [n, 3, 2, ps, ps]
        let patch_len = n * 3 * 2 * ps * ps;
        put(
            &mut tensors,
            "model.visual.patch_embed.proj.weight",
            vec![0.01; patch_len],
            vec![n, 3, 2, ps, ps],
        );
        put(
            &mut tensors,
            "model.visual.patch_embed.proj.bias",
            vec![0.0; n],
            vec![n],
        );
        put(
            &mut tensors,
            "model.visual.pos_embed.weight",
            vec![0.02; 4 * n],
            vec![1, 4, n],
        );
        put(
            &mut tensors,
            "model.visual.merger.linear_fc1.weight",
            vec![0.03; n_ff * n * merge_sq],
            vec![n_ff, n * merge_sq],
        );
        put(
            &mut tensors,
            "model.visual.merger.norm.weight",
            vec![1.0; n],
            vec![n],
        );
        put(
            &mut tensors,
            "model.visual.merger.norm.bias",
            vec![0.0; n],
            vec![n],
        );
        put(
            &mut tensors,
            "model.visual.merger.linear_fc1.bias",
            vec![0.0; n_ff],
            vec![n_ff],
        );
        put(
            &mut tensors,
            "model.visual.merger.linear_fc2.weight",
            vec![0.04; cfg.llm_hidden_size * n_ff],
            vec![cfg.llm_hidden_size, n_ff],
        );
        put(
            &mut tensors,
            "model.visual.merger.linear_fc2.bias",
            vec![0.0; cfg.llm_hidden_size],
            vec![cfg.llm_hidden_size],
        );
        let p = "model.visual.blocks.0";
        put(
            &mut tensors,
            &format!("{p}.attn.qkv.weight"),
            vec![0.05; 3 * n * n],
            vec![3 * n, n],
        );
        put(
            &mut tensors,
            &format!("{p}.attn.qkv.bias"),
            vec![0.0; 3 * n],
            vec![3 * n],
        );
        put(
            &mut tensors,
            &format!("{p}.attn.proj.weight"),
            vec![0.06; n * n],
            vec![n, n],
        );
        put(
            &mut tensors,
            &format!("{p}.attn.proj.bias"),
            vec![0.0; n],
            vec![n],
        );
        put(
            &mut tensors,
            &format!("{p}.mlp.linear_fc1.weight"),
            vec![0.07; n_ff * n],
            vec![n_ff, n],
        );
        put(
            &mut tensors,
            &format!("{p}.mlp.linear_fc1.bias"),
            vec![0.0; n_ff],
            vec![n_ff],
        );
        put(
            &mut tensors,
            &format!("{p}.mlp.linear_fc2.weight"),
            vec![0.08; n * n_ff],
            vec![n, n_ff],
        );
        put(
            &mut tensors,
            &format!("{p}.mlp.linear_fc2.bias"),
            vec![0.0; n],
            vec![n],
        );
        for norm in ["norm1", "norm2"] {
            put(
                &mut tensors,
                &format!("{p}.{norm}.weight"),
                vec![1.0; n],
                vec![n],
            );
            put(
                &mut tensors,
                &format!("{p}.{norm}.bias"),
                vec![0.0; n],
                vec![n],
            );
        }

        let mut map = WeightMap::from_tensors(tensors);
        let w = MmProjWeights::from_hf_visual(&cfg, &mut map).expect("from_hf_visual");
        assert_eq!(w.blocks.len(), 1);
        assert_eq!(w.patch_embd_0.len(), n * 3 * ps * ps);
        assert_eq!(w.patch_embd_1.len(), w.patch_embd_0.len());
        assert_eq!(w.mm_norm_w.len(), n);
        assert_eq!(w.mm_norm_b.len(), n);
        assert_eq!(w.ffn_up_identity_len_via_block(), n_ff * n);
        // Visual keys should be drained.
        assert!(!map.keys().any(|k| k.starts_with("model.visual.")));
    }

    impl MmProjWeights {
        fn ffn_up_identity_len_via_block(&self) -> usize {
            self.blocks[0].ffn_up_w.len()
        }
    }
}
