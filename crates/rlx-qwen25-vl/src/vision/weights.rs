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

//! Qwen2.5-VL mmproj weight loader — llama.cpp tensor names from `qwen2vl.cpp`.

use super::config::MmProjConfig;
use anyhow::{Context, Result, anyhow};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};

#[derive(Debug, Clone)]
pub struct VisionBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Option<Vec<f32>>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Option<Vec<f32>>,
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
}

#[derive(Debug, Clone)]
pub struct MmProjWeights {
    pub patch_embd_0: Vec<f32>,
    pub patch_embd_1: Vec<f32>,
    pub patch_bias: Vec<f32>,
    pub position_embd: Vec<f32>,
    pub pre_ln_w: Vec<f32>,
    pub pre_ln_b: Option<Vec<f32>>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Option<Vec<f32>>,
    pub mm_0_w: Vec<f32>,
    pub mm_0_b: Vec<f32>,
    pub mm_1_w: Vec<f32>,
    pub mm_1_b: Vec<f32>,
    pub blocks: Vec<VisionBlockWeights>,
}

pub fn load_vision_weights(path: &str) -> Result<(MmProjConfig, MmProjWeights)> {
    let mut loader = GgufLoader::from_file(path)?;
    let cfg = MmProjConfig::from_gguf(loader.file())?;
    crate::config::validate_mmproj_config(&cfg)?;
    let w = MmProjWeights::from_loader(&cfg, &mut loader)?;
    Ok((cfg, w))
}

impl MmProjWeights {
    pub fn from_loader(cfg: &MmProjConfig, loader: &mut GgufLoader) -> Result<Self> {
        if loader.file().tensors.contains_key("v.blk.0.attn_q.weight") {
            Self::from_loader_mtmd(cfg, loader)
        } else {
            Self::from_loader_legacy(cfg, loader)
        }
    }

    fn from_loader_legacy(cfg: &MmProjConfig, loader: &mut GgufLoader) -> Result<Self> {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff.max(n * 4);
        let ps = cfg.patch_size;
        let patch_elems = n * 3 * ps * ps;

        let patch_embd_0 = take_tensor(loader, "v.patch_embd.weight")?;
        let patch_embd_1 = take_tensor(loader, "v.patch_embd.weight.1")?;
        let patch_bias = take_tensor_or_zeros(loader, "v.patch_embd.bias", n)?;
        let position_embd = take_tensor(loader, "vision.position_embd.weight")?;
        check_len("v.patch_embd.weight", &patch_embd_0, patch_elems)?;
        check_len("v.patch_embd.weight.1", &patch_embd_1, patch_elems)?;
        check_len("v.patch_embd.bias", &patch_bias, n)?;

        let (pre_ln_w, pre_ln_b) = load_norm(loader, "vision.pre_ln", n, cfg.use_rms_norm)?;
        let (post_ln_w, post_ln_b) = load_norm(loader, "vision.post_ln", n, cfg.use_rms_norm)?;

        let merge_sq = cfg.n_merge * cfg.n_merge;
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

            let (ln1_w, ln1_b) = load_norm(loader, &format!("{p}.ln1"), n, cfg.use_rms_norm)?;
            let (ln2_w, ln2_b) = load_norm(loader, &format!("{p}.ln2"), n, cfg.use_rms_norm)?;

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
            mm_0_w,
            mm_0_b,
            mm_1_w,
            mm_1_b,
            blocks,
        })
    }

    /// llama.cpp / ggml-org mtmd layout (`v.blk.*`, split Q/K/V, `mm.2` merger).
    fn from_loader_mtmd(cfg: &MmProjConfig, loader: &mut GgufLoader) -> Result<Self> {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let mm_ff = cfg.n_ff.max(n * 4);
        let ps = cfg.patch_size;
        let patch_elems = n * 3 * ps * ps;

        let patch_embd_0 = take_tensor(loader, "v.patch_embd.weight")?;
        let patch_embd_1 = take_tensor(loader, "v.patch_embd.weight.1")?;
        let patch_bias = take_tensor_or_zeros(loader, "v.patch_embd.bias", n)?;
        check_len("v.patch_embd.weight", &patch_embd_0, patch_elems)?;
        check_len("v.patch_embd.weight.1", &patch_embd_1, patch_elems)?;

        let pre_ln_w = Vec::new();
        let pre_ln_b = None;
        let post_ln_w = take_tensor(loader, "v.post_ln.weight")?;
        check_len("v.post_ln.weight", &post_ln_w, n)?;
        let post_ln_b = None;

        let merge_sq = cfg.n_merge * cfg.n_merge;
        let mm_in = n * merge_sq;
        let mm_0_w = take_tensor(loader, "mm.0.weight")?;
        let mm_0_b = take_tensor(loader, "mm.0.bias")?;
        let mm_1_w = take_tensor(loader, "mm.2.weight")?;
        let mm_1_b = take_tensor(loader, "mm.2.bias")?;
        check_len("mm.0.weight", &mm_0_w, mm_ff * mm_in)?;
        check_len("mm.0.bias", &mm_0_b, mm_ff)?;
        check_len("mm.2.weight", &mm_1_w, cfg.llm_hidden_size * mm_ff)?;
        check_len("mm.2.bias", &mm_1_b, cfg.llm_hidden_size)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("v.blk.{il}");
            let q_w = take_tensor(loader, &format!("{p}.attn_q.weight"))?;
            let k_w = take_tensor(loader, &format!("{p}.attn_k.weight"))?;
            let v_w = take_tensor(loader, &format!("{p}.attn_v.weight"))?;
            let q_b = take_tensor(loader, &format!("{p}.attn_q.bias"))?;
            let k_b = take_tensor(loader, &format!("{p}.attn_k.bias"))?;
            let v_b = take_tensor(loader, &format!("{p}.attn_v.bias"))?;
            let qkv_w = stack_qkv(q_w, k_w, v_w, n * n)?;
            let qkv_b = stack_qkv(q_b, k_b, v_b, n)?;
            let attn_out_w = take_tensor(loader, &format!("{p}.attn_out.weight"))?;
            let attn_out_b = take_tensor(loader, &format!("{p}.attn_out.bias"))?;
            check_len(&format!("{p}.attn_out.weight"), &attn_out_w, n * n)?;
            check_len(&format!("{p}.attn_out.bias"), &attn_out_b, n)?;

            let ln1_w = take_tensor(loader, &format!("{p}.ln1.weight"))?;
            check_len(&format!("{p}.ln1.weight"), &ln1_w, n)?;
            let ln2_w = take_tensor(loader, &format!("{p}.ln2.weight"))?;
            check_len(&format!("{p}.ln2.weight"), &ln2_w, n)?;

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

            blocks.push(VisionBlockWeights {
                ln1_w,
                ln1_b: None,
                ln2_w,
                ln2_b: None,
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
            });
        }

        Ok(Self {
            patch_embd_0,
            patch_embd_1,
            patch_bias,
            position_embd: vec![0.0; n],
            pre_ln_w,
            pre_ln_b,
            post_ln_w,
            post_ln_b,
            mm_0_w,
            mm_0_b,
            mm_1_w,
            mm_1_b,
            blocks,
        })
    }

    /// Synthetic weights for unit / quick-check tests.
    pub fn synthetic(cfg: &MmProjConfig) -> Self {
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff.max(n * 4);
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
            pre_ln_b: None,
            post_ln_w: ones(n),
            post_ln_b: None,
            mm_0_w: ramp(n_ff * n * merge_sq, 0.01),
            mm_0_b: ramp(n_ff, 0.02),
            mm_1_w: ramp(cfg.llm_hidden_size * n_ff, 0.03),
            mm_1_b: ramp(cfg.llm_hidden_size, 0.04),
            blocks: (0..cfg.n_layer)
                .map(|_| VisionBlockWeights {
                    ln1_w: ones(n),
                    ln1_b: None,
                    ln2_w: ones(n),
                    ln2_b: None,
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
                })
                .collect(),
        }
    }
}

fn load_norm(
    loader: &mut GgufLoader,
    prefix: &str,
    n: usize,
    rms: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let w = take_tensor(loader, &format!("{prefix}.weight"))?;
    check_len(&format!("{prefix}.weight"), &w, n)?;
    let b = if rms {
        None
    } else {
        let bias = take_tensor(loader, &format!("{prefix}.bias"))?;
        check_len(&format!("{prefix}.bias"), &bias, n)?;
        Some(bias)
    };
    Ok((w, b))
}

fn take_tensor(loader: &mut GgufLoader, key: &str) -> Result<Vec<f32>> {
    loader
        .take(key)
        .with_context(|| format!("missing mmproj tensor {key}"))
        .map(|(v, _)| v)
}

fn take_tensor_or_zeros(loader: &mut GgufLoader, key: &str, n: usize) -> Result<Vec<f32>> {
    match loader.take(key) {
        Ok((v, _)) => {
            check_len(key, &v, n)?;
            Ok(v)
        }
        Err(_) => Ok(vec![0.0; n]),
    }
}

fn stack_qkv(a: Vec<f32>, b: Vec<f32>, c: Vec<f32>, row_width: usize) -> Result<Vec<f32>> {
    check_len("qkv_a", &a, row_width)?;
    check_len("qkv_b", &b, row_width)?;
    check_len("qkv_c", &c, row_width)?;
    let mut out = a;
    out.extend(b);
    out.extend(c);
    Ok(out)
}

fn check_len(name: &str, data: &[f32], expected: usize) -> Result<()> {
    if data.len() != expected {
        return Err(anyhow!("{name}: len {} != expected {expected}", data.len()));
    }
    Ok(())
}
