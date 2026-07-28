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

//! Compiled Trellis DiT torso — AdaLN + SDPA + GatedResidual + NeoX RoPE.
//!
//! Mirrors [`crate::dit_host::dit_forward`] / `ModulatedTransformerCrossBlock`
//! using fused rlx ops. Interleaved host RoPE is matched by de-interleaving
//! Q/K projection rows (+ RMS gammas) at weight bind time and applying the
//! stock NeoX `rope` op (see [`crate::rope::deinterleave_perm`]).

use crate::config::DitConfig;
use crate::dit_host::shared_modulation;
use crate::rope::{self, RopeTables};
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{AdaNormKind, MaskKind};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

const ADA_EPS: f32 = 1e-6;
const QK_RMS_EPS: f32 = 1e-12;

/// Compiled DiT session for a fixed `(n_pos, n_cond)` layout.
pub struct CompiledDit {
    compiled: CompiledGraph,
    pub n_pos: usize,
    pub n_cond: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub head_dim: usize,
    device: Device,
}

impl CompiledDit {
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run one DiT eval. `tokens` / `cond` are channels-last flat buffers;
    /// `t_mod` is `[6·C]` from [`shared_modulation`]; `cos`/`sin` are NeoX
    /// tables `[n_pos · (head_dim/2)]` from [`RopeTables::neox`].
    pub fn forward(
        &mut self,
        tokens: &[f32],
        t_mod: &[f32],
        cond: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>> {
        let c6 = 6 * self.model_channels;
        let half = self.head_dim / 2;
        ensure!(
            tokens.len() == self.n_pos * self.in_channels,
            "tokens len {} != n_pos {} × in_ch {}",
            tokens.len(),
            self.n_pos,
            self.in_channels
        );
        ensure!(t_mod.len() == c6, "t_mod len {} != 6×C {}", t_mod.len(), c6);
        ensure!(
            cond.len().is_multiple_of(self.n_cond) && !cond.is_empty(),
            "cond length {} not divisible by n_cond {}",
            cond.len(),
            self.n_cond
        );
        ensure!(
            cos.len() == self.n_pos * half && sin.len() == self.n_pos * half,
            "rope tables must be [n_pos × half]"
        );

        let outs = self.compiled.run(&[
            ("tokens", tokens),
            ("t_mod", t_mod),
            ("cond", cond),
            ("cos", cos),
            ("sin", sin),
        ]);
        outs.into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("compiled DiT returned no output"))
    }
}

/// Build + compile a Trellis DiT for fixed token / conditioner lengths.
pub fn compile_dit(
    cfg: &DitConfig,
    weights: &WeightMap,
    device: Device,
    n_pos: usize,
    n_cond: usize,
) -> Result<CompiledDit> {
    ensure!(n_pos > 0 && n_cond > 0, "n_pos and n_cond must be > 0");
    ensure!(
        cfg.args.share_mod,
        "compiled DiT requires share_mod (shared 6×C adaLN modulation)"
    );
    ensure!(
        cfg.args.qk_rms_norm && cfg.args.qk_rms_norm_cross,
        "compiled DiT expects qk_rms_norm"
    );

    let mut wm_copy = clone_weight_map(weights)?;
    // De-interleave Q/K rows (+ biases / gammas) so NeoX rope matches host interleaved RoPE.
    preprocess_rope_weights(&mut wm_copy, cfg)?;

    let built = build_dit_flow(cfg, &mut wm_copy, n_pos, n_cond)?;
    let typed = built.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

    Ok(CompiledDit {
        compiled,
        n_pos,
        n_cond,
        in_channels: cfg.args.in_channels,
        out_channels: cfg.args.out_channels,
        model_channels: cfg.args.model_channels,
        head_dim: cfg.head_dim(),
        device,
    })
}

/// Host timestep modulation + compiled forward (pads nothing).
#[allow(clippy::too_many_arguments)]
pub fn dit_forward_compiled(
    compiled: &mut CompiledDit,
    cfg: &DitConfig,
    wm: &WeightMap,
    tokens: &[f32],
    coords: &[f32],
    n_pos: usize,
    cond: &[f32],
    n_cond: usize,
    t: f32,
) -> Result<Vec<f32>> {
    ensure!(
        n_pos == compiled.n_pos,
        "n_pos {} != compiled {}",
        n_pos,
        compiled.n_pos
    );
    ensure!(
        n_cond == compiled.n_cond,
        "n_cond {} != compiled {}",
        n_cond,
        compiled.n_cond
    );
    let t_mod = shared_modulation(cfg, wm, t)?;
    let tables = RopeTables::neox(coords, n_pos, compiled.head_dim, 3, cfg.args.rope_freq);
    compiled.forward(tokens, &t_mod, cond, &tables.cos, &tables.sin)
}

/// Pad tokens/coords to `compiled.n_pos`, run, then slice the real prefix.
#[allow(clippy::too_many_arguments)]
pub fn dit_forward_compiled_padded(
    compiled: &mut CompiledDit,
    cfg: &DitConfig,
    wm: &WeightMap,
    tokens: &[f32],
    coords: &[f32],
    n_real: usize,
    cond: &[f32],
    n_cond: usize,
    t: f32,
) -> Result<Vec<f32>> {
    ensure!(n_real > 0, "n_real must be > 0");
    ensure!(
        n_real <= compiled.n_pos,
        "n_real {} exceeds compiled bucket {}",
        n_real,
        compiled.n_pos
    );
    ensure!(n_cond == compiled.n_cond, "n_cond mismatch");
    let in_ch = compiled.in_channels;
    let out_ch = compiled.out_channels;
    let mut tok = vec![0.0f32; compiled.n_pos * in_ch];
    tok[..n_real * in_ch].copy_from_slice(tokens);
    let mut coords_p = vec![0.0f32; compiled.n_pos * 3];
    coords_p[..n_real * 3].copy_from_slice(coords);
    // Pad remaining coords with a far-away voxel so RoPE phases don't collide
    // with the active set (attention still mixes pad tokens — acceptable for
    // flow-matching when the sampler state for pad channels is zero).
    for i in n_real..compiled.n_pos {
        coords_p[i * 3] = 1.0e4;
        coords_p[i * 3 + 1] = 1.0e4;
        coords_p[i * 3 + 2] = 1.0e4;
    }
    let out = dit_forward_compiled(
        compiled,
        cfg,
        wm,
        &tok,
        &coords_p,
        compiled.n_pos,
        cond,
        n_cond,
        t,
    )?;
    Ok(out[..n_real * out_ch].to_vec())
}

/// Bucket size for variable-length SLat DiTs.
pub fn bucket_n_pos(n: usize, max_tokens: usize) -> usize {
    const BUCKETS: &[usize] = &[
        256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152,
    ];
    let cap = max_tokens.max(n);
    if let Some(b) = BUCKETS.iter().copied().find(|&b| b >= n && b <= cap) {
        return b;
    }
    n.next_power_of_two().clamp(n, cap)
}

fn clone_weight_map(wm: &WeightMap) -> Result<WeightMap> {
    let mut tensors = HashMap::new();
    for key in wm.keys() {
        let (data, shape) = wm.get(key).with_context(|| format!("clone weight {key}"))?;
        tensors.insert(key.to_string(), (data.to_vec(), shape.to_vec()));
    }
    Ok(WeightMap::from_tensors(tensors))
}

fn preprocess_rope_weights(wm: &mut WeightMap, cfg: &DitConfig) -> Result<()> {
    let c = cfg.args.model_channels;
    let nh = cfg.num_heads();
    let hd = cfg.head_dim();
    let cond = cfg.args.cond_channels;
    let mut tensors = HashMap::new();
    for key in wm.keys() {
        let (data, shape) = wm.get(key).expect("key");
        tensors.insert(key.to_string(), (data.to_vec(), shape.to_vec()));
    }

    for blk in 0..cfg.args.num_blocks {
        let p = format!("blocks.{blk}");
        deinterleave_linear_rows_map(
            &mut tensors,
            &format!("{p}.self_attn.to_qkv.weight"),
            3 * c,
            c,
            nh,
            hd,
            0,
        )?;
        deinterleave_linear_rows_map(
            &mut tensors,
            &format!("{p}.self_attn.to_qkv.weight"),
            3 * c,
            c,
            nh,
            hd,
            c,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.self_attn.to_qkv.bias"),
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.self_attn.to_qkv.bias"),
            nh,
            hd,
            c,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.self_attn.q_rms_norm.gamma"),
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.self_attn.k_rms_norm.gamma"),
            nh,
            hd,
            0,
        )?;

        deinterleave_linear_rows_map(
            &mut tensors,
            &format!("{p}.cross_attn.to_q.weight"),
            c,
            c,
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.cross_attn.to_q.bias"),
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.cross_attn.q_rms_norm.gamma"),
            nh,
            hd,
            0,
        )?;

        deinterleave_linear_rows_map(
            &mut tensors,
            &format!("{p}.cross_attn.to_kv.weight"),
            2 * c,
            cond,
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.cross_attn.to_kv.bias"),
            nh,
            hd,
            0,
        )?;
        deinterleave_bias_map(
            &mut tensors,
            &format!("{p}.cross_attn.k_rms_norm.gamma"),
            nh,
            hd,
            0,
        )?;
    }
    *wm = WeightMap::from_tensors(tensors);
    Ok(())
}

fn deinterleave_linear_rows_map(
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
    out_rows: usize,
    in_cols: usize,
    heads: usize,
    hd: usize,
    row_offset: usize,
) -> Result<()> {
    let (data, shape) = tensors
        .get_mut(key)
        .with_context(|| format!("missing weight {key}"))?;
    ensure!(
        shape.as_slice() == [out_rows, in_cols],
        "{key}: expected shape [{out_rows}, {in_cols}], got {shape:?}"
    );
    apply_deinterleave_rows(data, out_rows, in_cols, heads, hd, row_offset);
    Ok(())
}

fn deinterleave_bias_map(
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
    heads: usize,
    hd: usize,
    offset: usize,
) -> Result<()> {
    let (data, _) = tensors
        .get_mut(key)
        .with_context(|| format!("missing weight {key}"))?;
    apply_deinterleave_vec(data, heads, hd, offset);
    Ok(())
}

fn apply_deinterleave_rows(
    w: &mut [f32],
    _out_rows: usize,
    in_cols: usize,
    heads: usize,
    hd: usize,
    row_offset: usize,
) {
    let perm = rope::deinterleave_perm(hd);
    let mut tmp = vec![0.0f32; hd * in_cols];
    for h in 0..heads {
        let base = row_offset + h * hd;
        for i in 0..hd {
            let src = base + perm[i];
            tmp[i * in_cols..(i + 1) * in_cols]
                .copy_from_slice(&w[src * in_cols..(src + 1) * in_cols]);
        }
        for i in 0..hd {
            w[(base + i) * in_cols..(base + i + 1) * in_cols]
                .copy_from_slice(&tmp[i * in_cols..(i + 1) * in_cols]);
        }
    }
}

fn apply_deinterleave_vec(v: &mut [f32], heads: usize, hd: usize, offset: usize) {
    let perm = rope::deinterleave_perm(hd);
    let mut tmp = vec![0.0f32; hd];
    for h in 0..heads {
        let base = offset + h * hd;
        for i in 0..hd {
            tmp[i] = v[base + perm[i]];
        }
        v[base..base + hd].copy_from_slice(&tmp);
    }
}

fn build_dit_flow(
    cfg: &DitConfig,
    weights: &mut WeightMap,
    n_pos: usize,
    n_cond: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let c = cfg.args.model_channels;
    let in_ch = cfg.args.in_channels;
    let out_ch = cfg.args.out_channels;
    let cond_ch = cfg.args.cond_channels;
    let nh = cfg.num_heads();
    let hd = cfg.head_dim();
    let half = hd / 2;
    let mlp_h = cfg.mlp_hidden();
    let num_blocks = cfg.args.num_blocks;

    let flow = ModelFlow::new("trellis_dit")
        .with_profile(CompileProfile::encoder())
        .input("tokens", Shape::new(&[1, n_pos, in_ch], f))
        .input("t_mod", Shape::new(&[1, 1, 6 * c], f))
        .input("cond", Shape::new(&[1, n_cond, cond_ch], f))
        .input("cos", Shape::new(&[n_pos, half], f))
        .input("sin", Shape::new(&[n_pos, half], f))
        .stage(plugin_named("bind_rope", move |emit, hidden| {
            let cos = emit.flow_input("cos")?;
            let sin = emit.flow_input("sin")?;
            emit.set_named("dit_cos", cos.hir_id());
            emit.set_named("dit_sin", sin.hir_id());
            let ones = emit.synth_param("dit_rms_ones", vec![1.0f32; hd], Shape::new(&[hd], f));
            let zeros = emit.synth_zeros("dit_rms_zeros", hd);
            emit.set_named("dit_rms_ones", ones);
            emit.set_named("dit_rms_zeros", zeros);
            let ln_ones = emit.synth_param("dit_ln_ones", vec![1.0f32; c], Shape::new(&[c], f));
            let ln_zeros = emit.synth_zeros("dit_ln_zeros", c);
            emit.set_named("dit_ln_ones", ln_ones);
            emit.set_named("dit_ln_zeros", ln_zeros);
            Ok(hidden)
        }))
        .stage(plugin_named("input_layer", move |emit, hidden| {
            let x = hidden.ok_or_else(|| anyhow::anyhow!("input_layer needs tokens"))?;
            let w = emit.load_param("input_layer.weight", true)?;
            let b = emit.load_param("input_layer.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let y = {
                let m = gb.mm(x.hir_id(), w);
                gb.add(m, b)
            };
            Ok(Some(emit.wrap(y, Shape::new(&[1, n_pos, c], f))))
        }))
        .repeat_layers(num_blocks, move |blk| {
            trellis_block(blk, n_pos, n_cond, c, cond_ch, nh, hd, mlp_h)
        })
        .stage(plugin_named("out_layer", move |emit, hidden| {
            let x = hidden.ok_or_else(|| anyhow::anyhow!("out_layer needs hidden"))?;
            let ones = emit.named("dit_ln_ones")?;
            let zeros = emit.named("dit_ln_zeros")?;
            let w = emit.load_param("out_layer.weight", true)?;
            let b = emit.load_param("out_layer.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let ln = gb.ln(x.hir_id(), ones, zeros, ADA_EPS);
            let y = {
                let m = gb.mm(ln, w);
                gb.add(m, b)
            };
            Ok(Some(emit.wrap(y, Shape::new(&[1, n_pos, out_ch], f))))
        }))
        .output("out");

    flow.build_with(&mut WeightMapSource(weights), None)
}

fn trellis_block(
    blk: usize,
    n_pos: usize,
    n_cond: usize,
    c: usize,
    cond_ch: usize,
    nh: usize,
    hd: usize,
    mlp_h: usize,
) -> rlx_flow::FlowStage {
    let p = format!("blocks.{blk}");
    plugin_named(format!("block{blk}"), move |emit, hidden| {
        let x_in = hidden.ok_or_else(|| anyhow::anyhow!("block {blk} needs hidden"))?;
        let f = DType::F32;
        let out_shape = Shape::new(&[1, n_pos, c], f);
        let t_mod = emit.flow_input("t_mod")?;
        let cond = emit.flow_input("cond")?;
        let cos = emit.named("dit_cos")?;
        let sin = emit.named("dit_sin")?;
        let rms_ones = emit.named("dit_rms_ones")?;
        let rms_zeros = emit.named("dit_rms_zeros")?;

        // modulation = blocks.i.modulation + t_mod → 6 chunks
        let mod_b = emit.load_param(&format!("{p}.modulation"), false)?;
        let mut gb = HirMut::new(emit.hir());
        let mod_r = gb.reshape_(mod_b, vec![1, 1, (6 * c) as i64]);
        let m = gb.add(t_mod.hir_id(), mod_r);
        let shift_msa = gb.narrow_(m, 2, 0, c);
        let scale_msa = gb.narrow_(m, 2, c, c);
        let gate_msa = gb.narrow_(m, 2, 2 * c, c);
        let shift_mlp = gb.narrow_(m, 2, 3 * c, c);
        let scale_mlp = gb.narrow_(m, 2, 4 * c, c);
        let gate_mlp = gb.narrow_(m, 2, 5 * c, c);

        // ── self-attn ──
        let residual = x_in.hir_id();
        let h = gb.ada_layer_norm(
            residual,
            scale_msa,
            shift_msa,
            AdaNormKind::LayerNorm,
            ADA_EPS,
        );

        let qkv_w = {
            // End `HirMut` borrow of `emit.hir()` before `load_param`.
            #[allow(clippy::drop_non_drop)]
            drop(gb);
            emit.load_param(&format!("{p}.self_attn.to_qkv.weight"), true)?
        };
        let qkv_b = emit.load_param(&format!("{p}.self_attn.to_qkv.bias"), false)?;
        let sa_q_g = emit.load_param(&format!("{p}.self_attn.q_rms_norm.gamma"), false)?;
        let sa_k_g = emit.load_param(&format!("{p}.self_attn.k_rms_norm.gamma"), false)?;
        let sa_out_w = emit.load_param(&format!("{p}.self_attn.to_out.weight"), true)?;
        let sa_out_b = emit.load_param(&format!("{p}.self_attn.to_out.bias"), false)?;

        let mut gb = HirMut::new(emit.hir());
        let qkv = {
            let m = gb.mm(h, qkv_w);
            gb.add(m, qkv_b)
        };
        let q = gb.narrow_(qkv, 2, 0, c);
        let k = gb.narrow_(qkv, 2, c, c);
        let v = gb.narrow_(qkv, 2, 2 * c, c);
        let q = multihead_rms(&mut gb, q, sa_q_g, rms_ones, rms_zeros, n_pos, nh, hd, c);
        let k = multihead_rms(&mut gb, k, sa_k_g, rms_ones, rms_zeros, n_pos, nh, hd, c);
        let q = gb.rope(q, cos, sin, hd);
        let k = gb.rope(k, cos, sin, hd);
        let attn = gb.attention_kind(q, k, v, nh, hd, MaskKind::None, out_shape.clone());
        let sa = {
            let m = gb.mm(attn, sa_out_w);
            gb.add(m, sa_out_b)
        };
        let x = gb.gated_residual(residual, sa, gate_msa);

        // ── cross-attn ──
        let n2_w = {
            #[allow(clippy::drop_non_drop)]
            drop(gb);
            emit.load_param(&format!("{p}.norm2.weight"), false)?
        };
        let n2_b = emit.load_param(&format!("{p}.norm2.bias"), false)?;
        let ca_q_w = emit.load_param(&format!("{p}.cross_attn.to_q.weight"), true)?;
        let ca_q_b = emit.load_param(&format!("{p}.cross_attn.to_q.bias"), false)?;
        let ca_kv_w = emit.load_param(&format!("{p}.cross_attn.to_kv.weight"), true)?;
        let ca_kv_b = emit.load_param(&format!("{p}.cross_attn.to_kv.bias"), false)?;
        let ca_q_g = emit.load_param(&format!("{p}.cross_attn.q_rms_norm.gamma"), false)?;
        let ca_k_g = emit.load_param(&format!("{p}.cross_attn.k_rms_norm.gamma"), false)?;
        let ca_out_w = emit.load_param(&format!("{p}.cross_attn.to_out.weight"), true)?;
        let ca_out_b = emit.load_param(&format!("{p}.cross_attn.to_out.bias"), false)?;

        let mut gb = HirMut::new(emit.hir());
        let h2 = gb.ln(x, n2_w, n2_b, ADA_EPS);
        let cq = {
            let m = gb.mm(h2, ca_q_w);
            gb.add(m, ca_q_b)
        };
        let ckv = {
            let m = gb.mm(cond.hir_id(), ca_kv_w);
            gb.add(m, ca_kv_b)
        };
        let ck = gb.narrow_(ckv, 2, 0, c);
        let cv = gb.narrow_(ckv, 2, c, c);
        let cq = multihead_rms(&mut gb, cq, ca_q_g, rms_ones, rms_zeros, n_pos, nh, hd, c);
        let ck = multihead_rms(&mut gb, ck, ca_k_g, rms_ones, rms_zeros, n_cond, nh, hd, c);
        let cattn = gb.attention_kind(cq, ck, cv, nh, hd, MaskKind::None, out_shape.clone());
        let ca = {
            let m = gb.mm(cattn, ca_out_w);
            gb.add(m, ca_out_b)
        };
        let x = gb.add(x, ca);

        // ── MLP ──
        let mlp0_w = {
            #[allow(clippy::drop_non_drop)]
            drop(gb);
            emit.load_param(&format!("{p}.mlp.mlp.0.weight"), true)?
        };
        let mlp0_b = emit.load_param(&format!("{p}.mlp.mlp.0.bias"), false)?;
        let mlp2_w = emit.load_param(&format!("{p}.mlp.mlp.2.weight"), true)?;
        let mlp2_b = emit.load_param(&format!("{p}.mlp.mlp.2.bias"), false)?;

        let mut gb = HirMut::new(emit.hir());
        let residual = x;
        let h3 = gb.ada_layer_norm(x, scale_mlp, shift_mlp, AdaNormKind::LayerNorm, ADA_EPS);
        let up = {
            let m = gb.mm(h3, mlp0_w);
            gb.add(m, mlp0_b)
        };
        let act = gb.gelu_approx(up);
        let down = {
            let m = gb.mm(act, mlp2_w);
            gb.add(m, mlp2_b)
        };
        let out = gb.gated_residual(residual, down, gate_mlp);

        let _ = (cond_ch, mlp_h); // shapes checked via weight load
        Ok(Some(emit.wrap(out, out_shape)))
    })
}

fn multihead_rms(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    ones: HirNodeId,
    zeros: HirNodeId,
    n: usize,
    nh: usize,
    hd: usize,
    c: usize,
) -> HirNodeId {
    let flat = gb.reshape_(x, vec![1, (n * nh) as i64, hd as i64]);
    let nrm = gb.rms_norm(flat, ones, zeros, QK_RMS_EPS);
    let nrm4 = gb.reshape_(nrm, vec![1, n as i64, nh as i64, hd as i64]);
    let g4 = gb.reshape_(gamma, vec![1, 1, nh as i64, hd as i64]);
    let out = gb.mul(nrm4, g4);
    gb.reshape_(out, vec![1, n as i64, c as i64])
}
