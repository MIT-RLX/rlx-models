// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Running the real weights **without dequantizing them**.
//!
//! `tests/real_weights.rs` calls `WeightMap::from_weight_loader_dequant_all`,
//! which expands every K-quant block to f32 — convenient, and about 5.8× the
//! bytes at `UD-IQ1_S`. The alternative is to leave the GGUF blocks packed and
//! let each projection be a fused `Op::DequantMatMul`: same arithmetic, one
//! pass, no f32 weight ever materialized.
//!
//! rlx already has the seam — `WeightSource::take_packed`, served by
//! `rlx_core::flow_bridge::PackedWeightLoaderSource` — so this needs no graph
//! surgery, only that the crate's projections go through
//! [`rlx_glm5next::common::linear`], which consults it.
//!
//! These tests assert the two paths **agree numerically** and report how much
//! smaller the packed one is. Same 337 MB subset and the same
//! `RLX_GLM5NEXT_GGUF` gate as `real_weights.rs`; without it they skip.
//!
//! ## What does not pack
//!
//! Only 2-D projections. Norms, `ssm_a`, `dt_bias` and `exp_probs_b` are f32 in
//! the checkpoint anyway; the depthwise `ssm_conv1d_*` kernels and MLA's
//! per-head `attn_k_b` / `attn_v_b` are 3-D; and the routed expert banks go
//! through `GroupedMatMul`, which has no packed form here. The expert banks are
//! the ones that matter for a whole-model run — 2.2 GB for a single MoE layer —
//! so this is a step toward that, not the end of it.

use rlx_core::flow_bridge::PackedWeightLoaderSource;
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, WeightSource};
use rlx_glm5next::kda::{KdaDims, emit_kda_attention};
use rlx_glm5next::mla::{MlaDims, emit_mla_attention};
use rlx_glm5next::moe::{MoeDims, emit_glm5next_moe};
use rlx_glm5next::{Glm5NextConfig, IndexerDims};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;

const ENV: &str = "RLX_GLM5NEXT_GGUF";

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn path() -> Option<String> {
    let p = std::env::var(ENV).ok().filter(|s| !s.is_empty())?;
    if !std::path::Path::new(&p).exists() {
        eprintln!("skip: {ENV}={p} does not exist");
        return None;
    }
    Some(p)
}

macro_rules! path_or_skip {
    () => {
        match path() {
            Some(p) => p,
            None => {
                eprintln!("skip: set {} to a glm5next GGUF subset", ENV);
                return;
            }
        }
    };
}

fn cfg(p: &str) -> Glm5NextConfig {
    Glm5NextConfig::from_gguf_path(p).expect("parse glm5next config")
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0
        })
        .collect()
}

/// Bytes of weight the built graph carries: f32 params plus packed U8 blobs.
fn weight_bytes(built: &BuiltModel) -> (usize, usize) {
    let typed: usize = built.typed_params.iter().map(|(_, b, _)| b.len()).sum();
    let dense: usize = built.params().values().map(|v| v.len() * 4).sum();
    (dense, typed)
}

/// Build one block twice — dequantized and packed — and return
/// `(dense_out, packed_out, dense_bytes, packed_bytes)`.
fn both_paths<F>(
    p: &str,
    seq: usize,
    hidden: usize,
    build: F,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, usize, usize)
where
    F: Fn(&mut dyn WeightSource, usize) -> BuiltModel + Copy,
{
    // ── dequantized: every K-quant expanded to f32 up front ──
    let mut loader = rlx_core::weight_loader::load_from_path(p).expect("open");
    let mut wm = WeightMap::from_weight_loader_dequant_all(loader.as_mut()).expect("dequant");
    let built = build(&mut WeightMapSource(&mut wm), seq);
    let (d_dense, d_typed) = weight_bytes(&built);
    let dense_out = compile_built(built, dev())
        .expect("compile dense")
        .run(&[("x", x)])
        .pop()
        .expect("out");

    // ── packed: the GGUF blocks go straight into DequantMatMul ──
    let mut loader = rlx_core::weight_loader::load_from_path(p).expect("open");
    let built = build(&mut PackedWeightLoaderSource(loader.as_mut()), seq);
    let (p_dense, p_typed) = weight_bytes(&built);
    let packed_out = compile_built(built, dev())
        .expect("compile packed")
        .run(&[("x", x)])
        .pop()
        .expect("out");

    let _ = hidden;
    (dense_out, packed_out, d_dense + d_typed, p_dense + p_typed)
}

fn compare(dense: &[f32], packed: &[f32], what: &str, d_bytes: usize, p_bytes: usize) {
    assert_eq!(dense.len(), packed.len(), "{what}: length");
    assert!(
        packed.iter().all(|v| v.is_finite()),
        "{what}: packed produced non-finite values"
    );
    let worst = dense
        .iter()
        .zip(packed)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = dense.iter().map(|v| v.abs()).fold(1e-6, f32::max);
    eprintln!(
        "{what}: weights {:.1} MB dequantized vs {:.1} MB packed ({:.1}× smaller); \
         max |Δ| = {worst:.3e} over scale {scale:.3e}",
        d_bytes as f64 / 1e6,
        p_bytes as f64 / 1e6,
        d_bytes as f64 / p_bytes.max(1) as f64,
    );
    // Both paths dequantize the *same* blocks; only the order of the dequant
    // relative to the accumulation differs, so this is accumulation noise.
    assert!(
        worst / scale < 1e-5,
        "{what}: packed diverges from dequantized; max |Δ| = {worst} over {scale}"
    );
    assert!(
        p_bytes < d_bytes,
        "{what}: packed should carry fewer weight bytes ({p_bytes} vs {d_bytes})"
    );
}

/// The KDA block (`blk.0`) — all-`Q5_K` projections plus f32 SSM parameters.
#[test]
fn packed_kda_block_matches_dequantized() {
    let p = path_or_skip!();
    let c = cfg(&p);
    let seq = 8;
    let hidden = c.hidden_size;
    let d = KdaDims {
        hidden,
        num_heads: c.linear_num_heads,
        head_dim: c.linear_head_dim,
        conv_kernel: c.linear_conv_kernel_dim,
        lower_bound: c.linear_lower_bound,
        eps: c.rms_norm_eps,
        seq,
    };
    let x = fill(seq * hidden, 61);
    let build = move |src: &mut dyn WeightSource, seq: usize| {
        let hs = Shape::new(&[1, seq, hidden], DType::F32);
        ModelFlow::new("kda")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", hs.clone())
            .plugin_named("blk", move |emit, _p| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_kda_attention(emit, "blk.0", x, d)?;
                Ok(Some(emit.wrap(out, hs.clone())))
            })
            .output("out")
            .build_with(src, None)
            .expect("build KDA block")
    };
    let (dense, packed, db, pb) = both_paths(&p, seq, hidden, build, &x);
    compare(&dense, &packed, "KDA blk.0", db, pb);
}

/// The MLA block (`blk.3`). Its per-head `attn_k_b` / `attn_v_b` are 3-D and
/// stay f32, so the saving is smaller than KDA's — which is exactly the shape of
/// the remaining work.
#[test]
fn packed_mla_block_matches_dequantized() {
    let p = path_or_skip!();
    let c = cfg(&p);
    let seq = 8;
    let hidden = c.hidden_size;
    let mla = MlaDims {
        hidden,
        num_heads: c.num_attention_heads,
        q_lora_rank: c.q_lora_rank,
        kv_lora_rank: c.kv_lora_rank,
        qk_nope_head_dim: c.qk_nope_head_dim,
        v_head_dim: c.v_head_dim,
        eps: c.rms_norm_eps,
        seq,
    };
    let idx = IndexerDims {
        hidden,
        q_lora_rank: c.q_lora_rank,
        n_heads: c.index_n_heads,
        head_dim: c.index_head_dim,
        topk: c.index_topk,
        kpool: c.index_kpool,
        always_select_tail: c.index_kpool_always_select_tail,
        seq,
        force_emit: false,
    };
    let x = fill(seq * hidden, 67);
    let build = move |src: &mut dyn WeightSource, seq: usize| {
        let hs = Shape::new(&[1, seq, hidden], DType::F32);
        ModelFlow::new("mla")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", hs.clone())
            .plugin_named("blk", move |emit, _p| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_mla_attention(emit, "blk.3", x, mla, idx)?;
                Ok(Some(emit.wrap(out, hs.clone())))
            })
            .output("out")
            .build_with(src, None)
            .expect("build MLA block")
    };
    let (dense, packed, db, pb) = both_paths(&p, seq, hidden, build, &x);
    compare(&dense, &packed, "MLA blk.3", db, pb);
}

/// The packed path must actually be taken — i.e. the graph carries U8 blobs and
/// noticeably fewer f32 bytes. Without this, a `take_packed` that silently
/// returned `None` would make the agreement tests pass trivially.
#[test]
fn packed_path_is_actually_used() {
    let p = path_or_skip!();
    let c = cfg(&p);
    let seq = 4;
    let hidden = c.hidden_size;
    let d = KdaDims {
        hidden,
        num_heads: c.linear_num_heads,
        head_dim: c.linear_head_dim,
        conv_kernel: c.linear_conv_kernel_dim,
        lower_bound: c.linear_lower_bound,
        eps: c.rms_norm_eps,
        seq,
    };
    let hs = Shape::new(&[1, seq, hidden], DType::F32);
    let mut loader = rlx_core::weight_loader::load_from_path(&p).expect("open");
    let built = ModelFlow::new("kda")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", hs.clone())
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_kda_attention(emit, "blk.0", x, d)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out")
        .build_with(&mut PackedWeightLoaderSource(loader.as_mut()), None)
        .expect("build packed");

    let (dense_bytes, typed_bytes) = weight_bytes(&built);
    eprintln!("packed KDA blk.0: {typed_bytes} U8 bytes + {dense_bytes} f32 bytes");
    assert!(
        typed_bytes > 0,
        "no packed blobs were registered — the projections did not take the \
         DequantMatMul path"
    );
    // blk.0's four projections are Q5_K; everything left in f32 (norms, ssm_a,
    // dt_bias, the conv kernels) is small by comparison.
    assert!(
        typed_bytes > dense_bytes,
        "expected the packed blobs to dominate: {typed_bytes} U8 vs {dense_bytes} f32"
    );
}

// ─────────────────────── routed expert banks ───────────────────────

/// A [`WeightSource`] that narrows the MoE router and its bias to the first `n`
/// experts.
///
/// `scripts/glm5next_subset.py --experts N` slices the routed *banks* (each
/// expert is a contiguous byte range, so 8 of 288 is ~60 MB instead of 2.2 GB)
/// but leaves `ffn_gate_inp` / `exp_probs_b` describing all 288 — they are tiny
/// and slicing them in the file would misrepresent it. This does it on the way
/// in, so both the dense and the packed build see a consistent 8-expert layer.
struct SlicedExperts<'a> {
    inner: &'a mut dyn WeightSource,
    experts: usize,
}

impl WeightSource for SlicedExperts<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.inner.take(key, transpose)?;
        if key.ends_with("ffn_gate_inp.weight") {
            // Router is [experts, hidden]; `transpose` yields [hidden, experts].
            return Ok(if transpose {
                let (h, e) = (shape[0], shape[1]);
                let keep = self.experts.min(e);
                let mut out = Vec::with_capacity(h * keep);
                for r in 0..h {
                    out.extend_from_slice(&data[r * e..r * e + keep]);
                }
                (out, vec![h, keep])
            } else {
                let keep = self.experts.min(shape[0]);
                let cols = shape[1];
                (data[..keep * cols].to_vec(), vec![keep, cols])
            });
        }
        if key.ends_with("exp_probs_b.bias") {
            let keep = self.experts.min(shape[0]);
            return Ok((data[..keep].to_vec(), vec![keep]));
        }
        Ok((data, shape))
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(key)
    }
    fn take_packed(&mut self, key: &str) -> anyhow::Result<Option<rlx_flow::GgufPackedLinear>> {
        self.inner.take_packed(key)
    }
    fn take_packed_bank(&mut self, key: &str) -> anyhow::Result<Option<rlx_flow::GgufPackedBank>> {
        self.inner.take_packed_bank(key)
    }
}

/// The routed MoE on **real** `IQ2_XXS` / `IQ3_XXS` expert banks, packed vs
/// dequantized.
///
/// This is the path a whole-model run needs and the one nothing else covers:
/// `Op::DequantGroupedMatMul` over the quant blob instead of `GroupedMatMul`
/// over an F32 bank. It also exercises two quant types no other test touches —
/// the banks are `IQ2_XXS` (gate/up) and `IQ3_XXS` (down) at `UD-IQ1_S`.
///
/// Needs a subset built with `--experts 8`; skips otherwise, since the plain
/// subset has no routed banks at all.
#[test]
fn packed_moe_matches_dequantized() {
    let p = path_or_skip!();
    let mut c = cfg(&p);
    const E: usize = 8;

    // Bail out politely if this subset has no expert banks.
    {
        let loader = rlx_core::weight_loader::load_from_path(&p).expect("open");
        if !loader
            .remaining_keys()
            .iter()
            .any(|k| k == "blk.3.ffn_gate_exps.weight")
        {
            eprintln!("skip: {p} has no routed expert banks (rebuild with --experts 8)");
            return;
        }
    }

    c.n_routed_experts = E;
    c.num_experts_per_tok = c.num_experts_per_tok.min(E);
    let seq = 4;
    let hidden = c.hidden_size;
    let moe = MoeDims {
        paged: false,
        hidden,
        moe_inter: c.moe_intermediate_size,
        n_routed: E,
        top_k: c.num_experts_per_tok,
        n_group: c.n_group,
        topk_group: c.topk_group,
        routed_scaling: c.routed_scaling_factor,
        swiglu_limit: Some(c.swiglu_limit),
        seq,
    };
    let x = fill(seq * hidden, 71);

    let build = move |src: &mut dyn WeightSource| {
        let mut src = SlicedExperts {
            inner: src,
            experts: E,
        };
        let hs = Shape::new(&[1, seq, hidden], DType::F32);
        ModelFlow::new("moe")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", hs.clone())
            .plugin_named("blk", move |emit, _p| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_glm5next_moe(emit, "blk.3", x, moe)?;
                Ok(Some(emit.wrap(out, hs.clone())))
            })
            .output("out")
            .build_with(&mut src, None)
            .expect("build MoE block")
    };

    // ── dequantized: the 8-expert banks expanded to f32 ──
    let mut loader = rlx_core::weight_loader::load_from_path(&p).expect("open");
    let mut wm = WeightMap::from_weight_loader_dequant_all(loader.as_mut()).expect("dequant");
    let built = build(&mut WeightMapSource(&mut wm));
    let (d_dense, d_typed) = weight_bytes(&built);
    let dense = compile_built(built, dev())
        .expect("compile dense MoE")
        .run(&[("x", x.as_slice())])
        .pop()
        .expect("out");

    // ── packed: banks stay as GGUF blocks, one DequantGroupedMatMul per pick ──
    let mut loader = rlx_core::weight_loader::load_from_path(&p).expect("open");
    let built = build(&mut PackedWeightLoaderSource(loader.as_mut()));
    let (p_dense, p_typed) = weight_bytes(&built);
    assert!(
        p_typed > 0,
        "no packed blobs registered — the MoE did not take the grouped \
         DequantMatMul path"
    );
    let packed = compile_built(built, dev())
        .expect("compile packed MoE")
        .run(&[("x", x.as_slice())])
        .pop()
        .expect("out");

    compare(
        &dense,
        &packed,
        &format!("MoE blk.3 ({E} real experts)"),
        d_dense + d_typed,
        p_dense + p_typed,
    );
}
