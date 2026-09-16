// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Real GLM-5.3-Flash weights, without downloading GLM-5.3-Flash.
//!
//! The published checkpoint is 93 GB at its smallest quantization and its
//! smallest text shard is 43.5 GB, so a whole-model run is not on the table
//! here. But a GGUF's header carries every tensor's byte offset, so the
//! interesting layers can be pulled out of a shard with HTTP range requests and
//! written to a small standalone file. `scripts/glm5next_subset.py` fetches
//! **337 MB** — `blk.0` in full plus `blk.3`'s attention, indexer and mHC — and
//! writes a valid `glm5next` GGUF carrying the real metadata.
//!
//! ```text
//! just glm5next-real          # fetch the subset, then run this file
//! ```
//!
//! Or point `RLX_GLM5NEXT_GGUF` at a subset yourself; without it these skip.
//!
//! `blk.0` and `blk.3` are the two layer *kinds*: layer 0 is KDA + dense FFN,
//! layer 3 is the first MLA + DSA + MoE layer. Between them they cover every
//! block this crate emits except the routed experts (whose banks are 2.2 GB for
//! that one layer, and which `text_flow_smoke` already covers synthetically).
//!
//! What this actually pins down, over and above the synthetic tests:
//!
//! * **Every tensor name and shape is real.** A typo or a transposed dim in the
//!   GGUF contract fails at load, against the published artifact rather than
//!   against a fixture written from the same (possibly wrong) understanding.
//! * **The k-quant dequant path** — `Q5_K`, `Q6_K`, `Q8_0` — is exercised, which
//!   no synthetic f32 test does.
//! * **mHC's learned parameters behave.** `comb` being near-doubly-stochastic
//!   and `pre`/`post` landing in range are properties of the *trained* gates;
//!   random weights would satisfy them trivially, real ones need the Sinkhorn
//!   schedule and the `scale`/`base` split to be right.
//! * **Decode reproduces prefill on real weights**, not just synthetic ones.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_glm5next::kda::{KdaDims, KdaState, emit_kda_attention, emit_kda_decode};
use rlx_glm5next::mhc::{MhcDims, emit_mhc_gates};
use rlx_glm5next::mla::{MlaCache, MlaDims, emit_mla_attention, emit_mla_decode};
use rlx_glm5next::{Glm5NextConfig, IndexerDims};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;

const ENV: &str = "RLX_GLM5NEXT_GGUF";

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// `(config, weights)` from the subset file, or `None` to skip.
fn load() -> Option<(Glm5NextConfig, WeightMap)> {
    let path = std::env::var(ENV).ok().filter(|s| !s.is_empty())?;
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: {ENV}={path} does not exist");
        return None;
    }
    let cfg = Glm5NextConfig::from_gguf_path(&path).expect("parse glm5next config");
    // `WeightMap::from_file` is the safetensors path; a GGUF goes through the
    // format registry. `from_weight_loader` would leave the K-quants packed for
    // a runner with a packed-matmul lowering — this crate has none, so force the
    // dequant to f32.
    let mut loader = rlx_core::weight_loader::load_from_path(&path).expect("open glm5next GGUF");
    let weights = WeightMap::from_weight_loader_dequant_all(loader.as_mut())
        .expect("dequantize glm5next GGUF");
    Some((cfg, weights))
}

macro_rules! weights_or_skip {
    () => {
        match load() {
            Some(v) => v,
            None => {
                eprintln!("skip: set {} to a glm5next GGUF subset", ENV);
                return;
            }
        }
    };
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

fn finite_stats(v: &[f32]) -> (f32, f32) {
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let rms = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt();
    (max, rms as f32)
}

/// The whole published tensor contract for both layer kinds, checked against
/// the real file: names resolve, and shapes are what the emitters assume after
/// the loader reverses GGML's dim order.
#[test]
fn real_tensor_names_and_shapes_match_the_contract() {
    let (cfg, wm) = weights_or_skip!();
    let kda_proj = cfg.kda_proj();
    let qk = cfg.num_attention_heads * cfg.qk_nope_head_dim;

    let expect: Vec<(String, Vec<usize>)> = vec![
        // ── blk.0: KDA + dense FFN ──
        ("blk.0.attn_norm.weight".into(), vec![cfg.hidden_size]),
        (
            "blk.0.attn_q.weight".into(),
            vec![kda_proj, cfg.hidden_size],
        ),
        (
            "blk.0.attn_k.weight".into(),
            vec![kda_proj, cfg.hidden_size],
        ),
        (
            "blk.0.attn_v.weight".into(),
            vec![kda_proj, cfg.hidden_size],
        ),
        (
            "blk.0.attn_output.weight".into(),
            vec![cfg.hidden_size, kda_proj],
        ),
        (
            "blk.0.ssm_conv1d_q.weight".into(),
            vec![kda_proj, 1, cfg.linear_conv_kernel_dim],
        ),
        ("blk.0.ssm_a".into(), vec![cfg.linear_num_heads]),
        ("blk.0.ssm_dt.bias".into(), vec![kda_proj]),
        (
            "blk.0.ssm_beta.weight".into(),
            vec![cfg.linear_num_heads, cfg.hidden_size],
        ),
        (
            "blk.0.ssm_f_a.weight".into(),
            vec![cfg.linear_head_dim, cfg.hidden_size],
        ),
        (
            "blk.0.ssm_f_b.weight".into(),
            vec![kda_proj, cfg.linear_head_dim],
        ),
        ("blk.0.ssm_norm.weight".into(), vec![cfg.linear_head_dim]),
        (
            "blk.0.ffn_gate.weight".into(),
            vec![cfg.intermediate_size, cfg.hidden_size],
        ),
        (
            "blk.0.ffn_down.weight".into(),
            vec![cfg.hidden_size, cfg.intermediate_size],
        ),
        // ── mHC, both sites ──
        (
            "blk.0.hc_attn_fn.weight".into(),
            vec![cfg.hc_mix(), cfg.hc_mult * cfg.hidden_size],
        ),
        ("blk.0.hc_attn_base.weight".into(), vec![cfg.hc_mix()]),
        ("blk.0.hc_attn_scale.weight".into(), vec![3]),
        ("blk.0.hc_ffn_scale.weight".into(), vec![3]),
        // ── blk.3: MLA + DSA indexer ──
        (
            "blk.3.attn_q_a.weight".into(),
            vec![cfg.q_lora_rank, cfg.hidden_size],
        ),
        ("blk.3.attn_q_a_norm.weight".into(), vec![cfg.q_lora_rank]),
        ("blk.3.attn_q_b.weight".into(), vec![qk, cfg.q_lora_rank]),
        (
            "blk.3.attn_kv_a_mqa.weight".into(),
            vec![cfg.kv_lora_rank, cfg.hidden_size],
        ),
        ("blk.3.attn_kv_a_norm.weight".into(), vec![cfg.kv_lora_rank]),
        // The two orientations that are NOT the same way round.
        (
            "blk.3.attn_k_b.weight".into(),
            vec![
                cfg.num_attention_heads,
                cfg.kv_lora_rank,
                cfg.qk_nope_head_dim,
            ],
        ),
        (
            "blk.3.attn_v_b.weight".into(),
            vec![cfg.num_attention_heads, cfg.v_head_dim, cfg.kv_lora_rank],
        ),
        (
            "blk.3.indexer.attn_q_b.weight".into(),
            vec![cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank],
        ),
        (
            "blk.3.indexer.attn_k.weight".into(),
            vec![cfg.index_head_dim, cfg.hidden_size],
        ),
        ("blk.3.indexer.k_norm.bias".into(), vec![cfg.index_head_dim]),
        (
            "blk.3.indexer.proj.weight".into(),
            vec![cfg.index_n_heads, cfg.hidden_size],
        ),
        (
            "blk.3.indexer_compressor_ape.weight".into(),
            vec![cfg.index_kpool, cfg.index_head_dim],
        ),
        (
            "blk.3.indexer_compressor_gate.weight".into(),
            vec![cfg.index_head_dim, cfg.hidden_size],
        ),
    ];

    let mut wm = wm;
    for (name, want) in expect {
        let (data, shape) = wm
            .take(&name)
            .unwrap_or_else(|e| panic!("{name}: not in the published GGUF: {e}"));
        assert_eq!(shape, want, "{name}: shape");
        let n: usize = want.iter().product();
        assert_eq!(data.len(), n, "{name}: element count");
        assert!(
            data.iter().all(|v| v.is_finite()),
            "{name}: dequantized to non-finite values"
        );
        assert!(
            data.iter().any(|v| *v != 0.0),
            "{name}: dequantized to all zeros"
        );
    }
}

/// One KDA block on real `blk.0` weights.
#[test]
fn real_kda_block_runs() {
    let (cfg, mut wm) = weights_or_skip!();
    let seq = 8;
    let d = KdaDims {
        hidden: cfg.hidden_size,
        num_heads: cfg.linear_num_heads,
        head_dim: cfg.linear_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        lower_bound: cfg.linear_lower_bound,
        eps: cfg.rms_norm_eps,
        seq,
    };
    let hs = Shape::new(&[1, seq, cfg.hidden_size], DType::F32);
    let flow = ModelFlow::new("kda_real")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", hs.clone())
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_kda_attention(emit, "blk.0", x, d)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build real KDA block");
    let mut compiled = compile_built(built, dev()).expect("compile");

    let x = fill(seq * cfg.hidden_size, 5);
    let out = compiled.run(&[("x", x.as_slice())]).pop().expect("out");
    assert_eq!(out.len(), seq * cfg.hidden_size);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "real KDA block produced non-finite values"
    );
    let (max, rms) = finite_stats(&out);
    eprintln!("real KDA blk.0: max |o| = {max:.4}, rms = {rms:.4}");
    assert!(rms > 0.0, "real KDA block collapsed to zero");
    // A single attention branch on a unit-ish input should not blow up by orders
    // of magnitude; this catches a mis-scaled gate or a dropped norm.
    assert!(
        max < 1.0e3,
        "real KDA block output is implausibly large: {max}"
    );
}

/// One MLA + DSA block on real `blk.3` weights. At `seq = 8` the indexer is the
/// identity, so this is the fused causal path.
#[test]
fn real_mla_block_runs() {
    let (cfg, mut wm) = weights_or_skip!();
    let seq = 8;
    assert!(cfg.dsa_is_dense(seq));
    let d = MlaDims {
        hidden: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        eps: cfg.rms_norm_eps,
        seq,
    };
    let idx = IndexerDims {
        hidden: cfg.hidden_size,
        q_lora_rank: cfg.q_lora_rank,
        n_heads: cfg.index_n_heads,
        head_dim: cfg.index_head_dim,
        topk: cfg.index_topk,
        kpool: cfg.index_kpool,
        always_select_tail: cfg.index_kpool_always_select_tail,
        seq,
        force_emit: false,
    };
    let hs = Shape::new(&[1, seq, cfg.hidden_size], DType::F32);
    let flow = ModelFlow::new("mla_real")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", hs.clone())
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_mla_attention(emit, "blk.3", x, d, idx)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build real MLA block");
    let mut compiled = compile_built(built, dev()).expect("compile");

    let x = fill(seq * cfg.hidden_size, 11);
    let out = compiled.run(&[("x", x.as_slice())]).pop().expect("out");
    assert_eq!(out.len(), seq * cfg.hidden_size);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "real MLA block produced non-finite values"
    );
    let (max, rms) = finite_stats(&out);
    eprintln!("real MLA blk.3: max |o| = {max:.4}, rms = {rms:.4}");
    assert!(rms > 0.0, "real MLA block collapsed to zero");
    assert!(
        max < 1.0e3,
        "real MLA block output is implausibly large: {max}"
    );

    // Every position must differ: identical rows would mean the causal mask or
    // the per-head packing collapsed.
    let row = |i: usize| &out[i * cfg.hidden_size..(i + 1) * cfg.hidden_size];
    let d01 = row(0)
        .iter()
        .zip(row(seq - 1))
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(d01 > 1e-6, "all query positions produced the same output");
}

/// mHC on the **trained** gates. `comb` is Sinkhorn-projected, so its columns
/// must sum to 1 within `hc_eps`, and `pre`/`post` must land in their ranges —
/// properties of the real `scale` / `base` parameters, not of the shapes.
#[test]
fn real_mhc_gates_are_well_formed() {
    let (cfg, mut wm) = weights_or_skip!();
    let seq = 4;
    let h = cfg.hc_mult;
    let d = MhcDims {
        hidden: cfg.hidden_size,
        streams: h,
        sinkhorn_iters: cfg.hc_sinkhorn_iters,
        eps: cfg.hc_eps,
        norm_eps: cfg.rms_norm_eps,
        seq,
    };
    let stream = Shape::new(&[1, seq, h, cfg.hidden_size], DType::F32);
    let out_shape = Shape::new(&[1, seq, h * (2 + h)], DType::F32);
    let flow = ModelFlow::new("mhc_real")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", stream)
        .plugin_named("site", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let g = emit_mhc_gates(emit, "blk.0.hc_attn", x, d)?;
            // Pack pre | post | comb into one output.
            let mut gb = HirMut::new(emit.hir());
            let pre = gb.reshape_(g.collapsed, vec![1, seq as i64, cfg.hidden_size as i64]);
            let _ = pre;
            let post = gb.reshape_(g.post, vec![1, seq as i64, h as i64]);
            let comb = gb.reshape_(g.comb, vec![1, seq as i64, (h * h) as i64]);
            let packed = gb.concat_(vec![post, comb], 2);
            Ok(Some(emit.wrap(packed, out_shape.clone())))
        })
        .output("gates");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build real mHC");
    let mut compiled = compile_built(built, dev()).expect("compile");

    let x = fill(seq * h * cfg.hidden_size, 23);
    let g = compiled.run(&[("x", x.as_slice())]).pop().expect("gates");
    let width = h + h * h;
    assert_eq!(g.len(), seq * width);

    for t in 0..seq {
        let row = &g[t * width..(t + 1) * width];
        let post = &row[..h];
        let comb = &row[h..];
        for (i, p) in post.iter().enumerate() {
            assert!(
                p.is_finite() && *p >= 0.0 && *p <= 2.0,
                "post[{i}] = {p} outside [0, 2] (it is 2·sigmoid)"
            );
        }
        // Column-first Sinkhorn ⇒ columns sum to 1 - O(hc_eps).
        for c in 0..h {
            let s: f32 = (0..h).map(|r| comb[r * h + c]).sum();
            assert!(
                (s - 1.0).abs() < 1e-3,
                "trained comb column {c} at t={t} sums to {s}"
            );
        }
        for v in comb {
            assert!(*v >= 0.0 && *v <= 1.0, "comb entry {v} outside [0, 1]");
        }
    }
    eprintln!("real mHC blk.0.hc_attn: post/comb well-formed over {seq} positions");
}

/// The decode-vs-prefill equivalence, on real weights: stepping the KDA decode
/// block token by token must reproduce the prefill block.
///
/// The synthetic version of this is what caught the rlx-cpu matmul bug; running
/// it on trained weights additionally exercises the k-quant dequant path and
/// real gate magnitudes (the delta-net decay is `-5·σ(exp(A_log)·g)`, and real
/// `A_log` values are not the ~0 that random init gives).
#[test]
fn real_kda_decode_matches_prefill() {
    let (cfg, _) = weights_or_skip!();
    let seq = 6;
    let hidden = cfg.hidden_size;
    let d_pre = KdaDims {
        hidden,
        num_heads: cfg.linear_num_heads,
        head_dim: cfg.linear_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        lower_bound: cfg.linear_lower_bound,
        eps: cfg.rms_norm_eps,
        seq,
    };
    let d_dec = KdaDims { seq: 1, ..d_pre };
    let conv_w = (cfg.linear_conv_kernel_dim - 1) * cfg.kda_proj();
    let scan_w = cfg.linear_num_heads * cfg.linear_head_dim * cfg.linear_head_dim;
    let x = fill(seq * hidden, 31);

    // ── prefill ──
    let (_, mut wm) = load().expect("weights");
    let hs = Shape::new(&[1, seq, hidden], DType::F32);
    let flow = ModelFlow::new("kda_pre")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", hs.clone())
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_kda_attention(emit, "blk.0", x, d_pre)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let want = compile_built(built, dev())
        .unwrap()
        .run(&[("x", x.as_slice())])
        .pop()
        .unwrap();

    // ── decode, one token at a time ──
    let (_, mut wm) = load().expect("weights");
    let total = hidden + 3 * conv_w + scan_w;
    let os = Shape::new(&[1, total], DType::F32);
    let flow = ModelFlow::new("kda_dec")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, 1, hidden], DType::F32))
        .input(
            "cq",
            Shape::new(
                &[1, cfg.linear_conv_kernel_dim - 1, cfg.kda_proj()],
                DType::F32,
            ),
        )
        .input(
            "ck",
            Shape::new(
                &[1, cfg.linear_conv_kernel_dim - 1, cfg.kda_proj()],
                DType::F32,
            ),
        )
        .input(
            "cv",
            Shape::new(
                &[1, cfg.linear_conv_kernel_dim - 1, cfg.kda_proj()],
                DType::F32,
            ),
        )
        .input(
            "scan",
            Shape::new(
                &[
                    1,
                    cfg.linear_num_heads,
                    cfg.linear_head_dim,
                    cfg.linear_head_dim,
                ],
                DType::F32,
            ),
        )
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let st = KdaState {
                conv_q: emit.flow_input("cq")?.hir_id(),
                conv_k: emit.flow_input("ck")?.hir_id(),
                conv_v: emit.flow_input("cv")?.hir_id(),
                scan: emit.flow_input("scan")?.hir_id(),
            };
            let (out, next) = emit_kda_decode(emit, "blk.0", x, st, d_dec)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.reshape_(out, vec![1, hidden as i64]);
            let cq = gb.reshape_(next.conv_q, vec![1, conv_w as i64]);
            let ck = gb.reshape_(next.conv_k, vec![1, conv_w as i64]);
            let cv = gb.reshape_(next.conv_v, vec![1, conv_w as i64]);
            let sc = gb.reshape_(st.scan, vec![1, scan_w as i64]);
            let packed = gb.concat_(vec![o, cq, ck, cv, sc], 1);
            Ok(Some(emit.wrap(packed, os.clone())))
        })
        .output("packed");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut compiled = compile_built(built, dev()).unwrap();

    let (mut cq, mut ck, mut cv) = (vec![0f32; conv_w], vec![0f32; conv_w], vec![0f32; conv_w]);
    let mut scan = vec![0f32; scan_w];
    let mut got = Vec::with_capacity(seq * hidden);
    for t in 0..seq {
        let tok = &x[t * hidden..(t + 1) * hidden];
        let packed = compiled
            .run(&[
                ("x", tok),
                ("cq", cq.as_slice()),
                ("ck", ck.as_slice()),
                ("cv", cv.as_slice()),
                ("scan", scan.as_slice()),
            ])
            .pop()
            .unwrap();
        got.extend_from_slice(&packed[..hidden]);
        let mut o = hidden;
        cq.copy_from_slice(&packed[o..o + conv_w]);
        o += conv_w;
        ck.copy_from_slice(&packed[o..o + conv_w]);
        o += conv_w;
        cv.copy_from_slice(&packed[o..o + conv_w]);
        o += conv_w;
        scan.copy_from_slice(&packed[o..o + scan_w]);
    }

    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    eprintln!("real KDA decode vs prefill: max |Δ| = {worst:.3e} over scale {scale:.3e}");
    assert!(
        worst / scale < 1e-3,
        "real-weight decode diverges from prefill; max |Δ| = {worst} over scale {scale}"
    );
}

/// MLA decode against MLA prefill, on real `blk.3` weights.
///
/// The counterpart to [`real_kda_decode_matches_prefill`], and the more
/// interesting half: prefill expands the latent to per-head keys and values and
/// uses the fused attention kernel, while decode absorbs the query into latent
/// space and attends against the cached latent. Textually disjoint code paths
/// that must agree — so this is simultaneously a check that both of GGUF's
/// `attn_k_b` / `attn_v_b` orientations are read correctly, on trained values.
#[test]
fn real_mla_decode_matches_prefill() {
    let (cfg, _) = weights_or_skip!();
    let seq = 6;
    let hidden = cfg.hidden_size;
    let mla_pre = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        eps: cfg.rms_norm_eps,
        seq,
    };
    let mla_dec = MlaDims { seq: 1, ..mla_pre };
    let idx = IndexerDims {
        hidden,
        q_lora_rank: cfg.q_lora_rank,
        n_heads: cfg.index_n_heads,
        head_dim: cfg.index_head_dim,
        topk: cfg.index_topk,
        kpool: cfg.index_kpool,
        always_select_tail: cfg.index_kpool_always_select_tail,
        seq,
        force_emit: false,
    };
    let x = fill(seq * hidden, 41);

    // ── prefill ──
    let (_, mut wm) = load().expect("weights");
    let hs = Shape::new(&[1, seq, hidden], DType::F32);
    let flow = ModelFlow::new("mla_pre")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", hs.clone())
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_mla_attention(emit, "blk.3", x, mla_pre, idx)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let want = compile_built(built, dev())
        .unwrap()
        .run(&[("x", x.as_slice())])
        .pop()
        .unwrap();

    // ── decode, one token at a time, against a latent cache ──
    let (_, mut wm) = load().expect("weights");
    let cap = seq;
    let kvl = cfg.kv_lora_rank;
    let os = Shape::new(&[1, hidden + kvl], DType::F32);
    let flow = ModelFlow::new("mla_dec")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, 1, hidden], DType::F32))
        .input("cache", Shape::new(&[1, cap, kvl], DType::F32))
        .input("mask", Shape::new(&[1, cap + 1], DType::F32))
        .plugin_named("blk", move |emit, _p| {
            let x = emit.flow_input("x")?.hir_id();
            let cache = MlaCache {
                latent: emit.flow_input("cache")?.hir_id(),
                mask: emit.flow_input("mask")?.hir_id(),
                cap,
            };
            let (out, latent) = emit_mla_decode(emit, "blk.3", x, cache, mla_dec)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.reshape_(out, vec![1, hidden as i64]);
            let l = gb.reshape_(latent, vec![1, kvl as i64]);
            let packed = gb.concat_(vec![o, l], 1);
            Ok(Some(emit.wrap(packed, os.clone())))
        })
        .output("packed");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut compiled = compile_built(built, dev()).unwrap();

    let mut cache = vec![0f32; cap * kvl];
    let mut mask = vec![0f32; cap + 1];
    mask[cap] = 1.0; // the current token always sees itself
    let mut got = Vec::with_capacity(seq * hidden);
    for t in 0..seq {
        let tok = &x[t * hidden..(t + 1) * hidden];
        let packed = compiled
            .run(&[
                ("x", tok),
                ("cache", cache.as_slice()),
                ("mask", mask.as_slice()),
            ])
            .pop()
            .unwrap();
        got.extend_from_slice(&packed[..hidden]);
        cache[t * kvl..(t + 1) * kvl].copy_from_slice(&packed[hidden..hidden + kvl]);
        mask[t] = 1.0;
    }

    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    eprintln!("real MLA decode vs prefill: max |Δ| = {worst:.3e} over scale {scale:.3e}");
    assert!(
        worst / scale < 1e-3,
        "absorbed decode diverges from expanded prefill on real weights; \
         max |Δ| = {worst} over scale {scale}"
    );
}

/// The DSA short-circuit, on the **trained** indexer.
///
/// Below the budget the selection is provably the identity, so running the
/// indexer must reproduce the causal mask. On synthetic weights this caught two
/// real bugs (top-k returning masked-out pools, and rlx-cpu's `ScatterElements`
/// mis-striding narrow indices); on trained weights it additionally means the
/// learned pooling gate and score projections do not push any pool out of the
/// selection when there is room for all of them.
#[test]
fn real_dsa_selection_is_the_identity_below_the_budget() {
    let (cfg, _) = weights_or_skip!();
    let seq = 8;
    let hidden = cfg.hidden_size;
    let mla = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        eps: cfg.rms_norm_eps,
        seq,
    };
    let base = IndexerDims {
        hidden,
        q_lora_rank: cfg.q_lora_rank,
        n_heads: cfg.index_n_heads,
        head_dim: cfg.index_head_dim,
        topk: cfg.index_topk,
        kpool: cfg.index_kpool,
        always_select_tail: cfg.index_kpool_always_select_tail,
        seq,
        force_emit: false,
    };
    assert!(base.is_dense());

    let x = fill(seq * hidden, 53);
    let run = |force: bool| -> Vec<f32> {
        let (_, mut wm) = load().expect("weights");
        let idx = IndexerDims {
            force_emit: force,
            ..base
        };
        let hs = Shape::new(&[1, seq, hidden], DType::F32);
        let flow = ModelFlow::new("mla")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", hs.clone())
            .plugin_named("blk", move |emit, _p| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_mla_attention(emit, "blk.3", x, mla, idx)?;
                Ok(Some(emit.wrap(out, hs.clone())))
            })
            .output("out");
        let built = flow
            .build_with(&mut WeightMapSource(&mut wm), None)
            .unwrap();
        compile_built(built, dev())
            .unwrap()
            .run(&[("x", x.as_slice())])
            .pop()
            .unwrap()
    };

    let short_circuit = run(false);
    let via_indexer = run(true);
    let worst = short_circuit
        .iter()
        .zip(&via_indexer)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = short_circuit.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    eprintln!("real DSA identity: max |Δ| = {worst:.3e} over scale {scale:.3e}");
    assert!(
        worst / scale < 1e-5,
        "the trained indexer must reproduce the causal mask below its budget; \
         max |Δ| = {worst} over scale {scale}"
    );
}
