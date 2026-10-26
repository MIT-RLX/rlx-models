// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `rlx-qwen4exp` against **real published weights**.
//!
//! Qwen3.8-Flash-Next's smallest quantization is tens of gigabytes, so this
//! runs on a range-fetched subset instead:
//!
//! ```sh
//! python3 scripts/qwen4exp_subset.py out.gguf     # ~186 MB
//! RLX_QWEN4EXP_GGUF=out.gguf cargo test -p rlx-qwen4exp --test real_weights
//! ```
//!
//! The subset carries `blk.1` (linear attention **plus the model's one PLE
//! module**) and `blk.3` (gated attention plus the QSA indexer), with 8 of each
//! block's 512 routed experts — the model's two layer kinds, and between them
//! every emitter in the crate except the LM head. Without the env var every
//! test here passes trivially, so CI stays offline.
//!
//! What real weights buy that synthetic ones cannot:
//!
//! * **Sign and scale facts.** `ssm_a` being the converter's precomputed
//!   `-exp(A_log)` rather than a raw `A_log` is the difference between a
//!   working GatedDeltaNet and a memoryless one, and only the real tensor can
//!   say which it is. Same for the norm gains being pre-offset by 1.
//! * **A non-degenerate activation scale.** Synthetic weights are uniform
//!   noise; a transposed projection or a mis-split fused tensor often still
//!   produces finite output from them. Trained weights have structure, so the
//!   output distribution is a real signal.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, Emit, ModelFlow};
use rlx_ir::{DType, HirNodeId, Shape};
use rlx_qwen4exp::Qwen4ExpConfig;
use rlx_qwen4exp::attention::{AttnDims, emit_attention};
use rlx_qwen4exp::config::LayerKind;
use rlx_qwen4exp::gdn::{GdnDims, emit_gdn};
use rlx_qwen4exp::hc::{HcDims, emit_hc_combine, emit_hc_mix, emit_hc_split};
use rlx_qwen4exp::indexer::QsaDims;
use rlx_qwen4exp::moe::{MoeDims, emit_moe};
use rlx_qwen4exp::ple::{PleDims, emit_ple};
use rlx_qwen4exp::rope::{mrope_tables, table_width, text_section_pos};
use rlx_runtime::Device;

/// Experts per bank in this file — see [`gguf_expert_count`], which is where
/// the derivation lives so a test cannot drift from what the runner does.
fn experts_in_file(path: &str) -> usize {
    rlx_qwen4exp::moe::gguf_expert_count(path)
        .expect("reading the subset header")
        .expect("the subset should carry at least one routed expert bank")
}

fn gguf_path() -> Option<String> {
    rlx_ir::env::var_result("RLX_QWEN4EXP_GGUF")
        .ok()
        .filter(|s| !s.is_empty())
}

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// Load the subset's config and weights.
///
/// `WeightMap::from_file` is the *safetensors* path; a GGUF has to go through
/// the format registry, and `_dequant_all` because the emitters here are given
/// a dense map rather than a packed source.
fn load() -> Option<(Qwen4ExpConfig, WeightMap)> {
    let path = gguf_path()?;
    let mut cfg = Qwen4ExpConfig::from_gguf_path(&path).expect("parse the subset's metadata");
    // The banks were sliced; the metadata still describes all 512, so take the
    // count from the tensors that are actually there.
    let experts = experts_in_file(&path);
    cfg.num_experts = experts;
    cfg.num_experts_per_tok = cfg.num_experts_per_tok.min(experts);
    let mut loader = rlx_core::weight_loader::load_from_path(&path).expect("open the subset GGUF");
    let weights =
        WeightMap::from_weight_loader_dequant_all(loader.as_mut()).expect("dequantize the subset");
    Some((cfg, weights))
}

fn stats(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// A deterministic hidden state in roughly the range a real residual carries.
fn hidden_state(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect()
}

/// The metadata in the subset is the published model's, verbatim.
#[test]
fn the_subsets_metadata_is_the_published_model() {
    let Some(path) = gguf_path() else { return };
    let cfg = Qwen4ExpConfig::from_gguf_path(&path).expect("parse");
    let published = Qwen4ExpConfig::qwen38_flash_next();
    assert_eq!(cfg.num_hidden_layers, published.num_hidden_layers);
    assert_eq!(cfg.hidden_size, published.hidden_size);
    assert_eq!(cfg.layer_types, published.layer_types);
    assert_eq!(cfg.ple_layers, published.ple_layers);
    assert_eq!(cfg.hc_count, published.hc_count);
    assert_eq!(cfg.indexer_budget, published.indexer_budget);
    // Reaching here means `validate` accepted the checkpoint's own PLE hash
    // constants against the ones derived from `(seed, vocab_size)`.
    assert!(cfg.ple_constants.is_some());
    assert_eq!(cfg.layer_kind(1), LayerKind::Linear, "blk.1 carries PLE");
    assert_eq!(cfg.layer_kind(3), LayerKind::Indexed);
}

/// The two sign/offset conventions this crate reads off the GGUF rather than
/// deriving, checked against the real tensors.
///
/// Both are silent if wrong: a positive `ssm_a` makes the recurrence diverge
/// instead of decay, and a zero-centred gamma scales every activation by
/// roughly zero. Neither raises.
#[test]
fn converter_conventions_hold_on_the_real_tensors() {
    let Some((cfg, weights)) = load() else { return };

    // `ssm_a` is `-exp(A_log)`: strictly negative, and spread over decades
    // because `A` was initialized uniform on [0.01, 16].
    let (a, _) = weights.get("blk.1.ssm_a").expect("blk.1.ssm_a");
    assert_eq!(a.len(), cfg.n_v_heads());
    assert!(
        a.iter().all(|x| *x < 0.0),
        "ssm_a must be the precomputed -exp(A_log); got a non-negative entry"
    );
    let min = a.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min < -1.0 && max > -1.0,
        "ssm_a should span decades (saw [{min}, {max}]); a range hugging -1 \
         would mean an extra -exp() was applied"
    );

    // Every `Qwen4ExpTextRMSNorm` gain is stored as `1 + w`, so it sits near 1.
    for key in [
        "blk.1.hc_attn_norm.weight",
        "blk.3.hc_ffn_norm.weight",
        "blk.1.ple_norm_key.weight",
        "blk.3.indexer.q_norm.weight",
        "blk.3.attn_q_norm.weight",
    ] {
        let (g, _) = weights.get(key).unwrap_or_else(|| panic!("{key} missing"));
        let (mean, _) = stats(g);
        assert!(
            (0.3..3.0).contains(&mean),
            "{key} has mean {mean}; a GGUF gain is pre-offset by 1, so a mean \
             near 0 means rms_norm_offset is set the wrong way"
        );
    }
    assert!(!cfg.rms_norm_offset);

    // The gated norm is the exception: `RMSNormGated`'s weight is used
    // directly, initialized to ones, and the converter does not shift it.
    let (g, _) = weights.get("blk.1.ssm_norm.weight").expect("ssm_norm");
    let (mean, _) = stats(g);
    assert!((0.3..3.0).contains(&mean), "ssm_norm mean {mean}");
}

/// Run `blk.1` — the hyper-connection site, the GatedDeltaNet, the PLE module
/// and the MoE — on real values.
#[test]
fn linear_block_with_ple_runs_on_real_weights() {
    let Some((cfg, mut weights)) = load() else {
        return;
    };
    let seq = 8;
    let hidden = cfg.hidden_size;
    let (hc, gdn, moe, ple) = (
        dims_hc(&cfg, seq),
        dims_gdn(&cfg, seq),
        dims_moe(&cfg, seq),
        dims_ple(&cfg, seq),
    );
    let stream = Shape::new(&[1, seq, cfg.hc_count, hidden], DType::F32);
    let emb_dim = cfg.ple_embed_dim();

    let flow = ModelFlow::new("qwen4exp_real_linear")
        .with_profile(CompileProfile::llama32_prefill())
        .input("h", Shape::new(&[1, seq, hidden], DType::F32))
        .input("emb", Shape::new(&[seq, emb_dim], DType::F32))
        .plugin_named("blk1", move |emit, _| {
            let h = emit.flow_input("h")?.hir_id();
            let mut x = emit_hc_split(emit, h, hc);
            let emb = emit.flow_input("emb")?.hir_id();
            x = emit_ple(emit, "blk.1", x, emb, ple)?;

            let gates = emit_hc_mix(emit, "blk.1.hc_attn", x, hc, true)?;
            let branch = emit_gdn(emit, "blk.1", gates.mixed, gdn)?;
            let x = emit_hc_combine(emit, branch, x, gates, hc);

            let gates = emit_hc_mix(emit, "blk.1.hc_ffn", x, hc, true)?;
            let branch = emit_moe(emit, "blk.1", gates.mixed, moe)?;
            let out = emit_hc_combine(emit, branch, x, gates, hc);
            Ok(Some(emit.wrap(out, stream.clone())))
        })
        .output("y");

    let built = flow
        .build_with(
            &mut rlx_core::flow_util::WeightMapSource(&mut weights),
            None,
        )
        .expect("build blk.1");
    let h = hidden_state(seq * hidden);
    let emb: Vec<f32> = (0..seq * emb_dim)
        .map(|i| ((i as f32) * 0.011).cos() * 0.05)
        .collect();
    let out = compile_built(built, dev())
        .expect("compile blk.1")
        .run(&[("h", h.as_slice()), ("emb", emb.as_slice())])
        .pop()
        .expect("blk.1 output");

    assert_eq!(out.len(), seq * cfg.hc_count * hidden);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "blk.1 produced non-finite values"
    );
    let (mean, sd) = stats(&out);
    // The residual is the input broadcast into 4 streams plus two injections;
    // a transposed projection or a mis-split `attn_qkv` blows this up or
    // collapses it, so bound it rather than only checking finiteness.
    assert!(
        sd > 1e-4 && sd < 50.0,
        "blk.1 output sd {sd} (mean {mean}) is outside anything a residual \
         stream should carry"
    );
}

/// Run `blk.3` — gated GQA attention behind the QSA indexer, plus its MoE.
#[test]
fn attention_block_with_qsa_runs_on_real_weights() {
    let Some((cfg, mut weights)) = load() else {
        return;
    };
    let seq = 8;
    let hidden = cfg.hidden_size;
    let (hc, attn, moe) = (
        dims_hc(&cfg, seq),
        dims_attn(&cfg, seq),
        dims_moe(&cfg, seq),
    );
    let qsa = dims_qsa(&cfg, seq, false);
    let stream = Shape::new(&[1, seq, cfg.hc_count, hidden], DType::F32);
    let (cos_d, sin_d) = mrope_tables(&cfg, seq, None);
    let width = table_width(&cfg);

    let flow = ModelFlow::new("qwen4exp_real_attn")
        .with_profile(CompileProfile::llama32_prefill())
        .input("h", Shape::new(&[1, seq, hidden], DType::F32))
        .plugin_named("blk3", move |emit, _| {
            let (cos, sin) = rope_params(emit, seq, width, &cos_d, &sin_d);
            let h = emit.flow_input("h")?.hir_id();
            let x = emit_hc_split(emit, h, hc);

            let gates = emit_hc_mix(emit, "blk.3.hc_attn", x, hc, true)?;
            let branch = emit_attention(emit, "blk.3", gates.mixed, cos, sin, attn, Some(qsa))?;
            let x = emit_hc_combine(emit, branch, x, gates, hc);

            let gates = emit_hc_mix(emit, "blk.3.hc_ffn", x, hc, true)?;
            let branch = emit_moe(emit, "blk.3", gates.mixed, moe)?;
            let out = emit_hc_combine(emit, branch, x, gates, hc);
            Ok(Some(emit.wrap(out, stream.clone())))
        })
        .output("y");

    let built = flow
        .build_with(
            &mut rlx_core::flow_util::WeightMapSource(&mut weights),
            None,
        )
        .expect("build blk.3");
    let h = hidden_state(seq * hidden);
    let out = compile_built(built, dev())
        .expect("compile blk.3")
        .run(&[("h", h.as_slice())])
        .pop()
        .expect("blk.3 output");

    assert_eq!(out.len(), seq * cfg.hc_count * hidden);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "blk.3 produced non-finite values"
    );
    let (mean, sd) = stats(&out);
    assert!(
        sd > 1e-4 && sd < 50.0,
        "blk.3 output sd {sd} (mean {mean}) is outside anything a residual \
         stream should carry"
    );
}

/// The QSA short-circuit, checked against the machinery it skips.
///
/// Below the budget the selection is the identity, so [`emit_attention`] drops
/// the indexer entirely and uses the fused causal kernel. That claim is an
/// algebraic one about block-plus-tail visibility; this is the test that it
/// holds *numerically*, on real weights, by running the full selection
/// (`force_emit`) and the short-circuit and comparing.
///
/// This is deliberately an equivalence against an *independently spelled*
/// path — a mask built from scores, top-k, a scatter and a bias, versus a
/// kernel-synthesized causal mask. A self-consistent test (dense config vs
/// sparse config) cannot find anything here, because both sides short-circuit.
#[test]
fn qsa_selection_is_the_identity_below_the_budget() {
    let Some((cfg, _)) = load() else { return };
    let seq = 8;
    assert!(
        cfg.qsa_is_dense(seq),
        "this test needs a length where the short-circuit applies"
    );

    let run = |force: bool| -> Vec<f32> {
        let (_, mut weights) = load().expect("weights");
        let hidden = cfg.hidden_size;
        let attn = dims_attn(&cfg, seq);
        let qsa = dims_qsa(&cfg, seq, force);
        let (cos_d, sin_d) = mrope_tables(&cfg, seq, None);
        let width = table_width(&cfg);
        let out_shape = Shape::new(&[1, seq, hidden], DType::F32);
        let flow = ModelFlow::new("qwen4exp_qsa")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", out_shape.clone())
            .plugin_named("attn", move |emit, _| {
                let (cos, sin) = rope_params(emit, seq, width, &cos_d, &sin_d);
                let x = emit.flow_input("x")?.hir_id();
                let y = emit_attention(emit, "blk.3", x, cos, sin, attn, Some(qsa))?;
                Ok(Some(emit.wrap(y, out_shape.clone())))
            })
            .output("y");
        let built = flow
            .build_with(
                &mut rlx_core::flow_util::WeightMapSource(&mut weights),
                None,
            )
            .expect("build attention");
        let x = hidden_state(seq * hidden);
        compile_built(built, dev())
            .expect("compile attention")
            .run(&[("x", x.as_slice())])
            .pop()
            .expect("attention output")
    };

    let dense = run(false);
    let selected = run(true);
    assert_eq!(dense.len(), selected.len());
    assert!(dense.iter().all(|v| v.is_finite()));
    let max_rel = dense
        .iter()
        .zip(&selected)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    // Not bit-exact: one path softmaxes against a kernel-synthesized mask, the
    // other against a `-1e30` bias, and the two differ in the last bits.
    assert!(
        max_rel < 2e-3,
        "the QSA selection and the dense short-circuit disagree by {max_rel} \
         relative — the identity claim does not hold"
    );
}

// ── dims helpers, so each test states only what it varies ──

fn dims_hc(cfg: &Qwen4ExpConfig, seq: usize) -> HcDims {
    HcDims::from_config(cfg, seq)
}

fn dims_gdn(cfg: &Qwen4ExpConfig, seq: usize) -> GdnDims {
    GdnDims::from_config(cfg, seq)
}

fn dims_attn(cfg: &Qwen4ExpConfig, seq: usize) -> AttnDims {
    AttnDims::from_config(cfg, seq)
}

fn dims_qsa(cfg: &Qwen4ExpConfig, seq: usize, force_emit: bool) -> QsaDims {
    QsaDims {
        force_emit,
        ..QsaDims::from_config(cfg, seq)
    }
}

fn dims_moe(cfg: &Qwen4ExpConfig, seq: usize) -> MoeDims {
    MoeDims::from_config(cfg, seq)
}

fn dims_ple(cfg: &Qwen4ExpConfig, seq: usize) -> PleDims {
    PleDims::from_config(cfg, seq)
}

fn rope_params(
    emit: &mut Emit<'_>,
    seq: usize,
    width: usize,
    cos: &[f32],
    sin: &[f32],
) -> (HirNodeId, HirNodeId) {
    let s = Shape::new(&[seq, width], DType::F32);
    (
        emit.synth_param("rope.cos", cos.to_vec(), s.clone()),
        emit.synth_param("rope.sin", sin.to_vec(), s),
    )
}

/// A two-layer model carved out of the subset: the real `blk.1` (GatedDeltaNet
/// + PLE) as layer 0, the real `blk.3` (gated attention + QSA) as layer 1.
///
/// The subset keeps the published block indices, so this renames them. Nothing
/// about the weights changes — only which layer slot the emitters look them up
/// under — which is what lets the whole-stack step flow run on a 186 MB file.
fn two_layer(cfg: &Qwen4ExpConfig) -> (Qwen4ExpConfig, WeightMap) {
    let mut c = cfg.clone();
    c.num_hidden_layers = 2;
    c.layer_types = vec![LayerKind::Linear, LayerKind::Indexed];
    c.compress_ratios = vec![0, c.indexer_compress_ratio];
    c.ple_layers = vec![0];
    c.num_nextn_predict_layers = 0;

    let path = gguf_path().expect("caller checked");
    let mut loader = rlx_core::weight_loader::load_from_path(&path).expect("open subset");
    let full = WeightMap::from_weight_loader_dequant_all(loader.as_mut()).expect("dequantize");
    let mut renamed: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> =
        std::collections::HashMap::new();
    for key in full.keys().map(str::to_string).collect::<Vec<_>>() {
        let (data, shape) = full.get(&key).expect("present");
        let (data, shape) = (data.to_vec(), shape.to_vec());
        let out = match key.strip_prefix("blk.1.") {
            Some(rest) => format!("blk.0.{rest}"),
            None => match key.strip_prefix("blk.3.") {
                Some(rest) => format!("blk.1.{rest}"),
                None => key.clone(),
            },
        };
        renamed.insert(out, (data, shape));
    }
    (c, WeightMap::from_tensors(renamed))
}

/// **Decode reproduces prefill on real weights.**
///
/// `tests/decode_equivalence.rs` runs this on synthetic weights, where every
/// tensor is uniform noise. Trained weights have structure — the decay
/// coefficients span decades, the convolution kernels are not symmetric, the
/// gate biases are far from zero — so an off-by-one in a carried window or a
/// mis-resumed scan shows up here at a magnitude that noise can hide.
#[test]
fn decode_reproduces_prefill_on_real_weights() {
    let Some((base, _)) = load() else { return };
    let (cfg, _) = two_layer(&base);
    let seq = 4usize;
    let hidden = cfg.hidden_size;
    let emb_dim = cfg.ple_embed_dim();

    // The subset carries no `token_embd`, so both graphs take `inputs_embeds`.
    let embeds: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let ple: Vec<f32> = (0..seq * emb_dim)
        .map(|i| ((i as f32) * 0.007).cos() * 0.05)
        .collect();
    let ids: Vec<f32> = (0..seq).map(|t| (t + 3) as f32).collect();

    // ── stateless prefill ──
    let prefill = {
        let (_, mut w) = two_layer(&base);
        let mut opts = rlx_qwen4exp::FlowOptions::prefill(seq);
        opts.output = rlx_qwen4exp::OutputKind::Hidden;
        opts.host_embed = true;
        opts.runtime_mrope = true;
        let built = rlx_qwen4exp::build_qwen4exp_text_flow_with_options(
            &cfg,
            &mut rlx_core::flow_util::WeightMapSource(&mut w),
            opts,
        )
        .expect("build prefill");
        let (cos, sin) = mrope_tables(&cfg, seq, None);
        compile_built(built, dev())
            .expect("compile prefill")
            .run(&[
                ("input_ids", ids.as_slice()),
                ("inputs_embeds", embeds.as_slice()),
                (rlx_qwen4exp::ROPE_COS_INPUT, cos.as_slice()),
                (rlx_qwen4exp::ROPE_SIN_INPUT, sin.as_slice()),
                (&rlx_qwen4exp::ple_embed_input(0), ple.as_slice()),
            ])
            .pop()
            .expect("prefill hidden")
    };
    assert_eq!(prefill.len(), seq * hidden);

    // ── one token at a time ──
    let (_, mut w) = two_layer(&base);
    let (built, layout) = rlx_qwen4exp::decode::build_qwen4exp_step_flow_with(
        &cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        seq,
        1,
        rlx_qwen4exp::OutputKind::Hidden,
        rlx_qwen4exp::ScanState::Portable,
        true,
    )
    .expect("build step");
    let names = rlx_qwen4exp::DecodeNames::new(&cfg);
    let mut session = rlx_qwen4exp::DecodeSession::new(&cfg, layout, seq);
    let mut compiled = compile_built(built, dev()).expect("compile step");

    for t in 0..seq {
        let (cos, sin) = mrope_tables(&cfg, 1, Some(&[text_section_pos(t)]));
        let id = [ids[t]];
        let e = &embeds[t * hidden..(t + 1) * hidden];
        let p = &ple[t * emb_dim..(t + 1) * emb_dim];
        let ple_name = rlx_qwen4exp::ple_embed_input(0);
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", &id),
            ("inputs_embeds", e),
            (rlx_qwen4exp::ROPE_COS_INPUT, cos.as_slice()),
            (rlx_qwen4exp::ROPE_SIN_INPUT, sin.as_slice()),
            (ple_name.as_str(), p),
        ];
        inputs.extend(session.inputs(&cfg, &names));
        let packed = compiled.run(&inputs).pop().expect("packed");
        let got = session.commit(&packed).expect("commit").to_vec();

        let want = &prefill[t * hidden..(t + 1) * hidden];
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
        assert!(
            max_abs / scale < 5e-3,
            "token {t}: real-weight decode and prefill differ by {max_abs} \
             ({:.3e} relative to a {scale} scale)",
            max_abs / scale
        );
    }
}

/// Paging real, quantized expert banks against the same layer resident.
///
/// The first exercise of the paged path on published weights, and the one that
/// matters: `blk.1`'s banks are **IQ2_XXS** (gate, up) and **IQ4_NL** (down) —
/// a paged read has to dequantize each fired expert on its way into the slot,
/// and the dtype is read out of the header rather than assumed because it
/// differs per bank and per layer (`blk.0`'s gate is IQ1_S).
///
/// The oracle is the same layer over the whole dequantized banks, so a wrong
/// expert offset, a misread dtype or a slot in the wrong order all show up as
/// a numeric disagreement rather than as plausible output.
#[test]
fn paging_real_quantized_banks_matches_the_resident_layer() {
    use rlx_distributed::{ExpertPager, register_gguf_expert_banks};
    use rlx_qwen4exp::paged_moe::{BANKS, PagedMoeLayer};
    use std::collections::HashMap;
    use std::sync::Arc;

    let Some(path) = gguf_path() else {
        eprintln!("RLX_QWEN4EXP_GGUF unset; skipping");
        return;
    };
    let Some((cfg, mut weights)) = load() else {
        return;
    };
    let seq = 2usize;
    // Which block to page is also the file's to say, not this test's: take the
    // first one whose routed banks are present.
    let layer = {
        let g = rlx_gguf::GgufFile::header_from_path(&path).expect("header");
        let mut found: Vec<usize> = g
            .tensors
            .values()
            .filter_map(|t| {
                t.name
                    .strip_prefix("blk.")?
                    .strip_suffix(".ffn_gate_exps.weight")?
                    .parse()
                    .ok()
            })
            .collect();
        found.sort_unstable();
        *found
            .first()
            .expect("the subset should carry a routed expert bank")
    };
    let prefix = format!("blk.{layer}");
    let prefix = prefix.as_str();
    let d = MoeDims::from_config(&cfg, seq);
    assert_eq!(d.num_experts, experts_in_file(&path));

    // Everything the layer reads except the banks; the paged layer supplies its
    // own slot-width placeholders for those.
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for key in [
        "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ] {
        let name = format!("{prefix}.{key}");
        let (data, shape) = weights
            .take(&name)
            .unwrap_or_else(|e| panic!("the subset should carry {name}: {e:#}"));
        tensors.insert(name, (data, shape));
    }
    // The dequantized banks, for the oracle only.
    let mut resident_tensors = tensors.clone();
    for bank in BANKS {
        let name = format!("{prefix}.{bank}.weight");
        let (data, shape) = weights
            .take(&name)
            .unwrap_or_else(|e| panic!("the subset should carry {name}: {e:#}"));
        resident_tensors.insert(name, (data, shape));
    }

    let x = hidden_state(seq * cfg.hidden_size);

    // ── resident: all eight experts, dequantized up front ──
    let want = {
        let mut wm = WeightMap::from_tensors(resident_tensors);
        let shape = Shape::new(&[1, seq, cfg.hidden_size], DType::F32);
        let s = shape.clone();
        let built = ModelFlow::new("qwen4exp_resident_real_moe")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", shape)
            .plugin_named("moe", {
                let prefix = prefix.to_string();
                move |emit, _prev| {
                    let inp = emit.flow_input("x")?.hir_id();
                    let out = emit_moe(emit, &prefix, inp, d)?;
                    Ok(Some(emit.wrap(out, s.clone())))
                }
            })
            .output("y")
            .build_with(&mut rlx_core::flow_util::WeightMapSource(&mut wm), None)
            .expect("build resident moe");
        compile_built(built, dev())
            .expect("compile resident moe")
            .run(&[("x", x.as_slice())])
            .pop()
            .expect("resident output")
    };
    assert!(want.iter().all(|v| v.is_finite()));
    let (_, sd) = stats(&want);
    assert!(sd > 1e-6, "the resident layer produced a constant: sd {sd}");

    // ── paged: the quantized banks, read per fired expert ──
    let mut pager = ExpertPager::new(64 << 20);
    let n = register_gguf_expert_banks(
        &mut pager,
        &path,
        &[
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ],
    )
    .expect("register the subset's banks");
    assert!(n >= 3, "expected at least blk.1's three banks, got {n}");

    let mut layer =
        PagedMoeLayer::from_gguf(prefix, layer, d, dev(), tensors, Arc::new(pager), &path)
            .expect("build the paged layer over real quantized banks");
    let got = layer.forward(&x).expect("paged forward");

    let max_abs = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
    assert!(
        max_abs / scale < 1e-5,
        "paged and resident real layers differ by {max_abs} \
         ({:.3e} relative to a {scale} scale)",
        max_abs / scale
    );

    // What it cost: the quantized bytes of what fired, not the whole banks.
    let s = layer.stats();
    eprintln!(
        "{prefix} paged: read {} KB for {} experts of {}, resident would be {} MB",
        s.bytes_read / 1024,
        d.top_k * seq,
        d.num_experts,
        layer.resident_bytes() / (1 << 20)
    );
    assert!(s.bytes_read > 0, "nothing was read from disk");
}
