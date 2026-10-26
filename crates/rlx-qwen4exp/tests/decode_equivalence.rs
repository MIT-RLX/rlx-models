// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Stepping the decode graph must reproduce a prefill of the same tokens.
//!
//! This is the load-bearing test for [`rlx_qwen4exp::decode`], because prefill
//! and decode spell **every stateful piece differently**:
//!
//! | piece | prefill | decode |
//! |---|---|---|
//! | GatedDeltaNet conv | one grouped `Op::Conv`, zero left-pad | sum of shifted copies over a carried window |
//! | GatedDeltaNet scan | `Op::GatedDeltaNet` | `…_carry` resuming a state matrix |
//! | PLE conv | grouped `Op::Conv`, dilation 3 | carried, dilation-aware taps |
//! | attention mask | `MaskKind::Causal`, synthesized | `MaskKind::Custom` over a `[1, cap+1]` validity row |
//! | n-gram hash window | the whole token array | a 2-token host-side history |
//!
//! Two independently spelled paths agreeing to f32 noise is evidence; a test
//! that compared decode to itself would be vacuous. The hazard is concrete:
//! every one of those five pieces has a plausible off-by-one that leaves the
//! output finite and the shapes right.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_qwen4exp::decode::{
    DecodeNames, DecodeSession, ScanState, build_qwen4exp_decode_flow_with,
};
use rlx_qwen4exp::rope::{mrope_tables, text_section_pos};
use rlx_qwen4exp::{
    DensePleTable, FlowOptions, OutputKind, PleHash, PleRowSource, Qwen4ExpConfig, ROPE_COS_INPUT,
    ROPE_SIN_INPUT, build_qwen4exp_text_flow_with_options, ngram_embeddings, ple_embed_input,
    synth,
};
use rlx_runtime::Device;

const TOKENS: [u32; 6] = [3, 11, 7, 11, 2, 19];

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// A table whose rows are a deterministic function of the row index, so prefill
/// and decode gather identical values for identical row indices — which is the
/// point: any disagreement then comes from the *indices*, i.e. the hash window.
fn table(cfg: &Qwen4ExpConfig) -> DensePleTable {
    let rows = PleHash::derive(cfg, 0).padded_rows as usize;
    let hd = cfg.ple_head_dim;
    DensePleTable {
        data: (0..rows * hd)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.004)
            .collect(),
        rows,
        head_dim: hd,
    }
}

fn weights(cfg: &Qwen4ExpConfig) -> WeightMap {
    let mut t = synth::synth_tensors(cfg, 11);
    // Never a graph parameter: the rows are gathered host-side.
    t.remove("per_layer_token_embd.weight");
    WeightMap::from_tensors(t)
}

#[test]
fn stepping_decode_reproduces_prefill() {
    let cfg = synth::tiny_config(2048);
    let seq = TOKENS.len();
    let hidden = cfg.hidden_size;
    let mut tbl = table(&cfg);
    let hash = PleHash::derive(&cfg, 0);

    // ── prefill: all `seq` tokens at once, hidden states out ──
    let prefill = {
        let mut w = weights(&cfg);
        let mut opts = FlowOptions::prefill(seq);
        opts.output = OutputKind::Hidden;
        opts.runtime_mrope = true;
        let built = build_qwen4exp_text_flow_with_options(
            &cfg,
            &mut rlx_core::flow_util::WeightMapSource(&mut w),
            opts,
        )
        .expect("build prefill");
        let (cos, sin) = mrope_tables(&cfg, seq, None);
        let emb = ngram_embeddings(&hash, &TOKENS, &[], cfg.ple_head_dim, |r| tbl.row(r))
            .expect("prefill ple rows");
        let ids: Vec<f32> = TOKENS.iter().map(|t| *t as f32).collect();
        compile_built(built, dev())
            .expect("compile prefill")
            .run(&[
                ("input_ids", ids.as_slice()),
                (ROPE_COS_INPUT, cos.as_slice()),
                (ROPE_SIN_INPUT, sin.as_slice()),
                (&ple_embed_input(0), emb.as_slice()),
            ])
            .pop()
            .expect("prefill hidden")
    };
    assert_eq!(prefill.len(), seq * hidden);

    // ── decode: one token per run, state threaded forward ──
    let cap = seq;
    let mut w = weights(&cfg);
    let (built, layout) = build_qwen4exp_decode_flow_with(
        &cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        cap,
        OutputKind::Hidden,
        ScanState::Portable,
    )
    .expect("build decode");
    let names = DecodeNames::new(&cfg);
    let mut session = DecodeSession::new(&cfg, layout, cap);
    let mut compiled = compile_built(built, dev()).expect("compile decode");

    for (t, &tok) in TOKENS.iter().enumerate() {
        // The embeddings for THIS step must be hashed against the history as it
        // was before the token was pushed.
        let emb = session
            .ple_embeddings(&cfg, 0, tok, &mut tbl)
            .expect("decode ple rows");
        let (cos, sin) = mrope_tables(&cfg, 1, Some(&[text_section_pos(t)]));
        let ids = [tok as f32];

        let ple_name = ple_embed_input(0);
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", &ids),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (ple_name.as_str(), emb.as_slice()),
        ];
        inputs.extend(session.inputs(&cfg, &names));

        let packed = compiled.run(&inputs).pop().expect("decode packed");
        let got = session.commit(&packed).expect("commit").to_vec();
        session.push_token(tok);

        let want = &prefill[t * hidden..(t + 1) * hidden];
        assert_eq!(got.len(), hidden);
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
        assert!(
            max_abs / scale < 2e-3,
            "token {t}: decode and prefill differ by {max_abs} absolute \
             ({:.3e} relative to a {scale} scale)",
            max_abs / scale
        );
    }
}

/// The PLE hash window has to slide with the decode, not restart.
///
/// Checked directly, because the equivalence test above would also pass if the
/// hash were wrong *identically* in both paths — and it is the one piece of
/// state that lives on the host rather than in the graph.
#[test]
fn the_decode_hash_window_matches_the_prefill_one() {
    let cfg = synth::tiny_config(2048);
    let hash = PleHash::derive(&cfg, 0);
    let mut session = DecodeSession::new(&cfg, layout_of(&cfg), TOKENS.len());
    let mut tbl = table(&cfg);

    let whole = ngram_embeddings(&hash, &TOKENS, &[], cfg.ple_head_dim, |r| tbl.row(r)).unwrap();
    let per_head = cfg.ple_embed_dim();
    for (t, &tok) in TOKENS.iter().enumerate() {
        let step = session.ple_embeddings(&cfg, 0, tok, &mut tbl).unwrap();
        assert_eq!(
            step,
            whole[t * per_head..(t + 1) * per_head],
            "token {t}: the decode hash window diverged from the prefill one"
        );
        session.push_token(tok);
    }
}

fn layout_of(cfg: &Qwen4ExpConfig) -> rlx_qwen4exp::decode::DecodeLayout {
    let mut w = weights(cfg);
    let (_, layout) = build_qwen4exp_decode_flow_with(
        cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        TOKENS.len(),
        OutputKind::Hidden,
        ScanState::Portable,
    )
    .expect("build decode for its layout");
    layout
}

/// A decode capacity past the QSA budget must be refused, not silently run as
/// dense attention — that would be a different model from the trained one.
#[test]
fn decode_refuses_to_outrun_the_qsa_budget() {
    // budget 8, ratio 4 → 2 blocks fit; cap 12 means 3 complete blocks.
    let cfg = synth::tiny_config(8);
    let mut w = weights(&cfg);
    let err = build_qwen4exp_decode_flow_with(
        &cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        12,
        OutputKind::Hidden,
        ScanState::Portable,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("indexer_budget"), "{err}");
}

/// Chunked prefill must reproduce a single-shot prefill — including a short
/// final chunk that is right-padded.
///
/// This is the same equivalence as `stepping_decode_reproduces_prefill`, run at
/// `chunk > 1`, which additionally exercises the pieces a single-token step
/// cannot: the `[chunk, cap+chunk]` attention bias (causal among the new tokens
/// *and* masked over the prefix), a multi-token carried convolution, and the
/// padding plumbing — the `pad_mask` that stops the scan state advancing over
/// padding and the `conv_shift` that makes the exported convolution window end
/// at the last real token.
#[test]
fn chunked_prefill_reproduces_a_single_shot_prefill() {
    let cfg = synth::tiny_config(2048);
    let hidden = cfg.hidden_size;
    let hash = PleHash::derive(&cfg, 0);
    let mut tbl = table(&cfg);

    // Single-shot reference over all six tokens.
    let seq = TOKENS.len();
    let reference = {
        let mut w = weights(&cfg);
        let mut opts = FlowOptions::prefill(seq);
        opts.output = OutputKind::Hidden;
        opts.runtime_mrope = true;
        let built = build_qwen4exp_text_flow_with_options(
            &cfg,
            &mut rlx_core::flow_util::WeightMapSource(&mut w),
            opts,
        )
        .expect("build prefill");
        let (cos, sin) = mrope_tables(&cfg, seq, None);
        let emb = ngram_embeddings(&hash, &TOKENS, &[], cfg.ple_head_dim, |r| tbl.row(r)).unwrap();
        let ids: Vec<f32> = TOKENS.iter().map(|t| *t as f32).collect();
        compile_built(built, dev())
            .expect("compile prefill")
            .run(&[
                ("input_ids", ids.as_slice()),
                (ROPE_COS_INPUT, cos.as_slice()),
                (ROPE_SIN_INPUT, sin.as_slice()),
                (&ple_embed_input(0), emb.as_slice()),
            ])
            .pop()
            .expect("prefill hidden")
    };

    // `chunk = 4` over six tokens: one full chunk, then two real of four.
    let chunk = 4;
    let cap = 8;
    let mut w = weights(&cfg);
    let (built, layout) = rlx_qwen4exp::decode::build_qwen4exp_step_flow(
        &cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        cap,
        chunk,
        OutputKind::Hidden,
        ScanState::Portable,
    )
    .expect("build chunked step");
    let names = DecodeNames::new(&cfg);
    let mut session = DecodeSession::with_chunk(&cfg, layout, cap, chunk);
    let mut compiled = compile_built(built, dev()).expect("compile chunked step");

    let mut consumed = 0usize;
    let mut history: Vec<u32> = Vec::new();
    while consumed < seq {
        let real = (seq - consumed).min(chunk);
        session.set_real_len(real).expect("real_len");

        // Right-pad the chunk with EOS, which is what the reference substitutes
        // for masked PLE positions.
        let mut ids: Vec<f32> = Vec::with_capacity(chunk);
        let mut toks: Vec<u32> = Vec::with_capacity(chunk);
        for i in 0..chunk {
            let t = if i < real {
                TOKENS[consumed + i]
            } else {
                cfg.ple_eos_token_id
            };
            toks.push(t);
            ids.push(t as f32);
        }
        let emb = ngram_embeddings(&hash, &toks, &history, cfg.ple_head_dim, |r| tbl.row(r))
            .expect("chunk ple rows");
        let positions: Vec<[usize; 4]> =
            (0..chunk).map(|i| text_section_pos(consumed + i)).collect();
        let (cos, sin) = mrope_tables(&cfg, chunk, Some(&positions));

        let ple_name = ple_embed_input(0);
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", ids.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (ple_name.as_str(), emb.as_slice()),
        ];
        inputs.extend(session.inputs(&cfg, &names));

        let packed = compiled.run(&inputs).pop().expect("chunk packed");
        let got = session.commit(&packed).expect("commit").to_vec();
        assert_eq!(got.len(), chunk * hidden);

        for i in 0..real {
            let t = consumed + i;
            let want = &reference[t * hidden..(t + 1) * hidden];
            let have = &got[i * hidden..(i + 1) * hidden];
            let max_abs = have
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
            assert!(
                max_abs / scale < 2e-3,
                "token {t} (chunk offset {i}): chunked and single-shot differ \
                 by {max_abs} ({:.3e} relative)",
                max_abs / scale
            );
        }

        // Carry the real tokens into the next chunk's hash window.
        for &t in &toks[..real] {
            history.push(t);
        }
        let keep = cfg.ple_ngram_size - 1;
        let excess = history.len().saturating_sub(keep);
        history.drain(..excess);
        consumed += real;
    }
    assert_eq!(session.position(), seq);
}

/// `set_real_len` has to reject a length the compiled graph cannot express,
/// rather than silently masking the wrong number of tokens.
#[test]
fn a_real_len_outside_the_chunk_is_rejected() {
    let cfg = synth::tiny_config(2048);
    let mut w = weights(&cfg);
    let (_, layout) = rlx_qwen4exp::decode::build_qwen4exp_step_flow(
        &cfg,
        &mut rlx_core::flow_util::WeightMapSource(&mut w),
        8,
        4,
        OutputKind::Hidden,
        ScanState::Portable,
    )
    .expect("build");
    let mut s = DecodeSession::with_chunk(&cfg, layout, 8, 4);
    assert!(s.set_real_len(4).is_ok());
    assert!(s.set_real_len(0).is_err());
    assert!(s.set_real_len(5).is_err());
}

/// A right-padded chunk must export the same state as an exactly-sized one.
///
/// This is what `pad_mask` and `conv_shift` are for, and the only test that can
/// see them: within a chunk, every real position's *output* is already correct
/// without them, because the convolution and the scan are both causal. What
/// padding breaks is the **exported** state — the scan would advance over the
/// pad tokens, and the convolution window would be gathered from them — which
/// only shows up in whatever runs next.
///
/// So this compares the exported state directly: five real tokens through a
/// `chunk = 5` graph versus the same five right-padded into a `chunk = 8` one.
#[test]
fn a_padded_chunk_exports_the_same_state_as_an_exact_one() {
    let cfg = synth::tiny_config(2048);
    let hash = PleHash::derive(&cfg, 0);
    let real = 5usize;
    let cap = 16usize;

    let run = |chunk: usize| -> (Vec<f32>, rlx_qwen4exp::decode::DecodeLayout) {
        let mut tbl = table(&cfg);
        let mut w = weights(&cfg);
        let (built, layout) = rlx_qwen4exp::decode::build_qwen4exp_step_flow(
            &cfg,
            &mut rlx_core::flow_util::WeightMapSource(&mut w),
            cap,
            chunk,
            OutputKind::Hidden,
            ScanState::Portable,
        )
        .expect("build step");
        let names = DecodeNames::new(&cfg);
        let mut session = DecodeSession::with_chunk(&cfg, layout.clone(), cap, chunk);
        session.set_real_len(real).expect("real_len");

        let toks: Vec<u32> = (0..chunk)
            .map(|i| {
                if i < real {
                    TOKENS[i]
                } else {
                    cfg.ple_eos_token_id
                }
            })
            .collect();
        let ids: Vec<f32> = toks.iter().map(|t| *t as f32).collect();
        let emb = ngram_embeddings(&hash, &toks, &[], cfg.ple_head_dim, |r| tbl.row(r)).unwrap();
        let positions: Vec<[usize; 4]> = (0..chunk).map(text_section_pos).collect();
        let (cos, sin) = mrope_tables(&cfg, chunk, Some(&positions));

        let ple_name = ple_embed_input(0);
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", ids.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (ple_name.as_str(), emb.as_slice()),
        ];
        inputs.extend(session.inputs(&cfg, &names));
        let packed = compile_built(built, dev())
            .expect("compile step")
            .run(&inputs)
            .pop()
            .expect("packed");
        (packed, layout)
    };

    let (exact, l_exact) = run(real);
    let (padded, l_padded) = run(8);

    // The GatedDeltaNet windows and scan matrices, and the PLE window, are all
    // chunk-independent — so they must agree value for value.
    let mut compared = 0;
    for (&(oa, ca, sa), &(ob, cb, sb)) in l_exact.gdn.iter().zip(&l_padded.gdn) {
        assert_eq!((ca, sa), (cb, sb));
        for j in 0..ca + sa {
            let (a, b) = (exact[oa + j], padded[ob + j]);
            assert!(
                (a - b).abs() < 2e-4,
                "GatedDeltaNet state element {j} differs: exact {a} vs padded {b} \
                 — the padded chunk advanced its state over the padding"
            );
        }
        compared += ca + sa;
    }
    for (&(oa, wa), &(ob, wb)) in l_exact.ple.iter().zip(&l_padded.ple) {
        assert_eq!(wa, wb);
        for j in 0..wa {
            let (a, b) = (exact[oa + j], padded[ob + j]);
            assert!(
                (a - b).abs() < 2e-4,
                "PLE conv window element {j} differs: exact {a} vs padded {b}"
            );
        }
        compared += wa;
    }
    assert!(compared > 0, "nothing was compared");
}
