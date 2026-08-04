//! PQ dense-init closes the quality gap *before any fine-tuning*: a `rlx-tiny`
//! synth model whose codebooks + u8 indices are product-quantized from a trained
//! dense `rlx-tinystories` model starts at a far lower loss than a random-init
//! one — the codebooks now *encode real weights*. This is the honest, IO-free
//! measurement the whole exercise is about. Pure CPU, offline.

use rlx_tensor::{DType, Device, LrSchedule};

/// A small, trivially-memorizable corpus (same shape as the smoke test's).
fn tiny_corpus() -> Vec<u8> {
    let base = "Once upon a time there was a little cat. The cat liked to play. ";
    let mut s = String::new();
    while s.len() < 8000 {
        s.push_str(base);
    }
    s.into_bytes()
}

/// Snapshot a model's params as `(name, data)`.
fn params_of_dense(m: &rlx_tensor::Func) -> Vec<(String, Vec<f32>)> {
    m.param_names()
        .into_iter()
        .map(|n| {
            let d = m.param_binding(&n).unwrap().to_vec();
            (n, d)
        })
        .collect()
}

#[test]
fn pq_dense_init_beats_random_init_untrained() {
    let corpus = tiny_corpus();

    // ── 1) Train a DENSE teacher (rlx-tinystories smoke) so its weights are
    //       actually informative (the corpus is trivially memorizable). ────────
    let dcfg = rlx_tinystories::config::GptConfig::smoke();
    let ddata = rlx_tinystories::data::Tokens::Bytes(&corpus);
    let mut dm = rlx_tinystories::model::init(
        rlx_tinystories::model::build(&dcfg, dcfg.batch, true, DType::F32),
        &dcfg,
        42,
    );
    let dbatcher = rlx_tinystories::data::Batcher::new(&dcfg);
    let mut drng = rlx_tinystories::Rng::new(1);
    let steps = 120usize;
    let mut dopt = rlx_tinystories::HybridOptimizer::new(3e-3, 2e-2, 0.05);
    let dsched = LrSchedule::WarmupCosine {
        base: 3e-3,
        min: 3e-4,
        warmup: 10,
        total: steps,
    };
    let mut dense_loss = f32::NAN;
    for step in 0..steps {
        let (tok, tgt) = dbatcher.sample(&ddata, &mut drng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let (next, loss) = dm.train_step_all_at_on(Device::Cpu, &mut dopt, &dsched, step, feed);
        dm = next;
        dense_loss = loss[0];
    }
    assert!(
        dense_loss.is_finite() && dense_loss < 3.0,
        "dense teacher did not learn: {dense_loss}"
    );
    let dense_params = params_of_dense(&dm);

    // ── 2) tiny config must match the dense architecture (so param shapes line
    //       up 1:1). Both `smoke()` share vocab/ctx/layers/heads/embd. ──────────
    let tcfg = rlx_tiny::config::GptConfig::smoke();
    assert_eq!(
        (
            tcfg.vocab,
            tcfg.block_size,
            tcfg.n_layer,
            tcfg.n_head,
            tcfg.n_embd
        ),
        (
            dcfg.vocab,
            dcfg.block_size,
            dcfg.n_layer,
            dcfg.n_head,
            dcfg.n_embd
        ),
        "smoke configs must share the architecture"
    );

    // ── 3) PQ-init tiny (codebooks+indices+LoRA from the dense weights) vs a
    //       random-init tiny of the exact same shape. ────────────────────────
    let synth = rlx_tiny::model::SynthInit::from_dense(&tcfg, &dense_params);
    let pq = rlx_tiny::model::init_dense(
        rlx_tiny::model::build_dense_init(&tcfg, tcfg.batch, true, &synth),
        &tcfg,
        &synth,
        7,
    );
    let rnd = rlx_tiny::model::init(rlx_tiny::model::build(&tcfg, tcfg.batch, true), &tcfg, 7);

    // ── 4) Compare UNTRAINED loss on the same batch. ─────────────────────────
    let tdata = rlx_tiny::data::Tokens::Bytes(&corpus);
    let tbatcher = rlx_tiny::data::Batcher::new(&tcfg);
    let mut trng = rlx_tiny::Rng::new(99);
    let (tok, tgt) = tbatcher.sample(&tdata, &mut trng);
    let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
    let pq_loss = pq.run_on(Device::Cpu, feed)[0][0];
    let rnd_loss = rnd.run_on(Device::Cpu, feed)[0][0];

    eprintln!(
        "PQ dense-init: dense_teacher={dense_loss:.4}  pq_init={pq_loss:.4}  random_init={rnd_loss:.4}  \
         (gap closed {:.1}%)",
        100.0 * (rnd_loss - pq_loss) / (rnd_loss - dense_loss).max(1e-6)
    );

    assert!(
        pq_loss.is_finite() && rnd_loss.is_finite(),
        "losses must be finite"
    );
    // Random byte-level CE starts near ln(256) ≈ 5.55.
    assert!(
        rnd_loss > 4.5,
        "random-init loss should be near ln(256): {rnd_loss}"
    );
    // The whole point: encoding the trained dense weights starts far lower.
    assert!(
        pq_loss < rnd_loss * 0.85,
        "PQ dense-init should be much lower than random init: pq={pq_loss} vs random={rnd_loss}"
    );
}
