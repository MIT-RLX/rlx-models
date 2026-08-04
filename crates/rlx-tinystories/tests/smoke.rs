//! CPU smoke test: the from-scratch GPT actually learns (loss falls sharply on
//! a tiny repetitive corpus) and the generation path runs. Fast and offline —
//! no TinyStories download, no GPU.

use rlx_tensor::{DType, Device, LrSchedule};

use rlx_tinystories::config::GptConfig;
use rlx_tinystories::data::{Batcher, Tokens};
use rlx_tinystories::sample::{GenOptions, generate};
use rlx_tinystories::{HybridOptimizer, Rng, model};

/// A small, highly-learnable corpus.
fn tiny_corpus() -> Vec<u8> {
    let base = "Once upon a time there was a little cat. The cat liked to play. ";
    let mut s = String::new();
    while s.len() < 8000 {
        s.push_str(base);
    }
    s.into_bytes()
}

#[test]
fn gpt_learns_and_generates_on_cpu() {
    let cfg = GptConfig::smoke();
    assert!(cfg.check().is_ok());
    let corpus = tiny_corpus();
    let data = Tokens::Bytes(&corpus);

    let mut m = model::init(model::build(&cfg, cfg.batch, true, DType::F32), &cfg, 42);
    let batcher = Batcher::new(&cfg);
    let mut rng = Rng::new(1);

    // Untrained loss on a batch (byte-level CE starts near ln(256) ≈ 5.55).
    let (tok0, tgt0) = batcher.sample(&data, &mut rng);
    let f0: &[(&str, &[f32])] = &[("tok_ids", &tok0), ("tgt_ids", &tgt0)];
    let loss0 = m.run_on(Device::Cpu, f0)[0][0];
    assert!(
        loss0.is_finite() && loss0 > 1.0,
        "suspicious init loss {loss0}"
    );

    let steps = 150usize;
    // Muon (2-D matrices) + AdamW (embeddings, biases, norms).
    let mut opt = HybridOptimizer::new(3e-3, 2e-2, 0.05);
    let sched = LrSchedule::WarmupCosine {
        base: 3e-3,
        min: 3e-4,
        warmup: 10,
        total: steps,
    };
    let mut last = loss0;
    for step in 0..steps {
        let (tok, tgt) = batcher.sample(&data, &mut rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let (next, loss) = m.train_step_all_at_on(Device::Cpu, &mut opt, &sched, step, feed);
        m = next;
        last = loss[0];
    }

    assert!(last.is_finite(), "loss went non-finite");
    assert!(
        last < loss0 * 0.7,
        "model did not learn: init {loss0:.3} → final {last:.3}"
    );

    // Generation path runs and produces the requested number of new tokens.
    let params: Vec<(String, Vec<f32>)> = m
        .param_names()
        .into_iter()
        .map(|n| (n.clone(), m.param_binding(&n).unwrap().to_vec()))
        .collect();
    let prompt = "Once upon";
    let out = generate(
        &cfg,
        &params,
        prompt,
        Device::Cpu,
        &GenOptions {
            max_new_tokens: 32,
            temperature: 0.8,
            top_k: 20,
            seed: 3,
        },
        None,
    );
    assert!(out.starts_with(prompt), "output lost the prompt: {out:?}");
    assert!(
        out.len() >= prompt.len() + 32,
        "expected 32 new bytes, got {:?}",
        out
    );
}
