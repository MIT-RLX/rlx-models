# rlx-eval

Evaluation harness for RLX language models: **perplexity** over a corpus and **lm-eval-style multiple-choice** scoring, both built on teacher-forced next-token log-probs. It's generic over the [`LmLogprobs`] trait, so any model exposing one full-sequence forward can be scored, and the whole harness is host-side (it consumes the `Vec<f32>` log-probs the model produces on any backend).

## Public API

```rust
use rlx_eval::{LmLogprobs, PerplexityConfig, perplexity, McItem, score_mc};

// Implement LmLogprobs for your model, then:
let ppl = perplexity(&mut model, &token_ids, &PerplexityConfig::default())?;

let items = vec![McItem { context: "...".into(), choices: vec!["A".into(), "B".into()], answer: 0 }];
let mc = score_mc(&mut model, &items)?;   // accuracy over teacher-forced choice log-probs
# anyhow::Ok(())
```

## How it fits

- [rlx-qwen3](../rlx-qwen3) and other LM crates supply the `LmLogprobs` implementation.
- Host-side only — backend-agnostic (works with whatever device the model ran on).
