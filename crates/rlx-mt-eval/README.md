# rlx-mt-eval

Host-side machine-translation metrics for RLX dubbing bake-offs.

Pure text — no model weights, no Metal. Safe to call from Mac CI / translator
prototyping; **not** part of the iOS ship path.

## Metrics

| Metric | Use |
|--------|-----|
| **chrF** | Primary surface score (script-agnostic; same formula as `rlx-translate::score`) |
| **BLEU-4** | Corpus / sentence BLEU with brevity penalty |
| **TER↓** | Word edit rate vs reference (lower better) |
| **entity F1** | Glossary / named-fact recall (Helix, 950, radial, …) |
| **timing** | Cue fill / overrun from `placed_duration_sec` |

## CLI

```bash
# Parallel line files
cargo run -p rlx-mt-eval -- score --hyp hyps.txt --ref refs.txt --lang fr

# translator-core result.json vs built-in motor gold
cargo run -p rlx-mt-eval -- score-result \
  --result /path/to/fr/result.json --lang fr --gold motor

# Multi-lang bench dir (fr/, de/, uk/ result.json)
cargo run -p rlx-mt-eval -- score-bench \
  --dir /path/to/bench-improve3 --langs fr,de,uk --gold motor

# Every run under a bake-off folder
cargo run -p rlx-mt-eval -- score-bakeoff \
  --dir /path/to/output/bakeoff --lang fr --gold motor
```

From the translator repo:

```bash
python3 scripts/eval_mt.py output/bench-improve3
python3 scripts/eval_mt.py output/bakeoff --bakeoff --json-out output/bakeoff/mt_scores.json
```

## Library

```rust
use rlx_mt_eval::{score_pair, motor_clip_entities};

let s = score_pair(hyp, reference, &motor_clip_entities("fr"));
```
