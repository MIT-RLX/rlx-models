# rlx-nllb

Native [RLX](https://github.com/MIT-RLX/rlx) implementation of Meta
[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) — an
**M2M100** encoder–decoder for multilingual machine translation (FLORES-200
language codes).

At parity with HF `transformers` on the real checkpoint on CPU, Metal and MLX:
encoder and decoder states to ~1e-6 relative, and greedy and beam-4 output
token-identical (`just test-nllb-parity [DEVICE]`).

## Architecture

Matches `facebook/nllb-200-distilled-600M` (`model_type: m2m_100`), read from
its `config.json`:

| Setting | Value |
|---------|-------|
| `d_model` | 1024 |
| encoder / decoder layers | 12 / 12 |
| attention heads | 16 |
| FFN dim | 4096 |
| activation | **relu** |
| `scale_embedding` | **true** (`× √d_model`) |
| vocab | 256206 |
| positions | sinusoidal (not stored), first real row `padding_idx + 1 = 2` |
| specials | bos=0, pad=1, eos=2, `decoder_start_token_id=2` |

Pre-norm layers (`LN → attention → residual`, `LN → FFN → residual`), a final
`model.{encoder,decoder}.layer_norm`, and the tied `model.shared` table as both
the embedding and the LM head.

## Download weights

```bash
just fetch-nllb        # → weights/mt/nllb-200-distilled-600M
```

The repo publishes `pytorch_model.bin`, not safetensors; it is read in place
(`rlx-torch-ckpt`, one tensor at a time). A safetensors file or shards work
too. The three tied copies of the embedding (`model.shared`, both
`embed_tokens`, `lm_head`) are read once.

## Use

```rust
use rlx_nllb::{flores_code, GenerateConfig, NllbRunner};
use rlx_runtime::Device;

assert_eq!(flores_code("en"), Some("eng_Latn"));

let mut runner = NllbRunner::builder()
    .weights("weights/mt/nllb-200-distilled-600M")
    .device(Device::Cpu)
    .build()?;

let opts = GenerateConfig::new(64).with_beams(4);
let fr = runner.translate("Hello, world!", "en", "fr", &opts)?;
println!("{fr}");
```

Lower-level API:

```rust
use rlx_nllb::{GenerateConfig, NllbConfig, NllbModel};

let cfg = NllbConfig::from_hf_config_json("…/config.json".as_ref())?;
let mut model = NllbModel::load("…".as_ref(), cfg, Device::Cpu)?;
let enc = model.encode_tokens(&input_ids)?;            // encoder + cross K/V
let opts = GenerateConfig::new(64).with_forced_bos(tgt_lang_id);
let ids = model.generate_greedy(&enc, &opts)?;          // or generate_beam
// One step at a time: start_decode(&enc, batch, max_len), step(&mut state, &tokens)
```

Generation follows HF: the decoder starts with `decoder_start_token_id` (2),
the first generated token is forced to the target language
(`forced_bos_token_id`), and beam search reproduces `_beam_search` (top
`2·num_beams` candidates, only top-`num_beams` finishers, length penalty over
generated tokens, `early_stopping`, no forced EOS unless
`forced_eos_token_id` is set).

## CLI

```bash
just nllb -- --weights weights/mt/nllb-200-distilled-600M \
  --text "Hello, how are you?" --src eng_Latn --tgt fra_Latn \
  --device metal --beams 4
```

(`just features=metal nllb …` for a GPU build.)

## Memory and speed

On CPU every compiled graph owns its weights unless graphs share a
param-sharing scope, so the runner keeps very few: an encoder graph (which also
emits every layer's cross-attention K/V) and one KV-cached decode-step graph.
The encoder computes every row, so its first bucket is small — 64 tokens,
which covers nearly every sentence — then 128 / 512 / 1024. The decode step
always reads at least 128 source rows (shorter cross K/V are zero-padded, pad
keys masked and skipped), so one step graph serves every bucket up to 128.
A second encoder bucket shares the first's weight pages. Weights are not held
on the host between builds: graphs are built from shapes, compiled, then the
tensors are streamed from the checkpoint into them one at a time. The host
keeps only the tied `[vocab, d]` table (1 GB).

A greedy step keeps its self-attention K/V resident in the step graph (only
the new row moves), takes the cross K/V and source mask as parameters set once
per sentence, and runs the 256k-row LM head across all cores
(`rlx_core::host_kernels::matmul_bt_few_rows`); the first generated token is
forced to the target language, so that step skips the LM head. About 13 ms
per token on an M4 Pro CPU (≈ 7 ms decoder, 6 ms LM head), plus ~25 ms to
encode a short sentence.

`just bench-nllb [DEVICE] [BEAMS]` against `transformers` 5.3 / torch 2.11
(f32, `num_beams=1`, `max_new_tokens=128`), English → Spanish, warm, run back
to back on the same machine under the same load (M4 Pro, CPU):

| | HF transformers | rlx-nllb |
|---|---|---|
| 9 tokens | 0.154–0.174 s | **0.13 s** |
| 15 tokens | 0.251–0.265 s | **0.20–0.21 s** |
| 59 tokens | 1.00–1.04 s | **0.79–0.81 s** |
| all three | 1.40–1.48 s | **1.13–1.15 s** |
| peak memory | 3.4 GB RSS | **2.74 GB** footprint |

Before this work the same run peaked at 14.9 GB and took ~2.3 s for the
59-token sentence. Beam-4 peaks at 3.1 GB. A source longer than 64 tokens
adds a second encoder graph whose weights are the first's pages: system memory
does not grow, but macOS per-process counters (RSS, `phys_footprint`) charge
shared pages once per mapping and report it as ~0.7 GB more.

## Tests

```bash
cargo test -p rlx-nllb                 # synthetic weights, no download
just test-nllb-parity [cpu|metal|mlx]  # real checkpoint vs HF transformers
just test-nllb-backends                # CPU vs GPU backends, synthetic
```

The unit tests check the three decoder spellings against each other on
synthetic weights in the real tensor layout (`rlx_nllb::synth`): the KV-cached
step over a bucket-padded source, the full causal prefix over the unpadded
encoder, and batched / reordered steps (beam search) against single ones.
`RLX_TEST_DEVICE` selects the device. The parity test (`tests/hf_parity_live.rs`,
reference from `scripts/nllb_reference.py`) compares tokenizer ids, the
encoder, both decoder paths, per-step top-8 log-probs and greedy / beam-4 ids.
