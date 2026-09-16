# rlx-nllb

Native [RLX](https://github.com/MIT-RLX/rlx) implementation of Meta
[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) — an
**M2M100** encoder–decoder for multilingual machine translation (FLORES-200
language codes).

Runs on every RLX backend (CPU / Metal / MLX / CUDA / ROCm / GPU / Vulkan) from
one IR graph — no per-backend code.

## Architecture

Matches `facebook/nllb-200-distilled-600M` (`model_type: m2m_100`):

| Knob | Value |
|------|-------|
| `d_model` | 1024 |
| encoder / decoder layers | 12 / 12 |
| attention heads | 16 |
| FFN dim | 4096 |
| activation | **relu** |
| `scale_embedding` | **true** (`× √d_model`) |
| vocab | 256206 |
| max positions | 1024 (learned, **offset 2**) |
| specials | bos=0, pad=1, eos=2, `decoder_start_token_id=2` |

BART-style post-norm stack with `layernorm_embedding`, tied `model.shared`
embedding as the LM head, and optional `model.{encoder,decoder}.layer_norm`.

Weight keys use the `model.` prefix (not Florence’s `language_model.model.`).

## Download weights

```bash
huggingface-cli download facebook/nllb-200-distilled-600M \
  --local-dir .cache/nllb-200-distilled-600M
```

Expect `model.safetensors` (or shards), `config.json`, and `tokenizer.json` in
that directory.

## Use

```rust
use rlx_nllb::{flores_code, GenerateConfig, NllbRunner};
use rlx_runtime::Device;

assert_eq!(flores_code("en"), Some("eng_Latn"));

let mut runner = NllbRunner::builder()
    .weights(".cache/nllb-200-distilled-600M")
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
let enc = model.encode_tokens(&input_ids)?;
let ids = model.generate_greedy(&enc, input_ids.len(), &GenerateConfig::new(64).with_forced_bos(tgt_lang_id))?;
```

## CLI

```bash
cargo run -p rlx-nllb --release -- \
  --weights .cache/nllb-200-distilled-600M \
  --text "Hello, how are you?" \
  --src eng_Latn --tgt fra_Latn \
  --device metal --beams 4
```

NLLB generation convention: the decoder starts with `eos` /
`decoder_start_token_id` (2), then the **first generated token is forced** to
the target FLORES language id (`forced_bos_token_id`).

## Tests

```bash
cargo test -p rlx-nllb
```

Unit tests cover FLORES aliases, the distilled preset, graph builders over a
synthetic tiny `WeightMap`, and a CPU encode → `decode_logits` smoke path.
