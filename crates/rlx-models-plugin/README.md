# rlx-models-plugin

rlx-models as an rlx **model plugin**: a shared library that rlx loads at run
time. After that, every rlx language binding opens these models with
`Model.open(path)` and calls each task's method: Python, JS, Swift, C/C++, Go,
Java and the rest. rlx itself never depends on rlx-models.

| Provider | Opens | Method |
|---|---|---|
| `rlx-models` | any checkpoint `auto_runner` recognises (GGUF or safetensors, every registered LM family) | `chat`, `generate_text`, `generate` |
| `rlx-models-whisper` | Whisper checkpoints (`config.json` `model_type: whisper`) | `transcribe(audio, sample_rate)` |
| `rlx-models-embed` | BERT / XLM-R sentence encoders (`model_type: bert` / `xlm-roberta`) | `embed(texts)` |

```bash
cargo build -p rlx-models-plugin --release            # add --features metal / cuda / all-models
mkdir -p ~/.rlx/plugins && cp target/release/librlx_models_plugin.* ~/.rlx/plugins/
```

```python
import pyrlx
lm = pyrlx.Model.open("Qwen3-0.6B-Q4_K_M.gguf")
lm.chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=24)   # 'Paris.'
pyrlx.Model.open("whisper-tiny", language="en").transcribe(pcm, 16000)
pyrlx.Model.open("multilingual-e5-base").embed(["query: …", "passage: …"])           # [2, 768]
```

The plugin is found in `~/.rlx/plugins` without any setting.
`RLX_MODEL_PLUGINS=<library or directory>` also works.

On iOS, Android or wasm, where nothing loads at run time, link this crate
into the app's library instead (`default-features = false` plus the families
and backend you ship):

```rust
pub use rlx_capi::*;
rlx_runtime::link_model_plugins!(rlx_models_plugin);   // rlx-runtime feature `linked-plugins`
```

The app calls `rlx_link_model_plugins()` once (Swift: `linkModelPlugins()`).
For an iOS app, rlx's `ios/build-xcframework.sh --capi-crate <Cargo.toml>`
packages the library as `RLX.xcframework`. See "Link one" in rlx's
`crates/bindings/rlx-model-plugin/README.md`.

## Options

These are keys of `Model.open`'s options, passed as a dict or as keywords in
Python.

| Option | Provider | Meaning |
|---|---|---|
| `device` | all | `cpu`, `metal`, `cuda`, …, as long as the plugin was built with that backend's feature |
| `max_seq` | `rlx-models` | the context length to allocate |
| `mmproj` | `rlx-models` | a multimodal projector, for vision-language checkpoints |
| `tokenizer` | `rlx-models` | a `tokenizer.json`. By default the provider uses one next to the checkpoint. With no tokenizer, `generate` (ids) works, but `chat` and `generate_text` do not |
| `language` | whisper | the spoken language (`"en"`). Pass it: whisper-tiny's language detection misreads clear English |
| `pooling`, `max_length` | embed | `"mean"` (default) or `"cls"`; tokens per text (256) |

`chat` uses the model's chat template: from the GGUF metadata, or from the
`chat_template` in a `tokenizer_config.json` next to the tokenizer.
`transcribe` resamples audio at other rates to Whisper's 16 kHz.

For a checkpoint the plugin can sniff, the `rlx-models` provider scores 80 in
`probe`. The task providers score 70 on their `model_type`. Both beat the
built-in `rlxp` provider. `options.provider` picks a provider explicitly.

## Adding a family

A language model needs nothing here. Register its `LmRunner` constructor with
`rlx_core::lm_registry::register_lm_runner`, as every family in
`rlx_models::run` does. `auto_runner` then builds it, and this plugin opens
it.

Any other task needs one trait plus one `Provider` in `src/tasks.rs`, listed
in the `export_model_plugin!` call in `src/lib.rs`. Whisper and the embedder
there are about 30 lines each:

```rust
impl SpeechToText for Whisper { fn transcribe(&mut self, audio, sample_rate, options) -> Result<String> { … } }
pub static WHISPER: Provider = Provider::new("rlx-models-whisper", "speech-recognition", |req| { … })
    .claims(&[Claim::HfModelType("whisper")]);
```

Checking against real weights is opt-in: `RLX_TEST_LM=<gguf>`,
`RLX_TEST_ASR=<whisper dir>`, `RLX_TEST_EMBED=<encoder dir>`, then
`cargo test -p rlx-models-plugin`.

The plugin ABI and the general authoring guide are in rlx's
`crates/bindings/rlx-model-plugin/README.md`.

License: GPL-3.0.
