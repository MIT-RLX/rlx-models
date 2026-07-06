# rlx-wav2vec2-bert

**[Wav2Vec2-BERT](https://huggingface.co/facebook/w2v-bert-2.0)** Conformer speech encoder for RLX. Given a `Wav2Vec2BertConfig` and safetensors weights, it builds and compiles the encoder graph (feature projection → N Conformer layers with FFN / self-attention / conv modules) and turns mono PCM into hidden states `[batch, seq, hidden]`. A native log-mel extractor is included, so a WAV goes end-to-end with no Python.

## Quick start

```bash
# Encode a 16 kHz mono WAV (falls back to a synthetic 1 s tone if --wav omitted)
cargo run -p rlx-wav2vec2-bert --release -- \
  --weights /path/to/w2v-bert-2.0/model.safetensors \
  --wav audio16k.wav --device cpu --seq 128

# or via the just recipe
just wav2vec2 --weights .../model.safetensors --wav audio16k.wav
```

CLI flags (`src/cli.rs`): `--weights` (required), `--wav`, `--config`, `--device cpu|metal|mlx|cuda|…`, `--batch`, `--seq`, `--dry`.

## Public API

```rust
use rlx_wav2vec2_bert::Wav2Vec2BertRunner;
use rlx_runtime::Device;

let mut runner = Wav2Vec2BertRunner::builder()
    .weights("/path/to/w2v-bert-2.0/model.safetensors")
    .device(Device::Cpu)
    .batch(1)
    .seq(128)          // graph is compiled for a fixed [batch, seq]
    .build()?;

// mono PCM in [-1, 1] @ 16 kHz → encoder hidden states [batch, seq, hidden]
let hidden = runner.encode_waveform(&pcm_16k)?;
// or: runner.encode_wav(path)?, runner.encode_features(&log_mel, mask)?
# anyhow::Ok(())
```

`Wav2Vec2BertConfig::w2v_bert_2_0()` gives the W2v-BERT 2.0 shape; `Wav2Vec2BertConfig::from_file` reads an HF `config.json`. The runner also exposes `config()`, `preprocess_config()`, and `extract_log_mel(waveform)`.

Graph builders are public for embedding the encoder in a larger model: `build_wav2vec2_bert_built`, `build_wav2vec2_bert_graph_sized`, `build_wav2vec2_bert_hir`, plus `Wav2Vec2BertFlow` and the `LogMelExtractor` / `Wav2Vec2BertPreprocessConfig` preprocessing types.

## How it fits

- `Wav2Vec2-BERT` is the speech front-end used by downstream ASR/audio stacks; the encoder graph is described with the shared `rlx-flow` DSL and compiled through [`rlx-runtime`](../rlx-runtime).
- Distinct from [`rlx-wav2vec2-asr`](../rlx-wav2vec2-asr) (classic Wav2Vec2 CTC forced alignment for word timestamps).

## Tests

```bash
cargo test -p rlx-wav2vec2-bert --release   # graph build + forward on synthetic weights, HF config parse
```
