# rlx-nemotron-asr

NVIDIA **Nemotron 3.5 ASR Streaming 0.6B** on RLX — a cache-aware
**FastConformer** encoder with a prompt-conditioned **RNN-T** decoder,
loaded **natively from the distributed `.nemo`** (no manual conversion).

Model card: <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>

## How it works

```
.nemo ──rlx-nemo──▶ config.yaml + state-dict (torch.save) + SentencePiece
  │
  ├─ log-mel frontend (mel.rs)            NeMo AudioToMelSpectrogramPreprocessor
  ├─ FastConformer encoder (encoder.rs)   dw_striding subsample → N conformer blocks
  │                                        (½FFN → rel-pos MHSA → conv → ½FFN → LN)
  │                                        — runs as an RLX graph (CPU/Metal/MLX/…)
  ├─ RNN-T decode (decoder.rs)            prediction LSTM + joint + greedy, host-side
  └─ SentencePiece detok (tokenizer.rs)  → transcript
```

Native `.nemo` reading lives in the sibling crate
[`rlx-nemo`](../../../rlx/crates/rlx-nemo) (tar + `torch.save` zip + a minimal
torch pickle VM + YAML). Everything in this crate is derived from the `.nemo`'s
`model_config.yaml`; no hyperparameters are hard-coded.

## CLI

```bash
# Transcribe a wav (any sample rate; resampled to the model's rate):
cargo run -p rlx-nemotron-asr --bin rlx_nemotron_asr -- \
    transcribe --nemo nemotron-3.5-asr-streaming-0.6b.nemo --wav clip.wav --device cpu

# Inspect a checkpoint's config + every state-dict tensor name/shape
# (use this to reconcile `src/weights.rs::keys` for a specific checkpoint):
cargo run -p rlx-nemotron-asr --bin rlx_nemotron_asr -- \
    dump-keys --nemo nemotron-3.5-asr-streaming-0.6b.nemo
```

## Status

**Verified on the real checkpoint** (`nvidia/nemotron-3.5-asr-streaming-0.6b`,
2.37 GB gzipped `.nemo`): transcribes correctly on CPU. JFK clip →
`And so my fellow American ask not what your country can do for you ask what you
can do for your country`.

Real-model specifics (all data-driven from the YAML, confirmed via `dump-keys`):
n_mels=128 / n_fft=512; the model's **own mel filterbank + window are used**
(`preprocessor.featurizer.{fb,window}`) for exact frontend parity; **no biases**
except LayerNorms; the conv module uses **LayerNorm** (`conv_norm_type: layer_norm`),
not BatchNorm; **causal** downsampling/conv with **ceil-mode** frequency reduction
(128→65→33→17); a **2-layer** prediction LSTM; and a **`prompt_kernel`** MLP that
fuses the `target_lang` one-hot into the encoder features before the joint.

**Language is required**: pass `--lang <code>` (default `en-US`); indices come from
the checkpoint's `prompt_dictionary`. An all-zeros prompt makes the transducer emit
only blanks.

Unit + smoke tests cover the loader, mel, RNN-T numerics, WAV, tokenizer, and a
synthetic full-encoder compile+run (`tests/encoder_smoke.rs`). For element-wise
NeMo parity use [`scripts/nemotron_asr_reference.py`](../../scripts/nemotron_asr_reference.py).
The encoder currently runs full-context per utterance; cache-aware streaming masks
(`att_context_size` `[56, k]`) are a follow-up at the runner level.
