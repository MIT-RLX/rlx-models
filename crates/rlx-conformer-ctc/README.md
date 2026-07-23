# rlx-conformer-ctc

NVIDIA **Conformer-CTC** ASR on RLX — classic Conformer encoder + CTC head,
loaded **natively from the distributed `.nemo`** (no ONNX / no Python at
runtime).

Primary checkpoint:
[`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small)
(~13 M params, English, SentencePiece unigram 1024 + blank).

## Quick start

```bash
just fetch-conformer-ctc

just conformer-ctc -- \
  transcribe \
  --nemo .cache/conformer-ctc/stt_en_conformer_ctc_small.nemo \
  --wav clip.wav \
  --device metal \
  --warm
```

`--warm` runs twice and prints cold (compile + run) vs warm (cached) ms.

## How it works

```
.nemo ──rlx-nemo──▶ config.yaml + state-dict + SentencePiece
  │
  ├─ log-mel frontend (mel.rs)      NeMo AudioToMelSpectrogramPreprocessor
  │                                  (uses checkpoint featurizer.fb / window)
  ├─ Conformer encoder (encoder.rs) striding×4 subsample → N conformer blocks
  │                                  (½FFN → rel-pos MHSA → conv+BN → ½FFN → LN)
  │                                  — RLX graph; CompileCache by mel bucket
  ├─ CTC head + greedy (ctc.rs)     ConvASRDecoder linear → collapse / drop blank
  └─ SentencePiece detok            → transcript
```

## CLI

```bash
# Transcribe (any sample rate; resampled to the model rate, usually 16 kHz):
just conformer-ctc -- \
  transcribe \
  --nemo .cache/conformer-ctc/stt_en_conformer_ctc_small.nemo \
  --wav clip.wav \
  --device cpu|metal|mlx|cuda|gpu|vulkan

# Inspect config + every state-dict tensor:
just conformer-ctc -- \
  dump-keys --nemo .cache/conformer-ctc/stt_en_conformer_ctc_small.nemo
```

## Library

```rust
use rlx_conformer_ctc::{ConformerCtc, wav};
use rlx_runtime::Device;

let mut asr = ConformerCtc::open(
    "stt_en_conformer_ctc_small.nemo".as_ref(),
    Device::Cpu,
)?;
let w = wav::parse(&std::fs::read("clip.wav")?)?;
let pcm = wav::resample(&w.samples, w.sample_rate, asr.config().sample_rate as u32);

// Optional: precompile the mel-length bucket before the first utterance.
asr.warm(/* mel_frames hint */ 768)?;

let text = asr.transcribe(&pcm)?;
let ids = asr.transcribe_ids(&pcm)?; // CTC piece ids (blank already removed)
```

`transcribe` / `transcribe_ids` take `&mut self` because the encoder
compile cache is updated on first use of each mel-length bucket.

## Features

| Feature | Purpose |
|---------|---------|
| `metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan` / `coreml` | Forwarded to `rlx-runtime` |
| `all-backends` | All of the above |
| `apple-silicon` | `metal` + `mlx` + `gpu` + `coreml` |
| `nvidia-gpu` | `cuda` |

## Backends

| Device | Notes |
|--------|--------|
| `cpu` | Always available |
| `metal` / `mlx` | Apple Silicon (`--features apple-silicon`) |
| `gpu` (wgpu) | Portable GPU; compile cache reuse supported |
| `cuda` | NVIDIA hosts (`--features nvidia-gpu`) |
| `vulkan` / `rocm` | When the host adapter is available |

Cross-backend matrix (cold_ms / warm_ms + transcript check):

```bash
just test-conformer-ctc-backends
just conformer-ctc-cuda-msi          # sync + CUDA on ssh msi
```

On the Librispeech sample (`stt_en_conformer_ctc_small`), warm Metal is
typically ~17 ms after a ~1.5 s cold compile; WGPU / CUDA / CPU also reuse the
cached encoder.

## Status

**Verified** on
[`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small)
(CPU, Metal, MLX, WGPU; CUDA on NVIDIA hosts). Sample transcript:

`well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait`

Targets NeMo `EncDecCTCModelBPE` with `encoder.subsampling: striding` (as in
`stt_en_conformer_ctc_small`: d_model=176, 16 layers, 4 heads, conv kernel 31,
80-mel, CTC vocab 1024). FastConformer `dw_striding` + RNN-T checkpoints belong
in [`rlx-nemotron-asr`](../rlx-nemotron-asr).

Unit tests cover config, mel, CTC greedy, WAV, and a synthetic encoder
compile+run (`tests/encoder_smoke.rs`). Real-checkpoint transcription is
exercised via the CLI and `examples/backend_matrix.rs`.

## See also

- Main repo [README](../../README.md)
- [AGENTS.md](../../AGENTS.md) — `just fetch-conformer-ctc`, `just conformer-ctc`
- [`rlx-nemo`](https://crates.io/crates/rlx-nemo) — native `.nemo` reader
- [`rlx-nemotron-asr`](../rlx-nemotron-asr) — FastConformer + RNN-T
