# rlx-pocket-tts

Native Rust port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts):
a lightweight (~100 M parameters) text-to-speech model that runs on CPU and
streams 24 kHz audio at faster than real time.

The implementation is an eager ndarray-based forward pass. It composes:

- **FlowLM** — 6-layer transformer (`d=1024`, 16 heads, RoPE, full causal,
  `GELU` FFN) followed by a per-step **SimpleMLPAdaLN flow head** (6 ResBlocks at
  `d=512`, 32-dim continuous latent output). Euler-integrated sampling.
- **Mimi codec (decoder path)** — `quantizer.output_proj` (Conv1d 32→512, k=1) →
  depthwise upsample (ConvTranspose1d 512→512, k=32, stride=16, groups=512) →
  2-layer projected transformer (`d=512`, 8 heads, sliding-window context=250,
  layer-scale 0.01) → SEANet decoder (ratios `[6, 5, 4]`) → 24 kHz PCM.

## Weights

Weights ship as a single safetensors file plus a SentencePiece tokenizer.
The ungated mirror is at
[`Verylicious/pocket-tts-ungated`](https://huggingface.co/Verylicious/pocket-tts-ungated).

```text
tts_b6369a24.safetensors    # 213 tensors, ~118 M params, BF16
tokenizer.model             # SentencePiece, vocab=4000
embeddings/<voice>.safetensors   # single tensor `audio_prompt` [1, 125, 1024]
```

`audio_prompt` is the post-projection conditioning sequence (125 frames at 12.5 Hz
= 10 s of voice) ready to feed straight into the FlowLM backbone. The official
`kyutai/tts-voices` repo stores fully-warmed KV caches in a different format;
this crate targets the ungated mirror's `audio_prompt` format.

## Usage

```rust
use rlx_pocket_tts::TtsModel;

let model = TtsModel::load("/path/to/pocket-tts")?;
let voice = model.load_voice("alba")?;
let audio = model.generate("Hello, world.", &voice)?;
audio.write_wav("output.wav")?;
```

Enable the `hf-download` feature for automatic download from Hugging Face.

## Validation

Whisper-validated end-to-end. Examples (`alba` voice, real prompts, transcribed
with `openai/whisper-large-v3` via `rlx-whisper`):

| Prompt | Whisper transcription |
|---|---|
| "The cat sat on the mat." | "The cat sat on the mat." |
| "My name is Alba and I work at MIT." | "My name is Alba and I work MIT." |
| "Hello world. I am Kyutai's Pocket TTS, running natively in Rust. I hope you like the way I sound." | "Hello world. I am Cutie's Pocket TTS, running natively in Rust. I hope you like the way I sound." |

Word coverage 89-100% per prompt — the only misses are proper-noun mishears
("Kyutai" → "Cutie") that also happen on the upstream Python reference. Run
`examples/whisper_check.rs` to validate after weight or model changes.

## Backends

- macOS / iOS with `accelerate` feature (default): CBLAS `sgemm` via Apple's
  Accelerate framework. Linker directive emitted by `build.rs`.
- Any other target / `--no-default-features`: pure-Rust matmul via
  `ndarray::dot`. Verified parity with Accelerate to max sample diff 3e-5
  (well below audibility), zero samples differ by > 1e-3 (~-60 dB).
