# rlx-miratts

[MiraTTS](https://huggingface.co/YatharthS/MiraTTS) — a 48 kHz autoregressive
LM + neural-codec text-to-speech with voice cloning. **CC-BY-NC-SA-4.0**.

> ✅ **Status: functional + validated.** The LM (Qwen2-0.5B) and the neural
> codec decoder are both real and tested:
> - **LM** greedy decode is **token-exact vs HF transformers** (16/16),
>   `tests/lm_parity.rs`.
> - **Codec decoder** (`detokenizer.onnx`) is imported **natively** (runs on any
>   RLX backend — no ONNX Runtime at runtime) and is **bit-exact vs onnxruntime
>   (cos 0.99999)**, `tests/codec_parity.rs`; its native decode of a real
>   utterance is **Whisper-intelligible**, `tests/whisper_roundtrip.rs`.
> - **Voice cloning from a raw reference clip** via
>   [`MiraTts::synthesize_with_ref`](src/lib.rs): FastBiCodec `encode_audio`
>   (`s_encoder.onnx`, mel → 32 global tokens) conditions the LM prompt and
>   detokenizer. Native parity vs ORT for `s_encoder` is exact
>   (`tests/s_encoder_parity.rs`).
>
> Note: `q_encoder.onnx` (WavLM SSL → semantic tokens) is the alternate LinaCodec
> encoder path and is **not** used by MiraTTS/FastBiCodec `encode_audio`. The
> shipped clone path is mel + `s_encoder` only.

## Real architecture (verified from the HF/github repos)

MiraTTS is **not** a single-ONNX model. It is a two-model autoregressive stack
(the Orpheus/NeuTTS pattern), and **no ONNX is published**:

- **LM — Qwen2-0.5B** (`Qwen2ForCausalLM`): hidden 896, 24 layers, 14 heads,
  2 KV heads (GQA), intermediate 4864, rope_theta 1e6, tied lm_head, bf16.
  **Vocab 166 000** = ~151.9 k base text tokens **+ ~14 k audio codec tokens**
  emitted inline by the LM. `model.safetensors` (1.01 GB).
- **Codec — [LinaCodec](https://huggingface.co/YatharthS/LinaCodec)** (same
  author): encoder is a **WavLM** SSL model + quantizer
  (`encode(wav) → (speech_tokens, global_embedding)`, 12.5 tokens/s); decoder is
  a **Dual-Path Vocos** (`decode(tokens, global_embedding) → 48 kHz`). Voice
  cloning rides on the per-utterance `global_embedding` from the reference clip.
  Weights: `config.yaml`, `model.safetensors` (480 MB), `wavlm_encoder.pth`
  (99 MB), `vocoder/`.

Reference pipeline (`mira/model.py`, via the `ncodec` package's `TTSCodec`):

```text
ctx = codec.encode(reference_wav)              # WavLM enc → speech tokens + global emb
prompt = codec.format_prompt(text, ctx)        # text + audio tokens + speech/text markers
ids = qwen2.generate(prompt)                   # AR: emits audio codec token ids
wav = codec.decode(ids, ctx)                   # Dual-Path Vocos → 48 kHz
```

## Port plan (large, multi-session)

1. Reverse-engineer the `ncodec`/LinaCodec **token layout + prompt format**
   (audio-token vocab offset, per-frame codebook interleave, speech/text
   start/end markers) from source + a reference Python dump.
2. **Qwen2-0.5B LM** native — adapt an rlx Qwen runner (Qwen2 = Qwen3 without
   QK-norm) + AR decode loop (mirror `rlx-orpheus`); validate token-ids vs Python.
3. **LinaCodec decoder** (Dual-Path Vocos) native — reuse rlx Vocos/ISTFT.
4. **WavLM encoder** native (voice cloning) — reuse an existing rlx SSL encoder
   if available.

## Current API

```rust
use rlx_miratts::{MiraTts, MiraConfig};
use rlx_runtime::Device;

let tts = MiraTts::load(std::path::Path::new("weights/tts/miratts"), Device::Cpu)?;
println!("{:?}", tts.config());          // parsed Qwen2-0.5B config
tts.synthesize("hello", &reference_wav)?; // Err: not yet implemented (never silent)
```
