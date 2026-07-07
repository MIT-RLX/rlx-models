# rlx-f5tts

F5-TTS voice-cloning text-to-speech for RLX — a **flow-matching DiT** (330M,
24 kHz) that clones a voice from a short reference clip + transcript.

> **License:** the F5-TTS weights are **CC-BY-NC-4.0 (non-commercial)**; the code
> is MIT. Use accordingly.

Runs the community [DakeQQ](https://github.com/DakeQQ/F5-TTS-ONNX) 3-file ONNX
export (`F5_Preprocess`, `F5_Transformer`, `F5_Decode`, all **f16**) with a thin
Rust orchestrator — everything numeric lives in the ONNX:

1. `preprocess(audio, text_ids, max_duration)` → noise + RoPE + CFG conditioning
2. **NFE denoising loop** (default 32): the DiT does classifier-free guidance +
   the ODE step internally; just feed the latent back with the step index
3. `decode(latent, ref_len)` → 24 kHz audio (Vocos vocoder + ISTFT folded in)

The Rust side is only text tokenization (char-level over `vocab.txt`), F5's
duration estimate, and the loop.

## Setup

Download the ONNX export ([`huggingfacess/F5-TTS-ONNX`](https://huggingface.co/huggingfacess/F5-TTS-ONNX))
and the vocab ([`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS) →
`F5TTS_v1_Base/vocab.txt`) into `weights/tts/f5tts/`:
`F5_Preprocess.onnx`, `F5_Transformer.onnx`, `F5_Decode.onnx`, `vocab.txt`.

## Usage

```bash
cargo run -p rlx-f5tts --release --bin rlx-f5tts -- \
    --ref-wav reference.wav \
    --ref-text "transcript of the reference audio" \
    --text "Text to speak in the cloned voice." \
    --out out.wav
```

`--nfe` (default 32; lower = faster, rougher), `--speed` (1.0), `--device`.

## Performance

NFE-32 over the 664 MB DiT is compute-heavy on CPU (~40–50 s for a short
utterance). Use `--nfe 16` for a faster, slightly rougher preview, or a GPU
execution provider (`metal`/`cuda`).

## Backends

Runs the three ONNX subgraphs on ONNX Runtime (CPU, plus CoreML / CUDA /
DirectML via `metal`/`mlx`/`cuda`/`gpu`).

## Note

English is char-level over `vocab.txt` (no phonemizer). Chinese needs pinyin
(`jieba`/`pypinyin`) conversion, which is not yet ported.
