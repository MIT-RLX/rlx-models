# rlx-moonshine-streaming

Moonshine v2 streaming ASR ([arXiv:2602.12241](https://arxiv.org/abs/2602.12241))
on RLX — `moonshine-ai/moonshine-streaming-{tiny,small,medium}`.

| Model | Params | Checkpoint (f32) | Open ASR WER |
|---|--:|--:|--:|
| [tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny) | 34 M | 176 MB | 12.00 % |
| [small](https://huggingface.co/moonshine-ai/moonshine-streaming-small) | 123 M | 561 MB | 7.84 % |
| [medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) | 245 M | 1.06 GB | 6.65 % |

```sh
hf download moonshine-ai/moonshine-streaming-tiny --local-dir weights/asr/moonshine-streaming-tiny
cargo run -p rlx-moonshine-streaming --release -- \
    --weights weights/asr/moonshine-streaming-tiny --wav assets/harness/jfk_16k.wav
```

## Architecture

```text
16 kHz PCM ─► 80-sample frames ─► CMVN ─► asinh(e^log_k · x)          (host)
  ─► linear 80→d ─► SiLU ─► causal conv k5/s2 ─► SiLU ─► causal conv k5/s2   (50 Hz)
  ─► N × [LN ─► windowed self-attn ─► LN ─► GELU MLP] ─► LN
  ─► + learned positions ─► proj to decoder width ─► Moonshine decoder
```

- The encoder has no positional encoding; each layer bounds attention by a
  `(left, right)` window — `(16, 4)` sees 15 frames back and 3 ahead,
  `(16, 0)` is causal. Position enters afterwards, on the decoder side.
- Encoder LayerNorms scale by `γ + 1` and have no bias.
- `moonshine-streaming-small` has a 620-wide encoder with 8 × 64 = 512-wide
  attention, so the explicit `head_dim` is used, not `hidden / heads`.
- The decoder is Moonshine v1's, built by `rlx_moonshine::flow::build_decoder_hidden_built`;
  the LM head is the untied `proj_out`.

Decoding re-runs the decoder over the whole prefix per token (graphs bucketed
by 16 tokens). That is fine for utterance-length audio; the encoder itself is
already streaming-shaped, so incremental encoding is the natural next step.

## Parity

`tests/hf_parity_live.rs` against transformers
(`scripts/moonshine_streaming_reference.py`) on `assets/harness/jfk_16k.wav`:
encoder states within 1.6e-6 – 3.5e-6 relative and greedy tokens identical for
all three sizes on CPU; tiny also on Metal, MLX and wgpu. Dropping the sliding
windows moves the encoder by 0.70.
