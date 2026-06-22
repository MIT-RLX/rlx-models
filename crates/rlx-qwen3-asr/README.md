# rlx-qwen3-asr

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) speech recognition for RLX — a
Qwen3-Omni audio encoder feeding a tied-head Qwen3 decoder.

```text
PCM 16 kHz ─► log-mel (Whisper, 128 bins)
           ─► audio tower: chunked Conv2d stem ─► sinusoid pos
                                              ─► windowed transformer (×18)
                                              ─► LayerNorm + proj1/GELU/proj2
           ─► fuse at <|audio_pad|> slots
           ─► Qwen3 decoder prefill (KV + last logits) ─► greedy decode
```

Weights: HF safetensors `Qwen/Qwen3-ASR-0.6B` / `Qwen/Qwen3-ASR-1.7B`
(`thinker.audio_tower.*`, `thinker.model.*`, `thinker.lm_head.weight`).

## Architecture notes

- **Audio encoder** (`encoder.rs`): mel is split into `2·n_window`-frame chunks,
  each independently downsampled by three stride-2 Conv2d layers (freq
  128→16, time per chunk →13), flattened and projected to `d_model=896`. A
  per-chunk sinusoidal position is added, CNN padding is dropped, and the
  transformer runs **block-diagonal windowed attention** (windows of
  `t_pc · n_window_infer/(2·n_window)` post-CNN frames). A 2-layer GELU adapter
  maps `896 → 1024` (the LM hidden size).
- **Decoder** (`lm_flow.rs`): reuses `rlx-qwen3`. Prefill consumes the fused
  `inputs_embeds` and exports per-layer K/V + last-token logits; subsequent
  tokens decode via the standard Qwen3 KV-cache flow. RoPE is the plain Qwen3
  table — the config's interleaved M-RoPE collapses to 1-D RoPE because audio
  and text share identical position ids.
- **Mel** (`audio.rs`): matches `transformers.WhisperFeatureExtractor`
  (`n_fft=400`, `hop=160`, 128 Slaney mel bins, reflect padding, per-utterance
  `(log10·clamp+4)/4` normalization).
- **Tokenizer** (`tokenizer.rs`, `tokenizer` feature): byte-level BPE from
  `vocab.json` + `merges.txt`; chat-template special tokens are spliced by id.

## CLI

```bash
cargo run -p rlx-qwen3-asr --features tokenizer -- \
  --weights /path/to/Qwen3-ASR-0.6B --wav speech.wav --device cpu
```

Flags: `--weights PATH [--config PATH] --wav PATH [--system TEXT]
[--device cpu|metal|mlx|cuda|…] [--max-tokens N] [--dry]`.

The library API (`AsrRunner`) also accepts raw prompt ids via
`generate(prompt_ids, mel)` when the tokenizer feature is off.

## Parity

Validated against an independent `transformers`-library reference
(`Qwen3OmniMoeAudioEncoder` + standard `Qwen3` + `WhisperFeatureExtractor`;
`qwen3_asr` itself isn't in transformers). On a 2 s clip:

| Stage | CPU | MLX | Metal |
|---|---|---|---|
| mel | 4e-5 | 4e-5 | 4e-5 |
| audio encoder | 1.3e-6 | 2.6e-6 | 1.2e-6 |
| decoder prefill (argmax) | ✓ match | ✓ match | ✓ match |
| **e2e token ids** | **exact** | **exact** | **exact** |

`cpu`/`metal`/`mlx`/`gpu`(wgpu, =`vulkan`) all build on Apple Silicon;
`cuda`/`rocm` need their SDKs. The audio encoder is bit-exact on every backend.

**Metal note:** the *fused* Metal decoder kernels accumulate fp drift that flips
argmax over the 28 decoder layers (the unfused path, and the CPU/MLX fused
paths, are all exact — confirmed by op-level bisection: every standalone
primitive is ≤2.6e-6, only the fused composition drifts). The runner therefore
builds the decoder graphs **unfused on Metal** (`skip_fusion`), restoring
token-exact output; CPU/MLX keep fusion. Reproduce parity with
`examples/parity.rs` against `ref_lib.py`'s dumps.

## Benchmark

`Qwen3-ASR-0.6B`, 5 s clip, Apple M4 Pro, release. Graphs are compiled once and
reused, so timings reflect backend compute (the per-step weight reload in
`AsrRunner::generate` — a backend-independent I/O cost — is excluded).
Reproduce with `examples/bench.rs`:

```bash
cargo run --release -p rlx-qwen3-asr --features metal --example bench -- <model_dir> metal 5
```

| Stage | CPU | MLX | Metal |
|---|--:|--:|--:|
| mel (host) | 4.9 ms | 4.8 ms | 4.7 ms |
| audio encoder (65 tok) | 8291.8 ms | 57.6 ms | 127.8 ms |
| prefill / TTFT (80 tok) | 406.8 ms | 117.3 ms | 24.8 ms |
| decode / step | 416.7 ms | 120.2 ms | 15.2 ms |
| decode throughput | 2.4 tok/s | 8.3 tok/s | 65.7 tok/s |
| **e2e compute** | **13.70 s** | **1.62 s** | **0.34 s** |
| **RTF** (×realtime) | 0.37× | 3.08× | **14.7×** |

**BSF — Backend Speedup Factor (× faster than CPU):**

| Metric | MLX | Metal |
|---|--:|--:|
| **e2e** | **8.5×** | **40.3×** |
| audio encoder | 143.9× | 64.9× |
| prefill (TTFT) | 3.5× | 16.4× |
| decode / step | 3.5× | 27.4× |

Metal is fastest end-to-end (40× over CPU, 14.7× realtime) even on the unfused
decoder path; MLX wins the conv2d-heavy encoder but loses the decoder to per-op
dispatch overhead. CPU is below realtime (the encoder dominates its cost).

## JFK transcription (real audio)

`examples/jfk.rs` runs the canonical 11 s JFK inaugural clip (16 kHz mono),
offline and in 6 s streaming chunks, reporting wall time, RTF, and WER vs a
plain-text reference. Output is verbatim-correct on every working backend; the
model prefixes a `language English` language-ID tag, which is the only delta
from the plain reference (hence the non-zero WER) and the only streaming-extra
(the tag repeats per chunk):

> "…And so, my fellow Americans, ask not what your country can do for you; ask
> what you can do for your country."

| Backend | offline | offline RTF | offline WER | streaming (2×6 s) | streaming WER |
|---|--:|--:|--:|--:|--:|
| CPU | 55.1 s | 0.2× | 9.1% | 66.9 s | 18.2% |
| MLX | 14.8 s | 0.7× | 9.1% | 19.6 s | 18.2% |
| Metal | 13.8 s | 0.8× | 9.1% | 25.6 s | 18.2% |

Decode uses a **bucketed compile cache**: the decode graph compiles once per
power-of-two `past_seq` bucket (≈1 bucket for a single utterance) and is reused
across all steps via a per-step custom mask, so weights load only at compile —
never per token. This replaced an earlier per-token rebuild+reload loop that ran
at RTF ~0.1× and **OOM'd Metal** (each step's new graph grew the Metal pipeline
cache until it exceeded shared RAM). The fix is ~4× faster on CPU and makes Metal
the fastest backend (RTF 0.8×, no OOM). Accuracy is unchanged (token-exact).
