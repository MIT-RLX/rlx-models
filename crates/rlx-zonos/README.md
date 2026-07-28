# rlx-zonos — Zonos v0.1 transformer

Apache-2.0 TTS from [Zyphra/Zonos](https://github.com/Zyphra/Zonos):
espeak phonemes → 1.6B GQA transformer (delay-pattern DAC codes) → Descript DAC @ 44.1 kHz.

## Status

| Piece | Status |
|-------|--------|
| Weights | `just fetch-zonos` → `weights/tts/zonos/` |
| espeak G2P | `--features espeak` |
| PrefixConditioner + CFG | host (optional `--speaker-emb` 128×f32) |
| 26×2048 GQA AR | **compiled**, CFG **batch=2**, cached prefill, **GPU-resident KV** (MLX/Metal/CUDA/…) |
| DAC decode | same `--device` |
| Defaults | **sample** + min_p (Zyphra); `--greedy` for short prompts; adaptive `--max-tokens`; EOS hold until ~1.05× phoneme duration |
| Eager host fallback | `RLX_ZONOS_EAGER=1` |
| Metal | F16 Linear weights; native GQA (no `repeat_kv`); all-thunk default (hybrid opt-in `RLX_ZONOS_MPSGRAPH_HYBRID=1`); `RLX_ZONOS_DISABLE_MPSGRAPH=1` forces all-thunk |
| Backend matrix | `just zonos-backends` |

Pangram (`RLX_MAX_TOKENS=256`), after GPU-resident KV + F16 weights + native GQA +
split-K decode attention + fused QKV + small-M F16 GEMM + batch-major KV:

| Backend | peak | Whisper | wall |
|---------|------|---------|------|
| Metal | 0.274 | **100%** | **~27s** |
| MLX | 0.274 | **100%** | **~28–31s** |
| CPU | 0.274 | **100%** | ~75s |

Metal is at/above MLX on Apple Silicon for this workload (was ~76s). Schedule-split hybrid (`RLX_ZONOS_MPSGRAPH_HYBRID=1`) is still slower than all-thunk F16. `RLX_DISABLE_GPU_KV=1` forces the host pad path.

Omit `--max-tokens` to size the AR budget from phoneme length + speaking rate
(up to ~45 s). Long paragraphs also hold off codebook-0 EOS until ~1.05× the
expected frame count so endings are not cut mid-phrase; prefer default sampling
over `--greedy` for multi-sentence text.

## Setup

```bash
just fetch-zonos
just fetch-parler-dac
just zonos-demo                 # MLX by default
just zonos-demo DEVICE=metal
just zonos-backends             # CPU + Metal + MLX + Whisper
# CUDA: host-pad KV by default (`RLX_ENABLE_GPU_KV=1` to try resident KV).
# NVIDIA: ~20 s fox, Whisper ok-row; cov can lag CPU (sampling) — CPU ~56% on short budget.
RLX_DEVICES=cpu,cuda just zonos-backends
```
