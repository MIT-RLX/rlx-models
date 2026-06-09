# Metal Qwen3 embeds prefill/decode — upstream repro

## End-to-end fusion target

Production should be **one warmed RLX session**, not three stitched backends:

```text
prefill graph (talker, inputs_embeds)
  → per-frame megakernel (talker lm_head + CP AR + talker decode)
  → batched speech decode graph (pre_transformer + conv/vocoder)
```

[`fused_e2e::E2EPipelinePlan`](src/fused_e2e.rs) logs which stages are `RlxFusedGraph` vs `CpuEager` today. **Fully fused AR** needs:

1. **Talker** — native Metal/CUDA `inputs_embeds` bucketed decode (Metal Graph/HIR diverge upstream).
2. **CP** — fused Qwen3 graph competitive with the 15-step CPU micro-kernel on 0.6B (compiled CP is slower today).
3. **Speech** — Metal-fused conv stack after batched `pre_transformer` (conv tail still CPU).

Tier-1 codec-frame megagraph: `build_qwen3_tts_codec_frame_built` + `CodecFrameFusedEngine` (CP prefill + 14 decode + talker decode). Greedy lm_head on host; opt in with `RLX_QWEN3_TTS_CODEC_FRAME_FUSED=1`.

## Summary

**CpCompiledEngine** on Metal was compiling graphs on `Device::Metal` directly while **TalkerEngine** routes compile caches through `talker_compile_device()` → **CPU** by default. That mismatch caused CP prefill divergence (~12 max_abs) even though the same Qwen3 builder works on Metal when wired like the talker.

**Fixed in rlx-qwen3-tts:** `CpCompiledEngine` uses `cp_compile_device()` (CPU graphs on Metal unless `CP_METAL=1`) and skips `metal_compile_guard` when compiling on CPU. Talker may still use `RLX_QWEN3_TTS_METAL_COMPILED=1` independently.

**Workaround defaults (unchanged for latency):**

- Talker: **CPU eager** on Metal (correct + fast)
- Code predictor: **CPU eager** on Metal
- Opt into CPU compiled talker: `RLX_QWEN3_TTS_TALKER_EAGER=0`
- Opt into native Metal talker compile caches: `RLX_QWEN3_TTS_METAL_COMPILED=1` + `RLX_QWEN3_TTS_TALKER_EAGER=0`
- Opt into CP compiled on Metal session (CPU graphs): `RLX_QWEN3_TTS_CP_COMPILED=1` or `RLX_QWEN3_TTS_CP_METAL=1` (without `METAL_COMPILED`, graphs compile on CPU)

**Native Metal bucketed decode (`RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1`):** Still diverges from eager/CPU-compiled decode on 0.6B talker (bisect: layer-1 `max_abs≈12`, full 28L `max_abs≈52`, wrong `g0` after frame-0). CPU-compiled decode graphs match eager (`max_abs<5e-5`). Default production hybrid: **CPU eager talker decode** on Metal GPU sessions (`talker_eager_decode_default`). Upstream `../rlx/rlx-metal` work required before enabling native decode by default.

**GPU-resident K/V (`RLX_QWEN3_TTS_GPU_KV`, default on Metal megakernel):** Re-enabled for native Metal decode. Metal/wgpu re-upload prefix K/V each step (`sync_gpu_kv_to_host` + `refresh_kv`) because GPU handle feeds do not persist across runs; CUDA/ROCm keep resident K/V until bucket change.

Regression (with `RLX_QWEN3_TTS_PARITY=1`):

```bash
RLX_QWEN3_TTS_METAL_COMPILED=1 RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1 \
  cargo test -p rlx-models --test qwen3_tts_talker_eager_vs_compiled metal_decode_with_eager_kv_matches_eager metal_native_decode_with_gpu_kv_matches_eager --release --features metal
cargo test -p rlx-models --test qwen3_tts_hf_parity --release --features metal
```

**Production RTF (Metal session, CPU eager talker+CP, `VECLIB_MAXIMUM_THREADS=1`):** steady 12-frame utterance **~1.20 RTF**; CP ~58ms/frame dominates codec AR. Compiled talker/CP graphs remain slower than eager on 0.6B; native Metal decode is for `TALKER_EAGER=0` paths.

MPSGraph reshape crash on padded bucket seq is fixed in local `../rlx/rlx-metal`; default remains `RLX_DISABLE_MPSGRAPH=1`.

## Repro (in-tree)

```bash
export RLX_QWEN3_TTS_DIR=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice
export RLX_QWEN3_TTS_PARITY=1
cargo test -p rlx-models --test qwen3_tts_cp_metal_upstream_repro --release --features metal -- --nocapture
```

Metal CP compiled prefill should match eager within tolerance (max_abs &lt; 0.05).

## Graph details

- Builder: `rlx_qwen3::build_qwen3_prefill_embeds_built` / `build_qwen3_decode_embeds_built`
- Weights: `talker.code_predictor.model.*` remapped to Qwen3-canonical names
- Layers: **5** (`code_predictor_config.num_hidden_layers`)
- Hidden: 1024, heads 16 / KV 8, head_dim 128
- RoPE: standard 1D
- Metal guard: `RLX_DISABLE_MPSGRAPH=1` unless `RLX_QWEN3_TTS_METAL_MPSGRAPH=1`

## What passes

- CPU compiled vs eager (talker + CP): **max_abs &lt; 5e-5**
- Metal CP compiled (CPU compile device) vs eager: **max_abs &lt; 0.05**
- CPU HF greedy codec golden: **22/22**
- Metal + CPU eager talker + CPU eager CP: **22/22** HF golden
- Metal `METAL_COMPILED=1` (eager prefill + CPU decode): **22/22** HF golden + talker isolation tests

## Env knobs

| Variable | Effect |
|----------|--------|
| `RLX_QWEN3_TTS_METAL_MPSGRAPH=1` | Opt into MPSGraph (reshape fix in rlx-metal; still has attention parity limits on fused QKV) |
| `RLX_QWEN3_TTS_METAL_COMPILED=1` | Metal prefill compile + eager prefill / CPU decode hybrid (native Metal decode still upstream) |
| `RLX_QWEN3_TTS_METAL_DECODE_HIR=1` | Experimental HIR Metal decode (diverges; default off) |
| `RLX_QWEN3_TTS_METAL_DECODE_NATIVE=1` | Native Metal bucketed decode thunks (default off; CPU graphs) |
| `RLX_QWEN3_TTS_FUSED_E2E=0` | Opt out of e2e RLX fusion target logging |
| `RLX_QWEN3_TTS_CP_METAL=1` | CP compiled backend on Metal session |
| `RLX_QWEN3_TTS_CP_METAL_UNFUSE=1` | Unfuse Metal CP compile profile |

Local `../rlx` patch (`.cargo/config.toml`) for upstream `rlx-metal` work.
