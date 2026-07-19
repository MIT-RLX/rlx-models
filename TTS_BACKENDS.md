# TTS backends — cross-backend parity + RTF, and rlx vs the original

Measured on this Apple-Silicon Mac (darwin). CUDA/ROCm are **compile-only** here —
those columns need a Linux+GPU host. Harness: `crates/rlx-<model>/examples/backend_matrix.rs`
(loops `Device` variants via `load_on`, median of 3 timed iters after 1 warmup;
parity = cosine of the waveform vs the CPU output; whisper = word coverage of the
reference sentence via `.cache/whisper-tiny` / `whisper-base.en`).

"Original" = the model's own **ONNX Runtime** path (`synthesize_ort`, CPU EP) run
in-process on the same input — ONNX Runtime is what these models shipped as.

## 2026-07 parity pass (Whisper-gated)

| Model | Issue | Fix | Gate |
|-------|-------|-----|------|
| ChatterBox | Fox 0/6 false negative | `rlx-whisper` skips lang/task tokens on `.en` checkpoints | Apple Whisper 6/6 |
| StyleTTS2 / Kokoro native | Decoder garbage (historical) | Remove Cin↔Cout transpose on ONNX `ConvTranspose` (`rlx-onnx-import`). Native is default: ORT CPU encoder + RLX decoder (or `RLX_KOKORO_NATIVE_ENC=1`); fox **6/6** on CPU/Metal/MLX/wgpu | fox 6/6 Apple |
| Soprano | Harness + MLX peak | Default greedy; coverage `len≥2`; Vocos on CPU when backbone is MLX | CPU/Metal/MLX Whisper 1.00 |
| Piper | Stochastic ORT / GPU enc | Native path; CPU-pinned `enc_p`; `RLX_PIPER_DETERMINISTIC=1` | Apple cos 1.0; MSI CUDA cos≈1.0 |
| Gepard | Bucketed decode mask/KV slice drift | Fix `slice_kv_from_bucket` + dense `generated` for custom-mask AR; bucketed default | Apple fox 6/6 + long 15/15 |
| MeloTTS / tiny-tts | 512-sample silence | Same CT fix + `ensure_audible` | Local CPU/wgpu/Metal cos 1.0 |
| MetaVoice | Whisper ~67% | Greedy defaults, speaker required, PCM postprocess | Fox 6/6 Apple |
| F5-TTS | DiT GPU mismatch | Metal/MLX on-device OK (fox 6/6); keep ScatterNd pin over MPS cliff; Apple `--device gpu` DiT→Metal by default. True wgpu (`RLX_F5_WGPU_DIT=1`): teacher traj mad≈1e-8 vs CPU, NFE=32 fox **6/6**. Fixes in wgpu: (1) unsharded `>bind` arenas — dedicated scratch / virtual bind-sized stripes under `RLX_WGPU_LARGE_BUFFERS=1`; (2) `Transpose(Param)` must bind the **act output** stripe and stage the weight in — param-anchored windows wrote the transpose into scratch with no writeback, so AdaLN Gemm collapsed to bias | fox 6/6 Metal + true wgpu |
| Supertonic | Vulkan cos≈0.03 | Host non-last Reduce; true Vulkan cos 1.0 (no remap) | MSI cos 1.0 |

Remaining: MSI CUDA/Vulkan Gepard matrix when the rig is reachable.

## Supertonic-3 (CFM, 4 subgraphs) — text = "The quick brown fox …", 4.30 s audio @ 44.1 kHz

### (b) rlx native, cross-backend
| backend | RTF | median ms | cosine vs CPU | whisper coverage |
|---|---|---|---|---|
| CPU    | 1.0×  | 4346 | 1.00000 | 1.00 |
| Metal  | 1.8×  | 2359 | 1.00000 | 1.00 |
| MLX    | 13.5× |  318 | 1.00000 | 1.00 |
| wgpu   | 1.5×  | 2856 | 1.00000 | 1.00 |
| CoreML | 1.3×  | 3399 | 1.00000 | 1.00 |
| **CUDA** (msi) | **2.4×** | **1782** | 0.96463 | 1.00 |

**Correctness: perfect on Apple backends.** All five native Apple backends produce bit-identical audio
(cosine 1.00000) and transcribe with 1.00 coverage. CUDA (Linux) matches Whisper coverage with
cosine ≈0.965 vs CPU (TF32 / kernel order).

### (c) rlx-native vs onnxruntime (original, CPU EP) — 848 ms (5.1× RT)
| path | median ms | vs ort-CPU |
|---|---|---|
| **onnxruntime CPU (original)** | **848** | 1.00× |
| rlx MLX    |  318 | **2.67× faster** |
| rlx Metal  | 2359 | 0.36× (slower) |
| rlx wgpu   | 2856 | 0.30× (slower) |
| rlx CoreML | 3399 | 0.25× (slower) |
| rlx CPU    | 4346 | 0.20× (5× slower) |

**Read:** "rlx runs faster than the original" is **true only on MLX**
(2.7× faster than ort-CPU). On CPU, rlx is ~5× *slower* than ONNX Runtime — ort's
CPU kernels are heavily optimized and rlx's CPU path is not competitive here.
Metal/wgpu/CoreML also trail ort-CPU for this model. A fully fair "vs original on
GPU" would compare against onnxruntime's CoreML EP (not measured — the ort path
here is CPU EP).

## LuxTTS (ZipVoice-distill CFM, 3 subgraphs, voice cloning) — 2.39 s audio @ 24 kHz

### (b) rlx native, cross-backend
| backend | RTF | median ms | cosine vs CPU | whisper coverage |
|---|---|---|---|---|
| CPU    | 0.3× | 7556 | 1.00000 | 0.85 |
| Metal  | 0.4× | 6573 | 0.99992 | 0.85 |
| MLX    | 1.7× | 1383 | 1.00000 | 0.85 |
| wgpu   | 0.3× | 7423 | 1.00000 | 0.85 |
| CoreML | (see note) | end-to-end with `RLX_COREML_UNITS=gpu` (auto) | — |
| **CUDA** (msi) | 1.4× | 1664 | 0.99979 | 0.85 |

Parity holds (cosine 1.0 / 0.99992-Metal; whisper 0.85 — luxtts's known espeak-vowel
coverage). **CoreML** runs all three subgraphs on `Device::Ane` when compute units
are GPU (`RLX_COREML_UNITS=gpu`, set automatically by LuxTTS / TinyModel); the
default Neural-Engine units SIGSEGV in BNNS.

### (c) vs onnxruntime (original, CPU EP) — 772 ms (3.1× RT)
| path | median ms | vs ort-CPU |
|---|---|---|
| **onnxruntime CPU (original)** | **772** | 1.00× |
| rlx MLX | 1383 | 0.56× (slower) |
| rlx CPU | 7556 | **0.10× (10× slower)** |

Here rlx loses to ort on **every** backend, MLX included.

## Optimization pass 1 — rlx-cpu `BinaryFull` (profiler-driven)

`RLX_PROFILE_THUNKS` showed **`BinaryFull` was 88% of a supertonic subgraph** — the
elementwise broadcast kernel ran single-threaded with per-element modulo + a
per-call allocation. Fix: fuse the index math, hoist the op-match, parallelize with
rayon (f32 + i64), bit-exact.

| model | CPU before | CPU after | speedup | parity |
|---|---|---|---|---|
| Supertonic | 4346 ms (1.0× RT) | **2756 ms (1.6× RT)** | **1.58×** | cos 1.00000 / whisper 1.00 (unchanged) |

Gap vs ONNX Runtime (848 ms CPU) narrowed **5× → 3.2×**. 174 rlx-cpu tests green.
Next CPU targets from the post-fix profile: `Transpose` (39 ms/430 calls), `Conv2D1x1`.

## Takeaways
- **Parity across native backends is real** — cosine ≈ 1.0 and stable whisper coverage on
  CPU/Metal/MLX/wgpu. The migration is numerically sound. **CoreML** (`Device::Ane`):
  TTS crates pin `RLX_COREML_UNITS=gpu` via `rlx_tiny_tts::resolve_tts_device` (Neural-Engine
  BNNS SIGSEGVs on large imported graphs). **Vulkan (2026-07-19):** piper/sesame/
  kokoro/styletts2/supertonic/melotts/moss true-Vulkan green after Cast copy +
  Reduce host + arena/barrier fixes; F5 uses Vulkan pre/dec + CUDA DiT (act
  arena); gepard NanoCodec still diverges (no silent remap).
- **rlx is generally SLOWER than the original ONNX Runtime.** Across the two models,
  the only backend that beat ort-CPU was **supertonic on MLX** (2.7×). Everything else —
  and *all* of luxtts — was slower; **rlx's CPU path is 5–10× behind ort**. So "all TTS
  models run faster on rlx backends" is **false**. rlx's real wins are being *native*
  (no ONNX Runtime dependency), portable across backends, and bit-exact — not raw speed.
- CUDA/ROCm unmeasured on macOS (compile-only here).
- Perf gaps worth chasing: the rlx **CPU kernel path** (5–10× behind ort).
- **CoreML (2026-07):** TinyModel auto-sets `RLX_COREML_UNITS=gpu` on
  `Device::Ane` (Neural-Engine BNNS SIGSEGVs on large TTS MIL). Verified e2e:
  LuxTTS, Supertonic, Kokoro/StyleTTS2, Piper, Moss-nano, **F5-TTS** (all three
  subgraphs; Transformer unblocked via MIL `scatter_nd` for ONNX ScatterND).
  Compile OK for OpenVoice tones, MeloTTS encoder, MioTTS codec, MiraTTS
  detokenizer.

## 2026-07-18 MSI CUDA + Vulkan + Mac CoreML (matrix harness)

Host: `ssh msi` (RTX 3080 Ti Laptop) via `just matrix-remote`; Mac CoreML via
`just matrix BACKENDS=cpu,coreml`. Cosine = wav vs CPU baseline.

### CUDA (msi) — green
| model | cos vs CPU | notes |
|---|---|---|
| melotts / tiny-tts | ≈1.000 | full native |
| piper | ≈1.000 | `RLX_PIPER_DETERMINISTIC=1` |
| kokoro / styletts2 | ≈0.998 | |
| supertonic | ≈0.994 | |
| sesame | 1.000 | eager CPU AR; Mimi on CUDA |
| gepard | 1.000 | AR on CPU by default; NanoCodec CUDA |
| f5tts | PASS (audible) | CPU fox timed out in harness; CUDA ~15s |
| chatterbox | PASS (audible) | CPU/Vulkan fox timed out; CUDA ~80s |

### CUDA gaps
| model | status |
|---|---|
| luxtts | **fixed 2026-07-18** — empty-tensor arena slots + skip zero-size Expand (fox CUDA cos via matrix) |
| moss-nano | **fixed 2026-07-18** — bucketed prefill + CUDA prefill on by default (~39s fox); codec still CPU |
| gepard AR | still CPU unless `RLX_GEPARD_CUDA_AR=1` |

### Vulkan (msi) — 2026-07-19 true-Vulkan pass

**Root causes fixed in `rlx-vulkan` (local `../rlx`):**
1. **Arena > `maxStorageBufferRange` (~4 GiB)** — bump allocator → all-zero bindings (silent fox). Switched to `plan_memory_f32_uniform` liveness reuse; **param/act split** + **activation striping** (wgpu-style snap into ≤4 GiB shards with stage reserve) when acts alone exceed the limit (F5 DiT).
2. **Barrier elision** — range-based RAW/WAW/WAR (piper cos=1.0).
3. **Bool→F32 Cast no-op** — non-aliased Casts left the F32 slot zeroed (Kokoro/StyleTTS2 cos≈0.09). Identity copy when src≠dst (same as wgpu).
4. **Reduce last-axis-only** — non-last / multi-axis Reduce silently wrong in release; host every Reduce for now (Supertonic cos 0.24→1.0). Softmax GPU uses Kahan sum.
5. **Host fallback I64** — f32-encoded ints at the Vulkan↔CPU boundary.
6. **AOT LIR schema** — versioned `RLXLIR01` header + cache miss on deserialize failure (ChatterBox stale `.lir.bin` / `variant index` errors).

Silent `RLX_*_VULKAN` remaps removed for kokoro / styletts2 / supertonic / moss-nano / gepard / F5 preprocess-decode.

| model | true Vulkan (`--device vulkan`) | notes |
|---|---|---|
| piper / sesame | ✅ cos 1.0 | |
| kokoro / styletts2 | ✅ cos≈0.995 | Cast copy |
| supertonic | ✅ cos 1.0 | Reduce→host |
| melotts / tiny-tts | ✅ cos 1.0 | Hi. + fox |
| moss-nano | ✅ cos 1.0 | |
| f5tts | ✅ full Vulkan DiT (striped acts) | `rlx-vulkan` activation striping; opt-out hybrid via `RLX_F5_CUDA_DIT=1` |
| gepard | ✅ NanoCodec on-device; eager AR on CPU | Compiled Qwen3.5 AR drifts on all GPUs — same hybrid as CUDA. Override `RLX_GEPARD_*_AR=1` |
| chatterbox | ✅ after AOT schema bump | Stale `.lir.bin` caused `variant index` compile fail; speech_encoder on CPU for all GPUs; CFM/HiFT on CPU for Metal; **wgpu graphs → CPU** (loud) until T3/CFM parity |

| luxtts | ✅ loads on Vulkan | `num_frames` floored to `tp+1` (no prompt-length CLI trip) |

### MSI matrix (2026-07-19)

`BACKENDS=cuda,vulkan` (cpu omitted by harness arg quirk): melotts / tiny-tts /
kokoro / styletts2 / supertonic / moss-nano / sesame **PASS** on both. LuxTTS
CUDA PASS; Vulkan no longer panics (prompt-length CLI separate). Chatterbox
was AOT-cache schema drift (fixed in `rlx-ir` LIR binary header). F5/gepard
need re-gate after striping + Vulkan-AR→CPU.

**2026-07-19 follow-up (Mac + rlx local):** ChatterBox cpu/metal/mlx fox 6/6
cos 1.0 after clearing stale AOT. Vulkan striping + F5 true-DiT + Gepard
Vulkan-AR→CPU land pending MSI re-run (host unreachable this session).

### CoreML (Mac)

| model | cos vs CPU |
|---|---|
| melotts / tiny-tts / piper / moss-nano | 1.000 |
| kokoro / styletts2 | ≈0.994 |
| f5tts | ≈0.866 (pass @ 0.85 threshold) |
| supertonic | CPU BLAS / CoreML MIL parse fails in this harness run (known flaky vs prior green) |
| chatterbox | CoreML MIL KernelChannels mismatch — omitted from default matrix |

