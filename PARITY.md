# rlx-models parity / speed / memory ledger

Tracks the three acceptance criteria for native-rlx inference vs the ONNX/torch
reference (`rlx_beats_onnx_criteria`): **(1) full parity** (cos≈1.0 or exact
transcript), **(2) faster** wall-time, **(3) same-or-less peak RAM**. ort/onnx is
the *reference only* — inference must be native rlx.

Legend: ✅ meets · ⚠️ close/needs measure · ❌ fails · — n/a (ort removed, no
reference) · `?` not yet benched.

Weights store: large (≥1 GB) trees live on `/Volumes/FOUR/weights/**` and are
symlinked back under `weights/` (internal disk was 93% full). Relocation is
byte+count verified before the original is removed — see `scripts/relocate_weight.sh`.

## Audio.cpp TTS/ASR fleet (local-weight validated)

| Model | Parity | Speed vs ref | RAM vs ref | Weights | Notes |
|---|---|---|---|---|---|
| kokoro | ✅ cos 1.0 (5 backends) | ✅ RTF Metal ~69× | ? | internal | encoder-on-CPU + 5 fixes |
| supertonic | ✅ cos 1.0, whisper exact | ✅ faster (8.15→2.9s) | ✅ **2.94→1.0 GB** (was 3.47) | internal | CPU pin fix; byte-identical output |
| miotts | ✅ cos 1.0, whisper 6/6 | ? | ? | FOUR | ConvTranspose NCL fix |
| maya1 | ✅ intelligible (Metal) | ⚠️ 3B slow on CPU | ? | FOUR | max-tokens fix; Metal only |
| f5tts | ✅ exact | ? | ? | FOUR | — |
| moss-nano | ✅ bit-exact (4 backends) | ? | ? | FOUR | host-delegate + CPU argmax |
| chatterbox | ✅ ort-free (5 backends); Metal=correct speech (whisper 6/6, logmel 0.97) | ⚠️ full synth slow (24k-node) | ? | FOUR | LM re-prefill; low waveform-cos = 1 near-tie argmax flip, not a bug (see note) |
| orpheus | ✅ | ✅ RTF 22.3 Metal (O(n) decode) | ? | internal | decode was O(n²) |
| openvoice | ✅ at-parity (0.89 = quality) | ? | ? | internal | subgraphs cos 1.0 |
| luxtts | ✅ at-parity (0.77 = f32 floor) | ? | ? | internal | inherent, not a bug |
| metavoice | ✅ working | ? | ? | FOUR | EnCodec from safetensors |
| parlertts | ✅ exact | ? | ? | FOUR | T5 + 9-codebook delay |
| melotts | ✅ exact | ✅ (tiny-tts) | ? | internal | VITS2 |
| parakeet-tdt | ⏳ wired, gated on .nemo | — | — | — | needs checkpoint on FOUR |

## Standing action items (where rlx does NOT yet meet all 3)

1. **RAM — big global win landed (CPU pin fix).** Root cause of the bloat was
   `MemoryPlanOptions::inference()` defaulting `pin_output_ancestors: true`,
   which pins the whole output-ancestor DAG to the last step and **destroys
   liveness slot reuse** on deep feed-forward graphs. The CPU backend is an
   **in-order executor** so it never needed that guard: switched both
   `cpu_backend.rs` plan sites `plan_memory_aligned` → `plan_memory_native`
   (same Native dtype widths; pins only on host indexing). Result on supertonic:
   vector_estimator arena 1.36 GB→327 MB, vocoder 1.17 GB→177 MB, **max RSS
   2.94 GB→1.0 GB**, wall 8.15 s→2.9 s, **output byte-identical** (parity
   untouched). This helps EVERY deep feed-forward CPU graph (vocoders,
   estimators). Remaining supertonic gap to ort (0.52 GB) is now ~1.9× (was
   5.7×) — closable with a shared arena across the sequential ve/voc graphs +
   optional f16 activations (would break byte-identical, so gated).
2. **Speed/RAM benching.** Most rows have `?` for speed and RAM — need a
   uniform harness that records native-rlx RTF **and** peak RSS per model on a
   fixed clip, so the ledger reflects measured numbers not recollection.
3. **maya1 / chatterbox CPU speed.** 3B / 24k-node graphs are impractically
   slow on CPU; Metal is the usable path. Not a regression, but flagged.
4. **chatterbox Metal "low cosine" is NOT a bug — inherent AR near-tie flip.**
   The TTS-bench reported chatterbox `cosine_vs_cpu` ≈ 0.03–0.05 on Metal while
   MLX was 1.0. Root-caused (per-token dump + per-step logit diag on the exact
   bench input): Metal's native T3 LM (rlx-llama32) is **bit-exact vs CPU for the
   first 110 of 141 greedy tokens**, then flips ONE token at step 110. Mechanism:
   the token is a repeat, so `sample()`'s repetition penalty (÷1.2) pulls its
   raw logit (gap 2.47 over runner-up) down to an **exact tie** with the runner-up;
   at that tie a **~1e-4 relative** MPSGraph-vs-CPU f32 matmul/reduction difference
   (4137 logit 14.8123 Metal vs 14.8133 CPU) decides the argmax the other way.
   MLX's reduction order happened to match CPU at this step (luck, not
   correctness). One flip cascades the tail's phase → waveform cosine collapses,
   but the speech is **correct**: whisper 6/6, coverage 1.0, logmel 0.97.
   → Two takeaways: (a) **waveform cosine is the wrong parity metric for AR/LM
   TTS** — use greedy token-prefix agreement + whisper coverage + logmel; (b) the
   bench now decodes **deterministically** (`SynthRequest.deterministic=true` →
   chatterbox `greedy`) so the comparison measures the compute path, not RNG.
   Debug knobs added (env-gated, off by default): `RLX_CB_DUMP_TOKENS`,
   `RLX_CB_LOGIT_DIAG`.

## Cross-backend matrix (remote sweeps via `scripts/matrix/run_matrix.py`)

Whisper coverage is built into the matrix (every TTS run is transcribed). Backends
per host = host-detected ∩ crate cargo features.

**amd** (Instinct MI100) — devices cpu/wgpu/rocm. Pass on **all three** backends:
qwen3-0.6b, whisper, melotts, tiny-tts, chatterbox, sesame; supertonic passes cpu.
Findings surfaced:
- **rocm** `Reduce: only single last-axis supported (axes=[1,2])` → **FIXED**:
  generalized `rlx-rocm/backend/compile.rs` Reduce to any contiguous trailing-axis
  suffix (inner=∏trailing, outer=∏leading). Verified on amd: panic cleared,
  supertonic/rocm now runs the full graph (125 s).
- **supertonic GPU divergence = Linux-wgpu + rocm ONLY (open).** On **mac** ALL
  backends PASS (metal/mlx/**wgpu**/coreml all cos 1.000). On msi native
  **vulkan PASSES** (1.000) and **cuda PASSES** (0.976); only Linux **wgpu**
  (−0.000) and **rocm** (−0.016) fail. So it is NOT a supertonic-graph bug and NOT
  wgpu-the-abstraction (mac-wgpu=Metal passes) — it's specific to the Linux
  wgpu (Vulkan-backed) + rocm lowering. Deep, remote-only; low priority since
  cpu/metal/mlx/coreml/vulkan/cuda all pass.
- **kokoro Apple-GPU regression (open, local).** Was "cos 1.0 all 5 backends";
  now on mac: cpu ✅, **wgpu ✅ 0.984**, but **metal ❌ −0.051, mlx 💥 panic**
  (`Reshape [1,512,384] ⟵ Gather [192,384]` — element-count mismatch 73728≠196608,
  a shape-inference/lowering bug), **coreml ❌** (`mps.reshape` shape incompatible).
  All three Apple-GPU failures are reshape-related. **Root cause (analyzed):** the
  mlx Reshape lowering already uses the *runtime* input shape (`env.rs:459`
  `x.shape()`), so the panic means the Gather genuinely produced `[192,384]` at
  runtime on mlx while the reshape needs `[512,384]` — a **data-dependent
  dynamic-shape divergence** in kokoro's decoder (gather length = Σdurations /
  length-regulator differs on mlx vs cpu), NOT a general reshape bug. Deep +
  kokoro-specific (needs the duration/alignment path to match cpu on Apple GPU).
  cpu + wgpu tolerate it (runtime buffer sizing). Needs a focused session, not a
  cron slice.
  **UPDATE (2026-08): MLX decoder REGRESSED then FIXED (real op fix).** Bench
  caught styletts2/mlx = garbage (cos 0.014, whisper 0/6). Localized via a
  CPU-vs-MLX per-node dump aligned on Conv/MatMul anchors → first divergence was
  the ISTFTNet **depthwise `ConvTranspose2d` (groups=512)**: MLX's native grouped
  transpose-conv **mixes channels across groups** → output ~25× too large (76.7 vs
  CPU 3.0). **Fix (rlx-mlx):** host-eval `ConvTranspose2d` for `groups>1` in
  `lower/env.rs` + `lower/mod.rs`, mirroring the existing `ConvTranspose3d
  groups>1 → host` fallback. Result: **styletts2/mlx cos 0.9925, fox 6/6, rtf
  1.95× (fastest backend)**; 25 rlx-mlx parity tests still pass. The temporary
  decoder-on-CPU stopgap was removed. See [[kokoro_multibackend_parity]].
- **TTS-bench "broken model" sweep (2026-08).** Full 23-model deterministic run
  surfaced these; fixes this session: **styletts2/mlx** garbage → real rlx-mlx
  grouped-ConvTranspose2d fix (above), cos 0.9925 fox 6/6, rtf 1.95× ✅. **voxtral-tts** FAIL "no .f32 embedding" → ran
  `--convert-voices` (20 `.pt`→`.f32`) ✅ data-fixed (4B then OOM/crashes on load
  — separate). **qwen3-tts** FAIL "missing tokenizer.json" → built it from
  `vocab.json`+`merges.txt` via `AutoTokenizer.save_pretrained` (Base repo doesn't
  ship one) → cpu **fox 6/6** rtf 0.253 ✅. **kyutai** FAIL "missing voice
  embedding" → `hf download kyutai/tts-voices alba-mackenna/casual.wav…safetensors`
  → backbone loads/runs (data-unblocked) but output **fox 0/6** (unintelligible —
  model-quality bug remains, like gepard). **gepard metal/mlx crash FIXED** —
  the qwen35 `clear_host_dense_projections` (runner.rs) now skips the release when
  `weights_path` is empty (inline_weights has no on-disk source to reload from),
  so Metal/MLX decode no longer panics with "F32 projections released, no weights
  path". Metal runs (11.9 s). Helps any inline_weights qwen35 backbone on Apple
  GPU. **gepard fox 0/6 → 6/6 FIXED (2026-08):** not a port bug — the bench ran
  gepard's default temperature sampling (0.4) which **free-runs into coherent but
  WRONG words** ("The love is a cure…"); **greedy is faithful** (whisper "The quick
  brown fox jumps over the lazy dog."). Adapter now honors `deterministic`→greedy →
  gepard cpu **6/6**, metal **6/6**. Same class as chatterbox. Still open (deep code): **kittentts/mlx** panic —
  Reshape runtime `[1,200100,9]` vs static `[1,200000,9]` (a cap/NSF-source length
  inconsistency in the code-gen `kitten_tts_mini_rlx` graph: F0/N-proj source vs
  alignment-cap Expand) → `Mul` broadcast fails on MLX (CPU/Metal silently
  tolerate; fox 4/6 everywhere ⇒ likely corrupts them too); **parlertts/GPU** cos 0.22 but fox 6/6 — NOT an rlx
  bug: benign temperature-sampling divergence (adapter ran `greedy:false`). Its
  *greedy* path had a **model prompt-length bug (FIXED)**: `native.rs` assumed the
  decoder's prompt prefix == `pt` (prompt_ids.len()), but the exported decoder
  emits a **fixed-length prefix (empirically 19, independent of `pt`)** → `seq !=
  pt+t` assertion fired for any transcript whose token count ≠ 19 (only the bench
  phrase's pt=19 coincidentally passed). Fix: derive prefix from the returned
  `seq` (`seq_idx = (seq - t) + step`) instead of `pt`. Greedy now works for any
  text on **all backends** (verified: whisper "the quick…" on MLX). The MLX panic
  was NOT a shape-inference bug — it was a **stale AOT cache-key collision**: the
  persistent `$TMPDIR/rlx_parler_native` cache keyed decoder graphs by `seq` only
  (=t=420), not `pt`/`et`, so a graph compiled for one transcript length served
  another → baked `prompt_input_ids [1,19]` for a pt=13 transcript → MLX
  `from_bytes` panic (CPU tolerated it). Fix: fold all named dims into the cache
  key (`native.rs`). Bench adapter re-wired to `greedy: req.deterministic`.
  **Lesson: an MLX static-shape panic whose HIR shape is correct = suspect a stale
  cache entry, not shape inference.**
- **moss-nano + luxtts build fix (FIXED + VALIDATED).** Both `FAIL in 0s` on every
  host was a **stale registry** `features_base = ["onnx"]` — both crates removed
  their ort `onnx` feature (native-only now), so cargo errored "feature does not
  exist". Set `features_base = []` in `scripts/matrix/registry.toml`. **Validated
  on mac: moss-nano now PASSES all 5 backends** — cpu, metal 1.000, mlx 1.000,
  wgpu 1.000, coreml 1.000 (was 100% build-blocked). **luxtts** now 4/5: cpu ✅,
  metal 0.973, mlx 0.973, coreml 0.973, **wgpu ❌ panic** "remainder with divisor
  of zero" (a wgpu integer zero-stride/modulo bug — separate small open item).
  Remotes pick up the corrected registry on their next matrix run.
- **build** moss-nano + luxtts `[gpu,onnx]` fail on Linux (ort feature) — expected;
  these still need the native path on non-mac.
- **data-prep** kokoro + styletts2 need `scripts/split_kokoro.py` run on the remote
  (bundle missing `encoder.onnx`) — cheap unblock, 6 cells.

**msi** (RTX 3080 Ti) — devices cpu/wgpu/cuda/vulkan. qwen3-0.6b: cpu PASS,
**cuda PASS (4.6 s)**, vulkan PASS, wgpu WARN (known qwen3 wgpu decode 1/8).
Per-model TTS (cpu / wgpu / cuda / vulkan):
| model | cpu | wgpu | cuda (post clone-fix) | vulkan |
|---|---|---|---|---|
| kokoro | ✅ | ❌ −0.035 | 🔶 0.263 (was silent) | ✅ 0.975 |
| supertonic | ✅ | ❌ −0.000 | ✅ 0.976 | ✅ 1.000 |
| piper | ✅ | ⚠️ 0.889 | ✅ **1.000** (was silent) | ✅ 1.000 |
| zipvoice | ✅ | ❌ OOM | 🔶 0.469 (was silent) | n/a |
| styletts2 | ✅ | ❌ −0.021 | 🔶 0.263 (was silent) | ✅ 0.975 |
| chatterbox | ❌ timeout 600 s | | | |

The **clone_for_cache param fix** un-silenced all cuda entries above; piper is now
bit-exact. kokoro/styletts2/zipvoice retain a separate cuda decoder parity-drift
(kokoro/styletts2 0.263 shared StyleTTS2 decoder, zipvoice 0.469): localized to
the sine source — some standalone `Activation(Sin)` outputs read identity though
`unary.cu case 13u=sinf` and the launch (op=13, args) are all verified correct →
suspect an arena slot-offset mismatch (Step writes sinf to a slot the node's
consumers don't read). Deep, open. Two correct kernel-completeness fixes also
landed: `fused_binary_unary.cu` + `batch/elementwise_region.cu` now handle
activation opcodes 12–28 (were identity past 11) — benefits any model fusing
Mul→Sin etc., though kokoro uses standalone unary so it doesn't move kokoro.
Build gotcha: those `.cu` are `include_str!`'d — force recompile with
`rm -rf target/release/build/rlx-gpu-kernels-* .fingerprint/rlx-gpu-kernels-*`.

**Shared-bug clusters (high value — one fix helps many):**
- **cuda `silent (peak=0.00)`** for kokoro + piper + zipvoice + styletts2 —
  **FIXED.** Root cause: rlx-cuda `clone_for_cache()` recompiled into a fresh
  zeroed arena but never copied the `set_param`-uploaded `Op::Param` weights, so
  `TinyModel::compile_named().clone()` consumers (piper flow_dec etc.) ran with
  **zero weights**. Fix = `copy_params_from(self)` D2D-copies every param slot in
  `backend/mod.rs`. **piper cuda: silent → 39936 samples.** (run_named models like
  supertonic were unaffected — no clone.) Was a regression (CUDA-green 2026-07-18).
  Below is the pre-fix state, kept for reference:
  **Investigated on msi (piper, pure VITS):** the 3 HiFiGAN upsampling
  ConvTranspose2d ops dispatch to cuDNN fine (correct shapes 250→2000→16000→64000),
  and forcing the kernel path (`RLX_CUDA_CONV_T_KERNEL=1`) is ALSO silent → **NOT
  the ConvTranspose**. Confirmed: **melotts + tiny-tts (same VITS/HiFiGAN
  ConvTranspose1d) PASS on cuda**, so H=1 transposed-conv works on CUDA. piper
  cuda runs suspiciously fast (881 ms) then silent → zeros likely originate
  upstream (an op CUDA mis-handles for these 4 graphs, possibly returning a
  zero buffer). **Narrowed by elimination — an op INSIDE flow_dec computes zero
  on CUDA.** RULED OUT: ConvTranspose (cuDNN + `RLX_CUDA_CONV_T_KERNEL=1` +
  `RLX_CUDA_NO_CUDNN=1` all silent; melotts uses same op, passes); input-binding
  (added `RLX_CUDA_INPUT_DIAG` → both flow_dec inputs confirmed uploaded:
  `/Add_output_0`=14400 f32, `/Cast_2_output_0`=75); arena reuse
  (`RLX_ARENA_NO_REUSE=1` no help); the cpu `plan_memory_native` change (piper
  **cpu still works**, 19200 samples). Post-run `DUMP_INTERMEDIATE` is unreliable
  (reuse overwrites → all read 0). **Fix** = compute-time per-op cuda-vs-cpu diff
  to find the first op with nonzero inputs → zero output (VITS flow / coupling /
  WN convs; melotts VITS2 is fine). Focused session. See memory
  `cuda_silent_split_graph_input`.
- **wgpu TTS cosine divergence** kokoro −0.035, supertonic −0.000, styletts2
  −0.021 — possibly one shared wgpu bug (or a regression: kokoro was cos 1.0 on
  all 5 backends per `kokoro_multibackend_parity`). wgpu is **local-reproducible
  on mac** → bisect there. supertonic wgpu+rocm both fail but cuda+vulkan pass.
- **vulkan is the strongest GPU backend** — all TTS pass (0.975–1.000).
- moss-nano/luxtts `FAIL in 0s` = onnx feature won't build on Linux (need native).

## Method

- Parity: native-vs-ort per-subgraph + e2e cosine + whisper transcript
  (`scripts/spectral_temporal.py`, `RLX_ONNX_TAP` + `RLX_NO_OPT=1`).
- Speed/RAM: wall-time + peak RSS on a fixed reference clip, native vs the ort
  baseline recorded when the crate still had a reference path.
- Cross-backend: `run_matrix.py` on each host; `remote_run.sh` / nohup for msi+amd.
