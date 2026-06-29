# Native-Parity Roadmap — Zero Alien Inference Runtimes

**Goal:** every model family in this workspace runs **100% on rlx backends**
(cpu / metal / mlx / cuda / rocm / wgpu) with **no rten, no ONNX/ort, no candle,
no moshi, no llama.cpp** in any shipped build — at **numerical parity** with the
reference implementation and at **equal-or-better speed**, via fully **fused
rlx-flow graphs** and **optimized (quantized / layout-planned) tensors**.

This is mostly *finishing in-flight migrations*, not greenfield work: most
families already have a native rlx path; the alien runtimes survive as defaults,
legacy fallbacks, or plumbing. This doc is the plan to drive each to done.

---

## 1. Definition of done (per family — three gates)

A family is "native-complete" only when all three pass:

1. **Parity gate.** Op-level *and* end-to-end numerical equivalence vs the
   reference, under tolerance (the `kitten_tts_mini_rlx` probe pattern: per-op
   `gelu`, `ffn`, `attention`, `qmatmul`, `scatter` tests vs reference, ~1e-3 f32;
   then stage outputs; then E2E). The reference (ort/candle/llama-cpp) is demoted
   to a **dev-only oracle** behind a `parity-*` feature — never in default/ship.

2. **Perf gate.** Native throughput **≥ reference** on the target backend
   (metal on Apple Silicon — benchmarked fastest, ~1144 docs/s for embeddings vs
   MLX ~833 / CPU ~296). Measured by a bench/example committed with the port.

3. **Removal gate.** Default features pull only rlx-native crates; the alien dep
   is `optional = true` behind `legacy-*`/`parity-*` (or deleted). `cargo tree`
   over the ship feature-set shows **zero** alien runtime (see §6 CI gate).

Never big-bang remove: each alien runtime stays behind a feature until its native
replacement passes all three gates.

---

## 2. Alien-runtime inventory (current state)

| Runtime | Crates | Native status today | What's left |
|---|---|---|---|
| **rten** (engine + tensor + imageproc) | `rlx-ocr` | Graph inference **already native** (`rlx` default). rten = 4 plumbing surfaces. | `tensor-ops` (pad/resize), `convert-rten` (weight load), `rten-tensor` (NdTensor I/O), `rten-imageproc` (geometry) |
| **ort / ONNX** | `rlx-kittentts`, `rlx-inflect-nano` | kitten: native `kitten_tts_mini_rlx` exists w/ op-parity suite, but `default=["onnx","rlx"]`. inflect-nano: only HiFi-GAN **vocoder** on ort. | Finish kitten parity + flip default; port Snake HiFi-GAN to rlx graph |
| **candle / moshi** | `rlx-mimi`, `rlx-moshi`, `rlx-kyutai-tts`, `rlx-clinicalbert` | mimi: native graph **already default**, candle = legacy `gpu-codec`. moshi: native port **in progress**. kyutai: transitive via mimi+moshi. clinicalbert: BERT on candle. | Delete mimi legacy; finish moshi native; port clinicalbert to rlx-bert |
| **llama.cpp** (`llama-cpp-2`) | `rlx-neutts`, `rlx-qwen35`, `rlx-models` | neutts: GGUF LM backbone via llama.cpp. qwen35/models: **parity oracle only** (optional). | Port neutts backbone to native `rlx-gguf`; keep qwen35 oracle dev-only |

candle reaches the tree **only** through `rlx-mimi → moshi → candle` (plus the
opt-in `parity-candle` in `rlx-models`); ort only through kittentts/inflect-nano;
rten only through rlx-ocr. So the islands are independent and can be cleared in
parallel.

---

## 3. Roadmap (phased by effort × ship-value)

### Phase A — harvest the near-done (low effort, immediate wins)

- **A1 · rlx-mimi: delete the candle/moshi codec path.** Native rlx-runtime
  codec is already the default; `gpu-codec` (candle/moshi) is "no longer
  required." Move `candle`/`candle-nn`/`moshi` to a `parity-mimi` dev feature (or
  delete), keep the codec round-trip parity test green. **Removes candle from
  mimi** → with B1, kyutai-tts goes candle-free.
- **A2 · rlx-ocr Stages 1–2: drop the heavy rten engine.**
  - *Stage 1 (convert-rten → offline):* run `export_rten_to_safetensors` **once
    offline**, publish `ocr-detection.safetensors` / `ocr-recognition.safetensors`
    to the model repo; the native loader (`prefer_safetensors_path`, the ST path
    consts) already consumes them. Drop `convert-rten` + `rten-model-file` from
    the ship build.
  - *Stage 2 (tensor-ops → native):* reimplement the only two ops used —
    `pad(BLACK_VALUE)` and `resize_image()` bilinear (3 sites in
    `rlx/detection.rs`) — via the `image` crate or an rlx-runtime resize. Flip
    `default = ["rlx","tensor-ops"]` → `["rlx"]`.
  - After A2 the entire rten **inference engine** (rten, rten-gemm,
    rten-vecmath, rten-shape-inference, rten-model-file) is gone; only the
    lightweight `rten-tensor` + `rten-imageproc` remain (Stages 3–4 / Phase B).

### Phase B — finish the in-flight ports (medium)

- **B1 · rlx-moshi: complete the native moshi LM.** Finish `rlx_gen` / `rlx_lm`;
  pass the depformer / gen / temporal parity tests (already being authored);
  demote `candle*`/`moshi` to `parity-moshi`. **Removes candle from moshi.**
  (A1 + B1 ⇒ `rlx-kyutai-tts` is fully candle-free.)
- **B2 · rlx-kittentts: flip to native.** Drive `kitten_tts_mini_rlx` through the
  op → stage → E2E parity ladder; set `default = ["native","rlx"]`; make `onnx`
  a `parity-kitten` dev feature. **Removes ort from kittentts.**
- **B3 · rlx-ocr Stages 3–4: zero rten.**
  - *Stage 3:* replace `rten-tensor` `NdTensor`/`NdTensorView` (≈55 sites, I/O
    container) with an rlx-runtime tensor or a thin `{data: Vec<f32>, shape}`.
  - *Stage 4:* replace `rten-imageproc` geometry (≈120 sites: `RotatedRect`,
    `Rect`/`RectF`, `LineF`/`PointF`, `Polygon`, `BoundingRect`). Most are trivial
    arithmetic; the two real algorithms in `detection/postprocess.rs` are
    **`find_contours`** (Suzuki–Abe / Moore-neighbor border following, ~100 LOC)
    and **`min_area_rect`** (rotating calipers on the convex hull, ~80 LOC).
    Vendor a small `geom` module. **Removes rten entirely.**

### Phase C — remaining ports (medium / large)

- **C1 · rlx-inflect-nano:** port the **Snake HiFi-GAN vocoder** to an rlx-flow
  graph (acoustic stage is already host-eager). Drop `ort`.
- **C2 · rlx-clinicalbert:** retarget to the native rlx BERT graph (reuse
  `rlx-bert` / `build_bert_graph_sized`). Drop `candle`.
- **C3 · rlx-neutts:** port the GGUF LM backbone to native `rlx-gguf`. Drop
  `llama-cpp-2`.

### Phase D — purge & lock

- Delete now-unused optional alien deps from every manifest (or quarantine behind
  documented `parity-*` dev-only features).
- In `rlx-models`, make the `ocr` / `kittentts` / `tts` features pull their crates
  with `default-features = false` so no legacy path leaks into the catalogue.
- Land the CI deny-list gate (§6).

---

## 4. Performance strategy (fused graph + optimized tensors)

The native path should *beat* the alien runtimes, not just match them — the lever
is whole-model graph compilation that ort/candle/rten (op-by-op dispatch) can't do.

- **One fused graph per model.** Build the full forward as a single rlx-flow
  `BuiltModel` (`build_*_graph_sized` + `attach_built_params`) so `rlx-opt` fuses
  elementwise/matmul/norm/attention chains and plans buffers once — instead of
  per-op kernel launches + intermediate allocations.
- **CompileProfile per backend.** Compile with the device-matched profile
  (`compile_options_for_profile`), cache the `CompiledGraph`, reuse across calls.
  Default to metal on Apple Silicon (benchmarked fastest); fall back cpu.
- **Optimized tensors.** Load weights in the reference's quantization
  (kitten qmatmul, GGUF q-formats via `rlx-gguf`; safetensors otherwise) and use
  native q-matmul kernels — the `kitten_tts_mini_rlx` `ffn_qmatmul_ref` /
  `f32_before_qmatmul` probes exist precisely to lock this down. Keep activations
  in the planned layout; avoid host round-trips (cf. the rlx-ocr note to preserve
  NCHW rank so `resize_image` stays on-graph).
- **No host-eager hot paths in ship builds.** Acceptable for correctness during a
  port (inflect-nano acoustic), but move on-graph before declaring the perf gate.
- **Dual gate on every port PR:** (a) parity diff < tol, (b) native latency ≤
  reference on metal **and** cpu, with the bench committed.

---

## 5. Parity test methodology (formalize the existing pattern)

Mirror `kitten_tts_mini_rlx`'s ladder for every family:

1. **Op probes** — each primitive vs reference output, tol ~1e-3 f32
   (`gelu_tanh_single`, `ffn_output_add_bias`, `attention1_probe`,
   `ffn_qmatmul_ref`, `alignment_scatter_probe`, …).
2. **Stage** — encoder / decoder / vocoder / detector-mask outputs vs reference.
3. **End-to-end** — final artifact (audio / mel / text-boxes / logits) under a
   numeric or perceptual tolerance.

The reference oracle stays behind a `parity-*` feature (the existing
`parity-candle` / `parity-pytorch` / `parity-llama` pattern + the new
`parity-onnx` / `parity-moshi` / `parity-mimi` / `parity-kitten`) — built in CI
only, never shipped.

---

## 6. CI deny-list gate (lock it shut)

A workspace test that fails if any alien runtime appears in the **ship**
feature-set:

```sh
# expect: empty
cargo tree --workspace --edges normal \
  -e no-dev --no-default-features --features <ship-set> \
  | grep -E '\b(ort|onnxruntime(-sys)?|candle-(core|nn|transformers)|moshi|rten|rten-(tensor|imageproc|gemm)|llama-cpp-2)\b'
```

Run per family as each phase lands; once all phases complete, run for the whole
workspace ship profile. Keep the `parity-*`-gated oracles excluded (they're
dev-only).

---

## 7. Sequencing & risk notes

- **Independent islands** — rten / ort / candle / llama.cpp don't overlap, so
  Phases A–C can proceed in parallel by area.
- **Quantized ops are the hardest parity** — moshi & kitten native q-kernels must
  match the reference's rounding bit-for-bit-ish; budget the most probe effort
  there.
- **Only genuinely new code** is the rlx-ocr CV geometry (`find_contours`,
  `min_area_rect`); everything else is graph translation against an existing
  native scaffold.
- **Order of value:** A1 + A2 (candle out of mimi, heavy rten out of ocr) are the
  cheapest big wins; B1/B2 finish the two hardest islands (candle/ort); C/D are
  cleanup.

---

### Quick status checklist

- [x] A1 rlx-mimi — candle/moshi codec demoted to dev-only `parity-mimi` (default tree candle-free; oracle still builds)
- [x] A2 rlx-ocr — **Stage 2 done**: native `host_resize` pad/resize (parity-tested vs rten, <1e-6 / <1e-4; <7 ms at 2K²); `tensor-ops` out of default → heavy rten engine gone from default + skill's feature set. Stage 1 = native safetensors path already works without `convert-rten`; offline-publish the `.safetensors` to retire `convert-rten` in skill (ops step).

  > **⚠ Bench reality check (CPU, `ocr_parity` + `ocr_perf_vs_reference`, real weights/image).** Removing rten is mechanically clean and my host ops are parity- + perf-neutral. Status of the two native-graph gates:
  > 1. **Recognition parity — FIXED.** Root cause was *not* numerical: the native recognition graph never computed the GRU at all — it loaded the weights into throwaway `_` vars and **zero-padded** the conv features (stale comment: "GRU op not yet present in rlx-ir"). rlx-ir *does* have `Op::Gru`; it just lacked HIR wiring. Added `HirOp::Gru` (enum + `HirMut::gru` + lowering + inspect, mirroring `Lstm`) and implemented two bidirectional GRUs in `rlx-ocr` (repacking ocrs ONNX weights `z,r,h` → rlx `r,z,n`, split bias). Result: `recognition_logits_match_reference` **23.3 → 0.00004** vs rten. CTC greedy decode is argmax over logits, so this is functional text parity too. Detection parity already passed.
  > 2. **Perf FAILS hard** — native `get_text` median **101,672 ms** vs rten/ocrs **906 ms** → **~112× slower on CPU**. Host ops are <20 ms; the rest is the rlx CPU graph (conv-heavy U-Net + CRNN) vs rten's optimised CPU engine. Needs the fused-graph + optimised-conv work (and/or metal, the real ship backend — not yet benched) before the perf gate passes.
  >
  > **`Op::Gru` cross-backend parity** (`rlx-runtime/tests/cpu_gru_parity.rs`, uni/bi/2-layer/wide): CPU-native ✅, **unfuse decomposition** (the path MLX/CoreML/CUDA-host/ROCm/wgpu/TPU/autodiff take) ✅, **Metal** ✅ (vs ref + vs cpu, M4 Pro), **MLX** ✅, **wgpu** ✅ — all ≤1e-4.
  >
  > **Takeaway:** A2 REMOVAL gate ✅; recognition PARITY gate ✅ (+ all backends); PERF gate still open (native conv graph, not rten plumbing). "Drop rten in skill" now waits only on perf.
- [ ] B1 rlx-moshi — finish native LM, demote candle to `parity-moshi`
- [ ] B2 rlx-kittentts — flip default to native, `onnx` → `parity-kitten`
- [ ] B3 rlx-ocr — Stage 3 (native tensor) + Stage 4 (native geometry) → zero rten
- [ ] C1 rlx-inflect-nano — native HiFi-GAN vocoder, drop ort
- [ ] C2 rlx-clinicalbert — native rlx BERT, drop candle
- [ ] C3 rlx-neutts — native rlx-gguf backbone, drop llama-cpp-2
- [ ] D  purge unused alien deps + `default-features=false` in rlx-models + CI deny-list
