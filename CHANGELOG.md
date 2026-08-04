# Changelog

## 0.2.14 — release hardening & repo hygiene (2026-08-03)

Workspace `[workspace.package].version` = **0.2.14**, pinned to upstream
**`rlx*`** **0.2.14** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.14** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### Notable changes

- **Release hygiene: the whole workspace is `fmt`- and `clippy`-clean.**
  `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D
  warnings` now pass across every crate. Besides formatting, this fixed ~50
  `clippy` findings (`manual_is_multiple_of`, `manual_checked_ops`,
  `unnecessary_cast`, `field_reassign_with_default`, `ptr_arg`, `needless_return`,
  `redundant_clone`, `repeat().take()` → `repeat_n`, …) and several examples/tests
  that had drifted from current APIs: six Gemma parity tests were missing newer
  `GemmaConfig` fields; the `gemma4_e2b` backend-parity test now uploads packed
  weights through the `PackedSrc` enum (`Owned`/`Borrow`/`F32`) like production;
  and the `backend_sweep` (`runner`) / `cmp_ort` (`onnx`) examples gained
  `required-features` so `--all-targets` skips them under default features instead
  of failing to compile.
- **Repo layout:** `rlx-tiny` / `rlx-tinystories` now default their trained
  `.rlxts` checkpoints under `weights/<model>/` (`weights/tinystories/…`,
  `weights/tiny/…`) rather than the repo root, and `checkpoint::save` creates the
  parent directory. Generated `memory_probe` / retention benchmark output dirs are
  consolidated under a git-ignored `bench_out/`.

## 0.2.8 — model coverage expansion (2026-06-21)

Workspace `[workspace.package].version` = **0.2.8**, pinned to upstream
**`rlx*`** **0.2.8** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.8** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### New model crates

Audio codecs: `rlx-snac`, `rlx-encodec`, `rlx-speechtokenizer`,
`rlx-wavtokenizer`, `rlx-xcodec`, `rlx-facodec`, `rlx-nanocodec`,
`rlx-mimi`, `rlx-dac`, `rlx-tsac`.

ASR / audio: `rlx-wav2vec2-asr`, `rlx-nemotron-asr`, `rlx-qwen3-asr`,
`rlx-funasr`, `rlx-diarize`, `rlx-aec`.

TTS / speech: `rlx-orpheus`, `rlx-kyutai-tts`, `rlx-pocket-tts`,
`rlx-inflect-nano`, `rlx-tiny-tts`, `rlx-vibevoice`, `rlx-moshi`.

Vision / VLM: `rlx-bioclip2`, `rlx-florence2`, `rlx-grounding-dino`.

LM: `rlx-eagle3`.

### Notable changes

- **Qwen3.6-27B-MTP-GGUF** (`qwen35` arch, `unsloth/Qwen3.6-27B-MTP-GGUF`) text
  generation is now coherent and matches llama.cpp. Fixed two GatedDeltaNet bugs
  in `rlx-qwen35`: (1) the decay gate applied a spurious `-exp()` to `ssm_a`,
  which the GGUF already stores as `-exp(A_log)` — collapsing the recurrent
  state; (2) the GQA q/k head expansion (16→48) used *interleave* instead of
  *tile*, flipping the sign of every middle head's output. Also fixed the Metal
  Q3_K dequant (`dequant_gguf.msl` was dropping 8 of 16 sub-block scales — fixes
  Q3_K for all models) and the qwen3vl vision `mmproj` (CLIP merger) loader.
- Removed the `rlx-tensor-host` crate (the host-kernel shim that existed only
  to dodge a crates.io name clash with the framework's `rlx-tensor`). Its host
  kernels now live in `rlx_core::host_kernels` (math unchanged). `rlx-grounding-dino`
  additionally moved its compute (Swin / text encoder / enhancer / decoder) onto
  the `rlx` graph path, with `nn.rs` rebacked on `rlx_cpu::blas`.
- `scripts/publish.sh` publish tiers regenerated from the workspace
  dependency graph to cover all publishable crates.

## 0.2.6 — RLX runtime alignment (2026-06-13)

Workspace and model runners now pin upstream **`rlx*`** **0.2.6** on crates.io
(`rlx-runtime`, `rlx-ir`, `rlx-flow`, …). Requires RLX **0.2.6** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### Model runners (dependency-only release)

Same Rust sources as **0.2.5**; `Cargo.toml` pins updated from `=0.2.5` to
`=0.2.6`:

- `rlx-neutts` 0.2.6
- `rlx-gemma` 0.2.6
- `rlx-minicpm5` 0.2.6
- `rlx-minimax` 0.2.6
- `rlx-nemotron` 0.2.6
- `rlx-models` 0.2.6 (facade; publish last)

Publish tiers 0–6 before the facade (`scripts/publish.sh --list`). After
`rlx-kittentts` **0.2.8** and the tier-5 runners above are on crates.io, Skill
can drop `[patch.crates-io]` path deps and use registry versions only.

### Also at 0.2.6+ in this workspace

- Full workspace `[workspace.package].version` = **0.2.6**
- `kitten_tts_mini_rlx` **0.2.7**, `rlx-kittentts` **0.2.8** (native RLX bundle path)
- `rlx-qwen3-tts`, `rlx-fft` at **0.2.7** where noted in `Cargo.toml`
