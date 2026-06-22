# Changelog

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
