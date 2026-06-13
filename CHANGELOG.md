# Changelog

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
