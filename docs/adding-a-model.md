# Adding a model

The repo has ~200 model crates and the same ten mistakes keep being available.
This is the order of operations that works here, with the reasons attached —
every step below exists because skipping it cost someone a debugging session.

The single most important point first: **most models do not need a new crate.**
Of four checkpoints added in one recent session, two needed no code at all, one
needed a module, and one needed a crate. Finding out which takes ten minutes.

## 1. Identify the architecture before writing anything

```sh
curl -s https://huggingface.co/api/models/<org>/<repo> | python3 -m json.tool | head -40
curl -s https://huggingface.co/<org>/<repo>/raw/main/config.json
```

What matters: `architectures`, `model_type`, and whether there is a nested
`text_config` / `vision_config` / `audio_config`. For a GGUF, the equivalent is
`general.architecture` in the header.

Then grep for it:

```sh
grep -rln "<model_type>" crates/*/src/          # already handled?
grep -rn "\"<gguf-arch>\"" crates/*/src/config.rs
```

Three outcomes, in order of likelihood:

| finding | what to do |
|---|---|
| the `model_type` already appears in a crate | nothing to build — go to step 5 and **prove** it with a fixture test |
| a sibling generation is handled (`qwen35` vs `qwen3_5`) | usually a config-reader delta, sometimes nothing |
| nothing matches | a new crate, step 4 |

`Cloudflare/clef` is `model_type: qwen3_5` and parsed on the first try, because
`Qwen35Config::from_hf_config_value` never gated on `model_type` — the hybrid
GatedDeltaNet/full-attention schedule, the output gate and the partial RoPE were
all already described. `jialinyyzz/humanizer` is `gemma4_unified`, which
`rlx-gemma` maps to `GemmaArch::Gemma4` along with the three flagship options it
turns on. Both were finished by writing a test, not by writing a port.

## 2. Get the tensor manifest for free

Never download a checkpoint to find out what is in it. Both formats put their
index in the header:

```python
# safetensors: first 8 bytes are the header length, then JSON
import json, struct, urllib.request
def rng(url, a, b):
    rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(rq).read()
n = struct.unpack("<Q", rng(url, 0, 7))[0]
hdr = json.loads(rng(url, 8, 8 + n - 1))      # name -> {shape, dtype, offsets}
```

A GGUF's tensor index is likewise header data, and a sharded safetensors repo
publishes `model.safetensors.index.json` directly. Tens of kilobytes gives you
every tensor's name, shape and dtype — which is the actual contract.

Use it to pin the contract **in both directions**: every tensor the emitter
reads must exist, and every tensor in the checkpoint must be read. See
`crates/rlx-qwen4exp/tests/tensor_manifest.rs`, which checks 1224 real tensor
names and orientations across 48 blocks with no weights present. That test is
what caught `indexer.q_proj` being *split* by the converter when HF ships it as
one tensor.

GGUF dimension order is the classic trap: **innermost first**. A projection
written `[in, out]` in the file arrives as `[out, in]`, and an expert bank's
expert count is the **last** dimension — `[2560, 640, 8]` is 8 experts of a
2560×640 projection, not 2560 of anything.

## 3. Read the reference, all of it

- HF: `transformers/src/transformers/models/<arch>/modeling_<arch>.py`, or
  `modular_<arch>.py` when it exists (it is the readable one). Raw URL:
  `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/<arch>/modeling_<arch>.py`
- llama.cpp: `src/models/<arch>.cpp` **and** `conversion/<arch>.py`. The
  converter is where tensor renames, splits and pre-negations live, and it is
  now per-arch rather than one giant `convert_hf_to_gguf.py`.

When the three sources agree you can port with confidence. When they disagree,
the converter usually wins for a GGUF and `modeling_*.py` for safetensors.

Small, sharp detail worth the lookup every time: a class like
`Gemma4ClippableLinear` turns out to be `clamp(x, in_min, in_max)` → bias-free
matmul → `clamp(y, out_min, out_max)`, with the bounds initialized to `±inf`.
Guessing that from the four extra scalars per linear in the manifest would have
been plausible and wrong in the details.

## 4. A new crate

Copy the shape of the closest sibling rather than inventing one.
`crates/rlx-qwen4exp` is a current, complete example: one module per
architectural concern (`hc`, `gdn`, `attention`, `indexer`, `moe`, `ple`,
`rope`), a `config` that validates, a `flow` that assembles, a `decode` for the
incremental path, a `runner` that implements `rlx_runtime::LmRunner`, and a
`synth` module whose fake weights every test uses.

If the crate exposes a `*RunnerBuilder`, take the shared half rather than
retyping it: `rlx_core::lm_builder_setters!(MyConfig)` inside the `impl` gives
the `weights` / `config` / `config_value` / `device` / `max_seq` /
`max_memory_gb` / `stream` setters over a
`common: LmBuilderCommon<MyConfig>` field, and `LmBuilderCommon` carries the
resolution helpers (`weights_path`, `device_or` / `device_or_auto`,
`max_seq_or`, `check_memory_cap`). `sample`, `format` and `packed_weights`
stay yours — the first two because their types live in `rlx-qwen3` and
`rlx-cli` (naming them in `rlx-models-core` would be a cycle), the third
because families genuinely disagree about what it means. Skip the macro
entirely if your builder lacks `max_memory_gb` or `stream`: it would add
setters your `build()` ignores, which is how `rlx-qwen35` ended up keeping
its own.

Then wire it up:

- **register a constructor** so the generic path can build it:
  `rlx_core::lm_registry::register_lm_runner(LmRunnerRegistration { runner: "x",
  build: |path, opts| Ok(Box::new(MyRunner::open(path, opts.device)?)) })`.
  Without this your crate works only through its own builder —
  `rlx-qwen4exp` and `rlx-minimax` both shipped complete `LmRunner`
  implementations that `auto_runner` could not reach, and their own test suites
  passed the whole time;
- a `model_registry` entry, so the architecture **sniff** resolves to that
  runner name. Both halves are needed: the entry without a constructor gives a
  "no registered constructor" error, and a constructor without an entry is dead
  code (a test asserts both directions);
- `Cargo.toml` + the workspace `members` list;
- a row in [MODELS.md](../MODELS.md) — `scripts/check-models-catalog.py`
  enforces one row per publishable crate and will fail the build without it;
- a tier in `scripts/publish.sh` **after** every crate it depends on, dev-deps
  included (`check-publishable.py` counts them);
- if a test uses `include_str!` on a fixture under `tests/fixtures/*.json`, add
  a `!` negation to `.gitignore` — the blanket rule there will otherwise exclude
  a compile-time input and break the build for everyone else.

### Derive, never hardcode

Read facts about a checkpoint from the checkpoint. A constant like
`SUBSET_EXPERTS = 8` that *overrides* the config is how a test silently measures
a different model than the builder constructs — `scripts/qwen4exp_subset.py`
takes `--experts N`, so the only correct source is the bank's own shape
(`moe::gguf_expert_count`). The same applies to block sizes (probe
`rlx_gguf::bytes_for_public`, do not write `32`), the number of experts, the
layer schedule, and which block a test should use.

State config→dims **once**. `XDims::from_config(cfg, seq)` with the genuine
knobs explicit at the call site beats four copies in prefill, decode, MTP and
the tests, one of which will drift.

## 5. Tests that can fail

This is where ports are won. In order of value:

1. **Fixtures from the real checkpoint.** The published `config.json` verbatim
   and the real tensor manifest, both committed. A hand-written second copy of
   the contract is the one that silently becomes wrong.
2. **Equivalences between independently spelled paths.** Decode == prefill.
   Chunked == single-shot. Padded state == exactly-sized state. Paged experts ==
   resident experts. Native tower == another crate's implementation of the same
   tower. Each compares two different pieces of code that must agree, so neither
   can be quietly wrong alone.
3. **Verify every equivalence can fail.** Break the thing it is supposed to
   catch and watch it go red, then put it back. A test that passes under
   sabotage is measuring nothing, and this is not hypothetical: a one-position
   M-RoPE slip measures only 2.1e-3 relative, which sails through a 2e-3
   tolerance copied from a sibling test.
4. **Pick tolerances from both sides of the gap.** Measure the legitimate noise
   (different reduction order: ~5e-8) and measure the defect you are guarding
   against (lost cache row: ~5e-4), then sit between them. "Looks close enough"
   is how a real divergence ships.
5. **Prefer bit-exactness when the paths should be identical.** If a runner
   compiles the same graph with the same inputs as the loop it replaces, assert
   equality — any difference is a threading mistake, not arithmetic.

### Real weights without the download

Range-fetch a subset into a valid standalone file. `scripts/qwen4exp_subset.py`
pulls two of 48 blocks with 8 of each block's 512 experts into a 186 MB GGUF, by
exploiting the fact that a quantized expert bank's expert axis is outermost, so
each expert is a contiguous byte range. It also narrows the router so top-k
cannot select a slab that is not present. `just qwen4exp-real` then runs the
real-weight tests against it.

Checkpoints at or above 1 GB belong on `/Volumes/FOUR`, not in the repo.

## 6. The gates

```sh
just fmt
just lint                        # clippy + rustdoc, -D warnings, default features
just lint-features --changed     # the states `just lint` cannot see
just lint-docs                   # rustdoc links alone, if that is all you touched
python3 scripts/check-models-catalog.py
python3 scripts/check-publishable.py
```

`just lint` builds **default features only**. Code behind an optional feature is
compiled by nothing, which is how a field declared `#[cfg(feature = "x")]` and
read without the gate breaks the build for everyone who has not enabled `x`.
Run `lint-features` after touching any `cfg`, and check **both** states — gating
a use site is how the mirror-image `unused` error gets introduced.

Do not commit unless asked.

## 7. A worked example of steps 1–3

[qwen-image-2.1-port-notes.md](qwen-image-2.1-port-notes.md) is this process
applied to a checkpoint that has *not* been ported: what the header says, the
full tensor contract, the three findings that change the graph, and what the
file does not contain. Written before any code, which is the point.

## 8. Write down what is not done

A port's Status section is part of the port. Say which paths have never run,
which backends are untested, and which equivalences a test is blind to —
`crates/rlx-qwen4exp/tests/vit_parity.rs` records that it cannot distinguish
erf from tanh GELU (they agree to ~3e-4) and that both towers receive the same
RoPE tables, so it tests the rotation's *application* and not its construction.
Implying coverage you do not have is worse than having less.
