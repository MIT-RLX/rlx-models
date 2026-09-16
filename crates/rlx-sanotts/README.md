# rlx-sanotts

[sanoTTS](https://github.com/ampixa/sanoTTS) on RLX — a native Rust port of the
**Root-A student stack**, a ~1.5 M-parameter distillation of Piper VITS that
fits in under 4 MB and runs comfortably faster than realtime on a CPU.

Weights: [`ampixa/sanoTTS`](https://huggingface.co/ampixa/sanoTTS) on Hugging Face.

## Architecture

`text → duration → acoustic latent → waveform`, three tiny convolutional
students and no attention anywhere:

| Stage | Kind | Shape |
| --- | --- | --- |
| duration | `duration_conv` — embed + 3 residual conv blocks | ids → frames per token |
| acoustic | `token_context` — token stack, duration-expanded, then a frame stack | ids + durations → `[192, frames]` |
| decoder | `piperlite` — 3 × (transposed conv + 3-branch residual bank) | latent → `frames × 256` samples |

The decoder upsamples 8 × 8 × 4 = 256 samples per frame, so `amy` emits 22 050 Hz
audio from an 86 Hz latent.

## Usage

```bash
# One voice package directory (manifest.json + weights.fp16.bin + piper-phoneme-config.json)
huggingface-cli download ampixa/sanoTTS --include 'amy-en-1p46m/*' --local-dir voices

cargo run -p rlx-sanotts --release --bin rlx-sanotts -- \
    say "Hello from a two megabyte voice." --voice-dir voices/amy-en-1p46m -o hello.wav

# Inspect a package, or dump the phoneme ids a text produces
cargo run -p rlx-sanotts --bin rlx-sanotts -- info --voice-dir voices/amy-en-1p46m
cargo run -p rlx-sanotts --bin rlx-sanotts -- ids "Hello there" --voice-dir voices/amy-en-1p46m
```

```rust
let synth = rlx_sanotts::Synthesizer::load("voices/amy-en-1p46m")?;
synth.synthesize("Hello from a two megabyte voice.")?.write("hello.wav")?;
```

Feed Piper phoneme ids directly when you want to bypass the text frontend
entirely (`Synthesizer::synthesize_ids`, or `--ids 1,0,28,0,…` on the CLI).

## Backends

Two execution paths share one set of weights:

- **`Backend::Host`** (`--device host`) — the host-eager reference kernels. This
  is the transcription of upstream's pure-numpy forward passes and the thing the
  parity fixtures gate.
- **`Backend::Graph(device)`** (`--device cpu|metal|mlx|cuda|rocm|gpu|vulkan|ane`)
  — the frame-rate acoustic stack and the entire decoder compiled into one
  rlx-ir graph. That is >95 % of the arithmetic; the duration model and the
  token-rate stack stay on the host, where they are microseconds of work at 64
  channels and where the data-dependent duration expansion has to happen anyway.

rlx-ir shapes are static, so a graph is only valid for the length it was built
for. Rather than compile one per utterance, graphs are built at a bucketed
*capacity* (default 32 frames, `RLX_SANOTTS_BUCKET`) and shorter inputs are
zero-padded up to it, then cached per `(device, capacity)` on the `Synthesizer`
and on disk (`RLX_SANOTTS_AOT`).

Zero-padding alone would be wrong: the first conv's bias turns the padded tail
into a nonzero constant and each later conv drags that leftward into real
frames. So every conv's bias add is followed by a multiply with a validity mask
that pins the tail back to zero — and every other op in the graph maps 0 to 0
(`leaky_relu`, `silu`, `tanh`, scalar multiplies, residual adds of masked
tensors, the transposed conv's zero-insertion), so that is sufficient. What
survives is reduction-order noise: on Metal/MLX/ANE the padded and unpadded runs
are bit-identical, on CPU the waveform differs by 1e-7, and the region next to
the padding — where a leaking mask would show first — is exactly zero
everywhere. `bucketing_is_exact` gates it.

Over 20 varied sentences (cold AOT cache, one process per sentence):

| `RLX_SANOTTS_BUCKET` | distinct graphs compiled | total |
| --- | --- | --- |
| 1 (off) | 20 | 13.45 s |
| 32 (default) | 14 | 12.34 s |
| 64 | 9 | 11.16 s |

Bigger buckets trade wasted compute per utterance (up to `bucket - 1` frames)
for a higher cache hit rate; a long-lived process reaches zero compiles either
way.

Measured on an M-series Mac, `amy-en-1p46m`, 2.40 s of audio, warm AOT cache:

| Backend | Realtime factor |
| --- | --- |
| `mlx` | 21.0× |
| `cpu` (graph) | 15.9× |
| `host` (eager) | 6.5× |
| `metal` | 2.4× — dominated by per-process pipeline-state compilation |

## Validation

`tests/reference_parity.rs` gates the port against upstream's pure-numpy
reference (`pypkg/sanotts/models.py`) on `amy-en-1p46m`:

- **durations match exactly** — they are integer frame counts, and one off-by-one
  desynchronizes every later frame. (numpy rounds half-to-even, unlike Rust's
  `f32::round`; the port matches numpy.)
- **latent and waveform** agree to float32 rounding noise, and the same fixtures
  gate every available graph device:

| Device | latent max abs | waveform max abs | correlation |
| --- | --- | --- | --- |
| Cpu | 1.5e-5 | 1.1e-6 | 1.000000000 |
| Metal | 1.7e-5 | 1.2e-6 | 1.000000000 |
| Mlx | 1.3e-5 | 2.3e-6 | 1.000000000 |
| Ane (CoreML) | 1.2e-5 | 2.6e-6 | 1.000000000 |

End to end from text, all backends agree within ±1 int16 LSB.

Also gated: the SHA-256 guard rejects a corrupted blob; bucketed graphs match
exact-length graphs; and the decoder post-filter — which no shipped voice has —
is checked by grafting a synthetic 2-layer post-filter onto the real decoder and
comparing the graph lowering against the host one on every device (with an
assertion that the synthetic filter actually changes the waveform, so the test
cannot pass against a no-op).

The voice package is not vendored (2.9 MB of weights). Point
`RLX_SANOTTS_VOICE_DIR` at a local copy or drop it in
`~/.cache/sanotts/amy-en-1p46m`; the tests skip when it is absent.

```bash
cargo test -p rlx-sanotts                      # host path
cargo test -p rlx-sanotts --features metal,mlx # + every available graph device
```

## Text frontend

sanoTTS inherits Piper's frontend: espeak-ng IPA with stress, punctuation
preserved, NFD-decomposed to individual codepoints, looked up one codepoint at a
time in the voice's `phoneme_id_map`, and framed as `^ _ (p _)* $`.

> Note the leading pad. sanoTTS emits `[BOS, PAD]` before the first phoneme,
> one pad more than `rlx-piper`'s framing. The models were trained on this exact
> layout, so the two are not interchangeable.

G2P is the pure-Rust [`espeak-ng`](https://crates.io/crates/espeak-ng) crate
(`text_to_phonemes_phonemizer`, which is the `phonemizer` package's
preserve-punctuation + flatten-clauses + keep-stress mode).

### espeak-ng version requirement

Two bugs in the pure-Rust espeak-ng affected this crate. **Both are fixed and
released** — the floor is now **0.2.0**, and the `[patch.crates-io]` this section
used to require is gone.

1. **`dictrules` (fixed in 0.1.3).** Before it, a voice's `dictrules` directive
   was never read, so `en-us` fell back to the base `en` rules — `fɹɒm` instead
   of `fɹʌm`. This is why the dependency floor is 0.1.3.
2. **en-US phoneme table (fixed in 0.2.0).** English voices share `en_dict`
   but not a phoneme table: `en` is en-GB, `en-us` is its own. The curated
   `EN_IPA_OVERRIDES`, derived from `espeak-ng -v en -q --ipa`, took precedence
   over the *active* table's own `i_IPA_NAME` phondata, so every American voice
   spoke RP — `həlˈəʊ` for "hello", non-rhotic `ˈəʊvə` for "over", `ɒ` for the
   LOT vowel. (Once 0.1.3 got `dictrules` right, `en-us` and `en-gb` diverged on
   dictionary lookups while still rendering *identical* IPA — which is how the
   phoneme table, rather than the rules, was identified as the cause.)

   Upstream fix: the active table's `i_IPA_NAME` is now authoritative and the
   overrides are a fallback, with en-GB's own GOAT-vowel entries (codes 144/145)
   scoped to the `en` table. en-GB output is unchanged — for every override
   code the `en` table's phondata is either identical to the override or absent.

Both land in **espeak-ng 0.2.0**, which is what this crate now requires. No
local patch is needed; a stale `[patch.crates-io] espeak-ng` entry pointing at a
sibling checkout will simply be reported as unused, because a patch only applies
when its version satisfies the requirement.

`g2p_produces_american_vowels` fails if the regression returns: it checks that
G2P for "hello over" is byte-identical to the ids for the reference IPA
`həlˈoʊ ˈoʊvɚ`, and different from the RP rendering.

### Bypassing G2P

Independently of any of that, both entry points take a pre-computed phoneme
stream — useful when you already have Piper's own ids, or a hand-written
pronunciation:

```bash
rlx-sanotts say x --phonemes "həlˈoʊ ˈoʊvɚ" --voice-dir voices/amy-en-1p46m -o out.wav
rlx-sanotts say x --ids 1,0,20,0,59,0,… --voice-dir voices/amy-en-1p46m -o out.wav
```

```rust
synth.synthesize_phonemes("həlˈoʊ ˈoʊvɚ", synth.default_length_scale())?;
synth.synthesize_ids(&ids, synth.default_length_scale())?;
```

## Voices

The Hugging Face repo ships seven packages across three languages. All seven
load (SHA-256 verified) and synthesize end to end here. `kristin` declares a
bare `en` espeak voice, which newer espeak-ng no longer exposes as a primary
voice; the frontend falls back to a regional variant rather than failing.

| Package | Voice | Params | Rate scale |
| --- | --- | --- | --- |
| `amy-en-1p1m` | `en_US-amy-medium` | 1 084 972 | 1.08 |
| `amy-en-1p46m` | `en_US-amy-medium` | 1 454 284 | 1.08 |
| `amy-en-1p8m` | `en_US-amy-medium` | 1 834 380 | 1.08 |
| `hfc-en-1p8m` | `en_US-hfc_female-medium` | 1 834 380 | 1.08 |
| `kristin-en-1p4m` | `en_US-kristin-medium` | 1 396 151 | 1.08 |
| `id-newstts-1p46m` | `id_ID-news_tts-medium` | 1 562 124 | 1.16 |
| `vi-vais1000-1p46m` | `vi_VN-vais1000-medium` | 1 565 484 | 1.16 |

```bash
just fetch-sanotts amy-en-1p46m
just sanotts "Hello from a two megabyte voice."
just test-sanotts-parity

# Non-English needs the full espeak dictionary set
cargo run -p rlx-sanotts --release --no-default-features \
    --features rlx-graph,espeak-all-languages --bin rlx-sanotts -- \
    say "Selamat pagi, apa kabar hari ini?" --voice-dir voices/id-newstts-1p46m -o id.wav
```

## Voice packages

Format `roota.raw-fp16.v1`: a `manifest.json` addressing tensors by
`offset_bytes`/`nbytes` into a flat little-endian fp16 blob, plus the Piper
phoneme config. The loader verifies the blob's declared size and SHA-256, so a
truncated or swapped file fails loudly instead of decoding into noise.

Only the subgraphs the shipped packages actually use are implemented
(`duration_conv` / `token_context` / `piperlite`, no output adapter, no
`pre_tanh_repair`); anything else errors rather than guessing a tensor layout.
The decoder post-filter is implemented on both paths even though no shipped
voice has one; the graph lowering is gated against the host one with synthetic
weights.

## Licence

GPL-3.0-only, matching both this workspace and upstream sanoTTS (which is GPLv3
because espeak-ng is).
