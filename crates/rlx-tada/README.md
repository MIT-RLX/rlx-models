# rlx-tada — HumeAI TADA

Native [rlx](https://github.com/eugenehp/rlx) port of **TADA** (Text-Acoustic
Dual Alignment), HumeAI's zero-shot voice cloner. No Python, no ONNX, no torch —
every stage compiles to rlx HIR and runs on any rlx backend.

## What makes TADA different

Most neural TTS emits discrete codec tokens and lets a separate duration model
decide timing. TADA does neither.

A forced aligner assigns **every text token exactly one 50 Hz frame**, so text
and audio ride the same autoregressive stream 1:1. At each step the backbone's
hidden state conditions a small DiT-style head, which runs a flow-matching ODE
to sample the token's **acoustic latent and its duration jointly** — as one
528-wide vector: 512 continuous acoustic dimensions concatenated with two
Gray-coded frame gaps. Nothing is quantized, and prosody is decided by the same
sample that decides timbre.

```text
  reference wav ─┬─► aligner (wav2vec2-CTC) ─► token↔frame assignment ─┐
                 └─► codec encoder ──────────► 50 Hz latents ──────────┤
                                                                       ▼
  target text ──► Llama-3.2 backbone ──► hidden ──► flow-matching head ──►
                       ▲                                    │
                       └──────── acoustic feedback ◄─────────┘
                                                             │
                                    latents + durations ─────┴─► codec
                                                       decoder ─► 24 kHz wav
```

## Layout

```text
<root>/
  tada-1b/config.json              HumeAI/tada-1b  (or tada-3b-ml/)
  tada-1b/model.safetensors        backbone + diffusion head (see below)
  tada-codec/encoder/model.safetensors
  tada-codec/decoder/model.safetensors
  tada-codec/aligner/model.safetensors        (or aligner-<lang>/)
  tokenizer/tokenizer.json         any ungated Llama-3.2 tokenizer mirror
```

### The `_decoder.*` trap

`tada-1b/model.safetensors` carries a *complete* codec decoder under a
`_decoder.` prefix. **It is not the decoder the model uses.** 195 of its 201
tensors differ from `HumeAI/tada-codec/decoder`, by up to 1.19. Upstream never
touches it: `TadaForCausalLM.from_pretrained` fetches the codec decoder from
`HumeAI/tada-codec` separately, and the bundled copy falls out of
`load_state_dict` as unexpected keys — silently.

Loading it instead is a perfect trap. The acoustic latents are still correct to
1e-5, the durations are still sensible, the waveform still has speech-like
energy — and Whisper transcribes the result as a single syllable. This port
therefore refuses to fall back to it and errors with a pointer to
`tada-codec/decoder` instead.

Upstream hard-codes the gated `meta-llama/Llama-3.2-1B` as its tokenizer source;
any mirror of the same vocabulary works (`unsloth/Llama-3.2-1B`). The special
token ids are resolved by name, so a mismatched vocabulary fails loudly instead
of silently masking the wrong prompt positions.

## Use

```bash
# Encode a reference speaker once — this is the expensive half.
rlx-tada prompt --weights $TADA --wav ref.wav \
  --text "What the reference audio says." --out morgan.tadaprompt

# Then speak, as often as you like.
rlx-tada speak --weights $TADA --prompt morgan.tadaprompt \
  --text "Deploy complete." --out hello.wav --device metal

rlx-tada info --prompt morgan.tadaprompt
```

Several clips of the same speaker condition better than one — repeat `--wav`
with a matching `--text`. Each clip is cleaned and validated on its own, then
concatenated with the transcripts joined in order:

```bash
rlx-tada prompt --weights $TADA \
  --wav take1.wav --text "First thing they said." \
  --wav take2.wav --text "Second thing they said." \
  --out morgan.tadaprompt
```

### Reference-audio hygiene

Reference clips are cleaned before encoding — DC offset removed, edge silence
trimmed, peak capped — and rejected if they are shorter than 2 s, longer than
30 s, or effectively silent. This lives in `rlx_core::voice_clone` and is shared
with the other cloning crates. It matters: on a realistic bad upload (DC offset,
recorded hot, 1.5 s of room tone at each end) it lifts speaker-cosine fidelity
from 0.642 mean / 0.480 worst to 0.726 / 0.653, measured over three utterances.
On an already-clean clip it costs nothing outside run-to-run spread.

`--raw-reference` skips both steps and encodes the clip exactly as supplied —
use it to reproduce a known-good prompt byte-for-byte, not for real recordings.

Two things are also checked and warned about, because both predict a poor clone
and neither is visible any other way:

* **A band-limited reference.** The codec is full-band 24 kHz, so a narrowband
  recording encodes a muffled voice and the clone inherits it. Measured: a studio
  clip carries 10.6% of its energy above 6 kHz and clones at 0.948 speaker
  cosine; a 1961 archival excerpt carries 0.195% and clones at 0.862.
* **A transcript that does not match the audio.** The aligner is a *forced*
  aligner — it places every token somewhere regardless — so the only symptom is
  the alignment score (`rlx-tada info` prints it): -0.34 for a correct
  transcript, -4.66 with five words of preamble that are not in the recording,
  -16.12 for an unrelated sentence.

On a clean reference TADA clones at **0.948** speaker cosine, slightly ahead of
ChatterBox's 0.939 on the same clip; on a band-limited one it is the less robust
of the two (0.862 vs 0.926). Upstream's solver defaults are kept: raising `--cfg`
to 3.0 helps the archival case and hurts the studio one, so it is left at 1.6.

Because TADA is autoregressive it can miss its stop condition and continue with
hallucinated speech. Output is checked for the `[speech][long silence][speech]`
signature and cut at the boundary; `--keep-runaway` disables that.

### Does it actually sound like the speaker?

```bash
cargo run -p rlx-wespeaker --release --features native,onnx \
  --example clone_fidelity -- --ort ref.wav cloned.wav
```

Above 0.7 is "same speaker" on the VoxCeleb convention. TADA scores **0.85 mean**
against the JFK reference clip.

`--ort` (ONNX Runtime, reference only) is worth using for measurement: the
native graph bakes a fixed 148-frame window, so it only ever sees the first
~1.5 s of a clip and its estimates are noticeably noisier. Both paths agree on
the direction of every comparison.

Non-English references need the matching aligner from `HumeAI/tada-codec`
(`--language de`, `es`, `fr`, `it`, `ja`, `pl`, `pt`, `ar`, `ch`). Generation
itself is language-agnostic; the alignment is baked into the prompt, so a prompt
built with the wrong aligner is wrong for every utterance made from it.

Library entry points: [`TadaModel::open`](src/model.rs) →
[`Synthesizer::synthesize`](src/synth.rs), and
[`PromptBuilder::build`](src/prompt_builder.rs).

## What was reused, what was written

| Piece | Source |
|---|---|
| Codec conv stacks (`WavEncoder`, `DACDecoder`) | `rlx-dac` graph builders — TADA's are DAC's verbatim |
| Llama-3.2 decoder, KV cache, RoPE | `rlx-llama32`'s flow, entered at `inputs_embeds` |
| `LocalAttentionEncoder` (both codec halves) | written here |
| `VibeVoiceDiffusionHead` + ODE solver | written here |
| wav2vec2-large CTC aligner | written here |
| Alignment DP, Gray coding, text normalization, resampling | written here |

The LM head is never built. For text-to-speech the token sequence is known up
front, so only hidden states matter — which drops a 128 k-wide matmul from every
step and means the 262 M-parameter embedding table never enters the graph.

The whole flow-matching solve — every Euler step, both guidance branches, the
blend — is emitted as **one** graph. Timesteps and guidance scales come from a
fixed schedule rather than from data, so they fold into constants and the ten
head evaluations become a single `run` per token.

## Performance

Measured on an Apple Mac mini, `tada-1b`, default 10 ODE steps and guidance on
(so every step runs two backbone branches).

| | before | after |
|---|---|---|
| 2.7 s of audio, CPU | 16.2 s — RTF 0.17× | **8.2 s — RTF 0.33×** |
| 11.1 s of audio, CPU | — | 18.5 s — RTF 0.60× |
| 11.1 s of audio, MLX | — | **10.1 s — RTF 1.10×** |
| peak RSS | 18.7 GB | **10.0 GB** |
| resident after load | 10.3 GB | 0.26 GB |
| voice prompt build | 1.93 s / 6.9 GB | **1.52 s / 5.1 GB** |

`RLX_TADA_PROFILE=1` prints a per-stage breakdown; `RLX_TADA_TRACE=1` prints the
predicted frame gaps, which is how a runaway duration shows itself before the
waveform gets long enough to notice.

### What actually cost the time

Profiling first, and the answer was not where it looked. Of the original 16 s,
about 9 s was **transposing weights**: checkpoints store `[out, in]`, every
matmul here wants `[in, out]`, and the obvious nested loop writes down a column
while reading along a row, so every store lands in its own cache line. On a 1 B
checkpoint that ran at ~1 GB/s. Replacing it with a cache-blocked, rayon-parallel
transpose (`rlx_core::weight_map::transpose_2d`, shared by every model crate)
took one backbone graph build from **3.88 s → 0.54 s** and the diffusion head
from **1.66 s → 0.31 s**.

The rest:

* **Nothing is materialized at load.** The backbone, head, codec, aligner and
  token embedding all read from the checkpoint on demand. Load went from 1.25 s
  and 10.3 GB resident to 0.17 s and 0.26 GB, and the old code additionally
  *cloned* its 3.9 GB weight map for every graph it built. Building a voice
  prompt — which runs the 447 M-parameter aligner — went from 1.93 s / 6.9 GB to
  **1.52 s / 5.1 GB**, and produces bit-identical output.
* **`pread`, not `mmap`.** Reading a 3.9 GB checkpoint through a mapping makes
  every page resident, and those pages are charged to the process at exactly the
  moment a multi-gigabyte arena is being allocated — ~2.6 GB of the peak, which
  `madvise(MADV_DONTNEED)` did not reclaim on macOS. Positional reads are also
  slightly faster here (0.55 s → 0.32 s per pass).
* **AOT LIR cache** for the backbone graphs, keyed by checkpoint identity plus
  shape, so lowering drops to ~1.7 ms.
* **Shape buckets** — prefill rounds to 32 positions and the KV cache to 64, so
  one compiled graph serves a range of utterances. Prefill padding sits past the
  end of a causal mask and the KV tail is already masked, so neither changes the
  result.
* **Arenas never overlap.** The solver is compiled lazily inside the loop rather
  than up front, and the backbone graphs are released before the codec runs.
* The token embedding (128 256 × 2048) is read one row at a time — an utterance
  touches a few dozen.

### F16 — measured, does not apply

Half precision looks like the obvious answer to a 10 GB f32 footprint, so it was
wired through every compiled graph and benchmarked. It does not work here, for a
reason worth writing down.

`CompileOptions::precision` is only honoured natively by the **CoreML** backend.
CPU, Metal, MLX and wgpu all route the compiled graph through
`cpu_low_precision::prepare_f32_exec_graph`, whose `needs_f32_exec` promotes the
whole graph back to F32 if *any* node other than `Param` / `Input` / `Constant`
is F16. And `PrecisionPolicy::AutoMixed` — the tuned default — deliberately
leaves matmul and data movement at F32, because F16 there diverged on other
models. Between the two, there is nothing left to save.

Measured on `tada-1b`, 2.7 s of audio, against the f32 baseline of 8.10 s /
10.25 GB (CPU) and 5.40 s / 12.32 GB (MLX):

| setting | CPU | MLX |
|---|---|---|
| `AutoMixed` | 8.15 s / 10.36 GB — **identical output** | 5.40 s / 12.32 GB — identical |
| `AlwaysF16` | 11.09 s / 9.14 GB — **0.24 s of silence** | 5.51 s / 12.32 GB — identical |

`AutoMixed` is a complete no-op on both. Three distinct LIR blobs really were
compiled (the cache keys differ and so do the bytes), and the graphs are the
same size — the dtype rewrite is recorded in the LIR and then undone at
execution. `AlwaysF16` does save 1.1 GB on CPU, because `Param` nodes are
exempt from the promotion and stay half — but the same change breaks the model:
the frame-gap field is Gray-coded and thresholded, so a little lost precision
turns durations into nonsense and the utterance collapses to a quarter second of
silence. On MLX it is simply promoted away.

BF16 is not an alternative: `PrecisionPolicy` has no BF16 variant, and
`rlx-core`'s own flow bridge already maps `Bf16 → Precision::F16` with the
comment "closest supported runtime precision today".

The knob was removed rather than shipped — a setting whose two values are
"nothing happens" and "silently emits silence" is a hazard. CoreML is the one
backend that would honour it, and is where to look next.

### What else did not work

Routing the **flow-matching solver** through the AOT cache made it *worse*: 176 ms
→ 298 ms per token and peak RSS 12.7 → 20.5 GB, because that pipeline's fusion
choices differ from `Session::compile`'s and for this graph they are wrong. It
would not have helped anyway — 1.66 s of the solver's 1.74 s was interning
parameters, not lowering. It is deliberately left on the direct path.

### What is left

Roughly 9 GB is the structural floor for synthesis: a graph build holds the f32
weights and the arena at the same time (7.8 GB for the backbone) plus the
solver's arena, and f32 is the only precision this path binds. Measured peak is
10.0 GB, so there is not much left without a streaming param-binding API
upstream — the flow hands back a complete `HashMap` of parameters before the
graph exists to receive them.

Per token, the solve is ~176 ms against a measured ~80 GB/s of parameter
traffic — bandwidth-bound and near optimal for 10 ODE steps, so the real lever
there is `--steps`, which is already a flag. The default stays at upstream's 10.
The backbone step is ~98 ms.


## Backends

All five Apple-available backends produce the same audio. 2.7 s of speech from a
cached voice prompt, `tada-1b`, default settings:

| backend | time | RTF | peak RSS | 78-test suite | waveform vs CPU |
|---|---|---|---|---|---|
| MLX | 5.9 s | 0.46× | 12.3 GB | pass | cosine 1.000 |
| Metal | 6.9 s | 0.39× | 10.5 GB | pass | cosine 1.000 |
| CPU | 8.8 s | 0.31× | 10.1 GB | pass | — |
| Vulkan | 15.7 s | 0.17× | 10.5 GB | pass | cosine 1.000 |
| CoreML / ANE | 38.3 s | 0.07× | 19.5 GB | pass | cosine 1.000 |
| wgpu | 76.7 s | 0.04× | **7.9 GB** | pass | cosine 1.000 |
| CUDA | — | — | — | builds | **not run** — no rig available |
| ROCm | — | — | — | builds | **not run** — no rig available |

Every backend that could be exercised produces the *same waveform* as CPU, not
merely intelligible audio. On a longer utterance MLX reaches RTF 1.10×. CoreML
and wgpu are correct but slow — neither has been tuned for this graph, and wgpu
is paying for the fix below. wgpu has the lowest footprint, because parking
parameters in a second buffer is exactly what its arena split does.

CUDA and ROCm compile with their features enabled and the crate contains no
host-specific branches, but the rigs were offline, so they are listed as
untested rather than supported. The 1 B backbone's f32 arena is ~3.9 GB, which
is worth knowing before pointing this at a small-VRAM card.

Three ways to run it:

```bash
just test-tada <device>      # the 78-test suite on one backend
just test-tada-backends      # …on every backend this host has
just tada-matrix             # the shared cross-backend harness (what a rig runs)
```

`tests/backend_matrix.rs` additionally runs one prefill of the **real** backbone
against CPU when `RLX_TADA_WEIGHTS` is set — the component fixtures are a few
megabytes and cannot reach code paths that only a gigabyte-scale arena selects.
The model is registered in `scripts/matrix/registry.toml`, so a CUDA or ROCm host
picks it up with no edits.

### An upstream wgpu bug this found

wgpu produced fluent-length audio that transcribed to nothing, with frame gaps
of `[15, 15, 15, 15, 15, 14, …]` — near-constant, and **identical for different
input text**. The backbone was ignoring its input entirely.

Every component parity suite passed on wgpu, and so did `e2e_parity`; the fault
only appears at checkpoint scale. With `RLX_WGPU_DEBUG=1`:

```
[rlx-wgpu] device limits: max_storage_binding=4.000GiB
[rlx-wgpu split] weight_params=148 weight_buf=3.657GiB act_arena=2.388GiB
```

Past the 4 GiB storage-binding cap, wgpu parks parameters in a second buffer
reached through staged copies. Those stages are bump-allocated into a *wrapping*
scratch reserve, so many parameters share one destination — and the deferred
host mirror is keyed by destination, so deferring them collapses the whole
sequence to whichever copy ran last. Every other weight then reads stale bytes.

`crates/backends/rlx-wgpu/src/backend/run.rs` had lost the guard for this:

```rust
let defer = !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");   // was: !src_is_weight && …
```

which had quietly turned that flag from a performance knob into a correctness
switch. Restored. The one-flag A/B (`RLX_WGPU_HOST_EAGER_H2D=1` fixing it) is
what identified the class before the code was read.

The blast radius is limited to wgpu runs whose arena crosses the bind cap —
which were producing wrong answers, not slow ones. It is also why wgpu is slow
in the table above: weight copies now round-trip eagerly, one device poll each.

## Validation

Every hand-derived component is pinned against a real forward pass from the
upstream module (`hume-tada` 0.1.9 on torch 2.11), not against a plausibility
check:

| Test | Reference | Agreement |
|---|---|---|
| `align_parity` | `aligner._align_text_tokens` | **exact** (discrete assignment) |
| `resample_parity` | `torchaudio.functional.resample` | < 1e-4 |
| `local_attn_parity` | `encoder.LocalAttentionEncoder` | < 1e-3 |
| `head_parity` | `nn.vibevoice.VibeVoiceDiffusionHead` | < 1e-4 |
| `codec_conv_parity` | `WavEncoder` / `DACDecoder` | < 1e-4 |
| `aligner_parity` | `transformers.Wav2Vec2ForCTC` | < 2e-3, exact per-frame argmax |
| `e2e_parity` | `TadaForCausalLM.generate` | < 5e-3 |

On the real `tada-1b` + `tada-codec` checkpoints, every stage was compared
against upstream running on the same weights:

| Stage | Agreement with upstream |
|---|---|
| Voice prompt: token ids, frame positions | **identical** |
| Voice prompt: 26 × 512 latents | 1.8e-5 (rms 0.84) |
| Plan: masked ids, gaps, prefill length | **identical** |
| Prefill embeddings (32 × 2048) | **bit-exact** |
| Prefill hidden state | 7e-6 |
| Decode hidden, while inputs still match | 1e-7 |
| Flow-matching solve, matched conditioning | 4e-5 |

Free-running trajectories at `noise_temperature = 0` **do** diverge after the
first model-predicted latent, and that is not a defect. Measured directly: at
that step a 1e-6 nudge to the conditioning moves the solve output by 1.03 — a
gain of ~10⁶. The ODE starts at the origin there, which is exactly where a field
trained from `N(0, I)` is least determined, so the trajectory sits on a
bifurcation. Neighbouring steps have gains of 153× and 420×. Trajectory-level
comparison at zero temperature is therefore not a valid parity test; stage-level
comparison with matched inputs is.

At the real default temperature the end-to-end check is intelligibility, and it
passes on CPU, Metal and MLX alike — `"The quick brown fox jumps over the lazy
dog."`, `"She sells sea shells by the sea shore."` and `"Deploy complete. All
tests passed."` all come back from Whisper verbatim.

`e2e_parity` runs a miniature TADA end to end with `noise_temperature = 0`, so
the ODE starts from the zero vector in both implementations and the outputs
compare value by value. The audio is meaningless; the wiring is the point —
prefill length, the five-step acoustic shift, the transition hand-off, prompt
text masking, and where the generated span starts. It needs a Llama-3.2
`tokenizer.json`; set `RLX_TADA_TOKENIZER` to a directory holding one, otherwise
it skips.

That test is also what caught the one real bug in the port: `decode_gray_code_to_time`
does **not** clamp each slot to `{0, 1}`, so a flow output that lands outside
`[-1, 1]` contributes a multi-valued term to the Gray integer rather than a bit.
Clamping "fixed" nothing visible and changed durations.

## Deliberate divergences from upstream

* **Predicted frame gaps are bounded** to `num_time_classes - 1` before they
  index a duration embedding. Upstream raises an `IndexError` there instead.
* **Noise is a seeded xorshift + Box–Muller stream**, not torch's Philox. The
  noise is the *start* of an ODE that contracts toward the conditional
  distribution, so any correctly-scaled Gaussian is a valid draw — but the same
  seed must give the same audio, and it does.
* **`--latent-noise 0`** is available to build a deterministic voice prompt.
  Upstream keeps the codec encoder's stochastic bottleneck on at inference
  (`std = 0.5`), which is the default here too; zero is off-distribution but
  gives an artifact you can diff.

## Backend note

The codec's FFN uses exact (erf) GELU, not the tanh approximation. Metal has a
known defect executing `FusedMatMulBiasAct{erf-Gelu}`; if codec output on Metal
disagrees with CPU, that fusion is the first thing to check.

## Upstream fix this port required

`rlx-llama32`'s `Llama32Flow::hidden_only()` was honored in prefill but silently
ignored in decode, which always built — and, with tied embeddings, always
demanded — `model.embed_tokens.weight`. That is a hard load failure for any
checkpoint whose embedding table is held outside the decoder's weight map.
Fixed in `crates/rlx-llama32/src/flow.rs`.
