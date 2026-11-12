# rlx-fish

Fish Audio text-to-speech and voice cloning on RLX, native Rust, and Audio8,
which shares S2's design:

| Model | Checkpoint | Architecture |
|-------|------------|--------------|
| **S2-Pro** | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) (9 GB bf16) | dual-AR: 36-layer Qwen3-4B-shaped *slow* backbone (qk-norm, GQA 32/8, interleaved RoPE) picks one semantic token per frame; a 4-layer *fast* transformer fills the frame's other nine codes |
| **Drama 3** | [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) (40 GB, block-FP8 experts), demo Space [`fishaudio/Drama-3`](https://huggingface.co/spaces/fishaudio/Drama-3) | 40-layer Qwen3.5-MoE hybrid (30 Gated DeltaNet + 10 gated-attention layers, 256 experts / 8 active + shared expert) and a 13-layer codebook *tower* that attends across frames |
| **Audio8-TTS-Realistic-Air** | [`Edge0/Audio8-TTS-Realistic-Air`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) (2.5 GB bf16 + f32 codec, Apache-2.0) | S2's dual-AR at 0.6 B: 24-layer slow stack (GQA 14/2, `wqkv` bias, no qk-norm) and a 4-layer fast stack; 23 languages, 2048-row context |

All three emit 10 codes per frame (21.5 frames/s) for the same codec,
`codec.pth`: a causal 44.1 kHz DAC whose RVQ sits behind a 4× downsample with
an 8-layer transformer on each side. S2 and Drama 3 ship one file; Audio8's
matches it tensor for tensor except for the post transformer, which is
narrower (GQA 16/8, FFN 1216).

## Run

```bash
just fetch-fish-s2        # → weights/tts/fish-s2-pro
just fish-s2 "Hello from Fish Audio on RLX." /tmp/s2.wav metal

just fetch-fish-drama3    # → weights/tts/fish-drama-3 (40 GB)
just fish-drama3 "Hello from Drama 3." /tmp/d3.wav metal

just fetch-audio8         # → weights/tts/audio8-realistic-air (2.5 GB)
just audio8 "Hello from Audio8 on RLX." /tmp/a8.wav metal
```

or directly:

```bash
cargo run -p rlx-fish --release --features metal -- audio8 \
  --model-dir weights/tts/audio8-realistic-air --device metal --text "…" --out /tmp/a8.wav \
  [--ref-wav ref.wav --ref-text "exactly what ref.wav says"]   # voice cloning
  [--batch 4]   # sentence by sentence, four at once (below)
cargo run -p rlx-fish --release --features metal -- s2 \
  --model-dir weights/tts/fish-s2-pro --device metal \
  --text "<|speaker:0|>Hi there. <|speaker:1|>Hello!" --out /tmp/s2.wav \
  [--ref-wav ref.wav --ref-text "what ref.wav says"]   # voice cloning
cargo run -p rlx-fish --release --features metal -- drama3 \
  --model-dir weights/tts/fish-drama-3 --device metal --text "…" --out /tmp/d3.wav
cargo run -p rlx-fish --release -- codec --model-dir weights/tts/fish-s2-pro \
  --wav in.wav --out roundtrip.wav
```

Sampling defaults are the Fish Audio SDK's (temperature 0.7, top-p 0.7,
top-k 30) with each model's repetition-aware resampling. Long text is chunked
the way each reference does: S2 by speaker turns (≤ 5 speakers, 200 bytes) in
a growing conversation; Drama 3 with the hosted server's splitter (300 bytes,
3 speakers, `[cues]` and `<tag>…</tag>` kept whole) and the first plus the
latest chunk as history. `RLX_FISH_VERBOSE=1` prints per-frame timings.

Audio8 takes its sampling defaults from the checkpoint's
`generation_config.json` (temperature 0.7, top-p 0.9, top-k 50, 512 frames;
the model card's examples use 0.8 / 0.95 / 50 / 1024). It speaks the whole text
in one pass, as its reference does, so an utterance has to fit what the
2048-row context leaves after the prompt (about 90 s with no reference voice).
Running into the frame limit prints a warning, since the speech may be cut off.
`rlx-export weights/tts/audio8-realistic-air -o audio8.rlxp` makes a package rlx
runs alone ([docs/exporting.md](../../docs/exporting.md)).

## How it runs

- **Codec** — two graphs (decode, encode) per 32-frame bucket. The codec is
  causal end to end, so frames padded after the real ones cannot change them.
  The RVQ lookup and nearest-neighbour search run on the host.
- **Weights** stay bf16 in the graphs (`RLX_WEIGHT_STORAGE=auto`, the
  default; `f32` widens). Backends widen in-kernel and multiply in f32, so the
  products equal the f32 path's at half the memory. On Metal, decode-sized
  (M ≤ 4) bf16 matmuls read the weight directly (`gemv_bf16w_splitk8`).
- **S2** — a prefill graph over the prompt, then one decode graph per frame
  whose KV cache (fixed 256-row buckets, unused slots masked) lives on the
  device: uploaded once, each step's new row copied in there
  (`RLX_DISABLE_GPU_KV=1` re-sends it from the host instead). A 4-layer fast
  graph runs 10 times per frame. The tied LM head is cut to the 4097 rows
  that can be sampled (semantic band + `<|im_end|>`).
- **Drama 3** — one graph per backbone layer (mixer, post norm, router logits,
  shared expert); the routed experts run on the host straight from the mapped
  FP8 shards. Decode rows multiply directly from the FP8 bytes (`e4m3 · scale`
  per 128×128 block); prefill batches dequantize each expert once and use
  BLAS, keeping an LRU of those (`--expert-cache`, bytes). 32 B expert
  parameters never enter a graph. DeltaNet layers carry their conv window and recurrent state;
  padded prefill rows get `beta = 0`, `g = 0` so they cannot advance it.

- **Audio8** — runs on [`S2Model`](src/s2.rs). `S2Config` carries what differs
  from S2: tensor names (`TensorLayout`), the slow `wqkv` bias, the codebook
  sum left unscaled, and the repetition-sampling settings from `config.json`.
  `FishCodec` reads each transformer's head counts and FFN width from its
  weights. `src/audio8.rs` is the one-pass driver.

## Parity

`scripts/fish_reference.py` runs the official PyTorch code (the `fish_speech`
package from the Drama-3 Space / fish-speech repo) and dumps fixtures. On the
dev box the sources and the fixtures used for the table below are kept in
`/Volumes/FOUR/ref/fish/{src,fixtures}` (`just test-fish /Volumes/FOUR/ref/fish/fixtures`).

```bash
python3 scripts/fish_reference.py codec  --src FISH_SRC --model-dir weights/tts/fish-s2-pro \
    --wav assets/jfk/jfk_voice_clone.wav --out /tmp/fish_ref
python3 scripts/fish_reference.py s2     --src FISH_SRC --model-dir weights/tts/fish-s2-pro --out /tmp/fish_ref --steps 8
python3 scripts/fish_reference.py s2     --src FISH_SRC --model-dir weights/tts/fish-s2-pro --out /tmp/fish_ref --layers 2 --no-eos
python3 scripts/fish_reference.py drama3 --src FISH_SRC --model-dir weights/tts/fish-drama-3 --out /tmp/fish_ref --layers 4 --no-eos --steps 4
RLX_FISH_DIR=$PWD/weights/tts/fish-s2-pro RLX_FISH_DRAMA3_DIR=$PWD/weights/tts/fish-drama-3 \
RLX_FISH_REF=/tmp/fish_ref cargo test -p rlx-fish --release -- --include-ignored
```

| Check (CPU; Metal, MLX and wgpu agree to the same order) | Result |
|---|---|
| codec decode, 24 random frames | max-abs 3.3e-6 |
| codec encode, 2 s of speech | latent max-abs 8.9e-5; 440/440 codes identical |
| S2-Pro full 36 layers, 8 greedy frames | hidden / band logits rel-L2 ≤ 4e-6; every frame identical; prompt tokens identical |
| Drama 3, first 4 layers + full tower, 4 greedy frames | hidden / main logits rel-L2 ≤ 2e-6, tower heads ≤ 5e-4; every frame identical; prompt tokens identical |

The full Drama 3 backbone cannot be run by the reference on a 64 GB machine
(it dequantizes the 32 GB of FP8 experts), hence the 4-layer cut (3 DeltaNet
+ 1 gated-attention layer — both mixer kinds). End-to-end output of both
models is checked with Whisper (`rlx-whisper`).

Audio8 has its own script and test, which run the code published with the
checkpoint: `ArkttsProcessor` for the prompts and `ArkttsModel` for greedy frames
(asserted equal to `generate()`). Use the card's versions, PyTorch 2.8.0 and
Transformers 4.57.3: on Transformers 5 `ArkttsProcessor.from_pretrained` raises
`TypeError` (`fix_mistral_regex` passed twice). `--src` holds the four `*.py`
files as a package named `arktts`, so `trust_remote_code` is not involved.

```bash
python3 scripts/audio8_reference.py codec --src SRC --model-dir weights/tts/audio8-realistic-air \
    --wav assets/jfk/jfk_voice_clone.wav --out /tmp/audio8_ref
python3 scripts/audio8_reference.py lm    --src SRC --model-dir weights/tts/audio8-realistic-air --out /tmp/audio8_ref
python3 scripts/audio8_reference.py lm    --src SRC --model-dir weights/tts/audio8-realistic-air --out /tmp/audio8_ref \
    --steps 12 --wav assets/jfk/jfk_voice_clone.wav \
    --ref-text "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
just test-audio8 /tmp/audio8_ref metal
```

| Audio8 check (full model, worst over CPU, Metal, MLX, wgpu) | Result |
|---|---|
| prompt rows vs `ArkttsProcessor`, with and without a reference voice | identical |
| 24 greedy frames, and 12 with a cloned voice | every token and code identical |
| slow hidden state / band logits / fast logits | rel-L2 ≤ 1.1e-5 / 1.1e-5 / 6.3e-5 (max-abs ≤ 8.2e-4) |
| codec decode, 40 random frames | max-abs 3.0e-6 |
| codec encode, 5.2 s of speech | latent max-abs 1.4e-4; 1120/1120 codes identical |
| what a sampled draw chooses among (396 pools: main token, repetition redraw, fast codes) vs the reference's logits processors, CPU | same candidates; probabilities within 2.1e-5 |
| `.rlxp` package vs the runner, CPU | 24 of 24 sampled frames, with and without a reference voice; codec cosine 1.0 |

The same suite on x86 Linux, in each backend's `RLX_FISH_DEVICE`:

| Backend (x86 Linux) | Result |
|---|---|
| CPU (two machines), wgpu (NVIDIA and AMD) | pass; hidden rel-L2 ≤ 1.1e-5, every frame identical |
| Vulkan on an NVIDIA GPU | pass; hidden rel-L2 ≤ 8.7e-6 |
| CUDA, default settings | pass; the comparisons run in strict f32, hidden rel-L2 ≤ 8.8e-6; greedy frames identical at the default TF32 too; 1120/1120 encoder codes |
| ROCm on an AMD RDNA3 integrated GPU | pass; hidden max-abs 1.4e-4 |
| Vulkan on that integrated GPU (RADV) | pass, hidden rel-L2 ≤ 1.1e-5; a full voice clone end to end with no GPU resets |

Three fixes got there, two of them upstream in `rlx`:

- **CUDA's TF32 and the codec encoder.** TF32 leaves the LM's greedy path
  alone (36 of 36 frames, 360 of 360 codes) but moves the encoder's latent by
  7e-3, enough to change 82 of 1120 codes of the reference clip, the prompt a
  cloned voice is built from. `FishCodec::encode_latent` now runs under
  `rlx_runtime::with_strict_f32`; the LM keeps TF32 (about 2× faster than
  strict f32 per frame).
- **ROCm, `cast_f32_to_half`.** The kernel that casts activations for a bf16
  or f16 weight included `cuda_fp16.h`, which hipRTC does not have, so every
  such matmul panicked on ROCm.
- **Vulkan submissions.** rlx-vulkan recorded each run of dispatches as one
  command buffer, so a long graph was one driver job, and amdgpu resets the
  ring after about 2 s. The codec decode is 17 M workgroups at 64 frames and
  35 M at 128; it now goes out in parts of at most 2^21.

The parity test's float comparisons run in strict f32. A backend's faster
default changes them by design, and it must not change the greedy frames,
which `greedy_frames_at_the_backends_default_precision` checks with no float
tolerance.

Tolerances sit between measured noise and a measured defect: hidden states
differ by 1.6e-4 at worst, while f32 RoPE tables in place of the reference's bf16
ones move them by 1.2e-2 to 1.9e-2 against a 2e-3 limit. Without weights,
`audio8::tests` pin the published config and all 226 tensors (read in both
directions), the processor's prompt, decode == prefill, and the plain codebook
sum.

Sampled tokens themselves cannot match the reference, since this crate's RNG
is not torch's, so sampling is compared up to the draw. Swapping S2's top-p and
temperature order moves a pool by 0.13 against a 2e-3 limit.

All 23 languages, one sampled sentence each on Metal, read back by Whisper
large-v3: mean character error 3.9%, 13 languages exact. What is left is the
model's reading (Indonesian and Malay say "cuaca" with a k), and quality varies
by language, as the model card says. Cantonese scores 44%, but Whisper has no
Cantonese and writes Mandarin, so that figure says little about the speech.

Voice cloning, WeSpeaker ResNet34-LM cosine (ONNX reference) against each
reference, mean of three clips (two for PyTorch):

| Output | JFK | voice A | voice B |
|---|---|---|---|
| rlx clone of JFK | **0.71** | 0.44 | 0.45 |
| PyTorch clone of JFK | **0.68** | 0.45 | 0.45 |
| rlx clone of A | 0.57 | **0.95** | 0.51 |
| PyTorch clone of A | 0.56 | **0.96** | 0.49 |
| rlx clone of B | 0.54 | 0.44 | **0.87** |
| PyTorch clone of B | 0.51 | 0.42 | **0.87** |
| no reference voice | 0.72 | 0.68 | 0.68 |

Read this by column and row together. The voice used without a reference sits
near the middle of the embedding space and scores about 0.7 against everyone,
so one clone's score against one reference shows nothing here: `clone_fidelity`
calls that uncloned speech "same speaker" as JFK. Voices A and B are synthetic
clips (KittenTTS 2 `80_regal_narrator_female`, `southern_elderly_female`); JFK
is an old, band-limited recording, and a real second JFK recording scores 0.92
against it.

Reference quirks reproduced on purpose: S2 and codec RoPE tables are
rounded to bf16 (their `freqs_cis` buffers are bf16); S2 applies top-p before
the temperature, Drama 3 after; the S2 repetition window skips the first
frame; Drama 3's window includes prompt rows.

## Performance (M4 Pro, Metal; shared machine, so ± load)

| | S2-Pro | Drama 3 |
|---|---|---|
| per frame (21.5 frames = 1 s audio) | ~0.2–0.33 s (slow step ~0.1–0.19 s + fast frame ~0.1–0.15 s) | ~4 s (experts ~2 s, backbone graphs ~1.3 s, tower ~0.7 s) |
| 16 s two-speaker dialogue, end to end | 145 s (LM 131 s, codec 14 s) | — |
| peak memory | 10.3 GB | 14.6 GB |

What got S2 there, each step parity-checked:

- bf16 weights read in-kernel by new Metal GEMVs (`rlx-metal`
  `gemv_bf16w_splitk8[_n64]`): slow step 312 → ~110 ms, memory 25 → 9.6 GB.
- Device-resident KV cache, and attention fed un-expanded K/V where the
  backend infers GQA (CPU, Metal, wgpu; elsewhere one query row's heads are
  regrouped instead): at 4096 cached rows the step went 1108 → 217 ms.
- A vectorized contiguous concat in `rlx-metal` for the cache append.
- Codec convs as per-tap matmuls in rows layout (Metal's direct conv ran
  below CPU speed): 16 s of audio decodes in 14 s instead of 42 s.
- Weights a sibling graph already placed in Metal's shared weight store are
  not uploaded again, and the prefill graph stays alive on Metal so its pages
  do: a new decode/prefill bucket costs ~0.1 s instead of ~13 s.

The floor for a bf16 S2 frame is ~15 GB of weight traffic (36-layer step +
10 fast steps), ~56 ms at this chip's bandwidth: real time needs both
near-peak GEMVs and fewer bytes (quantized weights).

Audio8 on the same machine, a 12.1 s utterance (261 frames), best of two runs
each on a heavily loaded machine (load average 40–77 on 14 cores, so these are
upper bounds):

| Device | LM per frame (slow + fast) | LM total | Codec | End to end | Peak memory |
|---|---|---|---|---|---|
| Metal | **32 ms** (10 + 22) | 14.2 s | 8.0 s | 22.3 s | 3.1 GB |
| MLX | 90 ms (34 + 56) | 38.1 s | 3.9 s | 42.1 s | 11.2 GB |
| CPU | 187 ms (69 + 118) | 51.0 s | 3.3 s | 54.4 s | 2.8 GB |
| wgpu | 178 ms (65 + 113) | 47.9 s | 2.2 s | 50.1 s | 5.0 GB |

Real time is 46.5 ms per frame, so on Metal the frame loop runs about 1.45×
faster than the audio it makes. End to end it does not: about 5 s of LM
startup (weight upload, graph builds, the prompt) and a Metal codec decode
that is slower than the CPU's. Timings moved by up to 2.5× between runs here.

### Several sentences at once

Decoding one frame reads every weight once; the arithmetic is small. So
`--batch N` (CLI), `audio8::synthesize_batch` (library) and
`speak(text, { batch: N })` (browser) cut the text into sentences and decode
up to N of them together: every matmul takes one row per sentence, attention
runs per sentence on its own cache (`[n, cap, kv]` per layer, every prompt
placed to end at one slot, each with its own RoPE positions), and each
sentence keeps its own sampler and seed (`seed + i`). Batched frames are
bit-identical to generating each sentence alone, on CPU and Metal, greedy and
sampled (`a_batch_samples_what_each_prompt_samples_alone`, the real-weight
`examples/batch_bench.rs`; the pipeline split is checked by
`a_pipeline_of_stages_batches_what_generate_batch_does`).

One voice across sentences: without `--ref-wav`, the first sentence is
generated alone and becomes the reference voice of the rest. On a
four-sentence paragraph WeSpeaker scores every later sentence 0.72–0.83
against the first (one pass: 0.66–0.78), and Whisper large-v3 transcribes
both versions correctly.

Measured with `examples/batch_bench.rs` (64 frames each, Metal, machine
under other load, so compare within a row):

| sentences | one by one | together |
|---|---|---|
| 2 | 39.7 frames/s | 51.9 frames/s |
| 4 | 38.8 frames/s | 69.6 frames/s (3.2× real time) |

This needed a Metal kernel: 2–8 rows against bf16 weights used to run one
thread per output column (2 rows: 7× slower a step than 1); `rlx-metal`'s
`gemv_bf16w_splitk8_rows` reads each weight once for all rows, and sums each
row in the one-row kernel's order. `--play` streams one pass and does not
batch.

## Limits

- An Audio8 package has no codec encoder: to clone a voice, pass its codes as
  `reference_codes` with `reference_text`. S2's package takes no reference at
  all. Packages were run on the CPU only.
- Neither model is real time. S2 is ~5–7× off; Drama 3 is dominated by its
  routed experts, which stream 1 GB of FP8 per frame through the page cache
  from wherever the checkpoint lives (an external disk here: 2.2 s cold vs
  0.6 s warm per frame for all 40 layers). Moving the experts onto the GPU is
  the next structural step.
- First use of S2 reads and uploads 7 GB of weights (~7 s warm, ~20 s from a
  cold external disk).
- CPU, Metal, MLX and wgpu pass every parity test above; CUDA, ROCm and
  Vulkan are wired but untested (no such hardware here).
