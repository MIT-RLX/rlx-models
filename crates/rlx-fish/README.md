# rlx-fish

Fish Audio text-to-speech and voice cloning on RLX, native Rust:

| Model | Checkpoint | Architecture |
|-------|------------|--------------|
| **S2-Pro** | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) (9 GB bf16) | dual-AR: 36-layer Qwen3-4B-shaped *slow* backbone (qk-norm, GQA 32/8, interleaved RoPE) picks one semantic token per frame; a 4-layer *fast* transformer fills the frame's other nine codes |
| **Drama 3** | [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) (40 GB, block-FP8 experts), demo Space [`fishaudio/Drama-3`](https://huggingface.co/spaces/fishaudio/Drama-3) | 40-layer Qwen3.5-MoE hybrid (30 Gated DeltaNet + 10 gated-attention layers, 256 experts / 8 active + shared expert) and a 13-layer codebook *tower* that attends across frames |

Both emit 10 codes per frame (21.5 frames/s) for the same codec, `codec.pth`
(shipped with both checkpoints): a causal 44.1 kHz DAC whose RVQ sits behind a
4× downsample with an 8-layer transformer on each side.

## Run

```bash
just fetch-fish-s2        # → weights/tts/fish-s2-pro
just fish-s2 "Hello from Fish Audio on RLX." /tmp/s2.wav metal

just fetch-fish-drama3    # → weights/tts/fish-drama-3 (40 GB)
just fish-drama3 "Hello from Drama 3." /tmp/d3.wav metal
```

or directly:

```bash
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

## Limits

- Neither model is real time. S2 is ~5–7× off; Drama 3 is dominated by its
  routed experts, which stream 1 GB of FP8 per frame through the page cache
  from wherever the checkpoint lives (an external disk here: 2.2 s cold vs
  0.6 s warm per frame for all 40 layers). Moving the experts onto the GPU is
  the next structural step.
- First use of S2 reads and uploads 7 GB of weights (~7 s warm, ~20 s from a
  cold external disk).
- CPU, Metal, MLX and wgpu pass every parity test above; CUDA, ROCm and
  Vulkan are wired but untested (no such hardware here).
