# rlx-minicpm-v

OpenBMB [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) in RLX: 4.6 / 4.7
(`model_type = minicpmv4_6 | minicpmv4_7`, e.g.
[MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)) with images
and video, and the resampler generation 2.5 / 2.6 / 4.0 / 4.5 (`model_type = minicpmv`,
[below](#resampler-generation-40--45)).

| piece | where |
|---|---|
| text model — Qwen3.5 hybrid trunk (Gated DeltaNet + gated attention), dense or MoE | [`rlx-qwen35`](../rlx-qwen35), loaded from the same HF directory |
| image slicing — thumbnail + up to 9 slices, sides a multiple of 56 px | `preprocess` |
| SigLIP-400M tower, window-attention merger after layer 6 (2×2 → 1), 2×2 merger into LM space | `vision` |
| canvas M-RoPE (4.7) / 1-D positions (4.6) | `positions` |
| chat template, `<image_id>N</image_id><image>…</image><slice>…</slice>` placeholders | `prompt` |

```sh
just fetch-minicpm-v                       # → weights/vision/minicpm-v-4.7-1b (3.1 GB, bf16)
just minicpm-v -- --model-dir weights/vision/minicpm-v-4.7-1b \
    --image crates/rlx-locateanything/fixtures/sample.jpg --prompt "Describe this image."
```

Flags: `--image` (repeatable), `--video` (repeatable, with `--max-frames N`,
default 128, and `--video-slices N`, default 1), `--max-tokens`,
`--temperature`, `--thinking` (opens a `<think>` block), `--bf16-vision` (keeps
the tower's matrices in bf16), `--max-seq`, `--device`. Pictures and clips go
into the prompt in the order given.

## How it runs

Every crop goes through the tower on its own — HF packs them into one
sequence but attends within each crop, so this is the same computation. The
tower graph is compiled per padded patch count (multiples of 256) and takes
the crop's geometry as data: padding is masked out of attention, and both
2×2 regroupings are row gathers with host-computed indices. Typical photos
slice into crops of at most 1024 patches, so one graph serves them all.

Canvas M-RoPE puts the thumbnail and the slices of a picture on one 2-D
canvas: slices tile it, the thumbnail is stretched across it with a rounded
`linspace`, and `<image>`/`</image>`/`<slice>`/`</slice>`/`\n` are pinned to
crop corners. The port of `get_rope_index` is literal — `\n` counts as a frame
marker, so the newline after an image joins the frame span.

## Parity

`just test-minicpm-v-parity [DIR] [IMAGE] [DEVICE]` dumps an f32 HF reference
with `scripts/minicpmv_reference.py` (needs `.venv-minicpmv`: transformers ≥
5.18, torch, pillow) and checks, stage by stage: crop grids and pixels, each
crop's merger output, prompt ids, every canvas position, prefill logits and
greedy tokens.

Measured on MiniCPM-V-4.7-1B, a 640×360 photo (thumbnail 24×44 patches + a
1×2 slice grid, 223 prompt tokens):

| stage | CPU | Metal |
|---|---|---|
| crop grids, pixels | identical | identical |
| merger output per crop | cos 1.0000000, max abs Δ ≤ 4.9e-5 | cos 1.0000000, max abs Δ ≤ 1.4e-4 |
| prompt ids, canvas positions | identical | identical |
| prefill logits | max abs Δ 6.9e-4, same argmax | max abs Δ 6.2e-4, same argmax |
| greedy, 24 tokens | identical | identical |

(merger outputs reach 4.8 in magnitude.) MLX gives the same: vision cos
1.0000000, logits 7.3e-4, same argmax. CUDA (RTX 3080 Ti Laptop,
`RLX_CUDA_PARITY=1` for strict f32 — by default cuBLAS uses TF32): pixels and
ids identical, vision cos 1.0000000 (max abs Δ 1.4e-4), logits 5.9e-4, 24/24
greedy tokens; the GGUF F16 pair the same (6.2e-4, identical text).

wgpu and Vulkan do not run the trunk yet, for reasons in `rlx-qwen35` that
plain Qwen3.5 shares: on wgpu its prefill arena outgrows the 4 GiB buffer
limit past ~64 tokens (5.2 GiB here; the in-graph projection-weight concats
alone take 1.4 GiB), and on Vulkan Qwen3.5-0.8B's own logits are 20 off
against transformers. The vision tower matches on Vulkan (cos 1.0000000).
ROCm is wired but not run.

The script saves the image as PIL decoded it (`<out>.png`) and the test reads
that: JPEG decoders disagree by a few 8-bit levels (PIL's libjpeg vs the
`image` crate), which is a decoder difference, not a model one. On the
original JPEG the thumbnail pixels differ by up to 4 levels.

## GGUF

The official release (`openbmb/MiniCPM-V-4.7-1B-gguf`) is an LM GGUF
(`general.architecture = qwen35`) plus a vision `mmproj` GGUF. Point
`--model-dir` at the directory and pick the quantization, or name the pair:

```sh
hf download openbmb/MiniCPM-V-4.7-1B-gguf MiniCPM-V-4_7-1B-Q4_K_M.gguf mmproj-model-f16.gguf \
    --local-dir weights/vision/minicpm-v-4.7-1b-gguf
just minicpm-v -- --model-dir weights/vision/minicpm-v-4.7-1b-gguf --prefer Q4_K_M \
    --image photo.jpg --prompt "Describe this image."
just minicpm-v -- --weights <lm.gguf> --mmproj <mmproj.gguf> --image photo.jpg --prompt "…"
```

The repo has no `config.json` or `tokenizer.json`. The config comes from the
mmproj's `clip.*` keys (the position grid from `v.position_embd`'s shape, the
window merger from whether `v.vit_merger.*` exists, so the `-4x` mmproj runs
4x mode), the marker ids from the LM's vocabulary, and prompts are tokenized
with the GGUF's own BPE. The mmproj tensors are the HF ones under llama.cpp
names; the loader renames them and pins the HF shapes.

A GGUF does not record that the trunk's M-RoPE is interleaved (llama.cpp
derives `IMROPE` from the `qwen35` architecture), so the runner sets it.
OpenBMB's llama.cpp-omni adds `qwen35.rope.section_order = zyxt`; that orders
llama.cpp's own 4-slot position buffer and changes no math.

Parity (`gguf_reference_parity` in
`tests/hf_parity_live.rs`) against the same HF reference:

| file | config, pixels, ids, positions | vision features | prefill logits | greedy, 24 tokens |
|---|---|---|---|---|
| `F16` + `mmproj-model-f16` | identical (config == `config.json`) | cos 1.0000000 | 6.9e-4, same argmax | identical |
| `Q4_K_M` + `mmproj-model-f16` | identical | cos 1.0000000 | 5.5 max abs; argmax flips where HF's own top-2 margin is 0.72 | "Two football coaches in suits on the pitch, one shouting and pointing, with the other observing intently during a match." |

## Video

`--video clip.mp4` samples frames the way `MiniCPMV4_6VideoProcessor` does —
one per second, or for a clip longer than `--max-frames` seconds one per 0.1 s
thinned to `--max-frames` with numpy's integer `linspace` — and decodes them
with the `ffmpeg` / `ffprobe` command-line tools (rgb24). Each frame then goes
through the picture path (resize, `--video-slices` slices) and becomes one
`<image>…</image>` block of `<|video_pad|>` tokens, frames back to back with no
`<image_id>`; canvas M-RoPE gives each frame its own canvas.

`just test-minicpm-v-video-parity CLIP [DIR] [FRAMES] [DEVICE]` runs
`scripts/minicpmv_video_reference.py`, which samples with the HF processor's own
`sample_frames` and decodes with the same `ffmpeg` call, so both sides read
identical pixels. On the model card's `football.mp4` (37 s, 1080p60), 16
frames, MiniCPM-V-4.7-1B:

| stage | CPU | Metal | MLX |
|---|---|---|---|
| sampled frame indices, crop grids, pixels | identical | identical | identical |
| features of all 16 crops | max abs Δ 1.2e-4, cos 1.0000000 | 2.0e-4 | 3.9e-4 |
| ids, canvas positions (1108 tokens) | identical | identical | identical |
| prefill logits | 8.5e-4, same argmax | 1.1e-3 | 1.0e-3 |
| greedy, 24 tokens | identical | identical | identical |

CUDA: features cos 1.0000000, ids and positions identical, logits 1.1e-3,
24/24 tokens. The rig's ffmpeg (8.0.1, Ubuntu) converts YUV to RGB differently
from the one that made the reference (8.1.2) — up to 3 levels — so the test
holds pixels to equality only when the two `ffmpeg -version` lines match (the
script records it) and runs every later stage on the reference's pixels.

`stack_frames > 1` (per-second composites of sub-frames) is not supported: the
HF processor builds the composites for the prompt but encodes the raw frames,
and its own forward then rejects the token / feature count mismatch (792 vs
1584 at 8 frames), so there is nothing to match.

## Resampler generation (4.0 / 4.5)

`model_type = minicpmv` checkpoints run OpenBMB's remote code: the same
SigLIP-400M tower without the window merger, then a perceiver resampler —
64 learned queries cross-attend to each crop's patches, with 2-D sin-cos
positions added to the keys — so every crop is 64 LM rows. `--model-dir`
picks this path from `config.json`.

| version | language model | how it runs here |
|---|---|---|
| 4.0 ([MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4)) | `LlamaForCausalLM`, GQA 32/2, head_dim 128, LongRoPE | `rlx-minicpm-sala`'s step graph (`SalaConfig::from_llama_value`) |
| 4.5 ([MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5)) | `Qwen3ForCausalLM` (per-head q/k RMSNorm) | the same, with `attn_qk_norm` |
| 2.6 ([MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6), gated; [FriendliAI/MiniCPM-V-2_6](https://huggingface.co/FriendliAI/MiniCPM-V-2_6) holds the same weights — every shard's sha256 matches) | `Qwen2ForCausalLM` (q/k/v biases); the template opens with a default system turn | the same, with `attn_qkv_bias` |
| 2.5 ([MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)) | Llama-3-8B, f16 | the same; f16 kept as f16 |

The trunk is the SALA crate's chunked step graph (all layers softmax
attention, no muP scaling); prompt rows come in as embeddings with the crop
features spliced into the `<unk>` slots. LongRoPE at these contexts (below
`original_max_position_embeddings`) divides each frequency by its
`short_factor`, attention factor 1.

The processor is its own: sides are multiples of 14 (not 56), the slice grid
is first-best (ties keep the earlier candidate), resizes are Pillow's bicubic
(`pil_resize_rgb8_exact`, bit-exact), and the placeholder is
`<image_id>i</image_id><image>` + 64 × `<unk>` + `</image>` plus slice rows.

`just test-minicpm-v-resampler-parity [DIR] [IMAGE] [DEVICE]` runs
`scripts/minicpmv_legacy_reference.py` in `.venv-sala` (transformers 4.56 —
the remote code predates 5.x; plus torchvision). `--widen` keeps the reference
in bf16 storage and widens each decoder layer to f32 for its own forward,
which is exactly an f32 load for a bf16 checkpoint (on MiniCPM-V-4 the two
references agree to 0.0). Same 640×360 photo:

| MiniCPM-V-4 | CPU | Metal | MLX |
|---|---|---|---|
| crop grids, pixels | identical | identical | identical |
| resampler output per crop (values up to 37.5) | max abs Δ 2.0e-4, cos 1.0000000 | 2.6e-4 | 2.8e-4 |
| prompt ids (221 tokens) | identical | identical | identical |
| prefill logits | 8.9e-5 | 9.3e-5 | 8.3e-5 |
| greedy | 18/18 identical | identical | identical |

CUDA (strict f32): resampler cos 1.0000000, ids identical, logits 1.1e-4,
same 18 tokens.

| MiniCPM-V-2_6 (8.1B, `--widen` reference) | Metal |
|---|---|
| crop grids, pixels | identical |
| resampler output per crop (values up to 22.4) | max abs Δ 2.0e-4, cos 1.0000000 |
| prompt ids (228 tokens, system turn included) | identical |
| prefill logits | 2.9e-4 (values up to 50.1) |
| greedy | 24/24 identical |

### 4.5 video

4.5 packs a clip: frames sampled at `--video-fps` (the card's `encode_video`:
uniform over the clip, each tagged with its time in 0.1 s steps), consecutive
frames grouped (`--packing`, else 1 to 3 by length), and each group becomes one
set of 64 rows — the resampler attends to all the group's patches at once,
each frame's keys carrying its 2-D position plus a temporal sin-cos position.
Only a group's first frame gets a placeholder (`use_image_id=False`,
`max_slice_nums=1`). `--video clip.mp4` with a 4.5 checkpoint takes this path.

`scripts/minicpmv_legacy_reference.py --video` ports `encode_video` onto
ffprobe / ffmpeg; on `football.mp4` at 1 fps, packed by 3 (37 frames, 13
groups), CPU: sampled frames and groups identical, pixels identical, group
features cos 1.0000000 (max abs Δ 1.3e-3), 890 prompt ids identical, logits
1.3e-4, 24/24 greedy tokens.

| MiniCPM-Llama3-V-2_5 (8.5B, f16, `--widen` reference) | CPU | Metal |
|---|---|---|
| crop grids, pixels | identical | identical |
| resampler output per crop (values up to 35.7) | max abs Δ 3.8e-4, cos 1.0000000 | 7.0e-4, cos 1.0000000 |
| prompt ids (314 tokens, Llama-3 template) | identical | identical |
| prefill logits | 1.3e-4 (values up to 23.3) | 8.8e-5 |
| greedy | — | 24/24 identical |

2.5 runs its tower in a way the later releases do not, and this crate
reproduces it: the crops of a prompt are padded to the longest one, and the
patch mask is built as `patch_attn_mask[i, :n] = True` on a `[B, 1, max]`
tensor, which slices the size-1 axis — every row of every crop counts as
valid. So a shorter crop attends to zero-pixel padding patches, its positions
are those of one `1 × max` row (transformers' Idefics2 tower reads the grid
from the mask), and its features depend on the other crops in the prompt.
`crop_feeds_padded` and `ResamplerRunner::encode_crops` do the same; the
resampler afterwards masks by the true sizes, as the reference does. The
position fraction follows transformers 4.5x (`j / n · (1 − 1e-6)`); the 4.40
code it was written against used `arange(0, 1 − 1e-6, 1 / n)`, which can land
one bucket lower on an exact boundary.

| MiniCPM-V-4_5 (8.7B, `--widen` reference) | CPU | Metal |
|---|---|---|
| crop grids, pixels | identical | identical |
| resampler output per crop (values up to 60.6) | max abs Δ 1.9e-4, cos 1.0000000 | 2.9e-4, cos 1.0000000 |
| prompt ids (221 tokens, `<think>` block closed) | identical | identical |
| prefill logits | 1.0e-4 (values up to 29.5) | 1.1e-4 |
| greedy | 24/24 identical | identical |

## Not covered

- MiniCPM-V 2.0 and earlier (different trunks and processors).
- 4.5 and 2.5 on CUDA: 16-17 GB of weights do not fit the 16 GB test card.
- `stack_frames > 1` video (see above).
- wgpu and Vulkan for 4.6 / 4.7 (see Parity); CUDA and ROCm are not run.
