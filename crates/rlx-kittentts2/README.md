# rlx-kittentts2

[KittenTTS 2](https://huggingface.co/KittenML/kitten-tts-2) on RLX: a 1.7B
speech language model with ternary weights and in-context voice conditioning,
decoded to 24 kHz audio by ResembleAI's chatterbox-turbo S3Gen. No PyTorch, no
ONNX Runtime; every network runs as an RLX graph.

```text
text ─ clean-up/chunk ─ Qwen3-1.7B (Q2_0) ─ S3 tokens (25 Hz) ─┐
                         ▲ spk_proj(x-vector) at position 0     │
                         ▲ reference transcript + tokens        │
  ┌─────────────────────────────────────────────────────────────┘
  └ conformer encoder ─ meanflow U-Net ×2 Euler steps ─ mel (50 Hz) ─ HiFT ─ 24 kHz
                         ▲ reference mel + CAM++ x-vector
```

This is a different model from [`rlx-kittentts`](../rlx-kittentts/README.md)
(KittenTTS mini 0.8, a 15M phoneme-to-waveform net). The eight 0.8 speakers
(Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo) are among KittenTTS 2's
123 voices.

## Quick start

```bash
just fetch-kittentts2           # ~2.2 GB: LM, voices, S3Gen
just kittentts2 "Hello there, how are you today?"           # → /tmp/kittentts2.wav
just kittentts2 "Guten Morgen." German metal
cargo run -p rlx-kittentts2 --release --features metal -- \
    --text "[joyful] We won <laugh> I can (((hardly))) believe it!" \
    --voice Kiki --preset expressive --device metal -o out.wav
rlx-kittentts2 --list-voices

# clone a voice from 5–30 s of one speaker (the transcript is part of the prompt)
just kittentts2-clone assets/jfk/jfk_voice_clone.wav \
    "Ask not what your country can do for you. Ask what you can do for your country." \
    "The quick brown fox jumps over the lazy dog."
```

The first load transcodes `lm/model-ternary.safetensors` into
`lm/model-q2_0.gguf` (about 3 s, written once, 1.0 GB). The transcode is exact:
every 128-wide ternary group `{-s, 0, +s}` with a bf16 scale becomes one `Q2_0`
block with the same scale in f16. It fails rather than rounding if a scale
cannot be represented exactly.

## Library

```rust
use rlx_kittentts2::{Device, GenerateOpts, KittenTts2, resolve_s3gen, write_wav};

let repo = std::path::Path::new("weights/tts/kitten-tts-2");
let mut tts = KittenTts2::open(repo, &resolve_s3gen(repo)?, Device::Metal)?;
let opts = GenerateOpts { voice: Some("Bruno".into()), seed: 7, ..Default::default() };
let (wav, _chunks) = tts.generate("Hello there.", &opts, |_| {})?;
// or a cloned voice:
let me = tts.clone_voice("me.wav".as_ref(), "What the clip says.")?;
let (wav, _) = tts.generate_with_voice("Hello there.", &me, &opts, |_| {})?;
write_wav("out.wav".as_ref(), &wav, tts.sample_rate())?;
```

## What is ported

| piece | upstream | here |
|---|---|---|
| speech LM | `Qwen3ForCausalLM` + `spk_proj` (Linear 512→2048 + LayerNorm) over position 0 | `rlx-qwen3` native packed generator; `Qwen3Generator::prefill_embeds_get_last_logits` takes the overwritten embedding |
| weights | `kittenml-ternary-1.58` (5 trits/byte + bf16 group scales) | lossless `Q2_0` GGUF (`ternary.rs`) |
| sampler | token-run penalty, windowed repetition penalty (end tokens exempt), temperature, top-k, top-p, min-p | `sampling.rs`, same order and tie rules |
| text | tag protection, bracket/dash/math-glyph clean-up, sentence chunking, seam joining | `text.rs` (verified against upstream on the same inputs) |
| S3Gen encoder | `UpsampleConformerEncoder` with ESPnet relative-position attention | `s3gen/encoder.rs` (rel-shift is the attention op's additive bias) |
| flow | meanflow `ConditionalDecoder`, 2 Euler steps | `s3gen/estimator.rs` |
| vocoder | HiFT (f0 predictor, 9-harmonic NSF source, iSTFT head) | `s3gen/hift.rs`; convolutions on device, source/STFT on host |
| voices | `voices/*.npz` + `cpp/default/voices.json` conditioning | `voices.rs` |
| cloning | S3 tokenizer (Whisper-shaped encoder + FSMN + FSQ), pyannote `XVectorSincNet`, CAM++, Matcha mel, Kaldi fbank | `reference/` (CAM++ graph from `rlx-funasr`) |
| long references | clips ≥ 30 s tokenized in 30 s windows, 26 s stride, 50 tokens dropped at each seam | `ReferenceEncoder::tokenize` |
| reference transcript | Whisper when the clone call has no transcript | `KittenTts2::transcribe_reference` via `rlx-whisper` (feature `transcribe`, default); the CLI looks for `--whisper DIR`, `RLX_WHISPER_DIR`, then `.cache/whisper-*` |
| text normalizer | `kitten_text_processing` 0.1.0: NeMo 1.2.0 English WFST tagger / verbalizer / post-processor, Moses detokenizer | `normalize/` — the grammars embedded (~0.8 MB, Apache-2.0), a native transducer walker; `--no-normalize` skips it |
| distilled decoder | `decoders/student_w4.pt`: half-width meanflow flow, 1 Euler step, BitNet layers (4-bit per-channel weights, int8 per-token activations), ≤ 4 s / 112-token prompt | `s3gen/student.rs`; `--decoder student_w4` |

## Parity

Run against `scripts/kittentts2_reference.py`, the upstream package in float32:

| stage | max \|Δ\| (CPU) | max \|Δ\| (Metal) | scale |
|---|---|---|---|
| first-step logits (158 241) | 1.2e-4 | 1.5e-4 | max \|logit\| 71.5 |
| greedy tokens (48, penalties on) | 23/48 ¹ | 48/48 | — |
| encoder `mu` | 1.2e-6 | 1.4e-6 | 2.2 |
| meanflow velocities | 1.7e-5 | 1.8e-5 | 14 |
| mel | 7.6e-6 | 7.6e-6 | 11.7 |
| HiFT f0 | 4.2e-7 | 5.4e-7 | 1.0 |
| waveform | 3.7e-5 | 3.3e-5 | 0.99 |

¹ Step 23 is a near-tie: after the penalties the top two tokens are 0.009
apart on CPU and 0.003 on Metal, while the two devices' logits differ by
~0.02. Teacher-forced (`lm_reference … OUT`, which writes each step's
penalized top-2 margin), every other step agrees. `KT2_STEPS=N` limits the
check to the first N steps on slow backends.

The student decoder matches `scripts/kittentts2_student_reference.py` exactly
with activation fake-quant off (`RLX_KITTENTTS2_STUDENT_NO_ACT_QUANT=1`) and to
cosine 0.99997 with it on (int8 rounding flips at different points). The
normalizer matches `kitten_text_processing` on 400/400 sentences
(`tests/normalize_parity.rs`, fixture from `scripts/kittentts2_normalize_fixture.py`).

Whisper (base.en) reads the fox pangram back 6/6.

Cloning, against `scripts/kittentts2_clone_reference.py` on a built-in voice's
own clip, each network fed the reference's exact inputs:

| stage | result |
|---|---|
| tokenizer log-mel / S3Gen mel / Kaldi fbank | ≤ 1e-4 / 1e-3 / 1e-3 |
| S3 tokenizer encoder, FSQ codes | ≤ 1e-3; codes identical |
| pyannote x-vector, CAM++ x-vector | ≤ 1e-3 relative; cosine > 0.9999 |
| whole clip → voice vs the shipped artifacts | S3 tokens 100 %, LM tokens 94 % (librosa's soxr vs a sinc resampler), x-vector cos 1.0000, speaker cos 0.992 (upstream's own Python scores 0.992 too) |

A 5 s JFK clip cloned onto the pangram: Whisper hears it word for word, and
its pyannote cosine to the reference is 0.38 against −0.04 for a built-in voice.
Clips ≥ 30 s give the reference's merged token count and > 98 % of its tokens
(`tests/clone_parity.rs`).

```bash
.venv-kitten2/bin/python scripts/kittentts2_reference.py \
    --repo weights/tts/kitten-tts-2 \
    --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
    --out weights/tts/kitten-tts-2/reference
just test-kittentts2-parity            # cpu; DEVICE=metal to switch
just test-kittentts2-whisper metal
.venv-kitten2/bin/python scripts/kittentts2_clone_reference.py \
    --repo weights/tts/kitten-tts-2 \
    --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
    --wav weights/tts/kitten-tts-2/voices/kitten_v08_voice_3_male.wav \
    --out weights/tts/kitten-tts-2/clone_reference
just test-kittentts2-clone
```

## Backends

| device | LM | S3Gen | notes |
|---|---|---|---|
| CPU | ✅ parity (near-tie flip at step 23, see above) | ✅ parity | prefill caches f32 dequantized weights (`RLX_DEQUANT_CACHE_MAX_BYTES` caps it) |
| Metal | ✅ parity | ✅ parity | recommended on macOS arm64 |
| MLX | ✅ parity | ✅ parity | native `quantized_matmul` for `Q2_0` (2-bit affine, group 128) |
| wgpu (macOS) | ✅ parity | ✅ parity | the tied head runs packed from the BF16 embedding; exact-length prefill (see below); decode 5–9 tok/s steady (Metal 13.7) on a loaded machine |
| wgpu (NVIDIA, Vulkan) | ✅ parity | — | RTX 3080 Ti; most ops run on the host there, so decode is slow (~0.1 tok/s): prefer `cuda` or `vulkan` |
| CUDA | ✅ parity | — | RTX 3080 Ti |
| Vulkan | ✅ parity | — | `Q2_0` GEMV on the GPU for decode; prefill on the host path |
| ROCm | wired, untested | wired, untested | no AMD GPU on the rig |

wgpu runs the prompt at its exact length rather than in 64-token buckets: a
bucketed graph selects the last real row with a `Gather`, and any gather
makes wgpu's memory plan pin every output's ancestors.

`RLX_KITTENTTS2_TIMING=1` prints per-stage times (prefill, decode, flow, vocode).

## Files

| path | what |
|---|---|
| `weights/tts/kitten-tts-2/` | `KittenML/kitten-tts-2`: `config.json`, `lm/`, `voices/`, `speaker/` (cloning), `cpp/default/voices.json` |
| `weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors` | `ResembleAI/chatterbox-turbo` (MIT); or set `RLX_KITTENTTS2_S3GEN` |

The KittenTTS 2 weights are under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md):
free for research, non-commercial and small commercial use, with attribution
requirements on redistribution.
