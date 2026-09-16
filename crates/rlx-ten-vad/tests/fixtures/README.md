# Parity fixtures

| file | what it is | produced by |
|------|------------|-------------|
| `synthetic_speech_16k.pcm` | 4 s of 16 kHz mono i16 LE (250 frames) | `rlx_ten_vad::synth::speech_like_clip()` |
| `reference_features.f32` | 250 × 41 f32 LE — the current-frame feature row | the upstream **C DSP**, `src/*.cc` built **`-ffp-contract=off`** |
| `reference_probs_onnx.f32` | 250 f32 LE — voice probability | **onnxruntime** on `src/onnx_model/ten-vad.onnx`, fed those features |
| `shipped_library_probs.f32` | 250 f32 LE — voice probability | the **prebuilt `libten_vad`** TEN-framework ships |

`reference_probs_onnx.f32` is the parity target: upstream's own DSP driving
upstream's own model. `reference_features.f32` is held to **bit equality** —
`src/ooura.rs` transliterates the reference's Ooura FFT and `src/pitch.rs` its
`f32` arithmetic, so every one of the 30 750 values must match exactly.

### Why `-ffp-contract=off` is not optional

clang on arm64 fuses `a*b + c` into a single `fma` by default, and the three
settings give three different spectra from the same source:

| `-ffp-contract=` | `r2c_1024` output vs this crate |
|---|---|
| `off` | **1024/1024 bit-identical** |
| `on` (clang's default) | 369/1024, max\|Δ\| 3.1e-5 |
| `fast` | 351/1024, max\|Δ\| 2.3e-5 |

So the C reference is not bit-reproducible across compilers or architectures,
and "bit-exact" only means anything against a stated build. `off` is the one
that matches plain IEEE-754 evaluation of the source, which is what a port can
be held to. `shipped_library_probs.f32` is tracked separately and
loosely — see "the prebuilt library" below.

The PCM is committed rather than generated at test time because `sin()` is not
bit-identical across libm versions, and every reference fixture describes one
exact waveform. `fixture_pcm_matches_the_generator` still checks the generator
against it (allowing ±1 LSB), so editing `src/synth.rs` fails loudly instead of
silently comparing against a stale reference.

No third-party audio is redistributed here.

## The prebuilt library

`libten_vad` embeds the DSP tables from `src/coeff.h` (the Hann-768 window and
the per-feature mean/std) **byte-for-byte**, so its frontend is the published
one. It does **not** contain the weights of its own `ten-vad.onnx` in `f32` or
`f16` at any byte alignment, in any order, and an `int8` correlation scan finds
nothing above 0.33 — so it runs a different build of the network. It also links
no onnxruntime.

Consequently the shipped binary sits ~9.6e-4 from the model in the same
repository, while this port sits ~6e-6 from it. Voice decisions agree in every
case. `shipped_library_decisions_agree` asserts that and pins the gap so a
regression in it is visible.

## Regenerating

```bash
git clone --depth 1 https://github.com/TEN-framework/ten-vad /tmp/ten-vad
cd /tmp/ten-vad
FX=/path/to/rlx-models/crates/rlx-ten-vad/tests/fixtures

# 1. the clip, straight from the crate's generator
cargo run -p rlx-ten-vad --release --example ten_vad_bench -- --dump-pcm $FX/synthetic_speech_16k.pcm
```

### 2. `reference_features.f32` — the upstream C DSP

Build with **`-ffp-contract=off`** (see above)
`src/{fftw.c,stft.cc,biquad.cc,pitch_est.cc,aed.cc}`, with the ORT-backed
`AUP_MODULE_AIVAD` in `aed_st.h` / `aed.cc` replaced by a stub whose `Process()`
writes its `[3, 41]` input out — that leaves the feature code path in `aed.cc`
untouched. Drive it through `AUP_Aed_proc` at `hopSz = 256`, and keep the
**last** of the three context rows per frame (the other two are the previous
frames' rows; `reference_parity.rs` rebuilds the stack the same way).

### 3. `reference_probs_onnx.f32` — the published model

```python
import numpy as np, onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 1     # as libten_vad does
sess = ort.InferenceSession('src/onnx_model/ten-vad.onnx', so,
                            providers=['CPUExecutionProvider'])
names = [i.name for i in sess.get_inputs()]
outs  = [o.name for o in sess.get_outputs()]

rows  = np.fromfile(f'{FX}/reference_features.f32', np.float32).reshape(-1, 41)
stack = np.zeros((3, 41), np.float32)
state = {n: np.zeros((1, 64), np.float32) for n in names[1:]}
probs = []
for row in rows:
    stack = np.vstack([stack[1:], row])
    r = sess.run(outs, {names[0]: stack[None], **state})
    probs.append(r[0].ravel()[0])
    state = dict(zip(names[1:], r[1:]))
np.array(probs, np.float32).tofile(f'{FX}/reference_probs_onnx.f32')
```

### 4. `shipped_library_probs.f32` — the prebuilt binary

```python
import sys, numpy as np
sys.path.insert(0, '/tmp/ten-vad/include')
from ten_vad import TenVad                      # ctypes wrapper around libten_vad

pcm = np.fromfile(f'{FX}/synthetic_speech_16k.pcm', np.int16)
vad = TenVad(256, 0.5)
np.array([vad.process(pcm[i * 256:(i + 1) * 256])[0]
          for i in range(len(pcm) // 256)], np.float32).tofile(f'{FX}/shipped_library_probs.f32')
```

Then `cargo test -p rlx-ten-vad --test reference_parity`.

Steps 3 and 4 need `numpy`, `onnxruntime`, and a platform TEN-framework ships a
binary for (macOS, Linux x64, Windows).

## Stage dumper

`examples/dump_stages.rs` writes this crate's `binpow` / `pitch` / `feat` per
frame in the same formats, and `--score FEATS` runs a `.feat.f32` file through
the network alone — that is how the frontend and the network were separated
when the numbers above were established.
