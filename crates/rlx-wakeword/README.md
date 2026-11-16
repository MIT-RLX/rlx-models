# rlx-wakeword

First-party streaming wakeword product on RLX (event API, multi-phrase, train/pack).

## Features

| Capability | Notes |
|------------|--------|
| Event API | `WakeEvent::{Idle, Candidate}` at configurable hop (default **40 ms**; 20 / 32 / 80) |
| Multi-phrase | One lite CNN head per phrase id |
| Train | RLX CNN SGD via `TrainBuilder` / `rlx-wakeword-train` (no PyTorch) |
| Ternary weights | `--ternary` → exact `{−1,0,+1}` (bake TQ2 / fused add-sub kernels) |
| Bundle | `manifest.json` + `phrase_<id>.safetensors` (+ optional `.rlxw` pack) |
| VAD gate | Earshot (`earshot` feature, **default**) |
| Speaker-id | Optional `--features speaker-id` enrollment gate |
| WASM | [`rlx-wakeword-wasm`](../rlx-wakeword-wasm) — Node, browser, Web Worker |

Core math: [`rlx-wakeword-core`](../rlx-wakeword-core) (`no_std` + alloc). Shared train/device helpers: [`rlx-wake`](../rlx-wake).

## Quick start

```bash
# Synth one phrase → bundle
just wakeword-train -- --synth --phrase hey_rlx --out-dir /tmp/wake_bundle

# N phrases + ternarize FC (bake / fuse friendly)
just wakeword-train -- --synth-n 4 --out-dir /tmp/wake4 --epochs 20 --ternary

# WAV dirs (repeatable)
just wakeword-train -- --out-dir /tmp/wake \
  --phrase hey=data/hey/pos:data/hey/neg \
  --phrase assist=data/assist/pos:data/assist/neg

# Auto: DIR/<id>/{positives,negatives}/
just wakeword-train -- --out-dir /tmp/wake --phrases-dir data/phrases

just wakeword-demo -- --wav clip.wav --bundle /tmp/wake_bundle --hop-ms 40
just test-wakeword
```

## Multi-phrase network (`rlx-wakeword-train-multi`)

One network with an output per phrase, run once per hop for all of them (the bundle's phrases share
`weights` and name their `output`). Trained on an RLX device with AdamW on the device
(`rlx-wake::train::multi`):

```bash
rlx-wakeword-train-multi --phrase hey=POS_DIR --phrase bye=POS_DIR --neg NEG_DIR … \
  --out-dir OUT --window-ms 2000 --model mid --device metal|cuda --epochs 100 \
  --pool-chunks 4 --res-blocks 4 --attn --pcen --bn \
  --noise NOISE_DIR --noise-p 0.5 --noise-snr 0,25 --mask-p 0.3 --pos-tail-ms 200 \
  --mine-ratio 2 --gain-db 12
# the loaded clips as one file, to train elsewhere:  … --save-pack r.pack --epochs 0
# then:                                              rlx-wakeword-train-multi --pack r.pack …
```

`--res-blocks` dilated residual blocks, `--attn` attention pooling, `--pcen` PCEN front end, `--bn`
batch normalization (folded into the convolutions on export), `--noise` noise mixed into training
windows on the fly, `--mask-p` SpecAugment, `--pos-tail-ms` positive windows ending after the
phrase, hard-negative mining every `--mine-every` epochs.

Manifest options read by `WakewordSession`: `min_hops` (hops in a row at the threshold),
`smooth_hops` (score = mean of the last N), per phrase `threshold` / `output` / `context_ms` /
`min_hops` / `smooth_hops` / `verify_threshold` / `verify_output`, and `verifier`
(`{weights, threshold, context_ms}`: a second network scored only when a phrase fires, which can
veto it). A worked example (Reachy Mini's wake words, with their data and training): `reachy-rs/crates/wakeword-peachy-r15`.

## Library

```rust
use rlx_wakeword::{TrainBuilder, TernaryOpts, WakewordBundle};

let bundle = TrainBuilder::new()
    .epochs(20)
    .hop_ms(40)?
    .ternary(TernaryOpts::fc_only()) // or ::all_weights()
    .phrase_dirs("hey_rlx", "pos/", "neg/")
    .out_dir("/tmp/wake")
    .run()?;

let mut sess = bundle.into_session()?;
for ev in sess.push(&pcm_16k_mono) {
    // WakeEvent::Idle { .. } | Candidate { phrase_id, score, .. }
}
```

### Speaker ID (`speaker-id`)

```bash
cargo build -p rlx-wakeword --features speaker-id
```

```rust
#[cfg(feature = "speaker-id")]
{
    use rlx_wakeword::{SpeakerIdConfig, SpeakerIdGate};
    let mut gate = SpeakerIdGate::new(SpeakerIdConfig {
        threshold: 0.65,
        require_match: true,
    });
    gate.enroll("alice", SpeakerIdGate::embed_from_pcm(&enroll_pcm, 64))?;
    let sess = bundle.into_session()?.with_speaker_gate(gate);
}
```

## Benches

```bash
just wakeword-multi-bench          # native N=2..10, f32 vs ternary
just wakeword-wasm-bench           # same table under wasm32 (Node)
just wakeword-wasm-web && just wakeword-wasm-worker-smoke
just wakeword-wasm-worker-serve    # http://127.0.0.1:8765/
```

Typical CPU (40 ms hop): N=10 f32 ≈ 3.7 ms/hop (RTF ~0.09); tern-all pack ≈ **0.08×** f32 size.
WASM: tern-all is often slightly faster than dense f32; RTF still ≪ 1.

## Cargo features

| Feature | Default | Role |
|---------|---------|------|
| `earshot` | yes | VAD gate via `rlx-vad` |
| `speaker-id` | no | Speaker enrollment / cosine gate |
| `all-backends` / `metal` / `cuda` / … | no | Forwarded to `rlx-wake` for device validation |

## Layout

| Path | Role |
|------|------|
| `WakewordSession` | Streaming PCM → events |
| `TrainBuilder` | Multi-phrase train → bundle |
| `WakewordBundle` | Load/save manifest + weights |
| `bin/rlx-wakeword` | Demo / device sweep |
| `bin/rlx-wakeword-train` | CLI train |

Compat engines (`rlx-openwakeword`, `rlx-nanowakeword`, `rlx-porcupine`, `rlx-voxrt`) remain for parity; new product work targets this crate.
