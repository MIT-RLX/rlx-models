# Cross-backend harness report

- host: `Linux`  backends: `cpu`, `wgpu`, `rocm`

| model | kind | cpu | wgpu | rocm |
|---|---|---|---|---|
| `qwen3-0.6b` | lm | 📦 | 📦 | 📦 |
| `whisper` | asr | 📦 | 📦 | 📦 |
| `melotts` | tts | 📦 | 📦 | 📦 |
| `tiny-tts` | tts | 📦 | 📦 | 📦 |
| `kokoro` | tts | ❌ | ❌ | ❌ |
| `supertonic` | tts | ✅ | ❌ 0.00364 | ❌ |
| `piper` | tts | 📦 | 📦 | n/a |
| `moss-nano` | tts | 🧱 | 🧱 | n/a |
| `luxtts` | tts | 🧱 | 🧱 | n/a |
| `zipvoice` | tts | 📦 | 📦 | n/a |
| `styletts2` | tts | ❌ | ❌ | ❌ |
| `chatterbox` | tts | 📦 | 📦 | 📦 |
| `sesame` | tts | 📦 | 📦 | 📦 |

## Needs attention

- ❌ `kokoro` / cpu: error: load native Kokoro from weights/tts/kokoro-82m: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
- ❌ `kokoro` / wgpu: error: load native Kokoro from weights/tts/kokoro-82m: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
- ❌ `kokoro` / rocm: error: load native Kokoro from weights/tts/kokoro-82m: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
- ❌ `supertonic` / wgpu: cosine 0.004 vs cpu
- ❌ `supertonic` / rocm: panic: rlx-rocm Reduce: only single last-axis supported (got axes=[1, 2], rank=3)
- ❌ `moss-nano` / cpu: see rlx-moss-nano.rlx-moss-nano.log
- ❌ `moss-nano` / wgpu: see rlx-moss-nano.rlx-moss-nano.log
- ❌ `luxtts` / cpu: see rlx-luxtts.rlx-luxtts.log
- ❌ `luxtts` / wgpu: see rlx-luxtts.rlx-luxtts.log
- ❌ `styletts2` / cpu: Error: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
- ❌ `styletts2` / wgpu: Error: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
- ❌ `styletts2` / rocm: Error: native Kokoro bundle missing encoder.onnx in /home/user/rlx-models/weights/tts/kokoro-82m/onnx/rlx-split (run scripts/split_kokoro.py)
