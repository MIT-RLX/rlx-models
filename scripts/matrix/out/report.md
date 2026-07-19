# Cross-backend harness report

- host: `Linux`  backends: `cuda`, `vulkan`
- cuda adapter: `NVIDIA GeForce RTX 3080 Ti Laptop GPU`
- vulkan adapter: `NVIDIA GeForce RTX 3080 Ti Laptop GPU`
- wgpu adapter: `NVIDIA GeForce RTX 3080 Ti Laptop GPU`

| model | kind | cuda | vulkan |
|---|---|---|---|
| `melotts` | tts | ✅ | ✅ |
| `tiny-tts` | tts | ✅ | ✅ |
| `kokoro` | tts | ✅ | ✅ |
| `supertonic` | tts | ✅ | ✅ |
| `moss-nano` | tts | ✅ | ✅ |
| `luxtts` | tts | ✅ | ❌ |
| `f5tts` | tts | · | · |
| `styletts2` | tts | ✅ | ✅ |
| `chatterbox` | tts | ❌ | ❌ |
| `sesame` | tts | ✅ | ✅ |
| `gepard` | tts | · | · |

## Needs attention

- ❌ `luxtts` / vulkan: panic: rlx-vulkan: no offset for node
- ❌ `chatterbox` / cuda: error: compile speech_encoder: invalid value: integer `3`, expected variant index 0 <= i < 2
- ❌ `chatterbox` / vulkan: error: compile speech_encoder: invalid value: integer `3`, expected variant index 0 <= i < 2
