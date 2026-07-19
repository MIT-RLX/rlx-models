# Bonsai-27B coherent backend check

- gguf: `/Users/Shared/rlx-models/weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf`
- prompt: What is the capital of France? Reply with one short sentence.
- max_tokens: 24  max_seq: 48

| backend | status | words | text |
|---------|--------|-------|------|
| metal | FAIL (rc=1) | 0 | _(see metal.log)_ |
| mlx | FAIL (rc=1) | 0 | _(see mlx.log)_ |
| coreml | FAIL (rc=1) | 0 | _(see coreml.log)_ |
| wgpu | FAIL (rc=1) | 0 | _(see wgpu.log)_ |
