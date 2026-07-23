# rlx-openai

Canonical **OpenAI-compatible HTTP server** for RLX chat language models.
One process can register several engines; clients select via the request
`model` field (`RegistryBackend`).

Library HTTP stack: [`rlx-serve`](../rlx-serve). Prefer this binary over
per-crate `--serve` and the Qwen3-only `rlx-serve` binary.

## Engines

| `--engine` | Cargo feature | Notes |
|------------|---------------|--------|
| `qwen3` | (always on) | `SingleEngine` / optional continuous + fused batching |
| `laguna` | `laguna` | packed GGUF `LagunaEngine` (greedy) |
| `qwen35` | `qwen35` | `SingleEngine` |
| `gemma` | `gemma` | `SingleEngine` |
| `llama32` | `llama32` | `SingleEngine` |
| `lfm` | `lfm` | `SingleEngine` |

`all-engines` enables every optional family. Backend features:
`apple-silicon`, `metal`, `mlx`, `cuda`, `all-backends`.

## Run

```bash
just features=apple-silicon,laguna openai-serve -- \
  --host 127.0.0.1 --port 8080 \
  --engine qwen3 --weights /path/to/qwen3 --model-id qwen3 --device metal \
  --engine laguna --weights /path/to/Laguna.gguf --tokenizer-dir /path/tok \
    --model-id laguna --device metal
```

Shared: `--host`, `--port`, `--max-tokens`. Per-engine flags apply until the
next `--engine`.

## Curl

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "laguna",
    "messages": [{"role":"user","content":"Say hello."}],
    "max_tokens": 32
  }' | jq .
```

TTS / Whisper / vision are not chat `Engine`s — use their own CLIs.
