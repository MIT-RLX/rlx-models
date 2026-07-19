# Bonsai-27B backend bench

- host: `msi` (`Linux 7.0.0-27-generic x86_64`)
- gguf: `/home/user/rlx-models/weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf`
- max_tokens: 16
- prompt_chars: 597
- env: RLX_QWEN35_BENCH=1 RLX_KV_CACHE_MAX_RESIDENT=1

| backend | prompt_tok | new_tok | prefill_ms | decode_ms | ms/tok | tok/s | status |
|---------|------------|---------|------------|-----------|--------|-------|--------|
| cuda | 155 | 16 | 15860.6 | 8719.6 | 545.0 | 1.835 | OK |
