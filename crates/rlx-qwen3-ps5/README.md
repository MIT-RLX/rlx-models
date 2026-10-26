# rlx-qwen3-ps5

Qwen3 GGUF decode on a jailbroken PS5's CPU, as an
[rlx-ps5](../../../rlx/crates/backends/rlx-ps5/README.md) payload: loads the
model with its K-quant weights packed, locks them in memory, answers a prompt
and reports prefill and decode tokens/s.

Standalone (its own `[workspace]`), so the rlx-models workspace build and CI
never see it. It expects rlx checked out next to rlx-models (`../rlx`) and puts
every rlx crate on that checkout through `[patch.crates-io]`, so the payload
holds one copy of each.

## Build and run

From the rlx checkout:

```sh
just ps5 setup                       # once: the PS5 SDK
just ps5 build --manifest-path "$PWD/../rlx-models/crates/rlx-qwen3-ps5/Cargo.toml"
just ps5 send qwen3-ps5 --manifest-path "$PWD/../rlx-models/crates/rlx-qwen3-ps5/Cargo.toml" --host <ps5-ip>
```

Copy the model to the console first (FTP), e.g.
`/data/rlx-ps5/models/Qwen3-0.6B-Q4_K_M.gguf` — the default path. Settings go
in `/data/rlx-ps5/qwen3.conf` (`MODEL`, `N`, `MAX_SEQ`, `PACKED`,
`PROMPT_IDS`); see the top of `src/main.rs`. Results are written to
`/data/rlx-ps5/qwen3.txt` as well.

On a host the same binary runs with environment variables:

```sh
cargo run --release -- # with MODEL=/path/to/Qwen3-0.6B-Q4_K_M.gguf RLX_WORKERS=3
```

## Memory

A payload has about 4–5 GB of flexible memory and can keep about 1.6 GB
locked (PS5LM's measurement: 2.7 GB locked froze the console). Locking
matters: unlocked payload memory is paged out while the browser is in front.

The `load:` line is taken before anything compiles; `peak resident over the
run:` is the number that matters. Qwen3-0.6B, 96 tokens, payload defaults:

| GGUF | peak over the run |
|---|---|
| Q4_K_M | **0.90 GB** (was 4.1 GB) |
| Q4_0 | 0.88 GB |
| Q5_K_M | 0.99 GB |
| Q6_K | 1.09 GB |
| Q8_0 | 1.37 GB |

Every compiled graph holds its own weight arena, so the payload compiles
exactly one — the decode graph — and sets (any of these can be overridden
with an `RLX_…=` line in `qwen3.conf`):

| option | effect |
|---|---|
| `RLX_PACKED_LM_HEAD=1` | head reads the packed `output.weight`, or the packed embedding table when the file has none, not an F32 copy |
| `RLX_QWEN3_HOST_EMBED=1` | token rows dequantized from the mmapped GGUF and fed as `inputs_embeds`; no graph holds the 593 MB F32 table |
| `RLX_QWEN3_DECODE_BUCKET_MIN=<MAX_SEQ>` | one decode bucket for the whole context: one compile, one arena |
| `RLX_QWEN3_PREFILL_DECODE=1` | the prompt goes through that decode graph a token at a time: no prefill graph (faster on a short prompt, slower on a long one) |
| `RLX_KV_CACHE_MAX_RESIDENT=1` | a superseded decode bucket is dropped, not kept |
| `RLX_QWEN3_GQA_NATIVE=1` | attention reads the 8 shared KV heads in place, no per-token copy to 16 |
| `RLX_GGUF_PACKED_Q4_1=1` | the few Q4_1 tensors of a Q4_0 file stay packed, not F32 |
| `RLX_CPU_EXEC_IN_POOL=1` | each step runs on a pool worker, so the ~225 parallel kernels per token join by spinning instead of parking the main thread |

The decode graph is compiled right after load (`decode graph:` line, ~0.6 s),
so no token waits for it.

## Expected speed on the console

Measured as the PS5 build runs it — no BLAS, `portable-gemm` + `int8-gemv`,
AVX2/FMA/F16C/BMI2 — on three pinned AVX2 cores (i9-12900H, Linux),
Qwen3-0.6B, 95 generated tokens, payload defaults, alternating runs with a
CPU-only llama.cpp on the same three cores (`llama-cli -t 3`, its generation
rate):

| GGUF | rlx, average | llama.cpp |
|---|---|---|
| Q4_K_M | **80.5 tok/s** | 82.5 tok/s |
| Q4_0 | 82.5 tok/s | 88 tok/s (87–95) |
| Q5_K_M | 73.5 tok/s | 75 tok/s |
| Q6_K | 70.6 tok/s | 73 tok/s |
| Q8_0 | 61.6 tok/s | 63 tok/s |

The average and the steady (median step) rate now agree: nothing compiles
mid-answer. A 21-token prompt takes ~0.3 s. llama.cpp prefills far faster
(~320 tok/s); here a long prompt costs ~13 ms per token.

Of a Q4_K_M step (~12 ms), ~10.4 ms is the packed matmuls — about 36 GB/s
of the ~53 GB/s the three cores can read — and ~1 ms attention. llama.cpp
keeps a lead on Q4_0 by repacking its rows eight at a time at load.

What got it here, from 11.0 tok/s and 4.1 GB:
- the KV cache resident in the CPU executable, kept in place in its arena
  slot (one row written per token, not the whole padded cache);
- int8 decode GEMVs for every format (`int8-gemv`), after llama.cpp's AVX2
  kernels: scales and min/offset terms kept in vector registers, eight f16
  block scales converted at once, two Q4_0/Q4_1 blocks per step, the
  activation quantized with AVX2;
- the packed LM head and host-side embedding, one compiled graph;
- GQA-native attention, split across heads, skipping masked keys and their
  softmax chunks, eight keys' dots per AVX register (bit-identical);
- each step run on a pool worker;
- the packed decode graph now takes the bucket's key mask. Before, the zero
  rows between the context and the bucket's top took part in attention, so
  the text changed with the bucket size; it no longer does.
