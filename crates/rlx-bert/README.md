# rlx-bert

BERT **encoder graph builder** for RLX. Given a [`BertConfig`](../rlx-models-core/src/config.rs) and a bag of weights, it emits an RLX IR graph for a BERT encoder — embeddings → N transformer layers → `hidden_states`.

This crate **only builds the graph**. It has no tokenizer, no weight loader, no CLI, and no inference loop. Loading files, compiling the graph for a device, feeding inputs, running, and pooling all live in **consumer crates**:

| Consumer | Uses rlx-bert for |
|---|---|
| [`rlx-embed`](../rlx-embed) | sentence-embedding runtime (MiniLM, sentence-transformers) — the main end-to-end path |
| [`rlx-clinicalbert`](../rlx-clinicalbert) | clinical-domain encoder + MLM/pooler head folded onto the encoder |
| [`rlx-grounding-dino`](../rlx-grounding-dino) | BERT text encoder for the vision-language model |

## Public API

```rust
use rlx_bert::{build_bert_graph_sized, build_bert_built, BertFlow};
use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;

let cfg = BertConfig::from_file("config.json".as_ref())?;   // HF config.json (or ::from_gguf)
let mut wm = WeightMap::from_file("model.safetensors")?;    // BF16/F16 auto-converted to F32

// Option A — a lowered IR `Graph` + params, ready to compile:
let (graph, params) = build_bert_graph_sized(&cfg, &mut wm, /*batch*/ 1, /*seq*/ 8)?;

// Option B — a `BuiltModel` (fold a task head on / customize compile before lowering):
let built = build_bert_built(&cfg, &mut wm, 1, 8)?;
// or the configurable builder directly:
let built = BertFlow::new(&cfg, 1, 8).with_profile(profile).build(&mut wm)?;
# anyhow::Ok(())
```

Dimensions are **baked into the graph** — `batch`/`seq` are concrete, so build again for a different shape. (RLX compiles for static shapes to get better fusion/codegen; consumers keep a small shape→graph cache and recompile on a new shape.)

## How it's built

Construction is one thin file (`bert.rs`) over a declarative flow (`flow.rs`). `BertFlow::build` describes the whole encoder with the shared `rlx_flow::ModelFlow` DSL:

```rust
ModelFlow::new("bert")
    .with_profile(CompileProfile::encoder())
    .input("input_ids", [batch, seq]).input("attention_mask", …)
    .input("token_type_ids", …).input("position_ids", …)
    .embed("embeddings.word_embeddings.weight")                     // token lookup       → [B,S,H]
    .gather_add("position_ids", "embeddings.position_embeddings.weight")
    .gather_add("token_type_ids", "embeddings.token_type_embeddings.weight")
    .layer_norm("embeddings.LayerNorm.{weight,bias}", eps)          // embedding LN
    .repeat_bert_layers(num_layers, prefix, qkv_style, hidden, heads, eps)
    .output("hidden_states")                                        // → [B, S, H]
    .build(&mut WeightMapSource(weights))
```

Two things are auto-detected from the weight keys at build time:

- **Prefix** — keys under `bert.` (a `BertFor…` checkpoint) get the `bert.` prefix; a bare encoder gets `""`.
- **QKV style** — `attention.self.{query,key,value}` ⇒ `BertQkvStyle::Bert` (BERT/MiniLM); otherwise `attention.attn.{q,k,v}` ⇒ `BertQkvStyle::Mpnet`. Either way Q/K/V are fused into one `[H, 3H]` matrix so attention is a single matmul.

### One encoder layer (post-LayerNorm topology)

`repeat_bert_layers` emits this per layer (defined in `rlx-flow/.../blocks/bert_layer.rs`):

```
INPUT [B,S,H]
  → MatMul(qkv_w) + qkv_b                → [B,S,3H]
  → Narrow ×3 (split last axis)          → Q, K, V each [B,S,H]
  → Attention(Q,K,V, attention_mask, heads, head_dim)   → [B,S,H]   # QKᵀ/√dₕ, mask, softmax, ·V
  → MatMul(out_w) + out_b
  → Add(INPUT)   → LayerNorm(ln1)        # residual THEN norm
  → MatMul(int_w) + int_b → GELU         → [B,S,intermediate]
  → MatMul(out2_w) + out2_b
  → Add(post-attn) → LayerNorm(ln2)      → [B,S,H]
```

`CompileProfile::encoder()` = `Direct` fusion (no KV-cache patterns — a stateless encoder), DCE + constant-folding, F32.

## End-to-end (how `rlx-embed` runs it)

```
BertConfig::from_file(config.json)
WeightMap::from_file(model.safetensors)
        │
build_bert_built(cfg, &mut wm, batch, seq)        ← this crate
        │
compile:  CPU+F32 → compile_built(built, device)
          GPU/precision → Session::new(device).compile_with(graph, opts) + set_param(...)
        ▼
CompiledGraph
        │  per request:
tokenize (HF tokenizers) → input_ids / attention_mask / token_type_ids   (u32)
   → cast to f32, flatten [B*S]; position_ids = 0..seq
   → compiled.run([input_ids, attention_mask, token_type_ids, position_ids])
   → hidden_states [B, S, H]
   → pool (mean / cls / none) → embeddings [B, H]
```

Note: every graph input is declared `DType::F32` — token ids are passed as floats that feed the embedding `Gather`.

## Build & test

```sh
cargo test -p rlx-bert                       # tiny-graph build + IR verification
cargo build -p rlx-bert --features metal     # GPU backends: metal | mlx | cuda | rocm | gpu | vulkan
```

Real-weights parity (downloads `sentence-transformers/all-MiniLM-L6-v2`, CPU vs TPU): `crates/rlx-models/tests/tpu_real_minilm.rs`.

## License

GPL-3.0-only. © 2026 Eugene Hauptmann, Nataliya Kosmyna.
