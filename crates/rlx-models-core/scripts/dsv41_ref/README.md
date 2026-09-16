# DeepSeek-V4.1 reference parity harness

Regenerates the fixtures behind
`crates/rlx-models-core/tests/dsv41_reference_parity.rs`.

The released `inference/model.py` runs unmodified on CPU once its tilelang
kernels are replaced. `kernel.py` here is that replacement: a
numerically-identical torch transliteration of `act_quant`, `fp4_act_quant`,
`fp8_gemm`, `fp4_gemm`, `sparse_attn` and `hc_split_sinkhorn`, written from the
prim_funcs rather than approximated.

```sh
./fetch.sh                                  # pull model.py / engram.py / vision.py
RLX_REF_NOQUANT=1 python3 dump.py           # text stack   -> dsv41_ref.json
RLX_REF_NOQUANT=1 python3 dump_vision.py    # ViT+aligner  -> dsv41_vision_ref.json
RLX_REF_NOQUANT=1 python3 dump_dspark.py    # draft head   -> dsv41_dspark_ref.json
```

Needs `torch`, `numpy` and `sympy`. No GPU and no checkpoint: every parameter
comes from `prng.py`, a name-keyed splitmix64 stream the Rust side reproduces
exactly, so the fixtures only have to carry shapes and outputs.

Two switches matter:

- **`RLX_REF_NOQUANT=1`** turns the in-place FP8/FP4 activation round-trips into
  no-ops. Those are precision simulation, not semantics, and the port computes
  the F32-exact value; leaving them on measures the quantization error instead of
  the port's.
- **`dump.py` pins `torch.topk`'s tie order** to lowest-index-wins. Torch leaves
  it unspecified and the Indexer produces exact ties constantly (it rectifies its
  head scores), so without pinning, neither implementation is reproducible, let
  alone comparable. `Op::TopK` uses the same rule.

`dump.py` writes every intermediate it can hook; the committed fixture is a
trimmed copy. Point `RLX_DSV41_REF` at the full one to bisect with
`cargo run -p rlx-models-core --example dsv41_bisect`.
