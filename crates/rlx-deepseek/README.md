# rlx-deepseek — DeepSeek-V3 / V3.1 (+ Kimi-K2) for RLX

Native RLX port of DeepSeek-V3, the SOTA-open model. Its two defining features —
both absent from the llama/qwen runners — are implemented here:

- **MLA (Multi-head Latent Attention)** — `mla.rs`. Low-rank compressed Q and KV
  (`q_a→rms→q_b`, `kv_a_with_mqa→rms→kv_b`), a decoupled RoPE head shared across
  query heads (MQA-style), `nope`/`rope` split + concat, value zero-padded to
  `qk_head_dim` for the fused attention op then sliced back to `v_head_dim`.
- **Fine-grained MoE** — `moe.rs`. Group-limited `noaux_tc` router (sigmoid +
  per-expert correction bias + n_group/topk_group group selection + top-k),
  reusing rlx-llada2's `group_limited_gate` op; 256 routed experts via
  `Op::GroupedMatMul` + one shared expert.

`flow.rs` assembles the decoder: `first_k_dense_replace` dense layers, then
MLA+MoE layers, `→ norm → lm_head`. **Kimi-K2 uses the same architecture**
(deepseek_v3 with 384 experts), so it comes almost free once weights load.

## Status

- Config, **MLA**, **fine-grained MoE**, and the **full text prefill graph**
  compile and run finite on **CPU + CUDA + wgpu + vulkan** (`cargo test
  -p rlx-deepseek`, `RLX_TEST_DEVICE=<dev>`: `config`, `mla_smoke`, `moe_smoke`,
  `text_flow_smoke` — all green on all four backends on NVIDIA).
- DeepSeek-V3 is 671B; real-checkpoint validation is out of local/NVIDIA RAM,
  deferred (see the mllama README for the parity workflow).

## Remaining

Runner + CLI; YaRN `mscale` folded into the attention scale (via
`attention_kind_opts` — v1 uses the base `qk_head_dim^-0.5`); optional KV-cache
(MLA caches the compressed `[kv_lora + rope]` latent); dispatch wiring.
