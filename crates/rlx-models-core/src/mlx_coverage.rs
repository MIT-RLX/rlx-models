// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! **mlx-community coverage classifier** — "can rlx run this checkpoint at
//! all", across *every* code path, not just the generic builder.
//!
//! [`crate::classify_config`] answers a narrower question — "can
//! `build_standard_decoder_packed` build this" — and gates the generic loader
//! in [`crate::standard_decoder::DecoderSpec::from_config_json`]. It must stay
//! generic-only: flipping it to `true` for a MoE/MLA/DeltaNet arch would make
//! the generic builder mis-build.
//!
//! Many mlx-community architectures the generic builder rejects (Gemma AltUp,
//! Qwen3.5 gated-DeltaNet, DeepSeek/Kimi MLA, GLM-MoE, gpt-oss sinks, …) DO
//! have a dedicated rlx crate that runs them. This module layers that
//! knowledge on top: [`classify_coverage`] consults the [`dedicated_coverage`]
//! table FIRST, and only falls back to the generic classifier when no crate
//! claims the arch. The result is the "does rlx run it (via any path)" verdict
//! the mlx-community catalog reports.
//!
//! Adding a family: wire its dispatch (a `model_registry` `model_type` → runner
//! mapping) and its crate's MLX-affine loader, then add ONE arm to
//! [`dedicated_coverage`]. `supported` flips in one place and the catalog stays
//! consistent with what actually runs.

use serde_json::Value;

use crate::standard_decoder::{ModelSupport, classify_config};

/// Which rlx code path runs a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageVia {
    /// Config-driven [`crate::build_standard_decoder_packed`] — no per-model
    /// crate. The dense / standard-MoE SwiGLU-or-GeGLU + RMSNorm family.
    Generic,
    /// A dedicated crate builder (e.g. `"rlx-gemma"`, `"rlx-qwen35"`). Names
    /// the crate so diagnostics and the catalog can point at it.
    Dedicated(&'static str),
    /// No rlx path yet.
    None,
}

/// Validation state behind a coverage verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageStatus {
    /// Run end-to-end against an mlx-lm oracle (or covered by a passing parity
    /// test in-repo). Numerically trusted. → `supported`.
    Validated,
    /// Builder + dispatch wired and the checkpoint loads + runs, but local
    /// cos-parity is deferred — the weights are too large for the dev box
    /// (100B–1T MoE). Runs; not yet numerically verified. → `supported`.
    WiredDeferred,
    /// The model's graph builder exists (and is finite on synthetic weights),
    /// but there is NO runnable real-checkpoint path yet — e.g. a headless
    /// graph library with no runner, or a build path that can only emit F32
    /// params so a real quantized giant can't be held in memory. Honest "not
    /// runnable, but the hard modelling is done". → NOT `supported`.
    BuilderOnly,
    /// Not runnable yet.
    Unsupported,
}

impl CoverageStatus {
    /// Whether this status means rlx can actually run a real checkpoint.
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            CoverageStatus::Validated | CoverageStatus::WiredDeferred
        )
    }
    /// Short tag for catalog rows.
    pub fn tag(self) -> &'static str {
        match self {
            CoverageStatus::Validated => "validated",
            CoverageStatus::WiredDeferred => "wired (validation deferred)",
            CoverageStatus::BuilderOnly => "builder-only (no runner yet)",
            CoverageStatus::Unsupported => "unsupported",
        }
    }
}

/// Full coverage verdict for one checkpoint's `config.json`.
#[derive(Debug, Clone)]
pub struct ModelCoverage {
    /// `true` when rlx can run this via SOME path (generic or dedicated crate).
    pub supported: bool,
    pub via: CoverageVia,
    pub status: CoverageStatus,
    /// Family / blocker note for humans.
    pub reason: String,
    /// Structural fields the generic classifier extracted (vocab/hidden/layers/
    /// experts/…) — always populated so the catalog can render them regardless
    /// of which path runs the model.
    pub base: ModelSupport,
}

/// A dedicated crate's claim over an architecture.
struct Dedicated {
    krate: &'static str,
    status: CoverageStatus,
    note: &'static str,
}

/// Map a normalized `model_type` / `architectures[0]` to the dedicated crate
/// that runs it, or `None` if no crate covers it (fall through to the generic
/// classifier).
///
/// Consulted BEFORE [`classify_config`], so a family the generic builder would
/// reject is still reported supported when a crate covers it. Keep the arms in
/// the same download-priority order the waves were implemented so this doubles
/// as an index of what's wired.
fn dedicated_coverage(arch: &str) -> Option<Dedicated> {
    let a = arch.to_ascii_lowercase();
    let hit = |krate, status, note| {
        Some(Dedicated {
            krate,
            status,
            note,
        })
    };

    // ── Gemma 3n / 4 AltUp (rlx-gemma `GemmaQatLoader`, mlx-affine) ──
    // gemma3n-E2B validated: prefill cosine 0.99993 vs mlx-lm on CPU (argmax
    // flips only on a 0.5%-tie between the top-2 logits — 4-bit backend rounding,
    // not a loader defect). gemma3 / gemma2 stay on the generic GeGLU path (do
    // NOT claim them here); gemma4moe (A4B routed experts) is NOT yet wired.
    if (a.starts_with("gemma3n") || a.starts_with("gemma4")) && !a.contains("moe") {
        return hit(
            "rlx-gemma",
            CoverageStatus::Validated,
            "Gemma 3n/4 AltUp — rlx-gemma mlx-affine loader (E2B prefill cos 0.99993 vs mlx-lm)",
        );
    }

    // ── MiniMax-M2 (`minimax`): GQA + full-qk-norm + partial-RoPE + MoE ──
    // Standard transformer (NOT Lightning/linear attention). build_minimax_prefill
    // reuses the validated partial-RoPE + fine-grained-MoE pieces (block_sparse_moe,
    // no shared experts). 230B → won't fit 64GB, so correct-by-construction wire,
    // not run-validated.
    if a == "minimax" || a == "minimax_m2" || a == "minimax-m2" {
        return hit(
            "rlx-minimax",
            CoverageStatus::WiredDeferred,
            "MiniMax-M2 GQA+qk-norm+partial-RoPE+MoE — build_minimax_prefill (reuses validated MoE/RoPE); 230B needs RAM",
        );
    }

    // ── Nemotron-H (`nemotron_h`): hybrid Mamba-2 / NoPE-attn / ReLU²-MoE ──
    // build_nemotron_h_prefill reuses the validated `Op::Mamba2` SSD scan +
    // deepseek-style MoE router; block layout from `hybrid_override_pattern`.
    // In-scope repos are 30B–120B MoE giants (NVFP4/affine) → correct-by-
    // construction wire (tensor names verified vs the 30B index), not run-
    // validated (won't fit 64GB).
    if a == "nemotron_h" || a == "nemotronh" || a == "nemotron-h" {
        return hit(
            "rlx-nemotron-h",
            CoverageStatus::WiredDeferred,
            "Nemotron-H hybrid Mamba-2/NoPE-attn/ReLU²-MoE — build_nemotron_h_prefill (reuses Op::Mamba2 + validated MoE); 30B–120B needs RAM",
        );
    }

    // ── Hunyuan-V3 (`hy_v3`, HYV3ForCausalLM): GQA + qk-norm + full-RoPE + MoE ──
    // Standard transformer (NOT MLA/SSM). build_hy_v3_prefill reuses the validated
    // deepseek-style MoE router (sigmoid + expert_bias select, route_norm,
    // ×router_scaling) + SwiGLU shared expert + per-head qk-norm (after RoPE,
    // Hunyuan-family convention) + dense layer-0. Newer than mlx-lm/transformers
    // 5.3.0 (no oracle); correct-by-construction from config + verified tensor
    // shapes. In-scope repos are 80L/192-expert giants → won't fit 64GB.
    if a == "hy_v3" || a == "hyv3" || a == "hunyuan_v3" || a == "hunyuanv3" {
        return hit(
            "rlx-hunyuan",
            CoverageStatus::WiredDeferred,
            "Hunyuan-V3 GQA+qk-norm+full-RoPE+deepseek-MoE — build_hy_v3_prefill (reuses validated MoE/RoPE/qk-norm); 80L/192-expert giant needs RAM",
        );
    }

    // ── gpt-oss: attention-with-sinks + MXFP4 packed MoE ──
    // Full `build_gpt_oss_prefill` (attention with per-head sinks + YaRN + mixed
    // per-module quant [affine attn/embed, mxfp4 experts] + clamped-SwiGLU MoE +
    // untied head) validated on gpt-oss-20b: last-token cos 0.99972 + exact
    // argmax (12650 " Paris") vs mlx-lm oracle.
    if a == "gpt_oss" || a == "gpt-oss" || a == "gptoss" {
        return hit(
            "rlx-gpt-oss",
            CoverageStatus::Validated,
            "gpt-oss attention-sinks + MXFP4 MoE — build_gpt_oss_prefill, cos 0.99972 + exact argmax vs mlx-lm",
        );
    }

    // ── Qwen3.5 / Qwen3-Next gated-DeltaNet hybrid (rlx-qwen35) ──
    // The DeltaNet linear-attention scan, hybrid full-attention layers, MoE and
    // MTP heads are all implemented graph ops; mlx-community affine 4-bit dirs
    // load via `MlxLoader` (dequant→F32 on take) through the qwen35 runner.
    // WiredDeferred: no local Qwen3.5 checkpoint to cos-validate (disk-limited).
    if a == "qwen3_5"
        || a == "qwen3next"
        || a == "qwen3_next"
        || a.starts_with("qwen3_5_")
        || a == "qwen35"
        || a.starts_with("qwen35_")
        || a.starts_with("qwen35moe")
    {
        return hit(
            "rlx-qwen35",
            CoverageStatus::WiredDeferred,
            "Qwen3.5 / Qwen3-Next gated-DeltaNet — rlx-qwen35 runner, mlx-affine via MlxLoader",
        );
    }

    // ── DeepSeek MLA + fine-grained MoE (build_deepseek_prefill, packed) ──
    // Numerically VALIDATED: cos 0.999995 + exact argmax vs mlx-lm on
    // DeepSeek-V2-Lite-16B (real MLA no-q-LoRA + softmax/sigmoid gate + shared
    // experts + YaRN). deepseek_v2 runs here; V3 671B / Kimi-K2 1T use the same
    // builder but need adequate RAM (packed ~335GB+) → WiredDeferred.
    // deepseek_v4 (+MTP) is a distinct 5-subsystem arch → its own arm below.
    if a == "deepseek_v2" {
        return hit(
            "rlx-deepseek",
            CoverageStatus::Validated,
            "DeepSeek MLA+MoE — build_deepseek_prefill; MULTI-TOKEN greedy match vs mlx-lm on V2-Lite-16B \
             (exact 6/6 continuation \"Paris. It is the largest\", prefill cos 0.999995)",
        );
    }
    if a == "deepseek_v3" || a == "kimi_k2" || a == "kimi_k25" || a == "kimi_k2_5" {
        return hit(
            "rlx-deepseek",
            CoverageStatus::WiredDeferred,
            "DeepSeek-V3 / Kimi-K2 MLA(q-LoRA)+MoE — build_deepseek_prefill (V2-Lite-16B validated; \
             q-LoRA q_a/q_b path added for V3/Kimi, which store kv_b_proj + q-LoRA); 671B–1T need RAM",
        );
    }

    // ── DeepSeek-V4-Flash (`deepseek_v4`, +`deepseek_v4_mtp`) — 5-subsystem
    //    research arch: Hyper-Connections (hc_mult parallel streams + Sinkhorn
    //    mixing) + o-LoRA MLA (q-LoRA + fused wkv + grouped wo_a/wo_b + attn-sink
    //    + sliding-window/compressed-KV attention) + overlapping KV Compressor +
    //    learned sparse Indexer (top-k) + sqrtsoftplus/clamped-SwiGLU MoE ──
    // build_deepseek_v4_prefill assembles it (ref inference/model.py). ALL cores
    // validated cos-exact vs inline references (probes: hc_probe,
    // dsv4_compressor_probe, dsv4_overlap_probe, dsv4_olora_probe,
    // dsv4_sinkattn_probe, dsv4_indexer_probe, dsv4_hash_route_probe) and the full
    // forward builds+runs finite (dsv4_assemble_probe, ratio-4 overlap + Indexer
    // active). Tensor names + layer layout CONFIRMED against the real
    // mlx-community/DeepSeek-V4-Flash-4bit index: attn.compressor.* /
    // attn.indexer.{compressor.*,wq_b,weights_proj} names match; Indexer only on
    // ratio-4 layers; gate.tid2eid hash routing on the first n_hash_layers (3).
    // Uses only 2D affine/MXFP4 dequant (already works; no 3D quant). MTP head
    // unconsumed (qwen3.5_mtp precedent). Giant + FP4 → e2e-on-real-checkpoint
    // deferred (correct-by-construction, unrunnable locally).
    // GA `DeepSeek-V4-Flash-0731` adds, over the April preview: YaRN on the
    // compressed-layer RoPE (build_deepseek_v4_stage rope_tables, cos-exact vs
    // precompute_freqs_cis); DSpark speculative decoding (build_dspark_stage +
    // build_dspark_markov_head/confidence_head + dspark_forward_head/greedy_accept —
    // heads cos-exact, stage builds+compiles, driver validated); fp8-block + fp4
    // (MXFP4) HOST dequant for the deepseek-ai fp8/fp4 original (dsv4_quant,
    // cos-exact) alongside the mlx-community affine repacks; and `deepseek4` GGUF
    // recognition (DeepseekV4Spec::from_gguf_metadata + hf_key_to_deepseek4_gguf,
    // validated vs the real bartowski MXFP4 header — GGUF is base-only, no DSpark).
    // Still WiredDeferred: every real GA checkpoint is a 96–167GB giant (>64GB RAM)
    // with no local oracle, so e2e→Validated remains a hardware ceiling.
    if a == "deepseek_v4" || a == "deepseek_v4_mtp" || a == "deepseek4" {
        return hit(
            "rlx-models-core",
            CoverageStatus::WiredDeferred,
            "DeepSeek-V4-Flash (GA 0731) — build_deepseek_v4_prefill (Hyper-Connections + o-LoRA MLA \
             + overlapping KV-Compressor + learned Indexer top-k + sqrtsoftplus/hash MoE + YaRN) plus \
             full DSpark speculative decoding (build_dspark_stage/heads/driver) and 3 quant load paths \
             (mlx-community affine, deepseek-ai fp8+fp4 via dsv4_quant, `deepseek4` GGUF MXFP4 via \
             from_gguf_metadata+name-map). All cores cos-exact vs inference/model.py; forward + DSpark \
             stage build+compile finite; config/metadata verified vs real 0731 config.json + GGUF \
             header. e2e→Validated blocked ONLY by hardware: 96–167GB checkpoints (>64GB RAM), no \
             smaller variant, no local oracle",
        );
    }

    // ── GLM-5 (`glm_moe_dsa`, GlmMoeDsaForCausalLM = DeepSeek-V3.2): absorbed-MLA
    //    (q-LoRA + per-head embed_q/unembed_out) + sparse Indexer + deepseek MoE ──
    // Routes through build_deepseek_prefill with q_lora_rank>0 + absorbed_mla=true.
    // The sparse "lightning indexer" is a NO-OP for prefill seq ≤ index_topk (2048)
    // → full attention (exact). Absorbed-MLA path validated by algebraic
    // equivalence vs the kv_b_proj path (examples/glm_dsa_mla_probe.rs, cos 1.0 /
    // err 0). MoE = build_deepseek_moe (noaux_tc sigmoid+bias, shared, ×scaling).
    // 78L/256-expert giant → WiredDeferred. glm4_moe (non-DSA) handled separately.
    if a == "glm_moe_dsa" || a == "glmmoedsa" {
        return hit(
            "rlx-deepseek",
            CoverageStatus::WiredDeferred,
            "GLM-5 (glm_moe_dsa = DeepSeek-V3.2) absorbed-MLA(q-LoRA)+indexer-skip+MoE — build_deepseek_prefill; absorbed-MLA validated cos 1.0 vs kv_b_proj path; 78L/256e giant",
        );
    }

    // ── Kimi-Linear (`kimi_linear`): KDA linear-attn + NoPE-MLA + deepseek MoE ──
    // build_kimi_linear_prefill: KDA (fine-grained per-key-dim gated delta-net,
    // the novel piece — numerically validated in isolation, cos 1.0 + max|err|
    // 1e-8 vs mlx-lm gated_delta_ops) + NoPE-MLA (reuses validated build_deepseek_mla
    // with identity RoPE) + deepseek MoE (validated). Kimi-Linear-48B-A3B → e2e
    // deferred (24GB + 20 unrolled-KDA layers), so WiredDeferred.
    if a == "kimi_linear" || a == "kimilinear" {
        return hit(
            "rlx-kimi-linear",
            CoverageStatus::WiredDeferred,
            "Kimi-Linear KDA(fine-grained gated delta-net)+NoPE-MLA+MoE — build_kimi_linear_prefill; KDA validated cos 1.0 vs mlx-lm, 48B e2e deferred",
        );
    }

    // ── GLM-4.5 MoE (glm4_moe): GQA + partial-RoPE + deepseek-style MoE ──
    // VALIDATED: cos 0.99920 + exact argmax vs mlx-lm on GLM-4.5-Air (106B, 2-bit)
    // via build_glm4moe_prefill. glm4_moe_lite reuses the same builder → same
    // status (both are 100B+, so WiredDeferred — they run, just RAM-bound).
    // glm_moe_dsa (DeepSeek-V3.2 absorbed-MLA + sparse indexer) is claimed above
    // via the rlx-deepseek arm — NOT here (it's not a partial-RoPE GQA model).
    if a == "glm4_moe" || a == "glm4_moe_lite" {
        return hit(
            "rlx-glm4moe",
            CoverageStatus::WiredDeferred,
            "GLM-4.5 MoE — build_glm4moe_prefill (GQA+partial-RoPE+MoE), cos 0.99920 + exact argmax vs mlx-lm (Air-106B 2-bit)",
        );
    }

    // ── LFM2 / LFM2.5 ShortConv+attention hybrid ──
    // Native `build_lfm2_prefill` (ShortConv mixer: in_proj→B·x→depthwise causal
    // conv→C·→out_proj; GQA-attn at full_attn_idxs; SwiGLU; tied head) validated
    // on LFM2-1.2B-4bit: last-token cos 0.99992 + exact argmax vs mlx-lm oracle.
    // lfm2moe (routed experts) not claimed.
    if (a == "lfm2" || a == "lfm" || a == "lfm25" || a == "lfm2_5") && !a.contains("moe") {
        return hit(
            "rlx-lfm",
            CoverageStatus::Validated,
            "LFM2/2.5 ShortConv+attn — build_lfm2_prefill, cos 0.99992 + exact argmax vs mlx-lm",
        );
    }

    // ── Laguna 256-expert MoE VLM (rlx-laguna) ──
    // Full mlx-community path complete + run: HF-config parse + mlx-dir loader
    // (`mlx_load::load_mlx_weights` → PackedMlx under the `language_model.` prefix)
    // + host affine forward (dense/attn `affine_matmul_bt` + stacked-`switch_mlp`
    // routed MoE `affine_expert_swiglu`, both unit-validated cos-exact) +
    // per-head qk-norm→RoPE(YaRN-partial full / default sliding) + softplus
    // `g_proj` gate — attention structure matches the `modeling_laguna.py`
    // reference. E2E RUN on Laguna-XS-2.1-4bit: greedy-generates the correct
    // answer ("The capital of France is Paris."). No external mlx-lm oracle
    // exists for laguna, so this is coherent+factually-correct generation, not an
    // oracle cos-match.
    if a == "laguna" {
        return hit(
            "rlx-laguna",
            CoverageStatus::Validated,
            "Laguna MoE VLM — mlx-dir loader + host affine forward (stacked switch_mlp experts, \
             qk-norm/YaRN-partial-RoPE/softplus g_proj); e2e greedy-generates correct output on Laguna-XS-2.1-4bit",
        );
    }

    None
}

/// Classify a parsed `config.json` for the mlx-community catalog: "does rlx run
/// it, and how". Dedicated-crate coverage wins over the generic verdict.
pub fn classify_coverage(v: &Value) -> ModelCoverage {
    let base = classify_config(v);
    if let Some(d) = dedicated_coverage(&base.arch) {
        return ModelCoverage {
            // A dedicated crate can be BuilderOnly (graph exists, no runner yet)
            // → not actually runnable, so `supported` follows the status.
            supported: d.status.is_supported(),
            via: CoverageVia::Dedicated(d.krate),
            status: d.status,
            reason: d.note.to_string(),
            base,
        };
    }
    if base.supported {
        let reason = base.reason.clone();
        return ModelCoverage {
            supported: true,
            via: CoverageVia::Generic,
            status: CoverageStatus::Validated,
            reason,
            base,
        };
    }
    let reason = base.reason.clone();
    ModelCoverage {
        supported: false,
        via: CoverageVia::None,
        status: CoverageStatus::Unsupported,
        reason,
        base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(model_type: &str) -> Value {
        json!({
            "model_type": model_type,
            "hidden_act": "silu",
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 8192,
            "rms_norm_eps": 1e-6,
        })
    }

    #[test]
    fn generic_dense_decoder_is_generic() {
        let c = classify_coverage(&cfg("qwen3"));
        assert!(c.supported);
        assert_eq!(c.via, CoverageVia::Generic);
        assert_eq!(c.status, CoverageStatus::Validated);
    }

    #[test]
    fn gemma3n_covered_by_dedicated_crate() {
        // Generic classifier rejects gemma3n (AltUp); coverage flips it to
        // supported via rlx-gemma.
        let base = classify_config(&cfg("gemma3n"));
        assert!(!base.supported, "generic must still reject gemma3n");
        let c = classify_coverage(&cfg("gemma3n"));
        assert!(c.supported);
        assert_eq!(c.via, CoverageVia::Dedicated("rlx-gemma"));
    }

    #[test]
    fn gemma4_variants_all_covered() {
        for mt in ["gemma4", "gemma4_unified", "gemma4_assistant"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt} should be covered");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-gemma"), "{mt}");
        }
    }

    #[test]
    fn qwen35_deltanet_covered() {
        for mt in ["qwen3_5", "qwen3_next", "qwen3_5_moe", "qwen3_5_mtp"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt} should be covered");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-qwen35"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
    }

    #[test]
    fn gpt_oss_validated() {
        // Full gpt-oss prefill (sinks + MXFP4 MoE) validated vs mlx-lm (cos 0.99972).
        for mt in ["gpt_oss", "gpt-oss"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt} runs via build_gpt_oss_prefill");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-gpt-oss"), "{mt}");
            assert_eq!(c.status, CoverageStatus::Validated, "{mt}");
        }
    }

    #[test]
    fn glm4_moe_wired() {
        for mt in ["glm4_moe", "glm4_moe_lite"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-glm4moe"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
        // glm_moe_dsa is now claimed via the rlx-deepseek (DSV32 absorbed-MLA) arm.
        assert_eq!(
            dedicated_coverage("glm_moe_dsa").map(|d| d.krate),
            Some("rlx-deepseek")
        );
    }

    #[test]
    fn glm_moe_dsa_wired() {
        // GLM-5 = DeepSeek-V3.2 (absorbed-MLA q-LoRA + sparse indexer + deepseek MoE).
        // build_deepseek_prefill (q_lora_rank>0, absorbed_mla=true); indexer no-op for
        // prefill; absorbed-MLA validated cos 1.0 vs kv_b_proj path. 78L/256e giant.
        for mt in ["glm_moe_dsa", "glmmoedsa"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-deepseek"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
    }

    #[test]
    fn deepseek_mla_validated_and_wired() {
        // MLA+MoE builder validated on V2-Lite-16B; giants wired-deferred (RAM).
        let v2 = classify_coverage(&cfg("deepseek_v2"));
        assert!(v2.supported && v2.status == CoverageStatus::Validated);
        for mt in ["deepseek_v3", "kimi_k2", "kimi_k25"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-deepseek"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
    }

    #[test]
    fn deepseek_v4_wired() {
        // DeepSeek-V4-Flash GA (+MTP, + `deepseek4` GGUF arch): 5-subsystem research
        // arch + DSpark, assembled by build_deepseek_v4_prefill/build_dspark_stage;
        // all cores cos-exact + full forward finite; 3 quant load paths.
        for mt in ["deepseek_v4", "deepseek_v4_mtp", "deepseek4"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-models-core"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
    }

    #[test]
    fn kimi_linear_wired() {
        // Kimi-Linear KDA + NoPE-MLA + MoE. KDA (novel fine-grained gated delta-net)
        // validated in isolation cos 1.0 vs mlx-lm gated_delta_ops; MLA/MoE reuse
        // validated deepseek builders. 48B-A3B e2e deferred → WiredDeferred.
        for mt in ["kimi_linear", "kimilinear"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-kimi-linear"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
        // Must NOT be swallowed by the deepseek/kimi-MLA arm.
        assert_eq!(
            dedicated_coverage("kimi_k2").map(|d| d.krate),
            Some("rlx-deepseek")
        );
    }

    #[test]
    fn minimax_m2_wired() {
        // MiniMax-M2 is a STANDARD transformer (GQA + full qk-norm + partial-RoPE
        // + block_sparse_moe), NOT Lightning/linear attention. build_minimax_prefill
        // reuses the validated partial-RoPE + MoE + qk-norm pieces; tensor names are
        // verified against the mlx-community/MiniMax-M2-4bit index. 230B won't fit
        // 64GB and there is no smaller MiniMax variant, so it stays WiredDeferred.
        for mt in ["minimax", "minimax_m2", "minimax-m2"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-minimax"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
    }

    #[test]
    fn hy_v3_wired() {
        // Hunyuan-V3 = GQA + per-head qk-norm + full-RoPE + deepseek-style MoE
        // (SwiGLU experts + shared expert). build_hy_v3_prefill reuses validated
        // pieces; tensor shapes verified vs mlx-community/Hy3-preview-4bit headers.
        // 80L/192-expert giant → WiredDeferred. Newer than mlx-lm/transformers 5.3.0.
        for mt in ["hy_v3", "hyv3", "hunyuan_v3"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-hunyuan"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
        // Old dense hunyuan (`hunyuan_v1_dense`) is generic-shaped — not claimed here.
        assert!(dedicated_coverage("hunyuan_v1_dense").is_none());
    }

    #[test]
    fn nemotron_h_wired() {
        // Nemotron-H = hybrid Mamba-2 / NoPE-attn / ReLU²-MoE. build_nemotron_h_prefill
        // reuses the validated Op::Mamba2 scan + deepseek MoE router; tensor names
        // verified vs mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 index. All
        // in-scope repos are 30B–120B MoE giants → WiredDeferred (won't fit 64GB).
        for mt in ["nemotron_h", "nemotronh", "nemotron-h"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt}");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-nemotron-h"), "{mt}");
            assert_eq!(c.status, CoverageStatus::WiredDeferred, "{mt}");
        }
        // Plain `nemotron` (dense Llama-shaped) is NOT the hybrid arch — must not
        // be claimed by this arm.
        assert!(dedicated_coverage("nemotron").is_none());
    }

    #[test]
    fn lfm2_validated() {
        // LFM2 ShortConv prefill validated vs mlx-lm oracle (cos 0.99992, exact argmax).
        for mt in ["lfm2", "lfm2_5", "lfm25"] {
            let c = classify_coverage(&cfg(mt));
            assert!(c.supported, "{mt} runnable via rlx-lfm build_lfm2_prefill");
            assert_eq!(c.via, CoverageVia::Dedicated("rlx-lfm"), "{mt}");
            assert_eq!(c.status, CoverageStatus::Validated, "{mt}");
        }
        // MoE variant not claimed.
        assert!(dedicated_coverage("lfm2moe").is_none());
    }

    #[test]
    fn laguna_validated() {
        // Laguna mlx-community path complete: mlx-dir loader + host affine forward
        // (stacked switch_mlp experts unit-validated) → e2e greedy-generates the
        // correct answer on Laguna-XS-2.1-4bit. Now supported/Validated.
        let mut v = cfg("laguna");
        v["num_experts"] = serde_json::json!(256);
        let c = classify_coverage(&v);
        assert!(c.supported, "laguna mlx now e2e-runnable");
        assert_eq!(c.via, CoverageVia::Dedicated("rlx-laguna"));
        assert_eq!(c.status, CoverageStatus::Validated);
    }

    #[test]
    fn gemma4moe_not_claimed() {
        // A4B routed-expert Gemma is genuinely unwired — must NOT be claimed by
        // the gemma arm just because the name starts with "gemma4".
        assert!(dedicated_coverage("gemma4moe").is_none());
        assert!(dedicated_coverage("gemma4_moe").is_none());
    }

    #[test]
    fn plain_gemma3_stays_generic() {
        // gemma3 (dense GeGLU) is generic-supported; must NOT be claimed by the
        // dedicated table (that would route it away from the working path).
        assert!(dedicated_coverage("gemma3").is_none());
        assert!(dedicated_coverage("gemma2").is_none());
    }

    #[test]
    fn unwired_family_stays_unsupported() {
        let c = classify_coverage(&cfg("rwkv"));
        assert!(!c.supported);
        assert_eq!(c.via, CoverageVia::None);
        assert_eq!(c.status, CoverageStatus::Unsupported);
    }
}
