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

//! Extensible GGUF / HF model-family registry.
//!
//! Single source of truth for:
//! - GGUF `general.architecture` → runner / [`GgufModelFamily`]
//! - HF `config.json` `model_type` → runner
//! - User-facing crate hints (`gguf_runner_hint`, auto-dispatch errors)
//!
//! Built-ins are registered by [`ensure_builtin_gguf_models`]. Third-party
//! crates call [`register_gguf_model`] at init (same pattern as
//! [`crate::weight_registry::register_weight_format`] and
//! [`crate::gguf_resolve::register_gguf_tensor_resolver`]).

use std::collections::HashMap;
use std::sync::{Once, OnceLock, RwLock};

use crate::gguf_support::GgufModelFamily;

/// One registered model family (GGUF arch tags and/or HF `model_type`s).
#[derive(Debug, Clone, Copy)]
pub struct GgufModelRegistration {
    /// Stable id for listing (e.g. `"qwen3"`, `"phi"`, `"flux2"`).
    pub id: &'static str,
    /// GGUF `general.architecture` values (exact match, case-sensitive as stored).
    pub arches: &'static [&'static str],
    /// HF `config.json` `model_type` values for safetensors auto-dispatch.
    pub hf_model_types: &'static [&'static str],
    /// Short `register_cli` / `rlx-run auto` name. `None` = sniffable for hints
    /// only (no multiplexer runner yet), e.g. embedding GGUFs.
    pub runner: Option<&'static str>,
    /// LM family for [`crate::gguf_support::assert_gguf_family`], when applicable.
    pub family: Option<GgufModelFamily>,
    /// One-line crate / usage hint for CLI and errors.
    pub hint: &'static str,
}

struct Registry {
    by_arch: HashMap<&'static str, GgufModelRegistration>,
    by_hf: HashMap<&'static str, GgufModelRegistration>,
    by_id: HashMap<&'static str, GgufModelRegistration>,
}

fn registry() -> &'static RwLock<Registry> {
    static R: OnceLock<RwLock<Registry>> = OnceLock::new();
    R.get_or_init(|| {
        RwLock::new(Registry {
            by_arch: HashMap::new(),
            by_hf: HashMap::new(),
            by_id: HashMap::new(),
        })
    })
}

/// Register (or replace) a model family. Idempotent per `id` and per arch /
/// `model_type` key — later registrations overwrite earlier ones for the same
/// key (mirrors weight-format registration).
pub fn register_gguf_model(reg: GgufModelRegistration) {
    let mut g = registry().write().expect("model registry poisoned");
    g.by_id.insert(reg.id, reg);
    for &arch in reg.arches {
        g.by_arch.insert(arch, reg);
    }
    for &mt in reg.hf_model_types {
        g.by_hf.insert(mt, reg);
    }
}

/// Look up a registration by GGUF architecture tag.
pub fn lookup_gguf_arch(arch: &str) -> Option<GgufModelRegistration> {
    ensure_builtin_gguf_models();
    registry()
        .read()
        .expect("model registry poisoned")
        .by_arch
        .get(arch)
        .copied()
}

/// Look up a registration by HF `model_type`.
pub fn lookup_hf_model_type(model_type: &str) -> Option<GgufModelRegistration> {
    ensure_builtin_gguf_models();
    registry()
        .read()
        .expect("model registry poisoned")
        .by_hf
        .get(model_type)
        .copied()
}

/// Look up by registration `id`.
pub fn lookup_gguf_model_id(id: &str) -> Option<GgufModelRegistration> {
    ensure_builtin_gguf_models();
    registry()
        .read()
        .expect("model registry poisoned")
        .by_id
        .get(id)
        .copied()
}

/// `rlx-run` runner name for a GGUF arch, if this family is auto-dispatchable.
pub fn runner_for_gguf_arch(arch: &str) -> Option<&'static str> {
    lookup_gguf_arch(arch).and_then(|r| r.runner)
}

/// `rlx-run` runner name for an HF `model_type`.
pub fn runner_for_hf_model_type(model_type: &str) -> Option<&'static str> {
    lookup_hf_model_type(model_type).and_then(|r| r.runner)
}

/// User-facing hint for a GGUF arch (crate + brief usage).
pub fn hint_for_gguf_arch(arch: &str) -> Option<&'static str> {
    lookup_gguf_arch(arch).map(|r| r.hint)
}

/// Snapshot of every registered model id (sorted).
pub fn registered_gguf_models() -> Vec<GgufModelRegistration> {
    ensure_builtin_gguf_models();
    let g = registry().read().expect("model registry poisoned");
    let mut v: Vec<GgufModelRegistration> = g.by_id.values().copied().collect();
    v.sort_by_key(|r| r.id);
    v
}

/// Map GGUF arch → [`GgufModelFamily`] (LM families only).
pub fn family_for_gguf_arch(arch: &str) -> Option<GgufModelFamily> {
    lookup_gguf_arch(arch).and_then(|r| r.family)
}

static BUILTINS: Once = Once::new();

/// Register every built-in GGUF / HF mapping. Safe to call repeatedly.
pub fn ensure_builtin_gguf_models() {
    BUILTINS.call_once(register_builtin_gguf_models);
}

fn register_builtin_gguf_models() {
    // ── Causal LMs ──────────────────────────────────────────────────────────
    register_gguf_model(GgufModelRegistration {
        id: "qwen3",
        arches: &["qwen3", "qwen2", "qwen25", "qwen2_5", "qwen2vl"],
        // qwen2 safetensors deliberately omitted — layout needs bias / no QK-norm.
        hf_model_types: &[
            "qwen3",
            "qwen3_moe",
            "qwen3moe",
            "qwen25",
            "qwen2_5",
            "qwen2.5",
            "qwen251",
            "qwen2_5_1",
        ],
        runner: Some("qwen3"),
        family: Some(GgufModelFamily::Qwen3),
        hint: "rlx-qwen3 (use `--packed` for large K-quant GGUF)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "qwen35",
        arches: &["qwen35", "qwen35moe", "qwen36", "qwen36moe"],
        hf_model_types: &[
            "qwen35",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_mtp",
            "qwen35_moe",
            "qwen35moe",
            "qwen3_next",
            "qwen3next",
            "qwen36",
            "qwen3_6",
            "qwen36_moe",
            "qwen36moe",
        ],
        runner: Some("qwen35"),
        family: Some(GgufModelFamily::Qwen35),
        hint: "rlx-qwen35 (mlx-community 4-bit affine dirs load via MlxLoader; `--packed` for GGUF)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "qwen3-vl",
        arches: &["qwen3vl", "qwen3vlmoe", "qwen3_vl", "qwen3-vl"],
        hf_model_types: &[],
        runner: Some("qwen3-vl"),
        family: Some(GgufModelFamily::Qwen3Vl),
        hint: "rlx-qwen3-vl (`Qwen3VlRunner::builder().weights(...).mmproj(...)`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "llama32",
        arches: &["llama"],
        hf_model_types: &["llama", "llama2", "llama3"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-llama32 (`--packed` for large K-quant GGUF)",
    });
    // Phi shares Llama32 weights/assert path but has its own CLI runner.
    register_gguf_model(GgufModelRegistration {
        id: "phi",
        arches: &["phi3", "phi4"],
        hf_model_types: &["phi3", "phi4"],
        runner: Some("phi"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-phi (Llama-shaped via rlx-llama32)",
    });
    // IBM Granite dense — Llama-shaped + embedding/residual/attention/logit
    // scalar multipliers, applied in the packed builder (validated on the
    // packed path for all devices). granitemoe / granitehybrid still need
    // MoE / Mamba blocks and stay in the unimplemented table.
    register_gguf_model(GgufModelRegistration {
        id: "granite",
        arches: &["granite"],
        hf_model_types: &["granite"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-granite / rlx-llama32 (dense Granite: embed/residual/attn/logit multipliers)",
    });
    // LG ExaOne 3.x — Llama-shaped decoder with NeoX RoPE. Runs via the
    // llama32 runner (packed GGUF path). ExaOne 4.0 (`exaone4`, hybrid
    // sliding/global + QK-norm) is NOT covered here.
    register_gguf_model(GgufModelRegistration {
        id: "exaone",
        arches: &["exaone"],
        hf_model_types: &["exaone"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-llama32 (ExaOne 3.x: Llama-shaped, NeoX RoPE)",
    });
    // Dense Llama-shaped arches with a small per-arch structural delta wired into
    // the packed rlx-llama32 builder (norm placement/kind, FFN shape, residual).
    // AllenAI OLMo-2 — post-sublayer RMSNorm + full-projection Q/K RMSNorm.
    register_gguf_model(GgufModelRegistration {
        id: "olmo2",
        arches: &["olmo2", "olmo"],
        hf_model_types: &["olmo2", "olmo"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-llama32 (OLMo-2: post-norm placement + Q/K RMSNorm)",
    });
    // NVIDIA Nemotron (dense) — LayerNorm(+bias), gate-less squared-ReLU FFN.
    register_gguf_model(GgufModelRegistration {
        id: "nemotron",
        arches: &["nemotron"],
        hf_model_types: &["nemotron"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-llama32 (dense Nemotron: LayerNorm + squared-ReLU gate-less FFN)",
    });
    // Cohere / Command-R / Cohere2 NOT registered: the parallel-residual path is
    // coded in rlx-llama32 but cohere2 (command-r7b) still outputs garbage
    // (needs per-layer sliding/full-attention + NoPE-on-global-layers). Kept in
    // `known_unimplemented_arch` until validated.
    // GLM-4 (four RMSNorms) / ChatGLM / GLM-Edge (two norms, fused gate∥up).
    register_gguf_model(GgufModelRegistration {
        id: "glm",
        arches: &["glm4", "chatglm"],
        hf_model_types: &["glm4", "chatglm", "glm"],
        runner: Some("llama32"),
        family: Some(GgufModelFamily::Llama32),
        hint: "rlx-llama32 (GLM-4 / ChatGLM: partial RoPE, GLM norm placement, fused gate∥up)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "mistral",
        arches: &["mistral3", "mistral4"],
        hf_model_types: &["mistral3", "mistral4", "ministral", "ministral3", "pixtral"],
        runner: Some("mistral"),
        family: Some(GgufModelFamily::Mistral),
        hint: "rlx-mistral (`--packed` for large K-quant GGUF)",
    });
    // Llama-4 (MoE + native vision) and Llama-3.2-Vision (mllama, cross-attn) are
    // safetensors-checkpoint runners (`from_checkpoint(dir, device)`), not
    // `LmRunner`/GGUF — `auto_runner` can't box them. Registered so sniffing
    // resolves the arch and points at the right builder instead of "unknown".
    register_gguf_model(GgufModelRegistration {
        id: "llama4",
        arches: &["llama4"],
        hf_model_types: &["llama4"],
        runner: Some("llama4"),
        family: None,
        hint: "rlx-llama4 (`Llama4Runner`/`Llama4VlRunner::from_checkpoint(dir, device)`; \
               MoE + native vision — not an LmRunner/GGUF path)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "mllama",
        arches: &["mllama"],
        hf_model_types: &["mllama"],
        runner: Some("mllama"),
        family: None,
        hint: "rlx-mllama (`MllamaRunner::from_checkpoint(dir, device)`; \
               Llama-3.2-Vision cross-attention — not an LmRunner/GGUF path)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "gemma",
        arches: &[
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3n",
            "gemma4",
            "gemma4moe",
            "gemma4_unified",
        ],
        // gemma4moe HF type stays unimplemented until routing is validated.
        hf_model_types: &[
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3n",
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
            "gemma4_unified_text",
        ],
        runner: Some("gemma"),
        family: Some(GgufModelFamily::Gemma),
        hint: "rlx-gemma (`--packed` for large K-quant GGUF)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "lfm",
        arches: &["lfm2", "lfm", "lfm25", "lfm2_5"],
        hf_model_types: &["lfm2", "lfm", "lfm25", "lfm2_5"],
        runner: Some("lfm"),
        family: Some(GgufModelFamily::Lfm),
        hint: "rlx-lfm (`LfmRunner::builder().weights`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "inkling",
        arches: &["inkling", "inkling_mm_model"],
        hf_model_types: &["inkling"],
        runner: Some("inkling"),
        family: Some(GgufModelFamily::Inkling),
        hint: "rlx-inkling (`--weights` GGUF sniff; RLX eager on --synth/--fixture)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "laguna",
        arches: &["laguna"],
        hf_model_types: &["laguna"],
        runner: Some("laguna"),
        family: Some(GgufModelFamily::Laguna),
        hint: "rlx-laguna (packed mmap generate; F32 expand off by default — \
               RLX_LAGUNA_ALLOW_F32_EXPAND=1 to opt in)",
    });

    // ── Vision / diffusion / speech (auto-dispatch runners) ─────────────────
    register_gguf_model(GgufModelRegistration {
        id: "flux2",
        arches: &["flux"],
        hf_model_types: &["flux", "flux2"],
        runner: Some("flux2"),
        family: None,
        hint: "rlx-flux2 denoiser (`Flux2Runner::builder().weights`) — VAE/TE stay safetensors",
    });
    register_gguf_model(GgufModelRegistration {
        id: "dinov2",
        arches: &["dinov2"],
        hf_model_types: &["dinov2", "dinov2_with_registers"],
        runner: Some("dinov2"),
        family: None,
        hint: "rlx-dinov2 (`DinoV2Runner::builder().weights`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "vjepa2",
        arches: &["vjepa2", "vjepa"],
        hf_model_types: &["vjepa2", "vjepa"],
        runner: Some("vjepa2"),
        family: None,
        hint: "rlx-vjepa2 (`Vjepa2Runner::builder().weights`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "sam1",
        arches: &["sam", "mobile-sam"],
        hf_model_types: &["sam", "sam_vit", "mobile-sam", "mobile_sam"],
        runner: Some("sam1"),
        family: None,
        hint: "rlx-sam (`Sam::from_safetensors_on`) — MobileSAM uses `mobile-sam` arch",
    });
    register_gguf_model(GgufModelRegistration {
        id: "sam2",
        arches: &["sam2"],
        hf_model_types: &["sam2"],
        runner: Some("sam2"),
        family: None,
        hint: "rlx-sam2 (`Sam2::from_safetensors_on`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "sam3",
        arches: &["sam3"],
        hf_model_types: &["sam3"],
        runner: Some("sam3"),
        family: None,
        hint: "rlx-sam3 (`Sam3::from_checkpoint_on`)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "wav2vec2-bert",
        arches: &["w2v-bert", "wav2vec2", "wav2vec"],
        hf_model_types: &["wav2vec2-bert", "wav2vec2_bert", "w2v-bert", "w2v_bert"],
        runner: Some("wav2vec2-bert"),
        family: None,
        hint: "rlx-wav2vec2-bert (`Wav2Vec2BertRunner::builder().weights`; keep config.json beside GGUF)",
    });
    register_gguf_model(GgufModelRegistration {
        id: "whisper",
        arches: &[],
        hf_model_types: &["whisper"],
        runner: Some("whisper"),
        family: None,
        hint: "rlx-whisper",
    });
    register_gguf_model(GgufModelRegistration {
        id: "minimax-m3",
        arches: &["minimax-m3", "minimax_m3"],
        hf_model_types: &["minimax_m3_vl", "minimax_m3"],
        runner: Some("minimax-m3"),
        family: None,
        hint: "rlx-minimax m3 (MSA block-sparse MoE; text prefill runner + vision tower)",
    });

    // ── Hint-only (no rlx-run auto runner today) ─────────────────────────────
    register_gguf_model(GgufModelRegistration {
        id: "embed",
        arches: &["bert", "modern-bert", "nomic-bert", "nomic-bert-moe"],
        hf_model_types: &[],
        runner: None,
        family: None,
        hint: "rlx-embed (`RlxEmbed::from_weights`)",
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_map_lm_arches() {
        ensure_builtin_gguf_models();
        assert_eq!(family_for_gguf_arch("qwen3"), Some(GgufModelFamily::Qwen3));
        assert_eq!(
            family_for_gguf_arch("qwen35moe"),
            Some(GgufModelFamily::Qwen35)
        );
        assert_eq!(
            family_for_gguf_arch("llama"),
            Some(GgufModelFamily::Llama32)
        );
        assert_eq!(family_for_gguf_arch("phi3"), Some(GgufModelFamily::Llama32));
        assert_eq!(runner_for_gguf_arch("phi3"), Some("phi"));
        // Granite dense + ExaOne 3.x route to the llama32 runner.
        assert_eq!(runner_for_gguf_arch("granite"), Some("llama32"));
        assert_eq!(
            family_for_gguf_arch("granite"),
            Some(GgufModelFamily::Llama32)
        );
        assert_eq!(runner_for_gguf_arch("exaone"), Some("llama32"));
        // MoE / hybrid Granite are NOT wired here.
        assert_eq!(runner_for_gguf_arch("granitemoe"), None);
        assert_eq!(runner_for_gguf_arch("granitehybrid"), None);
        assert_eq!(
            family_for_gguf_arch("mistral3"),
            Some(GgufModelFamily::Mistral)
        );
        assert_eq!(runner_for_gguf_arch("mistral3"), Some("mistral"));
        assert_eq!(
            family_for_gguf_arch("laguna"),
            Some(GgufModelFamily::Laguna)
        );
        assert_eq!(runner_for_gguf_arch("flux"), Some("flux2"));
        assert_eq!(runner_for_gguf_arch("bert"), None);
        assert!(hint_for_gguf_arch("bert").unwrap().contains("rlx-embed"));
        assert_eq!(runner_for_hf_model_type("whisper"), Some("whisper"));
        assert_eq!(runner_for_hf_model_type("gemma4"), Some("gemma"));
        assert_eq!(runner_for_hf_model_type("gemma4moe"), None);
        // Qwen3.5 / Qwen3-Next mlx-community model_types route to the qwen35 runner.
        assert_eq!(runner_for_hf_model_type("qwen3_5"), Some("qwen35"));
        assert_eq!(runner_for_hf_model_type("qwen3_next"), Some("qwen35"));
        assert_eq!(runner_for_hf_model_type("qwen3_5_moe"), Some("qwen35"));
        assert_eq!(runner_for_hf_model_type("qwen3_5_mtp"), Some("qwen35"));
        assert!(lookup_gguf_arch("clip").is_none());
    }

    #[test]
    fn third_party_can_register() {
        ensure_builtin_gguf_models();
        register_gguf_model(GgufModelRegistration {
            id: "test-arch-xyz",
            arches: &["test_arch_xyz"],
            hf_model_types: &["test_arch_xyz_hf"],
            runner: Some("test-arch-xyz"),
            family: None,
            hint: "test only",
        });
        assert_eq!(runner_for_gguf_arch("test_arch_xyz"), Some("test-arch-xyz"));
        assert_eq!(
            runner_for_hf_model_type("test_arch_xyz_hf"),
            Some("test-arch-xyz")
        );
    }
}
