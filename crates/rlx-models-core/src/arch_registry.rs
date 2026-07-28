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

//! Architecture registry (plan #82).
//!
//! Borrowed from MAX's `@register_pipelines_model("name")` decorator
//! pattern. Models register a string identifier + a small spec at
//! crate init; downstream code looks up `archs::lookup("bert")`
//! instead of importing a specific builder by name.
//!
//! Why a registry instead of plain imports?
//!   - Decouples burnembed (and other consumers) from rlx-models's
//!     concrete builder names. Adding a model = registering it,
//!     not editing every consumer's `match arch { ... }`.
//!   - Lets a third-party crate register its own arch without
//!     touching rlx-models.
//!
//! How registration happens:
//!   - Built-in archs: [`register_all`] (called once at startup, or
//!     idempotently any time).
//!   - Third-party: call [`register`] directly from your crate's
//!     init.
//!
//! No procedural-macro auto-registration today — Rust's stable
//! toolchain doesn't have a clean equivalent of MAX's decorator
//! without pulling in `linkme` or `inventory`. Add the macro when
//! a real consumer wants it.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Architecture family — coarse grouping for selection / dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchFamily {
    /// BERT-style encoder (padding mask, no RoPE).
    BertEncoder,
    /// NomicBERT-style encoder (padding mask + RoPE).
    NomicBertEncoder,
    /// Vision encoder (NomicVision, DINOv2 today).
    VisionEncoder,
    /// Segmentation model (SAM v1).
    Segmenter,
    /// Causal autoregressive language model (Qwen3, future Llama, etc.).
    CausalLLM,
    /// Speech / audio encoder (Wav2Vec2-BERT, etc.).
    SpeechEncoder,
    /// Image diffusion / flow-matching denoiser (FLUX.2, etc.).
    Diffusion,
    /// Future: MoE, SSM, etc.
    Other,
}

/// Registered architecture metadata. Builders are not stored here —
/// the cross-crate function pointer types vary by family. Consumers
/// look up `ArchSpec` by name to learn *which* family to use, then
/// call the family-typed builder directly.
#[derive(Debug, Clone)]
pub struct ArchSpec {
    /// Canonical name (e.g. `"bert"`, `"nomic-bert"`, `"nomic-vision"`).
    pub name: &'static str,
    pub family: ArchFamily,
    /// One-line description for `--list-archs` style tools.
    pub description: &'static str,
}

struct Registry {
    map: RwLock<HashMap<&'static str, ArchSpec>>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Registry {
        map: RwLock::new(HashMap::new()),
    })
}

/// Register one architecture. Idempotent — calling twice with the
/// same name replaces the prior entry, mirroring `register_backend`.
pub fn register(spec: ArchSpec) {
    let r = registry();
    let mut m = r.map.write().expect("arch registry poisoned");
    m.insert(spec.name, spec);
}

/// Look up an architecture spec by canonical name. Returns `None`
/// if the name isn't registered (caller can list `registered()`).
pub fn lookup(name: &str) -> Option<ArchSpec> {
    let r = registry();
    let m = r.map.read().expect("arch registry poisoned");
    m.get(name).cloned()
}

/// Snapshot of every currently-registered arch, sorted by name.
pub fn registered() -> Vec<ArchSpec> {
    let r = registry();
    let m = r.map.read().expect("arch registry poisoned");
    let mut v: Vec<ArchSpec> = m.values().cloned().collect();
    v.sort_by_key(|s| s.name);
    v
}

/// Register every built-in architecture. Idempotent — call from your
/// startup path (or rely on a consumer to call it; safe to invoke
/// multiple times).
pub fn register_all() {
    crate::model_registry::ensure_builtin_gguf_models();
    register(ArchSpec {
        name: "bert",
        family: ArchFamily::BertEncoder,
        description: "BERT-style encoder (MiniLM / BGE / mpnet).",
    });
    register(ArchSpec {
        name: "nomic-bert",
        family: ArchFamily::NomicBertEncoder,
        description: "NomicBERT encoder with RoPE + SwiGLU FFN.",
    });
    register(ArchSpec {
        name: "nomic-vision",
        family: ArchFamily::VisionEncoder,
        description: "NomicVision image encoder.",
    });
    register(ArchSpec {
        name: "dinov2",
        family: ArchFamily::VisionEncoder,
        description: "DINOv2 ViT (Meta) — self-supervised image encoder; optional register tokens.",
    });
    register(ArchSpec {
        name: "vjepa2",
        family: ArchFamily::VisionEncoder,
        description: "V-JEPA2 (Meta) — video ViT encoder + JEPA predictor + attentive pooler. ViT-G @ 384² / 64 frames.",
    });
    register(ArchSpec {
        name: "sam",
        family: ArchFamily::Segmenter,
        description: "Segment Anything v1 (Meta) — ViT-B/L/H encoder. Phase 1 ships encoder; decoder follows.",
    });
    register(ArchSpec {
        name: "sam2",
        family: ArchFamily::Segmenter,
        description: "Segment Anything 2 (Meta) — Hiera image encoder + FPN. Phase 1 ships encoder + neck; prompt/mask decoder + memory follow.",
    });
    register(ArchSpec {
        name: "sam3",
        family: ArchFamily::Segmenter,
        description: "Segment Anything 3 (Meta) — concept-conditioned image segmentation and video tracking.",
    });
    register(ArchSpec {
        name: "qwen3",
        family: ArchFamily::CausalLLM,
        description: "Qwen3 (Alibaba) — GQA + QK-norm + SwiGLU dense causal LM. Phase 1 ships prefill graph; KV cache / lm_head / sampling / MTP follow.",
    });
    register(ArchSpec {
        name: "llada2",
        family: ArchFamily::Diffusion,
        description: "LLaDA2 MoE (inclusionAI) — block diffusion LLM with sigmoid group-limited TopK router and TIDE predictive expert offload.",
    });
    register(ArchSpec {
        name: "llama32",
        family: ArchFamily::CausalLLM,
        description: "LLaMA-3.2 (Meta) — GQA + Llama 3 RoPE scaling + SwiGLU, no QK-norm. Safetensors + GGUF via rlx-run llama32.",
    });
    register(ArchSpec {
        name: "nanbeige",
        family: ArchFamily::CausalLLM,
        description: "Nanbeige4.2 — Looped Transformer (shared-weight layer reuse + per-loop KV); Llama-shaped via rlx-nanbeige / rlx-llama32.",
    });
    register(ArchSpec {
        name: "gemma",
        family: ArchFamily::CausalLLM,
        description: "Gemma / Gemma 2 (Google) — GQA + RoPE + GeGLU + embed scaling + tied weights. Safetensors + GGUF via rlx-run gemma.",
    });
    register(ArchSpec {
        name: "laguna",
        family: ArchFamily::CausalLLM,
        description: "Laguna (Poolside) — hybrid SWA MoE; packed GGUF generate (KV cache, optional Metal/MLX) via rlx-laguna; F32 expand off by default.",
    });
    register(ArchSpec {
        name: "wav2vec2-bert",
        family: ArchFamily::SpeechEncoder,
        description: "Wav2Vec2-BERT / W2v-BERT 2.0 (Meta) — Conformer speech encoder on log-mel features.",
    });
    register(ArchSpec {
        name: "flux2",
        family: ArchFamily::Diffusion,
        description: "FLUX.2 (Black Forest Labs) — rectified-flow image denoiser; CPU native or compiled HIR on metal/mlx/cuda/rocm/gpu/vulkan.",
    });
    register(ArchSpec {
        name: "neutts",
        family: ArchFamily::Other,
        description: "NeuTTS (Neuphonic) — voice-cloning TTS: GGUF backbone + NeuCodec FSQ/Vocos decoder.",
    });
    register(ArchSpec {
        name: "orpheus",
        family: ArchFamily::Other,
        description: "Orpheus TTS — Llama-3B speech LM (GGUF) + SNAC 24 kHz codec decoder.",
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        register_all();
        let bert = lookup("bert").expect("bert should be registered");
        assert_eq!(bert.family, ArchFamily::BertEncoder);
        assert!(lookup("nonexistent").is_none());
    }

    #[test]
    fn registered_is_sorted() {
        register_all();
        let names: Vec<&str> = registered().into_iter().map(|s| s.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
}
