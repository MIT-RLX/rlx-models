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

//! High-level runner API — re-exported from per-model crates.
//!
//! Prefer depending on a specific model crate (`rlx-qwen3`, …) and
//! its `rlx-<family>` binary when you only need one family.

pub use crate::sam_runner::{SamArch, SamPredictionAny, SamRunner, SamRunnerBuilder};
pub use rlx_cli::{
    AssembledTurn, ChatMessage, ChatTemplate, ChatTemplateSource, CompatSource,
    CompatibilityReport, CompatibilityStatus, GgufRequiredFields, LmRunner, MediaSource,
    ModelRunner, MtmdContext, MtmdTurn, SniffedFrom, SniffedRunner, UnimplementedArch,
    WeightFormat, arch_runner_name, auto_chat_template, auto_dispatch, auto_runner_name,
    auto_sniff, check_hf_repo, check_path, debug_resolve_name, dispatch, dispatch_help,
    known_unimplemented_arch, known_unimplemented_keys, list_mtp_keys, looks_like_hf_repo,
    model_type_runner_name, open_gguf_loader, open_loader, open_loader_resolved,
    open_loader_with_format, register_cli, register_runner, registered_runners, run_auto,
    run_check, run_inspect, run_registered,
};
pub use rlx_dinov2::{DinoV2Output, DinoV2Runner, DinoV2RunnerBuilder, DinoV2Variant};
pub use rlx_flux2::{Flux2Output, Flux2Runner, Flux2RunnerBuilder};
pub use rlx_gemma::{GemmaConfigSource, GemmaRunner, GemmaRunnerBuilder};
pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};
pub use rlx_qwen3::{Precision, Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder};
pub use rlx_qwen35::{Qwen35ConfigSource, Qwen35Runner, Qwen35RunnerBuilder};
#[cfg(feature = "uni2")]
pub use rlx_uni2::{Uni2Output, Uni2Runner, Uni2RunnerBuilder};
pub use rlx_vjepa2::{
    Vjepa2Output, Vjepa2PoolOutput, Vjepa2PredictOutput, Vjepa2Runner, Vjepa2RunnerBuilder,
};
pub use rlx_wav2vec2_bert::{Wav2Vec2BertRunner, Wav2Vec2BertRunnerBuilder};

/// Back-compat alias.
pub type ConfigSource = Qwen3ConfigSource;

use anyhow::{Result, bail};
use std::path::Path;

/// Sniff `path` for its GGUF / safetensors arch and return a boxed
/// runner that implements [`LmRunner`]. The factory uses the existing
/// [`auto_sniff`] arch-dispatch and constructs the per-family runner
/// via its default builder.
///
/// Today this covers the four `text` LM families with a stable
/// `predict_logits` API: `qwen3`, `qwen35`, `gemma`, `llama32`. Other
/// families (vision-language, diffusion, embed) don't fit the
/// `LmRunner` shape and return an error here. They keep their
/// per-family builders.
///
/// PLAN.md M3. The `LmRunner` trait gained a default `generate(..)`
/// in M8, so a boxed runner from this function can stream tokens too.
pub fn auto_runner(path: &Path) -> Result<Box<dyn LmRunner>> {
    auto_runner_with_mmproj(path, None)
}

/// Same as [`auto_runner`] but also attaches an mmproj vision encoder
/// when the model family supports multimodal prefill.
///
/// Supported with mmproj today:
/// - `qwen35` / `qwen36` (plain `Qwen35Runner`; MTP SpecRunner is skipped when
///   mmproj is set so vision wins over speculative decode)
/// - `gemma` (Gemma 4 multimodal projector)
/// - `qwen3-vl` (feature `qwen3-vl`)
/// - `lfm` with mmproj (feature `lfm-vl`)
/// - `mistral` / Ministral with mmproj (feature `mistral-vl`, Pixtral projector)
///
/// For other families, a non-`None` `mmproj` **errors** instead of being
/// silently ignored — callers that want text-only should pass `None`.
pub fn auto_runner_with_mmproj(path: &Path, mmproj: Option<&Path>) -> Result<Box<dyn LmRunner>> {
    let sniff = auto_sniff(path)?;
    let weights = sniff.path.as_path();
    // Packed-K-quant auto-detection is now inside each runner's
    // `.build()` (matches llama.cpp's behaviour — K-quant tensors stay
    // packed in memory, never materialise to a dense F32 matrix).
    let runner: Box<dyn LmRunner> = match sniff.runner_name {
        "qwen3" => {
            if mmproj.is_some() {
                bail!(
                    "auto_runner_with_mmproj: runner `qwen3` does not attach mmproj — \
                     use a `qwen3vl*` GGUF with the qwen3-vl runner, or omit mmproj for text-only"
                );
            }
            Box::new(Qwen3Runner::builder().weights(weights).build()?)
        }
        "qwen35" => {
            // When mmproj is requested, prefer plain Qwen35Runner (vision) over
            // MTP SpecRunner — speculative decode + mmproj is not wired together.
            let want_mtp = mmproj.is_none() && gguf_has_mtp_heads(weights).unwrap_or(false);
            if want_mtp {
                Box::new(
                    rlx_qwen35::Qwen35SpecRunner::builder()
                        .weights(weights)
                        .build()?,
                )
            } else {
                let mut b = Qwen35Runner::builder().weights(weights);
                if let Some(mp) = mmproj {
                    b = b.mmproj(mp);
                }
                Box::new(b.build()?)
            }
        }
        "gemma" => {
            let mut b = GemmaRunner::builder().weights(weights);
            if let Some(mp) = mmproj {
                b = b.mmproj(mp);
            }
            Box::new(b.build()?)
        }
        "llama32" => {
            if mmproj.is_some() {
                bail!(
                    "auto_runner_with_mmproj: runner `llama32` has no multimodal path — omit mmproj"
                );
            }
            Box::new(Llama32Runner::builder().weights(weights).build()?)
        }
        "mistral" => {
            if let Some(mp) = mmproj {
                #[cfg(feature = "mistral-vl")]
                {
                    Box::new(
                        rlx_mistral_vl::MistralVlRunner::builder()
                            .weights(weights)
                            .mmproj(mp)
                            .build()?,
                    ) as Box<dyn LmRunner>
                }
                #[cfg(not(feature = "mistral-vl"))]
                {
                    let _ = mp;
                    bail!(
                        "auto_runner_with_mmproj: mistral + mmproj requires feature `mistral-vl` \
                         (enable rlx-models/mistral-vl)"
                    );
                }
            } else {
                #[cfg(feature = "mistral")]
                {
                    Box::new(
                        rlx_mistral::MistralRunner::builder()
                            .weights(weights)
                            .build()?,
                    ) as Box<dyn LmRunner>
                }
                #[cfg(not(feature = "mistral"))]
                {
                    bail!(
                        "auto_runner: runner `mistral` requires feature `mistral` \
                         (enable rlx-models/mistral)"
                    );
                }
            }
        }
        "lfm" => {
            if let Some(mp) = mmproj {
                #[cfg(feature = "lfm-vl")]
                {
                    Box::new(
                        rlx_lfm_vl::LfmVlRunner::builder()
                            .weights(weights)
                            .mmproj(mp)
                            .build()?,
                    ) as Box<dyn LmRunner>
                }
                #[cfg(not(feature = "lfm-vl"))]
                {
                    let _ = mp;
                    bail!(
                        "auto_runner_with_mmproj: LFM + mmproj requires feature `lfm-vl` \
                         (enable rlx-models/lfm-vl)"
                    );
                }
            } else {
                Box::new(rlx_lfm::LfmRunner::builder().weights(weights).build()?)
            }
        }
        "qwen3-vl" => {
            #[cfg(feature = "qwen3-vl")]
            {
                let mut b = rlx_qwen3_vl::Qwen3VlRunner::builder().weights(weights);
                if let Some(mp) = mmproj {
                    b = b.mmproj(mp);
                }
                Box::new(b.build()?) as Box<dyn LmRunner>
            }
            #[cfg(not(feature = "qwen3-vl"))]
            {
                let _ = mmproj;
                bail!(
                    "auto_runner_with_mmproj: runner `qwen3-vl` requires feature `qwen3-vl` \
                     (enable rlx-models/qwen3-vl)"
                );
            }
        }
        // Llama-4 / Llama-3.2-Vision are safetensors-checkpoint, native-multimodal
        // runners without an `LmRunner` impl — `auto_runner` can't box them. Point
        // the caller at the right builder instead of the generic "no LmRunner".
        "llama4" => {
            let _ = mmproj;
            bail!(
                "auto_runner: `llama4` is a safetensors-checkpoint runner (MoE + native \
                 vision), not an LmRunner/GGUF path — build it directly with \
                 `rlx_llama4::Llama4Runner::from_checkpoint(dir, device)` (or `Llama4VlRunner` \
                 for image input)"
            );
        }
        "mllama" => {
            let _ = mmproj;
            bail!(
                "auto_runner: `mllama` (Llama-3.2-Vision, cross-attention) is a \
                 safetensors-checkpoint runner, not an LmRunner/GGUF path — build it directly \
                 with `rlx_mllama::MllamaRunner::from_checkpoint(dir, device)`"
            );
        }
        other => {
            if mmproj.is_some() {
                bail!(
                    "auto_runner_with_mmproj: runner `{other}` (sniffed from {:?}) does not \
                     support mmproj — omit mmproj for text-only, or use a supported VL family",
                    sniff.from
                );
            }
            bail!(
                "auto_runner: runner `{other}` (sniffed from {:?}) has no `LmRunner` impl yet — \
                 use its per-family builder directly",
                sniff.from
            )
        }
    };
    Ok(runner)
}

/// Peek at a GGUF's `<arch>.nextn_predict_layers` metadata key without
/// fully loading weights. Returns `Ok(true)` when the file declares ≥1
/// MTP head. Non-GGUF or missing-key → `Ok(false)`.
fn gguf_has_mtp_heads(path: &Path) -> Result<bool> {
    use rlx_gguf::{GgufFile, MetaValue};
    let is_gguf = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !is_gguf {
        return Ok(false);
    }
    let raw = GgufFile::from_path(path)?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("");
    // Try `<arch>.nextn_predict_layers` first; fall back to `qwen35.*` for
    // converters that reuse the qwen35 prefix on qwen36 files.
    for k in [
        format!("{arch}.nextn_predict_layers"),
        "qwen35.nextn_predict_layers".to_string(),
        "qwen36.nextn_predict_layers".to_string(),
    ] {
        if let Some(MetaValue::U32(n)) = raw.metadata.get(&k) {
            return Ok(*n > 0);
        }
    }
    Ok(false)
}

/// Encode `text` to LM token ids using a HuggingFace `tokenizer.json`
/// resolved next to the GGUF / safetensors at `weights_path`. Pass
/// `explicit_tokenizer` to override the auto-discovery (sibling
/// `<weights>.tokenizer.json` or `tokenizer.json` in the weights dir).
///
/// PLAN.md M8 — closes the loop between [`auto_chat_template`] (which
/// returns a rendered string) and [`LmRunner::predict_logits`] /
/// [`LmRunner::generate`] (which take raw token ids).
///
/// **Fallback (PLAN.md M8):** when no `tokenizer.json` is available
/// and the weights are a GGUF, `encode_prompt_auto` automatically
/// reconstructs a byte-level BPE tokenizer from
/// `tokenizer.ggml.{tokens, merges}`. Works for the GPT-2/Qwen/Llama
/// family (`tokenizer.ggml.model = "gpt2"`); SentencePiece tokenizers
/// (`tokenizer.ggml.model = "llama"` legacy) still require a sibling
/// `tokenizer.json`.
pub fn auto_tokenize(
    weights_path: &Path,
    text: &str,
    explicit_tokenizer: Option<&Path>,
) -> Result<Vec<u32>> {
    use anyhow::Context;
    match rlx_qwen35::encode_prompt_auto(weights_path, explicit_tokenizer, text) {
        Ok(ids) => Ok(ids),
        Err(e) => {
            // Augment with the GGUF-vocab fallback hint when applicable.
            let is_gguf = weights_path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            if !is_gguf {
                return Err(e);
            }
            Err(e).with_context(|| {
                format!(
                    "auto_tokenize: no `tokenizer.json` resolved for {weights_path:?}. \
                     The GGUF ships a vocab at `tokenizer.ggml.tokens` but \
                     reconstructing a BPE encoder from GGUF-only metadata is \
                     per-family work (PLAN.md M8 follow-up). Options: \
                     (1) place `tokenizer.json` next to the GGUF; \
                     (2) pass an explicit path via the `explicit_tokenizer` arg; \
                     (3) download the matching `tokenizer.json` from the model's \
                     HF repo and point at it"
                )
            })
        }
    }
}

/// Inverse of [`auto_tokenize`] — turn `ids` back into text, using the
/// same tokenizer resolution chain (sibling `tokenizer.json` →
/// `explicit_tokenizer` → GGUF-embedded byte-level BPE vocab).
///
/// `skip_special_tokens=true` removes EOS / chat-template control
/// tokens (`<|im_end|>`, `<|endoftext|>`, …) — what you want for
/// streaming user-facing chat output. Set `false` to keep them
/// (useful for debugging or stop-string matching).
pub fn auto_detokenize(
    weights_path: &Path,
    ids: &[u32],
    explicit_tokenizer: Option<&Path>,
    skip_special_tokens: bool,
) -> Result<String> {
    rlx_qwen35::decode_ids_auto(weights_path, explicit_tokenizer, ids, skip_special_tokens)
}
