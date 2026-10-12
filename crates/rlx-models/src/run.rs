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
/// Long-context KV context-store config (HNSW retrieval memory). Re-exported so
/// consumers can enable it via `Qwen3Runner::enable_kv_store[_with_encoder]`.
#[cfg(feature = "mmap-kv")]
pub use rlx_qwen3::KvStoreConfig;
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
/// Covers the LM families reachable from the multiplexer: `qwen3`, `qwen35`,
/// `qwen4exp`, `gemma`, `llama32`, `mistral`, `lfm`, `qwen3-vl`, `llama4`,
/// `mllama`. Other families (diffusion, embedding, audio) do not fit the
/// `LmRunner` shape and return an error naming the crate that does handle
/// them. They keep their per-family builders.
///
/// The runner comes back as the family built it, which means the **host-driven
/// decode pair may be unsupported**: `LmRunner::prefill_logits` /
/// `decode_logits` default to an error, and most families have not overridden
/// them. If you are driving the loop yourself — a server, a sampler with logit
/// bias, anything that owns token selection — use
/// [`auto_runner_servable`] instead, which guarantees the pair.
///
/// PLAN.md M3. The `LmRunner` trait gained a default `generate(..)`
/// in M8, so a boxed runner from this function can stream tokens too.
pub fn auto_runner(path: &Path) -> Result<Box<dyn LmRunner>> {
    auto_runner_with(path, &AutoOptions::default())
}

/// How to build a runner from a path.
///
/// An alias for [`rlx_core::lm_registry::LmBuildOptions`], which is where the
/// type lives so a crate registering its own family can name it without
/// depending on this facade.
pub type AutoOptions = rlx_core::lm_registry::LmBuildOptions;

/// Build a runner for any supported checkpoint.
///
/// The one entry point the others delegate to.
pub fn auto_runner_with(path: &Path, opts: &AutoOptions) -> Result<Box<dyn LmRunner>> {
    register_builtin_lm_runners();
    let sniff = auto_sniff(path)?;
    // Reject an exotic device before any family reads a config or loads
    // tensors: uniform message, and it fails before the I/O.
    rlx_core::device_capabilities::validate_lm_device(sniff.runner_name, opts.device)?;

    if !rlx_core::lm_registry::lm_runner_is_registered(sniff.runner_name) {
        // A known architecture with no constructor. The metadata registry
        // knows which crate handles it, so say so rather than leaving the
        // caller to grep — looked up by the **architecture**, since the table
        // is keyed on that and `phi3` -> runner `phi` is not an arch.
        let hint = match &sniff.from {
            rlx_cli::auto_dispatch::SniffedFrom::GgufArch(a) => {
                rlx_core::model_registry::hint_for_gguf_arch(a)
            }
            rlx_cli::auto_dispatch::SniffedFrom::SafetensorsConfig(t) => {
                rlx_core::model_registry::hint_for_hf_model_type(t)
            }
        }
        .map(|h| format!(" — try {h}"))
        .unwrap_or_default();
        bail!(
            "auto_runner: runner `{}` (sniffed from {:?}) has no registered `LmRunner` \
             constructor{hint}",
            sniff.runner_name,
            sniff.from
        );
    }

    // `build_lm_runner` honours `opts.servable` itself.
    rlx_core::lm_registry::build_lm_runner(sniff.runner_name, sniff.path.as_path(), opts)
}

/// [`auto_runner`], guaranteed to support the host-driven decode pair.
///
/// `prefill_logits` + `decode_logits` is the shape a server needs, because
/// sampling, logit bias, log-probs and stop handling all belong above the
/// model. The trait's default for that pair is *unsupported* and only a couple
/// of families override it, so a runner that came straight from
/// [`auto_runner`] may refuse the first request.
///
/// This wraps whatever was built in [`rlx_core::lm_host_driven::HostDriven`],
/// which answers each step by re-running `predict_logits` over the whole
/// context. A family that implements the pair natively is detected on the first
/// prefill and passed through untouched, so the wrapper costs nothing where it
/// is not needed — and where it is needed, it is **quadratic**. That is the
/// difference between a family being slow and being unavailable.
///
/// ```no_run
/// # use std::path::Path;
/// let mut runner = rlx_models::run::auto_runner_servable(Path::new("model.gguf"))?;
/// let logits = runner.prefill_logits(&[9707, 11])?;
/// # let _ = logits;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn auto_runner_servable(path: &Path) -> Result<Box<dyn LmRunner>> {
    auto_runner_with(path, &AutoOptions::default().servable())
}

/// [`auto_runner_servable`] with an mmproj vision encoder.
pub fn auto_runner_servable_with_mmproj(
    path: &Path,
    mmproj: Option<&Path>,
) -> Result<Box<dyn LmRunner>> {
    let opts = AutoOptions {
        mmproj: mmproj.map(|p| p.to_path_buf()),
        ..AutoOptions::default().servable()
    };
    auto_runner_with(path, &opts)
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
    let opts = AutoOptions {
        mmproj: mmproj.map(|p| p.to_path_buf()),
        ..Default::default()
    };
    auto_runner_with(path, &opts)
}

/// The builtin families, dispatched by name.
///
/// Registered into [`rlx_core::lm_registry`] by [`register_builtin_lm_runners`]
/// rather than being reachable only from a `match` here — that is what lets an
/// out-of-tree crate add a family, and what makes the set enumerable.
fn build_builtin(name: &str, path: &Path, opts: &AutoOptions) -> Result<Box<dyn LmRunner>> {
    let device = opts.device;
    let mmproj = opts.mmproj_path();
    let weights = path;
    // Packed-K-quant auto-detection is now inside each runner's
    // `.build()` (matches llama.cpp's behaviour — K-quant tensors stay
    // packed in memory, never materialise to a dense F32 matrix).
    let runner: Box<dyn LmRunner> = match name {
        "qwen3" => Box::new(
            Qwen3Runner::builder()
                .weights(weights)
                .device(device)
                .build()?,
        ),
        "qwen35" => {
            // When mmproj is requested, prefer plain Qwen35Runner (vision) over
            // MTP SpecRunner — speculative decode + mmproj is not wired together.
            let want_mtp = mmproj.is_none() && gguf_has_mtp_heads(weights).unwrap_or(false);
            if want_mtp {
                Box::new(
                    rlx_qwen35::Qwen35SpecRunner::builder()
                        .weights(weights)
                        .device(device)
                        .build()?,
                )
            } else {
                let mut b = Qwen35Runner::builder().weights(weights).device(device);
                if let Some(mp) = mmproj {
                    b = b.mmproj(mp);
                }
                Box::new(b.build()?)
            }
        }
        "minimax-m3" => {
            #[cfg(feature = "minimax")]
            {
                let w = weights.to_str().ok_or_else(|| {
                    anyhow::anyhow!("auto_runner: {weights:?} is not valid UTF-8")
                })?;
                // `None` config path: it is read from the `config.json` beside
                // the weights, which is how a HF checkpoint ships.
                Box::new(rlx_minimax::m3::runner::MiniMaxM3Runner::from_pretrained(
                    w, None, device,
                )?) as Box<dyn LmRunner>
            }
            #[cfg(not(feature = "minimax"))]
            {
                bail!(
                    "auto_runner: runner `minimax-m3` requires feature `minimax` \
                     (enable rlx-models/minimax)"
                );
            }
        }
        "qwen4exp" => {
            #[cfg(feature = "qwen4exp")]
            {
                use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions};
                // `open` attaches the n-gram table when the file carries it;
                // a split checkpoint needs `RunnerOptions::ple_table`, which
                // this entry point cannot guess — the runner then refuses on
                // the first step rather than reading zeros.
                Box::new(Qwen4ExpRunner::open(weights, RunnerOptions::new(device))?)
                    as Box<dyn LmRunner>
            }
            #[cfg(not(feature = "qwen4exp"))]
            {
                bail!(
                    "auto_runner: runner `qwen4exp` requires feature `qwen4exp` \
                     (enable rlx-models/qwen4exp)"
                );
            }
        }
        "gemma" => {
            let mut b = GemmaRunner::builder().weights(weights).device(device);
            if let Some(mp) = mmproj {
                b = b.mmproj(mp);
            }
            Box::new(b.build()?)
        }
        "llama32" => Box::new(
            Llama32Runner::builder()
                .weights(weights)
                .device(device)
                .build()?,
        ),
        "mistral" => {
            if let Some(mp) = mmproj {
                #[cfg(feature = "mistral-vl")]
                {
                    Box::new(
                        rlx_mistral_vl::MistralVlRunner::builder()
                            .weights(weights)
                            .mmproj(mp)
                            .device(device)
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
                            .device(device)
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
                // LFM2 / LFM2.5 GGUFs use the hybrid ShortConv+attention arch
                // (`Lfm2GgufRunner`); the legacy `LfmRunner` only handles the
                // older SSM variant and bails on ShortConv tensors.
                let is_gguf = weights
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|e| e.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false);
                if is_gguf {
                    Box::new(rlx_lfm::Lfm2GgufRunner::open(weights, device)?) as Box<dyn LmRunner>
                } else {
                    Box::new(
                        rlx_lfm::LfmRunner::builder()
                            .weights(weights)
                            .device(device)
                            .build()?,
                    )
                }
            }
        }
        "qwen3-vl" => {
            #[cfg(feature = "qwen3-vl")]
            {
                let mut b = rlx_qwen3_vl::Qwen3VlRunner::builder()
                    .weights(weights)
                    .device(device);
                if let Some(mp) = mmproj {
                    b = b.mmproj(mp);
                }
                Box::new(b.build()?) as Box<dyn LmRunner>
            }
            #[cfg(not(feature = "qwen3-vl"))]
            {
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
            bail!(
                "auto_runner: `llama4` is a safetensors-checkpoint runner (MoE + native \
                 vision), not an LmRunner/GGUF path — build it directly with \
                 `rlx_llama4::Llama4Runner::from_checkpoint(dir, device)` (or `Llama4VlRunner` \
                 for image input)"
            );
        }
        "mllama" => {
            bail!(
                "auto_runner: `mllama` (Llama-3.2-Vision, cross-attention) is a \
                 safetensors-checkpoint runner, not an LmRunner/GGUF path — build it directly \
                 with `rlx_mllama::MllamaRunner::from_checkpoint(dir, device)`"
            );
        }
        other => {
            // Unreachable via the registry: every name below is registered to
            // this function, and an unregistered one is refused earlier, where
            // the sniff is still in scope to name the crate that handles it.
            bail!("auto_runner: `{other}` has no builtin constructor");
        }
    };
    Ok(runner)
}

/// One `fn` per builtin name: a constructor is a plain `fn` pointer and cannot
/// capture its own name, so each gets a two-line wrapper over [`build_builtin`].
macro_rules! builtin_ctor {
    ($fname:ident, $name:literal) => {
        fn $fname(path: &Path, opts: &AutoOptions) -> Result<Box<dyn LmRunner>> {
            build_builtin($name, path, opts)
        }
    };
}

builtin_ctor!(ctor_qwen3, "qwen3");
builtin_ctor!(ctor_qwen35, "qwen35");
builtin_ctor!(ctor_qwen4exp, "qwen4exp");
builtin_ctor!(ctor_gemma, "gemma");
builtin_ctor!(ctor_llama32, "llama32");
builtin_ctor!(ctor_mistral, "mistral");
builtin_ctor!(ctor_lfm, "lfm");
builtin_ctor!(ctor_qwen3_vl, "qwen3-vl");
builtin_ctor!(ctor_llama4, "llama4");
builtin_ctor!(ctor_mllama, "mllama");
builtin_ctor!(ctor_minimax_m3, "minimax-m3");

/// Register every builtin family's constructor.
///
/// Idempotent and cheap; [`auto_runner_with`] calls it, so a caller never has
/// to. An out-of-tree crate adds itself exactly the same way, with
/// [`rlx_core::lm_registry::register_lm_runner`] — nothing here is privileged,
/// and registering the same name later overrides a builtin.
pub fn register_builtin_lm_runners() {
    use rlx_core::lm_registry::{LmRunnerCtor, LmRunnerRegistration, register_lm_runner};
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        // Capabilities are declared here, so `build_lm_runner` can refuse an
        // mmproj centrally instead of each family checking for itself — and a
        // family that forgets cannot silently ignore one.
        let regs = [
            LmRunnerRegistration::new("qwen3", ctor_qwen3 as LmRunnerCtor).mmproj_note(
                "use a `qwen3vl*` GGUF with the qwen3-vl runner, or omit mmproj for text-only",
            ),
            LmRunnerRegistration::new("qwen35", ctor_qwen35).with_mmproj(),
            LmRunnerRegistration::new("qwen4exp", ctor_qwen4exp).mmproj_note(
                "qwen4exp multimodal needs the vision seam (rlx_qwen4exp::vision), which this \
                 entry point does not wire",
            ),
            LmRunnerRegistration::new("gemma", ctor_gemma).with_mmproj(),
            LmRunnerRegistration::new("llama32", ctor_llama32)
                .mmproj_note("llama32 has no multimodal path"),
            LmRunnerRegistration::new("mistral", ctor_mistral).with_mmproj(),
            LmRunnerRegistration::new("lfm", ctor_lfm).with_mmproj(),
            LmRunnerRegistration::new("qwen3-vl", ctor_qwen3_vl).with_mmproj(),
            LmRunnerRegistration::new("llama4", ctor_llama4).mmproj_note(
                "llama4 is a safetensors-checkpoint runner; build it with \
                 `rlx_llama4::Llama4VlRunner::from_checkpoint` for image input",
            ),
            LmRunnerRegistration::new("mllama", ctor_mllama).mmproj_note(
                "mllama takes its vision tower from the checkpoint, not a separate mmproj",
            ),
            LmRunnerRegistration::new("minimax-m3", ctor_minimax_m3).mmproj_note(
                "the M3 VL variant takes its vision config from the checkpoint's config.json",
            ),
        ];
        for reg in regs {
            register_lm_runner(reg);
        }
    });
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
