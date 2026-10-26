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

//! Any checkpoint as an `.rlxp` package rlx runs alone, by the family
//! enabled beside this crate that takes it — the one exporter
//! ([`rlx_core::export`]) under each family's own graphs.
//!
//! ```no_run
//! use rlx_models::export::{ExportOptions, export_checkpoint};
//! let (exporter, report) = export_checkpoint(
//!     "SmolLM2-135M-Instruct".as_ref(),
//!     "smollm2.rlxp".as_ref(),
//!     &ExportOptions::default(),
//!     None, // picked from the checkpoint, as its runner is
//! )?;
//! println!("{}: {report}", exporter.name);
//! # anyhow::Ok(())
//! ```

use std::path::Path;

use anyhow::Result;

pub use rlx_core::export::check::{GenerateCheck, check_generate, open_package};
pub use rlx_core::export::{
    Checkpoint, ExportOptions, ExportReport, Exporter, Package, detect, exporter, exporters,
    register_exporter,
};

/// Register the exporter of every family enabled beside this crate.
/// Idempotent; [`export_checkpoint`] calls it. A crate of its own registers
/// the same way ([`register_exporter`]), and a later registration of a
/// name replaces a builtin.
pub fn register_builtin_exporters() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        #[cfg(feature = "qwen3")]
        rlx_qwen3::export::register();
        // After qwen3's: a GGUF Llama its own, the rest qwen3's.
        #[cfg(feature = "llama32")]
        rlx_llama32::export::register();
        #[cfg(feature = "nanbeige")]
        rlx_nanbeige::export::register();
        #[cfg(feature = "hy-mt")]
        rlx_hy_mt::export::register();
        #[cfg(feature = "neuralhash")]
        rlx_neuralhash::export::register();
        #[cfg(feature = "upscale")]
        rlx_upscale::export::register();
        #[cfg(feature = "birefnet")]
        rlx_birefnet::export::register();
        #[cfg(feature = "subject-lift")]
        rlx_subject_lift::export::register();
        #[cfg(feature = "conformer-ctc")]
        rlx_conformer_ctc::export::register();
        #[cfg(feature = "wespeaker")]
        rlx_wespeaker::export::register();
        #[cfg(feature = "timestral")]
        rlx_timestral::export::register();
        #[cfg(feature = "sam")]
        rlx_sam::export::register();
        #[cfg(feature = "pplx-decider")]
        rlx_pplx_decider::export::register();
        #[cfg(feature = "d1")]
        rlx_d1::export::register();
        #[cfg(feature = "kev")]
        rlx_kev::export::register();
        #[cfg(feature = "carbon")]
        rlx_carbon::export::register();
        #[cfg(feature = "minicpm-v")]
        rlx_minicpm_v::export::register();
        #[cfg(feature = "vad")]
        rlx_vad::export::register();
        #[cfg(feature = "moonshine")]
        rlx_moonshine::export::register();
        #[cfg(feature = "gemma")]
        rlx_gemma::export::register();
        #[cfg(feature = "qwen3-vl")]
        rlx_qwen3_vl::export::register();
        #[cfg(feature = "qwen25-vl")]
        rlx_qwen25_vl::export::register();
        #[cfg(feature = "qwen35")]
        rlx_qwen35::export::register();
        #[cfg(feature = "embed")]
        rlx_embed::export::register();
        #[cfg(feature = "whisper")]
        rlx_whisper::export::register();
        #[cfg(feature = "siglip2")]
        rlx_siglip2::export::register();
        #[cfg(feature = "lfm")]
        rlx_lfm::export::register();
        #[cfg(feature = "llama4")]
        rlx_llama4::export::register();
        #[cfg(feature = "minimax")]
        rlx_minimax::m3::export::register();
        #[cfg(feature = "dinov2")]
        rlx_dinov2::export::register();
        #[cfg(feature = "dinov3")]
        rlx_dinov3::export::register();
        #[cfg(feature = "uni2")]
        rlx_uni2::export::register();
        #[cfg(feature = "embeddinggemma")]
        rlx_embeddinggemma::export::register();
        #[cfg(feature = "jina-v5")]
        rlx_jina_v5::export::register();
        #[cfg(feature = "snac")]
        rlx_snac::export::register();
        #[cfg(feature = "nanocodec")]
        rlx_nanocodec::export::register();
        #[cfg(feature = "wavtokenizer")]
        rlx_wavtokenizer::export::register();
        #[cfg(feature = "xcodec")]
        rlx_xcodec::export::register();
        #[cfg(feature = "vibevoice")]
        rlx_vibevoice::export::register();
        #[cfg(feature = "facodec")]
        rlx_facodec::export::register();
        #[cfg(feature = "encodec")]
        rlx_encodec::export::register();
        #[cfg(feature = "speechtokenizer")]
        rlx_speechtokenizer::export::register();
        #[cfg(feature = "dac")]
        rlx_dac::export::register();
        #[cfg(feature = "mimi")]
        rlx_mimi::export::register();
        #[cfg(feature = "bioclip2")]
        rlx_bioclip2::export::register();
        #[cfg(feature = "ten-vad")]
        rlx_ten_vad::export::register();
        #[cfg(feature = "moonshine-streaming")]
        rlx_moonshine_streaming::export::register();
        #[cfg(feature = "kokoro")]
        rlx_kokoro::export::register();
        #[cfg(feature = "piper")]
        rlx_piper::export::register();
        #[cfg(feature = "supertonic")]
        rlx_supertonic::export::register();
        #[cfg(feature = "luxtts")]
        rlx_luxtts::export::register();
        #[cfg(feature = "zipvoice")]
        rlx_zipvoice::export::register();
        #[cfg(feature = "f5tts")]
        rlx_f5tts::export::register();
        #[cfg(feature = "melotts")]
        rlx_melotts::export::register();
        #[cfg(feature = "openvoice")]
        rlx_openvoice::export::register();
        #[cfg(feature = "parlertts")]
        rlx_parlertts::export::register();
        #[cfg(feature = "moss-nano")]
        rlx_moss_nano::export::register();
        #[cfg(feature = "orpheus")]
        rlx_orpheus::export::register();
        #[cfg(feature = "maya1")]
        rlx_maya1::export::register();
        #[cfg(feature = "neutts")]
        rlx_neutts::export::register();
        #[cfg(feature = "soprano")]
        rlx_soprano::export::register();
        #[cfg(feature = "chatterbox")]
        rlx_chatterbox::export::register();
        #[cfg(feature = "kittentts2")]
        rlx_kittentts2::export::register();
        #[cfg(feature = "miratts")]
        rlx_miratts::export::register();
        #[cfg(feature = "sesame")]
        rlx_sesame::export::register();
        #[cfg(feature = "zonos")]
        rlx_zonos::export::register();
        #[cfg(feature = "metavoice")]
        rlx_metavoice::export::register();
        #[cfg(feature = "gepard")]
        rlx_gepard::export::register();
        #[cfg(feature = "pocket-tts")]
        rlx_pocket_tts::export::register();
        #[cfg(feature = "voxtral-tts")]
        rlx_voxtral_tts::export::register();
        #[cfg(feature = "fish")]
        {
            rlx_fish::export::register();
            rlx_fish::drama3_export::register();
        }
        #[cfg(feature = "rlx-tts")]
        rlx_tts::export::register();
        #[cfg(feature = "tada")]
        rlx_tada::export::register();
        #[cfg(feature = "nemotron-asr")]
        rlx_nemotron_asr::export::register();
        #[cfg(feature = "parakeet")]
        rlx_parakeet::export::register();
        #[cfg(feature = "funasr")]
        rlx_funasr::export::register();
        #[cfg(feature = "qwen3-asr")]
        rlx_qwen3_asr::export::register();
        #[cfg(feature = "voxtral")]
        rlx_voxtral::export::register();
        #[cfg(feature = "wake")]
        rlx_wake::export::register();
        #[cfg(feature = "openwakeword")]
        rlx_openwakeword::export::register();
        #[cfg(feature = "sanotts")]
        rlx_sanotts::export::register();
        #[cfg(feature = "inflect-nano")]
        rlx_inflect_nano::export::register();
        #[cfg(feature = "kyutai-tts")]
        rlx_kyutai_tts::export::register();
        #[cfg(feature = "moshi")]
        rlx_moshi::export::register();
        #[cfg(feature = "qwen3-tts")]
        rlx_qwen3_tts::export::register();
    });
}

/// Export the checkpoint at `path` (a Hugging Face directory or a GGUF) as
/// the package `out`, by the family that takes it, or `name`'s exporter.
pub fn export_checkpoint(
    path: &Path,
    out: &Path,
    opts: &ExportOptions,
    name: Option<&str>,
) -> Result<(Exporter, ExportReport)> {
    register_builtin_exporters();
    rlx_core::export::export(path, out, opts, name)
}

/// The package's `generate` entry: what its model block says.
pub fn generate_entry(package: &Path) -> Result<serde_json::Value> {
    let module = rlx_runtime::Package::open(package)
        .and_then(|p| p.module())
        .map_err(|e| anyhow::anyhow!("{}: {e}", package.display()))?;
    Ok(module
        .model
        .as_ref()
        .and_then(|m| rlx_runtime::model::rlxp::entry_of_kind(m, "generate"))
        .unwrap_or_default())
}

/// `text` as the package's chat template renders a user's message (with
/// the assistant's turn opened), when it has one; else `text`.
pub fn as_chat(package: &Path, text: &str) -> Result<String> {
    let entry = generate_entry(package)?;
    let chat = &entry["chat"];
    let Some(source) = chat["chat_template"].as_str() else {
        return Ok(text.to_string());
    };
    let src = match source.strip_prefix("sidecar:") {
        Some(id) => rlx_runtime::Package::open(package)
            .and_then(|p| p.sidecar(id))
            .map_err(|e| anyhow::anyhow!("{}: {id}: {e}", package.display()))?,
        None => source.as_bytes().to_vec(),
    };
    let token = |k: &str| chat[k].as_str().map(String::from);
    rlx_text::chat::ChatTemplate::from_source(String::from_utf8_lossy(&src).into_owned())?
        .with_tokens(token("bos_token"), token("eos_token"))
        .render(&[rlx_text::chat::ChatMessage::user(text)], true)
}

/// Check the package `package`, exported from the checkpoint at
/// `checkpoint`, against the family's own [`LmRunner`] for it
/// ([`crate::run::auto_runner_with`]): the logits after each of `prompts`
/// and up to `n_new` greedy tokens, each ending at the package's first
/// stop id, on `device` (`"cpu"`, `"metal"`, …).
///
/// [`LmRunner`]: rlx_runtime::lm::LmRunner
#[cfg(feature = "runner")]
pub fn check_package(
    checkpoint: &Path,
    package: &Path,
    prompts: &[Vec<u32>],
    n_new: usize,
    device: &str,
) -> Result<GenerateCheck> {
    use anyhow::Context;
    let dev = rlx_runtime::parse_device(device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let opts = crate::run::AutoOptions {
        device: Some(dev),
        ..Default::default()
    };
    let mut runner = crate::run::auto_runner_with(checkpoint, &opts)?;
    // The package's tokenizer loads as the language bindings load it.
    rlx_text::install_model_tokenizer();
    let mut model = open_package(package, device)?;
    let lm = model
        .lm()
        .context("the package is not a language model (no generate entry)")?;
    let stop: Vec<u32> = generate_entry(package)?["stop_ids"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default();
    check_generate(runner.as_mut(), lm, prompts, n_new, &stop)
}
