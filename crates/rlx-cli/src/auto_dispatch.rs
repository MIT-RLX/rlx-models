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

//! Auto-dispatch: pick a registered model runner from a weights path.
//!
//! `auto_runner_name(path)` resolves the path (file or directory), sniffs
//! the model family (GGUF `general.architecture` for `.gguf`, sidecar
//! `config.json` `model_type` for safetensors), and maps it to the short
//! runner name a callsite registered with [`register_cli`](crate::register_cli)
//! (e.g. `"qwen3"`, `"gemma"`).
//!
//! `auto_dispatch(path, args)` is a one-shot: sniff, look up, run.
//!
//! Used by `skill` so callers don't need to hardcode `Qwen3Runner` vs
//! `GemmaRunner` per family.

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::gguf_support::{gguf_architecture_from_path, resolve_weights_file};
use std::path::{Path, PathBuf};

use crate::registry::run_registered;

/// Entry point for an `rlx-run auto WEIGHTS [args...]` subcommand.
///
/// Treats the first positional as the weights path (file or directory),
/// sniffs the runner, and forwards the remaining args to it. The
/// canonical wiring is `register_cli("auto", "...", rlx_cli::run_auto)`
/// in the multiplexer.
pub fn run_auto(args: &[String]) -> Result<()> {
    let Some(first) = args.first() else {
        bail!(
            "auto: expected WEIGHTS path as the first argument\n\
             usage: rlx-run auto <weights-path> [runner-args...]"
        );
    };
    if matches!(first.as_str(), "-h" | "--help" | "help") {
        println!(
            "rlx-run auto — sniff a GGUF / safetensors file and dispatch to the right runner\n\
             \n\
             USAGE:\n  rlx-run auto <weights-path> [runner-args...]\n\
             \n\
             The first argument is forwarded as the runner's --weights value;\n\
             remaining arguments are passed through unchanged."
        );
        return Ok(());
    }
    let path = Path::new(first);
    let sniff = auto_sniff(path)?;
    eprintln!(
        "[rlx-run auto] {} → runner `{}` (from {:?})",
        sniff.path.display(),
        sniff.runner_name,
        sniff.from
    );
    // Re-build argv: most per-family runners take `--weights PATH`. If the
    // caller already passed --weights, don't double it; otherwise inject.
    //
    // A GGUF model is a single self-contained file → forward the file. A
    // safetensors model is a multi-file directory (config.json + tokenizer +
    // shards); dedicated runners load the whole dir (mlx-community affine
    // dirs, sharded checkpoints), so forward the DIRECTORY, not one shard.
    let forward_path: PathBuf = match &sniff.from {
        SniffedFrom::SafetensorsConfig(_) => {
            if path.is_dir() {
                path.to_path_buf()
            } else {
                sniff
                    .path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| sniff.path.clone())
            }
        }
        SniffedFrom::GgufArch(_) => sniff.path.clone(),
    };
    let rest: Vec<String> = args[1..].to_vec();
    let has_weights_flag = rest
        .iter()
        .any(|a| a == "--weights" || a.starts_with("--weights="));
    let mut forwarded: Vec<String> = Vec::with_capacity(rest.len() + 2);
    if !has_weights_flag {
        forwarded.push("--weights".into());
        forwarded.push(forward_path.display().to_string());
    }
    forwarded.extend(rest);
    match run_registered(sniff.runner_name, &forwarded)? {
        Some(()) => Ok(()),
        None => bail!(
            "auto: runner `{}` not registered (sniffed from {:?}); register it via \
             `register_cli` in your binary's main",
            sniff.runner_name,
            sniff.from
        ),
    }
}

/// Source the sniffer used to identify the model family.
#[derive(Debug, Clone)]
pub enum SniffedFrom {
    /// `general.architecture` value read from a `.gguf` file.
    GgufArch(String),
    /// `model_type` value read from a sidecar `config.json`.
    SafetensorsConfig(String),
}

/// Result of sniffing a weights path.
#[derive(Debug, Clone)]
pub struct SniffedRunner {
    /// Concrete file we sniffed (after resolving a directory).
    pub path: PathBuf,
    /// Short runner name as registered with `register_cli`.
    pub runner_name: &'static str,
    /// Where the sniff came from — useful for diagnostics.
    pub from: SniffedFrom,
}

/// A catalog arch that RLX recognizes but has not yet implemented a runner
/// for. Returned by [`known_unimplemented_arch`] so error messages can point
/// at the PLAN.md milestone that unblocks the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnimplementedArch {
    /// Display name (e.g. `"Mistral 3.5"`).
    pub family: &'static str,
    /// PLAN.md milestone tag (e.g. `"M4"`).
    pub milestone: &'static str,
    /// One-line note for the user.
    pub note: &'static str,
}

/// Family-level metadata referenced by [`KNOWN_UNIMPLEMENTED`]. Static so
/// the phf map can hold `&'static UnimplementedArch`.
mod families {
    use super::UnimplementedArch;
    #[allow(dead_code)] // was KNOWN_UNIMPLEMENTED; mistral3/4 now route via rlx-mistral
    pub static MISTRAL: UnimplementedArch = UnimplementedArch {
        family: "Mistral 3+ / Ministral",
        milestone: "M4",
        note: "Llama-shaped with newer RoPE; share `rlx-llama-base` per PLAN.md M4",
    };
    pub static PHIMOE: UnimplementedArch = UnimplementedArch {
        family: "Phi MoE",
        milestone: "M4 + M5",
        note: "Phi + MoE routing; depends on shared MoE block — PLAN.md M4/M5",
    };
    pub static BONSAI: UnimplementedArch = UnimplementedArch {
        family: "Bonsai",
        milestone: "M4",
        note: "Llama-shaped; HF model_type only — usually ships as llama GGUF — PLAN.md M4",
    };
    pub static OMNICODER: UnimplementedArch = UnimplementedArch {
        family: "OmniCoder",
        milestone: "M4",
        note: "Qwen3-coder shaped — PLAN.md M4 (often tagged `qwen3` in GGUF)",
    };
    pub static MINIMAX: UnimplementedArch = UnimplementedArch {
        family: "MiniMax M2",
        milestone: "M5",
        note: "Lightning Attention; depends on `rlx-ssm` upstream — PLAN.md M5",
    };
    pub static GLM: UnimplementedArch = UnimplementedArch {
        family: "GLM 4 / 5",
        milestone: "M5",
        note: "GLM RoPE + RMSNorm placement — PLAN.md M5",
    };
    pub static GLM_MOE: UnimplementedArch = UnimplementedArch {
        family: "GLM 4 MoE",
        milestone: "M5",
        note: "GLM + MoE routing — PLAN.md M5",
    };
    pub static GPT_OSS: UnimplementedArch = UnimplementedArch {
        family: "gpt-oss",
        milestone: "M5",
        note: "OpenAI gpt-oss — confirm arch shape — PLAN.md M5",
    };
    #[allow(dead_code)] // dense nemotron now routes to the llama32 runner
    pub static NEMOTRON: UnimplementedArch = UnimplementedArch {
        family: "Nemotron",
        milestone: "M5",
        note: "Dense Nemotron arch — PLAN.md M5",
    };
    pub static NEMOTRON_H: UnimplementedArch = UnimplementedArch {
        family: "Nemotron-H",
        milestone: "M5",
        note: "Mamba+attention hybrid; depends on `rlx-ssm` upstream — PLAN.md M5/M7",
    };
    #[allow(dead_code)]
    pub static LFM: UnimplementedArch = UnimplementedArch {
        family: "LFM 2 / 2.5",
        milestone: "M5",
        note: "Liquid Foundation Models with custom SSM layers — PLAN.md M5",
    };
    pub static LFM_MOE: UnimplementedArch = UnimplementedArch {
        family: "LFM 2 MoE",
        milestone: "M5",
        note: "LFM + MoE — PLAN.md M5",
    };
    pub static QWEN3_MOE: UnimplementedArch = UnimplementedArch {
        family: "Qwen3 MoE",
        milestone: "M5",
        note: "Qwen3 + MoE routing block — PLAN.md M5 (often loadable via qwen3 runner once MoE lands)",
    };
    #[allow(dead_code)] // qwen3_next now routes to the qwen35 runner
    pub static QWEN3_NEXT: UnimplementedArch = UnimplementedArch {
        family: "Qwen3-Next",
        milestone: "M5",
        note: "Qwen3-Next variant — confirm arch deltas vs qwen3 — PLAN.md M5",
    };
    pub static GEMMA4: UnimplementedArch = UnimplementedArch {
        family: "Gemma 4 MoE (A4B)",
        milestone: "M2",
        note: "Gemma 4 MoE A4B routing block — dense + E2B/E4B run via the `gemma` runner — PLAN.md M2",
    };
    pub static QWEN3_MTP: UnimplementedArch = UnimplementedArch {
        family: "Qwen3 / Qwen3.6 + MTP",
        milestone: "M6",
        note: "multi-token-prediction draft heads — PLAN.md M6",
    };
    pub static LLADA: UnimplementedArch = UnimplementedArch {
        family: "LLaDA / LLaDA MoE (text-only)",
        milestone: "M5",
        note: "dense LLaDA arch in llama.cpp; rlx-llada2 currently targets the diffusion runner — PLAN.md M5",
    };
    pub static GRANITE: UnimplementedArch = UnimplementedArch {
        family: "Granite (IBM)",
        milestone: "M4",
        note: "Llama-shaped — PLAN.md M4",
    };
    pub static DEEPSEEK: UnimplementedArch = UnimplementedArch {
        family: "DeepSeek-V3 / Kimi-K2",
        milestone: "M5",
        note: "MLA + fine-grained MoE graph IS built in rlx-deepseek (finite on synthetic F32); \
               remaining = a runner + packed DequantMatMul re-plumb (rlx-flow path is F32-only) — \
               real 671B–1T checkpoint not runnable on this box yet",
    };
    pub static COHERE: UnimplementedArch = UnimplementedArch {
        family: "Command-R / Cohere",
        milestone: "M4",
        note: "parallel-residual coded; cohere2 (command-r7b) needs per-layer sliding/full-attn + NoPE-on-global — still garbage",
    };
}

/// Catalog families we know about but haven't implemented yet.
///
/// The keys are the **actual** GGUF `general.architecture` strings llama.cpp
/// uses (`src/llama-arch.cpp::LLM_ARCH_NAMES`) plus their HF `model_type`
/// aliases when those differ. Notably:
///
/// * Mistral 1/2 and Qwen 2.5 ship as `general.architecture = llama` /
///   `qwen2` respectively — they don't have their own llama.cpp arch tag.
///   Those tags route to the existing `llama32` / `qwen3` runners and are
///   *not* listed here.
/// * Mistral 3+ ships as `mistral3` / `mistral4` (real tags).
/// * Phi-4 ships as `phi3` (Phi-4 reuses the Phi-3 arch in llama.cpp).
///
/// Both GGUF arch tags and HF `model_type` values are accepted so
/// downstream callers don't keep two parallel lists.
static KNOWN_UNIMPLEMENTED: phf::Map<&'static str, &'static UnimplementedArch> = phf::phf_map! {
    // mistral3 / mistral4 route via `GgufModelFamily::Mistral` → `rlx-mistral`
    // (and `rlx-mistral-vl` when mmproj is attached).
    // Phi MoE — still pending.
    "phimoe" => &families::PHIMOE,
    // Catalog HF model_type aliases — same remap gap as phi3.
    "bonsai" => &families::BONSAI,
    "omnicoder" => &families::OMNICODER,
    // Hybrid / SSM families
    "minimax-m2" => &families::MINIMAX,
    "minimax_m2" => &families::MINIMAX,
    "minimax" => &families::MINIMAX,
    // dense glm4 / chatglm now route to the llama32 runner (per-arch packed
    // builder delta: GLM norm placement + partial RoPE + fused gate∥up).
    // glm5 / glm4moe remain unimplemented.
    "glm5" => &families::GLM,
    "glm4moe" => &families::GLM_MOE,
    "gpt-oss" => &families::GPT_OSS,
    "gpt_oss" => &families::GPT_OSS,
    // dense `nemotron` now routes to the llama32 runner (LayerNorm + squared-ReLU
    // gate-less FFN). The Mamba/attention hybrid `nemotron_h` still needs rlx-ssm.
    "nemotron_h" => &families::NEMOTRON_H,
    "nemotron_h_moe" => &families::NEMOTRON_H,
    // lfm2 / lfm / lfm25 / lfm2_5 are now routed through `rlx-lfm`'s
    // `LfmRunner` via `gguf_family_for_arch` → `GgufModelFamily::Lfm`.
    // Only the MoE variant remains unimplemented.
    "lfm2moe" => &families::LFM_MOE,
    // Qwen variants we don't run yet
    "qwen3moe" => &families::QWEN3_MOE,
    // qwen3next / qwen3_5 / qwen3_5_moe / qwen3_5_mtp now route to the qwen35
    // runner (gated-DeltaNet graph is implemented; mlx-community affine dirs
    // load via MlxLoader) — see model_registry `id:"qwen35"`.
    // Gemma 3 / 3n route through `gemma` (see `model_type_runner_name`).
    // Only the MoE A4B variant remains unimplemented.
    "gemma4moe" => &families::GEMMA4,
    // qwen3vl* route through `gguf_family_for_arch` → `GgufModelFamily::Qwen3Vl`.
    "qwen3_mtp" => &families::QWEN3_MTP,
    "qwen3-mtp" => &families::QWEN3_MTP,
    "qwen36_mtp" => &families::QWEN3_MTP,
    // Other catalog-adjacent families
    "llada" => &families::LLADA,
    "llada-moe" => &families::LLADA,
    // dense `granite` now routes to the llama32 runner (embed/residual/attn/
    // logit multipliers wired in the packed builder). MoE / hybrid remain.
    "granitemoe" => &families::GRANITE,
    "granitehybrid" => &families::GRANITE,
    "deepseek2" => &families::DEEPSEEK,
    "deepseek2-ocr" => &families::DEEPSEEK,
    // HF model_type keys (mlx-community): builder exists in rlx-deepseek, runner pending.
    "deepseek_v3" => &families::DEEPSEEK,
    "deepseek_v4" => &families::DEEPSEEK,
    "kimi_k2" => &families::DEEPSEEK,
    "kimi_k25" => &families::DEEPSEEK,
    // cohere/command-r/cohere2 stay unimplemented: parallel-residual coded but
    // cohere2 still garbage (per-layer sliding/full-attn + NoPE-on-global unwired).
    "cohere" => &families::COHERE,
    "command-r" => &families::COHERE,
    "cohere2" => &families::COHERE,
};

/// Look up an arch / model_type in the unimplemented-families table.
pub fn known_unimplemented_arch(arch_or_model_type: &str) -> Option<UnimplementedArch> {
    KNOWN_UNIMPLEMENTED.get(arch_or_model_type).map(|p| **p)
}

/// Snapshot of every (key, family) pair currently in the unimplemented
/// table — useful for `rlx-run check --list-unimplemented` style tooling.
pub fn known_unimplemented_keys() -> impl Iterator<Item = (&'static str, &'static UnimplementedArch)>
{
    KNOWN_UNIMPLEMENTED.entries().map(|(k, v)| (*k, *v))
}

/// Map a GGUF `general.architecture` tag to the short runner name.
///
/// Returns `None` for hint-only families (e.g. embedding GGUFs) and for
/// catalog families that aren't implemented yet — those get a richer error
/// via [`known_unimplemented_arch`] when sniffed.
pub fn arch_runner_name(arch: &str) -> Option<&'static str> {
    rlx_core::model_registry::runner_for_gguf_arch(arch)
}

/// Map an HF `config.json` `model_type` value to a short runner name.
///
/// HF naming differs from GGUF tags — `model_type: "llama"` covers Llama
/// 2 / 3 / 3.x, `qwen3` covers Qwen3 and Qwen3 MoE, etc.
///
/// Source of truth: [`rlx_core::model_registry`].
pub fn model_type_runner_name(model_type: &str) -> Option<&'static str> {
    rlx_core::model_registry::runner_for_hf_model_type(model_type)
}

/// Sniff `model_type` from the `config.json` next to a safetensors file.
fn read_model_type_from_sidecar(path: &Path) -> Result<Option<String>> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("safetensors path {path:?} has no parent dir"))?;
    let cfg = dir.join("config.json");
    if !cfg.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&cfg).with_context(|| format!("reading {cfg:?}"))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {cfg:?}"))?;
    Ok(v.get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned))
}

/// Resolve `path` to a single weight file, then sniff the runner.
pub fn auto_sniff(path: &Path) -> Result<SniffedRunner> {
    let file = resolve_weights_file(path)?;
    let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "gguf" => {
            let arch = gguf_architecture_from_path(&file)?;
            let runner = arch_runner_name(&arch).ok_or_else(|| {
                if let Some(u) = known_unimplemented_arch(&arch) {
                    anyhow!(
                        "{file:?}: GGUF architecture `{arch}` is {} ({}) — not yet implemented in rlx-models. {}",
                        u.family, u.milestone, u.note
                    )
                } else {
                    anyhow!(
                        "{file:?}: GGUF architecture `{arch}` has no registered rlx runner; \
                         see `rlx-run` for supported families"
                    )
                }
            })?;
            Ok(SniffedRunner {
                path: file,
                runner_name: runner,
                from: SniffedFrom::GgufArch(arch),
            })
        }
        "safetensors" => {
            let model_type = read_model_type_from_sidecar(&file)?.ok_or_else(|| {
                anyhow!("{file:?}: no `model_type` in sidecar config.json (auto-dispatch needs it)")
            })?;
            let runner = model_type_runner_name(&model_type).ok_or_else(|| {
                if let Some(u) = known_unimplemented_arch(&model_type) {
                    anyhow!(
                        "{file:?}: safetensors model_type `{model_type}` is {} ({}) — not yet implemented in rlx-models. {}",
                        u.family, u.milestone, u.note
                    )
                } else {
                    anyhow!(
                        "{file:?}: safetensors model_type `{model_type}` has no registered rlx runner"
                    )
                }
            })?;
            Ok(SniffedRunner {
                path: file,
                runner_name: runner,
                from: SniffedFrom::SafetensorsConfig(model_type),
            })
        }
        other => {
            bail!("{file:?}: unsupported extension `.{other}` (expected .gguf or .safetensors)")
        }
    }
}

/// Sniff `path` and return only the runner short name.
pub fn auto_runner_name(path: &Path) -> Result<&'static str> {
    Ok(auto_sniff(path)?.runner_name)
}

/// Sniff `path`, look up its runner in the registry, and run it with `args`.
///
/// `args` should be the per-runner argv *without* the leading subcommand.
/// Returns the runner name that was dispatched to.
pub fn auto_dispatch(path: &Path, args: &[String]) -> Result<&'static str> {
    let sniff = auto_sniff(path)?;
    match run_registered(sniff.runner_name, args)? {
        Some(()) => Ok(sniff.runner_name),
        None => bail!(
            "runner `{}` not registered (sniffed from {:?}); register it via \
             `register_cli` before calling auto_dispatch",
            sniff.runner_name,
            sniff.from
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_runner_maps_lm_families() {
        assert_eq!(arch_runner_name("qwen3"), Some("qwen3"));
        // qwen2 now routes to the qwen3 runner — the runner reads
        // attention_bias + qk_norm from the GGUF arch tag and emits
        // the right per-layer math.
        assert_eq!(arch_runner_name("qwen2"), Some("qwen3"));
        assert_eq!(arch_runner_name("qwen35"), Some("qwen35"));
        assert_eq!(arch_runner_name("qwen35moe"), Some("qwen35"));
        // Qwen3.6 reuses the qwen35 trunk (PLAN.md M1). qwen36_mtp still
        // routes through known_unimplemented_arch — base qwen36 routes
        // here so unsloth/Qwen3.6-27B-GGUF (no MTP) just works.
        assert_eq!(arch_runner_name("qwen36"), Some("qwen35"));
        assert_eq!(arch_runner_name("qwen36moe"), Some("qwen35"));
        // Qwen 2.5 / 2.5.1 ship as `qwen2` arch tag; explicit short
        // tags also route to the qwen3 runner (PLAN.md M4).
        assert_eq!(arch_runner_name("qwen25"), Some("qwen3"));
        assert_eq!(arch_runner_name("qwen2_5"), Some("qwen3"));
        assert_eq!(arch_runner_name("llama"), Some("llama32"));
        // Muse Glimmer runs its arch deltas on the llama32 packed GGUF path
        // (`DenseArch::MuseGlimmer`), so `rlx-run auto` routes it there.
        assert_eq!(arch_runner_name("muse-glimmer"), Some("llama32"));
        assert_eq!(arch_runner_name("phi3"), Some("phi"));
        assert_eq!(arch_runner_name("gemma"), Some("gemma"));
        assert_eq!(arch_runner_name("gemma2"), Some("gemma"));
        assert_eq!(arch_runner_name("gemma3"), Some("gemma"));
    }

    #[test]
    fn arch_runner_maps_vision_and_diffusion() {
        assert_eq!(arch_runner_name("dinov2"), Some("dinov2"));
        assert_eq!(arch_runner_name("sam"), Some("sam1"));
        assert_eq!(arch_runner_name("mobile-sam"), Some("sam1"));
        assert_eq!(arch_runner_name("sam2"), Some("sam2"));
        assert_eq!(arch_runner_name("sam3"), Some("sam3"));
        assert_eq!(arch_runner_name("flux"), Some("flux2"));
        assert_eq!(arch_runner_name("vjepa2"), Some("vjepa2"));
        assert_eq!(arch_runner_name("w2v-bert"), Some("wav2vec2-bert"));
    }

    #[test]
    fn arch_runner_returns_none_for_embed_and_unknown() {
        // Embed families aren't in the rlx-run dispatch table today.
        assert_eq!(arch_runner_name("bert"), None);
        assert_eq!(arch_runner_name("nomic-bert"), None);
        assert_eq!(arch_runner_name("totally-fake-arch"), None);
    }

    #[test]
    fn known_unimplemented_covers_plan_families() {
        // M4 — mistral3/4 now route via GgufModelFamily::Mistral
        assert_eq!(known_unimplemented_arch("mistral3"), None);
        assert_eq!(arch_runner_name("mistral3"), Some("mistral"));
        assert_eq!(arch_runner_name("mistral4"), Some("mistral"));
        assert_eq!(known_unimplemented_arch("phi3"), None);
        assert_eq!(known_unimplemented_arch("phi4"), None);
        assert_eq!(known_unimplemented_arch("gemma3"), None);
        assert_eq!(known_unimplemented_arch("gemma3n"), None);
        assert_eq!(
            known_unimplemented_arch("bonsai").map(|u| u.milestone),
            Some("M4")
        );
        // dense granite is now wired (routes to llama32); MoE / hybrid remain.
        assert_eq!(known_unimplemented_arch("granite"), None);
        assert_eq!(arch_runner_name("granite"), Some("llama32"));
        assert_eq!(arch_runner_name("exaone"), Some("llama32"));
        assert_eq!(
            known_unimplemented_arch("granitemoe").map(|u| u.milestone),
            Some("M4")
        );
        // M5 — MoE / SSM
        assert_eq!(
            known_unimplemented_arch("minimax-m2").map(|u| u.milestone),
            Some("M5")
        );
        // dense glm4 / chatglm / nemotron / olmo2 now route to llama32 (validated
        // coherent on real GGUFs); cohere2 stays unimplemented (still garbage);
        // MoE / SSM-hybrid variants remain unimplemented.
        assert_eq!(known_unimplemented_arch("glm4"), None);
        assert_eq!(arch_runner_name("glm4"), Some("llama32"));
        assert_eq!(arch_runner_name("chatglm"), Some("llama32"));
        assert_eq!(known_unimplemented_arch("nemotron"), None);
        assert_eq!(arch_runner_name("nemotron"), Some("llama32"));
        assert_eq!(
            known_unimplemented_arch("cohere2").map(|u| u.milestone),
            Some("M4")
        );
        assert_eq!(arch_runner_name("olmo2"), Some("llama32"));
        assert_eq!(
            known_unimplemented_arch("glm4moe").map(|u| u.milestone),
            Some("M5")
        );
        assert_eq!(
            known_unimplemented_arch("nemotron_h").map(|u| u.milestone),
            Some("M5")
        );
        // M6 — MTP
        assert_eq!(
            known_unimplemented_arch("qwen3_mtp").map(|u| u.milestone),
            Some("M6")
        );
        // M7 — VL (qwen3vl now routes via GgufModelFamily::Qwen3Vl)
        assert_eq!(known_unimplemented_arch("qwen3vl"), None);
        assert_eq!(crate::arch_runner_name("qwen3vl"), Some("qwen3-vl"));
        // Implemented or unknown — plain `mistral` is NOT a llama.cpp arch
        // tag (Mistral 1/2 use `llama`), so it should not be flagged.
        assert_eq!(known_unimplemented_arch("qwen3"), None);
        assert_eq!(known_unimplemented_arch("mistral"), None);
        assert_eq!(known_unimplemented_arch("totally-fake"), None);
    }

    #[test]
    fn auto_sniff_error_points_at_milestone_for_known_unimplemented() {
        // Build a tiny bonsai.gguf and check the error message.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let v = "bonsai";
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let path = std::env::temp_dir().join("rlx_auto_dispatch_bonsai_hint.gguf");
        std::fs::write(&path, &buf).unwrap();
        let err = auto_sniff(&path).expect_err("should error");
        let s = format!("{err:#}");
        assert!(s.contains("Bonsai"), "expected family name in error: {s}");
        assert!(s.contains("M4"), "expected milestone tag in error: {s}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn model_type_runner_maps_known() {
        assert_eq!(model_type_runner_name("qwen3"), Some("qwen3"));
        assert_eq!(model_type_runner_name("qwen3_moe"), Some("qwen3"));
        assert_eq!(model_type_runner_name("llama"), Some("llama32"));
        assert_eq!(model_type_runner_name("gemma3"), Some("gemma"));
        // Gemma 4 dense + E2B mobile route to the gemma runner.
        assert_eq!(model_type_runner_name("gemma4"), Some("gemma"));
        assert_eq!(model_type_runner_name("gemma4_text"), Some("gemma"));
        assert_eq!(model_type_runner_name("gemma4_unified"), Some("gemma"));
        // Gemma 4 dense is no longer flagged unimplemented; MoE A4B still is.
        assert!(known_unimplemented_arch("gemma4").is_none());
        assert!(known_unimplemented_arch("gemma4moe").is_some());
        assert_eq!(
            model_type_runner_name("dinov2_with_registers"),
            Some("dinov2")
        );
        assert_eq!(model_type_runner_name("whisper"), Some("whisper"));
        assert_eq!(model_type_runner_name("unknown"), None);
    }

    /// Builds a minimal GGUF file in a temp dir, then verifies auto_sniff
    /// picks the right runner name from `general.architecture`.
    #[test]
    fn auto_sniff_reads_gguf_arch() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        write_string(&mut buf, "general.architecture", "qwen3");
        // one f32 tensor with 4 elements
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let path = std::env::temp_dir().join("rlx_auto_dispatch_sniff.gguf");
        std::fs::write(&path, &buf).unwrap();
        let sniff = auto_sniff(&path).expect("sniff");
        assert_eq!(sniff.runner_name, "qwen3");
        match sniff.from {
            SniffedFrom::GgufArch(a) => assert_eq!(a, "qwen3"),
            other => panic!("wrong sniff source: {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    /// Register a fake runner under a known name, ask `run_auto` to dispatch
    /// to it, and capture what argv it received.
    #[test]
    fn run_auto_injects_weights_flag_when_missing() {
        use crate::registry::{ModelRunner, register_runner};
        use std::sync::{Mutex, OnceLock};

        static CAPTURED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        fn captured() -> &'static Mutex<Vec<String>> {
            CAPTURED.get_or_init(|| Mutex::new(Vec::new()))
        }

        struct Capture;
        impl ModelRunner for Capture {
            fn name(&self) -> &'static str {
                "qwen3"
            }
            fn description(&self) -> &'static str {
                "test capture"
            }
            fn run(&self, args: &[String]) -> Result<()> {
                *captured().lock().unwrap() = args.to_vec();
                Ok(())
            }
        }
        register_runner(Box::new(Capture));

        // Build a minimal qwen3 GGUF in a temp dir.
        let dir = std::env::temp_dir().join("rlx_auto_dispatch_run_auto");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let v = "qwen3";
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        // Caller passed no --weights → run_auto must inject it.
        run_auto(&[path.display().to_string(), "--prompt".into(), "hi".into()]).unwrap();
        let got = captured().lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                "--weights".to_string(),
                path.display().to_string(),
                "--prompt".into(),
                "hi".into()
            ]
        );

        // Caller already passed --weights → don't inject again.
        run_auto(&[
            path.display().to_string(),
            "--weights".into(),
            "/other/path".into(),
            "--prompt".into(),
            "hi".into(),
        ])
        .unwrap();
        let got = captured().lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                "--weights".to_string(),
                "/other/path".into(),
                "--prompt".into(),
                "hi".into(),
            ]
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn auto_sniff_reads_safetensors_sidecar() {
        let dir = std::env::temp_dir().join("rlx_auto_dispatch_sidecar");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("config.json");
        std::fs::write(&cfg, br#"{"model_type":"llama"}"#).unwrap();
        let st = dir.join("model.safetensors");
        // Empty file is fine — sniffer never opens the safetensors payload.
        std::fs::write(&st, b"").unwrap();
        let sniff = auto_sniff(&st).expect("sniff");
        assert_eq!(sniff.runner_name, "llama32");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A safetensors / mlx-community model is a multi-file directory; `run_auto`
    /// must forward the DIRECTORY (so the runner can load config + tokenizer +
    /// all shards), not the single resolved shard file. Contrast with the GGUF
    /// case (single self-contained file) exercised by `run_auto_*` above.
    #[test]
    fn run_auto_forwards_dir_for_safetensors() {
        use crate::registry::{ModelRunner, register_runner};
        use std::sync::{Mutex, OnceLock};
        static CAP: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        fn cap() -> &'static Mutex<Vec<String>> {
            CAP.get_or_init(|| Mutex::new(Vec::new()))
        }
        struct C;
        impl ModelRunner for C {
            fn name(&self) -> &'static str {
                "llama32"
            }
            fn description(&self) -> &'static str {
                "test capture safetensors dir"
            }
            fn run(&self, args: &[String]) -> Result<()> {
                *cap().lock().unwrap() = args.to_vec();
                Ok(())
            }
        }
        register_runner(Box::new(C));

        let dir = std::env::temp_dir().join("rlx_auto_dispatch_st_dir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        // Resolver picks this shard; run_auto must still forward its parent dir.
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();

        run_auto(&[dir.display().to_string(), "--max-tokens".into(), "4".into()]).unwrap();
        let got = cap().lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                "--weights".to_string(),
                dir.display().to_string(), // the DIR, not dir/model.safetensors
                "--max-tokens".into(),
                "4".into(),
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
