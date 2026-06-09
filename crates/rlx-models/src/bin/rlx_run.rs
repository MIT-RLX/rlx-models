// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Optional multiplexer over per-model binaries (`rlx-qwen3`, `rlx-flux2`, …).
// Prefer the per-crate binary when you only need one family — faster to build.

use rlx_cli::{dispatch, register_cli, run_inspect};
use std::process::ExitCode;

fn register_builtins() {
    register_cli(
        "qwen3",
        "Run a Qwen3 LM (safetensors or gguf)",
        rlx_qwen3::cli::run,
    );
    register_cli(
        "llama32",
        "Run a LLaMA-3.2 / Llama 3.x LM (safetensors or gguf)",
        rlx_llama32::cli::run,
    );
    register_cli(
        "minicpm5",
        "Run MiniCPM5 (Llama-shaped; openbmb/MiniCPM5-1B)",
        rlx_minicpm5::cli_run,
    );
    register_cli(
        "qwen35",
        "Run a Qwen3.5 / Qwen3.6 GGUF (hybrid gated-DeltaNet + attention)",
        rlx_qwen35::cli::run,
    );
    register_cli("sam1", "Segment Anything v1", rlx_sam::cli::run_sam1);
    register_cli("sam2", "Segment Anything v2", rlx_sam2::cli::run);
    register_cli(
        "sam3",
        "Segment Anything v3 (text-conditioned)",
        rlx_sam3::cli::run,
    );
    register_cli(
        "dinov2",
        "DINOv2 ViT encoder / classifier",
        rlx_dinov2::cli::run,
    );
    register_cli(
        "vjepa2",
        "V-JEPA2 video ViT encoder (ViT-G)",
        rlx_vjepa2::cli::run,
    );
    register_cli(
        "wav2vec2-bert",
        "W2v-BERT 2.0 Conformer speech encoder",
        rlx_wav2vec2_bert::cli::run,
    );
    register_cli("flux2", "FLUX.2 denoiser transformer", rlx_flux2::cli::run);
    register_cli(
        "flux2-serve",
        "FLUX.2 persistent server (JSON-lines on stdin)",
        rlx_flux2::cli::run_serve,
    );
    register_cli(
        "inspect",
        "Dump tensor list / format / MTP keys",
        run_inspect,
    );
}

fn main() -> ExitCode {
    register_builtins();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-run: {e:#}");
            ExitCode::FAILURE
        }
    }
}
