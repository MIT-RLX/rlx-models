//! spec_decode_bench — speculative decoding (prototype 3).
//!
//! Uses the existing `SpecDecoder` engine + `Qwen3Speculator` adapter (both
//! already unit-tested). Draft = Q4_K_M gguf, target = F32 safetensors (same
//! tokenizer). Measures the ACCEPT RATE — tokens accepted per round — which is
//! the speculative-decoding ceiling: how often the cheap draft agrees with the
//! target. (Wall-clock is not expected to beat plain decode here: the shipped
//! verify path is a sequential host-decode loop, and both models run the same
//! F32 path so there is no compute asymmetry — see the printed note.)
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example spec_decode_bench --features metal

use rlx_qwen3::{Qwen3Config, Qwen3Generator, Qwen3Speculator};
use rlx_runtime::Device;
use rlx_runtime::spec_decode::SpecDecoder;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let dir = "/Users/Shared/weights/qwen3-0.6b";
    // Both draft and target run NATIVE-PACKED (Op::DequantMatMul from GGUF, ~4×
    // fewer weight bytes/token than F32) — so the draft is genuinely cheaper.
    let target_gguf = std::env::var("SPEC_TARGET").unwrap_or_else(|_| {
        "/Users/Shared/weights/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    let gguf = std::env::var("SPEC_DRAFT").unwrap_or_else(|_| {
        "/Users/Shared/weights/qwen3-0.6b-gguf/Qwen3-0.6B-Q2_K.gguf".to_string()
    });
    for k in [
        "RLX_QWEN3_F16_WEIGHTS",
        "RLX_QWEN3_BAKE_WEIGHTS",
        "RLX_QWEN3_GQA_NATIVE",
    ] {
        if std::env::var_os(k).is_none() {
            unsafe { std::env::set_var(k, "1") };
        }
    }
    let cfg_t = Qwen3Config::from_file(Path::new(dir).join("config.json").as_path())?;
    let cfg_d = Qwen3Config::from_file(Path::new(dir).join("config.json").as_path())?;
    let tok = Tokenizer::from_file(Path::new(dir).join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    eprintln!("[spec] loading target packed: {target_gguf}");
    let target = Qwen3Generator::new_native_packed_decode(cfg_t, &target_gguf, Device::Metal)?;
    eprintln!("[spec] loading draft  packed: {gguf}");
    let draft = Qwen3Generator::new_native_packed_decode(cfg_d, &gguf, Device::Metal)?;

    let n_draft: usize = std::env::var("SPEC_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let want: usize = std::env::var("SPEC_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let mut dec = SpecDecoder::new(
        Qwen3Speculator::new(draft),
        Qwen3Speculator::new(target),
        n_draft,
        42,
    );

    let prompt = tok
        .encode(
            "The history of the Roman Empire is a story of ambition and conquest. It began",
            false,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();

    eprintln!("[spec] running {want}-token speculative decode, n_draft={n_draft} …");
    let mut ctx = prompt.clone();
    let mut rounds = 0usize;
    let t = Instant::now();
    while ctx.len() < prompt.len() + want {
        let new = dec.step(&ctx);
        if new.is_empty() {
            break;
        }
        ctx.extend(new);
        rounds += 1;
    }
    let secs = t.elapsed().as_secs_f64();
    let n_gen = ctx.len() - prompt.len();
    let per_round = n_gen as f64 / rounds.max(1) as f64;
    // tokens/round = 1 (guaranteed) + accepted-draft; accept fraction of proposals.
    let accept_frac = ((per_round - 1.0) / n_draft as f64).clamp(0.0, 1.0);
    println!("\n── speculative decoding: Q4 draft ⇒ F32 target ──");
    println!("n_draft={n_draft}  generated={n_gen} tokens in {rounds} rounds");
    println!(
        "  ACCEPT RATE: {per_round:.2} tokens/round (max {}), {:.0}% of drafted tokens accepted",
        n_draft + 1,
        accept_frac * 100.0
    );
    println!(
        "  → with a cheap draft + batched verify, that's a ~{per_round:.1}× step-amortization ceiling"
    );
    println!("  wall: {secs:.1}s = {:.2} tok/s", n_gen as f64 / secs);
    println!(
        "  NOTE: both draft+target run native-packed (draft IS cheaper by bytes).\n\
         But a quant of the SAME 0.6B has the same architecture → only ~1.3× cheaper\n\
         per token while acceptance drops (Q2 35% vs Q4 75%). Spec needs a draft\n\
         ~5-10× cheaper (a structurally smaller model / MTP head) — none exists <0.6B.\n\
         Wall-clock also blocked by the adapter re-prefilling context every round."
    );
    Ok(())
}
