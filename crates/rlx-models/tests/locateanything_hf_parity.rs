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

//! LocateAnything parity vs HuggingFace on real weights.
//!
//! ```bash
//! just fetch-locateanything
//! RLX_LOCATEANYTHING_DIR=.cache/locateanything/LocateAnything-3B \
//!   RLX_LOCATEANYTHING_PYTHON=python3 \
//!   cargo test -p rlx-models --test locateanything_hf_parity --release -- --test-threads 1
//! ```

use rlx_core::flow_util::compile_built;
use rlx_core::past_kv_input_names;
use rlx_locateanything::{
    LocateAnythingConfig, LocateAnythingWeightStore,
    embed::{argmax_token, fuse_inputs_embeds_from_store},
    generation::TokenIds,
    lm_flow::{
        build_locateanything_decode_built, build_locateanything_mtp_kv_built,
        build_locateanything_prefill_built, build_locateanything_prefill_mtp_built,
        compute_rope_chunk, compute_rope_slice, qwen3_config,
    },
    mask::{
        attn_bias_for_incremental, decode_custom_mask_from_row, last_row_decode_mask,
        mtp_decode_mask_padded, mtp_prefill_mask_2d,
    },
    moonvit::{MoonVitCache, encode_image, load_moonvit_weights},
    mtp::{decode_bbox_block, handle_pattern},
    preprocess::{PreprocessedImage, preprocess_path},
    projector::build_projector_built,
    session_cache::{LmSessionCaches, kv_state_from_runner},
};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::process::Command;

const PROJECTOR_CACHE: &str = ".rlx_locateanything_projector_parity.txt";
const MOONVIT_CACHE: &str = ".rlx_locateanything_moonvit_parity.txt";
const LM_PREFILL_CACHE: &str = ".rlx_locateanything_lm_prefill_parity.txt";
const LM_MTP_PREFILL_CACHE: &str = ".rlx_locateanything_lm_mtp_prefill_parity.txt";
const LM_MTP_KV_CACHE: &str = ".rlx_locateanything_lm_mtp_kv_parity.txt";
const LM_DECODE_AR_CACHE: &str = ".rlx_locateanything_lm_decode_ar_parity.txt";
const LM_DECODE_MTP_CACHE: &str = ".rlx_locateanything_lm_decode_mtp_parity.txt";
const LM_GREEDY_AR_CACHE: &str = ".rlx_locateanything_lm_greedy_ar_parity.txt";
const LM_GREEDY_FUSED_CACHE: &str = ".rlx_locateanything_lm_greedy_fused_parity.txt";
const E2E_HYBRID_CACHE: &str = ".rlx_locateanything_e2e_hybrid_parity.txt";
const E2E_FAST_CACHE: &str = ".rlx_locateanything_e2e_fast_parity.txt";
const PROMPT_TOKENIZER_CACHE: &str = ".rlx_locateanything_prompt_tokenizer_parity.txt";
const TASK_GROUND_SINGLE_CACHE: &str = ".rlx_locateanything_task_ground_single_parity.txt";
const E2E_GROUND_SINGLE_CACHE: &str = ".rlx_locateanything_e2e_ground_single_parity.txt";
const E2E_HYBRID_LONG_CACHE: &str = ".rlx_locateanything_e2e_hybrid_long_parity.txt";
const PROCESSOR_PROMPT_CACHE: &str = ".rlx_locateanything_processor_prompt_parity.txt";
const TASK_GROUND_MULTI_CACHE: &str = ".rlx_locateanything_task_ground_multi_parity.txt";
const TASK_DETECT_CACHE: &str = ".rlx_locateanything_task_detect_parity.txt";
const E2E_PROCESSOR_CACHE: &str = ".rlx_locateanything_e2e_processor_parity.txt";
const PREPROCESS_REAL_CACHE: &str = ".rlx_locateanything_preprocess_real_parity.txt";
const MOONVIT_REAL_CACHE: &str = ".rlx_locateanything_moonvit_real_parity.txt";
// CPU reference vs HF SDPA on native-resolution grids (larger than 56×56 synth).
const MOONVIT_REAL_MAX_ABS: f32 = 60.0;
const MOONVIT_REAL_MEAN_ABS: f32 = 0.45;
const E2E_PROCESSOR_REAL_CACHE: &str = ".rlx_locateanything_e2e_processor_real_parity.txt";
const REAL_PHRASE: &str = "person";
// Patch tensor vs HF image processor (BICUBIC vs CatmullRom may differ slightly).
const PREPROCESS_PATCH_MAX_ABS: f32 = 0.04;
const PREPROCESS_PATCH_MEAN_ABS: f32 = 0.003;
const PROMPT_N_IMAGE: usize = 4;
const PROMPT_PHRASE: &str =
    "Locate a single instance that matches the following description: red backpack.";
const E2E_GENERATE_NEW: usize = 3;
const LM_MTP_DECODE_CACHE: &str = ".rlx_locateanything_lm_mtp_decode_parity.txt";
const N_TOKENS: usize = 4;
const LM_PREFILL_SEQ: usize = 8;
const LM_MTP_SEQ: usize = 18;
const LM_MTP_PAST_LEN: usize = 12;
const LM_DECODE_PAST_LEN: usize = 12;
const LM_DECODE_TOKEN: u32 = 1000;
const LM_MTP_DECODE_PAST_LEN: usize = 17;
const LM_MTP_DECODE_TOKEN: u32 = 1001;
const LM_MTP_DECODE_BLOCK: usize = 6;
const LM_GREEDY_SEQ: usize = 8;
const LM_GREEDY_NEW: usize = 5;

const PROJECTOR_MAX_ABS: f32 = 2e-2;
const PROJECTOR_MEAN_ABS: f32 = 5e-3;
// CPU reference vs HF (SDPA vs manual attn, GELU, interp): within ~0.26 mean abs on 56×56 probe.
// Compiled encoder vs HF SDPA (spike outliers on a few dims; mean tracks well).
const MOONVIT_MAX_ABS: f32 = 26.0;
const MOONVIT_MEAN_ABS: f32 = 0.35;
const E2E_VIT_PROJECTOR_MAX: f32 = 4.0;
const E2E_VIT_PROJECTOR_MEAN: f32 = 0.65;
// Causal LM prefill (last-token logits, seq=8) vs HF Qwen2 SDPA reference.
const LM_PREFILL_MAX_ABS: f32 = 0.5;
const LM_PREFILL_MEAN_ABS: f32 = 0.02;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> Option<PathBuf> {
    let raw = std::env::var("RLX_LOCATEANYTHING_DIR").ok()?;
    let path = PathBuf::from(&raw);
    if path.is_absolute() || path.join("config.json").is_file() {
        return Some(path);
    }
    let rooted = workspace_root().join(&raw);
    if rooted.join("config.json").is_file() {
        Some(rooted)
    } else {
        Some(path)
    }
}

fn fixture_image() -> PathBuf {
    rlx_locateanything::fixtures::probe_image_path()
}

fn python_bin() -> PathBuf {
    let bin = std::env::var("RLX_LOCATEANYTHING_PYTHON").unwrap_or_else(|_| "python3".into());
    let path = PathBuf::from(&bin);
    // Only resolve paths with a directory component (e.g. `.venv/bin/python`).
    if bin.contains('/') {
        let joined = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../")
            .join(&path);
        joined.canonicalize().unwrap_or(joined)
    } else {
        path
    }
}

fn python_ok() -> bool {
    Command::new(python_bin())
        .args(["-c", "import transformers, torch"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn open_store(dir: &Path) -> LocateAnythingWeightStore {
    LocateAnythingWeightStore::open(&dir.join("model.safetensors.index.json"))
        .or_else(|_| LocateAnythingWeightStore::open(dir))
        .expect("weights")
}

fn run_hf_reference(model_dir: &Path, probe: &str, cache_name: &str) -> String {
    run_hf_reference_with_image(model_dir, probe, cache_name, None)
}

fn run_hf_reference_with_image(
    model_dir: &Path,
    probe: &str,
    cache_name: &str,
    image: Option<&Path>,
) -> String {
    let cache = model_dir.join(cache_name);
    if cache.is_file() && std::env::var("RLX_LOCATEANYTHING_PARITY_REFRESH").is_err() {
        return std::fs::read_to_string(&cache).expect("read HF parity cache");
    }
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/locateanything_parity_helpers/hf_reference.py");
    let mut cmd = Command::new(python_bin());
    cmd.arg(&helper)
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--probe")
        .arg(probe)
        .arg("--n-tokens")
        .arg(N_TOKENS.to_string());
    if let Some(img) = image {
        cmd.arg("--image").arg(img);
    }
    let out = cmd.output().expect("python hf_reference");
    assert!(
        out.status.success(),
        "hf_reference ({probe}) failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    std::fs::write(&cache, &text).expect("write HF parity cache");
    text
}

fn parse_f32_line(tag: &str, text: &str) -> Vec<f32> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(tag) {
            let rest = rest.trim();
            let mut it = rest.split_whitespace();
            let n: usize = it.next().expect("count").parse().expect("count parse");
            let vals: Vec<f32> = it.map(|s| s.parse().expect("float")).collect();
            assert_eq!(vals.len(), n, "{tag} length");
            return vals;
        }
    }
    panic!("missing line {tag}");
}

fn parse_u32_line(tag: &str, text: &str) -> Vec<u32> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(tag) {
            let rest = rest.trim();
            let mut it = rest.split_whitespace();
            let n: usize = it.next().expect("count").parse().expect("count parse");
            let vals: Vec<u32> = it.map(|s| s.parse().expect("u32")).collect();
            assert_eq!(vals.len(), n, "{tag} length");
            return vals;
        }
    }
    panic!("missing line {tag}");
}

fn parse_tag_line(text: &str, tag: &str) -> String {
    let prefix = format!("{tag} ");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return rest.trim().to_string();
        }
    }
    panic!("missing line {tag}");
}

fn parse_meta_usize(text: &str, key: &str) -> usize {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("META") {
            for part in rest.split_whitespace() {
                if let Some((k, v)) = part.split_once('=') {
                    if k == key {
                        return v.parse().expect("meta parse");
                    }
                }
            }
        }
    }
    panic!("missing META {key}");
}

fn bias_mask_delta(a: f32, b: f32) -> f32 {
    if a.is_infinite() && b.is_infinite() {
        0.0
    } else {
        (a - b).abs()
    }
}

fn compare_bias_masks(rlx: &[f32], hf: &[f32], label: &str) {
    assert_eq!(rlx.len(), hf.len(), "{label} length");
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    for (i, (a, b)) in rlx.iter().zip(hf.iter()).enumerate() {
        let d = bias_mask_delta(*a, *b);
        if d > 1e-5 {
            panic!("{label} mismatch at {i}: rlx={a} hf={b}");
        }
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let mean_abs = sum_abs / rlx.len() as f32;
    assert!(
        max_abs <= 1e-5 && mean_abs <= 1e-6,
        "{label} mismatch max_abs={max_abs} mean_abs={mean_abs}"
    );
}

fn compare_token_ids(rlx: &[u32], hf: &[u32], label: &str) {
    assert_eq!(rlx.len(), hf.len(), "{label} length");
    for (i, (a, b)) in rlx.iter().zip(hf.iter()).enumerate() {
        assert_eq!(a, b, "{label} mismatch at step {i}: rlx={a} hf={b}");
    }
}

fn rlx_greedy_ar_after_prefill(
    cfg: &LocateAnythingConfig,
    caches: &mut LmSessionCaches,
    inputs_embeds: &[f32],
    seq: usize,
    n_new: usize,
) -> Vec<u32> {
    let qcfg = qwen3_config(cfg);
    let layers = cfg.text_config.num_hidden_layers;
    let (logits, kv_flat) = caches
        .prefill_with_kv(cfg, seq, inputs_embeds)
        .expect("prefill");
    let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim();
    let mut kv = kv_state_from_runner(seq, &kv_flat, layers, kv_dim).expect("kv state");
    let mut out = vec![argmax_token(&logits)];
    for _ in 1..n_new {
        let past = kv.past_len;
        let (cos, sin) = compute_rope_slice(&qcfg, past);
        let step_logits = caches
            .decode_step_in_place(
                cfg,
                past,
                *out.last().expect("token"),
                &cos,
                &sin,
                None,
                &mut kv,
            )
            .expect("decode");
        out.push(argmax_token(&step_logits));
    }
    out
}

fn compare_vectors(rlx: &[f32], hf: &[f32], label: &str, max_tol: f32, mean_tol: f32) {
    assert_eq!(rlx.len(), hf.len(), "{label} length");
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    for (a, b) in rlx.iter().zip(hf.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let mean_abs = sum_abs / rlx.len() as f32;
    assert!(
        max_abs <= max_tol && mean_abs <= mean_tol,
        "{label} mismatch max_abs={max_abs} mean_abs={mean_abs}"
    );
}

fn ensure_hf_tokenizer_json(dir: &Path) {
    let dst = dir.join("tokenizer.json");
    if dst.is_file() {
        return;
    }
    let script = workspace_root().join("scripts/export_locateanything_tokenizer.py");
    let out = Command::new(python_bin())
        .arg(&script)
        .arg("--model-dir")
        .arg(dir)
        .output()
        .expect("export tokenizer.json");
    if !out.status.success() {
        eprintln!(
            "tokenizer export failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn parity_preflight(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        eprintln!("skip: no config.json in {dir:?}");
        return false;
    }
    if !python_ok() {
        eprintln!(
            "skip: transformers/torch not available ({})",
            python_bin().display()
        );
        return false;
    }
    ensure_hf_tokenizer_json(dir);
    true
}

#[test]
fn locateanything_projector_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "projector", PROJECTOR_CACHE);
    let hf_proj = parse_f32_line("PROJECTOR", &text);
    let vision_in = parse_f32_line("VISION_IN", &text);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_projector_weights().expect("projector weights");
    let built = build_projector_built(&cfg, &mut wm, 1, N_TOKENS).expect("projector built");
    let params = built.model.params().clone();
    let mut compiled = compile_built(built.model, Device::Cpu).expect("compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let rlx = compiled
        .run(&[("vision", vision_in.as_slice())])
        .into_iter()
        .next()
        .expect("projector out");

    compare_vectors(
        &rlx,
        &hf_proj,
        "projector",
        PROJECTOR_MAX_ABS,
        PROJECTOR_MEAN_ABS,
    );
}

#[test]
fn locateanything_moonvit_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let hf_vit = parse_f32_line("MOONVIT", &text);
    let patches = parse_f32_line("PATCHES", &text);
    let grid_h = parse_meta_usize(&text, "grid_h");
    let grid_w = parse_meta_usize(&text, "grid_w");
    let patch_dim = 3 * 14 * 14;

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut wm = store.load_vision_weights().expect("vision weights");
    let mut cache = MoonVitCache::default();
    let compiled = cache
        .encode(&cfg.vision_config, Some(&mut wm), &prep, Device::Cpu)
        .expect("compiled moonvit");

    compare_vectors(
        &compiled,
        &hf_vit,
        "moonvit compiled",
        MOONVIT_MAX_ABS,
        MOONVIT_MEAN_ABS,
    );
}

/// Compiled MoonViT on real weights (synthetic graphs match in `moonvit_compiled`).
#[test]
fn locateanything_moonvit_compiled_real_weights() {
    let Some(dir) = model_dir() else {
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }
    let text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &text);
    let grid_h = parse_meta_usize(&text, "grid_h");
    let grid_w = parse_meta_usize(&text, "grid_w");
    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };
    let mut wm_cpu = store.load_vision_weights().expect("vision");
    let vit = load_moonvit_weights(&mut wm_cpu, &cfg.vision_config).expect("vit");
    let cpu = encode_image(&vit, &prep).expect("cpu");
    let mut wm = store.load_vision_weights().expect("vision");
    let mut cache = MoonVitCache::default();
    let compiled = cache
        .encode(&cfg.vision_config, Some(&mut wm), &prep, Device::Cpu)
        .expect("compiled");
    // CPU uses manual softmax; compiled uses `attention_kind` (closer to HF SDPA).
    const COMPILED_CPU_MAX: f32 = 10.0;
    const COMPILED_CPU_MEAN: f32 = 0.12;
    compare_vectors(
        &compiled,
        &cpu,
        "moonvit compiled",
        COMPILED_CPU_MAX,
        COMPILED_CPU_MEAN,
    );
}

#[test]
fn locateanything_vit_then_projector_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &text);
    let hf_proj = parse_f32_line("PROJECTOR_FROM_VIT", &text);
    let grid_h = parse_meta_usize(&text, "grid_h");
    let grid_w = parse_meta_usize(&text, "grid_w");
    let n_tokens = parse_meta_usize(&text, "n_merged");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };
    let mut wm_vit = store.load_vision_weights().expect("vision");
    let mut vit_cache = MoonVitCache::default();
    let merged = vit_cache
        .encode(&cfg.vision_config, Some(&mut wm_vit), &prep, Device::Cpu)
        .expect("compiled vit");
    let mut wm = store.load_projector_weights().expect("projector");
    let built = build_projector_built(&cfg, &mut wm, 1, n_tokens).expect("projector");
    let params = built.model.params().clone();
    let mut compiled = compile_built(built.model, Device::Cpu).expect("compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let rlx = compiled
        .run(&[("vision", merged.as_slice())])
        .into_iter()
        .next()
        .expect("projector out");

    compare_vectors(
        &rlx,
        &hf_proj,
        "projector(vit)",
        E2E_VIT_PROJECTOR_MAX,
        E2E_VIT_PROJECTOR_MEAN,
    );
}

#[test]
fn locateanything_lm_prefill_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_prefill", LM_PREFILL_CACHE);
    let hf_logits = parse_f32_line("LOGITS_LAST", &text);
    let inputs = parse_f32_line("INPUTS_EMBEDS", &text);
    let seq = parse_meta_usize(&text, "seq");
    assert_eq!(seq, LM_PREFILL_SEQ);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");
    let built =
        build_locateanything_prefill_built(&cfg, &mut wm, 1, seq, false, true).expect("prefill");
    let params = built.params.clone();
    let mut compiled = compile_built(built, Device::Cpu).expect("compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let rlx = compiled
        .run(&[("inputs_embeds", inputs.as_slice())])
        .into_iter()
        .next()
        .expect("logits");

    compare_vectors(
        &rlx,
        &hf_logits,
        "lm prefill",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );
}

#[test]
fn locateanything_lm_mtp_prefill_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_mtp_prefill", LM_MTP_PREFILL_CACHE);
    let hf_logits = parse_f32_line("LOGITS_LAST", &text);
    let inputs = parse_f32_line("INPUTS_EMBEDS", &text);
    let hf_bias = parse_f32_line("ATTN_BIAS", &text);
    let input_ids = parse_u32_line("INPUT_IDS", &text);
    let seq = parse_meta_usize(&text, "seq");
    let block_size = parse_meta_usize(&text, "block_size");
    let text_mask = parse_meta_usize(&text, "text_mask") as u32;
    assert_eq!(seq, LM_MTP_SEQ);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let rlx_mask = mtp_prefill_mask_2d(&input_ids, text_mask, block_size, false, false);
    let per = seq * seq;
    compare_bias_masks(&rlx_mask, &hf_bias[..per], "mtp mask_2d (head0)");

    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");
    let built =
        build_locateanything_prefill_mtp_built(&cfg, &mut wm, 1, seq, true).expect("mtp prefill");
    let params = built.params.clone();
    let mut compiled = compile_built(built, Device::Cpu).expect("compile");
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let rlx = compiled
        .run(&[
            ("inputs_embeds", inputs.as_slice()),
            ("attn_bias", hf_bias.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("logits");

    compare_vectors(
        &rlx,
        &hf_logits,
        "lm mtp prefill",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );
}

#[test]
fn locateanything_lm_mtp_kv_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_mtp_kv", LM_MTP_KV_CACHE);
    let hf_logits = parse_f32_line("LOGITS_LAST", &text);
    let inputs_prefix = parse_f32_line("INPUTS_PREFIX", &text);
    let inputs_query = parse_f32_line("INPUTS_QUERY", &text);
    let hf_bias = parse_f32_line("ATTN_BIAS_INC", &text);
    let input_ids = parse_u32_line("INPUT_IDS", &text);
    let seq = parse_meta_usize(&text, "seq");
    let past_len = parse_meta_usize(&text, "past_len");
    let q_len = parse_meta_usize(&text, "q_len");
    let block_size = parse_meta_usize(&text, "block_size");
    let nh = parse_meta_usize(&text, "num_heads");
    let text_mask = parse_meta_usize(&text, "text_mask") as u32;
    assert_eq!(seq, LM_MTP_SEQ);
    assert_eq!(past_len, LM_MTP_PAST_LEN);
    assert_eq!(q_len, block_size);

    let rlx_mask = mtp_prefill_mask_2d(&input_ids, text_mask, block_size, true, false);
    let rlx_bias = attn_bias_for_incremental(1, nh, past_len, q_len, &rlx_mask, seq);
    compare_bias_masks(&rlx_bias, &hf_bias, "mtp incremental attn_bias");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");

    let prefill = build_locateanything_prefill_built(&cfg, &mut wm, 1, past_len, true, true)
        .expect("causal prefill kv");
    let prefill_params = prefill.params.clone();
    let mut prefill_compiled = compile_built(prefill, Device::Cpu).expect("compile prefill");
    for (n, d) in &prefill_params {
        prefill_compiled.set_param(n, d);
    }
    let prefill_outs = prefill_compiled.run(&[("inputs_embeds", inputs_prefix.as_slice())]);
    let kv_flat: Vec<Vec<f32>> = prefill_outs[1..].to_vec();

    let qcfg = qwen3_config(&cfg);
    let (rope_cos, rope_sin) = compute_rope_chunk(&qcfg, past_len, q_len);
    let layers = cfg.text_config.num_hidden_layers;
    let key_past = past_kv_input_names(layers);

    let mtp = build_locateanything_mtp_kv_built(&cfg, &mut wm, 1, past_len, q_len).expect("mtp kv");
    let mtp_params = mtp.params.clone();
    let mut mtp_compiled = compile_built(mtp, Device::Cpu).expect("compile mtp kv");
    for (n, d) in &mtp_params {
        mtp_compiled.set_param(n, d);
    }
    let mut run_in: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", inputs_query.as_slice()),
        ("attn_bias", hf_bias.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
    ];
    for i in 0..layers {
        run_in.push((key_past[2 * i].as_str(), kv_flat[2 * i].as_slice()));
        run_in.push((key_past[2 * i + 1].as_str(), kv_flat[2 * i + 1].as_slice()));
    }
    let rlx = mtp_compiled
        .run(&run_in)
        .into_iter()
        .next()
        .expect("mtp kv logits");

    compare_vectors(
        &rlx,
        &hf_logits,
        "lm mtp kv",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );
}

#[test]
fn locateanything_lm_decode_ar_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_decode_ar", LM_DECODE_AR_CACHE);
    let hf_logits = parse_f32_line("LOGITS_LAST", &text);
    let inputs_prefix = parse_f32_line("INPUTS_PREFIX", &text);
    let token = parse_u32_line("TOKEN", &text)[0];
    let past_len = parse_meta_usize(&text, "past_len");
    assert_eq!(past_len, LM_DECODE_PAST_LEN);
    assert_eq!(token, LM_DECODE_TOKEN);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");

    let prefill = build_locateanything_prefill_built(&cfg, &mut wm, 1, past_len, true, true)
        .expect("causal prefill kv");
    let prefill_params = prefill.params.clone();
    let mut prefill_compiled = compile_built(prefill, Device::Cpu).expect("compile prefill");
    for (n, d) in &prefill_params {
        prefill_compiled.set_param(n, d);
    }
    let prefill_outs = prefill_compiled.run(&[("inputs_embeds", inputs_prefix.as_slice())]);
    let kv_flat: Vec<Vec<f32>> = prefill_outs[1..].to_vec();

    let qcfg = qwen3_config(&cfg);
    let (rope_cos, rope_sin) = compute_rope_slice(&qcfg, past_len);
    let layers = cfg.text_config.num_hidden_layers;
    let key_past = past_kv_input_names(layers);
    let token_f = [token as f32];

    let decode =
        build_locateanything_decode_built(&cfg, &mut wm, 1, past_len, false).expect("decode");
    let decode_params = decode.params.clone();
    let mut decode_compiled = compile_built(decode, Device::Cpu).expect("compile decode");
    for (n, d) in &decode_params {
        decode_compiled.set_param(n, d);
    }
    let mut run_in: Vec<(&str, &[f32])> = vec![
        ("input_ids", token_f.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
    ];
    for i in 0..layers {
        run_in.push((key_past[2 * i].as_str(), kv_flat[2 * i].as_slice()));
        run_in.push((key_past[2 * i + 1].as_str(), kv_flat[2 * i + 1].as_slice()));
    }
    let rlx = decode_compiled
        .run(&run_in)
        .into_iter()
        .next()
        .expect("decode logits");

    compare_vectors(
        &rlx,
        &hf_logits,
        "lm decode ar",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );
}

#[test]
fn locateanything_lm_decode_mtp_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_decode_mtp", LM_DECODE_MTP_CACHE);
    let hf_logits = parse_f32_line("LOGITS_LAST", &text);
    let inputs_prefix = parse_f32_line("INPUTS_PREFIX", &text);
    let hf_mask_row = parse_f32_line("MASK_ROW", &text);
    let token = parse_u32_line("TOKEN", &text)[0];
    let past_len = parse_meta_usize(&text, "past_len");
    let block_size = parse_meta_usize(&text, "block_size");
    assert_eq!(past_len, LM_MTP_DECODE_PAST_LEN);
    assert_eq!(token, LM_MTP_DECODE_TOKEN);
    assert_eq!(block_size, LM_MTP_DECODE_BLOCK);

    let rlx_row = last_row_decode_mask(block_size, past_len);
    compare_bias_masks(&rlx_row, &hf_mask_row, "mtp decode mask row");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");

    let prefill = build_locateanything_prefill_built(&cfg, &mut wm, 1, past_len, true, true)
        .expect("causal prefill kv");
    let prefill_params = prefill.params.clone();
    let mut prefill_compiled = compile_built(prefill, Device::Cpu).expect("compile prefill");
    for (n, d) in &prefill_params {
        prefill_compiled.set_param(n, d);
    }
    let prefill_outs = prefill_compiled.run(&[("inputs_embeds", inputs_prefix.as_slice())]);
    let kv_flat: Vec<Vec<f32>> = prefill_outs[1..].to_vec();

    let qcfg = qwen3_config(&cfg);
    let (rope_cos, rope_sin) = compute_rope_slice(&qcfg, past_len);
    let layers = cfg.text_config.num_hidden_layers;
    let key_past = past_kv_input_names(layers);
    let token_f = [token as f32];
    let cap_len = past_len + 1;
    let custom_mask = mtp_decode_mask_padded(block_size, past_len, cap_len);
    let from_row = decode_custom_mask_from_row(&rlx_row);
    assert_eq!(custom_mask.len(), cap_len);
    assert_eq!(custom_mask, from_row, "mtp_decode_mask_padded vs row");

    let decode =
        build_locateanything_decode_built(&cfg, &mut wm, 1, past_len, true).expect("mtp decode");
    let decode_params = decode.params.clone();
    let mut decode_compiled = compile_built(decode, Device::Cpu).expect("compile decode");
    for (n, d) in &decode_params {
        decode_compiled.set_param(n, d);
    }
    let mut run_in: Vec<(&str, &[f32])> = vec![
        ("input_ids", token_f.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
        ("mask", custom_mask.as_slice()),
    ];
    for i in 0..layers {
        run_in.push((key_past[2 * i].as_str(), kv_flat[2 * i].as_slice()));
        run_in.push((key_past[2 * i + 1].as_str(), kv_flat[2 * i + 1].as_slice()));
    }
    let rlx = decode_compiled
        .run(&run_in)
        .into_iter()
        .next()
        .expect("mtp decode logits");

    compare_vectors(
        &rlx,
        &hf_logits,
        "lm decode mtp",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );
}

#[test]
fn locateanything_lm_greedy_ar_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_greedy_ar", LM_GREEDY_AR_CACHE);
    let hf_tokens = parse_u32_line("GENERATED_IDS", &text);
    let inputs = parse_f32_line("INPUTS_EMBEDS", &text);
    let seq = parse_meta_usize(&text, "seq");
    let n_new = parse_meta_usize(&text, "n_new");
    assert_eq!(seq, LM_GREEDY_SEQ);
    assert_eq!(n_new, LM_GREEDY_NEW);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut caches = LmSessionCaches::new(Device::Cpu, 64);
    caches.ensure_lm_store(std::sync::Arc::new(store.clone()));

    let rlx = rlx_greedy_ar_after_prefill(&cfg, &mut caches, &inputs, seq, n_new);
    compare_token_ids(&rlx, &hf_tokens, "lm greedy ar");
}

#[test]
fn locateanything_lm_greedy_fused_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_greedy_fused", LM_GREEDY_FUSED_CACHE);
    let hf_tokens = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let vision = parse_f32_line("VISION_EMBEDS", &text);
    let seq = parse_meta_usize(&text, "seq");
    let n_new = parse_meta_usize(&text, "n_new");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let fused = fuse_inputs_embeds_from_store(&cfg, &store, &prompt_ids, &vision).expect("fuse");
    assert_eq!(fused.len(), seq * cfg.text_config.hidden_size);

    let mut caches = LmSessionCaches::new(Device::Cpu, 256);
    caches.ensure_lm_store(std::sync::Arc::new(store.clone()));

    let rlx = rlx_greedy_ar_after_prefill(&cfg, &mut caches, &fused, seq, n_new);
    compare_token_ids(&rlx, &hf_tokens, "lm greedy fused");
}

#[test]
fn locateanything_fuse_embeds_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_greedy_fused", LM_GREEDY_FUSED_CACHE);
    let hf_fused = parse_f32_line("FUSED_EMBEDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let vision = parse_f32_line("VISION_EMBEDS", &text);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let rlx_fused =
        fuse_inputs_embeds_from_store(&cfg, &store, &prompt_ids, &vision).expect("fuse");
    compare_vectors(&rlx_fused, &hf_fused, "fuse_inputs_embeds", 1e-4, 1e-5);
}

#[test]
fn locateanything_runner_slow_from_embeds_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_greedy_fused", LM_GREEDY_FUSED_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let vision = parse_f32_line("VISION_EMBEDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let fused = fuse_inputs_embeds_from_store(&cfg, &store, &prompt_ids, &vision).expect("fuse");

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Slow)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let full = runner
        .generate_from_embeds(&prompt_ids, &fused, None)
        .expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "runner slow greedy");
}

#[test]
fn locateanything_runner_slow_rlx_vision_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_greedy_fused", LM_GREEDY_FUSED_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Slow)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let full = runner.generate(&prompt_ids, &prep).expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "runner slow rlx vision");
}

#[test]
fn locateanything_lm_mtp_block_decode_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "lm_mtp_decode", LM_MTP_DECODE_CACHE);
    let hf_logits = parse_f32_line("LOGITS_BLOCK", &text);
    let hf_box = parse_u32_line("BOX_TOKENS", &text);
    let hf_pat = parse_u32_line("PATTERN_TOKENS", &text);
    let inputs_prefix = parse_f32_line("INPUTS_PREFIX", &text);
    let inputs_query = parse_f32_line("INPUTS_QUERY", &text);
    let hf_bias = parse_f32_line("ATTN_BIAS_INC", &text);
    let past_len = parse_meta_usize(&text, "past_len");
    let q_len = parse_meta_usize(&text, "block_size");
    let vocab = parse_meta_usize(&text, "vocab");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let mut wm = store.load_language_model_weights().expect("lm weights");

    let prefill = build_locateanything_prefill_built(&cfg, &mut wm, 1, past_len, true, true)
        .expect("prefill");
    let prefill_params = prefill.params.clone();
    let mut prefill_compiled = compile_built(prefill, Device::Cpu).expect("compile prefill");
    for (n, d) in &prefill_params {
        prefill_compiled.set_param(n, d);
    }
    let prefill_outs = prefill_compiled.run(&[("inputs_embeds", inputs_prefix.as_slice())]);
    let kv_flat: Vec<Vec<f32>> = prefill_outs[1..].to_vec();

    let qcfg = qwen3_config(&cfg);
    let (rope_cos, rope_sin) = compute_rope_chunk(&qcfg, past_len, q_len);
    let layers = cfg.text_config.num_hidden_layers;
    let key_past = past_kv_input_names(layers);

    let mtp = build_locateanything_mtp_kv_built(&cfg, &mut wm, 1, past_len, q_len).expect("mtp kv");
    let mtp_params = mtp.params.clone();
    let mut mtp_compiled = compile_built(mtp, Device::Cpu).expect("compile mtp kv");
    for (n, d) in &mtp_params {
        mtp_compiled.set_param(n, d);
    }
    let mut run_in: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", inputs_query.as_slice()),
        ("attn_bias", hf_bias.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
    ];
    for i in 0..layers {
        run_in.push((key_past[2 * i].as_str(), kv_flat[2 * i].as_slice()));
        run_in.push((key_past[2 * i + 1].as_str(), kv_flat[2 * i + 1].as_slice()));
    }
    let rlx_logits = mtp_compiled
        .run(&run_in)
        .into_iter()
        .next()
        .expect("mtp logits");
    assert_eq!(rlx_logits.len(), hf_logits.len());
    compare_vectors(
        &rlx_logits,
        &hf_logits,
        "mtp block logits",
        LM_PREFILL_MAX_ABS,
        LM_PREFILL_MEAN_ABS,
    );

    let ids = TokenIds::from_config(&cfg);
    let rlx_box = decode_bbox_block(&rlx_logits, vocab, &ids, "hybrid");
    if hf_box.is_empty() {
        assert!(rlx_box.is_none(), "rlx expected no box");
    } else {
        let rlx_box = rlx_box.expect("rlx box");
        compare_token_ids(&rlx_box, &hf_box, "mtp box decode");
        let rlx_pat = handle_pattern(&rlx_box, &ids, "hybrid");
        compare_token_ids(&rlx_pat.tokens, &hf_pat, "mtp handle_pattern");
    }
}

#[test]
fn locateanything_runner_hybrid_e2e_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "e2e_hybrid", E2E_HYBRID_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");
    assert_eq!(n_new, E2E_GENERATE_NEW);

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Hybrid)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let full = runner
        .generate(&prompt_ids, &prep)
        .expect("hybrid generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "runner hybrid e2e");
}

#[test]
fn locateanything_runner_fast_e2e_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "e2e_fast", E2E_FAST_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");
    assert_eq!(n_new, E2E_GENERATE_NEW);

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Fast)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let full = runner.generate(&prompt_ids, &prep).expect("fast generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "runner fast e2e");
}

#[test]
fn locateanything_prompt_tokenizer_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "prompt_tokenizer", PROMPT_TOKENIZER_CACHE);
    let hf_ids = parse_u32_line("PROMPT_IDS", &text);
    let n_image = parse_meta_usize(&text, "n_image");
    assert_eq!(n_image, PROMPT_N_IMAGE);

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let tok = rlx_locateanything::tokenizer::load_tokenizer(&dir).expect("tokenizer");
    let rlx_ids =
        rlx_locateanything::tokenizer::build_user_prompt_ids(&cfg, &tok, PROMPT_PHRASE, n_image)
            .expect("prompt ids");
    compare_token_ids(&rlx_ids, &hf_ids, "prompt tokenizer");
}

#[test]
fn locateanything_task_ground_single_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "task_ground_single", TASK_GROUND_SINGLE_CACHE);
    let hf_text = parse_tag_line(&text, "USER_TEXT");
    let phrase = parse_tag_line(&text, "PHRASE");
    let rlx_text = rlx_locateanything::prompts::ground_single(&phrase);
    assert_eq!(rlx_text, hf_text, "ground-single task prompt");
}

#[test]
fn locateanything_runner_ground_single_hybrid_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "e2e_ground_single", E2E_GROUND_SINGLE_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let phrase = "red backpack";
    let user_text = rlx_locateanything::prompts::ground_single(phrase);

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Hybrid)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let prompt_ids = runner
        .build_prompt_from_text(&user_text, &prep)
        .expect("prompt");
    let full = runner.generate(&prompt_ids, &prep).expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "ground-single hybrid e2e");
}

#[test]
fn locateanything_runner_hybrid_long_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "e2e_hybrid_long", E2E_HYBRID_LONG_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let prompt_ids = parse_u32_line("INPUT_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");
    assert_eq!(n_new, 8);

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Hybrid)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let full = runner.generate(&prompt_ids, &prep).expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "hybrid long e2e");
}

#[test]
fn locateanything_processor_prompt_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "processor_prompt", PROCESSOR_PROMPT_CACHE);
    let hf_ids = parse_u32_line("PROMPT_IDS", &text);
    let user_with_ph = parse_tag_line(&text, "USER_TEXT");
    let n_image = parse_meta_usize(&text, "n_image");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let tok = rlx_locateanything::tokenizer::load_tokenizer(&dir).expect("tokenizer");
    let rlx_ids = rlx_locateanything::processor_prompt::build_processor_prompt_ids(
        &dir,
        &cfg,
        &tok,
        &user_with_ph,
        n_image,
    )
    .expect("processor prompt ids");
    compare_token_ids(&rlx_ids, &hf_ids, "processor prompt");
}

#[test]
fn locateanything_task_ground_multi_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "task_ground_multi", TASK_GROUND_MULTI_CACHE);
    let hf_text = parse_tag_line(&text, "USER_TEXT");
    let phrase = parse_tag_line(&text, "PHRASE");
    let rlx_text = rlx_locateanything::prompts::ground_multi(&phrase);
    assert_eq!(rlx_text, hf_text, "ground-multi task prompt");
}

#[test]
fn locateanything_task_detect_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "task_detect", TASK_DETECT_CACHE);
    let hf_text = parse_tag_line(&text, "USER_TEXT");
    let cats_line = parse_tag_line(&text, "CATEGORIES");
    let cats: Vec<&str> = cats_line.split("</c>").collect();
    let rlx_text = rlx_locateanything::prompts::detect(&cats);
    assert_eq!(rlx_text, hf_text, "detect task prompt");
}

#[test]
fn locateanything_runner_processor_hybrid_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !parity_preflight(&dir) {
        return;
    }

    let text = run_hf_reference(&dir, "e2e_processor", E2E_PROCESSOR_CACHE);
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");

    let moonvit_text = run_hf_reference(&dir, "moonvit", MOONVIT_CACHE);
    let patches = parse_f32_line("PATCHES", &moonvit_text);
    let grid_h = parse_meta_usize(&moonvit_text, "grid_h");
    let grid_w = parse_meta_usize(&moonvit_text, "grid_w");
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let user_with_ph =
        rlx_locateanything::processor_prompt::ground_single_with_image_placeholder("red backpack");

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Hybrid)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let prompt_ids = runner
        .build_prompt_processor(&user_with_ph, &prep)
        .expect("processor prompt");
    let full = runner.generate(&prompt_ids, &prep).expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "processor hybrid e2e");
}

fn real_photo_preflight(dir: &Path) -> bool {
    if !parity_preflight(dir) {
        return false;
    }
    let img = fixture_image();
    if !img.is_file() {
        eprintln!("skip: missing fixture image {}", img.display());
        return false;
    }
    true
}

#[test]
fn locateanything_preprocess_real_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !real_photo_preflight(&dir) {
        return;
    }
    let img = fixture_image();
    let text =
        run_hf_reference_with_image(&dir, "preprocess_real", PREPROCESS_REAL_CACHE, Some(&img));
    let hf_patches = parse_f32_line("PATCHES", &text);
    let grid_h = parse_meta_usize(&text, "grid_h");
    let grid_w = parse_meta_usize(&text, "grid_w");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let prep = preprocess_path(&img, &cfg).expect("preprocess");
    assert_eq!(prep.grid_h, grid_h, "grid_h");
    assert_eq!(prep.grid_w, grid_w, "grid_w");
    assert_eq!(prep.patches.len(), hf_patches.len(), "patches len");
    compare_vectors(
        &prep.patches,
        &hf_patches,
        "preprocess patches (real photo)",
        PREPROCESS_PATCH_MAX_ABS,
        PREPROCESS_PATCH_MEAN_ABS,
    );
}

#[test]
fn locateanything_moonvit_real_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !real_photo_preflight(&dir) {
        return;
    }
    let img = fixture_image();
    let text = run_hf_reference_with_image(&dir, "moonvit_real", MOONVIT_REAL_CACHE, Some(&img));
    let hf_vit = parse_f32_line("MOONVIT", &text);
    let patches = parse_f32_line("PATCHES", &text);
    let grid_h = parse_meta_usize(&text, "grid_h");
    let grid_w = parse_meta_usize(&text, "grid_w");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let store = open_store(&dir);
    let prep = PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim: 3 * 14 * 14,
        pixel_w: (grid_w * 14) as u32,
        pixel_h: (grid_h * 14) as u32,
    };

    let mut wm = store.load_vision_weights().expect("vision");
    let vit = load_moonvit_weights(&mut wm, &cfg.vision_config).expect("vit");
    let cpu = encode_image(&vit, &prep).expect("cpu moonvit");
    compare_vectors(
        &cpu,
        &hf_vit,
        "moonvit real photo",
        MOONVIT_REAL_MAX_ABS,
        MOONVIT_REAL_MEAN_ABS,
    );
}

#[test]
fn locateanything_runner_processor_real_hf_parity() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    if !real_photo_preflight(&dir) {
        return;
    }
    let img_path = fixture_image();
    let text = run_hf_reference_with_image(
        &dir,
        "e2e_processor_real",
        E2E_PROCESSOR_REAL_CACHE,
        Some(&img_path),
    );
    let hf_new = parse_u32_line("GENERATED_IDS", &text);
    let n_new = parse_meta_usize(&text, "n_new");

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let prep = preprocess_path(&img_path, &cfg).expect("preprocess");

    let user_with_ph =
        rlx_locateanything::processor_prompt::ground_single_with_image_placeholder(REAL_PHRASE);

    let mut runner = rlx_locateanything::LocateAnythingRunner::builder()
        .weights(dir.as_path())
        .device(Device::Cpu)
        .max_new_tokens(n_new)
        .generation_mode(rlx_locateanything::GenerationMode::Hybrid)
        .temperature(0.0)
        .repetition_penalty(1.0)
        .build()
        .expect("runner");
    let prompt_ids = runner
        .build_prompt_processor(&user_with_ph, &prep)
        .expect("processor prompt");
    let kh = runner.cfg.vision_config.merge_kernel_size[0];
    let kw = runner.cfg.vision_config.merge_kernel_size[1];
    let n_slots = (prep.grid_h / kh) * (prep.grid_w / kw);
    let img_tok = runner.cfg.image_token_index;
    let n_placeholders = prompt_ids.iter().filter(|&&t| t == img_tok).count();
    assert_eq!(
        n_placeholders, n_slots,
        "processor prompt must include {n_slots} image_token_index ({img_tok}) placeholders, got {n_placeholders}"
    );
    let full = runner.generate(&prompt_ids, &prep).expect("generate");
    let rlx_new = &full[prompt_ids.len()..];
    compare_token_ids(rlx_new, &hf_new, "processor hybrid real photo e2e");
}
