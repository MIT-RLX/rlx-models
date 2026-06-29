// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// HuggingFace Qwen2.5-VL parity vs Docker (or local Python) reference dump.
//
// ```bash
// # build reference image once
// crates/rlx-models/tests/qwen25_vl_parity_helpers/build.sh
//
// # vision-only (mmproj + image; no LM GGUF)
// RLX_QWEN25_VL_DOCKER=1 \
// RLX_QWEN25_VL_HF_DIR=/path/to/Qwen2.5-VL-7B-Instruct \
// RLX_QWEN25_VL_MMPROJ_PATH=/path/to/mmproj.gguf \
// RLX_QWEN25_VL_IMAGE=/path/to/sample.png \
// cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_vision_embed_parity --features qwen25-vl --release -- --nocapture
//
// # full multimodal logits
// RLX_QWEN25_VL_GGUF_PATH=/path/to/lm.gguf \
// ... same env ...
// cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_hf_logits_parity --features qwen25-vl --release -- --nocapture
// ```

mod qwen25_vl_parity_helpers {
    include!("qwen25_vl_parity_helpers/mod.rs");
}

use qwen25_vl_parity_helpers::{
    ReferenceDump, cosine_distance, load_reference_dump, max_abs_diff, run_docker_reference,
    run_reference, run_reference_with_env, top1_match,
};
use rlx_qwen25_vl::vision::{Qwen25VlVisionEncoder, load_rgb_image};
use rlx_qwen25_vl::{AifConfig, AifDynamicsMode, AifProbe, Qwen25VlRunner, Qwen25VlRunnerBuilder};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const TOL_VISION_COS: f64 = 0.02;
const TOL_HIDDEN_COS: f64 = 0.05;
const WARN_LOGITS_MAX_DIFF: f32 = 2.0;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn ref_runner_ready() -> bool {
    std::env::var("RLX_QWEN25_VL_DOCKER").ok().as_deref() == Some("1")
        || std::env::var("RLX_QWEN25_VL_PYTHON").is_ok()
}

fn hf_model_ready() -> bool {
    std::env::var("RLX_QWEN25_VL_HF_DIR").is_ok()
        || std::env::var("RLX_QWEN25_VL_DOWNLOAD").ok().as_deref() == Some("1")
}

fn load_image_path() -> Option<PathBuf> {
    env_path("RLX_QWEN25_VL_IMAGE").filter(|p| p.exists())
}

fn dump_reference(image: &Path) -> ReferenceDump {
    dump_reference_with_env(image, &[])
}

fn dump_reference_with_env(image: &Path, extra_env: &[(&str, &str)]) -> ReferenceDump {
    let out_dir = tempfile::tempdir().expect("tempdir");
    let use_docker = std::env::var("RLX_QWEN25_VL_DOCKER").ok().as_deref() == Some("1");
    run_reference_with_env(image, out_dir.path(), use_docker, extra_env).expect("reference dump")
}

fn encode_rlx_vision(
    mmproj: &Path,
    rgb: &[u8],
    w: usize,
    h: usize,
    reference: &ReferenceDump,
) -> rlx_qwen25_vl::vision::VisionEncodeOutput {
    let (tw, th) = (
        reference.resized_w.expect("reference resized_w"),
        reference.resized_h.expect("reference resized_h"),
    );
    let mut enc = Qwen25VlVisionEncoder::from_mmproj(mmproj, tw, th).expect("vision encoder");
    enc.encode_rgb_resized(rgb, w, h, Some(tw), Some(th))
        .expect("rlx vision encode")
}

fn assert_vision_token_grid(reference: &ReferenceDump, n_tokens: usize) {
    if let (Some(gh), Some(gw)) = (reference.grid_h, reference.grid_w) {
        let merge = 2usize;
        let expected = gh * gw / (merge * merge);
        assert_eq!(
            reference.n_vision_tokens, expected,
            "reference vision token count vs grid_thw"
        );
        assert_eq!(
            n_tokens, expected,
            "rlx vision tokens vs hf grid ({gh}x{gw} patches)"
        );
    }
}

#[test]
fn qwen25_vl_vision_embed_parity() {
    if !ref_runner_ready() {
        eprintln!("skip vision parity: set RLX_QWEN25_VL_DOCKER=1 or RLX_QWEN25_VL_PYTHON");
        return;
    }
    if !hf_model_ready() {
        eprintln!("skip vision parity: set RLX_QWEN25_VL_HF_DIR or RLX_QWEN25_VL_DOWNLOAD=1");
        return;
    }
    let Some(mmproj) = env_path("RLX_QWEN25_VL_MMPROJ_PATH").filter(|p| p.exists()) else {
        eprintln!("skip vision parity: set RLX_QWEN25_VL_MMPROJ_PATH");
        return;
    };
    let Some(image) = load_image_path() else {
        eprintln!("skip vision parity: set RLX_QWEN25_VL_IMAGE");
        return;
    };

    let reference = dump_reference(&image);
    let Some(hf_emb) = reference.vision_embeddings.as_ref() else {
        eprintln!(
            "skip vision parity: reference missing vision_embeddings.npy (rebuild docker image)"
        );
        return;
    };

    let (rgb, w, h) = load_rgb_image(image.to_str().expect("utf8")).expect("load image");
    let vision = encode_rlx_vision(&mmproj, &rgb, w, h, &reference);

    assert_vision_token_grid(&reference, vision.n_tokens);
    assert_eq!(vision.n_tokens, reference.n_vision_tokens);

    let proj = reference
        .vision_proj_dim
        .unwrap_or(vision.embeddings.len() / vision.n_tokens);
    assert_eq!(hf_emb.len(), vision.n_tokens * proj);
    assert_eq!(vision.embeddings.len(), hf_emb.len());

    let (mad, idx) = max_abs_diff(&vision.embeddings, hf_emb);
    let cos = cosine_distance(&vision.embeddings, hf_emb);

    eprintln!(
        "qwen25_vl_vision_parity tokens={} proj={} cos={cos:.4e} mad={mad:.4} idx={idx} \
         resize={:?}x{:?} grid={:?}x{:?} rlx0={:?} hf0={:?}",
        vision.n_tokens,
        proj,
        reference.resized_w,
        reference.resized_h,
        reference.grid_w,
        reference.grid_h,
        &vision.embeddings[..4.min(vision.embeddings.len())],
        &hf_emb[..4.min(hf_emb.len())],
    );

    if cos > TOL_VISION_COS {
        eprintln!(
            "warning: vision embed cosine {cos:.4e} > {TOL_VISION_COS} \
             (quant / window-attn / resize may still diverge)"
        );
    }
    assert!(
        cos <= TOL_VISION_COS * 5.0,
        "vision embed cosine {cos:.4e} > {}",
        TOL_VISION_COS * 5.0
    );
}

#[test]
fn qwen25_vl_hf_logits_parity() {
    if !ref_runner_ready() {
        eprintln!("skip qwen25_vl_hf_parity: set RLX_QWEN25_VL_DOCKER=1 or RLX_QWEN25_VL_PYTHON");
        return;
    }
    if !hf_model_ready() {
        eprintln!("skip qwen25_vl_hf_parity: set RLX_QWEN25_VL_HF_DIR or RLX_QWEN25_VL_DOWNLOAD=1");
        return;
    }
    let Some(gguf) = env_path("RLX_QWEN25_VL_GGUF_PATH").filter(|p| p.exists()) else {
        eprintln!("skip qwen25_vl_hf_parity: set RLX_QWEN25_VL_GGUF_PATH");
        return;
    };
    let Some(mmproj) = env_path("RLX_QWEN25_VL_MMPROJ_PATH").filter(|p| p.exists()) else {
        eprintln!("skip qwen25_vl_hf_parity: set RLX_QWEN25_VL_MMPROJ_PATH");
        return;
    };
    let Some(image) = load_image_path() else {
        eprintln!("skip qwen25_vl_hf_parity: set RLX_QWEN25_VL_IMAGE");
        return;
    };

    let reference = dump_reference(&image);
    let (rgb, w, h) = load_rgb_image(image.to_str().expect("utf8")).expect("load image");

    let mut runner = Qwen25VlRunnerBuilder::default()
        .weights(&gguf)
        .mmproj(&mmproj)
        .device(Device::Cpu)
        .max_seq(reference.seq_len.max(128))
        .build()
        .expect("runner");

    let target = match (reference.resized_w, reference.resized_h) {
        (Some(tw), Some(th)) => (Some(tw), Some(th)),
        _ => (None, None),
    };
    let vision = runner
        .encode_image_resized(&rgb, w, h, target.0, target.1)
        .expect("vision encode");

    assert_vision_token_grid(&reference, vision.n_tokens);
    if vision.n_tokens != reference.n_vision_tokens {
        eprintln!(
            "skip qwen25_vl_hf_logits_parity: vision token count rlx={} hf={}",
            vision.n_tokens, reference.n_vision_tokens
        );
        return;
    }

    let logits = runner
        .prefill_from_token_ids(
            &reference.input_ids,
            reference.vision_start_idx,
            reference.n_vision_tokens,
            &vision,
            0,
        )
        .expect("rlx prefill");

    let hidden = runner.last_prefill_hidden().expect("last hidden").to_vec();

    assert_eq!(logits.len(), reference.vocab_size);
    assert_eq!(hidden.len(), reference.hidden_size);

    let (logits_mad, logits_idx) = max_abs_diff(&logits, &reference.logits);
    let logits_cos = cosine_distance(&logits, &reference.logits);
    let (hidden_mad, hidden_idx) = max_abs_diff(&hidden, &reference.hidden);
    let hidden_cos = cosine_distance(&hidden, &reference.hidden);
    let top1 = top1_match(&logits, &reference.logits);

    eprintln!(
        "qwen25_vl_hf_parity seq={} vision={}@{} top1={top1} \
         logits cos={logits_cos:.4e} mad={logits_mad:.4} idx={logits_idx} \
         hidden cos={hidden_cos:.4e} mad={hidden_mad:.4} idx={hidden_idx}",
        reference.seq_len, reference.n_vision_tokens, reference.vision_start_idx,
    );

    if logits_mad > WARN_LOGITS_MAX_DIFF {
        eprintln!("warning: logits max_abs_diff {logits_mad:.4} > {WARN_LOGITS_MAX_DIFF}");
    }
    assert!(
        hidden_cos <= TOL_HIDDEN_COS || top1,
        "hidden cosine {hidden_cos:.4e} > {TOL_HIDDEN_COS} and top1 token mismatch"
    );
}

#[test]
fn qwen25_vl_reference_dump_reload() {
    if !ref_runner_ready() || !hf_model_ready() {
        return;
    }
    let Some(image) = load_image_path() else {
        return;
    };

    let out_dir = tempfile::tempdir().expect("tempdir");
    let use_docker = std::env::var("RLX_QWEN25_VL_DOCKER").ok().as_deref() == Some("1");
    if use_docker {
        run_docker_reference(&image, out_dir.path()).expect("docker reference");
    } else {
        run_reference(&image, out_dir.path(), false).expect("python reference");
    };
    let reference = load_reference_dump(out_dir.path()).expect("reload");
    assert_eq!(reference.input_ids.len(), reference.seq_len);
    assert_eq!(reference.logits.len(), reference.vocab_size);
    assert_eq!(reference.hidden.len(), reference.hidden_size);
    if let Some(ref emb) = reference.vision_embeddings {
        let proj = reference.vision_proj_dim.unwrap_or(1);
        assert_eq!(emb.len(), reference.n_vision_tokens * proj);
    }
    eprintln!(
        "qwen25_vl reference reload ok: seq={} vision={} mu={:?}",
        reference.seq_len,
        reference.n_vision_tokens,
        reference.vision_mu_scores.as_ref().map(|m| m.len()),
    );
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn build_aif_probe(reference: &ReferenceDump) -> Option<AifProbe> {
    if let Some(dynamics) = reference.vision_dynamics.clone() {
        return Some(AifProbe::build(dynamics));
    }
    reference.vision_mu_scores.as_ref().map(|mu| {
        let flat_layers = reference
            .vision_token_entropy
            .as_ref()
            .map(|_| 1usize)
            .unwrap_or(1);
        let dynamics: Vec<Vec<f32>> = if flat_layers == 1 {
            mu.iter().map(|&m| vec![m]).collect()
        } else {
            mu.iter().map(|&m| vec![m; flat_layers]).collect()
        };
        AifProbe::build(dynamics)
    })
}

#[test]
fn qwen25_vl_aif_mu_decode() {
    if !ref_runner_ready() {
        eprintln!("skip aif mu decode: set RLX_QWEN25_VL_DOCKER=1 or RLX_QWEN25_VL_PYTHON");
        return;
    }
    if !hf_model_ready() {
        eprintln!("skip aif mu decode: set RLX_QWEN25_VL_HF_DIR or RLX_QWEN25_VL_DOWNLOAD=1");
        return;
    }
    let Some(gguf) = env_path("RLX_QWEN25_VL_GGUF_PATH").filter(|p| p.exists()) else {
        eprintln!("skip aif mu decode: set RLX_QWEN25_VL_GGUF_PATH");
        return;
    };
    let Some(mmproj) = env_path("RLX_QWEN25_VL_MMPROJ_PATH").filter(|p| p.exists()) else {
        eprintln!("skip aif mu decode: set RLX_QWEN25_VL_MMPROJ_PATH");
        return;
    };
    let Some(image) = load_image_path() else {
        eprintln!("skip aif mu decode: set RLX_QWEN25_VL_IMAGE");
        return;
    };

    let reference = dump_reference(&image);
    let Some(probe) = build_aif_probe(&reference) else {
        eprintln!(
            "skip aif mu decode: reference missing vision_dynamics.npy (rebuild docker image)"
        );
        return;
    };
    assert_eq!(probe.mu.len(), reference.n_vision_tokens);

    let (rgb, w, h) = load_rgb_image(image.to_str().expect("utf8")).expect("load image");

    let mut runner = Qwen25VlRunnerBuilder::default()
        .weights(&gguf)
        .mmproj(&mmproj)
        .device(Device::Cpu)
        .max_seq(reference.seq_len.max(128))
        .build()
        .expect("runner");

    let target = match (reference.resized_w, reference.resized_h) {
        (Some(tw), Some(th)) => (Some(tw), Some(th)),
        _ => (None, None),
    };
    let vision = runner
        .encode_image_resized(&rgb, w, h, target.0, target.1)
        .expect("vision encode");

    if vision.n_tokens != reference.n_vision_tokens {
        eprintln!(
            "skip aif mu decode: vision token count rlx={} hf={}",
            vision.n_tokens, reference.n_vision_tokens
        );
        return;
    }

    runner
        .prefill_from_token_ids(
            &reference.input_ids,
            reference.vision_start_idx,
            reference.n_vision_tokens,
            &vision,
            0,
        )
        .expect("rlx prefill");
    let next = argmax_token(&reference.logits);

    runner.clear_aif_decode();
    let baseline = runner.decode_step(next).expect("baseline decode");

    runner
        .prefill_from_token_ids(
            &reference.input_ids,
            reference.vision_start_idx,
            reference.n_vision_tokens,
            &vision,
            0,
        )
        .expect("rlx prefill");
    let span = runner.vision_key_span().expect("vision span");
    let aif = AifConfig::from(&probe);
    let rlx_ratio = aif.mask_ratio();
    runner.apply_aif_config(&aif).expect("apply aif");
    let blocked = aif.blocked_keys(span);
    let masked = runner.decode_step(next).expect("masked decode");

    if let Some(hf_ratio) = reference.aif_mask_ratio {
        assert!(
            (rlx_ratio - hf_ratio).abs() < 1e-4,
            "adaptive ratio rlx={rlx_ratio} hf={hf_ratio}"
        );
    }
    if let Some(hf_s0) = reference.aif_s0 {
        assert!(
            (probe.s0 - hf_s0).abs() < 1e-4,
            "S0 rlx={} hf={hf_s0}",
            probe.s0
        );
    }

    if let Some(ref hf_blocked) = reference.aif_blocked_keys {
        let mut a = blocked.clone();
        let mut b = hf_blocked.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "blocked key set vs HF reference");
    }

    assert!(
        !blocked.is_empty(),
        "AIF should block at least one visual key"
    );
    assert_ne!(
        baseline, masked,
        "μ-guided AIF mask should change decode logits"
    );

    eprintln!(
        "qwen25_vl_aif_mu_decode ok: ratio={rlx_ratio:.3} blocked={} next={next}",
        blocked.len(),
    );
}

fn dynamics_atol() -> f32 {
    // Default tolerance for Q4_K_M LM + native graph Q/K vs HF F32 attentions.
    std::env::var("RLX_AIF_DYNAMICS_ATOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.06)
}

fn aif_parity_ready() -> bool {
    ref_runner_ready()
        && hf_model_ready()
        && env_path("RLX_QWEN25_VL_GGUF_PATH")
            .filter(|p| p.exists())
            .is_some()
        && env_path("RLX_QWEN25_VL_MMPROJ_PATH")
            .filter(|p| p.exists())
            .is_some()
        && load_image_path().is_some()
}

fn assert_native_probe_matches_reference(
    runner: &Qwen25VlRunner,
    native: &AifProbe,
    reference: &ReferenceDump,
    label: &str,
) {
    let hf_dynamics = reference
        .vision_dynamics
        .as_ref()
        .expect("reference vision_dynamics");
    assert_eq!(native.dynamics.len(), hf_dynamics.len());
    assert_eq!(native.dynamics[0].len(), hf_dynamics[0].len());

    let atol = dynamics_atol();
    for (vi, (rlx_row, hf_row)) in native.dynamics.iter().zip(hf_dynamics.iter()).enumerate() {
        for (l, (&a, &b)) in rlx_row.iter().zip(hf_row.iter()).enumerate() {
            assert!(
                (a - b).abs() <= atol,
                "{label} dynamics[{vi}][{l}] rlx={a} hf={b} atol={atol}"
            );
        }
    }

    if let Some(ref hf_mu) = reference.vision_mu_scores {
        assert_eq!(native.mu.len(), hf_mu.len());
        for (i, (&a, &b)) in native.mu.iter().zip(hf_mu.iter()).enumerate() {
            assert!(
                (a - b).abs() <= atol,
                "{label} mu[{i}] rlx={a} hf={b} atol={atol}"
            );
        }
    }

    if let Some(hf_ratio) = reference.aif_mask_ratio {
        assert!(
            (native.mask_ratio - hf_ratio).abs() <= atol,
            "{label} mask_ratio rlx={} hf={hf_ratio}",
            native.mask_ratio
        );
    }
    if let Some(hf_s0) = reference.aif_s0 {
        assert!(
            (native.s0 - hf_s0).abs() <= atol,
            "{label} S0 rlx={} hf={hf_s0}",
            native.s0
        );
    }

    let span = runner.vision_key_span().expect("vision span");
    let blocked = AifConfig::from(native).blocked_keys(span);
    if let Some(ref hf_blocked) = reference.aif_blocked_keys {
        let mut a = blocked.clone();
        let mut b = hf_blocked.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "{label} blocked keys vs HF reference");
    }

    eprintln!(
        "{label} ok: n_vis={} layers={} ratio={:.3} atol={atol}",
        native.dynamics.len(),
        native.dynamics[0].len(),
        native.mask_ratio,
    );
}

fn run_native_aif_probe(
    gguf: &Path,
    mmproj: &Path,
    image: &Path,
    reference: &ReferenceDump,
    mode: AifDynamicsMode,
    use_probe_prefill: bool,
) -> (Qwen25VlRunner, AifProbe) {
    let (rgb, w, h) = load_rgb_image(image.to_str().expect("utf8")).expect("load image");

    let mut runner = Qwen25VlRunnerBuilder::default()
        .weights(gguf)
        .mmproj(mmproj)
        .device(Device::Cpu)
        .max_seq(reference.seq_len.max(128))
        .aif_dynamics_mode(mode)
        .build()
        .expect("runner");

    let target = match (reference.resized_w, reference.resized_h) {
        (Some(tw), Some(th)) => (Some(tw), Some(th)),
        _ => (None, None),
    };
    let vision = runner
        .encode_image_resized(&rgb, w, h, target.0, target.1)
        .expect("vision encode");

    assert_eq!(
        vision.n_tokens, reference.n_vision_tokens,
        "vision token count rlx={} hf={}",
        vision.n_tokens, reference.n_vision_tokens
    );

    if use_probe_prefill {
        runner
            .prefill_from_token_ids_probe(
                &reference.input_ids,
                reference.vision_start_idx,
                reference.n_vision_tokens,
                &vision,
                0,
            )
            .expect("rlx probe prefill");
    } else {
        runner
            .prefill_from_token_ids(
                &reference.input_ids,
                reference.vision_start_idx,
                reference.n_vision_tokens,
                &vision,
                0,
            )
            .expect("rlx prefill");
    }

    let native = runner.probe_aif_native().expect("native probe");
    (runner, native)
}

#[test]
fn qwen25_vl_aif_native_dynamics() {
    if !aif_parity_ready() {
        eprintln!(
            "skip native dynamics: set RLX_QWEN25_VL_DOCKER=1 or RLX_QWEN25_VL_PYTHON, HF dir, GGUF, mmproj, image"
        );
        return;
    }
    let gguf = env_path("RLX_QWEN25_VL_GGUF_PATH").unwrap();
    let mmproj = env_path("RLX_QWEN25_VL_MMPROJ_PATH").unwrap();
    let image = load_image_path().unwrap();

    let reference = dump_reference_with_env(&image, &[("RLX_AIF_DYNAMICS", "prefill_v2t")]);
    let Some(_) = reference.vision_dynamics.clone() else {
        eprintln!(
            "skip native dynamics: reference missing vision_dynamics.npy (rebuild docker image)"
        );
        return;
    };

    let (runner, native) = run_native_aif_probe(
        &gguf,
        &mmproj,
        &image,
        &reference,
        AifDynamicsMode::PrefillV2t,
        true,
    );
    assert_native_probe_matches_reference(
        &runner,
        &native,
        &reference,
        "qwen25_vl_aif_native_dynamics",
    );
}

#[test]
fn qwen25_vl_aif_native_decode_step_dynamics() {
    if !aif_parity_ready() {
        eprintln!(
            "skip decode-step dynamics: set RLX_QWEN25_VL_DOCKER=1 or RLX_QWEN25_VL_PYTHON, HF dir, GGUF, mmproj, image"
        );
        return;
    }
    let gguf = env_path("RLX_QWEN25_VL_GGUF_PATH").unwrap();
    let mmproj = env_path("RLX_QWEN25_VL_MMPROJ_PATH").unwrap();
    let image = load_image_path().unwrap();

    let reference = dump_reference_with_env(&image, &[("RLX_AIF_DYNAMICS", "decode_step")]);
    let Some(_) = reference.vision_dynamics.clone() else {
        eprintln!(
            "skip decode-step dynamics: reference missing vision_dynamics.npy (rebuild docker image)"
        );
        return;
    };
    if reference.aif_dynamics_mode.as_deref() != Some("decode_step") {
        eprintln!(
            "skip decode-step dynamics: reference dump lacks aif_dynamics_mode=decode_step (rebuild ref image)"
        );
        return;
    }

    let (runner, native) = run_native_aif_probe(
        &gguf,
        &mmproj,
        &image,
        &reference,
        AifDynamicsMode::DecodeStep,
        false,
    );
    assert_native_probe_matches_reference(
        &runner,
        &native,
        &reference,
        "qwen25_vl_aif_native_decode_step_dynamics",
    );
}
