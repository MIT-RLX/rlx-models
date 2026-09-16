//! Does this port actually agree with the implementation it was ported from?
//!
//! Everything else in this crate is *self*-consistent: finite differences check
//! the ops, CPU/Metal parity checks the kernels, bit-exact invariances check the
//! scheduling. None of that would catch a systematically different estimator, and
//! one such difference was in fact here — see the tap convention below.
//!
//! Requires `weights/Qwen3-0.6B` (an f32 checkpoint both sides can read; the
//! Qwen3.5 one on hand is quantized and would only ever give the Jacobian *of the
//! quantized model*) and a Python environment with `torch`, `transformers` and
//! the reference `jlens` on the path. Skips itself when either is missing.
//!
//! ```bash
//! python3 crates/rlx-jlens/scripts/reference_jacobian.py \
//!     --model weights/Qwen3-0.6B --out /tmp/ref.safetensors \
//!     --layers 3,4,5 --target 6 --seq 16 --dim-batch 8 --skip-first 1
//! cargo test -p rlx-jlens --features qwen3 --release --test reference_parity -- --nocapture
//! ```

#![cfg(feature = "qwen3")]

use std::collections::BTreeMap;

use rlx_jlens::models::qwen3::Qwen3LensModel;
use rlx_jlens::{FitConfig, LensModel, StackLens};
use rlx_runtime::Device;

const REF: &str = "/tmp/ref.safetensors";
const TARGET: usize = 6;
const LAYERS: [usize; 3] = [3, 4, 5];
const DIM_BATCH: usize = 8;
const SKIP_FIRST: usize = 1;
/// Exactly what `reference_jacobian.py` printed, so no tokenizer sits between
/// the two implementations — a tokenization difference would surface as a
/// numerical one and be blamed on the estimator.
const IDS: [u32; 14] = [
    785, 6722, 315, 9625, 374, 12095, 323, 279, 6722, 315, 6323, 374, 26194, 13,
];

fn reference() -> Option<BTreeMap<usize, Vec<f32>>> {
    let bytes = std::fs::read(REF).ok()?;
    let st = safetensors::SafeTensors::deserialize(&bytes).ok()?;
    let mut out = BTreeMap::new();
    for (name, view) in st.tensors() {
        let layer = name.strip_prefix("J.")?.parse::<usize>().ok()?;
        assert_eq!(
            view.dtype(),
            safetensors::tensor::Dtype::F32,
            "{name} must be f32"
        );
        let v: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(layer, v);
    }
    Some(out)
}

fn model() -> Option<Qwen3LensModel> {
    let dir = match std::env::var("RLX_JLENS_QWEN3_DIR") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/Qwen3-0.6B"),
    };
    if !dir.join("config.json").exists() {
        return None;
    }
    Qwen3LensModel::open(&dir).ok()
}

/// rlx's `J_l` must equal the reference's `J_l`, entry for entry.
///
/// It did not, originally: the reference hooks each block's forward **output**,
/// while this crate tapped each block's **input**, so rlx's `J_l` reproduced the
/// reference's `J_{l-1}` — to 2e-4, i.e. correct arithmetic under a
/// one-layer-shifted labelling. Since a lens file is meant to be interchangeable
/// with the reference's, that mislabels every layer in the artifact.
/// `layer_exit_taps` is the fix; this test is what would catch its regression.
#[test]
fn matches_the_python_reference() {
    let Some(reference) = reference() else {
        eprintln!("no {REF}; run scripts/reference_jacobian.py first — skipping");
        return;
    };
    let Some(model) = model() else {
        eprintln!("no Qwen3 checkpoint; skipping");
        return;
    };

    let ids: Vec<f32> = IDS.iter().map(|&t| t as f32).collect();
    let layers = LAYERS.to_vec();
    let mut lens = StackLens::new(
        &model,
        &layers,
        TARGET,
        ids.len(),
        FitConfig {
            dim_batch: DIM_BATCH,
            skip_first: SKIP_FIRST,
            device: Device::Cpu,
        },
    )
    .expect("stack lens");
    let batched = lens.replicate_tokens(&ids).expect("replicate");
    let js = lens.jacobians(&batched).expect("jacobians");

    let d = model.d_model();
    let mut worst_rel = 0.0f32;
    for (slot, &layer) in layers.iter().enumerate() {
        let r = reference
            .get(&layer)
            .unwrap_or_else(|| panic!("reference has no layer {layer}"));
        let x = &js[slot].values;
        assert_eq!(r.len(), d * d, "reference J.{layer} is the wrong size");
        assert_eq!(x.len(), d * d, "rlx J.{layer} is the wrong size");

        let num: f32 = r.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = r.iter().map(|a| a * a).sum();
        let rel = (num / den).sqrt();
        let worst = r
            .iter()
            .zip(x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let diag = |v: &[f32]| (0..d).map(|i| v[i * d + i]).sum::<f32>() / d as f32;
        eprintln!(
            "J.{layer}: relF = {rel:.2e}  max|diff| = {worst:.2e}  \
             mean diag ref {:.4} vs rlx {:.4}",
            diag(r),
            diag(x)
        );
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < 5e-3,
            "layer {layer} disagrees with the reference: relF {rel:.3e}"
        );
    }
    eprintln!("worst relF across layers: {worst_rel:.2e}");
}

/// Guard the convention itself, not just the numbers.
///
/// If the taps ever slip back to block inputs, the test above would still catch
/// it — but only where a reference dump is present. This states the invariant
/// directly: transporting layer `target`'s own exit residual to layer `target`
/// is the identity, which is true of exit taps and false of entry taps.
#[test]
fn tapping_the_target_layer_is_the_identity() {
    let Some(model) = model() else {
        eprintln!("no Qwen3 checkpoint; skipping");
        return;
    };
    let target = 3usize;
    let ids: Vec<f32> = IDS.iter().map(|&t| t as f32).collect();
    let mut lens = StackLens::new(
        &model,
        &[target],
        target,
        ids.len(),
        FitConfig {
            dim_batch: 8,
            skip_first: 1,
            device: Device::Cpu,
        },
    )
    .expect("stack lens");
    let batched = lens.replicate_tokens(&ids).expect("replicate");
    let js = lens.jacobians(&batched).expect("jacobians");

    let d = model.d_model();
    let j = &js[0].values;
    let mut worst_diag = 0.0f32;
    let mut worst_off = 0.0f32;
    for i in 0..d {
        for k in 0..d {
            let v = j[i * d + k];
            if i == k {
                worst_diag = worst_diag.max((v - 1.0).abs());
            } else {
                worst_off = worst_off.max(v.abs());
            }
        }
    }
    eprintln!(
        "J(target->target): worst |diag-1| = {worst_diag:.2e}, worst off-diagonal = {worst_off:.2e}"
    );
    assert!(
        worst_diag < 1e-4 && worst_off < 1e-4,
        "tapping the target layer should give the identity, but |diag-1| reaches \
         {worst_diag:.2e} and the off-diagonal reaches {worst_off:.2e} — the taps \
         are probably back on block inputs"
    );
}
