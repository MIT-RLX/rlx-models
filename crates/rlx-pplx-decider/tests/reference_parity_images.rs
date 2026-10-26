//! Image requests against Perplexity's own `pplx_decider` package.
//!
//! Fixtures from `scripts/pplx_decider_reference.py` over
//! `tests/fixtures/requests_images.json` (PNG images under `fixtures/img`,
//! so both sides decode identical pixels):
//!
//! * `ref_images_f32.json` — bf16 safetensors upcast to f32;
//! * `ref_images_q8_0.json` — the backbone from the dequantized Q8_0 GGUF and
//!   the tower from the F16 mmproj, i.e. the weights this crate runs.
//!
//! Each stage that can be silently wrong is checked on its own, in pipeline
//! order: resized pixels (exact), expanded token ids and markers (exact),
//! M-RoPE positions (exact), the vision tower's output, then the readout.
//!
//! JPEG requests are compared only with the `libjpeg-turbo` feature: the
//! pure-Rust decoder is close to PIL's libjpeg-turbo but not identical
//! (~1.4% of a photo's bytes, up to 3 levels), which moves logits by ~0.03.
//!
//! The preprocessing checks need no weights. The rest load
//! `PPLX_DECIDER_GGUF` (default `weights/lm/pplx-decider-flash-gguf`, mmproj
//! beside it) on `PPLX_DECIDER_DEVICE` (default `cpu`).

use std::path::PathBuf;

use rlx_pplx_decider::protocol::Request;
use rlx_pplx_decider::{Json, OpenOptions, PplxDecider, vision};
use sha2::{Digest, Sha256};

const REQUESTS: &str = include_str!("fixtures/requests_images.json");
const REF_F32: &str = include_str!("fixtures/ref_images_f32.json");
const REF_Q8_0: &str = include_str!("fixtures/ref_images_q8_0.json");

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn requests() -> Vec<Json> {
    Json::parse(REQUESTS).unwrap().as_array().unwrap().to_vec()
}

fn results(reference: &str) -> Vec<Json> {
    Json::parse(reference)
        .unwrap()
        .get("results")
        .unwrap()
        .as_array()
        .unwrap()
        .to_vec()
}

fn num(x: &Json) -> f64 {
    match x {
        Json::Float(f) => *f,
        Json::Int(i) => *i as f64,
        Json::UInt(u) => *u as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn nums(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn ints(v: &Json) -> Vec<usize> {
    nums(v).into_iter().map(|x| x as usize).collect()
}

/// Whether this build can reproduce `entry`'s pixels exactly.
fn exact_decode(entry: &str) -> bool {
    vision::JPEG_EXACT || !entry.to_ascii_lowercase().ends_with(".jpg")
}

#[test]
fn preprocessing_is_bit_exact() {
    for (i, (r, want)) in requests().iter().zip(results(REF_F32)).enumerate() {
        let req = Request::from_json(r, 255).unwrap();
        let grids = want.get("image_grid_thw").unwrap().as_array().unwrap();
        let shas = want.get("image_sha256").unwrap().as_array().unwrap();
        for (k, entry) in req.images.iter().enumerate() {
            if !exact_decode(entry) {
                eprintln!("request {i} image {k}: JPEG, skipped without the libjpeg-turbo feature");
                continue;
            }
            let img = vision::RgbImage::open(entry, Some(&fixtures())).unwrap();
            let pre = vision::preprocess(&img, 65_536, 1_638_400).unwrap();
            let g = ints(&grids[k]);
            assert_eq!(
                (pre.grid_h * 2, pre.grid_w * 2),
                (g[1], g[2]),
                "request {i} image {k}: patch grid"
            );
            let sha = format!("{:x}", Sha256::digest(&pre.resized.pixels));
            assert_eq!(
                sha,
                shas[k].as_str().unwrap(),
                "request {i} image {k}: resized pixels differ from torchvision's"
            );
        }
    }
}

fn open() -> Option<PplxDecider> {
    let path = rlx_ir::env::read("PPLX_DECIDER_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/lm/pplx-decider-flash-gguf")
        });
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (set PPLX_DECIDER_GGUF)",
            path.display()
        );
        return None;
    }
    let device = rlx_ir::env::read("PPLX_DECIDER_DEVICE").unwrap_or_else(|_| "cpu".into());
    let opts = OpenOptions {
        image_base: Some(fixtures()),
        ..OpenOptions::default()
    };
    Some(PplxDecider::open_with(&path, rlx_cli::parse_device(&device).unwrap(), &opts).unwrap())
}

#[test]
fn real_weights_match_the_reference() {
    let Some(mut d) = open() else { return };
    let q8 = results(REF_Q8_0);
    let f32_ref = results(REF_F32);
    let t = d.meta().temperature;
    let h = d.config().hidden_size;
    let probs = |l: &[f64]| {
        rlx_pplx_decider::protocol::probabilities(
            &l.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            t,
        )
    };

    let (mut worst_vision, mut worst_logit, mut worst_q8_p, mut worst_f32_p) =
        (0f64, 0f64, 0f64, 0f64);
    for (i, r) in requests().iter().enumerate() {
        let req = d.request(r).unwrap();
        if !req.images.iter().all(|e| exact_decode(e)) {
            eprintln!("request {i}: JPEG, skipped without the libjpeg-turbo feature");
            continue;
        }
        let plan = d.plan(&req).unwrap();
        assert_eq!(plan.chunks.len(), 1);
        let enc = &plan.chunks[0].1;

        let ids: Vec<usize> = enc.input_ids.iter().map(|&x| x as usize).collect();
        assert_eq!(
            ids,
            ints(q8[i].get("input_ids").unwrap()),
            "request {i}: token ids"
        );
        assert_eq!(
            enc.markers,
            ints(q8[i].get("markers").unwrap()),
            "request {i}: markers"
        );

        let pos = enc.positions().unwrap();
        let want = q8[i].get("position_ids").unwrap().as_array().unwrap();
        for axis in 0..3 {
            let got: Vec<usize> = pos.iter().map(|p| p[axis]).collect();
            assert_eq!(got, ints(&want[axis]), "request {i}: M-RoPE axis {axis}");
        }

        // The tower: first values of its first, middle and last rows (the
        // reference concatenates every image's rows).
        let vision = d.vision_embeddings(&plan.images).unwrap();
        let rows: Vec<&[f32]> = vision.iter().flat_map(|v| v.chunks_exact(h)).collect();
        assert_eq!(
            rows.len(),
            num(q8[i].get("vision_rows").unwrap()) as usize,
            "request {i}: vision rows"
        );
        let head = q8[i].get("vision_head").unwrap().as_array().unwrap();
        for (k, &row) in [0, rows.len() / 2, rows.len() - 1].iter().enumerate() {
            for (a, b) in rows[row].iter().zip(nums(&head[k])) {
                worst_vision = worst_vision.max((f64::from(*a) - b).abs());
            }
        }

        let logits = d.logits(&req).unwrap();
        for ql in &logits {
            let a = nums(q8[i].get("logits").unwrap().get(&ql.id).unwrap());
            let b = nums(f32_ref[i].get("logits").unwrap().get(&ql.id).unwrap());
            let ours: Vec<f64> = ql.logits.iter().map(|&x| f64::from(x)).collect();
            let (p, pa, pb) = (probs(&ours), probs(&a), probs(&b));
            for k in 0..a.len() {
                worst_logit = worst_logit.max((ours[k] - a[k]).abs());
                worst_q8_p = worst_q8_p.max((p[k] - pa[k]).abs());
                worst_f32_p = worst_f32_p.max((p[k] - pb[k]).abs());
            }
        }
    }
    eprintln!(
        "vision max abs(Δ) {worst_vision:.3e}; logits vs Q8_0+mmproj reference {worst_logit:.3e}, \
         abs(Δp) {worst_q8_p:.3e}; abs(Δp) vs bf16 {worst_f32_p:.3e}"
    );
    assert!(
        worst_vision < VISION_TOL,
        "vision tower off by {worst_vision}"
    );
    assert!(
        worst_logit < LOGIT_TOL,
        "logits off the reference by {worst_logit}"
    );
    assert!(
        worst_q8_p < 2e-3,
        "probabilities off the reference by {worst_q8_p}"
    );
    assert!(
        worst_f32_p < 0.03,
        "probabilities off bf16 by {worst_f32_p}"
    );
}

/// Measured on Metal: the tower lands 2.2e-4 from the reference (its
/// outputs peak near |8|, so ~3e-5 relative); the logits 2.2e-4. The logit
/// tolerance matches the text suite's: rotating image tokens with the
/// contiguous instead of the interleaved M-RoPE layout — invisible to
/// text-only prompts — moves a logit by 0.24.
const VISION_TOL: f64 = 5e-3;
const LOGIT_TOL: f64 = 5e-3;
