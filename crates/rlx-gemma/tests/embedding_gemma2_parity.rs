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

//! `embeddinggemma-2` text and audio embeddings, and the null-space audio
//! head, against HF + sentence-transformers, stage by stage.
//!
//! ```text
//! python scripts/embeddinggemma2_reference.py --model-dir weights/embed/embeddinggemma-2 \
//!     --head weights/embed/embeddinggemma-2-nullspace-audio/nullspace_audio_head.safetensors \
//!     --out fixtures/embeddinggemma2 --dtype float32 WAV...
//! RLX_EMBEDDINGGEMMA2_DIR=weights/embed/embeddinggemma-2 \
//! RLX_EMBEDDINGGEMMA2_HEAD=weights/embed/embeddinggemma-2-nullspace-audio/nullspace_audio_head.safetensors \
//! RLX_EMBEDDINGGEMMA2_REF=fixtures/embeddinggemma2/float32 \
//!     cargo test -p rlx-gemma --release --test embedding_gemma2_parity -- --nocapture
//! ```
//!
//! `RLX_EMBEDDINGGEMMA2_DEVICE` picks the device (default `cpu`). Skips
//! without the variables.

use std::path::{Path, PathBuf};

use rlx_gemma::embedding_gemma2_model::EmbeddingGemma2;
use rlx_gemma::embedding_gemma2_nullspace::{HEAD_TAPS, NullSpaceAudioHead};
use rlx_runtime::Device;

fn env_path(k: &str) -> Option<PathBuf> {
    rlx_ir::env::var(k).map(PathBuf::from)
}

fn f32s(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn ids(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let d: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    d / (na * nb).max(1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Worst per-row cosine of two `[rows · width]` matrices.
fn worst_row_cos(a: &[f32], b: &[f32], width: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.chunks_exact(width)
        .zip(b.chunks_exact(width))
        .map(|(x, y)| cos(x, y))
        .fold(1.0, f64::min)
}

fn device() -> Device {
    match rlx_ir::env::var("RLX_EMBEDDINGGEMMA2_DEVICE").as_deref() {
        None | Some("cpu") => Device::Cpu,
        Some(d) => rlx_cli::parse_device(d).unwrap_or_else(|e| panic!("{d}: {e}")),
    }
}

struct Setup {
    model: EmbeddingGemma2,
    head: NullSpaceAudioHead,
    reference: PathBuf,
}

fn setup() -> Option<Setup> {
    let (Some(dir), Some(head), Some(reference)) = (
        env_path("RLX_EMBEDDINGGEMMA2_DIR"),
        env_path("RLX_EMBEDDINGGEMMA2_HEAD"),
        env_path("RLX_EMBEDDINGGEMMA2_REF"),
    ) else {
        eprintln!("RLX_EMBEDDINGGEMMA2_DIR / _HEAD / _REF unset: skipping");
        return None;
    };
    Some(Setup {
        model: EmbeddingGemma2::open(&dir, device()).expect("open model"),
        head: NullSpaceAudioHead::open(&head).expect("open head"),
        reference,
    })
}

fn cases(root: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    v.sort();
    v
}

#[test]
fn audio_and_text_match_the_reference() {
    let Some(mut s) = setup() else { return };
    let h = s.model.config().text.hidden_size;

    // u_perp is orthogonal to every output of this checkpoint.
    let proj = s.model.embedding_projection().unwrap().to_vec();
    let overlap = s.head.max_overlap(&proj, h);
    eprintln!("u_perp · embedding_projection: worst normalized overlap {overlap:.2e}");
    assert!(
        overlap < 1e-3,
        "u_perp is not orthogonal to the projection: {overlap}"
    );

    let audio = cases(&s.reference, "audio_");
    assert!(
        !audio.is_empty(),
        "no audio cases in {}",
        s.reference.display()
    );
    for d in &audio {
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        let samples = f32s(&d.join("samples.f32"));
        let want_ids = ids(&d.join("input_ids.i64"));

        // Features: our extractor against HF's.
        let hf_feats = f32s(&d.join("features.f32"));
        let (feats, frames) = s.model.audio_features(&samples);
        assert_eq!(feats.len(), hf_feats.len(), "{name}: frame count");
        let feat_err = max_abs(&feats, &hf_feats);

        // Tower on HF's features, so its error is the tower's own.
        let (outs, seq) = s.model.run_tower(&hf_feats, frames, &[2, 6, 12]).unwrap();
        let tower_cos = worst_row_cos(&outs[0], &f32s(&d.join("tower_out.f32")), 1536);
        let l2_cos = worst_row_cos(&outs[1], &f32s(&d.join("tower_l2.f32")), 1024);
        let l6_cos = worst_row_cos(&outs[2], &f32s(&d.join("tower_l6.f32")), 1024);
        let l12_cos = worst_row_cos(&outs[3], &f32s(&d.join("tower_l12.f32")), 1024);
        let soft = s.model.embed_audio_tokens(&outs[0], seq).unwrap();
        let soft_cos = worst_row_cos(&soft, &f32s(&d.join("soft_tokens.f32")), h);

        // Token layout: the prompt part of the reference ids, then ours.
        let boa = s.model.config().boa_token_id;
        let prompt_len = want_ids
            .iter()
            .position(|&t| t == boa)
            .expect("no <|audio>");
        let prompt = &want_ids[..prompt_len];
        assert_eq!(
            s.model.audio_prompt_ids(prompt, seq),
            want_ids,
            "{name}: token ids"
        );

        // End to end from samples, with the head's taps.
        let a = s.model.embed_audio(prompt, &samples, &HEAD_TAPS).unwrap();
        let base_cos = cos(&a.embedding, &f32s(&d.join("base.f32")));
        let l2 = NullSpaceAudioHead::pool_mean_std(&a.taps[0], a.audio_tokens);
        let l6 = NullSpaceAudioHead::pool_mean(&a.taps[1], a.audio_tokens);
        let pool_cos = cos(&l2, &f32s(&d.join("pool_l2_ms.f32")))
            .min(cos(&l6, &f32s(&d.join("pool_l6_mean.f32"))));
        let z = s.head.z(&l2, &l6).unwrap();
        let z_cos = cos(&z, &f32s(&d.join("z.f32")));
        let emb = s.head.apply(&a).unwrap();
        let emb_cos = cos(&emb, &f32s(&d.join("embedding.f32")));
        // The head on the reference's own inputs isolates its arithmetic.
        let head_only = s
            .head
            .inject(
                &f32s(&d.join("base.f32")),
                &s.head
                    .z(
                        &f32s(&d.join("pool_l2_ms.f32")),
                        &f32s(&d.join("pool_l6_mean.f32")),
                    )
                    .unwrap(),
            )
            .unwrap();
        let head_err = max_abs(&head_only, &f32s(&d.join("embedding.f32")));

        eprintln!(
            "{name}: frames {frames} tokens {} | feats max|Δ| {feat_err:.1e} | tower cos {tower_cos:.6} \
             l2 {l2_cos:.6} l6 {l6_cos:.6} l12 {l12_cos:.6} | soft {soft_cos:.6} | base {base_cos:.7} \
             | pools {pool_cos:.7} z {z_cos:.7} | head+base {emb_cos:.7} | head alone max|Δ| {head_err:.1e}",
            want_ids.len()
        );
        assert!(feat_err < 2e-3, "{name}: features {feat_err}");
        for (what, c) in [
            ("tower", tower_cos),
            ("l2", l2_cos),
            ("l6", l6_cos),
            ("l12", l12_cos),
            ("soft", soft_cos),
        ] {
            assert!(c > 0.9999, "{name}: {what} worst row cosine {c}");
        }
        assert!(
            base_cos > 0.99999,
            "{name}: base embedding cosine {base_cos}"
        );
        assert!(z_cos > 0.9999, "{name}: z cosine {z_cos}");
        assert!(emb_cos > 0.99999, "{name}: head embedding cosine {emb_cos}");
        assert!(head_err < 1e-5, "{name}: head arithmetic {head_err}");
    }

    for d in cases(&s.reference, "text_") {
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        let got = s.model.embed_ids(&ids(&d.join("input_ids.i64"))).unwrap();
        let c = cos(&got, &f32s(&d.join("base.f32")));
        eprintln!("{name}: base cos {c:.7}");
        assert!(c > 0.99999, "{name}: text embedding cosine {c}");
    }
}

/// What the head promises: text embeddings untouched, the injected part
/// orthogonal to every base embedding, so text–audio dot products move only
/// by the final renormalization.
#[test]
fn head_is_invisible_to_text() {
    let Some(mut s) = setup() else { return };
    let text_dir = cases(&s.reference, "text_")
        .into_iter()
        .next()
        .expect("a text case");
    let text = s
        .model
        .embed_ids(&ids(&text_dir.join("input_ids.i64")))
        .unwrap();
    for d in cases(&s.reference, "audio_") {
        let samples = f32s(&d.join("samples.f32"));
        let want_ids = ids(&d.join("input_ids.i64"));
        let boa = s.model.config().boa_token_id;
        let prompt = &want_ids[..want_ids.iter().position(|&t| t == boa).unwrap()];
        let a = s.model.embed_audio(prompt, &samples, &HEAD_TAPS).unwrap();
        let with = s.head.apply(&a).unwrap();
        // with = (v + u z) / ‖v + u z‖ with ⟨t, u z⟩ = 0, so
        // ⟨t, with⟩ · ‖v + u z‖ = ⟨t, v⟩; ‖v + u z‖ = √2 for unit v ⟂ u z.
        let dot =
            |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| f64::from(a * b)).sum::<f64>();
        let before = dot(&text, &a.embedding);
        let after = dot(&text, &with) * 2f64.sqrt();
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        eprintln!("{name}: text·audio {before:.6} without, {after:.6} with the head (× √2)");
        assert!(
            (before - after).abs() < 1e-4,
            "{name}: the head moved a text–audio dot product"
        );
    }
}
