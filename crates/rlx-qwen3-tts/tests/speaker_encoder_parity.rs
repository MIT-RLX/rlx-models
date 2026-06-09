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

//! Layer-by-layer parity check vs the Python `Qwen3TTSSpeakerEncoder`.
//!
//! Requires the safetensors fixture baked by:
//!   `scripts/qwen3_tts_bake_speaker_parity.py`
//!
//! Skips silently if either the Base checkpoint or the fixture is missing.

use ndarray::{Array2, Array3};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::speaker_encoder::{
    config::SpeakerEncoderConfig, mel::log_mel, open_speaker_encoder,
};
use safetensors::SafeTensors;
use std::path::PathBuf;

fn base_dir() -> Option<PathBuf> {
    let env = std::env::var("RLX_QWEN3_TTS_BASE_DIR")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.is_dir());
    if let Some(p) = env {
        return Some(p);
    }
    let p = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    p.is_dir().then_some(p)
}

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/speaker_parity.safetensors");
    p
}

struct Fixture {
    bytes: Vec<u8>,
}

impl Fixture {
    fn load() -> Option<Self> {
        let path = fixture_path();
        path.is_file().then(|| Self {
            bytes: std::fs::read(&path).expect("read fixture"),
        })
    }

    fn view(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.bytes).expect("parse fixture safetensors")
    }
}

fn tensor_f32(st: &SafeTensors<'_>, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing {name}: {e}"));
    let shape = t.shape().to_vec();
    let bytes = t.data();
    assert!(bytes.len() % 4 == 0);
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    (data, shape)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn relative_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
    max_abs_diff(a, b) / scale
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
fn speaker_encoder_layer_parity() {
    let Some(model_dir) = base_dir() else {
        eprintln!("skip: no Base checkpoint (set RLX_QWEN3_TTS_BASE_DIR)");
        return;
    };
    let Some(fixture) = Fixture::load() else {
        eprintln!(
            "skip: fixture {} missing — run scripts/qwen3_tts_bake_speaker_parity.py",
            fixture_path().display()
        );
        return;
    };
    let st = fixture.view();
    let cfg = SpeakerEncoderConfig::from_model_dir(&model_dir).expect("config");
    let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
    let enc = open_speaker_encoder(&store, &cfg).expect("open speaker encoder");

    // ---- mel front-end ----
    let (pcm, pcm_shape) = tensor_f32(&st, "pcm");
    assert_eq!(pcm_shape.len(), 1, "pcm should be 1-D");
    let (mel_ref, mel_shape) = tensor_f32(&st, "mel");
    assert_eq!(mel_shape, vec![1, mel_shape[1], cfg.mel_dim]);
    let t_frames = mel_shape[1];
    let mel = log_mel(&pcm, &cfg.mel_params()).expect("log_mel");
    assert_eq!(mel.dim(), (cfg.mel_dim, t_frames), "mel shape");
    // mel_ref is [1, T, mel_dim]; ours is [mel_dim, T]. Compare elementwise.
    let mut mel_ours_aligned = Array2::<f32>::zeros((t_frames, cfg.mel_dim));
    for ti in 0..t_frames {
        for m in 0..cfg.mel_dim {
            mel_ours_aligned[[ti, m]] = mel[[m, ti]];
        }
    }
    let mel_max_abs = max_abs_diff(mel_ours_aligned.as_slice().unwrap(), &mel_ref);
    let mel_rel = relative_diff(mel_ours_aligned.as_slice().unwrap(), &mel_ref);
    println!("[parity] mel: max_abs={mel_max_abs:.3e} rel={mel_rel:.3e}");
    assert!(mel_rel < 5e-4, "mel relative diff too large: {mel_rel:.3e}");

    // ---- per-block intermediates ----
    let mut hidden = enc.initial.forward(mel.view());
    for i in 0..=enc.blocks.len() {
        let name = format!("block_{i}_out");
        let (ref_data, ref_shape) = tensor_f32(&st, &name);
        // Ref shape is [1, C, T] — drop batch dim.
        assert_eq!(ref_shape.len(), 3);
        let c = ref_shape[1];
        let t = ref_shape[2];
        let (cc, tt) = hidden.dim();
        assert_eq!(
            (cc, tt),
            (c, t),
            "{name} shape ours {cc}x{tt} vs ref {c}x{t}"
        );
        let ours = hidden.as_slice().unwrap();
        let rel = relative_diff(ours, &ref_data);
        println!("[parity] {name}: rel={rel:.3e}");
        assert!(rel < 5e-3, "{name} diverged: rel={rel:.3e}");
        if i < enc.blocks.len() {
            hidden = enc.blocks[i].forward(hidden.view());
        }
    }

    // ---- MFA + ASP + FC ----
    let (mfa_ref, mfa_shape) = tensor_f32(&st, "mfa");
    let cat_c: usize = enc.cfg.enc_channels[1..enc.cfg.enc_channels.len() - 1]
        .iter()
        .sum();
    let mut cat = Array2::<f32>::zeros((cat_c, hidden.dim().1));
    // Re-run blocks to capture all outputs.
    let mut h2 = enc.initial.forward(mel.view());
    let mut outs = Vec::new();
    for b in &enc.blocks {
        h2 = b.forward(h2.view());
        outs.push(h2.clone());
    }
    let mut off = 0;
    for h in &outs {
        let c = h.dim().0;
        for ci in 0..c {
            for ti in 0..h.dim().1 {
                cat[[off + ci, ti]] = h[[ci, ti]];
            }
        }
        off += c;
    }
    let mfa = enc.mfa.forward(cat.view());
    assert_eq!(mfa_shape[1..], [mfa.dim().0, mfa.dim().1]);
    let rel_mfa = relative_diff(mfa.as_slice().unwrap(), &mfa_ref);
    println!("[parity] mfa: rel={rel_mfa:.3e}");
    assert!(rel_mfa < 5e-3);

    let (asp_ref, asp_shape) = tensor_f32(&st, "asp");
    let asp = enc.asp.forward(mfa.view());
    assert_eq!(asp_shape[1..], [asp.dim().0, asp.dim().1]);
    let rel_asp = relative_diff(asp.as_slice().unwrap(), &asp_ref);
    println!("[parity] asp: rel={rel_asp:.3e}");
    assert!(rel_asp < 1e-2);

    let (fc_ref, fc_shape) = tensor_f32(&st, "fc");
    let fc = enc.fc.forward(asp.view());
    assert_eq!(fc_shape[1..], [fc.dim().0, fc.dim().1]);
    let rel_fc = relative_diff(fc.as_slice().unwrap(), &fc_ref);
    println!("[parity] fc: rel={rel_fc:.3e}");
    assert!(rel_fc < 1e-2);

    // ---- full forward + cosine vs reference x-vector ----
    let (xvec_ref, xvec_shape) = tensor_f32(&st, "xvec");
    assert_eq!(xvec_shape, vec![1, cfg.enc_dim]);
    let xvec = enc.forward(mel.view());
    let cos = cosine(&xvec, &xvec_ref);
    let rel = relative_diff(&xvec, &xvec_ref);
    let max_abs = max_abs_diff(&xvec, &xvec_ref);
    println!(
        "[parity] xvec: dim={} cos={cos:.6} rel={rel:.3e} max_abs={max_abs:.3e}",
        xvec.len()
    );
    assert!(cos > 0.9999, "x-vector cosine too low: {cos:.6}");
}

#[allow(dead_code)]
fn _zero_use_for_array3() {
    let _ = Array3::<f32>::zeros((1, 1, 1));
}
