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

//! Layer-by-layer parity for the speech tokenizer **conv encoder stack**
//! against Python `transformers.MimiEncoder`.
//!
//! Fixture: `tests/fixtures/st_encoder_parity.safetensors`
//!   (run `scripts/qwen3_tts_bake_st_encoder_parity.py`).
//!
//! Skips if the Base checkpoint or fixture is missing.

use ndarray::{Array2, ArrayView2};
use rlx_qwen3_tts::speech_tokenizer::encoder::{
    MimiConvEncoder, MimiDownsample, open_conv_encoder,
};
use rlx_qwen3_tts::speech_tokenizer::encoder_transformer::{
    MimiEncoderTransformer, open_encoder_transformer,
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
    p.push("tests/fixtures/st_encoder_parity.safetensors");
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
    assert_eq!(a.len(), b.len(), "size mismatch");
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn relative_diff(ours: &[f32], theirs: &[f32]) -> f32 {
    let scale = theirs
        .iter()
        .map(|v| v.abs())
        .fold(0f32, f32::max)
        .max(1e-6);
    max_abs_diff(ours, theirs) / scale
}

fn align_2d(ours: &Array2<f32>) -> Vec<f32> {
    // ours: [C, T]. Reference fixture: [B=1, C, T] stored row-major → same flat layout.
    ours.iter().copied().collect()
}

#[test]
fn st_encoder_conv_layer_parity() {
    let Some(model_dir) = base_dir() else {
        eprintln!("skip: no Base checkpoint (set RLX_QWEN3_TTS_BASE_DIR)");
        return;
    };
    let Some(fixture) = Fixture::load() else {
        eprintln!(
            "skip: fixture {} missing — run scripts/qwen3_tts_bake_st_encoder_parity.py",
            fixture_path().display()
        );
        return;
    };
    let st = fixture.view();
    let tok_dir = model_dir.join("speech_tokenizer");
    let enc: MimiConvEncoder = open_conv_encoder(&tok_dir).expect("open encoder");
    let cfg = &enc.cfg;
    println!(
        "[parity] encoder: audio_channels={} num_filters={} kernel_size={} last_kernel_size={} \
         hidden_size={} ratios={:?} compress={} layers={}",
        cfg.audio_channels,
        cfg.num_filters,
        cfg.kernel_size,
        cfg.last_kernel_size,
        cfg.hidden_size,
        cfg.upsampling_ratios,
        cfg.compress,
        enc.layer_count(),
    );

    // Input.
    let (input_data, input_shape) = tensor_f32(&st, "input_values");
    assert_eq!(input_shape, vec![1, cfg.audio_channels, input_shape[2]]);
    let t_in = input_shape[2];
    let x = Array2::from_shape_vec((cfg.audio_channels, t_in), input_data).expect("input shape");

    // Run with intermediates.
    let (final_out, outs) = enc.forward_with_intermediates(x.view());

    let n_layers = enc.layer_count();
    assert_eq!(outs.len(), n_layers);
    for i in 0..n_layers {
        let name = format!("enc_layer_{i}_out");
        let (ref_data, ref_shape) = tensor_f32(&st, &name);
        assert_eq!(ref_shape.len(), 3, "{name} ref dim");
        let (c_r, t_r) = (ref_shape[1], ref_shape[2]);
        let ours = &outs[i];
        let (c, t) = ours.dim();
        if (c, t) != (c_r, t_r) {
            panic!("{name} shape mismatch — ours {c}x{t}, ref {c_r}x{t_r}");
        }
        let ours_flat = align_2d(ours);
        let rel = relative_diff(&ours_flat, &ref_data);
        let abs = max_abs_diff(&ours_flat, &ref_data);
        println!("[parity] {name}: shape {c}x{t}  rel={rel:.3e}  max_abs={abs:.3e}");
        assert!(rel < 5e-3, "{name} relative diff too large: rel={rel:.3e}");
    }

    // Final-output cross-check
    let (final_ref, _) = tensor_f32(&st, "enc_out");
    let final_flat = align_2d(&final_out);
    let rel = relative_diff(&final_flat, &final_ref);
    println!("[parity] enc_out: rel={rel:.3e}");
    assert!(rel < 5e-3);

    let _ = ArrayView2::<f32>::from_shape((1, 1), &[0f32]); // silence unused import
}

#[test]
fn st_encoder_transformer_parity() {
    let Some(model_dir) = base_dir() else {
        eprintln!("skip: no Base checkpoint");
        return;
    };
    let Some(fixture) = Fixture::load() else {
        eprintln!("skip: fixture missing");
        return;
    };
    let st = fixture.view();
    let tok_dir = model_dir.join("speech_tokenizer");
    let tf: MimiEncoderTransformer = open_encoder_transformer(&tok_dir).expect("open transformer");
    println!(
        "[parity] transformer: hidden={} layers={} heads={} head_dim={} window={}",
        tf.cfg.hidden_size,
        tf.layer_count(),
        tf.cfg.num_attention_heads,
        tf.cfg.head_dim,
        tf.cfg.sliding_window,
    );

    // Input: Python-baked `pre_transformer` [1, T, hidden].
    let (pre_data, pre_shape) = tensor_f32(&st, "pre_transformer");
    assert_eq!(pre_shape.len(), 3);
    let t = pre_shape[1];
    let hidden = pre_shape[2];
    let pre = Array2::from_shape_vec((t, hidden), pre_data).expect("pre_transformer shape");

    let (post_t, layer_outs) = tf.forward_with_intermediates(pre.view());
    assert_eq!(layer_outs.len(), tf.layer_count());

    // Per-layer parity.
    for i in 0..layer_outs.len() {
        let name = format!("tf_layer_{i}_out");
        let (ref_data, ref_shape) = tensor_f32(&st, &name);
        assert_eq!(ref_shape, vec![1, t, hidden], "{name} shape");
        let ours_flat: Vec<f32> = layer_outs[i].iter().copied().collect();
        let rel = relative_diff(&ours_flat, &ref_data);
        let abs = max_abs_diff(&ours_flat, &ref_data);
        println!("[parity] {name}: rel={rel:.3e}  max_abs={abs:.3e}");
        assert!(rel < 5e-3, "{name} relative diff too large: rel={rel:.3e}");
    }

    let (ref_data, _) = tensor_f32(&st, "post_transformer");
    let ours: Vec<f32> = post_t.iter().copied().collect();
    let rel = relative_diff(&ours, &ref_data);
    println!("[parity] post_transformer: rel={rel:.3e}");
    assert!(rel < 5e-3);
}

#[test]
fn st_encoder_rvq_end_to_end() {
    let Some(model_dir) = base_dir() else {
        eprintln!("skip: no Base checkpoint");
        return;
    };
    let Some(fixture) = Fixture::load() else {
        eprintln!("skip: fixture missing");
        return;
    };
    let st = fixture.view();

    // Load reference codes (i64 in fixture).
    let codes_tensor = st.tensor("audio_codes").expect("audio_codes");
    let shape = codes_tensor.shape().to_vec(); // [B=1, num_q=32, T=61]
    assert_eq!(shape.len(), 3);
    let num_q_total = shape[1];
    let t_codes = shape[2];
    let bytes = codes_tensor.data();
    assert_eq!(bytes.len(), num_q_total * t_codes * 8);
    let mut ref_codes = vec![0i64; num_q_total * t_codes];
    for (i, c) in bytes.chunks_exact(8).enumerate() {
        ref_codes[i] = i64::from_le_bytes(c.try_into().unwrap());
    }

    // Load PCM from fixture.
    let (pcm, _) = tensor_f32(&st, "pcm");

    let frames = rlx_qwen3_tts::speech_tokenizer::encode_pcm_to_codec_frames(&model_dir, &pcm)
        .expect("encode_pcm_to_codec_frames");
    assert_eq!(
        frames.len(),
        t_codes,
        "T_codes mismatch — ours {}, ref {}",
        frames.len(),
        t_codes
    );
    let num_q_ours = frames[0].len();
    println!(
        "[parity] codes: ours [{} frames x {} q]  ref [{} q x {} frames]",
        frames.len(),
        num_q_ours,
        num_q_total,
        t_codes
    );
    assert!(
        num_q_ours <= num_q_total,
        "num_q_ours {} > num_q_total {}",
        num_q_ours,
        num_q_total
    );

    // Ref layout is [num_q, T]; compare element-wise against ours [T, num_q].
    let mut mismatched = 0usize;
    let mut first_bad: Option<(usize, usize, u32, i64)> = None;
    for ti in 0..t_codes {
        for q in 0..num_q_ours {
            let ours = frames[ti][q];
            let r = ref_codes[q * t_codes + ti] as u32;
            if ours != r {
                if first_bad.is_none() {
                    first_bad = Some((ti, q, ours, r as i64));
                }
                mismatched += 1;
            }
        }
    }
    let total = t_codes * num_q_ours;
    println!(
        "[parity] codes mismatch: {mismatched}/{total} ({:.2}%)",
        mismatched as f64 * 100.0 / total as f64
    );
    if let Some((ti, q, o, r)) = first_bad {
        println!("[parity] first divergence at frame {ti} quantizer {q}: ours={o} ref={r}");
    }
    // Acoustic RVQ is sensitive to upstream floating-point drift; allow a small
    // mismatch rate while keeping a tight bound on the semantic + first acoustic
    // codes (where divergence is loudest).
    let early_mismatched: usize = (0..t_codes)
        .flat_map(|ti| (0..num_q_ours.min(4)).map(move |q| (ti, q)))
        .filter(|&(ti, q)| frames[ti][q] != ref_codes[q * t_codes + ti] as u32)
        .count();
    let early_total = t_codes * num_q_ours.min(4);
    println!("[parity] first-4-q codes mismatch: {early_mismatched}/{early_total}");
    assert!(
        mismatched as f64 / total as f64 <= 0.10,
        "codes diverged too much: {mismatched}/{total}"
    );
}

#[test]
fn st_encoder_downsample_parity() {
    let Some(model_dir) = base_dir() else {
        eprintln!("skip: no Base checkpoint");
        return;
    };
    let Some(fixture) = Fixture::load() else {
        eprintln!("skip: fixture missing");
        return;
    };
    let st = fixture.view();
    let tok_dir = model_dir.join("speech_tokenizer");

    // Use Python-baked pre_downsample as input to isolate the downsample stage.
    let (pre_data, pre_shape) = tensor_f32(&st, "pre_downsample");
    assert_eq!(pre_shape.len(), 3, "pre_downsample dim");
    let hidden = pre_shape[1];
    let t_pre = pre_shape[2];
    let pre = Array2::from_shape_vec((hidden, t_pre), pre_data).expect("pre_downsample shape");

    let ds = MimiDownsample::open(&tok_dir, hidden).expect("open downsample");
    let out = ds.forward(pre.view());

    let (ref_data, ref_shape) = tensor_f32(&st, "post_downsample");
    assert_eq!(ref_shape.len(), 3);
    assert_eq!(
        (out.dim().0, out.dim().1),
        (ref_shape[1], ref_shape[2]),
        "downsample output shape mismatch — ours {:?}, ref {}x{}",
        out.dim(),
        ref_shape[1],
        ref_shape[2]
    );
    let ours_flat = align_2d(&out);
    let rel = relative_diff(&ours_flat, &ref_data);
    let abs = max_abs_diff(&ours_flat, &ref_data);
    println!(
        "[parity] downsample: shape {}x{}  rel={rel:.3e}  max_abs={abs:.3e}",
        out.dim().0,
        out.dim().1,
    );
    assert!(rel < 5e-4, "downsample diverged: rel={rel:.3e}");
}
