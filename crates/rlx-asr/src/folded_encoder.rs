// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Folded encoder: mel → input_proj_eff → optional h-map → body → CTC log-probs.
//!
//! Body map preference:
//! 1. CE residual MLP sidecars `body_mlp_ce.*` (`enc = h@B + tanh(h@W1+b1)@W2+b2`)
//! 2. Else `body_map_ls.B` (`enc = h @ B`)
//! 3. Else pack `frontend.body_residual_ls.R` (`enc = h @ R`), optional `body_out_ls.A`
//!
//! Live path also applies `body_h_map_ls.M` after input_proj when frontend is LS
//! (`h = h @ M`), bridging our-mel projections toward Apple teacher-h.
//!
//! Mirrors `tools/e2e_native_whole.py::forward_folded`.

use crate::env::{
    FrontendMode, body_mlp_ce_enabled, body_out_ls_dir, body_out_ls_enabled, frontend_ls_dir,
    frontend_mode, h_map_enabled,
};
use crate::gguf_io::AsrPack;
use crate::spec::{AED_WINDOW_FRAMES, DECODER_DIM, MEL_BINS, SUBSAMPLE, VOCAB};
use anyhow::{Context, Result, bail};
use std::path::Path;

/// Mel frames per folded chunk (see `tools/audio_io.py::FRAMES`).
pub const FOLDED_MEL_FRAMES: usize = 389;
/// Subsampled encoder time steps per chunk.
pub const FOLDED_OUT_T: usize = 64;

fn enc_key(suffix: &str) -> String {
    format!("encoder.{suffix}")
}

/// CE-trained residual body MLP sidecars (`body_mlp_ce.*.bin`).
struct BodyMlpCe {
    /// Linear trunk `[D,D]` (same layout as body_map).
    b: Vec<f32>,
    w1: Vec<f32>, // [D, HID]
    b1: Vec<f32>, // [HID]
    w2: Vec<f32>, // [HID, D]
    b2: Vec<f32>, // [D]
    hid: usize,
}

/// Loaded folded encoder + CTC head from an ASR pack.
pub struct FoldedEncoder {
    input_w: Vec<f32>,
    input_w_t: bool,
    input_b: Option<Vec<f32>>,
    body_r: Vec<f32>,
    /// Preferred Apple-capture LS map `enc = h @ B` (replaces pack bodyR when present).
    body_map: Option<Vec<f32>>,
    /// Optional CE residual MLP: `enc = h@B + tanh(h@W1+b1)@W2 + b2` (HID=512).
    body_mlp: Option<BodyMlpCe>,
    ctc_w: Vec<f32>,
    ctc_w_t: bool,
    ctc_b: Option<Vec<f32>>,
    /// Per-bin affine calibration (optional, silence target).
    cal_a: Option<Vec<f32>>,
    cal_b: Option<Vec<f32>>,
    /// Cross-wav LS affine to Apple speech features (optional).
    ls_a: Option<Vec<f32>>,
    ls_b: Option<Vec<f32>>,
    /// Live-path h-space LS: `h = h @ M` after input_proj (our-mel → teacher-h).
    h_map: Option<Vec<f32>>,
    /// Optional LS correction after bodyR (`enc = (h @ R) @ A`). Not applied when `body_map` is set.
    body_out_a: Option<Vec<f32>>,
}

impl FoldedEncoder {
    /// Load required tensors from a weight pack.
    pub fn from_pack(pack: &AsrPack) -> Result<Self> {
        let input_w_raw = pack
            .f32_tensor(&enc_key("frontend.input_proj_eff.W"))
            .context("encoder.frontend.input_proj_eff.W")?;
        let body_r = pack
            .f32_tensor(&enc_key("frontend.body_residual_ls.R"))
            .context("encoder.frontend.body_residual_ls.R")?;

        // Published pack stores `input_proj_eff.W` as [80, 512] (mel @ W).
        // Note MEL_BINS*DIM == DIM*MEL_BINS, so length alone cannot disambiguate —
        // prefer the published [in, out] layout (w_transposed=false).
        let (input_w, input_w_t) = if input_w_raw.len() == MEL_BINS * DECODER_DIM {
            (input_w_raw, false)
        } else {
            bail!(
                "input_proj_eff.W len {} (expected {})",
                input_w_raw.len(),
                MEL_BINS * DECODER_DIM
            );
        };

        if body_r.len() != DECODER_DIM * DECODER_DIM {
            bail!("body_residual_ls.R len {}", body_r.len());
        }

        let ctc_w_key = if pack.has(&enc_key("head.ctc.W_ls")) {
            enc_key("head.ctc.W_ls")
        } else {
            enc_key("head.ctc.W")
        };
        let ctc_w_raw = pack
            .f32_tensor(&ctc_w_key)
            .with_context(|| ctc_w_key.clone())?;
        let (ctc_w, ctc_w_t) = if ctc_w_raw.len() == VOCAB * DECODER_DIM {
            (ctc_w_raw, true)
        } else if ctc_w_raw.len() == DECODER_DIM * VOCAB {
            (ctc_w_raw, false)
        } else {
            bail!("CTC W len {}", ctc_w_raw.len());
        };

        let input_b = pack
            .f32_tensor(&enc_key("frontend.input_proj_eff.b"))
            .ok()
            .filter(|b| b.len() == DECODER_DIM);
        let ctc_b = pack
            .f32_tensor(&enc_key("head.ctc.b_ls"))
            .or_else(|_| pack.f32_tensor(&enc_key("head.ctc.b")))
            .ok()
            .filter(|b| b.len() == VOCAB);

        let (cal_a, cal_b) = fit_silence_calibration(pack).unwrap_or((None, None));
        let (ls_a, ls_b) = load_ls_cross_wav(pack)
            .map(|(a, b)| (Some(a), Some(b)))
            .unwrap_or((None, None));
        let body_map = load_body_map_ls(pack);
        let body_mlp = if body_mlp_ce_enabled() {
            load_body_mlp_ce()
        } else {
            None
        };
        let h_map = if h_map_enabled() {
            load_h_map_ls()
        } else {
            None
        };
        let body_out_a = if body_map.is_none() && body_mlp.is_none() && body_out_ls_enabled() {
            load_body_out_ls(pack)
        } else {
            None
        };

        Ok(Self {
            input_w,
            input_w_t,
            input_b,
            body_r,
            body_map,
            body_mlp,
            ctc_w,
            ctc_w_t,
            ctc_b,
            cal_a,
            cal_b,
            ls_a,
            ls_b,
            h_map,
            body_out_a,
        })
    }

    pub fn is_available(pack: &AsrPack) -> bool {
        pack.has(&enc_key("frontend.input_proj_eff.W"))
            && pack.has(&enc_key("frontend.body_residual_ls.R"))
            && (pack.has(&enc_key("head.ctc.W_ls")) || pack.has(&enc_key("head.ctc.W")))
    }

    pub fn has_h_map(&self) -> bool {
        self.h_map.is_some()
    }

    pub fn has_body_mlp(&self) -> bool {
        self.body_mlp.is_some()
    }

    /// Preprocess mel frames per `RLX_ASR_FRONTEND` before encoder forward.
    pub fn preprocess_mel(&self, mel: &mut [Vec<f32>]) {
        match frontend_mode() {
            FrontendMode::Raw => {}
            FrontendMode::Calibrated => self.calibrate_mel(mel),
            FrontendMode::LsCrossWav => {
                if let (Some(a), Some(b)) = (&self.ls_a, &self.ls_b) {
                    apply_affine_mel(mel, a, b);
                }
            }
        }
    }

    /// Apply silence-fbank calibration when loaded and non-degenerate.
    pub fn calibrate_mel(&self, mel: &mut [Vec<f32>]) {
        let (Some(a), Some(b)) = (&self.cal_a, &self.cal_b) else {
            return;
        };
        if a.iter().all(|&v| v.abs() < 1e-4) {
            return;
        }
        for frame in mel.iter_mut() {
            if frame.len() != MEL_BINS {
                continue;
            }
            for i in 0..MEL_BINS {
                frame[i] = frame[i] * a[i] + b[i];
            }
        }
    }

    /// Run folded forward on one 389-frame mel window → `(enc [T×D], logp [T×V])`.
    pub fn forward_chunk(&self, feat389: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>)> {
        if feat389.len() < FOLDED_MEL_FRAMES {
            bail!(
                "folded chunk needs {} mel frames, got {}",
                FOLDED_MEL_FRAMES,
                feat389.len()
            );
        }
        let mut mel_sub = vec![0f32; FOLDED_OUT_T * MEL_BINS];
        for t in 0..FOLDED_OUT_T {
            let idx = t * SUBSAMPLE;
            let src = &feat389[idx];
            if src.len() != MEL_BINS {
                bail!("mel frame {idx} has {} bins", src.len());
            }
            mel_sub[t * MEL_BINS..(t + 1) * MEL_BINS].copy_from_slice(src);
        }

        let mut h = vec![0f32; FOLDED_OUT_T * DECODER_DIM];
        matmul_rows(
            &mel_sub,
            FOLDED_OUT_T,
            MEL_BINS,
            DECODER_DIM,
            &self.input_w,
            self.input_w_t,
            self.input_b.as_deref(),
            &mut h,
        );
        if let Some(m) = &self.h_map {
            apply_square_map_inplace(&mut h, FOLDED_OUT_T, m);
        }

        let mut enc = vec![0f32; FOLDED_OUT_T * DECODER_DIM];
        if let Some(mlp) = &self.body_mlp {
            apply_body_mlp_ce(&h, FOLDED_OUT_T, mlp, &mut enc);
        } else {
            let body = self.body_map.as_deref().unwrap_or(&self.body_r);
            for t in 0..FOLDED_OUT_T {
                let h_row = &h[t * DECODER_DIM..(t + 1) * DECODER_DIM];
                let e_row = &mut enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
                for d in 0..DECODER_DIM {
                    let mut acc = 0f32;
                    for k in 0..DECODER_DIM {
                        acc += h_row[k] * body[k * DECODER_DIM + d];
                    }
                    e_row[d] = acc;
                }
            }
            if self.body_map.is_none()
                && let Some(a) = &self.body_out_a
            {
                apply_body_out(&mut enc, FOLDED_OUT_T, a);
            }
        }

        let logp = self.ctc_logp(&enc, FOLDED_OUT_T)?;
        Ok((enc, logp))
    }

    /// Full utterance: chunk mel → concat CTC log-probs + encoder cache for AED.
    pub fn forward_mel(&self, mel: &[Vec<f32>]) -> Result<crate::encoder::EncoderOutputs> {
        self.forward_mel_limited(mel, None)
    }

    /// Like [`Self::forward_mel`] but process at most `max_chunks` windows (probe / early abort).
    ///
    /// Caller must apply [`Self::preprocess_mel`] first when using LS/calibrated frontend.
    pub fn forward_mel_limited(
        &self,
        mel: &[Vec<f32>],
        max_chunks: Option<usize>,
    ) -> Result<crate::encoder::EncoderOutputs> {
        if mel.is_empty() {
            bail!("empty mel");
        }
        // Non-overlapping 389-frame windows (matches `audio_io.mel_windows` default hop).
        let mut chunks = mel_windows(mel, FOLDED_MEL_FRAMES, FOLDED_MEL_FRAMES);
        if let Some(n) = max_chunks {
            chunks.truncate(n);
        }
        let mut all_logp = Vec::new();
        let mut all_enc = Vec::new();
        for chunk in &chunks {
            let (enc, logp) = self.forward_chunk(chunk)?;
            all_enc.extend(enc);
            all_logp.extend(logp);
        }
        let n_frames = all_logp.len() / VOCAB;
        let encoder_cache = enc_to_aed_cache(&all_enc);
        Ok(crate::encoder::EncoderOutputs {
            wp_logprob: all_logp,
            encoder_cache,
            n_frames,
        })
    }

    fn ctc_logp(&self, enc: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        let mut logits = vec![0f32; n_frames * VOCAB];
        for t in 0..n_frames {
            let row = &enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
            let out = &mut logits[t * VOCAB..(t + 1) * VOCAB];
            if self.ctc_w_t {
                for v in 0..VOCAB {
                    let mut s = 0f32;
                    for d in 0..DECODER_DIM {
                        s += row[d] * self.ctc_w[v * DECODER_DIM + d];
                    }
                    out[v] = s;
                }
            } else {
                for v in 0..VOCAB {
                    let mut s = 0f32;
                    for d in 0..DECODER_DIM {
                        s += row[d] * self.ctc_w[d * VOCAB + v];
                    }
                    out[v] = s;
                }
            }
            if let Some(b) = &self.ctc_b {
                for v in 0..VOCAB {
                    out[v] += b[v];
                }
            }
            log_softmax_row(out);
        }
        Ok(logits)
    }
}

fn matmul_rows(
    a: &[f32],
    n_rows: usize,
    k: usize,
    n_cols: usize,
    w: &[f32],
    w_transposed: bool,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for r in 0..n_rows {
        let a_row = &a[r * k..(r + 1) * k];
        let o_row = &mut out[r * n_cols..(r + 1) * n_cols];
        for c in 0..n_cols {
            let mut s = 0f32;
            for i in 0..k {
                let wv = if w_transposed {
                    w[c * k + i]
                } else {
                    w[i * n_cols + c]
                };
                s += a_row[i] * wv;
            }
            o_row[c] = s;
        }
        if let Some(b) = bias {
            for c in 0..n_cols {
                o_row[c] += b[c];
            }
        }
    }
}

fn log_softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let v = -(row.len() as f32).ln();
        row.fill(v);
        return;
    }
    let mut sum = 0f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum <= 0.0 || !sum.is_finite() {
        let v = -(row.len() as f32).ln();
        row.fill(v);
        return;
    }
    let log_sum = sum.ln();
    for x in row.iter_mut() {
        *x = (*x).ln() - log_sum;
    }
}

/// Pad/truncate encoder frames into AED window cache.
fn enc_to_aed_cache(enc: &[f32]) -> Vec<f32> {
    let n_enc_frames = enc.len() / DECODER_DIM;
    let mut cache = vec![0f32; AED_WINDOW_FRAMES * DECODER_DIM];
    if n_enc_frames == 0 {
        return cache;
    }
    let copy_frames = n_enc_frames.min(AED_WINDOW_FRAMES);
    let start = n_enc_frames.saturating_sub(copy_frames);
    for (i, t) in (start..start + copy_frames).enumerate() {
        let src = &enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        cache[i * DECODER_DIM..(i + 1) * DECODER_DIM].copy_from_slice(src);
    }
    cache
}

/// Split mel into `frame_len`-frame windows with hop `step`.
///
/// When the tail is shorter than `frame_len`, emits one final padded window so
/// leftover frames are not dropped (e.g. 766 frames / hop 389 → 2 windows).
/// Partial windows pad with the last real frame (not frame 0) to avoid injecting
/// leading-silence spectrum into the tail chunk.
pub fn mel_windows(mel: &[Vec<f32>], frame_len: usize, step: usize) -> Vec<Vec<Vec<f32>>> {
    let step = step.max(1);
    let fill = if mel.is_empty() {
        vec![0f32; MEL_BINS]
    } else {
        mel[mel.len() - 1].clone()
    };
    if mel.len() < frame_len {
        let mut padded: Vec<Vec<f32>> = mel.to_vec();
        while padded.len() < frame_len {
            padded.push(fill.clone());
        }
        return vec![padded];
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    while start + frame_len <= mel.len() {
        windows.push(mel[start..start + frame_len].to_vec());
        start += step;
    }
    if start < mel.len() {
        let rem = mel.len() - start;
        // Skip tiny tails — padded short windows often decode as CTC stutter.
        if rem * 2 >= frame_len {
            let mut padded: Vec<Vec<f32>> = mel[start..].to_vec();
            while padded.len() < frame_len {
                padded.push(fill.clone());
            }
            windows.push(padded);
        }
    }
    if windows.is_empty() {
        let mut padded: Vec<Vec<f32>> = mel.to_vec();
        while padded.len() < frame_len {
            padded.push(fill.clone());
        }
        windows.push(padded);
    }
    windows
}

/// Fit per-bin affine calibration from silence fbank (see `audio_io.fit_silence_calibration`).
fn load_body_map_ls(pack: &AsrPack) -> Option<Vec<f32>> {
    let key = "encoder.body_map_ls.B";
    if pack.has(key) {
        let b = pack.f32_tensor(key).ok()?;
        if b.len() == DECODER_DIM * DECODER_DIM {
            return Some(b);
        }
    }
    for dir in sidecar_dirs(body_out_ls_dir()) {
        if let Some(b) = load_f32_sidecar_matrix(&dir, "body_map_ls.B.bin") {
            return Some(b);
        }
    }
    None
}

const BODY_MLP_HID: usize = 512;

fn load_body_mlp_ce() -> Option<BodyMlpCe> {
    if let Some(dir) = sidecar_dirs(None).into_iter().next() {
        let b = load_f32_file(&dir.join("body_mlp_ce.B.bin"), DECODER_DIM * DECODER_DIM)?;
        let w1 = load_f32_file(&dir.join("body_mlp_ce.W1.bin"), DECODER_DIM * BODY_MLP_HID)?;
        let b1 = load_f32_file(&dir.join("body_mlp_ce.b1.bin"), BODY_MLP_HID)?;
        let w2 = load_f32_file(&dir.join("body_mlp_ce.W2.bin"), BODY_MLP_HID * DECODER_DIM)?;
        let b2 = load_f32_file(&dir.join("body_mlp_ce.b2.bin"), DECODER_DIM)?;
        return Some(BodyMlpCe {
            b,
            w1,
            b1,
            w2,
            b2,
            hid: BODY_MLP_HID,
        });
    }
    None
}

fn load_h_map_ls() -> Option<Vec<f32>> {
    for dir in sidecar_dirs(None) {
        if let Some(m) = load_f32_sidecar_matrix(&dir, "body_h_map_ls.M.bin") {
            return Some(m);
        }
    }
    None
}

fn apply_square_map_inplace(h: &mut [f32], n_frames: usize, m: &[f32]) {
    if m.len() != DECODER_DIM * DECODER_DIM {
        return;
    }
    let mut tmp = vec![0f32; n_frames * DECODER_DIM];
    for t in 0..n_frames {
        let row = &h[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        let out = &mut tmp[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        for d in 0..DECODER_DIM {
            let mut acc = 0f32;
            for k in 0..DECODER_DIM {
                acc += row[k] * m[k * DECODER_DIM + d];
            }
            out[d] = acc;
        }
    }
    h.copy_from_slice(&tmp);
}

fn load_f32_file(path: &Path, expected: usize) -> Option<Vec<f32>> {
    if !path.is_file() {
        return None;
    }
    let raw = std::fs::read(path).ok()?;
    if raw.len() != expected * 4 {
        return None;
    }
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn apply_body_mlp_ce(h: &[f32], n_frames: usize, mlp: &BodyMlpCe, enc: &mut [f32]) {
    let hid = mlp.hid;
    for t in 0..n_frames {
        let h_row = &h[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        let e_row = &mut enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        // linear trunk: h @ B
        for d in 0..DECODER_DIM {
            let mut acc = 0f32;
            for k in 0..DECODER_DIM {
                acc += h_row[k] * mlp.b[k * DECODER_DIM + d];
            }
            e_row[d] = acc;
        }
        // residual: tanh(h @ W1 + b1) @ W2 + b2
        // W1 stored [D, HID] row-major → (h @ W1)[j] = sum_k h[k]*W1[k*HID+j]
        let mut hidden = vec![0f32; hid];
        for j in 0..hid {
            let mut acc = mlp.b1[j];
            for k in 0..DECODER_DIM {
                acc += h_row[k] * mlp.w1[k * hid + j];
            }
            hidden[j] = acc.tanh();
        }
        // W2 stored [HID, D] → (hid @ W2)[d] = sum_j hidden[j]*W2[j*D+d]
        for d in 0..DECODER_DIM {
            let mut acc = mlp.b2[d];
            for j in 0..hid {
                acc += hidden[j] * mlp.w2[j * DECODER_DIM + d];
            }
            e_row[d] += acc;
        }
    }
}

fn load_body_out_ls(pack: &AsrPack) -> Option<Vec<f32>> {
    let key = "encoder.body_out_ls.A";
    if pack.has(key) {
        let a = pack.f32_tensor(key).ok()?;
        if a.len() == DECODER_DIM * DECODER_DIM {
            return Some(a);
        }
    }
    for dir in sidecar_dirs(body_out_ls_dir()) {
        if let Some(a) = load_f32_sidecar_matrix(&dir, "body_out_ls.A.bin") {
            return Some(a);
        }
    }
    None
}

fn load_f32_sidecar_matrix(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(name);
    if !p.is_file() {
        return None;
    }
    let raw = std::fs::read(p).ok()?;
    if raw.len() != DECODER_DIM * DECODER_DIM * 4 {
        return None;
    }
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn apply_body_out(enc: &mut [f32], n_frames: usize, a: &[f32]) {
    if a.len() != DECODER_DIM * DECODER_DIM {
        return;
    }
    let mut tmp = vec![0f32; n_frames * DECODER_DIM];
    for t in 0..n_frames {
        let row = &enc[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        let out = &mut tmp[t * DECODER_DIM..(t + 1) * DECODER_DIM];
        for d in 0..DECODER_DIM {
            let mut acc = 0f32;
            for k in 0..DECODER_DIM {
                acc += row[k] * a[k * DECODER_DIM + d];
            }
            out[d] = acc;
        }
    }
    enc.copy_from_slice(&tmp);
}

fn load_f32_sidecar(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(name);
    if !p.is_file() {
        return None;
    }
    let raw = std::fs::read(p).ok()?;
    if raw.len() % 4 != 0 {
        return None;
    }
    let v: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if v.len() >= MEL_BINS {
        Some(v[..MEL_BINS].to_vec())
    } else {
        None
    }
}

fn load_ls_cross_wav(pack: &AsrPack) -> Option<(Vec<f32>, Vec<f32>)> {
    let a_key = "encoder.frontend.ls_cross_wav.a";
    let b_key = "encoder.frontend.ls_cross_wav.b";
    if pack.has(a_key) && pack.has(b_key) {
        let a = pack.f32_tensor(a_key).ok()?;
        let b = pack.f32_tensor(b_key).ok()?;
        if a.len() >= MEL_BINS && b.len() >= MEL_BINS {
            return Some((a[..MEL_BINS].to_vec(), b[..MEL_BINS].to_vec()));
        }
    }
    for dir in sidecar_dirs(frontend_ls_dir()) {
        if let (Some(a), Some(b)) = (
            load_f32_sidecar(&dir, "frontend_fbank_ls_cross_wav_a.bin"),
            load_f32_sidecar(&dir, "frontend_fbank_ls_cross_wav_b.bin"),
        ) {
            return Some((a, b));
        }
    }
    None
}

/// Sidecar lookup: explicit env dir, then `RLX_ASR_DIR` root.
fn sidecar_dirs(explicit: Option<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Some(d) = explicit {
        out.push(d);
    }
    let root = crate::env::asr_dir();
    if root.is_dir() && !out.iter().any(|p| p == &root) {
        out.push(root);
    }
    out
}

fn apply_affine_mel(mel: &mut [Vec<f32>], a: &[f32], b: &[f32]) {
    if a.len() < MEL_BINS || b.len() < MEL_BINS {
        return;
    }
    for frame in mel.iter_mut() {
        if frame.len() != MEL_BINS {
            continue;
        }
        for i in 0..MEL_BINS {
            frame[i] = frame[i] * a[i] + b[i];
        }
    }
}

fn fit_silence_calibration(pack: &AsrPack) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
    let silence = pack.silence_fbank()?;
    if silence.len() < MEL_BINS {
        return Ok((None, None));
    }
    let silence = &silence[..MEL_BINS];
    let raw =
        crate::frontend::log_mel_fbank(&vec![0f32; crate::frontend::SAMPLE_RATE as usize], 16_000)?;
    if raw.is_empty() {
        return Ok((None, None));
    }
    let mut m = vec![0f32; MEL_BINS];
    for frame in &raw {
        for i in 0..MEL_BINS {
            m[i] += frame[i];
        }
    }
    for v in &mut m {
        *v /= raw.len() as f32;
    }
    let mut num = 0f32;
    let mut den = 0f32;
    for i in 0..MEL_BINS {
        num += m[i] * silence[i];
        den += m[i] * m[i];
    }
    if den <= 1e-8 {
        return Ok((None, None));
    }
    let a = num / den;
    let a_vec = vec![a; MEL_BINS];
    let b: Vec<f32> = (0..MEL_BINS).map(|i| silence[i] - a * m[i]).collect();
    Ok((Some(a_vec), Some(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_windows_short_pad() {
        let mel: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0; MEL_BINS]).collect();
        let w = mel_windows(&mel, FOLDED_MEL_FRAMES, 4);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].len(), FOLDED_MEL_FRAMES);
    }

    #[test]
    fn mel_windows_covers_tail() {
        // 766 frames / hop 389 previously dropped the last 377 frames.
        let mel: Vec<Vec<f32>> = (0..766).map(|i| vec![i as f32; MEL_BINS]).collect();
        let w = mel_windows(&mel, FOLDED_MEL_FRAMES, FOLDED_MEL_FRAMES);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].len(), FOLDED_MEL_FRAMES);
        assert_eq!(w[1].len(), FOLDED_MEL_FRAMES);
        assert_eq!(w[1][0][0], 389.0);
        assert_eq!(w[1][376][0], 765.0);
        assert_eq!(w[1][377][0], 765.0); // pad uses last real frame
    }

    #[test]
    fn folded_encoder_from_published_pack() {
        let root = crate::asr_dir();
        let Some(path) = crate::gguf_io::resolve_pack_path(&root) else {
            return;
        };
        let pack = AsrPack::open(&path).expect("open pack");
        if !FoldedEncoder::is_available(&pack) {
            return;
        }
        let enc = FoldedEncoder::from_pack(&pack).expect("load folded");
        let feat: Vec<Vec<f32>> = (0..FOLDED_MEL_FRAMES)
            .map(|t| {
                (0..MEL_BINS)
                    .map(|b| ((t + b) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let (_, logp) = enc.forward_chunk(&feat).unwrap();
        assert_eq!(logp.len(), FOLDED_OUT_T * VOCAB);
        assert!(
            logp.iter().any(|v| v.is_finite()),
            "expected finite CTC log-probs"
        );
        let best = logp
            .chunks(VOCAB)
            .flat_map(|row| row.iter().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(best.is_finite(), "best logprob {best}");
    }

    #[test]
    fn dump_ask_not_enc_parity() {
        let root = std::path::Path::new("/Users/Shared/translator/models/rlx-asr");
        if !root.join("model.rlxp").is_file() {
            return;
        }
        // SAFETY: test-only process-local env for sidecar loading.
        unsafe {
            std::env::set_var("RLX_ASR_DIR", root);
            std::env::set_var("RLX_ASR_H_MAP", "1");
            std::env::set_var("RLX_ASR_BODY_MLP", "1");
            std::env::set_var("RLX_ASR_FRONTEND", "ls");
        }
        let Ok(shape_s) = std::fs::read_to_string("/tmp/rust_ask_mel_shape.txt") else {
            return;
        };
        let mut parts = shape_s.split_whitespace();
        let t: usize = parts.next().unwrap().parse().unwrap();
        let f: usize = parts.next().unwrap().parse().unwrap();
        let raw = std::fs::read("/tmp/rust_ask_mel.bin").unwrap();
        let mel_flat: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut mel: Vec<Vec<f32>> = mel_flat.chunks(f).map(|r| r.to_vec()).collect();
        assert_eq!(mel.len(), t);
        let pack = AsrPack::open(root.join("model.rlxp")).unwrap();
        let enc = FoldedEncoder::from_pack(&pack).unwrap();
        enc.preprocess_mel(&mut mel);
        let mut feat = mel[..FOLDED_MEL_FRAMES.min(mel.len())].to_vec();
        while feat.len() < FOLDED_MEL_FRAMES {
            feat.push(feat.last().cloned().unwrap_or_else(|| vec![0f32; MEL_BINS]));
        }
        let (e, logp) = enc.forward_chunk(&feat).unwrap();
        let bytes: Vec<u8> = e.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write("/tmp/rust_ask_enc.bin", bytes).unwrap();
        let nb = logp
            .chunks(VOCAB)
            .filter(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    != Some(0)
            })
            .count();
        std::fs::write("/tmp/rust_ask_enc_meta.txt", format!("nonblank={nb}\n")).unwrap();
    }
}
