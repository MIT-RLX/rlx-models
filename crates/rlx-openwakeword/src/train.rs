// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train openWakeWord **phrase head** in RLX (embedding frozen).

use anyhow::Result;
use rlx_wake::ops::{gemv_bias, relu, sigmoid};
use rlx_wake::train::dataset::LabeledClip;
use rlx_wake::train::report::TrainReport;
use rlx_wake::train::sgd::{SgdConfig, bce_dlogit, bce_loss, sgd_step};
use rlx_wake::{MelConfig, MelFrontend, OWW_CHUNK_SAMPLES};

use crate::embedding::{
    EMBED_DIM, EmbeddingNet, EmbeddingWeights, MEL_STEP, MEL_WINDOW,
};
use crate::phrase::{EMBED_HISTORY, PhraseWeights};

fn collect_embed_window(
    embed: &EmbeddingNet,
    pcm: &[f32],
) -> Option<[[f32; EMBED_DIM]; EMBED_HISTORY]> {
    let mut mel = MelFrontend::new(MelConfig::default());
    let mut mel_frames: Vec<Vec<f32>> = Vec::new();
    let mut embeds: Vec<[f32; EMBED_DIM]> = Vec::new();
    let mut i = 0usize;
    while i < pcm.len() {
        let end = (i + OWW_CHUNK_SAMPLES).min(pcm.len());
        let mut chunk = pcm[i..end].to_vec();
        if chunk.len() < OWW_CHUNK_SAMPLES {
            chunk.resize(OWW_CHUNK_SAMPLES, 0.0);
        }
        let flat = mel.push(&chunk);
        let n_mels = mel.n_mels();
        let n_new = flat.len() / n_mels.max(1);
        for j in 0..n_new {
            mel_frames.push(flat[j * n_mels..(j + 1) * n_mels].to_vec());
        }
        while mel_frames.len() >= MEL_WINDOW {
            let mut window = Vec::with_capacity(MEL_WINDOW * n_mels);
            for f in mel_frames.iter().take(MEL_WINDOW) {
                window.extend_from_slice(f);
            }
            embeds.push(embed.forward(&window));
            for _ in 0..MEL_STEP.min(mel_frames.len()) {
                mel_frames.remove(0);
            }
        }
        i += OWW_CHUNK_SAMPLES;
    }
    if embeds.len() < EMBED_HISTORY {
        return None;
    }
    let mut hist = [[0.0f32; EMBED_DIM]; EMBED_HISTORY];
    for (k, e) in embeds.iter().rev().take(EMBED_HISTORY).enumerate() {
        hist[EMBED_HISTORY - 1 - k] = *e;
    }
    Some(hist)
}

fn flatten(hist: &[[f32; EMBED_DIM]; EMBED_HISTORY]) -> Vec<f32> {
    let mut flat = vec![0.0f32; EMBED_HISTORY * EMBED_DIM];
    for (i, e) in hist.iter().enumerate() {
        flat[i * EMBED_DIM..(i + 1) * EMBED_DIM].copy_from_slice(e);
    }
    flat
}

fn phrase_train_step(w: &mut PhraseWeights, x: &[f32], label: f32, lr: f32, wd: f32) -> f32 {
    let in_dim = EMBED_HISTORY * EMBED_DIM;
    let mut h_pre = vec![0.0f32; w.hidden];
    gemv_bias(w.hidden, in_dim, &w.fc1_w, x, &w.fc1_b, &mut h_pre);
    let h: Vec<f32> = h_pre.iter().copied().map(relu).collect();
    let mut logit = [0.0f32];
    gemv_bias(1, w.hidden, &w.fc2_w, &h, &w.fc2_b, &mut logit);
    let prob = sigmoid(logit[0]);
    let loss = bce_loss(prob, label);
    let dlogit = bce_dlogit(prob, label);

    let mut dfc2_w = vec![0.0f32; w.hidden];
    for i in 0..w.hidden {
        dfc2_w[i] = dlogit * h[i];
    }
    let mut dh = vec![0.0f32; w.hidden];
    for i in 0..w.hidden {
        dh[i] = dlogit * w.fc2_w[i];
        if h_pre[i] <= 0.0 {
            dh[i] = 0.0;
        }
    }
    let mut dfc1_w = vec![0.0f32; w.hidden * in_dim];
    let mut dfc1_b = vec![0.0f32; w.hidden];
    for o in 0..w.hidden {
        dfc1_b[o] = dh[o];
        for i in 0..in_dim {
            dfc1_w[o * in_dim + i] = dh[o] * x[i];
        }
    }
    sgd_step(&mut w.fc2_w, &dfc2_w, lr, wd);
    sgd_step(&mut w.fc2_b, &[dlogit], lr, wd);
    sgd_step(&mut w.fc1_w, &dfc1_w, lr, wd);
    sgd_step(&mut w.fc1_b, &dfc1_b, lr, wd);
    loss
}

/// Train phrase head with frozen embedding (all RLX).
pub fn train_phrase_head(
    phrase: &mut PhraseWeights,
    embed_w: &EmbeddingWeights,
    clips: &[LabeledClip],
    sgd: &SgdConfig,
) -> Result<TrainReport> {
    let embed = EmbeddingNet::new(embed_w.clone());
    let mut feats = Vec::new();
    for c in clips {
        if let Some(hist) = collect_embed_window(&embed, &c.pcm) {
            feats.push((flatten(&hist), c.label));
        }
    }
    if feats.is_empty() {
        anyhow::bail!("no clips produced a full embedding window (need ~1.5s+ audio each)");
    }

    let mut initial = 0.0f32;
    let mut final_loss = 0.0f32;
    for epoch in 0..sgd.epochs {
        let mut sum = 0.0f32;
        for (x, y) in &feats {
            sum += phrase_train_step(phrase, x, *y, sgd.lr, sgd.weight_decay);
        }
        let mean = sum / feats.len() as f32;
        if epoch == 0 {
            initial = mean;
        }
        final_loss = mean;
        if sgd.log_every > 0 && epoch % sgd.log_every == 0 {
            eprintln!(
                "[rlx-openwakeword-train] epoch={epoch} loss={mean:.4} keyword={}",
                phrase.keyword
            );
        }
    }
    let mut correct = 0usize;
    for (x, y) in &feats {
        let mut h_pre = vec![0.0f32; phrase.hidden];
        gemv_bias(
            phrase.hidden,
            EMBED_HISTORY * EMBED_DIM,
            &phrase.fc1_w,
            x,
            &phrase.fc1_b,
            &mut h_pre,
        );
        let h: Vec<f32> = h_pre.iter().copied().map(relu).collect();
        let mut logit = [0.0f32];
        gemv_bias(
            1,
            phrase.hidden,
            &phrase.fc2_w,
            &h,
            &phrase.fc2_b,
            &mut logit,
        );
        let p = sigmoid(logit[0]);
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - *y).abs() < 0.5 {
            correct += 1;
        }
    }
    Ok(TrainReport {
        epochs: sgd.epochs,
        final_loss,
        initial_loss: initial,
        train_acc: correct as f32 / feats.len() as f32,
        keyword: phrase.keyword.clone(),
    })
}
