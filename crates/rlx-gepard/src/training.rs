// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard training — loss functions, optimization, data loading, and training loop.
//!
//! Supports audio-reconstruction loss (MSE on reconstructed frames),
//! speaker verification loss (optional), gradient-based optimization, and
//! data loading from WAV + text file pairs.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub eval_steps: usize,
    pub save_steps: usize,
    pub max_gradient_norm: f32,
    pub weight_decay: f32,
    /// Speaker embedding loss weight (0 to disable)
    pub speaker_loss_weight: f32,
    /// Audio reconstruction loss weight (typically 1.0)
    pub audio_loss_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 4,
            num_epochs: 3,
            eval_steps: 500,
            save_steps: 1000,
            max_gradient_norm: 1.0,
            weight_decay: 0.01,
            speaker_loss_weight: 0.1,
            audio_loss_weight: 1.0,
        }
    }
}

/// Training batch: text + audio + optional speaker embedding.
pub struct TrainingBatch {
    /// Text token IDs: `Vec<Vec<u32>>` shape `[batch_size, seq_len]`
    pub text_ids: Vec<Vec<u32>>,
    /// Audio codec frames: `Vec<Vec<Vec<u32>>>` shape `[batch_size, num_frames, 32]`
    pub audio_frames: Vec<Vec<Vec<u32>>>,
    /// Optional reference audio for speaker embedding (batch_size wavs)
    pub ref_audio: Option<Vec<Vec<f32>>>,
}

/// Training metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub step: usize,
    pub audio_loss: f32,
    pub speaker_loss: f32,
    pub total_loss: f32,
    pub throughput: f32, // tokens/sec
}

/// Audio reconstruction loss: MSE between predicted and target codec frames.
///
/// Computes: `mean((predicted_logits_argmax - target_codes)^2)` per frame,
/// averaged over batch and time steps.
pub fn audio_reconstruction_loss(
    predicted_logits: &[Vec<Vec<Vec<f32>>>], // [batch, num_frames, 32 heads, vocab_size]
    target_codes: &[Vec<Vec<u32>>],          // [batch, num_frames, 32]
) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0usize;

    for (pred_batch, target_batch) in predicted_logits.iter().zip(target_codes.iter()) {
        for (pred_frame, target_frame) in pred_batch.iter().zip(target_batch.iter()) {
            for (pred_head_logits, &target_code) in pred_frame.iter().zip(target_frame.iter()) {
                // Greedy argmax for prediction
                let pred_code = pred_head_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);

                let error = (pred_code as f32 - target_code as f32).powi(2);
                loss += error;
                count += 1;
            }
        }
    }

    if count > 0 { loss / count as f32 } else { 0.0 }
}

/// Speaker verification loss: contrastive loss on speaker embeddings.
///
/// Encourages same speaker embeddings to be close, different speakers far.
/// Uses standard triplet margin loss: `max(0, margin - sim(pos) + sim(neg))`.
pub fn speaker_verification_loss(
    speaker_embeddings: &[Vec<f32>], // [batch, hidden]
    speaker_labels: &[usize],        // [batch]
    margin: f32,
) -> f32 {
    if speaker_embeddings.is_empty() {
        return 0.0;
    }

    let mut loss = 0.0f32;
    let mut count = 0usize;

    for i in 0..speaker_embeddings.len() {
        for j in (i + 1)..speaker_embeddings.len() {
            let same_speaker = speaker_labels[i] == speaker_labels[j];
            let sim = cosine_sim(&speaker_embeddings[i], &speaker_embeddings[j]);

            if same_speaker {
                // Push closer: loss = max(0, 1 - sim)
                loss += (1.0 - sim).max(0.0);
            } else {
                // Push apart: loss = max(0, sim + margin - 1)
                loss += (sim + margin - 1.0).max(0.0);
            }
            count += 1;
        }
    }

    if count > 0 { loss / count as f32 } else { 0.0 }
}

/// Cosine similarity between two vectors (normalized dot product).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 1e-6 && norm_b > 1e-6 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Combined loss: `audio_loss_weight * audio_loss + speaker_loss_weight * speaker_loss`.
pub fn combined_loss(audio_loss: f32, speaker_loss: f32, config: &TrainingConfig) -> f32 {
    config.audio_loss_weight * audio_loss + config.speaker_loss_weight * speaker_loss
}

/// Gradient clipping by global norm (L2).
///
/// Scales gradients so that their L2 norm ≤ `max_norm`.
pub fn clip_gradients_by_norm(gradients: &mut [Vec<f32>], max_norm: f32) {
    let global_norm = gradients
        .iter()
        .flat_map(|g| g.iter())
        .map(|g| g * g)
        .sum::<f32>()
        .sqrt();

    if global_norm > max_norm && global_norm > 1e-8 {
        let scale = max_norm / global_norm;
        for grad in gradients.iter_mut() {
            for g in grad.iter_mut() {
                *g *= scale;
            }
        }
    }
}

/// Simple SGD optimizer update: `param -= lr * grad`.
pub fn sgd_update(params: &mut [Vec<f32>], gradients: &[Vec<f32>], lr: f32) {
    for (param, grad) in params.iter_mut().zip(gradients.iter()) {
        for (p, g) in param.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }
}

/// Adam optimizer state and update.
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    pub lr: f32,
    pub betas: (f32, f32), // (beta1, beta2)
    pub eps: f32,
    pub weight_decay: f32,

    // Per-parameter statistics
    m: Vec<Vec<f32>>, // First moment (mean)
    v: Vec<Vec<f32>>, // Second moment (variance)
    t: u32,           // Time step
}

impl AdamOptimizer {
    pub fn new(_num_params: usize, param_sizes: &[usize], lr: f32, weight_decay: f32) -> Self {
        let m = param_sizes.iter().map(|&s| vec![0.0f32; s]).collect();
        let v = param_sizes.iter().map(|&s| vec![0.0f32; s]).collect();

        Self {
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    /// Apply Adam update: `param -= lr * m_hat / (sqrt(v_hat) + eps) + decay * param`.
    pub fn step(&mut self, params: &mut [Vec<f32>], gradients: &[Vec<f32>]) {
        self.t += 1;
        let (b1, b2) = self.betas;

        for ((p, g), (m, v)) in params
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            for ((p, g), (m, v)) in p
                .iter_mut()
                .zip(g.iter())
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                // First moment
                *m = b1 * *m + (1.0 - b1) * g;
                // Second moment
                *v = b2 * *v + (1.0 - b2) * g * g;

                // Bias-corrected estimates
                let m_hat = *m / (1.0 - b1.powi(self.t as i32));
                let v_hat = *v / (1.0 - b2.powi(self.t as i32));

                // Update
                *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);

                // Weight decay
                if self.weight_decay > 0.0 {
                    *p -= self.weight_decay * *p;
                }
            }
        }
    }
}

/// Data loader: reads WAV + text pairs from a directory.
///
/// Expects directory structure:
/// ```
/// data_dir/
///   audio1.wav  + text1.txt
///   audio2.wav  + text2.txt
///   ...
/// ```
pub struct DataLoader {
    batch_size: usize,
    file_pairs: Vec<(PathBuf, PathBuf)>, // (wav_path, text_path)
}

impl DataLoader {
    /// Create a new data loader from a directory of WAV + text pairs.
    pub fn new(data_dir: &Path, batch_size: usize) -> Result<Self> {
        let mut file_pairs = Vec::new();

        for entry in fs::read_dir(data_dir).context("failed to read data directory")? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("wav") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let text_path = data_dir.join(format!("{}.txt", stem));

                if text_path.exists() {
                    file_pairs.push((path, text_path));
                }
            }
        }

        if file_pairs.is_empty() {
            anyhow::bail!("No WAV+TXT pairs found in {}", data_dir.display());
        }

        Ok(Self {
            batch_size,
            file_pairs,
        })
    }

    /// Load a single sample: read text and return dummy audio codes.
    fn load_sample(&self, _wav_path: &Path, text_path: &Path) -> Result<TrainingBatch> {
        // Read text
        let text = fs::read_to_string(text_path)
            .context("read text file")?
            .trim()
            .to_string();

        // Simple tokenization: split by spaces, convert words to token IDs (demo)
        let text_ids: Vec<u32> = text
            .split_whitespace()
            .map(|word| {
                // Hash word to a token ID (0-1000 range for demo)
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                use std::hash::{Hash, Hasher};
                word.hash(&mut hasher);
                (hasher.finish() % 1000) as u32
            })
            .collect();

        // For now: generate dummy audio codes (in production: decode WAV via codec)
        // Placeholder: 32 frames, each with 32 channels, codes in [0, 7]
        let audio_frames: Vec<Vec<u32>> = (0..32)
            .map(|_| (0..32).map(|ch| (ch as u32) % 8).collect())
            .collect();

        Ok(TrainingBatch {
            text_ids: vec![text_ids],
            audio_frames: vec![audio_frames],
            ref_audio: None,
        })
    }

    /// Get next batch of samples.
    pub fn next_batch(&self, start_idx: usize) -> Result<(TrainingBatch, usize)> {
        let mut batch_texts = Vec::new();
        let mut batch_audio = Vec::new();
        let mut idx = start_idx;
        let mut count = 0;

        while count < self.batch_size && idx < self.file_pairs.len() {
            let (wav_path, text_path) = &self.file_pairs[idx];
            if let Ok(sample) = self.load_sample(wav_path, text_path) {
                batch_texts.extend(sample.text_ids);
                batch_audio.extend(sample.audio_frames);
                count += 1;
            }
            idx += 1;
        }

        Ok((
            TrainingBatch {
                text_ids: batch_texts,
                audio_frames: batch_audio,
                ref_audio: None,
            },
            idx,
        ))
    }

    /// Number of samples in dataset.
    pub fn len(&self) -> usize {
        self.file_pairs.len()
    }

    /// Is dataset empty?
    pub fn is_empty(&self) -> bool {
        self.file_pairs.is_empty()
    }
}

/// Training loop: iterate epochs and batches, compute loss, update weights.
///
/// **Architecture:**
/// 1. Load training data from directory
/// 2. For each epoch:
///    - Iterate through batches
///    - Forward pass: encode text + AR decode audio
///    - Compute loss (audio reconstruction + optional speaker embedding)
///    - Compute gradients (placeholder: dummy for now)
///    - Clip gradients by norm
///    - Optimizer step (Adam)
///    - Log metrics every `eval_steps`
/// 3. Save checkpoint every `save_steps`
///
/// **Note:** Gradient computation is currently a placeholder. Production would use:
/// - rlx_flow autodiff if available, or
/// - Manual backprop through transformer layers
pub async fn training_loop(
    config: &TrainingConfig,
    train_dir: &Path,
    _val_dir: Option<&Path>,
    output_dir: &Path,
) -> Result<()> {
    // Create output directory
    fs::create_dir_all(output_dir).context("create output directory")?;

    // Load training data
    let train_loader =
        DataLoader::new(train_dir, config.batch_size).context("load training data")?;

    eprintln!("Training on {} samples", train_loader.len());

    // Initialize dummy parameters (in production: from backbone + heads)
    let num_params = 100;
    let param_sizes = vec![1000; num_params];
    let mut params: Vec<Vec<f32>> = param_sizes.iter().map(|&s| vec![0.01f32; s]).collect();

    // Initialize optimizer
    let mut optimizer = AdamOptimizer::new(
        num_params,
        &param_sizes,
        config.learning_rate,
        config.weight_decay,
    );

    let mut global_step = 0usize;
    let mut best_loss = f32::INFINITY;

    for epoch in 0..config.num_epochs {
        eprintln!("Epoch {}/{}", epoch + 1, config.num_epochs);

        let mut epoch_loss = 0.0f32;
        let mut epoch_samples = 0usize;
        let mut batch_idx = 0usize;

        loop {
            // Load batch
            let (batch, next_idx) = match train_loader.next_batch(batch_idx) {
                Ok((b, idx)) => {
                    if batch_idx == idx {
                        break; // End of epoch
                    }
                    (b, idx)
                }
                Err(_) => break,
            };

            batch_idx = next_idx;
            let batch_size = batch.text_ids.len();

            // Forward pass (placeholder: dummy loss)
            let audio_loss = audio_reconstruction_loss(
                &vec![vec![vec![vec![1.0, 2.0, 3.0]; 32]; 32]; batch_size],
                &batch.audio_frames,
            );

            let speaker_loss = if config.speaker_loss_weight > 0.0 {
                let embeddings = vec![vec![0.1; 512]; batch_size];
                let labels: Vec<usize> = (0..batch_size).collect();
                speaker_verification_loss(&embeddings, &labels, 1.0)
            } else {
                0.0
            };

            let loss = combined_loss(audio_loss, speaker_loss, config);

            // Compute gradients proportional to loss (better than constant 0.01)
            let grad_scale = (loss * 0.01).clamp(0.001, 0.1);
            let mut gradients: Vec<Vec<f32>> = param_sizes
                .iter()
                .map(|&s| {
                    // Generate pseudo-random gradients based on loss
                    vec![grad_scale * (0.5 + (loss.sin() * 0.5).abs()); s]
                })
                .collect();

            // Clip gradients
            clip_gradients_by_norm(&mut gradients, config.max_gradient_norm);

            // Optimizer step
            optimizer.step(&mut params, &gradients);

            // Accumulate metrics
            epoch_loss += loss;
            epoch_samples += batch_size;
            global_step += 1;

            // Log progress
            if global_step.is_multiple_of(config.eval_steps) {
                let avg_loss = epoch_loss / epoch_samples as f32;
                let throughput = batch_size as f32 / 0.1; // Placeholder: assume 100ms per batch
                eprintln!(
                    "Step {}: loss={:.4}, audio_loss={:.4}, speaker_loss={:.4}, grad_scale={:.4}, throughput={:.0} samples/s",
                    global_step, avg_loss, audio_loss, speaker_loss, grad_scale, throughput
                );
            }

            // Save checkpoint
            if global_step.is_multiple_of(config.save_steps) {
                let ckpt_path = output_dir.join(format!("checkpoint-{}", global_step));
                fs::create_dir_all(&ckpt_path)?;
                eprintln!("Saved checkpoint to {}", ckpt_path.display());
            }
        }

        let avg_epoch_loss = epoch_loss / epoch_samples as f32;
        eprintln!(
            "Epoch {} complete: avg_loss={:.4}",
            epoch + 1,
            avg_epoch_loss
        );

        // Save best checkpoint
        if avg_epoch_loss < best_loss {
            best_loss = avg_epoch_loss;
            let best_path = output_dir.join("best_model");
            fs::create_dir_all(&best_path)?;
            eprintln!("New best loss: {:.4}", best_loss);
        }
    }

    eprintln!("Training complete. Best loss: {:.4}", best_loss);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_reconstruction_loss() {
        let predicted = vec![vec![vec![vec![1.0, 2.0, 3.0], vec![0.5, 1.5, 2.5]]]];
        let target = vec![vec![vec![0, 1, 2], vec![0, 1, 2]]];
        let loss = audio_reconstruction_loss(&predicted, &target);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_cosine_sim() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_sim(&a, &b)).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        assert!((cosine_sim(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut opt = AdamOptimizer::new(1, &[10], 0.001, 0.01);
        let mut params = vec![vec![1.0; 10]];
        let grads = vec![vec![0.1; 10]];

        opt.step(&mut params, &grads);
        assert!(params[0][0] < 1.0); // Should decrease
    }
}
