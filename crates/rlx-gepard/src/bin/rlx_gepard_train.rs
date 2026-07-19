// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard TTS training binary.

use anyhow::Result;
use clap::Parser;
use rlx_gepard::training::{TrainingConfig, training_loop};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "rlx-gepard-train",
    author,
    version,
    about = "Train Gepard TTS on custom data"
)]
struct Args {
    /// Path to training data directory (containing .wav and .txt pairs)
    #[arg(long, required = true)]
    train_dir: PathBuf,

    /// Path to validation data directory
    #[arg(long)]
    val_dir: Option<PathBuf>,

    /// Path to pretrained weights directory
    #[arg(long, required = true)]
    weights: PathBuf,

    /// Output directory for checkpoints
    #[arg(long, default_value = "./checkpoints")]
    output_dir: PathBuf,

    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    learning_rate: f32,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Number of epochs
    #[arg(long, default_value = "3")]
    num_epochs: usize,

    /// Save checkpoint every N steps
    #[arg(long, default_value = "1000")]
    save_steps: usize,

    /// Evaluate every N steps
    #[arg(long, default_value = "500")]
    eval_steps: usize,

    /// Maximum gradient norm (for clipping)
    #[arg(long, default_value = "1.0")]
    max_gradient_norm: f32,

    /// Weight decay (L2 regularization)
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Speaker loss weight (0 to disable)
    #[arg(long, default_value = "0.1")]
    speaker_loss_weight: f32,

    /// Device: cpu, metal, mlx, cuda, rocm, coreml, wgpu, vulkan
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Number of workers for data loading
    #[arg(long, default_value = "4")]
    num_workers: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Gepard TTS Training ===");
    println!("Train dir:    {:?}", args.train_dir);
    println!("Val dir:      {:?}", args.val_dir);
    println!("Weights:      {:?}", args.weights);
    println!("Output:       {:?}", args.output_dir);
    println!("Device:       {}", args.device);
    println!("Batch size:   {}", args.batch_size);
    println!("Epochs:       {}", args.num_epochs);
    println!("LR:           {}", args.learning_rate);
    println!("Workers:      {}", args.num_workers);
    println!();

    // Verify input directories exist
    if !args.train_dir.exists() {
        anyhow::bail!("Training directory not found: {:?}", args.train_dir);
    }

    if let Some(ref val_dir) = args.val_dir {
        if !val_dir.exists() {
            anyhow::bail!("Validation directory not found: {:?}", val_dir);
        }
    }

    // Create training config
    let config = TrainingConfig {
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        num_epochs: args.num_epochs,
        eval_steps: args.eval_steps,
        save_steps: args.save_steps,
        max_gradient_norm: args.max_gradient_norm,
        weight_decay: args.weight_decay,
        speaker_loss_weight: args.speaker_loss_weight,
        audio_loss_weight: 1.0,
    };

    // Run training loop
    println!("Starting training loop...");
    training_loop(
        &config,
        &args.train_dir,
        args.val_dir.as_deref(),
        &args.output_dir,
    )
    .await?;

    println!();
    println!("✓ Training complete!");
    println!("Checkpoints saved to: {}", args.output_dir.display());

    Ok(())
}
