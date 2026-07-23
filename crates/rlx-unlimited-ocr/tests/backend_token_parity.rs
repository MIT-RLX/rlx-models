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

//! Exact greedy token-ID parity: Metal/wgpu (and CUDA if present) vs CPU
//! compiled MoE LM path on a fixed short prompt + image.
//!
//! Packs LM weights once and shares them across devices. Env-gated — full
//! MoE compile+run needs tens of GB of RAM:
//!
//! ```bash
//! RLX_UNLIMITED_OCR_TOKEN_PARITY=1 cargo test -p rlx-unlimited-ocr \
//!   --test backend_token_parity --features apple-silicon --release -- --test-threads 1
//! ```

use rlx_runtime::Device;
use rlx_unlimited_ocr::{
    CompiledLm, DeepEncoder, IMAGE_TOKEN_ID, ImageMode, LmWeightPrecision, PackedLmWeights,
    Projector, SampleOpts, UnlimitedOcrConfig, UnlimitedOcrWeightStore, fuse_inputs_embeds,
    preprocess_path, require_model_dir, require_probe_image, sample_token,
};
use std::sync::Arc;

fn parity_enabled() -> bool {
    matches!(
        std::env::var("RLX_UNLIMITED_OCR_TOKEN_PARITY").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

struct SharedFixture {
    packed: Arc<PackedLmWeights>,
    prompt_ids: Vec<u32>,
    inputs_embeds: Vec<f32>,
    opts: SampleOpts,
}

fn load_fixture() -> Option<SharedFixture> {
    let model_dir = require_model_dir()?;
    let image = require_probe_image()?;
    let cfg = UnlimitedOcrConfig::from_model_dir(&model_dir).ok()?;
    cfg.validate().ok()?;
    let store = UnlimitedOcrWeightStore::open(&model_dir).ok()?;

    let mut encoder = DeepEncoder::from_config(&cfg);
    let mut projector = Projector::from_config(&cfg.projector);
    encoder.load(&store).ok()?;
    projector.load(&store).ok()?;
    let packed = Arc::new(
        PackedLmWeights::from_store_with_precision(&store, &cfg, LmWeightPrecision::F32).ok()?,
    );

    let prep = preprocess_path(&image, ImageMode::Base { size: 1024 }).ok()?;
    let prompt_ids = rlx_unlimited_ocr::build_prompt_ids(
        &model_dir,
        "<image>document parsing.",
        &[prep.clone()],
        cfg.bos_token_id,
        IMAGE_TOKEN_ID,
    )
    .ok()?;

    let vision = encoder.encode_and_project(&[prep], &projector).ok()?;
    let mut inputs_embeds = packed.embed_tokens_lookup(&prompt_ids).ok()?;
    fuse_inputs_embeds(
        &prompt_ids,
        &mut inputs_embeds,
        cfg.hidden_size,
        IMAGE_TOKEN_ID,
        &vision,
    )
    .ok()?;

    Some(SharedFixture {
        packed,
        prompt_ids,
        inputs_embeds,
        opts: SampleOpts {
            max_new_tokens: 4,
            ..SampleOpts::default()
        },
    })
}

fn generate_on_device(fx: &SharedFixture, device: Device) -> Vec<u32> {
    let mut lm = CompiledLm::new(device, Arc::clone(&fx.packed));
    let mut token_ids = fx.prompt_ids.clone();
    let prompt_len = token_ids.len();
    let (mut logits, mut kv) = lm.prefill(&fx.inputs_embeds, prompt_len).expect("prefill");
    for _ in 0..fx.opts.max_new_tokens {
        let next = sample_token(&logits, &fx.opts, &token_ids);
        token_ids.push(next);
        if next == fx.packed.config.eos_token_id {
            break;
        }
        let step_embed = lm.embed_tokens(&[next]).expect("embed");
        let pos = token_ids.len() - 1;
        logits = lm.decode_step(&step_embed, pos, &mut kv).expect("decode");
    }
    token_ids[prompt_len..].to_vec()
}

fn parity_vs_cpu(device: Device) {
    if !parity_enabled() {
        eprintln!("skip token parity {device:?}: set RLX_UNLIMITED_OCR_TOKEN_PARITY=1");
        return;
    }
    if !rlx_runtime::is_available(device) {
        eprintln!("skip token parity {device:?}: backend not available");
        return;
    }
    let Some(fixture) = load_fixture() else {
        eprintln!("skip token parity: model dir / probe image missing");
        return;
    };
    let cpu = generate_on_device(&fixture, Device::Cpu);
    // Drop CPU session graphs before compiling on the GPU device.
    let other = generate_on_device(&fixture, device);
    assert_eq!(
        other, cpu,
        "token mismatch {device:?} vs CPU\n  cpu={cpu:?}\n  {device:?}={other:?}"
    );
}

#[test]
fn token_parity_cpu_self() {
    if !parity_enabled() {
        eprintln!("skip token_parity_cpu_self: set RLX_UNLIMITED_OCR_TOKEN_PARITY=1");
        return;
    }
    let Some(fixture) = load_fixture() else {
        eprintln!("skip token_parity_cpu_self: model dir / probe image missing");
        return;
    };
    let a = generate_on_device(&fixture, Device::Cpu);
    let b = generate_on_device(&fixture, Device::Cpu);
    assert_eq!(a, b, "CPU compiled path not deterministic");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn token_parity_metal_vs_cpu() {
    parity_vs_cpu(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn token_parity_mlx_vs_cpu() {
    if !parity_enabled() {
        eprintln!("skip token parity Mlx: set RLX_UNLIMITED_OCR_TOKEN_PARITY=1");
        return;
    }
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("skip token parity Mlx: backend not available");
        return;
    }
    eprintln!(
        "skip token parity Mlx: GroupedMatMul runtime shape unsupported on MLX for this MoE graph"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn token_parity_wgpu_vs_cpu() {
    parity_vs_cpu(Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn token_parity_cuda_vs_cpu() {
    parity_vs_cpu(Device::Cuda);
}
