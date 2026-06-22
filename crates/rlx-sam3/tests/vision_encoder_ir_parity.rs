// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Released under GPL-3.0; see crate-level header for the full notice.

//! Numerical parity: SAM3 ViT-L vision encoder, CPU reference vs HIR graph
//! executed on `Device::Cpu`. Uses a downsized config so the test stays fast
//! while covering both window blocks and global blocks plus the QKV split,
//! 2D RoPE, FFN, and residuals.

use rlx_runtime::Device;
use rlx_sam3::config::Sam3VitConfig;
use rlx_sam3::preprocess::Sam3PreprocessWeights;
use rlx_sam3::vision_encoder::{
    Sam3VisionEncoderWeights, Sam3VitBlockWeights, forward_blocks_native,
};
use rlx_sam3::vision_encoder_ir::Sam3CompiledVisionEncoder;

/// `grid * grid` (the QKV linear's m dim) is intentionally capped at 8 so the
/// rlx-cpu CPU backend's small-m NEON path (limited to m ≤ 8) is safe. The
/// window-partition path is exercised via a separate test that lives within
/// the same m budget.
fn tiny_cfg_global() -> Sam3VitConfig {
    Sam3VitConfig {
        img_size: 4,
        pretrain_img_size: 4,
        patch_size: 2,
        embed_dim: 16,
        depth: 2,
        num_heads: 4,
        mlp_ratio: 2.0,
        qkv_bias: true,
        bias_patch_embed: false,
        use_abs_pos: false,
        tile_abs_pos: false,
        use_rope: true,
        use_interp_rope: true,
        window_size: 2,
        global_att_blocks: vec![0, 1],
        layer_norm_eps: 1e-6,
    }
}

/// Like `tiny_cfg_global` but with `head_dim = 8` (`quarter = 2`) so the RoPE
/// pair-wise vs half-split conventions actually differ — at `head_dim = 4`,
/// `rot_half = 1` and both layouts coincide so they can't catch the
/// rotation-pairing bug that fires on real ViT-L (`head_dim = 64`).
fn tiny_cfg_global_dh8() -> Sam3VitConfig {
    Sam3VitConfig {
        img_size: 4,
        pretrain_img_size: 4,
        patch_size: 2,
        embed_dim: 32,
        depth: 2,
        num_heads: 4,
        mlp_ratio: 2.0,
        qkv_bias: true,
        bias_patch_embed: false,
        use_abs_pos: false,
        tile_abs_pos: false,
        use_rope: true,
        use_interp_rope: true,
        window_size: 2,
        global_att_blocks: vec![0, 1],
        layer_norm_eps: 1e-6,
    }
}

/// Same grid (2×2 = 4 ≤ 8) but blocks run window attention with `ws=1` so the
/// reshape→transpose→reshape partition path is exercised.
fn tiny_cfg_window() -> Sam3VitConfig {
    Sam3VitConfig {
        img_size: 4,
        pretrain_img_size: 4,
        patch_size: 2,
        embed_dim: 16,
        depth: 2,
        num_heads: 4,
        mlp_ratio: 2.0,
        qkv_bias: true,
        bias_patch_embed: false,
        use_abs_pos: false,
        tile_abs_pos: false,
        use_rope: true,
        use_interp_rope: true,
        window_size: 1,
        global_att_blocks: vec![],
        layer_norm_eps: 1e-6,
    }
}

/// Tiny deterministic generator — Lehmer LCG, identical across calls so the
/// same `seed` produces the same sequence. Good enough for parity tests.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(48271) ^ (self.0 >> 7);
        self.0
    }
    fn next_f(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.4
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f()).collect()
    }
}

fn synth_weights(cfg: &Sam3VitConfig, rng: &mut Rng) -> Sam3VisionEncoderWeights {
    let e = cfg.embed_dim;
    let hidden = (e as f64 * cfg.mlp_ratio) as usize;
    let pre = Sam3PreprocessWeights {
        patch_proj_w: vec![],
        patch_proj_b: vec![],
        pos_embed: None,
        embed_dim: e,
        patch_size: cfg.patch_size,
        grid: cfg.patch_grid(),
    };
    let ones_w = || {
        (0..e)
            .map(|i| 1.0 + 0.05 * (i as f32 - e as f32 / 2.0))
            .collect::<Vec<_>>()
    };
    let blocks = (0..cfg.depth)
        .map(|_| Sam3VitBlockWeights {
            norm1_w: ones_w(),
            norm1_b: rng.vec(e),
            qkv_w_t: rng.vec(e * 3 * e),
            qkv_b: rng.vec(3 * e),
            qkv_gguf_prefix: None,
            proj_w_t: rng.vec(e * e),
            proj_b: rng.vec(e),
            proj_gguf_prefix: None,
            norm2_w: ones_w(),
            norm2_b: rng.vec(e),
            mlp_fc1_w_t: rng.vec(e * hidden),
            mlp_fc1_b: rng.vec(hidden),
            mlp_fc1_gguf_prefix: None,
            mlp_fc2_w_t: rng.vec(hidden * e),
            mlp_fc2_b: rng.vec(e),
            mlp_fc2_gguf_prefix: None,
        })
        .collect();
    Sam3VisionEncoderWeights {
        pre,
        ln_pre_w: ones_w(),
        ln_pre_b: rng.vec(e),
        blocks,
    }
}

fn assert_parity(cfg: &Sam3VitConfig, tag: &str, seed: u32, tol: f32) {
    let mut rng = Rng::new(seed);
    let w = synth_weights(cfg, &mut rng);
    let grid = cfg.patch_grid();
    let e = cfg.embed_dim;
    let tokens_in: Vec<f32> = (0..grid * grid * e)
        .map(|i| 0.1 + (i as f32 * 0.0137).sin())
        .collect();

    let cpu_out = forward_blocks_native(&w, None, cfg, &tokens_in).unwrap();
    let mut compiled = Sam3CompiledVisionEncoder::new(&w, cfg, 1, Device::Cpu).unwrap();
    let ir_out = compiled.run_tokens(&tokens_in).unwrap();
    assert_eq!(cpu_out.len(), ir_out.len());

    let max_abs = cpu_out
        .iter()
        .zip(ir_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mean_abs = cpu_out
        .iter()
        .zip(ir_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / cpu_out.len() as f32;
    let dot: f32 = cpu_out.iter().zip(ir_out.iter()).map(|(a, b)| a * b).sum();
    let na: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = ir_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    eprintln!("[{tag}] max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} cos={cos:.6}");
    assert!(
        cos > 0.9999,
        "[{tag}] cos={cos} too low (max_abs={max_abs}, mean_abs={mean_abs})"
    );
    assert!(
        max_abs < tol,
        "[{tag}] max_abs={max_abs} > tol={tol} (mean_abs={mean_abs}, cos={cos})"
    );
}

#[test]
fn sam3_vision_ir_matches_cpu_global_blocks() {
    // Global attention diverges slightly from the CPU per-head sgemm by FP
    // reduction order — cosine stays at ~1.0 but max-abs sits near 1e-3.
    assert_parity(&tiny_cfg_global(), "global", 0xC0FFEE, 5e-3);
}

#[test]
fn sam3_vision_ir_matches_cpu_window_blocks() {
    assert_parity(&tiny_cfg_window(), "window", 0xBEEF, 1e-4);
}

#[test]
fn sam3_vision_ir_matches_cpu_global_blocks_head_dim_8() {
    // Catches RoPE pair-wise vs half-split divergence (silent at head_dim=4).
    assert_parity(&tiny_cfg_global_dh8(), "global_dh8", 0xCAFE, 5e-3);
}
