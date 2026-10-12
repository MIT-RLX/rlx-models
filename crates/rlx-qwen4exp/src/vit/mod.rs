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

//! The vision tower — `Qwen4ExpVisionModel`, a Qwen3-VL ViT.
//!
//! ```text
//!   patches = conv(image, patch_embd) + conv(image, patch_embd.1)   stride = patch
//!           → reorder into merge blocks → + patch_bias → + pos_embd(interpolated)
//!   27 ×   ( LN1 → QKV+b → axial-2D RoPE(q,k) → SDPA → out+b → +res
//!            LN2 → fc1+b → GELU_tanh → fc2+b                 → +res )
//!   merged  = LN(h, v.post_ln) → reshape [n_out, n·merge²]
//!   out     = mm.2(GELU_erf(mm.0(merged)))                    [1, n_out, 2560]
//! ```
//!
//! Six details carry the architecture, and each has a plausible wrong answer:
//!
//! * **The patch embedding is two convolutions summed.** Qwen3-VL's patch
//!   projection spans a 2-frame temporal window; a still image is replicated
//!   across both frames, which the converter stores as `v.patch_embd.weight`
//!   and `v.patch_embd.weight.1`. Using only the first halves the patch signal.
//! * **Tokens are reordered into spatial-merge blocks before the transformer**,
//!   not after. The merger's `[n_out, n·merge²]` reshape is only a reshape
//!   because each group of `merge²` patches is already contiguous — so the
//!   position embedding has to be permuted the same way, and the RoPE positions
//!   generated in that order.
//! * **The position embedding is a learned square table, bilinearly resized**
//!   to the actual patch grid (`num_position_embeddings = 2304 = 48²`), exactly
//!   as HF's `get_vision_interpolation_indices_and_weights` does with
//!   `mode = bilinear`. Nearest-sampling it is visually subtle and numerically
//!   wrong everywhere.
//! * **The RoPE is axial and covers the whole head.** `inv_freq` has
//!   `head_dim / 4` entries; the first half of the head gets
//!   `[row freqs | column freqs]` and the second half duplicates it, so
//!   `rotate_half` at `head_dim / 2` rotates row against row and column against
//!   column. See [`vision_rope_tables`].
//! * **The two GELUs are different functions.** The block FFN is
//!   `hidden_act = gelu_pytorch_tanh` (the tanh approximation); the merger is
//!   `nn.GELU()` with PyTorch's default `approximate='none'` — the exact erf
//!   form. The two agree to ~3e-4 absolute, so no end-to-end comparison
//!   resolves the difference: this one rests on reading the reference.
//! * **`v.post_ln` is the *merger* norm, not a post-block norm.** llama.cpp's
//!   qwen3vl mmproj has no separate pre- or post-block LayerNorm: the single
//!   `v.post_ln` is `Qwen3VLVisionPatchMerger.norm`, applied once before the
//!   spatial reshape. Applying it twice re-normalizes and strips the learned
//!   affine.
//!
//! ## Why this is a port and not a call into `rlx-qwen35`
//!
//! The architecture is shared (llama.cpp's converter says so:
//! `class Qwen4ExpVisionModel(Qwen3VLVisionModel)`), but the *implementation*
//! here is this crate's: the tower is a [`ModelFlow`](rlx_flow::ModelFlow) over a
//! [`WeightSource`](rlx_flow::WeightSource), so every projection goes through [`crate::common::linear`]
//! and takes the **packed** path on a quantized mmproj — `mmproj-Q8_0.gguf`
//! never materializes an f32 weight. It also means no host-side transposes at
//! all: the GGUF loader already reports `[out, in]`, and `v.patch_embd.weight`
//! already arrives as `Op::Conv`'s `[out_c, in_c, kh, kw]`.

mod config;
mod flow;
mod preprocess;

pub use config::VisionConfig;
pub use flow::{
    build_qwen4exp_vision_flow, build_qwen4exp_vision_flow_with_source, emit_vision_tower,
    vision_tensor_names, vision_tensor_shapes,
};
pub use preprocess::{
    preprocess_rgb, resize_rgb_nearest, rgb_to_nchw_f32, smart_resize, spatial_merge_gather_idx,
    vision_position_hw, vision_rope_tables,
};

/// Graph-input name for the normalized NCHW image.
pub const IMAGE_INPUT: &str = "image";
/// Graph-input name for the axial-RoPE cosine table.
pub const VISION_ROPE_COS_INPUT: &str = "vision_rope_cos";
/// Graph-input name for the axial-RoPE sine table.
pub const VISION_ROPE_SIN_INPUT: &str = "vision_rope_sin";

#[cfg(test)]
mod tests;
