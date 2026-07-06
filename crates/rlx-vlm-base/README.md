# rlx-vlm-base

Shared **vision-language / omni base types** for RLX multimodal runners. It defines the small set of traits and structs that every VLM/Omni family crate reuses — a per-image preprocessor, a vision-tower trait, an MLP projector trait, an audio-encoder trait, and a turn-ordered multimodal prompt — so the family crates stay thin.

This crate has **no dependencies beyond `anyhow`**: no weights, no graphs, no tokenizer, no CLI. It is a pure contract layer. Implementations live in the consumer crates below.

## How it fits

| Consumer | Implements |
|---|---|
| [`rlx-qwen3-vl`](../rlx-qwen3-vl) | `ImagePreprocessor`, `VisionTower`, `Projector` for Qwen3-VL |
| [`rlx-lfm-vl`](../rlx-lfm-vl) | the same three traits for LFM2.5-VL |
| [`rlx-nemotron-omni`](../rlx-nemotron-omni) | vision traits + `AudioEncoder` (over `rlx-whisper`) |

## Public API

```rust
use rlx_vlm_base::{
    Modality, ImagePatches, ImagePreprocessor, VisionTower, Projector, AudioEncoder,
    MultimodalPrompt, PromptChunk,
};

// A multimodal turn is an ordered list of (modality, payload) chunks.
let mut prompt = MultimodalPrompt::default();
prompt.push(PromptChunk::Text(vec![1, 2, 3]));          // caller-tokenized text ids
prompt.push(PromptChunk::Image(/* ImagePatches */ patches));
prompt.push(PromptChunk::Audio { samples, sample_rate: 16_000 });

assert!(!prompt.is_text_only());
assert_eq!(prompt.num_chunks(), 3);
```

The traits a family crate implements:

```rust
pub trait ImagePreprocessor: Send {
    fn preprocess_path(&self, path: &std::path::Path) -> anyhow::Result<ImagePatches>;
    fn preprocess_bytes(&self, bytes: &[u8]) -> anyhow::Result<ImagePatches>;
}

pub trait VisionTower: Send {              // patches -> [num_patches, hidden]
    fn embed(&mut self, patches: &ImagePatches) -> anyhow::Result<Vec<f32>>;
    fn hidden_size(&self) -> usize;
}

pub trait Projector: Send {                // vision hidden -> LM embedding space
    fn project(&mut self, vision_embed: &[f32], num_patches: usize) -> anyhow::Result<Vec<f32>>;
    fn output_dim(&self) -> usize;
}

pub trait AudioEncoder: Send {             // PCM -> hidden embeddings (omni)
    fn embed_audio(&mut self, samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<f32>>;
    fn hidden_size(&self) -> usize;
}
```

`ImagePatches` carries the resized/patchified image (`patches`, `grid_h`, `grid_w`, `patch_h`, `patch_w`, `channels`) with `num_patches()` = `grid_h * grid_w` and `patch_dim()` = `channels * patch_h * patch_w`; the exact layout is family-specific. `Modality` (`Text` / `Image` / `Audio`) tags each chunk so the runner knows when to invoke an encoder instead of consuming raw token ids.

## Status

TYPE SKELETON (PLAN.md M7). The traits and structs are stable; concrete implementations land in the family crates as M7 progresses.
