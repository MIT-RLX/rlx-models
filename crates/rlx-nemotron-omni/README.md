# rlx-nemotron-omni

NVIDIA **Nemotron-3 Nano Omni** runner for RLX — text + vision + audio. The target is a 30B A3B-routed MoE that accepts all three modalities. This crate wires the multimodal encoders: a SigLIP-variant **vision tower** + LLaVA-style projector, and an **audio encoder** adapter over [`rlx-whisper`](../rlx-whisper)'s mel encoder.

## Status

PLAN.md M7 (per modality):

| Modality | Status |
|----------|--------|
| Vision (`NemotronOmniVisionRunner`) | implemented — SigLIP-variant ViT + LLaVA projector (`vision_tower.…` / `mm_projector.…`) |
| Audio (`NemotronOmniAudioEncoder`) | implemented — adapter over `rlx_whisper::WhisperRunner`, implements `rlx_vlm_base::AudioEncoder` |
| LM text path | deferred — Nemotron-H hybrid Mamba2+attention; final wiring waits on the `rlx-nemotron` runner + `rlx-ssm` `Mamba2Block` wrapper |

The binary validates the GGUF `general.architecture` (one of `nemotron-omni` / `nemotron_omni` / `nemotron3-omni` / `nemotron_h_omni`) and points you at the library API. The LM text path lives in [`rlx-nemotron`](../rlx-nemotron).

## Public API

```rust
use rlx_nemotron_omni::{NemotronOmniVisionRunner, NemotronOmniAudioEncoder};
use rlx_vlm_base::AudioEncoder;
use rlx_runtime::Device;
use std::path::Path;

// Vision: image -> [num_patches, lm_hidden]
let mut vision = NemotronOmniVisionRunner::builder()
    .mmproj("mmproj-nemotron-omni.gguf")
    .hf_config("config.json")
    .device(Device::Cpu)
    .build()?;
let img_embeds: Vec<f32> = vision.embed_image_path(Path::new("photo.jpg"))?;

// Audio: PCM f32 -> hidden embeddings (via a whisper mel encoder)
let mut audio = NemotronOmniAudioEncoder::from_weights_path("whisper.safetensors")?;
let aud_embeds: Vec<f32> = audio.embed_audio(&pcm_f32, 16_000)?;
# anyhow::Ok(())
```

`NemotronOmniVisionRunner` implements the [`rlx-vlm-base`](../rlx-vlm-base) `VisionTower` + `Projector` traits (also `embed_image_bytes` / `embed_patches`); `NemotronOmniAudioEncoder` implements `AudioEncoder` (wrap an existing runner with `::new(WhisperRunner)`). `AudioEncoderBox` is an owned trait-object wrapper; `SyntheticAudioEncoder` is a deterministic test stand-in (not a real encoder). `build_nemotron_omni_vision` returns a `NemotronOmniVisionBuilt` for custom compilation.

## How it fits

- Shared multimodal traits: [`rlx-vlm-base`](../rlx-vlm-base).
- Audio encoder: [`rlx-whisper`](../rlx-whisper).
- LM text path (hybrid Mamba2/attention): [`rlx-nemotron`](../rlx-nemotron).
- Sibling VLM runners: [`rlx-qwen3-vl`](../rlx-qwen3-vl), [`rlx-lfm-vl`](../rlx-lfm-vl), [`rlx-qwen25-vl`](../rlx-qwen25-vl).
