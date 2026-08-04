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

//! # rlx-audio-blocks
//!
//! Shared, reusable building blocks for RLX audio models — the RLX analogue of
//! [`audio.cpp`](https://github.com/0xShug0/audio.cpp)'s `framework/` layer. The
//! goal is "improve once, benefit many families": a new audio model port becomes
//! mostly wiring of these components plus a thin model-specific glue crate.
//!
//! Many building blocks already live in dedicated RLX crates and are the canonical
//! home for their weights and graphs (e.g. BigVGAN in `rlx-neutts`/`rlx-facodec`,
//! Vocos + ISTFT in `rlx-wavtokenizer`, CAM++ in `rlx-funasr`, native T5 in
//! `rlx-parlertts`, Conformer in `rlx-wav2vec2-bert`). This crate is deliberately
//! **checkpoint-free**: it collects the pure, model-agnostic *algorithms* (decode
//! loops, samplers, DSP math) that those crates and new ports both need, and — as
//! the campaign proceeds — re-exports the canonical graph modules behind a single
//! import surface.
//!
//! ## Modules
//!
//! - [`decoders`] — sequence decoders that are pure host algorithms, independent
//!   of any single model's weights. Currently: the Token-and-Duration Transducer
//!   (TDT) greedy decoder used by Parakeet-TDT and TDT-variant Nemotron models.
//! - [`sampling`] — a seedable RNG plus noise schedules and denoise steppers
//!   (DDPM betas/`alphas_cumprod`, flow-matching Euler, SD3 shift) and
//!   classifier-free guidance, shared by the diffusion / flow-matching audio
//!   generators (Stable-Audio, Seed-VC, ACE-Step, VibeVoice).
//! - [`codec`] — codec-token utilities (the RVQ delay pattern used by
//!   MusicGen/Parler/Higgs-style autoregressive audio LMs).
//!
//! Roadmap (added as the port campaign reaches each family): `vocoders`
//! (BigVGAN/HiFT/Vocos re-exports), `speech_encoders` (WavLM/HuBERT/CAM++),
//! `text_encoders` (T5), and streaming KV/conv primitives.

pub mod codec;
pub mod decoders;
pub mod sampling;
