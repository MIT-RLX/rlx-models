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

//! Open Genmoji ([EvanZhouDev/open-genmoji](https://github.com/EvanZhouDev/open-genmoji))
//! on RLX: Genmoji-style emoji from a short description.
//!
//! The pipeline, as its author reverse-engineered the vendor's:
//! 1. **Prompt assistant** ([`assistant`]): a local rlx LM (Qwen3.5-0.8B by
//!    default, any registered family by choice) rewrites `flying pig` into a
//!    detailed emoji prompt under the Open Genmoji metaprompt — the step that
//!    makes the LoRA produce emoji rather than illustrations.
//! 2. **FLUX.1 `[dev]` + emoji LoRA** ([`rlx_flux1`]): trained on macOS emoji at
//!    160×160 on white.
//! 3. **Sticker** ([`postprocess`], [`Sticker`]): ×4 super-resolution
//!    (Real-ESRGAN via `rlx-upscale`), known-white-background matting at that
//!    resolution, resample to 800×800 for use as an iOS sticker.

pub mod assistant;
pub mod cli;
pub mod postprocess;

pub use assistant::{
    HttpAssistant, LocalAssistant, Metaprompt, PromptAssistant, clean_reply, open_lm,
    register_builtin_assistants, rewrite,
};
pub use cli::{Matter, Sticker, SubjectSegmenter, resolve_matter};
pub use postprocess::{
    Cutout, apply_alpha_on_white, background_pockets, remove_white_background,
    remove_white_background_guided, remove_white_background_hinted, remove_white_background_with,
    resize_rgba, subject_box, upscale, white_prior,
};
