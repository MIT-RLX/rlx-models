// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Fixed streaming Conformer ASR dimensions (80-mel → 6× subsample → CTC/AED).

pub const MEL_BINS: usize = 80;
pub const SUBSAMPLE: usize = 6;
pub const SAMPLE_RATES: [u32; 2] = [8000, 16000];

pub const VOCAB: usize = 6081;
pub const BLANK: u32 = 0;
pub const SOS: u32 = 2;
pub const EOS: u32 = 3;
pub const PRINTABLE_LO: u32 = 4;
pub const PRINTABLE_HI: u32 = 6080;

pub const DECODER_DIM: usize = 512;
pub const DECODER_LAYERS: usize = 3;
pub const DECODER_HEADS: usize = 8;
pub const DECODER_HEAD_DIM: usize = 64;
pub const DECODER_FFN: usize = 2048;
pub const BEAM: usize = 5;
pub const AED_WINDOW_FRAMES: usize = 512;
pub const AED_MAX_HISTORY: usize = 256;

pub const CHUNK_FRAMES: usize = 64;
pub const LOOKAHEAD_FRAMES: usize = 16;

/// Encoder AED window flattened: `[AED_WINDOW_FRAMES, DECODER_DIM]`.
pub const ENC_ELEMS: usize = AED_WINDOW_FRAMES * DECODER_DIM;
pub const AED_CACHE_IN_ELEMS: usize = BEAM * AED_MAX_HISTORY * 2 * DECODER_LAYERS * DECODER_DIM;

pub const CTC_SCALE: f32 = 0.3;
pub const AED_SCALE: f32 = 0.7;
pub const RESCORE_CTC_SCALE: f32 = 0.7;
pub const RESCORE_AED_SCALE: f32 = 0.3;
