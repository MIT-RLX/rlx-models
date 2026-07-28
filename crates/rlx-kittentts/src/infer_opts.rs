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

//! Native compile sizing and ONNX duration → waveform length mapping.

/// Vocoder samples per duration unit (matches ONNX Kitten mini 0.8).
pub const SAMPLES_PER_DURATION_UNIT: usize = 600;

/// Minimum compiled sequence length for native graphs (vocoder stable below ~16 slots).
pub const MIN_NATIVE_SEQUENCE_LENGTH: usize = 16;

/// Default chunk cap when no compiled sequence length is available (legacy tests).
pub const MAX_NATIVE_CHUNK_SLOTS: usize = 48;

/// Max padded IPA ids per infer pass (runtime token cap from compile).
pub fn infer_chunk_slots(sequence_length: usize) -> usize {
    if let Ok(raw) = std::env::var("KITTEN_RLX_CHUNK_SLOTS") {
        if let Ok(n) = raw.parse::<usize>() {
            return n.max(1);
        }
    }
    let budget = sequence_length
        .saturating_sub(DURATION_COMPILE_HEADROOM)
        .saturating_sub(2)
        .max(MIN_NATIVE_SEQUENCE_LENGTH);
    budget.clamp(MIN_NATIVE_SEQUENCE_LENGTH, 48)
}

/// Typical max duration units per token for vocoder buffer sizing.
const TYPICAL_MAX_DURATION_UNITS_PER_TOKEN: usize = 8;

/// Headroom above token×duration estimate for native waveform compile buffer.
const WAVEFORM_HEADROOM: usize = 12_000;

const COMPILE_WAVEFORM_FLOOR: usize = 24_000;

fn compile_waveform_cap(runtime_tokens: usize, engine_cap: usize) -> usize {
    let est = runtime_tokens
        .saturating_mul(SAMPLES_PER_DURATION_UNIT)
        .saturating_mul(TYPICAL_MAX_DURATION_UNITS_PER_TOKEN)
        .saturating_add(WAVEFORM_HEADROOM)
        .max(COMPILE_WAVEFORM_FLOOR);
    est.min(engine_cap.max(COMPILE_WAVEFORM_FLOOR))
}

/// Extra compile slots so duration epilogue buffers do not alias token rows
/// (matches `kitten_tts_mini_rlx::bundle_compile::DURATION_COMPILE_HEADROOM`).
const DURATION_COMPILE_HEADROOM: usize = 6;

/// Graphs compiled at this slot count or above *may* use the wide-seq vocoder path.
///
/// Historically we chunked above 32 because F0_conv/N_conv were stub-sized on wide
/// imports (Concat ASR∥F0∥N collapsed). After `repair_f0n_conv_shapes`, single-pass
/// at ~75 tokens is Whisper-intelligible; keep a higher ceiling so long phrases do
/// not get sliced into tiny exploding tail chunks.
pub const WIDE_COMPILE_SLOT_THRESHOLD: usize = 128;

/// Padded id length → compile slot length (`+ DURATION_COMPILE_HEADROOM`).
pub fn compile_slot_length(padded_len: usize) -> usize {
    padded_len.saturating_add(DURATION_COMPILE_HEADROOM)
}

/// Wide-seq vocoder: chunk only when compile slot would exceed [`WIDE_COMPILE_SLOT_THRESHOLD`].
pub fn needs_narrow_vocoder_chunking(padded_len: usize) -> bool {
    compile_slot_length(padded_len) >= WIDE_COMPILE_SLOT_THRESHOLD
}

/// Max padded `input_ids` per infer chunk so compile slot stays below [`WIDE_COMPILE_SLOT_THRESHOLD`].
pub fn narrow_chunk_slots() -> usize {
    WIDE_COMPILE_SLOT_THRESHOLD
        .saturating_sub(DURATION_COMPILE_HEADROOM)
        .saturating_sub(1)
        .max(MIN_NATIVE_SEQUENCE_LENGTH)
}

/// Typical duration units/token for wave-budget chunk sizing (hello is ~2–3;
/// worst-case compile uses 8). Was 4 — quiet long audio under 48 k Vulkan.
/// **3** packs ~17 tokens into the 32 k wgpu cap / ~44 into 80 k Vulkan.
/// Do not drop to 2: denser packing overflows the compile wave buffer
/// (truncated audio / collapsed peak on NVIDIA).
const CHUNK_DURATION_UNITS_PER_TOKEN: usize = 3;

/// Minimum padded ids we try to keep when merging a tiny tail chunk.
const MIN_CHUNK_IDS: usize = 12;

/// Max padded ids whose *typical* wave estimate fits in `max_wave`.
pub fn tokens_for_waveform_budget(max_wave: usize) -> usize {
    let per = SAMPLES_PER_DURATION_UNIT.saturating_mul(CHUNK_DURATION_UNITS_PER_TOKEN);
    if per == 0 {
        return 8;
    }
    (max_wave / per).clamp(8, MAX_NATIVE_CHUNK_SLOTS)
}

/// Typical-duration sample estimate for `token_len` padded ids.
#[inline]
fn typical_wave_samples(token_len: usize) -> usize {
    token_len
        .saturating_mul(SAMPLES_PER_DURATION_UNIT)
        .saturating_mul(CHUNK_DURATION_UNITS_PER_TOKEN)
}

/// Chunk width for this utterance (narrow vocoder vs length / wave caps).
pub fn effective_chunk_slots(sequence_length: usize, padded_len: usize) -> usize {
    effective_chunk_slots_with_wave(sequence_length, padded_len, usize::MAX)
}

/// Like [`effective_chunk_slots`], but also caps by waveform compile buffer.
pub fn effective_chunk_slots_with_wave(
    sequence_length: usize,
    padded_len: usize,
    max_wave: usize,
) -> usize {
    let engine = infer_chunk_slots(sequence_length).min(sequence_length.max(1));
    let wave_limited = max_wave != 0 && max_wave != usize::MAX;
    let wave_cap = if wave_limited {
        tokens_for_waveform_budget(max_wave)
    } else {
        usize::MAX
    };

    if needs_narrow_vocoder_chunking(padded_len) {
        return narrow_chunk_slots().min(engine).min(wave_cap);
    }

    // Single-pass when the compiled engine holds the utterance and the wave
    // buffer covers a typical-duration estimate.
    if padded_len <= sequence_length
        && (!wave_limited || typical_wave_samples(padded_len) <= max_wave)
    {
        return padded_len;
    }

    engine.min(wave_cap).max(1)
}

/// Split ids into infer chunks (wide-seq, engine length, or wave-buffer limits).
pub fn chunk_plan(ids: &[i64], sequence_length: usize) -> Vec<(Vec<i64>, usize)> {
    chunk_plan_with_wave(ids, sequence_length, usize::MAX)
}

/// Like [`chunk_plan`], capping chunk width so each piece fits `max_wave`.
pub fn chunk_plan_with_wave(
    ids: &[i64],
    sequence_length: usize,
    max_wave: usize,
) -> Vec<(Vec<i64>, usize)> {
    let slots = effective_chunk_slots_with_wave(sequence_length, ids.len(), max_wave);
    if ids.len() <= slots {
        return vec![(ids.to_vec(), 0)];
    }
    merge_tiny_tail_chunks(chunk_padded_ids_with_offsets(ids, slots), slots)
}

/// Merge a too-small final chunk into the previous one when the merge still
/// fits `slots` (tiny vocoder tails are unstable).
fn merge_tiny_tail_chunks(
    mut chunks: Vec<(Vec<i64>, usize)>,
    slots: usize,
) -> Vec<(Vec<i64>, usize)> {
    while chunks.len() > 1 {
        let last_len = chunks.last().map(|(c, _)| c.len()).unwrap_or(0);
        if last_len >= MIN_CHUNK_IDS {
            break;
        }
        let (tail, tail_start) = chunks.pop().unwrap();
        let (prev, prev_start) = chunks.pop().unwrap();
        let mut merged = Vec::with_capacity(prev.len() + tail.len());
        merged.push(0);
        if prev.len() > 2 {
            merged.extend_from_slice(&prev[1..prev.len() - 1]);
        }
        if tail.len() > 2 {
            merged.extend_from_slice(&tail[1..tail.len() - 1]);
        }
        merged.push(0);
        if merged.len() > slots {
            chunks.push((prev, prev_start));
            chunks.push((tail, tail_start));
            break;
        }
        chunks.push((merged, prev_start));
    }
    chunks
}

pub fn native_compile_token_cap(token_len: usize) -> usize {
    if needs_narrow_vocoder_chunking(token_len) {
        narrow_chunk_slots()
    } else {
        token_len.max(1)
    }
}

/// Waveform compile buffer for one infer chunk (see [`recommended_native_compile_opts`]).
pub fn max_waveform_samples_for_tokens(token_len: usize) -> usize {
    let compile_tokens = native_compile_token_cap(token_len);
    compile_waveform_cap(compile_tokens, usize::MAX)
}

/// Recommended `(sequence_length, max_waveform_samples)` for native compile.
///
/// Sized for **one infer chunk** (≤ [`narrow_chunk_slots`] when vocoder chunking applies),
/// not the full padded IPA length — keeps compile arenas in the low hundreds of MB, not tens of GB.
/// Graphs still compile with `token_cap + DURATION_COMPILE_HEADROOM` via [`SeqCompileCache`].
pub fn recommended_native_compile_opts(token_len: usize) -> (usize, usize) {
    let compile_tokens = native_compile_token_cap(token_len);
    (compile_tokens, max_waveform_samples_for_tokens(token_len))
}

/// Split padded IPA ids `[0, …, 0]` into chunks that fit `max_slots`.
pub fn chunk_padded_ids(ids: &[i64], max_slots: usize) -> Vec<Vec<i64>> {
    chunk_padded_ids_with_offsets(ids, max_slots)
        .into_iter()
        .map(|(chunk, _)| chunk)
        .collect()
}

/// Like [`chunk_padded_ids`], but each chunk carries its start index in the full id slice.
pub fn chunk_padded_ids_with_offsets(ids: &[i64], max_slots: usize) -> Vec<(Vec<i64>, usize)> {
    if ids.len() <= max_slots {
        return vec![(ids.to_vec(), 0)];
    }
    if ids.len() < 2 {
        return vec![(ids.to_vec(), 0)];
    }
    let content = &ids[1..ids.len() - 1];
    let budget = max_slots.saturating_sub(2).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < content.len() {
        let end = (start + budget).min(content.len());
        let mut chunk = vec![0i64];
        chunk.extend_from_slice(&content[start..end]);
        chunk.push(0);
        chunks.push((chunk, if start == 0 { 0 } else { 1 + start }));
        start = end;
    }
    chunks
}

/// Slice ORT duration for a chunk aligned to the full padded id sequence.
pub fn ort_duration_slice(full: &[i64], chunk_start: usize, chunk_len: usize) -> Vec<i64> {
    let end = (chunk_start + chunk_len).min(full.len());
    if chunk_start >= full.len() {
        return vec![0i64; chunk_len];
    }
    let mut out = full[chunk_start..end].to_vec();
    if out.len() < chunk_len {
        out.resize(chunk_len, 0);
    }
    out
}

/// Pad duration to compile slot length and encode as i64 LE bytes for `DURATION_CARRY`.
pub fn duration_carry_bytes(duration: &[i64], compile_seq: usize) -> Vec<u8> {
    let mut padded = duration.to_vec();
    if padded.len() < compile_seq {
        padded.resize(compile_seq, 0);
    } else if padded.len() > compile_seq {
        padded.truncate(compile_seq);
    }
    padded.iter().flat_map(|d| d.to_le_bytes()).collect()
}

/// Content token count inside a padded `[0, …, 0]` id slice.
pub fn content_token_count(ids: &[i64]) -> usize {
    ids.len().saturating_sub(2).max(1)
}

/// ONNX waveform sample count from the `duration` output and active token count.
pub fn waveform_samples_from_duration(duration: &[i64], token_len: usize) -> Option<usize> {
    if token_len == 0 || duration.is_empty() {
        return None;
    }
    let n = token_len.min(duration.len());
    let sum: i64 = duration[..n]
        .iter()
        .copied()
        .filter(|&d| d > 0 && d < 10_000)
        .sum();
    if sum <= 0 {
        return None;
    }
    Some(sum as usize * SAMPLES_PER_DURATION_UNIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_token_compile_opts() {
        let (seq, wave) = recommended_native_compile_opts(8);
        assert_eq!(seq, 8);
        assert_eq!(
            wave,
            8 * SAMPLES_PER_DURATION_UNIT * TYPICAL_MAX_DURATION_UNITS_PER_TOKEN
                + WAVEFORM_HEADROOM
        );
    }

    #[test]
    fn long_phrase_compile_opts_no_forced_chunk() {
        // LONG_IPA ~75 tokens: single-pass after F0_conv fix (threshold 128).
        let (seq, wave) = recommended_native_compile_opts(74);
        assert_eq!(seq, 74);
        assert!(
            wave >= 74 * SAMPLES_PER_DURATION_UNIT,
            "wave buffer should cover the full utterance, got {wave}"
        );
        assert!(!needs_narrow_vocoder_chunking(74));
    }

    #[test]
    fn wave_budget_forces_chunk_under_wgpu_cap() {
        let ids: Vec<i64> = std::iter::once(0)
            .chain(1..=73i64)
            .chain(std::iter::once(0))
            .collect();
        assert_eq!(ids.len(), 75);
        // Full wave → single pass when engine is wide enough.
        let one = chunk_plan_with_wave(&ids, 128, 400_000);
        assert_eq!(one.len(), 1);
        // Discrete wgpu safe cap → multiple chunks that fit the sample budget.
        for &cap_wave in &[24_000usize, 32_000] {
            let many = chunk_plan_with_wave(&ids, 128, cap_wave);
            assert!(
                many.len() > 1,
                "expected chunking under {cap_wave} wave, got {}",
                many.len()
            );
            let tok_cap = tokens_for_waveform_budget(cap_wave);
            assert!(
                many.iter().all(|(c, _)| c.len() <= tok_cap),
                "chunk wider than wave budget {tok_cap} at {cap_wave}: {:?}",
                many.iter().map(|(c, _)| c.len()).collect::<Vec<_>>()
            );
            assert!(
                typical_wave_samples(tok_cap) <= cap_wave || tok_cap == 8,
                "token budget {tok_cap} exceeds wave {cap_wave}"
            );
        }
    }

    #[test]
    fn chunk_plan_splits_only_above_wide_threshold() {
        let ids: Vec<i64> = std::iter::once(0)
            .chain(1..=200i64)
            .chain(std::iter::once(0))
            .collect();
        assert!(needs_narrow_vocoder_chunking(ids.len()));
        let chunks = chunk_plan(&ids, 256);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|(c, _)| compile_slot_length(c.len())
            < WIDE_COMPILE_SLOT_THRESHOLD
            || c.len() >= 12));
    }

    #[test]
    fn chunk_long_ipa_ids() {
        let ids: Vec<i64> = std::iter::once(0)
            .chain(1..=50i64)
            .chain(std::iter::once(0))
            .collect();
        let chunks = chunk_padded_ids(&ids, MAX_NATIVE_CHUNK_SLOTS);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.len() <= MAX_NATIVE_CHUNK_SLOTS));
        let chunks128 = chunk_padded_ids(&ids, 128);
        assert_eq!(chunks128.len(), 1);
    }

    #[test]
    fn chunk_offsets_align_with_full_ids() {
        let ids: Vec<i64> = std::iter::once(0)
            .chain(1..=50i64)
            .chain(std::iter::once(0))
            .collect();
        let chunks = chunk_padded_ids_with_offsets(&ids, MAX_NATIVE_CHUNK_SLOTS);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[1].1, 47);
    }

    #[test]
    fn ort_duration_slice_pads_tail() {
        let full = vec![0i64, 3, 2, 1, 0];
        let slice = ort_duration_slice(&full, 0, 5);
        assert_eq!(slice, full);
    }

    #[test]
    fn onnx_hello_duration_sum() {
        let dur = [19i64, 2, 1, 2, 3, 2, 3, 2];
        assert_eq!(
            waveform_samples_from_duration(&dur, 8),
            Some(34 * SAMPLES_PER_DURATION_UNIT)
        );
    }
}
