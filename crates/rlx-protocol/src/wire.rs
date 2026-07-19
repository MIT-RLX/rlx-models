//! Staged-inference wire codec.
//!
//! Ported and renamed from mesh-llm's `skippy-protocol` binary activation
//! codec. The core wire object is the [`ActivationFrame`]: a `tokens × width`
//! block of hidden-state activations that flows from one pipeline stage to the
//! next, carried on the wire in one of three dtypes ([`ActivationDType`]).
//!
//! Framing is manual and length-prefixed so the format is transport-agnostic
//! (QUIC / TCP / in-process channel — none of which live in this crate). Every
//! frame is self-describing: it carries its own token count, activation width,
//! dtype tag, and per-token `state_flags` sideband bits.
//!
//! Stage-control messages ([`StageControl`]) are plain serde structs encoded
//! with `serde_json` — deliberately *not* protobuf, so this crate needs no
//! `protoc`/`prost` build dependency.

use std::convert::TryFrom;

use half::f16;
use serde::{Deserialize, Serialize};

/// ALPN identifier advertised by a stage endpoint. Bumped from the skippy
/// `skippy-stage/2` token to the rlx namespace, generation 2.
pub const STAGE_ALPN: &[u8] = b"rlx-stage/2";

/// Human-readable subprotocol name (advertised alongside the ALPN).
pub const STAGE_SUBPROTOCOL_NAME: &str = "rlx-stage";
/// Major version of the stage subprotocol.
pub const STAGE_SUBPROTOCOL_MAJOR: u32 = 2;
/// Schema version stamped into stage-control payloads.
pub const SCHEMA_VERSION: u32 = 1;

/// Stream sub-id carrying stage-control messages ([`StageControl`]).
pub const STAGE_STREAM_CONTROL: u8 = 0x01;
/// Stream sub-id carrying activation transport frames ([`ActivationFrame`]).
pub const STAGE_STREAM_TRANSPORT: u8 = 0x02;
/// Stream sub-id carrying artifact (weight-slice) transfers.
pub const STAGE_STREAM_ARTIFACT_TRANSFER: u8 = 0x03;

/// Maximum size of a single encoded stage frame on the wire.
pub const MAX_STAGE_FRAME_BYTES: usize = 8 * 1024 * 1024;
/// Maximum decoded (F32-expanded) activation payload byte count. Guards against
/// hostile length prefixes that would decode to an enormous allocation.
pub const MAX_STAGE_DECODED_ACTIVATION_BYTES: usize = 512 * 1024 * 1024;

/// Per-token sideband flags that ride alongside an activation frame. These
/// mirror the skippy `state_flags` sideband bits: some model families emit
/// extra per-token planes (RWKV7 `v_first`, Gemma3n AltUp) that widen the
/// payload by an integer multiplier.
pub mod state_flags {
    /// RWKV7 `v_first` sideband present — payload carries 2× tokens worth of rows.
    pub const RWKV7_V_FIRST_SIDEBAND: i32 = 1 << 6;
    /// Gemma3n AltUp sideband present — payload carries 4× tokens worth of rows.
    pub const GEMMA3N_ALTUP_SIDEBAND: i32 = 1 << 7;
}

/// Number of activation rows carried per logical token, given the sideband
/// flags. RWKV7 doubles, Gemma3n AltUp quadruples, otherwise 1×.
pub fn activation_payload_multiplier(state_flag_bits: i32) -> usize {
    if (state_flag_bits & state_flags::GEMMA3N_ALTUP_SIDEBAND) != 0 {
        4
    } else if (state_flag_bits & state_flags::RWKV7_V_FIRST_SIDEBAND) != 0 {
        2
    } else {
        1
    }
}

/// Error type for the wire codec. Manual framing failures surface here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// Buffer ended before a fully-described frame could be read.
    Truncated { needed: usize, got: usize },
    /// A negative token count or width was requested.
    NegativeDimension,
    /// A length/element multiply overflowed `usize`.
    Overflow,
    /// The dtype tag byte did not name a known [`ActivationDType`].
    UnknownDType(u8),
    /// The payload length did not match the length implied by the header.
    PayloadSizeMismatch { expected: usize, got: usize },
    /// The decoded (F32) payload would exceed [`MAX_STAGE_DECODED_ACTIVATION_BYTES`].
    DecodedTooLarge { bytes: usize },
    /// The frame magic prefix was wrong.
    BadMagic,
    /// A JSON stage-control payload failed to (de)serialize.
    Json(String),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { needed, got } => {
                write!(f, "truncated frame: needed {needed} bytes, got {got}")
            }
            Self::NegativeDimension => write!(f, "negative activation dimension"),
            Self::Overflow => write!(f, "activation size computation overflowed"),
            Self::UnknownDType(tag) => write!(f, "unknown activation wire dtype tag {tag}"),
            Self::PayloadSizeMismatch { expected, got } => {
                write!(
                    f,
                    "activation payload size mismatch: expected {expected}, got {got}"
                )
            }
            Self::DecodedTooLarge { bytes } => {
                write!(
                    f,
                    "decoded activation payload of {bytes} bytes exceeds maximum"
                )
            }
            Self::BadMagic => write!(f, "activation frame magic mismatch"),
            Self::Json(msg) => write!(f, "stage control json error: {msg}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Wire dtype for an activation payload.
///
/// * `F32` — 4 bytes/element, no transform.
/// * `F16` — 2 bytes/element, IEEE half via the `half` crate.
/// * `Q8`  — per-token f32 scale (`4·tokens` bytes) followed by `i8` codes
///   (`tokens·width` bytes); reconstructs to `code · scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ActivationDType {
    F32 = 0,
    F16 = 1,
    Q8 = 2,
}

impl ActivationDType {
    /// Numeric wire tag.
    pub fn tag(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for ActivationDType {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q8),
            other => Err(WireError::UnknownDType(other)),
        }
    }
}

/// Frame magic: "RLXA" (RLX Activation), little-endian.
const FRAME_MAGIC: u32 = 0x414c_5852; // 'R''L''X''A' reversed for LE readability

/// A single activation transport frame.
///
/// `data` holds the *wire-form* payload for `dtype` (already F16-narrowed or
/// Q8-quantized when applicable). `tokens` is the logical token count and
/// `width` the embedding width (`n_embd`); the physical row count is
/// `tokens × activation_payload_multiplier(state_flags)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationFrame {
    /// Logical token count.
    pub tokens: u32,
    /// Activation / embedding width (`n_embd`).
    pub width: u32,
    /// Wire dtype of `data`.
    pub dtype: ActivationDType,
    /// Wire-form payload bytes.
    pub data: Vec<u8>,
    /// Per-token sideband flags (see [`state_flags`]).
    pub state_flags: i32,
}

impl ActivationFrame {
    /// Physical row count = logical tokens × sideband multiplier.
    fn physical_tokens(&self) -> Result<usize, WireError> {
        (self.tokens as usize)
            .checked_mul(activation_payload_multiplier(self.state_flags))
            .ok_or(WireError::Overflow)
    }

    /// Expected wire-form byte length of `data` for this frame's dtype/shape.
    pub fn wire_bytes(&self) -> Result<usize, WireError> {
        wire_bytes(self.dtype, self.physical_tokens()?, self.width as usize)
    }

    /// Build an F32 frame from a raw little-endian f32 byte payload.
    ///
    /// Length is validated against `tokens · multiplier · width · 4`.
    pub fn from_f32_bytes(
        tokens: u32,
        width: u32,
        state_flags: i32,
        f32_payload: Vec<u8>,
    ) -> Result<Self, WireError> {
        let physical = (tokens as usize)
            .checked_mul(activation_payload_multiplier(state_flags))
            .ok_or(WireError::Overflow)?;
        let expected = wire_bytes(ActivationDType::F32, physical, width as usize)?;
        if f32_payload.len() != expected {
            return Err(WireError::PayloadSizeMismatch {
                expected,
                got: f32_payload.len(),
            });
        }
        Ok(Self {
            tokens,
            width,
            dtype: ActivationDType::F32,
            data: f32_payload,
            state_flags,
        })
    }

    /// Build a frame in `dtype`, transforming a raw little-endian f32 payload
    /// into the requested wire form (identity for F32, narrow for F16, quantize
    /// for Q8).
    pub fn encode_from_f32(
        dtype: ActivationDType,
        tokens: u32,
        width: u32,
        state_flags: i32,
        f32_payload: &[u8],
    ) -> Result<Self, WireError> {
        let physical = (tokens as usize)
            .checked_mul(activation_payload_multiplier(state_flags))
            .ok_or(WireError::Overflow)?;
        let expected_f32 = wire_bytes(ActivationDType::F32, physical, width as usize)?;
        if expected_f32 > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(WireError::DecodedTooLarge {
                bytes: expected_f32,
            });
        }
        if f32_payload.len() != expected_f32 {
            return Err(WireError::PayloadSizeMismatch {
                expected: expected_f32,
                got: f32_payload.len(),
            });
        }
        let data = match dtype {
            ActivationDType::F32 => f32_payload.to_vec(),
            ActivationDType::F16 => encode_f32_to_f16_bytes(f32_payload)?,
            ActivationDType::Q8 => encode_f32_to_q8_bytes(f32_payload, physical, width as usize)?,
        };
        Ok(Self {
            tokens,
            width,
            dtype,
            data,
            state_flags,
        })
    }

    /// Decode this frame's payload back to a raw little-endian f32 byte buffer.
    pub fn to_f32_bytes(&self) -> Result<Vec<u8>, WireError> {
        let physical = self.physical_tokens()?;
        match self.dtype {
            ActivationDType::F32 => {
                if self.data.len() > MAX_STAGE_DECODED_ACTIVATION_BYTES {
                    return Err(WireError::DecodedTooLarge {
                        bytes: self.data.len(),
                    });
                }
                Ok(self.data.clone())
            }
            ActivationDType::F16 => decode_f16_to_f32_bytes(&self.data),
            ActivationDType::Q8 => {
                decode_q8_to_f32_bytes(&self.data, physical, self.width as usize)
            }
        }
    }

    /// Encode the frame with length-prefixed manual framing.
    ///
    /// Layout (all integers little-endian):
    /// `magic(u32) | tokens(u32) | width(u32) | state_flags(i32) | dtype(u8) |
    /// pad(3) | payload_len(u32) | payload[payload_len]`.
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.data.len();
        let mut out = Vec::with_capacity(4 + 4 + 4 + 4 + 4 + 4 + payload_len);
        out.extend_from_slice(&FRAME_MAGIC.to_le_bytes());
        out.extend_from_slice(&self.tokens.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.state_flags.to_le_bytes());
        out.push(self.dtype.tag());
        out.extend_from_slice(&[0u8; 3]); // pad to 4-byte alignment
        out.extend_from_slice(&(payload_len as u32).to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// Decode a frame produced by [`ActivationFrame::encode`].
    ///
    /// Validates the magic, dtype tag, and that the trailing payload length
    /// matches the length implied by the shape/dtype header. Returns the number
    /// of bytes consumed alongside the frame so callers can stream multiple.
    pub fn decode_prefixed(input: &[u8]) -> Result<(Self, usize), WireError> {
        const HEADER: usize = 4 + 4 + 4 + 4 + 4 + 4;
        if input.len() < HEADER {
            return Err(WireError::Truncated {
                needed: HEADER,
                got: input.len(),
            });
        }
        let magic = u32::from_le_bytes(input[0..4].try_into().unwrap());
        if magic != FRAME_MAGIC {
            return Err(WireError::BadMagic);
        }
        let tokens = u32::from_le_bytes(input[4..8].try_into().unwrap());
        let width = u32::from_le_bytes(input[8..12].try_into().unwrap());
        let state_flags = i32::from_le_bytes(input[12..16].try_into().unwrap());
        let dtype = ActivationDType::try_from(input[16])?;
        // input[17..20] is padding.
        let payload_len = u32::from_le_bytes(input[20..24].try_into().unwrap()) as usize;
        let end = HEADER.checked_add(payload_len).ok_or(WireError::Overflow)?;
        if input.len() < end {
            return Err(WireError::Truncated {
                needed: end,
                got: input.len(),
            });
        }
        let data = input[HEADER..end].to_vec();

        let frame = Self {
            tokens,
            width,
            dtype,
            data,
            state_flags,
        };
        // Validate the declared payload length against the shape/dtype header.
        let expected = frame.wire_bytes()?;
        if frame.data.len() != expected {
            return Err(WireError::PayloadSizeMismatch {
                expected,
                got: frame.data.len(),
            });
        }
        Ok((frame, end))
    }

    /// Decode a single frame, requiring `input` to be exactly one frame.
    pub fn decode(input: &[u8]) -> Result<Self, WireError> {
        let (frame, consumed) = Self::decode_prefixed(input)?;
        if consumed != input.len() {
            return Err(WireError::PayloadSizeMismatch {
                expected: consumed,
                got: input.len(),
            });
        }
        Ok(frame)
    }
}

/// Wire-form byte count for a `physical_tokens × width` block in `dtype`.
pub fn wire_bytes(
    dtype: ActivationDType,
    physical_tokens: usize,
    width: usize,
) -> Result<usize, WireError> {
    let elements = physical_tokens
        .checked_mul(width)
        .ok_or(WireError::Overflow)?;
    match dtype {
        ActivationDType::F32 => elements.checked_mul(4).ok_or(WireError::Overflow),
        ActivationDType::F16 => elements.checked_mul(2).ok_or(WireError::Overflow),
        ActivationDType::Q8 => physical_tokens
            .checked_mul(4)
            .and_then(|scales| scales.checked_add(elements))
            .ok_or(WireError::Overflow),
    }
}

fn encode_f32_to_f16_bytes(input: &[u8]) -> Result<Vec<u8>, WireError> {
    if input.len() & 3 != 0 {
        return Err(WireError::PayloadSizeMismatch {
            expected: input.len() & !3,
            got: input.len(),
        });
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(out)
}

fn decode_f16_to_f32_bytes(input: &[u8]) -> Result<Vec<u8>, WireError> {
    if input.len() & 1 != 0 {
        return Err(WireError::PayloadSizeMismatch {
            expected: input.len() & !1,
            got: input.len(),
        });
    }
    let decoded = input.len().checked_mul(2).ok_or(WireError::Overflow)?;
    if decoded > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(WireError::DecodedTooLarge { bytes: decoded });
    }
    let mut out = Vec::with_capacity(decoded);
    for chunk in input.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&f16::from_bits(bits).to_f32().to_le_bytes());
    }
    Ok(out)
}

/// Q8 encode: per-token `max(|x|)/127` scale (LE f32) block, then `i8` codes.
fn encode_f32_to_q8_bytes(
    input: &[u8],
    physical_tokens: usize,
    width: usize,
) -> Result<Vec<u8>, WireError> {
    let expected = physical_tokens
        .checked_mul(width)
        .and_then(|e| e.checked_mul(4))
        .ok_or(WireError::Overflow)?;
    if input.len() != expected {
        return Err(WireError::PayloadSizeMismatch {
            expected,
            got: input.len(),
        });
    }
    let mut scales = Vec::with_capacity(physical_tokens * 4);
    let mut packed = Vec::with_capacity(physical_tokens * width);
    for token_index in 0..physical_tokens {
        let row_offset = token_index * width * 4;
        let row = &input[row_offset..row_offset + width * 4];
        let mut max_abs = 0.0_f32;
        for chunk in row.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            max_abs = max_abs.max(value.abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.extend_from_slice(&scale.to_le_bytes());
        for chunk in row.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
            packed.push(quantized as u8);
        }
    }
    scales.extend_from_slice(&packed);
    Ok(scales)
}

/// Q8 decode: reconstruct `code · scale` per row.
fn decode_q8_to_f32_bytes(
    input: &[u8],
    physical_tokens: usize,
    width: usize,
) -> Result<Vec<u8>, WireError> {
    let scale_bytes = physical_tokens.checked_mul(4).ok_or(WireError::Overflow)?;
    let value_bytes = physical_tokens
        .checked_mul(width)
        .ok_or(WireError::Overflow)?;
    let expected = scale_bytes
        .checked_add(value_bytes)
        .ok_or(WireError::Overflow)?;
    if input.len() != expected {
        return Err(WireError::PayloadSizeMismatch {
            expected,
            got: input.len(),
        });
    }
    let decoded = value_bytes.checked_mul(4).ok_or(WireError::Overflow)?;
    if decoded > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(WireError::DecodedTooLarge { bytes: decoded });
    }
    let mut out = Vec::with_capacity(decoded);
    for token_index in 0..physical_tokens {
        let scale_offset = token_index * 4;
        let scale = f32::from_le_bytes([
            input[scale_offset],
            input[scale_offset + 1],
            input[scale_offset + 2],
            input[scale_offset + 3],
        ]);
        let row_offset = scale_bytes + token_index * width;
        for value in &input[row_offset..row_offset + width] {
            let signed = *value as i8;
            out.extend_from_slice(&((signed as f32) * scale).to_le_bytes());
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage-control messages (serde_json, no protobuf).
// ---------------------------------------------------------------------------

/// Load-mode for a stage's weights, mirroring the skippy `LoadMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoadMode {
    RuntimeSlice,
    LayerPackage,
    ArtifactSlice,
}

/// A neighbouring stage this stage streams activations to/from.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PeerConfig {
    pub stage_id: String,
    pub stage_index: u32,
    pub endpoint: String,
}

/// Command asking a node to load a contiguous layer range as a pipeline stage.
///
/// Ported from the skippy `LoadStage` proto message; carries the same fields
/// (stage/layer range, upstream/downstream peers, wire dtype, activation width)
/// as a plain serde struct.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LoadStage {
    pub topology_id: String,
    pub run_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Wire dtype the stage should emit downstream.
    pub wire_dtype: ActivationDType,
    /// Activation / embedding width (`n_embd`).
    pub activation_width: u32,
    #[serde(default)]
    pub load_mode: Option<LoadMode>,
    #[serde(default)]
    pub package_ref: Option<String>,
    #[serde(default)]
    pub manifest_sha256: Option<String>,
    #[serde(default)]
    pub bind_addr: Option<String>,
    #[serde(default)]
    pub upstream: Option<PeerConfig>,
    #[serde(default)]
    pub downstream: Option<PeerConfig>,
}

/// Query for a stage's current status.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct GetStatus {
    pub topology_id: String,
    pub run_id: String,
    pub stage_id: String,
}

/// Ask a node to pre-stage (download/materialize) a layer range before load.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Prepare {
    pub load_stage: LoadStage,
    #[serde(default)]
    pub coordinator_id: Option<String>,
}

/// Runtime state of a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRuntimeState {
    Unassigned,
    Assigned,
    Loading,
    Ready,
    Failed,
    Stopped,
}

/// Progress / status report emitted by a stage.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StatusUpdate {
    pub topology_id: String,
    pub run_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub state: StageRuntimeState,
    #[serde(default)]
    pub bind_addr: Option<String>,
    #[serde(default)]
    pub activation_width: Option<u32>,
    #[serde(default)]
    pub wire_dtype: Option<ActivationDType>,
    #[serde(default)]
    pub bytes_done: Option<u64>,
    #[serde(default)]
    pub bytes_total: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Stage-control message envelope. Encoded as `serde_json`, carried on the
/// [`STAGE_STREAM_CONTROL`] stream.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageControl {
    LoadStage(LoadStage),
    GetStatus(GetStatus),
    Prepare(Prepare),
    StatusUpdate(StatusUpdate),
}

impl StageControl {
    /// Serialize to a JSON byte buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WireError> {
        serde_json::to_vec(self).map_err(|e| WireError::Json(e.to_string()))
    }

    /// Deserialize from a JSON byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        serde_json::from_slice(bytes).map_err(|e| WireError::Json(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_f32_payload(tokens: usize, width: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens * width * 4);
        for t in 0..tokens {
            for w in 0..width {
                let v = (t as f32) * 0.5 - (w as f32) * 0.25 + 1.0;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn activation_frame_roundtrip_f32() {
        let (tokens, width) = (5u32, 7u32);
        let payload = sample_f32_payload(tokens as usize, width as usize);
        let frame =
            ActivationFrame::encode_from_f32(ActivationDType::F32, tokens, width, 0, &payload)
                .unwrap();
        let encoded = frame.encode();
        let decoded = ActivationFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(decoded.dtype, ActivationDType::F32);
        // F32 is exact.
        assert_eq!(decoded.to_f32_bytes().unwrap(), payload);
    }

    #[test]
    fn activation_frame_roundtrip_f16() {
        let (tokens, width) = (3u32, 4u32);
        let payload = sample_f32_payload(tokens as usize, width as usize);
        let frame =
            ActivationFrame::encode_from_f32(ActivationDType::F16, tokens, width, 0, &payload)
                .unwrap();
        assert_eq!(frame.data.len(), (tokens * width * 2) as usize);
        let encoded = frame.encode();
        let (decoded, consumed) = ActivationFrame::decode_prefixed(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.dtype, ActivationDType::F16);
        // F16 is lossy but should be close.
        let original = f32_vec(&payload);
        let roundtrip = f32_vec(&decoded.to_f32_bytes().unwrap());
        assert_eq!(original.len(), roundtrip.len());
        for (a, b) in original.iter().zip(roundtrip.iter()) {
            assert!((a - b).abs() < 0.01, "f16 drift too large: {a} vs {b}");
        }
    }

    #[test]
    fn activation_frame_roundtrip_q8() {
        let (tokens, width) = (4u32, 6u32);
        let payload = sample_f32_payload(tokens as usize, width as usize);
        let frame =
            ActivationFrame::encode_from_f32(ActivationDType::Q8, tokens, width, 0, &payload)
                .unwrap();
        // Q8 wire size: per-token f32 scales + per-element u8 codes.
        assert_eq!(frame.data.len(), (tokens * 4 + tokens * width) as usize);
        let encoded = frame.encode();
        let decoded = ActivationFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.dtype, ActivationDType::Q8);
        // Q8 is lossy; per-row error bounded by the row's scale (max_abs/127).
        let original = f32_vec(&payload);
        let roundtrip = f32_vec(&decoded.to_f32_bytes().unwrap());
        assert_eq!(original.len(), roundtrip.len());
        for row in 0..tokens as usize {
            let row_slice = &original[row * width as usize..(row + 1) * width as usize];
            let max_abs = row_slice.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let tol = (max_abs / 127.0) + 1e-6;
            for i in 0..width as usize {
                let a = original[row * width as usize + i];
                let b = roundtrip[row * width as usize + i];
                assert!(
                    (a - b).abs() <= tol,
                    "q8 drift too large: {a} vs {b} tol {tol}"
                );
            }
        }
    }

    #[test]
    fn q8_sideband_multiplier_widens_payload() {
        let (tokens, width) = (2u32, 3u32);
        let mult = activation_payload_multiplier(state_flags::RWKV7_V_FIRST_SIDEBAND);
        assert_eq!(mult, 2);
        let physical = tokens as usize * mult;
        let payload = sample_f32_payload(physical, width as usize);
        let frame = ActivationFrame::encode_from_f32(
            ActivationDType::Q8,
            tokens,
            width,
            state_flags::RWKV7_V_FIRST_SIDEBAND,
            &payload,
        )
        .unwrap();
        let encoded = frame.encode();
        let decoded = ActivationFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.state_flags, state_flags::RWKV7_V_FIRST_SIDEBAND);
        assert_eq!(
            decoded.data.len(),
            (physical * 4 + physical * width as usize)
        );
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = vec![0u8; 24];
        bytes[0] = 0xde;
        assert!(matches!(
            ActivationFrame::decode_prefixed(&bytes),
            Err(WireError::BadMagic)
        ));
    }

    #[test]
    fn decode_rejects_truncated() {
        let payload = sample_f32_payload(2, 2);
        let frame =
            ActivationFrame::encode_from_f32(ActivationDType::F32, 2, 2, 0, &payload).unwrap();
        let encoded = frame.encode();
        let err = ActivationFrame::decode_prefixed(&encoded[..encoded.len() - 3]).unwrap_err();
        assert!(matches!(err, WireError::Truncated { .. }));
    }

    #[test]
    fn stage_control_load_roundtrip() {
        let msg = StageControl::LoadStage(LoadStage {
            topology_id: "topo-a".into(),
            run_id: "run-a".into(),
            model_id: "qwen3".into(),
            stage_id: "stage-1".into(),
            stage_index: 1,
            layer_start: 8,
            layer_end: 16,
            wire_dtype: ActivationDType::F16,
            activation_width: 4096,
            load_mode: Some(LoadMode::LayerPackage),
            package_ref: Some("hf://repo/model".into()),
            manifest_sha256: None,
            bind_addr: Some("127.0.0.1:0".into()),
            upstream: Some(PeerConfig {
                stage_id: "stage-0".into(),
                stage_index: 0,
                endpoint: "127.0.0.1:5000".into(),
            }),
            downstream: None,
        });
        let bytes = msg.to_bytes().unwrap();
        let back = StageControl::from_bytes(&bytes).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn stage_control_status_roundtrip() {
        let msg = StageControl::StatusUpdate(StatusUpdate {
            topology_id: "topo-a".into(),
            run_id: "run-a".into(),
            stage_id: "stage-1".into(),
            stage_index: 1,
            layer_start: 8,
            layer_end: 16,
            state: StageRuntimeState::Ready,
            bind_addr: Some("127.0.0.1:51234".into()),
            activation_width: Some(4096),
            wire_dtype: Some(ActivationDType::Q8),
            bytes_done: Some(10),
            bytes_total: Some(20),
            error: None,
        });
        let bytes = msg.to_bytes().unwrap();
        assert_eq!(StageControl::from_bytes(&bytes).unwrap(), msg);
    }
}
