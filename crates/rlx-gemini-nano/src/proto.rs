//! Schema-less protobuf wire-format decoder.
//!
//! MediaPipe-converted LLM files carry their architecture description
//! (layer count, head counts, norm and activation choices, …) as a serialized
//! protobuf inside a TFLite metadata buffer. The `.proto` is not shipped with
//! the model, so this decodes the wire format into field numbers and raw
//! values; [`crate::inspect`] prints the tree so the fields can be identified.

use anyhow::{Result, anyhow, bail};

#[derive(Clone, Debug, PartialEq)]
pub enum WireValue {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    /// Length-delimited: a string, bytes, packed repeated scalars or a
    /// nested message. Which one is ambiguous without the schema.
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct WireField {
    pub number: u32,
    pub value: WireValue,
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut out = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf
            .get(*pos)
            .ok_or_else(|| anyhow!("truncated varint at {pos}"))?;
        *pos += 1;
        out |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(out);
        }
    }
    bail!("varint longer than 10 bytes")
}

/// Decode one message level. Fails on anything that is not well-formed, which
/// is also how [`looks_like_message`] tells nested messages from strings.
pub fn decode(buf: &[u8]) -> Result<Vec<WireField>> {
    let mut pos = 0;
    let mut out = Vec::new();
    while pos < buf.len() {
        let key = varint(buf, &mut pos)?;
        let number = u32::try_from(key >> 3).map_err(|_| anyhow!("field number overflow"))?;
        if number == 0 {
            bail!("field number 0 at {pos}");
        }
        let take = |pos: &mut usize, n: usize| -> Result<&[u8]> {
            let end = pos
                .checked_add(n)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| anyhow!("field {number} runs past end"))?;
            let s = &buf[*pos..end];
            *pos = end;
            Ok(s)
        };
        let value = match key & 7 {
            0 => WireValue::Varint(varint(buf, &mut pos)?),
            1 => WireValue::Fixed64(u64::from_le_bytes(take(&mut pos, 8)?.try_into()?)),
            2 => {
                let n = usize::try_from(varint(buf, &mut pos)?)?;
                WireValue::Bytes(take(&mut pos, n)?.to_vec())
            }
            5 => WireValue::Fixed32(u32::from_le_bytes(take(&mut pos, 4)?.try_into()?)),
            w => bail!("unsupported wire type {w} for field {number}"),
        };
        out.push(WireField { number, value });
    }
    Ok(out)
}

/// Heuristic: printable UTF-8 is a string, otherwise a clean decode is a
/// nested message.
pub fn looks_like_message(bytes: &[u8]) -> bool {
    if bytes.is_empty() || is_printable(bytes) {
        return false;
    }
    decode(bytes).is_ok()
}

pub fn is_printable(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok_and(|s| s.chars().all(|c| !c.is_control() || c == '\n'))
}

/// Render the decoded tree, one field per line, nested messages indented.
pub fn render(bytes: &[u8]) -> Result<String> {
    let mut out = String::new();
    render_into(bytes, 0, &mut out)?;
    Ok(out)
}

fn render_into(bytes: &[u8], depth: usize, out: &mut String) -> Result<()> {
    use std::fmt::Write;
    let pad = "  ".repeat(depth);
    for f in decode(bytes)? {
        match &f.value {
            WireValue::Varint(v) => writeln!(out, "{pad}{}: {v}", f.number)?,
            WireValue::Fixed32(v) => {
                writeln!(out, "{pad}{}: {} (f32 {})", f.number, v, f32::from_bits(*v))?
            }
            WireValue::Fixed64(v) => {
                writeln!(out, "{pad}{}: {} (f64 {})", f.number, v, f64::from_bits(*v))?
            }
            WireValue::Bytes(b) if looks_like_message(b) => {
                writeln!(out, "{pad}{} {{", f.number)?;
                render_into(b, depth + 1, out)?;
                writeln!(out, "{pad}}}")?;
            }
            WireValue::Bytes(b) if is_printable(b) => {
                writeln!(out, "{pad}{}: {:?}", f.number, String::from_utf8_lossy(b))?
            }
            WireValue::Bytes(b) => writeln!(out, "{pad}{}: <{} bytes>", f.number, b.len())?,
        }
    }
    Ok(())
}
