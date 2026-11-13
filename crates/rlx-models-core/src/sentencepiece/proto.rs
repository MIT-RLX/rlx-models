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

//! The protobuf wire format, as far as `ModelProto` needs it.

use anyhow::{Context, Result, bail};

pub(super) struct Field<'a> {
    pub field: u64,
    pub wire: u64,
    /// Payload of a length-delimited or fixed-width field.
    pub bytes: &'a [u8],
    /// Value of a varint field.
    pub varint: u64,
}

impl Field<'_> {
    pub fn f32(&self) -> Result<f32> {
        Ok(f32::from_le_bytes(
            self.bytes
                .try_into()
                .context("a float field is not 4 bytes")?,
        ))
    }

    pub fn string(&self, what: &str) -> Result<String> {
        String::from_utf8(self.bytes.to_vec()).with_context(|| format!("{what} is not UTF-8"))
    }

    pub fn bool(&self) -> bool {
        self.varint != 0
    }
}

fn varint(b: &[u8], mut at: usize) -> Result<(u64, usize)> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&byte) = b.get(at) else {
            bail!("a varint runs past the end");
        };
        at += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((v, at));
        }
    }
    bail!("a varint longer than 64 bits")
}

fn take(b: &[u8], at: usize, n: usize) -> Result<&[u8]> {
    match at.checked_add(n).and_then(|end| b.get(at..end)) {
        Some(s) => Ok(s),
        None => bail!("a field runs past the end"),
    }
}

/// Each field of a message, in order.
pub(super) fn fields(b: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < b.len() {
        let (key, next) = varint(b, at)?;
        at = next;
        let (field, wire) = (key >> 3, key & 7);
        let (bytes, value) = match wire {
            0 => {
                let (v, next) = varint(b, at)?;
                at = next;
                (&b[0..0], v)
            }
            1 => {
                let s = take(b, at, 8)?;
                at += 8;
                (s, 0)
            }
            2 => {
                let (len, next) = varint(b, at)?;
                let s = take(b, next, len as usize)?;
                at = next + len as usize;
                (s, 0)
            }
            5 => {
                let s = take(b, at, 4)?;
                at += 4;
                (s, 0)
            }
            w => bail!("wire type {w} in a SentencePiece model"),
        };
        out.push(Field {
            field,
            wire,
            bytes,
            varint: value,
        });
    }
    Ok(out)
}
