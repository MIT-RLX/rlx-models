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

//! Weight access with no allocator and no JSON parse.
//!
//! The safetensors blob is embedded and its layout is a build-time constant
//! ([`crate::weights_layout`]), so a tensor is a plain slice of a `&'static
//! [f32]` — no header parse, no heap, nothing to fail at run time.

use crate::weights_layout::LAYOUT;
use crate::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN, MEL_BANDS, WINDOW_SIZE};

/// Byte length of the embedded blob — checked against the file at test time.
const BLOB_LEN: usize = 305_080;

/// `include_bytes!` yields alignment 1; wrap it so the `f32` view is sound.
#[repr(align(4))]
struct Align4<T>(T);

static BLOB: Align4<[u8; BLOB_LEN]> = Align4(*include_bytes!(
    "../../rlx-ten-vad/weights/ten_vad.safetensors"
));

/// The blob as `f32`s.
///
/// Sound: `Align4` guarantees 4-byte alignment, the length is a multiple of 4,
/// and every bit pattern is a valid `f32` (safetensors stores little-endian
/// IEEE-754, which is the layout of `f32` on every target rlx builds for).
/// The whole f32 blob. Public so the fixed-point generator can walk it by
/// [`crate::weights_layout::LAYOUT`] rather than re-parsing safetensors.
pub fn blob_f32() -> &'static [f32] {
    // SAFETY: see above — alignment, length and validity all hold.
    unsafe { core::slice::from_raw_parts(BLOB.0.as_ptr().cast::<f32>(), BLOB_LEN / 4) }
}

fn tensor(name: &str) -> &'static [f32] {
    let all = blob_f32();
    let mut i = 0;
    while i < LAYOUT.len() {
        let (n, off, len) = LAYOUT[i];
        if n.as_bytes() == name.as_bytes() {
            return &all[off..off + len];
        }
        i += 1;
    }
    panic!("ten-vad: no such tensor in the embedded blob");
}

/// The DSP frontend's tables.
///
/// Borrowed rather than `&'static` so a host can hand over weights it loaded at
/// run time (`rlx-ten-vad`'s `--weights` override) as easily as an MCU hands
/// over [`embedded`]'s slices into flash.
pub struct CoreWeights<'a> {
    /// Hann-768 analysis window.
    pub window: &'a [f32],
    /// Per-feature standardization, `[FEATURE_LEN]` each.
    pub feature_mean: &'a [f32],
    pub feature_std: &'a [f32],
}

/// The network's tensors, in rlx layout (LSTM gate order `i, f, g, o`).
pub struct NetWeights<'a> {
    pub conv0_depthwise: &'a [f32],
    pub conv0_pointwise: &'a [f32],
    pub conv0_bias: &'a [f32],
    pub sep1_depthwise: &'a [f32],
    pub sep1_pointwise: &'a [f32],
    pub sep1_bias: &'a [f32],
    pub sep2_depthwise: &'a [f32],
    pub sep2_pointwise: &'a [f32],
    pub sep2_bias: &'a [f32],
    pub lstm1_weight_ih: &'a [f32],
    pub lstm1_weight_hh: &'a [f32],
    pub lstm1_bias: &'a [f32],
    pub lstm2_weight_ih: &'a [f32],
    pub lstm2_weight_hh: &'a [f32],
    pub lstm2_bias: &'a [f32],
    pub dense1_weight: &'a [f32],
    pub dense1_bias: &'a [f32],
    pub dense2_weight: &'a [f32],
    pub dense2_bias: &'a [f32],
}

/// The embedded frontend tables.
pub fn embedded() -> CoreWeights<'static> {
    let w = CoreWeights {
        window: tensor("stft.window"),
        feature_mean: tensor("feature.mean"),
        feature_std: tensor("feature.std"),
    };
    debug_assert_eq!(w.window.len(), WINDOW_SIZE);
    debug_assert_eq!(w.feature_mean.len(), FEATURE_LEN);
    debug_assert_eq!(FEATURE_LEN, MEL_BANDS + 1);
    debug_assert_eq!(CONTEXT_FRAMES, 3);
    w
}

/// The embedded network tensors.
pub fn embedded_net() -> NetWeights<'static> {
    let w = NetWeights {
        conv0_depthwise: tensor("conv0.depthwise.weight"),
        conv0_pointwise: tensor("conv0.pointwise.weight"),
        conv0_bias: tensor("conv0.bias"),
        sep1_depthwise: tensor("sep1.depthwise.weight"),
        sep1_pointwise: tensor("sep1.pointwise.weight"),
        sep1_bias: tensor("sep1.bias"),
        sep2_depthwise: tensor("sep2.depthwise.weight"),
        sep2_pointwise: tensor("sep2.pointwise.weight"),
        sep2_bias: tensor("sep2.bias"),
        lstm1_weight_ih: tensor("lstm1.weight_ih"),
        lstm1_weight_hh: tensor("lstm1.weight_hh"),
        lstm1_bias: tensor("lstm1.bias"),
        lstm2_weight_ih: tensor("lstm2.weight_ih"),
        lstm2_weight_hh: tensor("lstm2.weight_hh"),
        lstm2_bias: tensor("lstm2.bias"),
        dense1_weight: tensor("dense1.weight"),
        dense1_bias: tensor("dense1.bias"),
        dense2_weight: tensor("dense2.weight"),
        dense2_bias: tensor("dense2.bias"),
    };
    debug_assert_eq!(w.lstm1_weight_hh.len(), 4 * HIDDEN * HIDDEN);
    w
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// The baked layout must still describe the file it was generated from.
    #[test]
    fn layout_matches_blob() {
        let bytes = include_bytes!("../../rlx-ten-vad/weights/ten_vad.safetensors");
        assert_eq!(bytes.len(), BLOB_LEN, "blob length changed");
        let hdr_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let hdr = core::str::from_utf8(&bytes[8..8 + hdr_len]).expect("header utf8");
        for (name, off, len) in LAYOUT {
            // Locate `"name":{...,"data_offsets":[a,b]}` without a JSON parser.
            let key = alloc_key(name);
            let at = hdr
                .find(&key)
                .unwrap_or_else(|| panic!("{name} missing from header"));
            let tail = &hdr[at..];
            let da = tail.find("\"data_offsets\":[").expect("data_offsets") + 16;
            let rest = &tail[da..];
            let comma = rest.find(',').expect("offset comma");
            let close = rest.find(']').expect("offset close");
            let a: usize = rest[..comma].trim().parse().expect("a");
            let b: usize = rest[comma + 1..close].trim().parse().expect("b");
            assert_eq!((8 + hdr_len + a) / 4, off, "{name}: offset drifted");
            assert_eq!((b - a) / 4, len, "{name}: length drifted");
        }
    }

    fn alloc_key(name: &str) -> String {
        format!("\"{name}\":")
    }

    #[test]
    fn embedded_tensors_have_the_right_shapes() {
        let w = embedded();
        assert_eq!(w.window.len(), WINDOW_SIZE);
        assert_eq!(w.window[0], 0.0);
        assert!((w.window[384] - 1.0).abs() < 1e-6);
        let n = embedded_net();
        assert_eq!(n.lstm1_weight_ih.len(), 4 * HIDDEN * 80);
        assert_eq!(n.dense2_weight.len(), 32);
    }
}
