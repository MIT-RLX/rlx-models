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

//! Trained weights on disk.
//!
//! ```text
//! magic    "RLXDW004"   8 bytes
//! inputs   u32          input planes the network takes
//! widths   u32 x 4      the four channel widths
//! head     u32          0 direct, 1 kernel-predicting
//! radius   u32          kernel head only, else 0
//! sampling u32          0 strided, 1 pooled
//! guides   u32          0 absolute standard error, 1 relative
//! count    u32          number of parameter tensors
//! per tensor: elems u32, then that many f32
//! ```
//!
//! The architecture is stored, not just the weights, because it decides every
//! tensor's shape: a file trained at one shape cannot be loaded into a network
//! built at another, and reading element counts alone would let that mistake
//! through as a network of the wrong shape rather than as an error.
//!
//! Earlier versions are still read, each declaring what it shipped with:
//! `001` and `002` were nine input planes, `003` eleven.
//!
//! `RLXDW001` and `RLXDW002` are still read. `001` predates the head and
//! sampling choices, so it is direct-prediction with strided resampling; both
//! predate the guide-encoding field, so they are the absolute standard error
//! that version shipped with.
//!
//! The guide encoding changes no tensor's shape, which is exactly why it has to
//! be written down: weights trained against one and run against another load
//! cleanly and are quietly wrong.

use anyhow::{Result, bail, ensure};
use std::path::Path;

use crate::model::{Arch, DenoiseNet, Guides, Head, Sampling, Widths};

const MAGIC: &[u8; 8] = b"RLXDW004";
/// No input-count field: eleven planes, the only shape that version had.
const MAGIC_V3: &[u8; 8] = b"RLXDW003";
/// No guide-encoding field either: nine planes, absolute standard error.
const MAGIC_V2: &[u8; 8] = b"RLXDW002";
/// The original: no architecture fields either, so direct and strided.
const MAGIC_V1: &[u8; 8] = b"RLXDW001";

/// Serialise weights, in `net.params()` order.
pub fn save(net: &DenoiseNet, params: &[Vec<f32>], path: impl AsRef<Path>) -> Result<()> {
    ensure!(
        params.len() == net.params().len(),
        "{} tensors for a network with {}",
        params.len(),
        net.params().len()
    );
    let arch = net.arch();
    let w = arch.widths;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(arch.inputs as u32).to_le_bytes());
    for v in [w.level0, w.level1, w.level2, w.level3] {
        bytes.extend_from_slice(&(v as u32).to_le_bytes());
    }
    let (head, radius) = match arch.head {
        Head::Direct => (0u32, 0u32),
        Head::Kernel { radius } => (1u32, radius as u32),
    };
    let sampling = match arch.sampling {
        Sampling::Strided => 0u32,
        Sampling::Pooled => 1u32,
    };
    for v in [head, radius, sampling, arch.guides.code()] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for (spec, values) in net.params().iter().zip(params) {
        ensure!(
            values.len() == spec.elems(),
            "{}: {} values for a {:?} tensor",
            spec.name,
            values.len(),
            spec.shape
        );
        bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| anyhow::anyhow!("denoise: cannot write {}: {e}", path.as_ref().display()))?;
    Ok(())
}

/// Read weights back, with the network they belong to.
pub fn load(path: impl AsRef<Path>) -> Result<(DenoiseNet, Vec<Vec<f32>>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("denoise: cannot read {}: {e}", path.display()))?;
    ensure!(
        bytes.len() >= 28,
        "weights file is {} bytes, too short",
        bytes.len()
    );
    let v1 = &bytes[..8] == MAGIC_V1;
    let v2 = &bytes[..8] == MAGIC_V2;
    let v3 = &bytes[..8] == MAGIC_V3;
    if &bytes[..8] != MAGIC && !v1 && !v2 && !v3 {
        bail!(
            "weights do not start with {}",
            String::from_utf8_lossy(MAGIC)
        );
    }
    let u32_at = |o: usize| -> Result<u32> { Ok(u32::from_le_bytes(bytes[o..o + 4].try_into()?)) };
    // `004` puts the input count first; everything older had exactly one.
    let (inputs, base) = if v1 || v2 {
        (crate::model::CORE_CHANNELS, 8)
    } else if v3 {
        (11usize, 8)
    } else {
        ensure!(
            bytes.len() >= 12,
            "weights file is truncated before its input count"
        );
        (u32_at(8)? as usize, 12)
    };
    ensure!(inputs > 0, "a network cannot take {inputs} input planes");
    let widths = Widths {
        level0: u32_at(base)? as usize,
        level1: u32_at(base + 4)? as usize,
        level2: u32_at(base + 8)? as usize,
        level3: u32_at(base + 12)? as usize,
    };
    let after_widths = base + 16;
    // `RLXDW001` carries no architecture fields; it predates both choices, so
    // it is direct prediction with strided resampling.
    let (arch, mut off) = if v1 {
        (
            Arch {
                inputs,
                widths,
                head: Head::Direct,
                sampling: Sampling::Strided,
                guides: Guides::AbsoluteError,
            },
            after_widths,
        )
    } else {
        ensure!(
            bytes.len() >= after_widths + 12,
            "weights file is {} bytes, too short for a header",
            bytes.len()
        );
        let head = match u32_at(after_widths)? {
            0 => Head::Direct,
            1 => Head::Kernel {
                radius: u32_at(after_widths + 4)? as usize,
            },
            other => bail!("unknown head {other}"),
        };
        let sampling = match u32_at(after_widths + 8)? {
            0 => Sampling::Strided,
            1 => Sampling::Pooled,
            other => bail!("unknown sampling {other}"),
        };
        // `002` stops after `sampling`; `003` and later add the guide encoding.
        let (guides, header) = if v2 {
            (Guides::AbsoluteError, after_widths + 12)
        } else {
            let at = after_widths + 12;
            ensure!(
                bytes.len() >= at + 8,
                "weights file is truncated before its guides field"
            );
            let code = u32_at(at)?;
            (
                Guides::from_code(code)
                    .ok_or_else(|| anyhow::anyhow!("unknown guide encoding {code}"))?,
                at + 4,
            )
        };
        (
            Arch {
                inputs,
                widths,
                head,
                sampling,
                guides,
            },
            header,
        )
    };
    let count = u32_at(off)? as usize;
    off += 4;
    let net = DenoiseNet::with_arch(arch);
    ensure!(
        count == net.params().len(),
        "file holds {count} tensors, a {arch:?} network has {}",
        net.params().len()
    );

    let mut params = Vec::with_capacity(count);
    for spec in net.params() {
        ensure!(
            off + 4 <= bytes.len(),
            "{}: truncated before its length",
            spec.name
        );
        let elems = u32_at(off)? as usize;
        off += 4;
        ensure!(
            elems == spec.elems(),
            "{}: file holds {elems} values, this network wants {}",
            spec.name,
            spec.elems()
        );
        ensure!(off + elems * 4 <= bytes.len(), "{}: truncated", spec.name);
        let mut values = Vec::with_capacity(elems);
        for chunk in bytes[off..off + elems * 4].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into()?));
        }
        off += elems * 4;
        params.push(values);
    }
    Ok((net, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, Head, Sampling, init_params};

    /// The guide encoding must survive a round trip and be refused when it
    /// disagrees. It changes no tensor shape, so nothing else would catch it.
    #[test]
    fn the_guide_encoding_round_trips_and_older_files_declare_theirs() {
        let arch = Arch {
            widths: Widths::tiny(),
            head: Head::Direct,
            sampling: Sampling::Pooled,
            guides: Guides::RelativeError,
            ..Arch::default()
        };
        let net = DenoiseNet::with_arch(arch);
        let params = init_params(&net, 9);
        let path = std::env::temp_dir().join("rlx_denoise_guides.bin");
        save(&net, &params, &path).expect("save");
        let (back, _) = load(&path).expect("load");
        assert_eq!(back.arch().guides, Guides::RelativeError);

        // Rewritten as `RLXDW002`, the same bytes have to read as the encoding
        // that format shipped with — not as today's default.
        let mut bytes = std::fs::read(&path).expect("read");
        // Rewrite as `003`, which has every field but the input count.
        bytes[..8].copy_from_slice(b"RLXDW003");
        bytes.drain(8..12);
        std::fs::write(&path, &bytes).expect("write");
        let (older, _) = load(&path).expect("load v3");
        assert_eq!(
            older.arch().inputs,
            11,
            "a pre-input-count file must declare the shape it shipped with"
        );
        assert_eq!(older.arch().guides, Guides::RelativeError);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn every_architecture_round_trips() {
        for (i, arch) in [
            Arch {
                widths: Widths::tiny(),
                head: Head::Direct,
                sampling: Sampling::Strided,
                ..Arch::default()
            },
            Arch::optix_like(Widths::tiny()),
            Arch::oidn_like(Widths::tiny()),
        ]
        .into_iter()
        .enumerate()
        {
            let net = DenoiseNet::with_arch(arch);
            let params = init_params(&net, 3);
            let path = std::env::temp_dir().join(format!("rlx_denoise_arch{i}.bin"));
            save(&net, &params, &path).expect("save");
            let (back, values) = load(&path).expect("load");
            assert_eq!(
                back.arch(),
                arch,
                "architecture did not survive the round trip"
            );
            assert_eq!(values, params);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn weights_round_trip() {
        let net = DenoiseNet::new(Widths::tiny());
        let params = init_params(&net, 3);
        let path = std::env::temp_dir().join("rlx_denoise_roundtrip.bin");
        save(&net, &params, &path).expect("save");
        let (back_net, back) = load(&path).expect("load");
        assert_eq!(back_net.widths(), net.widths());
        assert_eq!(back, params);
        let _ = std::fs::remove_file(path);
    }

    /// Weights trained when the network took fewer input planes must not load
    /// into one that takes more. The widths match, the tensor count matches,
    /// and only the first convolution's element count gives it away.
    #[test]
    fn a_checkpoint_from_another_channel_count_is_refused() {
        let net = DenoiseNet::new(Widths::tiny());
        let mut params = init_params(&net, 4);
        // enc0 is [w0, IN_CHANNELS, K, K]; shrink it to what 9 input planes
        // would have produced.
        let w0 = net.widths().level0;
        params[0].truncate(w0 * 9 * 3 * 3);
        let path = std::env::temp_dir().join("rlx_denoise_channels.bin");

        // `save` refuses to write it, which is the first line of defence.
        assert!(save(&net, &params, &path).is_err());

        // And a file that reached disk some other way is refused on load.
        let good = init_params(&net, 4);
        save(&net, &good, &path).expect("save");
        let mut bytes = std::fs::read(&path).expect("read");
        // Find the first tensor's length field by its value rather than by a
        // hardcoded offset — the header has grown twice, and an offset written
        // out by hand breaks silently every time it does.
        let want = net.params()[0].elems() as u32;
        let off = (0..bytes.len() - 4)
            .find(|&i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) == want)
            .expect("the first tensor's length is in the header");
        let shrunk = (w0 * 9 * 3 * 3) as u32;
        bytes[off..off + 4].copy_from_slice(&shrunk.to_le_bytes());
        std::fs::write(&path, &bytes).expect("write");
        let err = load(&path).expect_err("a 9-plane enc0 must not load");
        assert!(
            err.to_string().contains("enc0"),
            "the error should name the tensor: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_from_another_shape_is_rejected() {
        let small = DenoiseNet::new(Widths::tiny());
        let path = std::env::temp_dir().join("rlx_denoise_shape.bin");
        save(&small, &init_params(&small, 1), &path).expect("save");

        // Corrupt one width: the tensor sizes then no longer match, which has
        // to be an error rather than a network quietly built at the wrong size.
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[8..12].copy_from_slice(&64u32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("write");
        assert!(load(&path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
