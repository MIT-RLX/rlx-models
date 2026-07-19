//! Decode HF-exported codes with rlx-mimi and write a WAV.
use anyhow::Result;
use rlx_mimi::{MimiCodec, MimiCodes};
use rlx_runtime::Device;
use rlx_sesame::session::write_wav;

fn main() -> Result<()> {
    let npy = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/hf_fox_codes.npy".into());
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/mimi_from_hf_codes.wav".into());
    let mimi_dir = std::env::args()
        .nth(3)
        .unwrap_or_else(|| ".cache/mimi".into());

    // Minimal npy reader for float64/int64 2d — HF saved int64? actually int from numpy default int
    let bytes = std::fs::read(&npy)?;
    let frames = parse_npy_i64_2d(&bytes)?;
    eprintln!(
        "frames={} codebooks={}",
        frames.len(),
        frames.first().map(|r| r.len()).unwrap_or(0)
    );
    eprintln!("frame0={:?}", &frames[0][..8.min(frames[0].len())]);

    let mimi = MimiCodec::open_on(&mimi_dir, Device::Cpu)?;
    let codes = MimiCodes {
        num_quantizers: frames[0].len(),
        frames: frames
            .iter()
            .map(|r| r.iter().map(|&x| x as u32).collect())
            .collect(),
    };
    let pcm = mimi.decode_codes(&codes)?;
    write_wav(&out, &pcm, 24_000)?;
    eprintln!("wrote {out} samples={}", pcm.len());
    Ok(())
}

fn parse_npy_i64_2d(bytes: &[u8]) -> Result<Vec<Vec<i64>>> {
    anyhow::ensure!(bytes.starts_with(b"\x93NUMPY"), "not npy");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])?;
    // descr '<i8' or '<i4'
    let data = &bytes[10 + header_len..];
    let shape = parse_shape(header)?;
    anyhow::ensure!(shape.len() == 2, "expected 2d got {shape:?}");
    let (rows, cols) = (shape[0], shape[1]);
    let is_i8 = header.contains("'<i8'") || header.contains("\"<i8\"");
    let is_i4 = header.contains("'<i4'") || header.contains("\"<i4\"");
    let mut frames = Vec::with_capacity(rows);
    if is_i8 {
        anyhow::ensure!(data.len() >= rows * cols * 8, "short i8 data");
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                let o = (r * cols + c) * 8;
                row.push(i64::from_le_bytes(data[o..o + 8].try_into()?));
            }
            frames.push(row);
        }
    } else if is_i4 {
        anyhow::ensure!(data.len() >= rows * cols * 4, "short i4 data");
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                let o = (r * cols + c) * 4;
                row.push(i32::from_le_bytes(data[o..o + 4].try_into()?) as i64);
            }
            frames.push(row);
        }
    } else {
        anyhow::bail!("unsupported npy dtype in {header}");
    }
    Ok(frames)
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let start = header
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("no shape"))?;
    let end = header
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("no shape end"))?;
    let inner = &header[start + 1..end];
    Ok(inner
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() { None } else { t.parse().ok() }
        })
        .collect())
}
