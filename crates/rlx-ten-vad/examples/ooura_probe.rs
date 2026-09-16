//! Dev tool: run this crate's Ooura transform on a raw f32 file and dump both
//! the format2 spectrum and the rescaled format1 one, for A/B against the C.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let x: Vec<f32> = std::fs::read(&a[0])?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut spec = vec![0f32; 1024];
    rlx_ten_vad::ooura::r2c(&x, &mut spec);
    let mut out: Vec<u8> = spec.iter().flat_map(|v| v.to_le_bytes()).collect();
    rlx_ten_vad::ooura::inplace_transf(1, &mut spec);
    rlx_ten_vad::ooura::rescale_fft_out(&mut spec);
    out.extend(spec.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(&a[1], out)?;
    Ok(())
}
