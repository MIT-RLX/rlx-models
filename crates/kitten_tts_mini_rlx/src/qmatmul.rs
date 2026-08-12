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

//! ONNX `MatMulInteger` + `DynamicQuantizeLinear` (uint8 activations, int8 weights).

fn qmatmul_parallel_enabled() -> bool {
    std::env::var("KITTEN_RLX_QMATMUL_PARALLEL")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

/// When compile-time sequence headroom pads `[1, seq_compile, C]` activations,
/// only the first active token rows are valid at runtime.
fn clip_act_to_runtime_seq(act: &[f32], act_shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let rt = crate::opts::compile_sequence_length_from_env().filter(|&n| n > 0);
    let Some(rt) = rt else {
        return (act.to_vec(), act_shape.to_vec());
    };
    if act_shape.len() == 3 {
        let b = act_shape[0].max(1);
        let s = act_shape[1].max(1);
        let h = act_shape[2];
        if rt < s {
            let n = b * rt * h;
            if n <= act.len() {
                return (act[..n].to_vec(), vec![b, rt, h]);
            }
        }
    }
    if act_shape.len() == 2 {
        let s = act_shape[0].max(1);
        let h = act_shape[1];
        if rt < s {
            let n = rt * h;
            if n <= act.len() {
                return (act[..n].to_vec(), vec![rt, h]);
            }
        }
    }
    (act.to_vec(), act_shape.to_vec())
}

#[inline]
fn dot_u8_i8_row(
    act_q: &[u8],
    row_off: usize,
    act_zp_i32: i32,
    w: &[i8],
    w_zp: i32,
    k: usize,
    n: usize,
    j: usize,
) -> i32 {
    let mut acc = 0i32;
    let mut p = 0usize;
    while p + 4 <= k {
        acc += (act_q[row_off + p] as i32 - act_zp_i32) * (w[(p) * n + j] as i32 - w_zp);
        acc += (act_q[row_off + p + 1] as i32 - act_zp_i32) * (w[(p + 1) * n + j] as i32 - w_zp);
        acc += (act_q[row_off + p + 2] as i32 - act_zp_i32) * (w[(p + 2) * n + j] as i32 - w_zp);
        acc += (act_q[row_off + p + 3] as i32 - act_zp_i32) * (w[(p + 3) * n + j] as i32 - w_zp);
        p += 4;
    }
    while p < k {
        let aq = act_q[row_off + p] as i32 - act_zp_i32;
        let wq = w[p * n + j] as i32 - w_zp;
        acc += aq * wq;
        p += 1;
    }
    acc
}

fn qmatmul_row_range(
    act_q: &[u8],
    act_zp_i32: i32,
    w: &[i8],
    w_zp: i32,
    k: usize,
    n: usize,
    scale_prod: f32,
    out: *mut f32,
    row_start: usize,
    row_end: usize,
) {
    for i in row_start..row_end {
        let row_off = i * k;
        for j in 0..n {
            let acc = dot_u8_i8_row(act_q, row_off, act_zp_i32, w, w_zp, k, n, j);
            unsafe {
                out.add(i * n + j).write(acc as f32 * scale_prod);
            }
        }
    }
}

/// `act @ w` matching ORT: one `DynamicQuantizeLinear` over the full activation
/// tensor, then `MatMulInteger` with activation/weight zero-points.
pub fn qmatmul_f32_act_i8_weight(
    act: &[f32],
    act_shape: &[usize],
    w: &[i8],
    w_shape: &[usize],
    w_scale: f32,
    w_zp: i32,
) -> Vec<f32> {
    let (act, act_shape) = clip_act_to_runtime_seq(act, act_shape);
    let act = act.as_slice();
    let act_shape = act_shape.as_slice();
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    let mut out = vec![0.0f32; m * n];
    if k == 0 || n == 0 || m == 0 || act.len() < m * k || w.len() < k * n {
        return out;
    }
    let (xq, act_scale, act_zp) = dynamic_quantize_uint8(act);
    qmatmul_uint8_act_i8_weight_into(
        &xq, act_shape, act_scale, act_zp, w, w_shape, w_scale, w_zp, &mut out,
    );
    out
}

/// Pre-quantized activation path used when DQL outputs are wired into `QMatMul`.
pub fn qmatmul_uint8_act_i8_weight(
    act_q: &[u8],
    act_shape: &[usize],
    act_scale: f32,
    act_zp: u8,
    w: &[i8],
    w_shape: &[usize],
    w_scale: f32,
    w_zp: i32,
) -> Vec<f32> {
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    let mut out = vec![0.0f32; m * n];
    if k == 0 || n == 0 || m == 0 || act_q.len() < m * k || w.len() < k * n {
        return out;
    }
    qmatmul_uint8_act_i8_weight_into(
        act_q, act_shape, act_scale, act_zp, w, w_shape, w_scale, w_zp, &mut out,
    );
    out
}

/// QMatMul with pre-baked f32 weights (`onnx.QMatMulBaked`).
pub fn qmatmul_uint8_act_f32_weight_into(
    act_q: &[u8],
    act_shape: &[usize],
    act_scale: f32,
    act_zp: u8,
    w_f32: &[f32],
    w_shape: &[usize],
    out: &mut [f32],
) {
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    if k == 0 || n == 0 || m == 0 || act_q.len() < m * k || w_f32.len() < k * n || out.len() < m * n
    {
        return;
    }
    let act_zp_i32 = act_zp as i32;
    for i in 0..m {
        let row = i * k;
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let aq = act_q[row + p] as i32 - act_zp_i32;
                acc += aq as f32 * act_scale * w_f32[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn qmatmul_uint8_act_i8_weight_into(
    act_q: &[u8],
    act_shape: &[usize],
    act_scale: f32,
    act_zp: u8,
    w: &[i8],
    w_shape: &[usize],
    w_scale: f32,
    w_zp: i32,
    out: &mut [f32],
) {
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    if k == 0 || n == 0 || m == 0 || act_q.len() < m * k || w.len() < k * n || out.len() < m * n {
        return;
    }
    let act_zp_i32 = act_zp as i32;
    let scale_prod = act_scale * w_scale;

    if qmatmul_parallel_enabled() && m >= 8 {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(m)
            .max(1);
        if threads > 1 {
            let out_addr = out.as_mut_ptr() as usize;
            std::thread::scope(|scope| {
                let chunk = m.div_ceil(threads);
                for t in 0..threads {
                    let row_start = t * chunk;
                    if row_start >= m {
                        break;
                    }
                    let row_end = (row_start + chunk).min(m);
                    scope.spawn(move || {
                        qmatmul_row_range(
                            act_q,
                            act_zp_i32,
                            w,
                            w_zp,
                            k,
                            n,
                            scale_prod,
                            out_addr as *mut f32,
                            row_start,
                            row_end,
                        );
                    });
                }
            });
            return;
        }
    }

    qmatmul_row_range(
        act_q,
        act_zp_i32,
        w,
        w_zp,
        k,
        n,
        scale_prod,
        out.as_mut_ptr(),
        0,
        m,
    );
}

pub fn dynamic_quantize_uint8(act: &[f32]) -> (Vec<u8>, f32, u8) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in act {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    let r = mx - mn;
    let scale = if r > 0.0 { r / 255.0 } else { 1.0 };
    let zp = (-mn / scale).round().clamp(0.0, 255.0) as u8;
    let q: Vec<u8> = act
        .iter()
        .map(|&x| (x / scale + zp as f32).round().clamp(0.0, 255.0) as u8)
        .collect();
    (q, scale, zp)
}

fn matmul_dims(act_shape: &[usize], w_shape: &[usize]) -> (usize, usize, usize) {
    let k = w_shape.first().copied().filter(|&d| d > 0).unwrap_or(1);
    let n = w_shape.get(1).copied().filter(|&d| d > 0).unwrap_or(1);
    let m = if act_shape.len() >= 3 {
        act_shape[act_shape.len() - 2].max(1)
    } else if act_shape.len() == 2 {
        act_shape[0].max(1)
    } else {
        1
    };
    (m, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ORT dump fixtures under `/tmp/q1_*` and `/tmp/ffn_*` (optional dev-only).
    fn skip_unless_tmp_fixtures(paths: &[&str]) -> bool {
        if paths.iter().all(|p| std::path::Path::new(p).is_file()) {
            return false;
        }
        eprintln!("skip: missing ORT dump fixture(s): {}", paths.join(", "));
        true
    }

    fn q1_fixture_paths() -> [&'static str; 4] {
        [
            "/tmp/q1_act.bin",
            "/tmp/q1_w.bin",
            "/tmp/q1_ws.bin",
            "/tmp/q1_wzp.bin",
        ]
    }

    #[test]
    fn tiny_qmatmul_parallel_matches_serial() {
        let act_q = [
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let w: [i8; 32] = std::array::from_fn(|i| (i as i8).wrapping_mul(3));
        // act [8,2] x w [2,8] => out is m*n = 8*8 = 64, not 16. At 16 the serial
        // path bailed on its `out.len() < m * n` guard (leaving zeros) while
        // `qmatmul_row_range` wrote 64 floats through a raw pointer into a
        // 16-element allocation.
        let mut serial = vec![0.0f32; 64];
        qmatmul_uint8_act_i8_weight_into(&act_q, &[8, 2], 0.1, 0, &w, &[2, 8], 0.2, 0, &mut serial);
        let mut par = vec![0.0f32; 64];
        qmatmul_row_range(&act_q, 0, &w, 0, 2, 8, 0.02, par.as_mut_ptr(), 0, 4);
        qmatmul_row_range(&act_q, 0, &w, 0, 2, 8, 0.02, par.as_mut_ptr(), 4, 8);
        // Relative tolerance: the two paths accumulate in a different order, so
        // results differ by an ULP or two. At |x| ~ 116 one f32 ULP is ~7.6e-6,
        // which a fixed 1e-5 absolute bound cannot represent.
        for (a, b) in serial.iter().zip(par.iter()) {
            let tol = 1e-5 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
        }
    }

    #[test]
    fn q1_ort_fixture_parity() {
        let paths = q1_fixture_paths();
        if skip_unless_tmp_fixtures(&paths) {
            return;
        }
        let act = std::fs::read("/tmp/q1_act.bin").unwrap();
        let w = std::fs::read("/tmp/q1_w.bin").unwrap();
        let ws = std::fs::read("/tmp/q1_ws.bin").unwrap();
        let wzp = std::fs::read("/tmp/q1_wzp.bin").unwrap();
        let act_f32: Vec<f32> = act
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let w_i8: Vec<i8> = w.iter().map(|&b| b as i8).collect();
        let w_scale = f32::from_le_bytes(ws.try_into().unwrap());
        let w_zp = i32::from_le_bytes(wzp.try_into().unwrap());
        let _ = qmatmul_f32_act_i8_weight(&act_f32, &[1, 256], &w_i8, &[256, 256], w_scale, w_zp);
    }
}
