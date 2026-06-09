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

//! Reference bidirectional LSTM for ONNX `com.microsoft::DynamicQuantizeLSTM`.
//!
//! Weight layout matches ONNX Runtime contrib (see `dynamic_quantize_lstm.cc`):
//! - `W`: `[num_directions, input_size, 4 * hidden_size]` (int8)
//! - `R`: `[num_directions, hidden_size, 4 * hidden_size]` (int8)
//! - `B`: `[num_directions, 8 * hidden_size]` (f32)
//! - `X`: `[seq_len, batch, input_size]`
//! - `Y`: `[seq_len, num_directions, batch, hidden_size]`

#[derive(Clone, Copy, Debug)]
pub struct LstmAttrs {
    pub hidden_size: usize,
    pub bidirectional: bool,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn dequant_i8(data: &[i8], scale: f32, zp: i32) -> Vec<f32> {
    data.iter()
        .map(|&q| (q as i32 - zp) as f32 * scale)
        .collect()
}

fn gemv(
    m: usize,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> Result<(), String> {
    if y.len() < m {
        return Err(format!("gemv: y len {} < m {m}", y.len()));
    }
    if x.len() < n {
        return Err(format!("gemv: x len {} < n {n}", x.len()));
    }
    if m > 0 && n > 0 {
        let need = (m - 1) * lda + n;
        if a.len() < need {
            return Err(format!(
                "gemv: a len {} < required {need} (m={m} lda={lda} n={n})",
                a.len()
            ));
        }
    }
    for i in 0..m {
        let mut sum = if beta == 0.0 { 0.0 } else { beta * y[i] };
        for j in 0..n {
            sum += a[i * lda + j] * x[j];
        }
        y[i] = sum;
    }
    Ok(())
}

fn lstm_direction(
    x: &[f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    w: &[f32],
    r: &[f32],
    b: &[f32],
    reverse: bool,
    y_dir: &mut [f32],
) -> Result<(), String> {
    let h4 = hidden_size * 4;
    let mut h = vec![0.0f32; hidden_size];
    let mut c = vec![0.0f32; hidden_size];
    let mut gates = vec![0.0f32; h4];

    for step in 0..seq_len {
        let t = if reverse { seq_len - 1 - step } else { step };
        for batch_idx in 0..batch {
            let x_off = (t * batch + batch_idx) * input_size;
            if x_off + input_size > x.len() {
                return Err(format!(
                    "LSTM: X slice OOB off={x_off} need={input_size} len={}",
                    x.len()
                ));
            }
            let x_t = &x[x_off..x_off + input_size];

            gates.fill(0.0);
            gemv(h4, input_size, w, input_size, x_t, 0.0, &mut gates)?;
            gemv(h4, hidden_size, r, hidden_size, &h, 1.0, &mut gates)?;
            for i in 0..h4 {
                gates[i] += b[i];
            }

            for i in 0..hidden_size {
                gates[i] = sigmoid(gates[i]);
                gates[hidden_size + i] = sigmoid(gates[hidden_size + i]);
                gates[2 * hidden_size + i] = sigmoid(gates[2 * hidden_size + i]);
                gates[3 * hidden_size + i] = gates[3 * hidden_size + i].tanh();
            }

            for i in 0..hidden_size {
                let i_gate = gates[i];
                let o_gate = gates[hidden_size + i];
                let f_gate = gates[2 * hidden_size + i];
                let c_gate = gates[3 * hidden_size + i];
                c[i] = f_gate * c[i] + i_gate * c_gate;
                h[i] = o_gate * c[i].tanh();
            }

            let y_off = (t * batch + batch_idx) * hidden_size;
            if y_off + hidden_size > y_dir.len() {
                return Err(format!(
                    "LSTM: Y slice OOB off={y_off} need={hidden_size} len={}",
                    y_dir.len()
                ));
            }
            y_dir[y_off..y_off + hidden_size].copy_from_slice(&h);
        }
    }
    Ok(())
}

/// Run LSTM with f32 `W`/`R` (already dequantized if needed).
///
/// `x_dims` when `Some` must be ONNX layout `[seq_len, batch, input_size]`.
pub fn dynamic_lstm_f32(
    x: &[f32],
    x_dims: Option<&[usize]>,
    w: &[f32],
    r: &[f32],
    b: &[f32],
    attrs: LstmAttrs,
    y: &mut [f32],
) -> Result<(), String> {
    let h = attrs.hidden_size;
    if h == 0 {
        return Err("hidden_size=0".into());
    }
    let num_dirs = if attrs.bidirectional { 2 } else { 1 };
    let h4 = h * 4;
    let w_dir = w.len() / num_dirs;
    let input_size = w_dir / h4;
    if input_size == 0 || w_dir != input_size * h4 {
        return Err(format!("bad W len {} for h={h}", w.len()));
    }
    let (seq_len, batch) = if let Some(d) = x_dims {
        if d.len() != 3 {
            return Err(format!("LSTM X rank {} != 3", d.len()));
        }
        if d[2] != input_size {
            return Err(format!(
                "LSTM X input dim {} != weight input_size {input_size}",
                d[2]
            ));
        }
        let need = d[0] * d[1] * d[2];
        if x.len() < need {
            return Err(format!(
                "X len {} < active {}*{}*{input_size}",
                x.len(),
                d[0],
                d[1]
            ));
        }
        (d[0], d[1])
    } else {
        let batch = x.len() / input_size;
        if batch == 0 {
            return Err("empty X".into());
        }
        let seq_len = x.len() / (batch * input_size);
        if seq_len * batch * input_size != x.len() {
            return Err(format!(
                "X len {} != seq*batch*input (seq={seq_len} batch={batch} in={input_size})",
                x.len()
            ));
        }
        (seq_len, batch)
    };
    let need_y = seq_len * num_dirs * batch * h;
    if y.len() < need_y {
        return Err(format!(
            "Y len {} < active seq*dirs*batch*h ({seq_len}*{num_dirs}*{batch}*{h}), x_dims={x_dims:?}, x_len={}",
            y.len(),
            x.len()
        ));
    }

    let w_stride = input_size * h4;
    let r_stride = h * h4;
    let b_stride = 8 * h;

    for dir in 0..num_dirs {
        let w_slice = &w[dir * w_stride..(dir + 1) * w_stride];
        let r_slice = &r[dir * r_stride..(dir + 1) * r_stride];
        let b_d = &b[dir * b_stride..(dir + 1) * b_stride];

        let mut y_tmp = vec![0.0f32; seq_len * batch * h];
        lstm_direction(
            x,
            seq_len,
            batch,
            input_size,
            h,
            w_slice,
            r_slice,
            b_d,
            dir == 1,
            &mut y_tmp,
        )?;
        for t in 0..seq_len {
            for bidx in 0..batch {
                for hi in 0..h {
                    let src = (t * batch + bidx) * h + hi;
                    let dst = t * (num_dirs * batch * h) + dir * (batch * h) + bidx * h + hi;
                    y[dst] = y_tmp[src];
                }
            }
        }
    }
    // Static compile allocates Y for max seq; zero tail so downstream MatMul
    // does not read stale cells when runtime seq < compiled seq.
    let total_t = y.len() / (num_dirs * batch * h).max(1);
    for t in seq_len..total_t {
        for dir in 0..num_dirs {
            for bidx in 0..batch {
                for hi in 0..h {
                    let dst = t * (num_dirs * batch * h) + dir * (batch * h) + bidx * h + hi;
                    y[dst] = 0.0;
                }
            }
        }
    }
    Ok(())
}

/// Run DynamicQuantizeLSTM (int8 `W`/`R` with per-direction scale/zp).
pub fn dynamic_quantize_lstm(
    x: &[f32],
    x_dims: Option<&[usize]>,
    w_i8: &[i8],
    r_i8: &[i8],
    b: &[f32],
    w_scale: &[f32],
    w_zp: &[i32],
    r_scale: &[f32],
    r_zp: &[i32],
    attrs: LstmAttrs,
    y: &mut [f32],
) -> Result<(), String> {
    let h = attrs.hidden_size;
    if h == 0 {
        return Err("hidden_size=0".into());
    }
    let num_dirs = if attrs.bidirectional { 2 } else { 1 };
    let h4 = h * 4;
    let w_dir = w_i8.len() / num_dirs;
    let input_size = w_dir / h4;
    if input_size == 0 || w_dir != input_size * h4 {
        return Err(format!("bad W len {} for h={h}", w_i8.len()));
    }

    let w_stride = input_size * h4;
    let r_stride = h * h4;

    let mut w_all = Vec::with_capacity(w_i8.len());
    let mut r_all = Vec::with_capacity(r_i8.len());
    for dir in 0..num_dirs {
        let w_scale_d = w_scale.get(dir).copied().unwrap_or(w_scale[0]);
        let w_zp_d = w_zp.get(dir).copied().unwrap_or(w_zp[0]);
        let r_scale_d = r_scale.get(dir).copied().unwrap_or(r_scale[0]);
        let r_zp_d = r_zp.get(dir).copied().unwrap_or(r_zp[0]);
        w_all.extend(dequant_i8(
            &w_i8[dir * w_stride..(dir + 1) * w_stride],
            w_scale_d,
            w_zp_d,
        ));
        r_all.extend(dequant_i8(
            &r_i8[dir * r_stride..(dir + 1) * r_stride],
            r_scale_d,
            r_zp_d,
        ));
    }
    dynamic_lstm_f32(x, x_dims, &w_all, &r_all, b, attrs, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::load_weights;

    #[test]
    fn predictor_lstm_produces_nonzero_output_on_ones_input() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
        if !dir.join("model.safetensors").is_file() {
            return;
        }
        let bundle = dir.join("rlx_bundle/weights.safetensors");
        if !bundle.is_file() {
            return;
        }
        let w = load_weights(&dir).expect("weights");
        let w_i8 = load_i8_tensor(&bundle, "onnx::LSTM_6243_quantized");
        let r_i8 = load_i8_tensor(&bundle, "onnx::LSTM_6244_quantized");
        let b = w
            .f32
            .get("onnx::LSTM_6242")
            .map(|(v, _)| v.as_slice())
            .expect("onnx::LSTM_6242");
        let w_scale: Vec<f32> = w
            .f32
            .get("onnx::LSTM_6243_scale")
            .map(|(v, _)| v.clone())
            .expect("scale");
        let w_zp: Vec<i32> = bundle_zp_i32(&dir, "onnx::LSTM_6243_zero_point");
        let r_scale: Vec<f32> = w
            .f32
            .get("onnx::LSTM_6244_scale")
            .map(|(v, _)| v.clone())
            .expect("r_scale");
        let r_zp: Vec<i32> = bundle_zp_i32(&dir, "onnx::LSTM_6244_zero_point");

        let seq = 128usize;
        let batch = 1usize;
        let input_size = 640usize;
        let x = vec![1.0f32; seq * batch * input_size];
        let mut y = vec![0.0f32; seq * 2 * batch * 256];
        let attrs = LstmAttrs {
            hidden_size: 256,
            bidirectional: true,
        };
        dynamic_quantize_lstm(
            &x,
            Some(&[seq, batch, input_size]),
            &w_i8,
            &r_i8,
            b,
            &w_scale,
            &w_zp,
            &r_scale,
            &r_zp,
            attrs,
            &mut y,
        )
        .expect("lstm");
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-4,
            "predictor LSTM output flat (max_abs={max_abs}); check dequant/layout"
        );
    }

    fn load_i8_tensor(path: &std::path::Path, name: &str) -> Vec<i8> {
        let bytes = std::fs::read(path).expect("read weights");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("st");
        let t = st.tensor(name).expect("tensor");
        t.data().iter().map(|&b| b as i8).collect()
    }

    fn bundle_zp_i32(dir: &std::path::Path, name: &str) -> Vec<i32> {
        let path = dir.join("rlx_bundle/weights.safetensors");
        let bytes = std::fs::read(path).expect("bundle weights");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("st");
        let t = st.tensor(name).expect("zp");
        match t.dtype() {
            safetensors::tensor::Dtype::I8 => t.data().iter().map(|&b| b as i8 as i32).collect(),
            safetensors::tensor::Dtype::I64 => t
                .data()
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            safetensors::tensor::Dtype::F32 => t
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()).round() as i32)
                .collect(),
            _ => vec![0],
        }
    }
}
