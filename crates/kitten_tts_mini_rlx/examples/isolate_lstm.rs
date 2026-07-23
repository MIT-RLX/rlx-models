//! Isolate the DynamicQuantizeLSTM kernel: feed ORT's exact lstms.0 input to the
//! native LSTM and compare to ORT's lstms.0 LSTM output.

use std::process::Command;

use kitten_tts_mini_rlx::lstm::{dynamic_quantize_lstm, LstmAttrs};
use safetensors::SafeTensors;

fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ort_two(names: [&str; 2]) -> Vec<Vec<f32>> {
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, sys
from onnx import helper, TensorProto
m=onnx.load('.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
existing={{o.name for o in m.graph.output}}
want={names:?}
for n in want:
    if n not in existing:
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
onnx.save(m,'/tmp/kiso.onnx')
v=np.load('.cache/kittentts-mini-0.8/voices.npz'); style=v['expr-voice-2-m'][6:7].astype(np.float32)
ids=np.array([[0,50,83,156,54,57,135,10,0]],dtype=np.int64); speed=np.array([1.0],dtype=np.float32)
s=ort.InferenceSession('/tmp/kiso.onnx',providers=['CPUExecutionProvider'])
outs=s.run(None,{{'input_ids':ids,'style':style,'speed':speed}})
d={{o.name:np.asarray(a) for o,a in zip(s.get_outputs(),outs)}}
for n in want:
    a=d[n].astype(np.float32); sys.stdout.buffer.write(np.int64(a.size).tobytes()); sys.stdout.buffer.write(a.reshape(-1).tobytes())
"#
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .current_dir("/Users/Shared/rlx-models")
        .output()
        .expect("py");
    if !out.status.success() {
        panic!("ort: {}", String::from_utf8_lossy(&out.stderr));
    }
    let mut res = Vec::new();
    let mut off = 0usize;
    let b = &out.stdout;
    for _ in 0..2 {
        let n = i64::from_le_bytes(b[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        res.push(f32_vec(&b[off..off + n * 4]));
        off += n * 4;
    }
    res
}

fn main() {
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("weights/rlx_bundle/weights.safetensors");
    let bytes = std::fs::read(&bundle).expect("bundle");
    let st = SafeTensors::deserialize(&bytes).expect("st");
    let i8v = |n: &str| -> Vec<i8> { st.tensor(n).unwrap().data().iter().map(|&b| b as i8).collect() };
    let f32t = |n: &str| -> Vec<f32> { f32_vec(st.tensor(n).unwrap().data()) };
    let i8_to_i32 = |n: &str| -> Vec<i32> { st.tensor(n).unwrap().data().iter().map(|&b| b as i8 as i32).collect() };

    let w = i8v("onnx::LSTM_6094_quantized"); // [2,640,1024]
    let r = i8v("onnx::LSTM_6095_quantized"); // [2,256,1024]
    let b = f32t("onnx::LSTM_6093"); // [2,2048]
    let w_scale = f32t("onnx::LSTM_6094_scale");
    let w_zp = i8_to_i32("onnx::LSTM_6094_zero_point");
    let r_scale = f32t("onnx::LSTM_6095_scale");
    let r_zp = i8_to_i32("onnx::LSTM_6095_zero_point");

    let got = ort_two([
        "/text_encoder_1/Transpose_3_output_0",
        "/text_encoder/lstms.0/LSTM_output_0",
    ]);
    let x = &got[0]; // [8,1,640]
    let y_ort = &got[1]; // [8,2,1,256]
    println!("x len={} (expect {}), y_ort len={} (expect {})", x.len(), 8 * 640, y_ort.len(), 8 * 2 * 256);

    let seq = 8usize;
    let batch = 1usize;
    let input_size = 640usize;
    let hidden = 256usize;
    let h4 = 4 * hidden;

    let cmp = |y: &[f32]| -> (f32, usize) {
        let n = y.len().min(y_ort.len());
        let mut maxd = 0.0f32;
        let mut idx = 0;
        for j in 0..n {
            let d = (y[j] - y_ort[j]).abs();
            if d > maxd {
                maxd = d;
                idx = j;
            }
        }
        (maxd, idx)
    };

    // Variant A: as-is (native gemv treats W as [h4, input]).
    let mut y = vec![0.0f32; seq * 2 * batch * hidden];
    dynamic_quantize_lstm(
        x,
        Some(&[seq, batch, input_size]),
        &w,
        &r,
        &b,
        &w_scale,
        &w_zp,
        &r_scale,
        &r_zp,
        LstmAttrs { hidden_size: hidden, bidirectional: true },
        &mut y,
    )
    .expect("lstm");
    let (m, i) = cmp(&y);
    println!("A (as-is)           max_abs={m:.5} @idx={i}  y0={:?}", &y[..4]);

    // Variant B: transpose W [dir,input,h4]->[dir,h4,input], R [dir,hidden,h4]->[dir,h4,hidden].
    let transpose_dir = |src: &[i8], rows: usize, cols: usize| -> Vec<i8> {
        // per 2 directions
        let mut out = vec![0i8; src.len()];
        let stride = rows * cols;
        for d in 0..2 {
            let s = &src[d * stride..(d + 1) * stride];
            let o = &mut out[d * stride..(d + 1) * stride];
            for rr in 0..rows {
                for cc in 0..cols {
                    o[cc * rows + rr] = s[rr * cols + cc];
                }
            }
        }
        out
    };
    let wt = transpose_dir(&w, input_size, h4); // now [dir, h4, input]
    let rt = transpose_dir(&r, hidden, h4); // now [dir, h4, hidden]
    let mut y2 = vec![0.0f32; seq * 2 * batch * hidden];
    dynamic_quantize_lstm(
        x,
        Some(&[seq, batch, input_size]),
        &wt,
        &rt,
        &b,
        &w_scale,
        &w_zp,
        &r_scale,
        &r_zp,
        LstmAttrs { hidden_size: hidden, bidirectional: true },
        &mut y2,
    )
    .expect("lstm2");
    let (m2, i2) = cmp(&y2);
    println!("B (W,R transposed)  max_abs={m2:.5} @idx={i2}  y0={:?}", &y2[..4]);
    println!("ort                 y0={:?}", &y_ort[..4]);
}
