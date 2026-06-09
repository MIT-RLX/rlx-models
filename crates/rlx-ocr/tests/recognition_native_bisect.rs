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

//! Bisect native recognition vs safetensors reference (env-gated).

#![cfg(feature = "rlx")]

#[path = "env.rs"]
mod bench_env;

use anyhow::Result;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_core::vision_ops_ir::{conv2d_bias, max_pool2d_2x2};
use rlx_flow::CompileProfile;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_ocr::model::weights::OcrGraphBuilder;
use rlx_ocr::model::{
    RecognitionGraphConfig, build_recognition_after_g1_graph, build_recognition_after_g2_graph,
    build_recognition_after_logits_graph, build_recognition_conv_graph,
};
use rlx_ocr::weights::SafetensorsFile;
use rlx_runtime::{Device, Session};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::path::PathBuf;

fn max_abs_diff_4(a: NdTensorView<f32, 4>, b: NdTensorView<f32, 4>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn native_conv0_matches_safetensors_reference() -> Result<()> {
    let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skip native_conv0_matches_safetensors_reference: set OCR_MODEL_DIR");
        return Ok(());
    };
    let (_, rec_st) = rlx_ocr::resolve_model_dir(&dir)?;
    let h = 64usize;
    let w = 200usize;
    let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);

    let py = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import struct, sys, numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open
p = sys.argv[1]
with safe_open(p, framework="numpy") as f:
    wt = torch.from_numpy(f.get_tensor("conv.0.weight"))
    bt = torch.from_numpy(f.get_tensor("conv.0.bias"))
    x = torch.full((1,1,64,200), 0.82)
    y = F.relu(F.max_pool2d(F.conv2d(x, wt, bt, padding=1), 2))
z = y.detach()
sys.stdout.buffer.write(struct.pack('4I', *z.shape))
sys.stdout.buffer.write(z.numpy().astype('float32').tobytes())
"#,
        )
        .arg(rec_st.to_str().unwrap())
        .output()?;
    if !py.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&py.stderr));
    }
    let bytes = &py.stdout;
    let mut dims = [0u32; 4];
    for (i, d) in dims.iter_mut().enumerate() {
        *d = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?);
    }
    let shape: [usize; 4] = dims.map(|x| x as usize);
    let ref_t = NdTensor::from_data(
        shape,
        bytes[16..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>(),
    );

    let mut wm = SafetensorsFile::open(&rec_st)?.weight_map()?;
    let mut b = OcrGraphBuilder::new("bisect_conv0");
    let batch = 1usize;
    let image = b
        .m()
        .input("image", Shape::new(&[batch, 1, h, w], DType::F32));
    let weight = b.load_param(&mut wm, "conv.0.weight")?;
    let bias = b.load_param(&mut wm, "conv.0.bias")?;
    let mut g = b.m();
    let y = conv2d_bias(
        &mut g,
        image,
        weight,
        bias,
        batch,
        32,
        3,
        3,
        [1, 1],
        [1, 1],
        h,
        w,
    );
    let y = g.relu(y);
    let y = max_pool2d_2x2(&mut g, y, batch, 32, h, w);
    g.set_outputs(vec![y]);
    let (graph, params) = b.finish()?;
    let opts = compile_options_for_profile(&CompileProfile::encoder(), Device::Cpu);
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    let flat = compiled
        .run(&[(
            "image",
            input.iter().copied().collect::<Vec<_>>().as_slice(),
        )])
        .into_iter()
        .next()
        .unwrap();
    let rlx_t = NdTensor::from_data(shape, flat);
    let err = max_abs_diff_4(ref_t.view(), rlx_t.view());
    eprintln!("conv0 max abs diff {err}");
    assert!(err <= 1e-4, "conv0 max abs diff {err}");
    Ok(())
}

#[test]
fn native_conv_full_matches_safetensors_reference() -> Result<()> {
    let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skip native_conv_full_matches_safetensors_reference: set OCR_MODEL_DIR");
        return Ok(());
    };
    let (_, rec_st) = rlx_ocr::resolve_model_dir(&dir)?;
    let h = 64usize;
    let w = 200usize;
    let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);

    let py = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import struct, sys, numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open
p = sys.argv[1]
with safe_open(p, framework="numpy") as f:
    tensors = {k: torch.from_numpy(f.get_tensor(k)) for k in f.keys()}
W = tensors.get
x = torch.full((1,1,64,200), 0.82)
y = F.relu(F.max_pool2d(F.conv2d(x, W('conv.0.weight'), W('conv.0.bias'), padding=1), 2))
y = F.relu(F.max_pool2d(F.conv2d(y, W('onnx::Conv_367'), W('onnx::Conv_368'), padding=1), 2))
y = F.relu(F.conv2d(y, W('conv.7.weight'), W('conv.7.bias'), padding=1))
y = F.relu(F.max_pool2d(F.conv2d(y, W('onnx::Conv_370'), W('onnx::Conv_371'), padding=1), (2,1), stride=(2,1)))
y = F.relu(F.conv2d(y, W('conv.13.weight'), W('conv.13.bias'), padding=1))
y = F.relu(F.max_pool2d(F.conv2d(y, W('onnx::Conv_373'), W('onnx::Conv_374'), padding=1), (2,1), stride=(2,1)))
y = F.conv2d(y, W('onnx::Conv_376'), W('onnx::Conv_377'), stride=(1,1), padding=(1,1))
y = F.avg_pool2d(y, kernel_size=(4,1), stride=(4,1))
seq = y.shape[3]
y = y.reshape(1, 128, seq).permute(2, 0, 1)
z = y.detach()
sys.stdout.buffer.write(struct.pack('3I', *z.shape))
sys.stdout.buffer.write(z.numpy().astype('float32').tobytes())
"#,
        )
        .arg(rec_st.to_str().unwrap())
        .output()?;
    if !py.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&py.stderr));
    }
    let bytes = &py.stdout;
    let mut dims = [0u32; 3];
    for (i, d) in dims.iter_mut().enumerate() {
        *d = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?);
    }
    let shape: [usize; 3] = dims.map(|x| x as usize);
    let ref_t = NdTensor::from_data(
        shape,
        bytes[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>(),
    );

    let mut wm = SafetensorsFile::open(&rec_st)?.weight_map()?;
    let (graph, params) =
        build_recognition_conv_graph(&mut wm, RecognitionGraphConfig { batch: 1, width: w })?;
    let opts = compile_options_for_profile(&CompileProfile::encoder(), Device::Cpu);
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    let flat = compiled
        .run(&[(
            "image",
            input.iter().copied().collect::<Vec<_>>().as_slice(),
        )])
        .into_iter()
        .next()
        .unwrap();
    let rlx_t = NdTensor::from_data(shape, flat);
    let err = ref_t
        .iter()
        .zip(rlx_t.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("conv_full max abs diff {err} ref_shape={shape:?}");
    assert!(err <= 1e-3, "conv_full max abs diff {err}");
    Ok(())
}

/// Compare RTen reference logits vs ONNX Runtime (same `.onnx` topology).
#[test]
fn rten_logits_match_onnxruntime_reference() -> Result<()> {
    let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skip rten_logits_match_onnxruntime_reference");
        return Ok(());
    };
    let rec_rten = dir.join(rlx_ocr::weights::HF_RECOGNITION_RTEN);
    let onnx = "/tmp/ocr-rec.onnx";
    if !std::path::Path::new(onnx).is_file() {
        eprintln!("skip rten_logits_match_onnxruntime_reference: missing {onnx}");
        return Ok(());
    }
    let h = 64usize;
    let w = 200usize;
    let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
    let ref_rec = rlx_ocr::inference::RtenTextRecognizer::from_path(&rec_rten)?;
    let ref_out = ref_rec.run(input.clone())?;

    let py = std::process::Command::new("/tmp/ocr-onnx-venv/bin/python")
        .arg("-c")
        .arg(
            r#"
import struct, sys, numpy as np, onnxruntime as ort
x = np.full((1,1,64,200), 0.82, dtype=np.float32)
y = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'line_image': x})[0]
y = np.transpose(y, (1,0,2))
sys.stdout.buffer.write(struct.pack('3I', *y.shape))
sys.stdout.buffer.write(y.astype('float32').tobytes())
"#,
        )
        .arg(onnx)
        .output()?;
    if !py.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&py.stderr));
    }
    let bytes = &py.stdout;
    let shape: [usize; 3] = std::array::from_fn(|i| {
        u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize
    });
    let ort_t = NdTensor::from_data(
        shape,
        bytes[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>(),
    );
    assert_eq!(ref_out.shape(), ort_t.shape());
    let err = ref_out
        .iter()
        .zip(ort_t.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("rten vs onnxruntime max abs diff {err}");
    assert!(err <= 1e-4, "rten vs onnxruntime {err}");
    Ok(())
}

fn run_rlx_recognition_stage<const N: usize>(
    rec_st: &std::path::Path,
    w: usize,
    out_shape: [usize; N],
    build: fn(
        &mut rlx_core::weight_map::WeightMap,
        RecognitionGraphConfig,
    ) -> anyhow::Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)>,
) -> Result<NdTensor<f32, N>> {
    let h = 64usize;
    let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
    let mut wm = SafetensorsFile::open(rec_st)?.weight_map()?;
    let (graph, params) = build(&mut wm, RecognitionGraphConfig { batch: 1, width: w })?;
    let opts = compile_options_for_profile(&CompileProfile::encoder(), Device::Cpu);
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    let flat = compiled
        .run(&[(
            "image",
            input.iter().copied().collect::<Vec<_>>().as_slice(),
        )])
        .into_iter()
        .next()
        .unwrap();
    Ok(NdTensor::from_data(out_shape, flat))
}

fn ort_intermediates(onnx: &str) -> Result<std::collections::HashMap<String, NdTensor<f32, 3>>> {
    let py = std::process::Command::new("/tmp/ocr-onnx-venv/bin/python")
        .arg("-c")
        .arg(
            r#"
import struct, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper
path, tag = sys.argv[1], sys.argv[2]
model = onnx.load(path)
for n in ['chars','/Reshape_output_0','/gru/Reshape_output_0','/gru/Reshape_1_output_0','/output/output.0/Add_output_0']:
    if n not in {o.name for o in model.graph.output}:
        model.graph.output.append(helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
x = np.full((1,1,64,200), 0.82, np.float32)
outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {'line_image': x})))
for name in ['chars','/Reshape_output_0','/gru/Reshape_output_0','/gru/Reshape_1_output_0','/output/output.0/Add_output_0']:
    y = outs[name].astype('float32')
    sys.stdout.buffer.write(name.encode() + b'\0')
    sys.stdout.buffer.write(struct.pack('3I', *y.shape))
    sys.stdout.buffer.write(y.tobytes())
"#,
        )
        .arg(onnx)
        .arg("x")
        .output()?;
    if !py.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&py.stderr));
    }
    let mut out = std::collections::HashMap::new();
    let mut pos = 0usize;
    let bytes = py.stdout.as_slice();
    while pos < bytes.len() {
        let end = bytes[pos..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| anyhow::anyhow!("truncated ort intermediate payload"))?
            + pos;
        let name = std::str::from_utf8(&bytes[pos..end])?.to_string();
        pos = end + 1;
        let shape: [usize; 3] = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[pos + i * 4..pos + i * 4 + 4].try_into().unwrap()) as usize
        });
        pos += 12;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = bytes[pos..pos + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        pos += n * 4;
        out.insert(name, NdTensor::from_data(shape, data));
    }
    Ok(out)
}

#[test]
fn native_gru_stages_match_onnxruntime_reference() -> Result<()> {
    let Some(dir) = bench_env::env_var("OCR_MODEL_DIR", "OCRS_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skip native_gru_stages_match_onnxruntime_reference");
        return Ok(());
    };
    let onnx = "/tmp/ocr-rec.onnx";
    if !std::path::Path::new(onnx).is_file() {
        eprintln!("skip native_gru_stages: missing {onnx}");
        return Ok(());
    }
    let (_, rec_st) = rlx_ocr::resolve_model_dir(&dir)?;
    let w = 200usize;
    let ort = ort_intermediates(onnx)?;
    let rlx_conv = {
        let h = 64usize;
        let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
        let mut wm = SafetensorsFile::open(&rec_st)?.weight_map()?;
        let (graph, params) =
            build_recognition_conv_graph(&mut wm, RecognitionGraphConfig { batch: 1, width: w })?;
        let opts = compile_options_for_profile(&CompileProfile::encoder(), Device::Cpu);
        let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &[]);
        let flat = compiled
            .run(&[(
                "image",
                input.iter().copied().collect::<Vec<_>>().as_slice(),
            )])
            .into_iter()
            .next()
            .unwrap();
        NdTensor::from_data([51, 1, 128], flat)
    };
    let conv_err = rlx_conv
        .iter()
        .zip(ort["/Reshape_output_0"].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("conv vs ort /Reshape_output_0 max abs diff {conv_err}");

    let rlx_g1 =
        run_rlx_recognition_stage(&rec_st, w, [51, 1, 512], build_recognition_after_g1_graph)?;
    let rlx_g2 =
        run_rlx_recognition_stage(&rec_st, w, [51, 1, 512], build_recognition_after_g2_graph)?;
    let rlx_logits = run_rlx_recognition_stage(
        &rec_st,
        w,
        [51, 1, 97],
        build_recognition_after_logits_graph,
    )?;
    for (key, rlx, ort_key) in [
        ("after_g1", &rlx_g1, "/gru/Reshape_output_0"),
        ("after_g2", &rlx_g2, "/gru/Reshape_1_output_0"),
        ("after_logits", &rlx_logits, "/output/output.0/Add_output_0"),
    ] {
        let o = &ort[ort_key];
        assert_eq!(rlx.shape(), o.shape());
        let err = rlx
            .iter()
            .zip(o.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{key} max abs diff {err}");
        assert!(err <= 1e-3, "{key} max abs diff {err}");
    }

    let rlx_full = {
        use rlx_ocr::rlx::RlxTextRecognizer;
        let rec = RlxTextRecognizer::from_model_dir(&dir, Device::Cpu)?;
        let h = 64usize;
        let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
        rec.run_batch_logits(input)?
    };
    let rten_full = {
        use rlx_ocr::inference::RtenTextRecognizer;
        let rec_rten = dir.join(rlx_ocr::weights::HF_RECOGNITION_RTEN);
        let rec = RtenTextRecognizer::from_path(&rec_rten)?;
        let h = 64usize;
        let input = NdTensor::from_data([1, 1, h, w], vec![0.82f32; h * w]);
        rec.run(input)?
    };
    let err_rten = rlx_full
        .iter()
        .zip(rten_full.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("rlx vs rten log_probs max abs diff {err_rten}");
    let ort_chars = ort
        .get("chars")
        .ok_or_else(|| anyhow::anyhow!("missing ort chars output"))?;
    let mut rlx_sbc = rlx_full.clone();
    rlx_sbc.permute([1, 0, 2]);
    let err_full = rlx_sbc
        .iter()
        .zip(ort_chars.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("rlx log_probs vs ort chars max abs diff {err_full}");
    Ok(())
}
