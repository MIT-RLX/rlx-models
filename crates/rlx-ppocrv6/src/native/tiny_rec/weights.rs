use std::collections::HashMap;
use std::path::Path;

/// Loaded initializer tensors from `model.safetensors`.
pub struct LoadedWeights {
    pub f32: HashMap<String, (Vec<f32>, Vec<usize>)>,
    pub i64: HashMap<String, (Vec<i64>, Vec<usize>)>,
}

fn scale_for<'a>(f32: &'a HashMap<String, (Vec<f32>, Vec<usize>)>, name: &str) -> Option<&'a f32> {
    f32.get(&format!("{name}_scale"))
        .or_else(|| {
            name.strip_suffix("_quantized")
                .and_then(|base| f32.get(&format!("{base}_scale")))
        })
        .and_then(|(v, _)| v.first())
}

fn zp_for(f32: &HashMap<String, (Vec<f32>, Vec<usize>)>, name: &str) -> f32 {
    f32.get(&format!("{name}_zero_point"))
        .or_else(|| {
            name.strip_suffix("_quantized")
                .and_then(|base| f32.get(&format!("{base}_zero_point")))
        })
        .and_then(|(v, _)| v.first().copied())
        .unwrap_or(0.0)
}

/// Load decomposed weights from `dir` (`model.safetensors`).
pub fn load_weights(dir: &Path) -> anyhow::Result<LoadedWeights> {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut f32 = HashMap::new();
    let mut i64 = HashMap::new();
    let mut pending_quant: Vec<(String, Vec<usize>, Vec<u8>, safetensors::tensor::Dtype)> =
        Vec::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().iter().copied().collect();
        match view.dtype() {
            safetensors::tensor::Dtype::F32 => {
                let mut data = Vec::with_capacity(view.data().len() / 4);
                for chunk in view.data().chunks_exact(4) {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                f32.insert(name.to_string(), (data, shape));
            }
            safetensors::tensor::Dtype::I64 => {
                let mut data = Vec::with_capacity(view.data().len() / 8);
                for chunk in view.data().chunks_exact(8) {
                    data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                }
                i64.insert(name.to_string(), (data, shape));
            }
            safetensors::tensor::Dtype::F16 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                f32.insert(name.to_string(), (data, shape));
            }
            safetensors::tensor::Dtype::BF16 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                f32.insert(name.to_string(), (data, shape));
            }
            safetensors::tensor::Dtype::U8 | safetensors::tensor::Dtype::I8 => {
                pending_quant.push((name.to_string(), shape, view.data().to_vec(), view.dtype()));
            }
            _ => {}
        }
    }
    for (name, shape, raw, dt) in pending_quant {
        let scale = scale_for(&f32, &name).copied().unwrap_or(1.0);
        let zp = zp_for(&f32, &name);
        let data: Vec<f32> = match dt {
            safetensors::tensor::Dtype::U8 => raw.iter().map(|&b| (b as f32 - zp) * scale).collect(),
            _ => raw.iter().map(|&b| (b as i8 as f32 - zp) * scale).collect(),
        };
        let out_name = name.strip_suffix("_quantized").unwrap_or(&name).to_string();
        if !f32.contains_key(&out_name) {
            f32.insert(out_name.clone(), (data.clone(), shape.clone()));
        }
        if name.ends_with("_quantized") && !f32.contains_key(name.as_str()) {
            f32.insert(name.clone(), (data.clone(), shape.clone()));
        }
        let import_name = format!("{name}_f32_import");
        if !f32.contains_key(&import_name) {
            f32.insert(import_name, (data, shape));
        }
    }
    Ok(LoadedWeights { f32, i64 })
}

/// ONNX initializer names in this model (140 tensors).
pub const PARAM_NAMES: &[&str] = &[
    "\"HardSigmoid.0_alpha_0\"",
    "\"HardSigmoid.0_beta_0\"",
    "\"HardSigmoid.0_one_0\"",
    "\"HardSigmoid.0_zero_0\"",
    "\"HardSigmoid.1_alpha_0\"",
    "\"HardSigmoid.1_beta_0\"",
    "\"HardSigmoid.1_one_0\"",
    "\"HardSigmoid.1_zero_0\"",
    "\"HardSigmoid.2_alpha_0\"",
    "\"HardSigmoid.2_beta_0\"",
    "\"HardSigmoid.2_one_0\"",
    "\"HardSigmoid.2_zero_0\"",
    "\"HardSigmoid.3_alpha_0\"",
    "\"HardSigmoid.3_beta_0\"",
    "\"HardSigmoid.3_one_0\"",
    "\"HardSigmoid.3_zero_0\"",
    "\"HardSigmoid.5_alpha_0\"",
    "\"HardSigmoid.5_beta_0\"",
    "\"HardSigmoid.5_one_0\"",
    "\"HardSigmoid.5_zero_0\"",
    "\"batch_norm1d_0.b_0\"",
    "\"batch_norm1d_0.w_0\"",
    "\"batch_norm1d_0.w_1\"",
    "\"batch_norm1d_0.w_2\"",
    "\"batch_norm1d_1.b_0\"",
    "\"batch_norm1d_1.w_0\"",
    "\"batch_norm1d_1.w_1\"",
    "\"batch_norm1d_1.w_2\"",
    "\"batch_norm2d_0.b_0\"",
    "\"batch_norm2d_0.w_0\"",
    "\"batch_norm2d_0.w_1\"",
    "\"batch_norm2d_0.w_2\"",
    "\"batch_norm2d_1.b_0\"",
    "\"batch_norm2d_1.w_0\"",
    "\"batch_norm2d_1.w_1\"",
    "\"batch_norm2d_1.w_2\"",
    "\"conv2d_0.w_0\"",
    "\"conv2d_1.w_0\"",
    "\"conv2d_17.w_0\"",
    "\"conv2d_18.w_0\"",
    "\"conv2d_30.w_0\"",
    "\"conv2d_31.w_0\"",
    "\"conv2d_4.w_0\"",
    "\"conv2d_43.w_0\"",
    "\"conv2d_44.w_0\"",
    "\"conv2d_45.w_0\"",
    "\"conv2d_47.w_0\"",
    "\"conv2d_48.w_0\"",
    "\"conv2d_49.w_0\"",
    "\"conv2d_5.w_0\"",
    "\"conv2d_50.w_0\"",
    "\"conv2d_51.w_0\"",
    "\"conv2d_52.w_0\"",
    "\"conv2d_54.w_0\"",
    "\"conv2d_55.w_0\"",
    "\"conv2d_56.w_0\"",
    "\"conv2d_58.w_0\"",
    "\"conv2d_59.w_0\"",
    "\"conv2d_60.w_0\"",
    "\"conv2d_61.w_0\"",
    "\"conv2d_62.w_0\"",
    "\"conv2d_63.w_0\"",
    "\"conv2d_65.w_0\"",
    "\"conv2d_66.w_0\"",
    "\"conv2d_67.w_0\"",
    "\"conv2d_69.w_0\"",
    "\"conv2d_70.w_0\"",
    "\"conv2d_71.w_0\"",
    "\"conv2d_73.w_0\"",
    "\"conv2d_74.w_0\"",
    "\"conv2d_75.w_0\"",
    "\"helper.constant.0\"",
    "\"helper.constant.1\"",
    "\"helper.constant.10\"",
    "\"helper.constant.11\"",
    "\"helper.constant.12\"",
    "\"helper.constant.13\"",
    "\"helper.constant.14\"",
    "\"helper.constant.15\"",
    "\"helper.constant.16\"",
    "\"helper.constant.17\"",
    "\"helper.constant.18\"",
    "\"helper.constant.19\"",
    "\"helper.constant.2\"",
    "\"helper.constant.20\"",
    "\"helper.constant.21\"",
    "\"helper.constant.22\"",
    "\"helper.constant.23\"",
    "\"helper.constant.24\"",
    "\"helper.constant.25\"",
    "\"helper.constant.26\"",
    "\"helper.constant.27\"",
    "\"helper.constant.28\"",
    "\"helper.constant.29\"",
    "\"helper.constant.3\"",
    "\"helper.constant.4\"",
    "\"helper.constant.5\"",
    "\"helper.constant.6\"",
    "\"helper.constant.7\"",
    "\"helper.constant.8\"",
    "\"helper.constant.9\"",
    "\"linear_0.b_0\"",
    "\"linear_0.w_0\"",
    "\"linear_1.b_0\"",
    "\"linear_1.w_0\"",
    "\"p2o.pd_op.reshape.0.0\"",
    "\"p2o.pd_op.reshape.1.0\"",
    "\"p2o.pd_op.reshape.10.0\"",
    "\"p2o.pd_op.reshape.11.0\"",
    "\"p2o.pd_op.reshape.12.0\"",
    "\"p2o.pd_op.reshape.13.0\"",
    "\"p2o.pd_op.reshape.14.0\"",
    "\"p2o.pd_op.reshape.15.0\"",
    "\"p2o.pd_op.reshape.16.0\"",
    "\"p2o.pd_op.reshape.17.0\"",
    "\"p2o.pd_op.reshape.18.0\"",
    "\"p2o.pd_op.reshape.19.0\"",
    "\"p2o.pd_op.reshape.2.0\"",
    "\"p2o.pd_op.reshape.20.0\"",
    "\"p2o.pd_op.reshape.21.0\"",
    "\"p2o.pd_op.reshape.22.0\"",
    "\"p2o.pd_op.reshape.23.0\"",
    "\"p2o.pd_op.reshape.24.0\"",
    "\"p2o.pd_op.reshape.25.0\"",
    "\"p2o.pd_op.reshape.26.0\"",
    "\"p2o.pd_op.reshape.27.0\"",
    "\"p2o.pd_op.reshape.28.0\"",
    "\"p2o.pd_op.reshape.29.0\"",
    "\"p2o.pd_op.reshape.3.0\"",
    "\"p2o.pd_op.reshape.30.0\"",
    "\"p2o.pd_op.reshape.31.0\"",
    "\"p2o.pd_op.reshape.32.0\"",
    "\"p2o.pd_op.reshape.4.0\"",
    "\"p2o.pd_op.reshape.5.0\"",
    "\"p2o.pd_op.reshape.6.0\"",
    "\"p2o.pd_op.reshape.7.0\"",
    "\"p2o.pd_op.reshape.8.0\"",
    "\"p2o.pd_op.reshape.9.0\"",
    "\"p2o.pd_op.unsqueeze.0.0\"",
    "\"p2o.pd_op.unsqueeze.2.0\"",
];
