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
        let shape: Vec<usize> = view.shape().to_vec();
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
            safetensors::tensor::Dtype::U8 => {
                raw.iter().map(|&b| (b as f32 - zp) * scale).collect()
            }
            _ => raw.iter().map(|&b| (b as i8 as f32 - zp) * scale).collect(),
        };
        let out_name = name.strip_suffix("_quantized").unwrap_or(&name).to_string();
        f32.entry(out_name)
            .or_insert_with(|| (data.clone(), shape.clone()));
        if name.ends_with("_quantized") {
            f32.entry(name.clone())
                .or_insert_with(|| (data.clone(), shape.clone()));
        }
        let import_name = format!("{name}_f32_import");
        f32.entry(import_name).or_insert((data, shape));
    }
    Ok(LoadedWeights { f32, i64 })
}

/// ONNX initializer names in this model (264 tensors).
pub const PARAM_NAMES: &[&str] = &[
    "\"Conv.1_same_pads_0\"",
    "\"Conv.2_same_pads_0\"",
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
    "\"HardSigmoid.4_alpha_0\"",
    "\"HardSigmoid.4_beta_0\"",
    "\"HardSigmoid.4_one_0\"",
    "\"HardSigmoid.4_zero_0\"",
    "\"MaxPool.0_same_pads_0\"",
    "\"auto.cast.72\"",
    "\"auto.cast.74\"",
    "\"auto.cast.75\"",
    "\"auto.cast.82\"",
    "\"auto.cast.84\"",
    "\"auto.cast.85\"",
    "\"batch_norm2d_55.b_0\"",
    "\"batch_norm2d_55.w_0\"",
    "\"batch_norm2d_55.w_1\"",
    "\"batch_norm2d_55.w_2\"",
    "\"batch_norm2d_56.b_0\"",
    "\"batch_norm2d_56.w_0\"",
    "\"batch_norm2d_56.w_1\"",
    "\"batch_norm2d_56.w_2\"",
    "\"batch_norm2d_57.b_0\"",
    "\"batch_norm2d_57.w_0\"",
    "\"batch_norm2d_57.w_1\"",
    "\"batch_norm2d_57.w_2\"",
    "\"conv2d_101.w_0\"",
    "\"conv2d_102.w_0\"",
    "\"conv2d_103.w_0\"",
    "\"conv2d_105.w_0\"",
    "\"conv2d_106.w_0\"",
    "\"conv2d_107.w_0\"",
    "\"conv2d_109.w_0\"",
    "\"conv2d_110.w_0\"",
    "\"conv2d_111.w_0\"",
    "\"conv2d_112.w_0\"",
    "\"conv2d_113.w_0\"",
    "\"conv2d_114.w_0\"",
    "\"conv2d_116.w_0\"",
    "\"conv2d_117.w_0\"",
    "\"conv2d_118.w_0\"",
    "\"conv2d_120.w_0\"",
    "\"conv2d_121.w_0\"",
    "\"conv2d_122.w_0\"",
    "\"conv2d_24.w_0\"",
    "\"conv2d_25.w_0\"",
    "\"conv2d_34.w_0\"",
    "\"conv2d_35.w_0\"",
    "\"conv2d_44.w_0\"",
    "\"conv2d_45.w_0\"",
    "\"conv2d_57.w_0\"",
    "\"conv2d_58.w_0\"",
    "\"conv2d_65.w_0\"",
    "\"conv2d_66.w_0\"",
    "\"conv2d_67.w_0\"",
    "\"conv2d_68.w_0\"",
    "\"conv2d_69.w_0\"",
    "\"conv2d_7.w_0\"",
    "\"conv2d_70.w_0\"",
    "\"conv2d_71.w_0\"",
    "\"conv2d_72.w_0\"",
    "\"conv2d_74.w_0\"",
    "\"conv2d_75.w_0\"",
    "\"conv2d_76.w_0\"",
    "\"conv2d_78.w_0\"",
    "\"conv2d_79.w_0\"",
    "\"conv2d_8.w_0\"",
    "\"conv2d_80.w_0\"",
    "\"conv2d_82.w_0\"",
    "\"conv2d_83.w_0\"",
    "\"conv2d_84.w_0\"",
    "\"conv2d_85.w_0\"",
    "\"conv2d_86.w_0\"",
    "\"conv2d_87.w_0\"",
    "\"conv2d_89.w_0\"",
    "\"conv2d_90.w_0\"",
    "\"conv2d_91.w_0\"",
    "\"conv2d_93.w_0\"",
    "\"conv2d_94.w_0\"",
    "\"conv2d_95.w_0\"",
    "\"conv2d_97.w_0\"",
    "\"conv2d_98.w_0\"",
    "\"conv2d_99.w_0\"",
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
    "\"helper.constant.30\"",
    "\"helper.constant.31\"",
    "\"helper.constant.32\"",
    "\"helper.constant.33\"",
    "\"helper.constant.34\"",
    "\"helper.constant.35\"",
    "\"helper.constant.36\"",
    "\"helper.constant.37\"",
    "\"helper.constant.38\"",
    "\"helper.constant.39\"",
    "\"helper.constant.4\"",
    "\"helper.constant.40\"",
    "\"helper.constant.41\"",
    "\"helper.constant.42\"",
    "\"helper.constant.43\"",
    "\"helper.constant.44\"",
    "\"helper.constant.45\"",
    "\"helper.constant.46\"",
    "\"helper.constant.5\"",
    "\"helper.constant.52\"",
    "\"helper.constant.53\"",
    "\"helper.constant.54\"",
    "\"helper.constant.55\"",
    "\"helper.constant.56\"",
    "\"helper.constant.57\"",
    "\"helper.constant.58\"",
    "\"helper.constant.59\"",
    "\"helper.constant.6\"",
    "\"helper.constant.65\"",
    "\"helper.constant.66\"",
    "\"helper.constant.7\"",
    "\"helper.constant.72\"",
    "\"helper.constant.73\"",
    "\"helper.constant.74\"",
    "\"helper.constant.75\"",
    "\"helper.constant.76\"",
    "\"helper.constant.77\"",
    "\"helper.constant.78\"",
    "\"helper.constant.79\"",
    "\"helper.constant.8\"",
    "\"helper.constant.85\"",
    "\"helper.constant.86\"",
    "\"helper.constant.9\"",
    "\"helper.reshape.0\"",
    "\"helper.reshape.1\"",
    "\"helper.reshape.2\"",
    "\"helper.reshape.3\"",
    "\"helper.reshape.4\"",
    "\"helper.reshape.5\"",
    "\"helper.reshape.6\"",
    "\"helper.reshape.7\"",
    "\"helper.reshape.8\"",
    "\"helper.reshape.9\"",
    "\"helper.unsqueeze.0\"",
    "\"helper.unsqueeze.1\"",
    "\"helper.unsqueeze.3\"",
    "\"linear_0.b_0\"",
    "\"linear_0.w_0\"",
    "\"linear_1.b_0\"",
    "\"linear_1.w_0\"",
    "\"linear_2.b_0\"",
    "\"linear_2.w_0\"",
    "\"linear_3.b_0\"",
    "\"linear_3.w_0\"",
    "\"linear_4.b_0\"",
    "\"linear_4.w_0\"",
    "\"linear_5.b_0\"",
    "\"linear_5.w_0\"",
    "\"linear_6.b_0\"",
    "\"linear_6.w_0\"",
    "\"linear_7.b_0\"",
    "\"linear_7.w_0\"",
    "\"linear_8.b_0\"",
    "\"linear_8.w_0\"",
    "\"p2o.pd_op.full_int_array.61.0\"",
    "\"p2o.pd_op.full_int_array.62.0\"",
    "\"p2o.pd_op.full_int_array.64.0\"",
    "\"p2o.pd_op.full_int_array.65.0\"",
    "\"p2o.pd_op.full_int_array.66.0\"",
    "\"p2o.pd_op.full_int_array.67.0\"",
    "\"p2o.pd_op.full_int_array.68.0\"",
    "\"p2o.pd_op.full_int_array.69.0\"",
    "\"p2o.pd_op.full_int_array.72.0\"",
    "\"p2o.pd_op.full_int_array.73.0\"",
    "\"p2o.pd_op.full_int_array.74.0\"",
    "\"p2o.pd_op.full_int_array.75.0\"",
    "\"p2o.pd_op.full_int_array.76.0\"",
    "\"p2o.pd_op.full_int_array.77.0\"",
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
    "\"p2o.pd_op.reshape.33.0\"",
    "\"p2o.pd_op.reshape.34.0\"",
    "\"p2o.pd_op.reshape.35.0\"",
    "\"p2o.pd_op.reshape.36.0\"",
    "\"p2o.pd_op.reshape.37.0\"",
    "\"p2o.pd_op.reshape.38.0\"",
    "\"p2o.pd_op.reshape.39.0\"",
    "\"p2o.pd_op.reshape.4.0\"",
    "\"p2o.pd_op.reshape.40.0\"",
    "\"p2o.pd_op.reshape.41.0\"",
    "\"p2o.pd_op.reshape.42.0\"",
    "\"p2o.pd_op.reshape.43.0\"",
    "\"p2o.pd_op.reshape.44.0\"",
    "\"p2o.pd_op.reshape.45.0\"",
    "\"p2o.pd_op.reshape.46.0\"",
    "\"p2o.pd_op.reshape.47.0\"",
    "\"p2o.pd_op.reshape.48.0\"",
    "\"p2o.pd_op.reshape.49.0\"",
    "\"p2o.pd_op.reshape.5.0\"",
    "\"p2o.pd_op.reshape.50.0\"",
    "\"p2o.pd_op.reshape.51.0\"",
    "\"p2o.pd_op.reshape.52.0\"",
    "\"p2o.pd_op.reshape.53.0\"",
    "\"p2o.pd_op.reshape.6.0\"",
    "\"p2o.pd_op.reshape.7.0\"",
    "\"p2o.pd_op.reshape.8.0\"",
    "\"p2o.pd_op.reshape.9.0\"",
];
