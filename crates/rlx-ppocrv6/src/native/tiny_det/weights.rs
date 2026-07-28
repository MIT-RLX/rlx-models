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

/// ONNX initializer names in this model (224 tensors).
pub const PARAM_NAMES: &[&str] = &[
    "\"Conv.1_same_pads_0\"",
    "\"Conv.2_same_pads_0\"",
    "\"HardSigmoid.0_alpha_0\"",
    "\"HardSigmoid.0_beta_0\"",
    "\"HardSigmoid.0_one_0\"",
    "\"HardSigmoid.0_zero_0\"",
    "\"HardSigmoid.10_alpha_0\"",
    "\"HardSigmoid.10_beta_0\"",
    "\"HardSigmoid.10_one_0\"",
    "\"HardSigmoid.10_zero_0\"",
    "\"HardSigmoid.11_alpha_0\"",
    "\"HardSigmoid.11_beta_0\"",
    "\"HardSigmoid.11_one_0\"",
    "\"HardSigmoid.11_zero_0\"",
    "\"HardSigmoid.12_alpha_0\"",
    "\"HardSigmoid.12_beta_0\"",
    "\"HardSigmoid.12_one_0\"",
    "\"HardSigmoid.12_zero_0\"",
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
    "\"HardSigmoid.5_alpha_0\"",
    "\"HardSigmoid.5_beta_0\"",
    "\"HardSigmoid.5_one_0\"",
    "\"HardSigmoid.5_zero_0\"",
    "\"HardSigmoid.6_alpha_0\"",
    "\"HardSigmoid.6_beta_0\"",
    "\"HardSigmoid.6_one_0\"",
    "\"HardSigmoid.6_zero_0\"",
    "\"HardSigmoid.7_alpha_0\"",
    "\"HardSigmoid.7_beta_0\"",
    "\"HardSigmoid.7_one_0\"",
    "\"HardSigmoid.7_zero_0\"",
    "\"HardSigmoid.8_alpha_0\"",
    "\"HardSigmoid.8_beta_0\"",
    "\"HardSigmoid.8_one_0\"",
    "\"HardSigmoid.8_zero_0\"",
    "\"HardSigmoid.9_alpha_0\"",
    "\"HardSigmoid.9_beta_0\"",
    "\"HardSigmoid.9_one_0\"",
    "\"HardSigmoid.9_zero_0\"",
    "\"MaxPool.0_same_pads_0\"",
    "\"_v_680\"",
    "\"_v_683\"",
    "\"_v_686\"",
    "\"_v_689\"",
    "\"_v_692\"",
    "\"_v_695\"",
    "\"_v_698\"",
    "\"_v_701\"",
    "\"_v_704\"",
    "\"_v_707\"",
    "\"_v_710\"",
    "\"_v_713\"",
    "\"_v_716\"",
    "\"_v_719\"",
    "\"_v_722\"",
    "\"_v_725\"",
    "\"_v_728\"",
    "\"_v_731\"",
    "\"_v_734\"",
    "\"_v_737\"",
    "\"_v_740\"",
    "\"_v_743\"",
    "\"_v_746\"",
    "\"_v_749\"",
    "\"_v_752\"",
    "\"_v_755\"",
    "\"_v_758\"",
    "\"_v_761\"",
    "\"_v_764\"",
    "\"_v_767\"",
    "\"_v_770\"",
    "\"_v_773\"",
    "\"_v_776\"",
    "\"_v_779\"",
    "\"_v_782\"",
    "\"_v_785\"",
    "\"_v_788\"",
    "\"_v_791\"",
    "\"_v_794\"",
    "\"_v_797\"",
    "\"_v_800\"",
    "\"_v_803\"",
    "\"_v_806\"",
    "\"_v_809\"",
    "\"_v_812\"",
    "\"_v_815\"",
    "\"_v_818\"",
    "\"_v_821\"",
    "\"_v_824\"",
    "\"_v_827\"",
    "\"_v_830\"",
    "\"_v_833\"",
    "\"_v_836\"",
    "\"_v_839\"",
    "\"_v_842\"",
    "\"_v_845\"",
    "\"_v_848\"",
    "\"_v_851\"",
    "\"_v_854\"",
    "\"_v_857\"",
    "\"_v_860\"",
    "\"_v_863\"",
    "\"_v_866\"",
    "\"_v_869\"",
    "\"_v_872\"",
    "\"_v_875\"",
    "\"_v_878\"",
    "\"_v_881\"",
    "\"_v_884\"",
    "\"_v_887\"",
    "\"_v_890\"",
    "\"_v_893\"",
    "\"_v_896\"",
    "\"_v_899\"",
    "\"_v_902\"",
    "\"auto.cast.95\"",
    "\"auto.cast.98\"",
    "\"conv2d_108.w_0\"",
    "\"conv2d_109.w_0\"",
    "\"conv2d_110.w_0\"",
    "\"conv2d_111.w_0\"",
    "\"conv2d_112.w_0\"",
    "\"conv2d_114.w_0\"",
    "\"conv2d_115.w_0\"",
    "\"conv2d_116.w_0\"",
    "\"conv2d_118.w_0\"",
    "\"conv2d_119.w_0\"",
    "\"conv2d_120.w_0\"",
    "\"conv2d_121.w_0\"",
    "\"conv2d_122.w_0\"",
    "\"conv2d_123.w_0\"",
    "\"conv2d_125.w_0\"",
    "\"conv2d_126.w_0\"",
    "\"conv2d_127.w_0\"",
    "\"conv2d_129.w_0\"",
    "\"conv2d_130.w_0\"",
    "\"conv2d_131.w_0\"",
    "\"conv2d_132.w_0\"",
    "\"conv2d_133.w_0\"",
    "\"conv2d_134.w_0\"",
    "\"conv2d_136.w_0\"",
    "\"conv2d_137.w_0\"",
    "\"conv2d_138.w_0\"",
    "\"conv2d_140.w_0\"",
    "\"conv2d_141.w_0\"",
    "\"conv2d_142.w_0\"",
    "\"conv2d_144.w_0\"",
    "\"conv2d_145.w_0\"",
    "\"conv2d_146.w_0\"",
    "\"conv2d_148.w_0\"",
    "\"conv2d_149.w_0\"",
    "\"conv2d_150.w_0\"",
    "\"conv2d_151.w_0\"",
    "\"conv2d_152.w_0\"",
    "\"conv2d_153.w_0\"",
    "\"conv2d_155.w_0\"",
    "\"conv2d_156.w_0\"",
    "\"conv2d_157.w_0\"",
    "\"conv2d_159.w_0\"",
    "\"conv2d_160.w_0\"",
    "\"conv2d_161.w_0\"",
    "\"conv2d_162.w_0\"",
    "\"conv2d_163.w_0\"",
    "\"conv2d_164.w_0\"",
    "\"conv2d_165.w_0\"",
    "\"conv2d_166.w_0\"",
    "\"conv2d_20.w_0\"",
    "\"conv2d_21.w_0\"",
    "\"conv2d_33.w_0\"",
    "\"conv2d_34.w_0\"",
    "\"conv2d_43.w_0\"",
    "\"conv2d_44.w_0\"",
    "\"conv2d_56.w_0\"",
    "\"conv2d_57.w_0\"",
    "\"conv2d_64.w_0\"",
    "\"conv2d_65.w_0\"",
    "\"conv2d_66.w_0\"",
    "\"conv2d_7.w_0\"",
    "\"conv2d_70.w_0\"",
    "\"conv2d_71.w_0\"",
    "\"conv2d_72.w_0\"",
    "\"conv2d_73.w_0\"",
    "\"conv2d_74.w_0\"",
    "\"conv2d_75.w_0\"",
    "\"conv2d_79.w_0\"",
    "\"conv2d_8.w_0\"",
    "\"conv2d_80.w_0\"",
    "\"conv2d_81.w_0\"",
    "\"conv2d_82.w_0\"",
    "\"conv2d_83.w_0\"",
    "\"conv2d_84.w_0\"",
    "\"conv2d_88.w_0\"",
    "\"conv2d_89.w_0\"",
    "\"conv2d_90.w_0\"",
    "\"conv2d_91.w_0\"",
    "\"conv2d_92.w_0\"",
    "\"conv2d_93.w_0\"",
    "\"conv2d_97.w_0\"",
    "\"conv2d_98.w_0\"",
    "\"conv2d_99.w_0\"",
    "\"helper.constant.0\"",
    "\"helper.constant.1\"",
    "\"helper.constant.2\"",
    "\"helper.constant.39\"",
    "\"helper.constant.40\"",
    "\"helper.constant.45\"",
    "\"helper.constant.47\"",
    "\"p2o.pd_op.reshape.75.0\"",
    "\"p2o.pd_op.reshape.76.0\"",
];
