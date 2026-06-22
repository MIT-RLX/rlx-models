// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Pre-dequantize i8/u8 QMatMul weights to f32 companion params at load time.

use std::collections::HashMap;

use rlx_ir::DType;

pub type TypedParams = HashMap<String, (Vec<u8>, DType)>;

pub const BAKED_SUFFIX: &str = "__baked_f32";

pub fn baked_param_name(quant_name: &str) -> String {
    format!("{quant_name}{BAKED_SUFFIX}")
}

pub fn qmatmul_weight_bake_enabled() -> bool {
    if crate::compile_profile::env_flag("KITTEN_RLX_NO_QMATMUL_BAKE") {
        return false;
    }
    crate::compile_profile::qdq_fusion_enabled()
}

fn scale_for_quant(name: &str, params: &HashMap<String, Vec<f32>>) -> f32 {
    let direct = format!("{name}_scale");
    if let Some(v) = params.get(&direct).and_then(|s| s.first()) {
        return *v;
    }
    name.strip_suffix("_quantized")
        .and_then(|base| params.get(&format!("{base}_scale")))
        .and_then(|s| s.first())
        .copied()
        .unwrap_or(1.0)
}

fn zp_for_quant(name: &str, params: &HashMap<String, Vec<f32>>, typed: &TypedParams) -> f32 {
    let direct = format!("{name}_zero_point");
    if let Some(v) = params.get(&direct).and_then(|s| s.first()) {
        return *v;
    }
    if let Some((bytes, dt)) = typed.get(&direct) {
        return read_zp_f32(bytes, *dt);
    }
    name.strip_suffix("_quantized")
        .and_then(|base| params.get(&format!("{base}_zero_point")))
        .and_then(|s| s.first())
        .copied()
        .unwrap_or(0.0)
}

fn read_zp_f32(bytes: &[u8], dt: DType) -> f32 {
    match dt {
        DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as f32,
        DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f32,
        DType::U8 => bytes.first().copied().unwrap_or(0) as f32,
        DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        _ => 0.0,
    }
}

fn dequant_weight(bytes: &[u8], dtype: DType, scale: f32, zp: f32) -> Vec<f32> {
    match dtype {
        DType::U8 => bytes.iter().map(|&b| (b as f32 - zp) * scale).collect(),
        DType::I8 => bytes
            .iter()
            .map(|&b| (b as i8 as f32 - zp) * scale)
            .collect(),
        _ => Vec::new(),
    }
}

/// Build `{quant}_quantized__baked_f32` tensors for every typed quant weight.
pub fn bake_qmatmul_weights(
    typed: &TypedParams,
    params: &HashMap<String, Vec<f32>>,
) -> HashMap<String, Vec<f32>> {
    if !qmatmul_weight_bake_enabled() {
        return HashMap::new();
    }
    let mut out = HashMap::new();
    for (name, (bytes, dtype)) in typed {
        if !name.ends_with("_quantized") {
            continue;
        }
        if !matches!(dtype, DType::I8 | DType::U8) {
            continue;
        }
        let scale = scale_for_quant(name, params);
        let zp = zp_for_quant(name, params, typed);
        let baked = dequant_weight(bytes, *dtype, scale, zp);
        if baked.is_empty() {
            continue;
        }
        out.insert(baked_param_name(name), baked);
    }
    out
}
