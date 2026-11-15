// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! YuNet weights out of an ONNX export (feature `onnx`): its `Conv` nodes, in order, are
//! [`CONVS`]; every kernel, stride, group and shape is checked, so an export with another
//! architecture is refused rather than misread. Only for conversion and parity tests: the
//! model runs from [`Weights`] in an RLX graph.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use protobuf::Message;
use rlx_onnx_proto::onnx::{ModelProto, TensorProto, tensor_proto::DataType};

use crate::model::CONVS;
use crate::weights::Weights;

fn floats(t: &TensorProto) -> Result<Vec<f32>> {
    ensure!(
        t.data_type.map(|d| d.enum_value_or_default()) == Some(DataType::FLOAT),
        "{}: not f32",
        t.name()
    );
    Ok(match &t.raw_data {
        Some(raw) => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        None => t.float_data.clone(),
    })
}

/// The weights of a YuNet ONNX file.
pub fn weights_from_onnx(bytes: &[u8]) -> Result<Weights> {
    let model = ModelProto::parse_from_bytes(bytes).context("parse ONNX")?;
    let g = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow!("ONNX model without a graph"))?;
    let init: HashMap<&str, &TensorProto> = g.initializer.iter().map(|t| (t.name(), t)).collect();
    let convs: Vec<_> = g.node.iter().filter(|n| n.op_type() == "Conv").collect();
    ensure!(
        convs.len() == CONVS.len(),
        "{} Conv nodes, YuNet has {}",
        convs.len(),
        CONVS.len()
    );
    let mut tensors = HashMap::new();
    for (n, spec) in convs.iter().zip(&CONVS) {
        let ints = |a: &str| {
            n.attribute
                .iter()
                .find(|x| x.name() == a)
                .map(|x| x.ints.clone())
                .unwrap_or_default()
        };
        let group = n
            .attribute
            .iter()
            .find(|x| x.name() == "group")
            .and_then(|x| x.i)
            .unwrap_or(1) as usize;
        let (k, s, p) = (spec.k as i64, spec.stride as i64, (spec.k / 2) as i64);
        if ints("kernel_shape") != [k, k]
            || ints("strides") != [s, s]
            || ints("pads") != [p, p, p, p]
            || group != spec.groups
        {
            bail!(
                "{} ({}): kernel {:?} stride {:?} pads {:?} group {group}, want {k} / {s} / {p} / {}",
                spec.name,
                n.output[0],
                ints("kernel_shape"),
                ints("strides"),
                ints("pads"),
                spec.groups
            );
        }
        let w = init
            .get(n.input[1].as_str())
            .ok_or_else(|| anyhow!("{}: weight not a constant", spec.name))?;
        let dims: Vec<usize> = w.dims.iter().map(|&d| d as usize).collect();
        ensure!(
            dims == spec.weight_shape(),
            "{}: weight {dims:?}, want {:?}",
            spec.name,
            spec.weight_shape()
        );
        let bias = match n.input.get(2).filter(|b| !b.is_empty()) {
            Some(b) => floats(
                init.get(b.as_str())
                    .ok_or_else(|| anyhow!("{}: bias not a constant", spec.name))?,
            )?,
            None => vec![0.0; spec.out_c],
        };
        tensors.insert(format!("{}.weight", spec.name), floats(w)?);
        tensors.insert(format!("{}.bias", spec.name), bias);
    }
    let w = Weights { tensors };
    w.check()?;
    Ok(w)
}
