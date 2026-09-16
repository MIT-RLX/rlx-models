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

//! Espresso weight blobs → a [`WeightMap`] keyed by the names
//! [`crate::spec`] assigns.
//!
//! Layouts, matching Apple's storage:
//!
//! | layer           | blob                 | dtype | stored shape       | exposed as |
//! |-----------------|----------------------|-------|--------------------|------------|
//! | `convolution`   | `blob_weights_f16`   | f16   | `[C, K, Ny, Nx]`   | unchanged (rlx Conv2d wants `[C_out, C_in/g, kH, kW]`) |
//! | `convolution`   | `blob_biases`        | f32   | `[C]`              | unchanged |
//! | `inner_product` | `blob_weights_f16`   | f16   | `[nC, nB]` (out, in) | transposed to `[nB, nC]` for `x @ W` |
//! | `inner_product` | `w_f16_t`            | f16   | `[nB, nC]` already | unchanged |
//! | `inner_product` | `blob_biases`/`b_f32`| f32   | `[nC]`             | unchanged |
//!
//! Every blob's byte length is checked against the shape the layer declares —
//! a silent misread here would produce a well-formed but wrong hash.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;

use crate::espresso::{EspressoLayer, EspressoNet};
use crate::spec::{is_input_layer, tensor_base_for};

/// Extract every weight tensor referenced by the network.
pub fn from_espresso(net: &EspressoNet) -> Result<WeightMap> {
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut used: HashMap<String, usize> = HashMap::new();

    for (i, l) in net.layers.iter().enumerate() {
        if is_input_layer(&l.kind) {
            continue;
        }
        // Must mirror `spec::Deriver::tensor_base` call-for-call.
        let base = tensor_base_for(&mut used, &l.name);
        let extracted = match l.kind.as_str() {
            "convolution" => conv_tensors(net, l, &base),
            "inner_product" | "innerproduct" | "fully_connected" => ip_tensors(net, l, &base),
            "batchnorm" | "instancenorm" | "instancenorm_1d" => norm_tensors(net, l, &base),
            _ => Ok(Vec::new()),
        }
        .with_context(|| {
            format!(
                "espresso layer {i}/{} {:?} (type {})",
                net.layers.len(),
                l.name,
                l.kind
            )
        })?;
        for (name, data, shape) in extracted {
            tensors.insert(name, (data, shape));
        }
    }

    Ok(WeightMap::from_tensors(tensors))
}

type Tensor = (String, Vec<f32>, Vec<usize>);

fn conv_tensors(net: &EspressoNet, l: &EspressoLayer, base: &str) -> Result<Vec<Tensor>> {
    let out_c = l.req_int(&["C", "n_output_channels", "outputChannels"])? as usize;
    let k_in = l.req_int(&["K", "n_input_channels", "inputChannels"])? as usize;
    let kw = l.int_or(&["Nx", "kernel_x", "size_x", "kernelWidth"], 1) as usize;
    let kh = l.int_or(&["Ny", "kernel_y", "size_y", "kernelHeight"], 1) as usize;
    let groups = l.int_or(&["n_groups", "groups", "nGroups"], 1).max(1) as usize;
    ensure!(
        k_in.is_multiple_of(groups),
        "convolution {:?}: n_groups={groups} does not divide K={k_in}",
        l.name
    );
    // Espresso's `K` counts total input channels; the kernel is stored with the
    // per-group depth, i.e. `[C, K / n_groups, Ny, Nx]`. Depthwise layers rely
    // on this: `C=16, K=16, n_groups=16` stores 16·1·3·3, not 16·16·3·3.
    let per_group = k_in / groups;

    let widx = l
        .blob(&["blob_weights_f16", "w_f16", "weights_f16"])
        .ok_or_else(|| anyhow::anyhow!("convolution {:?} references no weight blob", l.name))?;
    let want = out_c * per_group * kh * kw;
    let w = net.blob_f16(widx)?;
    ensure!(
        w.len() == want,
        "convolution {:?}: weight blob {widx} holds {} f16 values, expected {want} \
         ([C={out_c}, K/groups={per_group}, Ny={kh}, Nx={kw}]; K={k_in}, n_groups={groups})",
        l.name,
        w.len()
    );

    let mut out = vec![(format!("{base}.weight"), w, vec![out_c, per_group, kh, kw])];
    if let Some(bidx) = l.blob(&["blob_biases", "b_f32"]) {
        let b = net.blob_f32(bidx)?;
        ensure!(
            b.len() == out_c,
            "convolution {:?}: bias blob {bidx} holds {} f32 values, expected {out_c}",
            l.name,
            b.len()
        );
        out.push((format!("{base}.bias"), b, vec![out_c]));
    }
    Ok(out)
}

/// Instance-norm parameters.
///
/// `blob_batchnorm_params` is a single f32 blob of `4·C` values **interleaved
/// per channel** as `[gamma, beta, mean, variance]`. With
/// `training_instancenorm = 1` the mean/variance entries are placeholders —
/// in the shipping NeuralHash net they are exactly 0 and 1 for all 4616
/// channels — because the statistics are recomputed from the input. Only
/// gamma (scale) and beta (shift) are extracted.
///
/// The interleaving is not cosmetic: reading the blob as four contiguous
/// `C`-blocks yields a scale vector that is 25% zeros and 25% ones, which
/// still runs and still produces a hash — just the wrong one.
fn norm_tensors(net: &EspressoNet, l: &EspressoLayer, base: &str) -> Result<Vec<Tensor>> {
    let c = l.req_int(&["C"])? as usize;
    let idx = l
        .blob(&["blob_batchnorm_params", "blob_params"])
        .ok_or_else(|| anyhow::anyhow!("norm {:?} references no parameter blob", l.name))?;
    let p = net.blob_f32(idx)?;
    ensure!(
        p.len() == 4 * c,
        "norm {:?}: parameter blob {idx} holds {} f32 values, expected {} (4 × C={c})",
        l.name,
        p.len(),
        4 * c
    );
    let scale: Vec<f32> = (0..c).map(|i| p[4 * i]).collect();
    let shift: Vec<f32> = (0..c).map(|i| p[4 * i + 1]).collect();
    Ok(vec![
        (format!("{base}.scale"), scale, vec![c]),
        (format!("{base}.shift"), shift, vec![c]),
    ])
}

fn ip_tensors(net: &EspressoNet, l: &EspressoLayer, base: &str) -> Result<Vec<Tensor>> {
    let n_in = l.req_int(&["nB", "n_input", "inputChannels"])? as usize;
    let n_out = l.req_int(&["nC", "n_output", "outputChannels"])? as usize;

    // `blob_weights_f16` is physically [nC, nB]; `w_f16_t` is already [nB, nC].
    let (widx, needs_transpose) = match l.blob(&["blob_weights_f16"]) {
        Some(i) => (i, true),
        None => (
            l.blob(&["w_f16_t"]).ok_or_else(|| {
                anyhow::anyhow!("inner_product {:?} references no weight blob", l.name)
            })?,
            false,
        ),
    };
    let raw = net.blob_f16(widx)?;
    ensure!(
        raw.len() == n_in * n_out,
        "inner_product {:?}: weight blob {widx} holds {} f16 values, expected {} (nB={n_in} × nC={n_out})",
        l.name,
        raw.len(),
        n_in * n_out
    );

    let w = if needs_transpose {
        // [nC, nB] → [nB, nC]
        let mut t = vec![0f32; n_in * n_out];
        for o in 0..n_out {
            for i in 0..n_in {
                t[i * n_out + o] = raw[o * n_in + i];
            }
        }
        t
    } else {
        raw
    };

    let mut out = vec![(format!("{base}.weight"), w, vec![n_in, n_out])];
    if let Some(bidx) = l.blob(&["blob_biases", "b_f32"]) {
        let b = net.blob_f32(bidx)?;
        ensure!(
            b.len() == n_out,
            "inner_product {:?}: bias blob {bidx} holds {} f32 values, expected {n_out}",
            l.name,
            b.len()
        );
        out.push((format!("{base}.bias"), b, vec![n_out]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::NeuralHashSpec;
    use half::f16;

    fn container(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut out = (entries.len() as u64).to_le_bytes().to_vec();
        for (idx, data) in entries {
            out.extend(idx.to_le_bytes());
            out.extend((data.len() as u64).to_le_bytes());
        }
        for (_, data) in entries {
            out.extend(data);
        }
        out
    }

    fn f16_blob(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect()
    }
    fn f32_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn tiny_net(weights: Vec<u8>) -> EspressoNet {
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"c","top":"a","bottom":"i",
           "C":2,"K":3,"Nx":1,"Ny":1,"has_biases":1,"blob_weights_f16":0,"blob_biases":1},
          {"type":"pool","name":"gap","top":"p","bottom":"a","avg_or_max":0,"is_global":1},
          {"type":"inner_product","name":"head","top":"e","bottom":"p","nB":2,"nC":3,
           "blob_weights_f16":2,"blob_biases":3}
        ]}"#;
        let mut shapes = HashMap::new();
        shapes.insert(
            "i".to_string(),
            crate::espresso::BlobShape {
                n: 1,
                c: 3,
                h: 4,
                w: 4,
            },
        );
        EspressoNet::from_parts(net_json, &weights, shapes).unwrap()
    }

    fn tiny_weights() -> Vec<u8> {
        container(&[
            (0, f16_blob(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])), // conv [2,3,1,1]
            (1, f32_blob(&[0.5, -0.5])),                    // conv bias [2]
            (2, f16_blob(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])), // ip [nC=3, nB=2]
            (3, f32_blob(&[0.0, 1.0, 2.0])),                // ip bias [3]
        ])
    }

    #[test]
    fn extracts_conv_and_ip_with_declared_shapes() {
        let net = tiny_net(tiny_weights());
        let mut wm = from_espresso(&net).unwrap();
        let (w, s) = wm.take("c.weight").unwrap();
        assert_eq!(s, vec![2, 3, 1, 1]);
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (b, s) = wm.take("c.bias").unwrap();
        assert_eq!((b, s), (vec![0.5, -0.5], vec![2]));
    }

    #[test]
    fn inner_product_weights_are_transposed_to_in_by_out() {
        let net = tiny_net(tiny_weights());
        let mut wm = from_espresso(&net).unwrap();
        let (w, s) = wm.take("head.weight").unwrap();
        // Stored [nC=3, nB=2] row-major = [[1,2],[3,4],[5,6]];
        // exposed [nB=2, nC=3] = [[1,3,5],[2,4,6]].
        assert_eq!(s, vec![2, 3]);
        assert_eq!(w, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn spec_and_weights_agree_on_names() {
        let net = tiny_net(tiny_weights());
        let spec = NeuralHashSpec::from_espresso(&net).unwrap();
        let wm = from_espresso(&net).unwrap();
        let mut from_wm: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
        from_wm.sort();
        let mut from_spec = spec.weight_names();
        from_spec.sort();
        assert_eq!(
            from_spec, from_wm,
            "spec op names and extracted tensor names must match exactly"
        );
    }

    #[test]
    fn duplicate_layer_names_stay_aligned() {
        // Two layers both named `conv`: the spec suffixes the second `#1`, and
        // the extractor must land on the same suffix or weights swap silently.
        let net_json = r#"{"layers": [
          {"type":"input","name":"i","top":"i","bottom":""},
          {"type":"convolution","name":"conv","top":"a","bottom":"i",
           "C":1,"K":3,"Nx":1,"Ny":1,"blob_weights_f16":0},
          {"type":"convolution","name":"conv","top":"b","bottom":"a",
           "C":1,"K":1,"Nx":1,"Ny":1,"blob_weights_f16":1}
        ]}"#;
        let weights = container(&[(0, f16_blob(&[1.0, 2.0, 3.0])), (1, f16_blob(&[9.0]))]);
        let mut shapes = HashMap::new();
        shapes.insert(
            "i".to_string(),
            crate::espresso::BlobShape {
                n: 1,
                c: 3,
                h: 4,
                w: 4,
            },
        );
        let net = EspressoNet::from_parts(net_json, &weights, shapes).unwrap();
        let spec = NeuralHashSpec::from_espresso(&net).unwrap();
        let mut wm = from_espresso(&net).unwrap();
        assert_eq!(spec.weight_names(), vec!["conv.weight", "conv#1.weight"]);
        assert_eq!(wm.take("conv.weight").unwrap().0, vec![1.0, 2.0, 3.0]);
        assert_eq!(wm.take("conv#1.weight").unwrap().0, vec![9.0]);
    }

    #[test]
    fn blob_size_mismatch_is_caught() {
        // Conv declares [2,3,1,1] = 6 values but the blob holds 4.
        let weights = container(&[
            (0, f16_blob(&[1.0, 2.0, 3.0, 4.0])),
            (1, f32_blob(&[0.5, -0.5])),
            (2, f16_blob(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            (3, f32_blob(&[0.0, 1.0, 2.0])),
        ]);
        let net = tiny_net(weights);
        // `WeightMap` is not `Debug`, so take the error out of the `Result`.
        let e = format!(
            "{:#}",
            from_espresso(&net).err().expect("expected an error")
        );
        assert!(e.contains("expected 6"), "{e}");
        assert!(e.contains("type convolution"), "{e}");
    }
}
