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

//! Per-network parity: each native graph vs the upstream `.onnx` on ONNX
//! Runtime (the reference), on the same inputs.
//!
//! `INDRA_UPSTREAM_DIR` = the `datalab-to/indra-ocr` snapshot (`*.onnx`),
//! `INDRA_MODEL_DIR` = its `rlx-indra-convert` output. Skips when unset.
//! Optional `INDRA_PARITY_DEVICE` (default `cpu`).
//!
//! ```sh
//! cargo test -p rlx-indra --features parity-ort --release --test indra_parity_ort -- --nocapture
//! ```

use ort::session::Session;
use ort::value::Tensor;
use rlx_indra::graph::{self, Weights};
use rlx_runtime::Device;
use std::path::PathBuf;

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let up = std::env::var_os("INDRA_UPSTREAM_DIR")?;
    let native = std::env::var_os("INDRA_MODEL_DIR")?;
    Some((up.into(), native.into()))
}

fn device() -> Device {
    std::env::var("INDRA_PARITY_DEVICE")
        .ok()
        .map(|s| rlx_runtime::parse_device(&s).expect("INDRA_PARITY_DEVICE"))
        .unwrap_or(Device::Cpu)
}

macro_rules! need_dirs {
    () => {
        match dirs() {
            Some(d) => d,
            None => {
                eprintln!("skip: set INDRA_UPSTREAM_DIR and INDRA_MODEL_DIR");
                return;
            }
        }
    };
}

/// Deterministic page-like ink: dark strokes on paper, a little noise.
fn ink(n: usize, h: usize, w: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32
    };
    let mut out = vec![0.0f32; n * h * w];
    for b in 0..n {
        let img = &mut out[b * h * w..(b + 1) * h * w];
        for _ in 0..(w * h / 40).max(8) {
            let (x0, y0) = ((rnd() * w as f32) as usize, (rnd() * h as f32) as usize);
            let (dx, dy) = (1 + (rnd() * 6.0) as usize, 1 + (rnd() * 10.0) as usize);
            let v = 0.5 + 0.5 * rnd();
            for y in y0..(y0 + dy).min(h) {
                for x in x0..(x0 + dx).min(w) {
                    img[y * w + x] = v;
                }
            }
        }
        for v in img.iter_mut() {
            *v = (*v + 0.05 * rnd()).min(1.0);
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn session(path: PathBuf) -> Session {
    Session::builder()
        .unwrap()
        .commit_from_file(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn rec_case(net: &str, widths: &[usize]) {
    let (up, native) = need_dirs!();
    let w = Weights::load(&native.join(format!("{net}.safetensors"))).unwrap();
    let spec = graph::rec::RecSpec::from_weights(&w).unwrap();
    let batch = widths.len();
    let width = widths.iter().max().unwrap().div_ceil(8) * 8;
    let mut x = ink(batch, spec.height, width, 7);
    for (b, &wd) in widths.iter().enumerate() {
        for y in 0..spec.height {
            x[(b * spec.height + y) * width + wd..(b * spec.height + y + 1) * width].fill(0.0);
        }
    }
    let t = width / graph::rec::STRIDE;

    let mut s = session(up.join(format!("{net}.onnx")));
    let xs = Tensor::from_array(([batch, 1, spec.height, width], x.clone())).unwrap();
    let ws = Tensor::from_array((
        [batch],
        widths.iter().map(|&v| v as i64).collect::<Vec<_>>(),
    ))
    .unwrap();
    let outs = s.run(ort::inputs!["x" => xs, "widths" => ws]).unwrap();
    let ids = outs["ids"].try_extract_tensor::<i64>().unwrap().1.to_vec();
    let conf = outs["conf"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    let blank = outs["blank"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .1
        .to_vec();
    let topv = outs["topv"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    let topi = outs["topi"].try_extract_tensor::<i64>().unwrap().1.to_vec();

    let (mask, inv) = graph::rec::line_masks(widths, t);
    let c = graph::Compiled::new(w, device(), move |w, k| graph::rec::build(w, k[0], k[1]));
    let outs = c
        .run(
            &[batch, width],
            &[("x", &x), ("mask", &mask), ("inv_count", &inv)],
        )
        .unwrap();
    let (best, lse, feats) = (&outs[0], &outs[1], &outs[2]);
    let head = graph::rec::Head::new(c.weights()).unwrap();
    let ch = head.channels();
    const K: usize = graph::rec::TOP_K;
    let (mut d_conf, mut d_blank, mut d_top, mut id_miss, mut top_miss) = (0f32, 0f32, 0f32, 0, 0);
    for r in 0..batch * t {
        let f = &feats[r * ch..(r + 1) * ch];
        let lp = |class: usize| head.logp(f, lse[r], class);
        if best[r] as i64 != ids[r] {
            id_miss += 1;
        }
        d_conf = d_conf.max((lp(best[r] as usize).exp() - conf[r]).abs());
        d_blank = d_blank.max((lp(0) - blank[r]).abs());
        for (k, (class, v)) in head.top_k(f, lse[r]).into_iter().enumerate() {
            d_top = d_top.max((v - topv[r * K + k]).abs());
            if class as i64 != topi[r * K + k] {
                top_miss += 1;
            }
        }
    }
    eprintln!(
        "{net} B={batch} W={width}: argmax miss {id_miss}/{}, top-8 index miss {top_miss}, \
         |Δconf| {d_conf:.2e}, |Δblank| {d_blank:.2e}, |Δtopv| {d_top:.2e}",
        batch * t
    );
    assert_eq!(id_miss, 0, "{net}: argmax differs");
    assert!(
        d_conf < 1e-4 && d_blank < 1e-3 && d_top < 1e-3,
        "{net}: values drift"
    );
}

#[test]
fn rec_base_matches_onnx() {
    rec_case("rec", &[400, 236, 96]);
}

#[test]
fn rec_groups_match_onnx() {
    for net in ["rec_cjk", "rec_rtl", "rec_world"] {
        rec_case(net, &[320, 120]);
    }
}

#[test]
fn det_matches_onnx() {
    let (up, native) = need_dirs!();
    let w = Weights::load(&native.join("det.safetensors")).unwrap();
    let (h, wd) = (320, 256);
    let x = ink(1, h, wd, 11);
    let mut s = session(up.join("det.onnx"));
    let xs = Tensor::from_array(([1usize, 1, h, wd], x.clone())).unwrap();
    let outs = s.run(ort::inputs!["x" => xs]).unwrap();
    let want = outs["out"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    let c = graph::Compiled::new(w, device(), |w, k| graph::det::build(w, k[0], k[1]));
    let got = &c.run(&[h, wd], &[("x", &x)]).unwrap()[0];
    let d = max_abs(got, &want);
    eprintln!("det {h}×{wd}: max |Δ| {d:.2e}");
    assert!(d < 1e-3, "det drift {d}");
}

#[test]
fn script_matches_onnx() {
    let (up, native) = need_dirs!();
    let w = Weights::load(&native.join("script.safetensors")).unwrap();
    let widths = [400usize, 120, 64];
    let width = 400;
    let x = ink(widths.len(), 32, width, 5);
    let mut s = session(up.join("script.onnx"));
    let xs = Tensor::from_array(([widths.len(), 1, 32, width], x.clone())).unwrap();
    let ws = Tensor::from_array((
        [widths.len()],
        widths.iter().map(|&v| v as i64).collect::<Vec<_>>(),
    ))
    .unwrap();
    let outs = s.run(ort::inputs!["x" => xs, "widths" => ws]).unwrap();
    let want = outs["p"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    let cols = width / graph::classifier::SCRIPT_STRIDE;
    let (mask, inv) = graph::classifier::script_masks(&widths, cols);
    let c = graph::Compiled::new(w, device(), |w, k| {
        graph::classifier::build_script(w, k[0], k[1])
    });
    let got = &c
        .run(
            &[widths.len(), width],
            &[("x", &x), ("mask", &mask), ("inv_count", &inv)],
        )
        .unwrap()[0];
    let d = max_abs(got, &want);
    eprintln!("script: max |Δp| {d:.2e}");
    assert!(d < 1e-5, "script drift {d}");
}

#[test]
fn orient_matches_onnx() {
    let (up, native) = need_dirs!();
    let w = Weights::load(&native.join("orient.safetensors")).unwrap();
    let side = 768;
    let x = ink(1, side, side, 3);
    let mut s = session(up.join("orient.onnx"));
    let xs = Tensor::from_array(([1usize, 1, side, side], x.clone())).unwrap();
    let outs = s.run(ort::inputs!["x" => xs]).unwrap();
    let want = outs["p"].try_extract_tensor::<f32>().unwrap().1.to_vec();
    let c = graph::Compiled::new(w, device(), |w, k| {
        graph::classifier::build_orient(w, 1, k[0])
    });
    let got = &c.run(&[side], &[("x", &x)]).unwrap()[0];
    let d = max_abs(got, &want);
    eprintln!("orient: max |Δp| {d:.2e}");
    assert!(d < 1e-5, "orient drift {d}");
}

#[test]
fn order_matches_onnx_padded() {
    let (up, native) = need_dirs!();
    let w = Weights::load(&native.join("order.safetensors")).unwrap();
    let n = 13;
    let padded = 16;
    // Plausible line features: boxes stacked down a page, text features in [0, 1].
    let mut feats = vec![0f32; n * 16];
    for i in 0..n {
        let f = &mut feats[i * 16..(i + 1) * 16];
        let y0 = 0.05 + 0.07 * i as f32;
        let x0 = if i % 3 == 0 { 0.1 } else { 0.55 };
        let (x1, y1) = (x0 + 0.35, y0 + 0.02);
        f[..10].copy_from_slice(&[
            x0,
            y0,
            x1,
            y1,
            (x0 + x1) / 2.,
            (y0 + y1) / 2.,
            x1 - x0,
            0.0,
            (17.5f32).ln() / 4.0,
            0.4,
        ]);
        f[10] = 0.6;
        f[11] = (i % 4) as f32 * 0.1;
        f[14] = (i % 2) as f32;
        f[15] = 1.0;
    }
    let mut s = session(up.join("order.onnx"));
    let fs = Tensor::from_array(([1usize, n, 16], feats.clone())).unwrap();
    let ms = Tensor::from_array(([1usize, n], vec![true; n])).unwrap();
    let outs = s.run(ort::inputs!["feats" => fs, "mask" => ms]).unwrap();
    let want: Vec<Vec<f32>> = ["a", "cls", "same", "tab"]
        .iter()
        .map(|k| outs[*k].try_extract_tensor::<f32>().unwrap().1.to_vec())
        .collect();

    let mut pf = feats.clone();
    pf.resize(padded * 16, 0.0);
    let mut bias = vec![0f32; padded];
    bias[n..].fill(graph::order::PAD_BIAS);
    let c = graph::Compiled::new(w, device(), |w, k| graph::order::build(w, k[0]));
    let got = c
        .run(&[padded], &[("feats", &pf), ("key_bias", &bias)])
        .unwrap();
    let crop = |m: &[f32], planes: usize| -> Vec<f32> {
        let mut out = Vec::new();
        for p in 0..planes {
            for i in 0..n {
                out.extend_from_slice(&m[(p * padded + i) * padded..(p * padded + i) * padded + n]);
            }
        }
        out
    };
    let a = crop(&got[0], 1);
    let cls = got[1][..n * 10].to_vec();
    let same = crop(&got[2], 1);
    let tab = crop(&got[3], 3);
    for (name, g, w) in [
        ("a", &a, &want[0]),
        ("cls", &cls, &want[1]),
        ("same", &same, &want[2]),
        ("tab", &tab, &want[3]),
    ] {
        let d = max_abs(g, w);
        eprintln!("order {name}: max |Δ| {d:.2e}");
        assert!(d < 1e-3, "order {name} drift {d}");
    }
}
