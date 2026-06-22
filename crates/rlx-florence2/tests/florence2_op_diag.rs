// RLX — Florence-2 GPU op diagnostic.
//
// Exercises the non-trivial DaViT ops (6-D window partition reshape/permute,
// 4-D batched matmul for channel attention, mean+keepdim, concat) on a tiny
// graph and compares each GPU backend to the CPU result, to localise backend
// coverage gaps. Weight-free; run with `--features apple-silicon`.

#![cfg(any(feature = "metal", feature = "mlx"))]

use rlx_core::flow_util::{built_from_hir_with_profile, compile_built};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut};
use rlx_ir::op::MaskKind;
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

fn run(device: Device, build: &str) -> Vec<f32> {
    let f = DType::F32;
    let mut hir = HirModule::new("diag").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    match build {
        "tr_reshape" | "tr_mul" => {
            params.insert("one".to_string(), vec![1.0f32]);
            params.insert(
                "tr".to_string(),
                (0..32 * 2304 * 32)
                    .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
                    .collect(),
            );
        }
        "bmm3d_b32" => {
            params.insert(
                "a32".to_string(),
                (0..32 * 32 * 2304)
                    .map(|i| (((i % 127) as f32 - 63.0) / 63.0) * 0.0208)
                    .collect(),
            );
            params.insert(
                "b32".to_string(),
                (0..32 * 2304 * 32)
                    .map(|i| ((i % 113) as f32 - 56.0) / 56.0)
                    .collect(),
            );
        }
        "bmm3d_b16" => {
            params.insert(
                "a16".to_string(),
                (0..16 * 32 * 2304)
                    .map(|i| (((i % 127) as f32 - 63.0) / 63.0) * 0.0208)
                    .collect(),
            );
            params.insert(
                "b16".to_string(),
                (0..16 * 2304 * 32)
                    .map(|i| ((i % 113) as f32 - 56.0) / 56.0)
                    .collect(),
            );
        }
        "chanattn_full" => {
            params.insert(
                "cx".to_string(),
                (0..2304 * 1024)
                    .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
                    .collect(),
            );
            params.insert(
                "cqkvw".to_string(),
                (0..1024 * 3072)
                    .map(|i| ((i % 127) as f32 - 63.0) / 630.0)
                    .collect(),
            );
            params.insert(
                "cprojw".to_string(),
                (0..1024 * 1024)
                    .map(|i| ((i % 113) as f32 - 56.0) / 560.0)
                    .collect(),
            );
            params.insert("csc".to_string(), vec![0.0208f32]);
        }
        "permute_mm" => {
            params.insert(
                "qkv".to_string(),
                (0..64 * 3 * 32 * 4)
                    .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
                    .collect(),
            );
        }
        "meancat_big" | "meancat_big_mat" | "concat_big" | "mean_big" => {
            params.insert(
                "ca".to_string(),
                (0..2048).map(|i| (i as f32) / 2048.0).collect(),
            );
            params.insert(
                "xb".to_string(),
                (0..576 * 2048)
                    .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
                    .collect(),
            );
            params.insert("one".to_string(), vec![1.0f32]);
        }
        "permute5d" => {
            params.insert(
                "p5".to_string(),
                (0..8 * 3 * 32 * 4).map(|i| (i as f32) * 0.001).collect(),
            );
        }
        "convstrided" => {
            params.insert(
                "w".to_string(),
                (0..6 * 6 * 2 * 2).map(|i| (i as f32) * 0.003).collect(),
            );
        }
        "convdw" => {
            params.insert(
                "w".to_string(),
                (0..6 * 1 * 3 * 3).map(|i| (i as f32) * 0.01).collect(),
            );
        }
        _ => {}
    }
    // input [1, 4, 4, 6] flattened as [1,16,6] tokens (B=1,H=W=4,C=6)
    let x = hir.input("x", Shape::new(&[1, 16, 6], f));
    let out = {
        let mut g = HirMut::new(&mut hir);
        match build {
            // 6-D window partition (ws=2) round-trip → [1,16,6]
            "window6d" => {
                let x4 = g.reshape_(x, vec![1, 4, 4, 6]);
                let x6 = g.reshape_(x4, vec![1, 2, 2, 2, 2, 6]);
                let p = g.transpose_(x6, vec![0, 1, 3, 2, 4, 5]);
                let win = g.reshape_(p, vec![4, 4, 6]);
                // reverse
                let w6 = g.reshape_(win, vec![1, 2, 2, 2, 2, 6]);
                let wp = g.transpose_(w6, vec![0, 1, 3, 2, 4, 5]);
                g.reshape_(wp, vec![1, 16, 6])
            }
            // 4-D batched matmul: [1,2,16,3] @ [1,2,3,16] -> [1,2,16,16]
            "bmm4d" => {
                let a = g.reshape_(x, vec![1, 2, 16, 3]);
                let at = g.transpose_(a, vec![0, 1, 3, 2]); // [1,2,3,16]
                let m = g.mm(a, at); // [1,2,16,16]
                g.reshape_(m, vec![1, 16, 16])
            }
            // 3-D batched matmul: [2,16,3] @ [2,3,16] -> [2,16,16]
            "bmm3d" => {
                let a = g.reshape_(x, vec![2, 16, 3]);
                let at = g.transpose_(a, vec![0, 2, 1]); // [2,3,16]
                let m = g.mm(a, at); // [2,16,16]
                g.reshape_(m, vec![1, 16, 16 * 2])
            }
            // strided conv: NCHW [1,6,4,4] w[6,6,2,2] s2 p0 -> [1,6,2,2]
            "convstrided" => {
                let nchw = g.reshape_(x, vec![1, 6, 4, 4]); // reuse 16*6 as 6*16
                let w = g.param("w", Shape::new(&[6, 6, 2, 2], f));
                let y = g.conv2d(
                    nchw,
                    w,
                    [2, 2],
                    [2, 2],
                    [0, 0],
                    1,
                    Shape::new(&[1, 6, 2, 2], f),
                );
                g.reshape_(y, vec![1, 4, 6])
            }
            // depthwise conv: NCHW [1,6,4,4] w[6,1,3,3] s1 p1 groups=6 -> [1,6,4,4]
            "convdw" => {
                let nchw = g.reshape_(x, vec![1, 6, 4, 4]);
                let w = g.param("w", Shape::new(&[6, 1, 3, 3], f));
                let y = g.conv2d(
                    nchw,
                    w,
                    [3, 3],
                    [1, 1],
                    [1, 1],
                    6,
                    Shape::new(&[1, 6, 4, 4], f),
                );
                g.reshape_(y, vec![1, 16, 6])
            }
            // Exact stage-2 channel-attn bmm: [32,32,2304] @ [32,2304,32] -> [32,32,32].
            "bmm3d_b32" => {
                let a = g.param("a32", Shape::new(&[32, 32, 2304], f)); // [Bg,Cg,N]
                let b2 = g.param("b32", Shape::new(&[32, 2304, 32], f)); // [Bg,N,Cg]
                let m = g.mm(a, b2); // [32,32,32]
                g.reshape_(m, vec![1, 32 * 32, 32])
            }
            // Same contraction at batch 16 (stage-1 channel-attn batch).
            "bmm3d_b16" => {
                let a = g.param("a16", Shape::new(&[16, 32, 2304], f));
                let b2 = g.param("b16", Shape::new(&[16, 2304, 32], f));
                let m = g.mm(a, b2);
                g.reshape_(m, vec![1, 16 * 32, 32])
            }
            // Real channel-attn pattern: transposed-view LHS into mm, then softmax.
            // q[32,2304,32] -> q_t[32,32,2304]; attn = mm(q_t, k[32,2304,32]) [32,32,32];
            // softmax(-1); out = mm(attn, v_t[32,32,2304]) [32,32,2304].
            "chanattn32" => {
                let q = g.param("a32", Shape::new(&[32, 2304, 32], f)); // [Bg,N,Cg]
                let k = g.param("b32", Shape::new(&[32, 2304, 32], f));
                let qt = g.transpose_(q, vec![0, 2, 1]); // [Bg,Cg,N] (view)
                let attn = g.mm(qt, k); // [Bg,Cg,Cg]
                let attn = g.sm(attn, -1);
                let vt = g.transpose_(k, vec![0, 2, 1]); // [Bg,Cg,N]
                let out = g.mm(attn, vt); // [Bg,Cg,N]
                g.reshape_(out, vec![1, 32 * 32, 2304])
            }
            // qkv split via 5-D permute → slice → reshape → mm (real channel-attn
            // front half). Tests mm of a non-contiguous reshape-of-permute on GPU.
            "permute_mm" => {
                let qkv = g.param("qkv", Shape::new(&[1, 64, 3 * 32 * 4], f)); // [b,nn,3C]
                let r = g.reshape_(qkv, vec![1, 64, 3, 32, 4]); // [b,nn,3,g,cg]
                let p = g.transpose_(r, vec![2, 0, 3, 1, 4]); // [3,b,g,nn,cg]
                let qn = g.narrow_(p, 0, 0, 1);
                let kn = g.narrow_(p, 0, 1, 1);
                let q = g.reshape_(qn, vec![32, 64, 4]); // [bg,nn,cg]
                let k = g.reshape_(kn, vec![32, 64, 4]);
                let qt = g.transpose_(q, vec![0, 2, 1]); // [bg,cg,nn]
                let m = g.mm(qt, k); // [bg,cg,cg]
                g.reshape_(m, vec![1, 32 * 4, 4])
            }
            // EXACT channel_attention at stage-2 size (b=1,N=2304,C=1024,g=32).
            "chanattn_full" => {
                let xin = g.param("cx", Shape::new(&[1, 2304, 1024], f));
                let qkvw = g.param("cqkvw", Shape::new(&[1024, 3072], f));
                let projw = g.param("cprojw", Shape::new(&[1024, 1024], f));
                let qkv = g.mm(xin, qkvw); // [1,2304,3072]
                let qkv5 = g.reshape_(qkv, vec![1, 2304, 3, 32, 32]);
                let qkv5p = g.transpose_(qkv5, vec![2, 0, 3, 1, 4]); // [3,1,32,2304,32]
                let qn = g.narrow_(qkv5p, 0, 0, 1);
                let q = g.reshape_(qn, vec![32, 2304, 32]);
                let kn = g.narrow_(qkv5p, 0, 1, 1);
                let k = g.reshape_(kn, vec![32, 2304, 32]);
                let vn = g.narrow_(qkv5p, 0, 2, 1);
                let v = g.reshape_(vn, vec![32, 2304, 32]);
                let sc = g.param("csc", Shape::new(&[1], f));
                let q = g.mul(q, sc);
                let qt = g.transpose_(q, vec![0, 2, 1]);
                let attn0 = g.mm(qt, k);
                let attn = g.sm(attn0, -1);
                let vt = g.transpose_(v, vec![0, 2, 1]);
                let ov = g.mm(attn, vt);
                let out = g.transpose_(ov, vec![0, 2, 1]); // [32,2304,32]
                let out4 = g.reshape_(out, vec![1, 32, 2304, 32]);
                let mut slices = Vec::new();
                for gi in 0..32 {
                    let sl = g.narrow_(out4, 1, gi, 1);
                    slices.push(g.reshape_(sl, vec![1, 2304, 32]));
                }
                let merged = g.concat_(slices, 2); // [1,2304,1024]
                g.mm(merged, projw) // [1,2304,1024]
            }
            // Reshape after a 4-D transpose (channel-attn head merge):
            // [1,32,2304,32] -transpose[0,2,1,3]-> [1,2304,32,32] -reshape-> [1,2304,1024].
            "tr_reshape" => {
                let a = g.param("tr", Shape::new(&[1, 32, 2304, 32], f));
                let t = g.transpose_(a, vec![0, 2, 1, 3]); // [1,2304,32,32] (non-contiguous)
                g.reshape_(t, vec![1, 2304, 1024])
            }
            // Transpose then element-wise (forces materialization) then reshape.
            "tr_mul" => {
                let a = g.param("tr", Shape::new(&[1, 32, 2304, 32], f));
                let t = g.transpose_(a, vec![0, 2, 1, 3]); // [1,2304,32,32]
                let one = g.param("one", Shape::new(&[1], f));
                let m = g.mul(t, one);
                g.reshape_(m, vec![1, 2304, 1024])
            }
            // 5-D permute [2,0,3,1,4] (channel-attn qkv split) at g=32.
            "permute5d" => {
                let p = g.param("p5", Shape::new(&[1, 8, 3, 32, 4], f)); // [b,nn,3,g,cg]
                let pp = g.transpose_(p, vec![2, 0, 3, 1, 4]); // [3,1,32,8,4]
                let s = g.narrow_(pp, 0, 0, 1);
                g.reshape_(s, vec![32, 8, 4])
            }
            // windowed attention op: batch=4 windows, seq=4, C=6, heads=2.
            "winattn" => {
                let q = g.reshape_(x, vec![4, 4, 6]);
                let attn = g.add_node(
                    attention_kind_op(2, 3, MaskKind::None, None, None),
                    vec![q, q, q],
                    Shape::new(&[4, 4, 6], f),
                );
                g.reshape_(attn, vec![1, 16, 6])
            }
            // mean keepdim over tokens + concat
            "meancat" => {
                let m = g.mean(x, vec![1], true); // [1,1,6]
                g.concat_(vec![m, x], 1) // [1,17,6]
            }
            // Real projection pooling: mean+keepdim then concat at [1,576,2048].
            "meancat_big" => {
                let xb = g.param("xb", Shape::new(&[1, 576, 2048], f));
                let m = g.mean(xb, vec![1], true); // [1,1,2048]
                g.concat_(vec![m, xb], 1) // [1,577,2048]
            }
            // Same, but materialize the pooled token before concat (workaround test).
            "meancat_big_mat" => {
                let xb = g.param("xb", Shape::new(&[1, 576, 2048], f));
                let m = g.mean(xb, vec![1], true); // [1,1,2048]
                let one = g.param("one", Shape::new(&[1], f));
                let m2 = g.mul(m, one); // force a fresh contiguous buffer
                g.concat_(vec![m2, xb], 1)
            }
            // concat alone (no mean): [1,1,2048] ++ [1,576,2048] along axis 1.
            "concat_big" => {
                let a = g.param("ca", Shape::new(&[1, 1, 2048], f));
                let xb = g.param("xb", Shape::new(&[1, 576, 2048], f));
                g.concat_(vec![a, xb], 1)
            }
            // mean+keepdim alone at large N: [1,576,2048] -> [1,1,2048].
            "mean_big" => {
                let xb = g.param("xb", Shape::new(&[1, 576, 2048], f));
                g.mean(xb, vec![1], true)
            }
            _ => x,
        }
    };
    hir.outputs = vec![out];
    let built = built_from_hir_with_profile(hir, params, CompileProfile::encoder()).unwrap();
    let mut compiled = compile_built(built, device).unwrap();
    let input: Vec<f32> = (0..16 * 6).map(|i| i as f32 * 0.01).collect();
    compiled.run(&[("x", &input)]).into_iter().next().unwrap()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn diag(device: Device, name: &str) {
    // Regression guards for the DaViT op set. `tr_reshape`/`tr_mul` pinned an
    // rlx-metal tiled-transpose bug (head-merge reshape after a 4-D `[0,2,1,3]`
    // transpose); `concat_big` pinned an rlx-mlx concat-axis alignment bug
    // (size-1 leading operand). Both fixed upstream — every op must match CPU.
    for op in [
        "window6d",
        "bmm3d",
        "bmm3d_b16",
        "convstrided",
        "convdw",
        "winattn",
        "permute5d",
        "permute_mm",
        "chanattn_full",
        "mean_big",
        "meancat",
        "tr_reshape",
        "tr_mul",
        "concat_big",
    ] {
        let cpu = run(Device::Cpu, op);
        let gpu = run(device, op);
        assert_eq!(
            cpu.len(),
            gpu.len(),
            "[{name}] {op}: length cpu={} gpu={}",
            cpu.len(),
            gpu.len()
        );
        let d = maxdiff(&cpu, &gpu);
        eprintln!("[diag/{name}] {op}: maxdiff={d:.6}");
        assert!(d < 1e-3, "[{name}] {op}: maxdiff {d} exceeds 1e-3");
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn diag_metal() {
    diag(Device::Metal, "metal");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn diag_mlx() {
    diag(Device::Mlx, "mlx");
}
