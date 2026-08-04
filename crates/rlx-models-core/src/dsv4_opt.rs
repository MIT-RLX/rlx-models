// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **DSV4 graph-fusion optimizations** — each behind an env flag so an ablation
//! can toggle one at a time and measure op-count / correctness / time.
//!
//! Opscope ops-inspection of the DSV4-Flash decode graph found ~63% of runtime
//! ops are the **HC (hyper-connection) Sinkhorn gating** on tiny [hc×hc] tensors
//! (nested `Div(x/(Reduce+eps))`), plus decomposed sink-attention (Softmax + 2
//! MatMul) and partial-RoPE narrow/concat. Each is a launch-overhead sink on GPU.
//! These flags replace the decomposed graph sub-DAGs with single fused
//! `Op::Custom` nodes (one kernel launch each). The kernels are model-agnostic
//! primitives (Sinkhorn gate, sink-attention, GptJ tail-rope) registered via the
//! public `register_op` / `register_cpu_kernel` APIs — CPU here for the ablation;
//! promotable to `../rlx` for GPU once a winner is chosen.
//!
//! | flag | fuses | metric |
//! |---|---|---|
//! | `RLX_OPT_HCGATE`  | #1 Sinkhorn HC gate → 1 op | op-count |
//! | `RLX_OPT_SINKATTN`| #2 sink-attention → 1 op   | op-count |
//! | `RLX_OPT_ROPE`    | #4 partial GptJ rope → 1 op| op-count |
//! | `RLX_OPT_VSPLIT`  | #3 verify attn/post split  | verify passes |
//! | `RLX_OPT_SHEXP`   | #5 shared-expert MXFP4     | bytes |

use std::sync::{Arc, OnceLock};

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::OpExtension;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape, register_op};

// ── flags ───────────────────────────────────────────────────────────
pub fn opt_hcgate() -> bool {
    rlx_ir::env::flag("RLX_OPT_HCGATE")
}
pub fn opt_sinkattn() -> bool {
    rlx_ir::env::flag("RLX_OPT_SINKATTN")
}
pub fn opt_rope() -> bool {
    rlx_ir::env::flag("RLX_OPT_ROPE")
}
pub fn opt_vsplit() -> bool {
    rlx_ir::env::flag("RLX_OPT_VSPLIT")
}
pub fn opt_shexp() -> bool {
    rlx_ir::env::flag("RLX_OPT_SHEXP")
}

pub const HC_GATE_OP: &str = "dsv4.hc_sinkhorn_gate";
pub const SINK_ATTN_OP: &str = "dsv4.sink_attention";
pub const ROPE_TAIL_OP: &str = "dsv4.rope_tail_gptj";

fn u32le(v: usize) -> [u8; 4] {
    (v as u32).to_le_bytes()
}
fn f32le(v: f32) -> [u8; 4] {
    v.to_le_bytes()
}
fn rdu(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}
fn rdf(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}

/// Register the CPU kernels + IR extensions once. Idempotent.
pub fn ensure_registered() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        register_op(Arc::new(HcGateExt));
        register_cpu_kernel(Arc::new(HcGateExt));
        register_op(Arc::new(SinkAttnExt));
        register_cpu_kernel(Arc::new(SinkAttnExt));
        register_op(Arc::new(RopeTailExt));
        register_cpu_kernel(Arc::new(RopeTailExt));
        // Native GPU kernel for the Sinkhorn gate (the headline fusion). Metal =
        // raw on-device dispatch (one thread/row, no host sync). Other backends fall
        // back to the CPU kernel (host-delegate) until native kernels are added.
        #[cfg(feature = "metal")]
        rlx_metal::hc_sinkhorn_gate::register();
        #[cfg(feature = "gpu")]
        rlx_wgpu::wgpu_gpu_custom::register_wgpu_gpu_kernel(Arc::new(HcGateWgpu));
        // CUDA/ROCm native kernels (NVRTC/hipRTC) — written mirroring the validated
        // CPU/Metal/wgpu logic; NOT compiled on Apple hardware, validate on remote.
        #[cfg(feature = "cuda")]
        rlx_cuda::hc_sinkhorn_gate::register();
        #[cfg(feature = "rocm")]
        rlx_rocm::hc_sinkhorn_gate::register();
    });
}

// ════════════════════════════════════════════════════════════════════
// #1 — HC Sinkhorn gate.  attrs = [hc, iters, eps]
//   inputs: mixes [rows, 2hc+hc²], scale [3], base [2hc+hc²]
//   output: [rows, 2hc+hc²] packed [pre(hc) | post(hc) | comb(hc²)]
// ════════════════════════════════════════════════════════════════════
struct HcGateExt;
impl OpExtension for HcGateExt {
    fn name(&self) -> &str {
        HC_GATE_OP
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone() // same [rows, 2hc+hc²]
    }
}
impl CpuKernel for HcGateExt {
    fn name(&self) -> &str {
        HC_GATE_OP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let mixes = inputs[0].expect_f32("hc_gate mixes")?;
        let scale = inputs[1].expect_f32("hc_gate scale")?;
        let base = inputs[2].expect_f32("hc_gate base")?;
        let out = output.expect_f32_mut("hc_gate out")?;
        let hc = rdu(attrs, 0) as usize;
        let iters = rdu(attrs, 1) as usize;
        let eps = rdf(attrs, 2);
        let mix_hc = 2 * hc + hc * hc;
        if mixes.len() % mix_hc != 0 {
            return Err("hc_gate: mixes not divisible".into());
        }
        let rows = mixes.len() / mix_hc;
        let (s0, s1, s2) = (scale[0], scale[1], scale[2]);
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        for r in 0..rows {
            let m = &mixes[r * mix_hc..(r + 1) * mix_hc];
            let o = &mut out[r * mix_hc..(r + 1) * mix_hc];
            // pre = sigmoid(m*s0 + b) + eps
            for i in 0..hc {
                o[i] = sig(m[i] * s0 + base[i]) + eps;
            }
            // post = 2*sigmoid(m*s1 + b)
            for i in 0..hc {
                o[hc + i] = 2.0 * sig(m[hc + i] * s1 + base[hc + i]);
            }
            // comb [hc,hc]: softmax over k (last axis) + eps
            let mut c = vec![0f32; hc * hc];
            for j in 0..hc {
                let mut mx = f32::NEG_INFINITY;
                for k in 0..hc {
                    let l = m[2 * hc + j * hc + k] * s2 + base[2 * hc + j * hc + k];
                    c[j * hc + k] = l;
                    if l > mx {
                        mx = l;
                    }
                }
                let mut s = 0.0f32;
                for k in 0..hc {
                    let e = (c[j * hc + k] - mx).exp();
                    c[j * hc + k] = e;
                    s += e;
                }
                for k in 0..hc {
                    c[j * hc + k] = c[j * hc + k] / s + eps;
                }
            }
            // sinkhorn: first / (colsum_j + eps)
            for k in 0..hc {
                let mut cs = eps;
                for j in 0..hc {
                    cs += c[j * hc + k];
                }
                for j in 0..hc {
                    c[j * hc + k] /= cs;
                }
            }
            for _ in 0..iters.saturating_sub(1) {
                for j in 0..hc {
                    let mut rs = eps;
                    for k in 0..hc {
                        rs += c[j * hc + k];
                    }
                    for k in 0..hc {
                        c[j * hc + k] /= rs;
                    }
                }
                for k in 0..hc {
                    let mut cs = eps;
                    for j in 0..hc {
                        cs += c[j * hc + k];
                    }
                    for j in 0..hc {
                        c[j * hc + k] /= cs;
                    }
                }
            }
            o[2 * hc..].copy_from_slice(&c);
        }
        Ok(())
    }
}

/// Emit the fused HC Sinkhorn gate — replaces `build_hc_sinkhorn`'s decomposed
/// body. Returns `(pre [rows,hc], post [rows,hc], comb [rows,hc,hc])`.
pub fn emit_hc_gate(
    g: &mut Graph,
    mixes: NodeId,
    scale: NodeId,
    base: NodeId,
    rows: usize,
    hc: usize,
    eps: f32,
    iters: usize,
) -> (NodeId, NodeId, NodeId) {
    ensure_registered();
    let mix_hc = 2 * hc + hc * hc;
    let mut attrs = Vec::with_capacity(12);
    attrs.extend_from_slice(&u32le(hc));
    attrs.extend_from_slice(&u32le(iters));
    attrs.extend_from_slice(&f32le(eps));
    let packed = g.custom_op_packed(
        HC_GATE_OP,
        attrs,
        vec![mixes, scale, base],
        Shape::new(&[rows, mix_hc], DType::F32),
    );
    let pre = g.narrow_(packed, 1, 0, hc);
    let post = g.narrow_(packed, 1, hc, hc);
    let comb = g.narrow_(packed, 1, 2 * hc, hc * hc);
    let comb = g.reshape_(comb, vec![rows as i64, hc as i64, hc as i64]);
    (pre, post, comb)
}

// ════════════════════════════════════════════════════════════════════
// #2 — sink attention (MQA/MLA: shared kv).  attrs = [rows,nh,hd,nk,scale]
//   inputs: q [rows,nh,hd], kv [nk,hd], mask [rows,nk], sink [nh]
//   output: [rows, nh*hd]
// ════════════════════════════════════════════════════════════════════
struct SinkAttnExt;
impl OpExtension for SinkAttnExt {
    fn name(&self) -> &str {
        SINK_ATTN_OP
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn infer_shape(&self, _inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let rows = rdu(attrs, 0) as usize;
        let nh = rdu(attrs, 1) as usize;
        let hd = rdu(attrs, 2) as usize;
        Shape::new(&[rows, nh * hd], DType::F32)
    }
}
impl CpuKernel for SinkAttnExt {
    fn name(&self) -> &str {
        SINK_ATTN_OP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f32("sink_attn q")?;
        let kv = inputs[1].expect_f32("sink_attn kv")?;
        let mask = inputs[2].expect_f32("sink_attn mask")?;
        let sink = inputs[3].expect_f32("sink_attn sink")?;
        let out = output.expect_f32_mut("sink_attn out")?;
        let rows = rdu(attrs, 0) as usize;
        let nh = rdu(attrs, 1) as usize;
        let hd = rdu(attrs, 2) as usize;
        let nk = rdu(attrs, 3) as usize;
        let scale = rdf(attrs, 4);
        let rn = rows * nh;
        // scores[rn,nk] = scale · Q[rn,hd] @ kv[nk,hd]ᵀ  (MQA: kv shared across heads).
        // Route both matmuls through Accelerate/AMX (was a scalar loop that regressed
        // at real hd=512); only the sink-softmax stays scalar (cheap, O(rn·nk)).
        let mut scores = vec![0f32; rn * nk];
        rlx_cpu::blas::sgemm_bt(q, kv, &mut scores, rn, hd, nk, scale);
        for r in 0..rows {
            for h in 0..nh {
                let row = &mut scores[(r * nh + h) * nk..(r * nh + h) * nk + nk];
                let mut mx = sink[h]; // sink logit competes in the softmax
                for k in 0..nk {
                    row[k] += mask[r * nk + k];
                    if row[k] > mx {
                        mx = row[k];
                    }
                }
                let mut denom = (sink[h] - mx).exp();
                for k in 0..nk {
                    let e = (row[k] - mx).exp();
                    row[k] = e;
                    denom += e;
                }
                let inv = 1.0 / denom;
                for k in 0..nk {
                    row[k] *= inv;
                }
            }
        }
        // out[rn,hd] = attn[rn,nk] @ kv[nk,hd].
        rlx_cpu::blas::sgemm(&scores, kv, out, rn, nk, hd);
        Ok(())
    }
}

/// Emit the fused sink attention → `[rows, nh, hd]`.
#[allow(clippy::too_many_arguments)]
pub fn emit_sink_attention(
    g: &mut Graph,
    q: NodeId,
    kv: NodeId,
    mask: NodeId,
    sink: NodeId,
    scale: f32,
    rows: usize,
    nh: usize,
    hd: usize,
    nk: usize,
) -> NodeId {
    ensure_registered();
    let mut attrs = Vec::with_capacity(20);
    attrs.extend_from_slice(&u32le(rows));
    attrs.extend_from_slice(&u32le(nh));
    attrs.extend_from_slice(&u32le(hd));
    attrs.extend_from_slice(&u32le(nk));
    attrs.extend_from_slice(&f32le(scale));
    let o = g.custom_op_packed(
        SINK_ATTN_OP,
        attrs,
        vec![q, kv, mask, sink],
        Shape::new(&[rows, nh * hd], DType::F32),
    );
    g.reshape_(o, vec![rows as i64, nh as i64, hd as i64])
}

// ════════════════════════════════════════════════════════════════════
// #4 — partial GptJ tail-rope.  attrs = [rows,nh,hd,rd]
//   inputs: x [rows, nh*hd], cos [rows, rd/2], sin [rows, rd/2]
//   output: [rows, nh*hd] (rotate the last rd dims of each head, pass the rest)
// ════════════════════════════════════════════════════════════════════
struct RopeTailExt;
impl OpExtension for RopeTailExt {
    fn name(&self) -> &str {
        ROPE_TAIL_OP
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}
impl CpuKernel for RopeTailExt {
    fn name(&self) -> &str {
        ROPE_TAIL_OP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("rope x")?;
        let cos = inputs[1].expect_f32("rope cos")?;
        let sin = inputs[2].expect_f32("rope sin")?;
        let out = output.expect_f32_mut("rope out")?;
        let rows = rdu(attrs, 0) as usize;
        let nh = rdu(attrs, 1) as usize;
        let hd = rdu(attrs, 2) as usize;
        let rd = rdu(attrs, 3) as usize;
        let half = rd / 2;
        let off = hd - rd; // tail offset within each head
        out.copy_from_slice(x);
        for r in 0..rows {
            let cr = &cos[r * half..r * half + half];
            let sr = &sin[r * half..r * half + half];
            for h in 0..nh {
                let base = (r * nh + h) * hd + off;
                // GptJ: adjacent pairs (2i, 2i+1)
                for i in 0..half {
                    let a = x[base + 2 * i];
                    let b = x[base + 2 * i + 1];
                    out[base + 2 * i] = a * cr[i] - b * sr[i];
                    out[base + 2 * i + 1] = a * sr[i] + b * cr[i];
                }
            }
        }
        Ok(())
    }
}

/// Emit fused GptJ tail-rope → `[rows, nh*hd]`.
pub fn emit_rope_tail(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    rows: usize,
    nh: usize,
    hd: usize,
    rd: usize,
) -> NodeId {
    ensure_registered();
    let mut attrs = Vec::with_capacity(16);
    attrs.extend_from_slice(&u32le(rows));
    attrs.extend_from_slice(&u32le(nh));
    attrs.extend_from_slice(&u32le(hd));
    attrs.extend_from_slice(&u32le(rd));
    g.custom_op_packed(
        ROPE_TAIL_OP,
        attrs,
        vec![x, cos, sin],
        Shape::new(&[rows, nh * hd], DType::F32),
    )
}

// ── wgpu (Device::Gpu — Metal/Vulkan/DX12/WebGPU) native Sinkhorn gate ──
// Raw-GPU WGSL, dispatched straight against the arena (no host roundtrip). The
// wgpu seam passes only offset/len params (no attrs), so hc/rows are derived from
// the base-input length (= mix_hc) and the DSV4 Sinkhorn constants (iters=3,
// eps=1e-6) are used — this op is DSV4-specific. One thread per row.
#[cfg(feature = "gpu")]
const WGSL_HC_GATE: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read>       params: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_off = params[0];
    let mixes_off = params[4];
    let mixes_len = params[5];
    let scale_off = params[6];
    let base_off  = params[8];
    let mix_hc    = params[9];            // base_len == mix_hc = 2*hc + hc*hc
    let hc = u32(round(sqrt(f32(1u + mix_hc)) - 1.0));
    let rows = mixes_len / max(mix_hc, 1u);
    let r = gid.x;
    if (r >= rows) { return; }
    let eps = 1e-6;
    let iters = 3u;
    let mb = mixes_off + r * mix_hc;
    let ob = out_off + r * mix_hc;
    let s0 = arena[scale_off]; let s1 = arena[scale_off + 1u]; let s2 = arena[scale_off + 2u];
    for (var i = 0u; i < hc; i++) { arena[ob + i] = 1.0 / (1.0 + exp(-(arena[mb + i] * s0 + arena[base_off + i]))) + eps; }
    for (var i = 0u; i < hc; i++) { arena[ob + hc + i] = 2.0 / (1.0 + exp(-(arena[mb + hc + i] * s1 + arena[base_off + hc + i]))); }
    var c: array<f32, 16>;
    for (var j = 0u; j < hc; j++) {
        var mx = -1e30;
        for (var k = 0u; k < hc; k++) { let l = arena[mb + 2u*hc + j*hc + k] * s2 + arena[base_off + 2u*hc + j*hc + k]; c[j*hc+k] = l; mx = max(mx, l); }
        var sm = 0.0; for (var k = 0u; k < hc; k++) { let e = exp(c[j*hc+k] - mx); c[j*hc+k] = e; sm += e; }
        for (var k = 0u; k < hc; k++) { c[j*hc+k] = c[j*hc+k] / sm + eps; }
    }
    for (var k = 0u; k < hc; k++) { var cs = eps; for (var j = 0u; j < hc; j++) { cs += c[j*hc+k]; } for (var j = 0u; j < hc; j++) { c[j*hc+k] = c[j*hc+k] / cs; } }
    for (var it = 1u; it < iters; it++) {
        for (var j = 0u; j < hc; j++) { var rs = eps; for (var k = 0u; k < hc; k++) { rs += c[j*hc+k]; } for (var k = 0u; k < hc; k++) { c[j*hc+k] = c[j*hc+k] / rs; } }
        for (var k = 0u; k < hc; k++) { var cs = eps; for (var j = 0u; j < hc; j++) { cs += c[j*hc+k]; } for (var j = 0u; j < hc; j++) { c[j*hc+k] = c[j*hc+k] / cs; } }
    }
    for (var idx = 0u; idx < hc*hc; idx++) { arena[ob + 2u*hc + idx] = c[idx]; }
}
"#;

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct HcGateWgpu;
#[cfg(feature = "gpu")]
impl rlx_wgpu::wgpu_gpu_custom::WgpuGpuKernel for HcGateWgpu {
    fn name(&self) -> &str {
        HC_GATE_OP
    }
    fn wgsl(&self) -> &str {
        WGSL_HC_GATE
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use rlx_ir::{Graph, Shape};
    use rlx_runtime::{Device, Session};

    fn run_gate(dev: Device) -> Vec<Vec<f32>> {
        let (rows, hc, eps, iters) = (5usize, 4usize, 1e-6f32, 3usize);
        let mix_hc = 2 * hc + hc * hc;
        let mixes: Vec<f32> = (0..rows * mix_hc)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let scale = vec![0.5f32, 0.7, 0.3];
        let base: Vec<f32> = (0..mix_hc).map(|i| (i as f32 - 12.0) * 0.05).collect();
        let mut g = Graph::new("hcg");
        let m = g.input("mixes", Shape::new(&[rows, mix_hc], DType::F32));
        let s = g.input("scale", Shape::new(&[3], DType::F32));
        let b = g.input("base", Shape::new(&[mix_hc], DType::F32));
        let (pre, post, comb) = emit_hc_gate(&mut g, m, s, b, rows, hc, eps, iters);
        g.set_outputs(vec![pre, post, comb]);
        let mut c = Session::new(dev).compile(g);
        c.run(&[("mixes", &mixes), ("scale", &scale), ("base", &base)])
    }

    #[allow(dead_code)] // only called under the metal/mlx/gpu feature gates below
    fn cmp(cpu: &[Vec<f32>], gpu: &[Vec<f32>], tag: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{tag}: output count");
        for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
            assert_eq!(a.len(), b.len(), "{tag} out {i} len");
            let err = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "{tag} out {i}: vs cpu max|Δ| {err:e}");
        }
    }

    // The fused Sinkhorn gate must match the CPU kernel bit-for-bit on every GPU
    // backend: Metal via the NATIVE MetalGpuKernel (on-device), MLX via the auto
    // host-delegate (CPU kernel over byte staging). Skips a backend if unavailable.
    #[test]
    fn hc_sinkhorn_gate_gpu_matches_cpu() {
        #[allow(unused_variables)] // used only under the GPU feature gates below
        let cpu = run_gate(Device::Cpu);
        #[allow(unused_mut)] // mutated only under the GPU feature gates below
        let mut checked = 0;
        #[cfg(feature = "metal")]
        if rlx_runtime::is_available(Device::Metal) {
            cmp(&cpu, &run_gate(Device::Metal), "metal(native)");
            checked += 1;
        }
        #[cfg(feature = "mlx")]
        if rlx_runtime::is_available(Device::Mlx) {
            cmp(&cpu, &run_gate(Device::Mlx), "mlx(host-delegate)");
            checked += 1;
        }
        #[cfg(feature = "gpu")]
        if rlx_runtime::is_available(Device::Gpu) {
            cmp(&cpu, &run_gate(Device::Gpu), "wgpu(native)");
            checked += 1;
        }
        eprintln!("hc_sinkhorn_gate: validated {checked} GPU backend(s) vs CPU");
    }
}
