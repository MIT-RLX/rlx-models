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

//! Bridge between `rlx-models` loaders/runtime and `rlx-flow`.

use std::path::Path;

use rlx_flow::CompileProfile;
use rlx_flow::{
    BuiltModel, FusionTargetKind, MixedPrecisionKind, ModelExecutionConfig, PrecisionKind,
};
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_opt::{FusionOptions, FusionTarget, PrecisionPolicy};
use rlx_runtime::Device;
use rlx_runtime::{CompileOptions, ModelCompilePipeline, Precision, Session, stages};

use crate::weight_loader::WeightLoader;

/// Adapt [`WeightLoader`] to [`rlx_flow::WeightSource`].
pub struct WeightLoaderSource<'a>(pub &'a mut dyn WeightLoader);

impl rlx_flow::WeightSource for WeightLoaderSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        if transpose {
            self.0.take_transposed(key)
        } else {
            self.0.take(key)
        }
    }
    // `take_packed` intentionally left at the trait default (`None`): the plain
    // adapter always dequantizes to F32, so existing flows are unchanged.
}

/// Packed-aware [`WeightLoader`] → [`rlx_flow::WeightSource`] adapter.
///
/// Identical to [`WeightLoaderSource`] for F32 weights, but also serves GGUF/MLX
/// quant blobs via [`rlx_flow::WeightSource::take_packed`], so the flow builder
/// emits fused `DequantMatMul` projections instead of dequantizing to F32 — one
/// packed graph that runs at any `m` (m=1 decode, m=N prefill). Opt-in: only
/// flows explicitly built with this adapter go packed, so no existing path
/// changes behavior. Any model can adopt packed matmuls just by building its
/// flow through this wrapper.
pub struct PackedWeightLoaderSource<'a>(pub &'a mut dyn WeightLoader);

impl rlx_flow::WeightSource for PackedWeightLoaderSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        if transpose {
            self.0.take_transposed(key)
        } else {
            self.0.take(key)
        }
    }

    /// Hand the flow builder the GGUF/MLX quant blob so `linear` emits a fused
    /// `DequantMatMul` instead of dequantizing to F32. `PackedWeightTensor` is
    /// `(w_q, scheme, [out_dim, in_dim])`; bias (when any) is applied by the
    /// caller as a separate add, matching the F32 matmul path.
    fn take_packed(&mut self, key: &str) -> anyhow::Result<Option<rlx_flow::GgufPackedLinear>> {
        Ok(self.0.take_packed(key)?.map(|(w_q, scheme, shape)| {
            let out_dim = shape.first().copied().unwrap_or(0);
            let in_dim = shape.get(1).copied().unwrap_or(0);
            rlx_flow::GgufPackedLinear {
                w_q,
                scheme,
                in_dim,
                out_dim,
                bias: Vec::new(),
            }
        }))
    }
}

/// Load a tier-1 profile from disk; fall back to `default` when missing or invalid.
pub fn load_compile_profile(path: &Path, default: CompileProfile) -> CompileProfile {
    CompileProfile::from_toml_path(path).unwrap_or(default)
}

/// Load `profile_file` next to `weights` (parent directory); fall back to `default`.
pub fn profile_near_weights(
    weights: &Path,
    profile_file: &str,
    default: CompileProfile,
) -> CompileProfile {
    let dir = weights.parent().unwrap_or_else(|| Path::new("."));
    load_compile_profile(&dir.join(profile_file), default)
}

/// Apply tier-1 profile options to runtime compile options.
pub fn apply_compile_profile(profile: &CompileProfile, opts: &mut CompileOptions) {
    opts.dce = profile.passes.dce;
    opts.constant_folding = profile.passes.constant_folding;
    opts.verbose = profile.passes.verbose;
    opts.assert_fusion_clean = profile.fusion.assert_clean;
    opts.fusion_opts = FusionOptions {
        skip_fusion: profile.fusion.skip,
        unfuse_elementwise_regions: profile.backend.metal.unfuse_regions
            || profile.backend.cpu.unfuse_regions,
        ..FusionOptions::default()
    };
    if let Some(target) = fusion_target_from_profile(profile.fusion.target) {
        opts.fusion_target = Some(target);
    }
    opts.precision = match profile.precision.compute {
        PrecisionKind::F32 => Precision::F32,
        PrecisionKind::F16 => Precision::F16,
        PrecisionKind::Bf16 => Precision::F16, // closest supported runtime precision today
    };
    opts.policy = match profile.precision.mixed {
        MixedPrecisionKind::None => None,
        MixedPrecisionKind::Auto => Some(PrecisionPolicy::AutoMixed),
    };
    // Opt-in f16 residual stream (Metal / packed Q1 decode).
    //   RLX_QWEN35_AMP=1 | RLX_METAL_AMP=1           → AutoMixed (residual+RMS f16)
    //   RLX_QWEN35_AMP=conservative | RLX_METAL_AMP=conservative
    //                                                 → AutoMixedConservative (RMS f32)
    // GatedDeltaNet + packed DequantMatMul prefill (m>1) stay f32 inside the
    // AMP pass. Isolated Q1 decode GEMV+residual f16 is parity-checked in
    // rlx-metal; full Bonsai-27B AMP is still experimental (default off).
    match rlx_ir::env::var("RLX_QWEN35_AMP")
        .or_else(|| rlx_ir::env::var("RLX_METAL_AMP"))
        .as_deref()
        .map(str::trim)
    {
        Some("1") | Some("auto") | Some("full") => {
            opts.policy = Some(PrecisionPolicy::AutoMixed);
        }
        Some("conservative") | Some("c") => {
            opts.policy = Some(PrecisionPolicy::AutoMixedConservative);
        }
        _ if rlx_ir::env::flag("RLX_QWEN35_AMP") || rlx_ir::env::flag("RLX_METAL_AMP") => {
            opts.policy = Some(PrecisionPolicy::AutoMixed);
        }
        _ => {}
    }
}

/// Dynamic HIR template/specialize — default passes only (matches legacy `DynamicDimCompileCache`).
pub fn compile_options_dynamic(binding: rlx_ir::DimBinding) -> CompileOptions {
    CompileOptions::new().dim_binding(binding)
}

/// Build [`CompileOptions`] from a tier-1 profile + device fusion target.
pub fn compile_options_from_profile(
    profile: &CompileProfile,
    device: Device,
    kernel_dispatch: KernelDispatchConfig,
) -> CompileOptions {
    let mut opts = CompileOptions::new();
    apply_compile_profile(profile, &mut opts);
    opts.kernel_dispatch = kernel_dispatch;
    if opts.fusion_target.is_none() {
        opts.fusion_target = Some(stages::fusion_target_for(device));
    }
    opts
}

/// Tier-1 profile + device (no execution variant binding).
pub fn compile_options_for_profile(profile: &CompileProfile, device: Device) -> CompileOptions {
    compile_options_from_profile(profile, device, KernelDispatchConfig::default())
}

/// Compile options for packed GGUF K-quant prefill (`Op::DequantMatMul`).
///
/// Fusion is disabled on **all** backends: hand-rolled packed graphs emit separate
/// RMSNorm + `DequantMatMul` nodes, and fusing them into `Op::FusedResidualRmsNorm`
/// assumes F32 matmul weights — K-quant logits skew badly on CPU (cos ~ -0.2 vs
/// F32 dequant). GPU backends already needed this for lowering coverage; CPU/Metal/MLX
/// need it for numerical parity too.
pub fn compile_options_for_packed_gguf_prefill_with_profile(
    profile: &CompileProfile,
    device: Device,
) -> CompileOptions {
    let mut profile = profile.clone();
    profile.fusion.skip = true;
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

/// Llama-shaped LM packed GGUF prefill (MiniCPM5, Llama 3.2, …).
pub fn compile_options_for_packed_gguf_prefill(device: Device) -> CompileOptions {
    compile_options_for_packed_gguf_prefill_with_profile(&CompileProfile::llama32_prefill(), device)
}

/// Backend env overrides while compiling or running packed GGUF graphs.
///
/// - **Metal** — `RLX_DISABLE_MPSGRAPH=1` (MPSGraph mishandles GGUF `DequantMatMul`).
/// - **MLX** — `RLX_MLX_MODE=lazy` (GGUF `DequantMatMul` host-dequant cannot use `mlx::compile`).
///
/// MLX mode is baked into the executable at compile time; use this guard around every
/// `Session::compile_with` / bucketed decode compile for packed GGUF (`rlx-llama32`,
/// `rlx-qwen3`, `rlx-gemma`, …).
pub fn packed_gguf_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    with_packed_gguf_backend_env(device, f)
}

fn with_packed_gguf_backend_env<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    let mlx_prev = if device == Device::Mlx {
        let prev = rlx_ir::env::var("RLX_MLX_MODE");
        rlx_ir::env::set("RLX_MLX_MODE", "lazy");
        prev
    } else {
        None
    };
    let metal = device == Device::Metal;
    if metal {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    }
    let out = f();
    if metal {
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    }
    if device == Device::Mlx {
        match mlx_prev {
            Some(ref v) => rlx_ir::env::set("RLX_MLX_MODE", v),
            None => rlx_ir::env::unset("RLX_MLX_MODE"),
        }
    }
    out
}

/// Device used to compile/run packed GGUF graphs.
///
/// **CPU**, **Metal**, **MLX**, **CUDA / ROCm**, **wgpu**, **Vulkan**, and
/// **CoreML** (`Device::Ane`) run natively (incl. Q1_0). MLX uses host-side
/// GGUF dequant per `DequantMatMul` with lazy eval for most schemes (see
/// [`packed_gguf_compile_guard`]); **Q1_0** expands on-device via
/// `dequant_q1_0.metal` (no f32 weight cache — Bonsai-27B blow-up).
/// wgpu keeps packed weights in a separate buffer when the activation
/// arena would exceed WebGPU's ~4 GiB `max_buffer_size`, and runs
/// GatedDeltaNet on-device (set `RLX_WGPU_GDN_HOST=1` for the old
/// readback path).
///
/// Vulkan keeps Q1_0 / Q4_K / Q6_K `DequantMatMul` on-device (row-loop GEMV
/// for prefill); GatedDeltaNet still uses the Vulkan host segment.
///
/// CoreML Q1_0 defaults to `RLX_COREML_Q1_MODE=lut` (1-bit palettization);
/// `MODE=f32` OOMs disk on 27B-class models.
///
/// Force CPU with `RLX_PACKED_GGUF_WGPU_HOST=1`,
/// `RLX_PACKED_GGUF_VULKAN_HOST=1`, or `RLX_PACKED_GGUF_COREML_HOST=1`.
pub fn packed_gguf_execution_device(device: Device) -> Device {
    match device {
        Device::Ane if rlx_ir::env::flag("RLX_PACKED_GGUF_COREML_HOST") => Device::Cpu,
        Device::Ane => Device::Ane,
        Device::Gpu if rlx_ir::env::flag("RLX_PACKED_GGUF_WGPU_HOST") => Device::Cpu,
        Device::Gpu => Device::Gpu,
        Device::Vulkan if rlx_ir::env::flag("RLX_PACKED_GGUF_VULKAN_HOST") => Device::Cpu,
        Device::Vulkan => Device::Vulkan,
        Device::Cpu | Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm => device,
        _ => device,
    }
}

/// SAM encoder / upscale / prompt-mask subgraphs.
pub fn compile_options_sam_encoder(device: Device) -> CompileOptions {
    compile_options_for_profile(&CompileProfile::sam_encoder(), device)
}

/// SAM3 detector encoder/decoder layers.
pub fn compile_options_sam3(device: Device) -> CompileOptions {
    compile_options_for_profile(&CompileProfile::sam3(), device)
}

/// SAM2 memory attention (fusion disabled — matches legacy `compile_opts_no_fusion`).
pub fn compile_options_sam2_memory_attention(device: Device) -> CompileOptions {
    compile_options_for_profile(&CompileProfile::sam2_memory_attention(), device)
}

/// Compile a vision subgraph with explicit tier-1 profile options.
pub fn compile_graph_with_profile(
    device: Device,
    graph: rlx_ir::Graph,
    profile: &CompileProfile,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    use rlx_runtime::Session;
    let opts = compile_options_for_profile(profile, device);
    Ok(Session::new(device).compile_with(graph, &opts))
}

/// Compile a SAM/SAM2/SAM3 vision subgraph with tier-1 encoder profile options.
pub fn compile_graph_sam(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::sam_encoder())
}

/// Bidirectional encoder defaults (BERT, DINOv2, Wav2Vec2, vision towers).
pub fn compile_graph_encoder(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::encoder())
}

/// Qwen3 prefill / full-sequence graphs.
pub fn compile_graph_qwen3_prefill(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::qwen3_prefill())
}

/// Qwen3 single-token decode graphs.
pub fn compile_graph_qwen3_decode(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::qwen3_decode())
}

/// Qwen3.5 prefill-cache / predict graphs.
pub fn compile_graph_qwen35_prefill(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::qwen35_prefill())
}

/// Qwen3.5 decode-step graphs.
pub fn compile_graph_qwen35_decode(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::qwen35_decode())
}

/// Gemma / Gemma 2 prefill graphs.
pub fn compile_graph_gemma_prefill(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::gemma_prefill())
}

/// Gemma / Gemma 2 decode-step graphs.
pub fn compile_graph_gemma_decode(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::gemma_decode())
}

/// Llama 3.2 prefill graphs.
pub fn compile_graph_llama32_prefill(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::llama32_prefill())
}

/// Llama 3.2 decode graphs.
pub fn compile_graph_llama32_decode(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_graph_with_profile(device, graph, &CompileProfile::llama32_decode())
}

/// Unprofiled compile (parity probes / bisect tests).
pub fn compile_graph_legacy(
    device: Device,
    graph: rlx_ir::Graph,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    use rlx_runtime::{CompileOptions, Session};
    Ok(Session::new(device).compile_with(graph, &CompileOptions::new()))
}

/// Compile HIR with SAM/SAM3 tier-1 profile options.
pub fn compile_hir_sam(
    device: Device,
    hir: rlx_ir::hir::HirModule,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_hir_with_profile(device, hir, &CompileProfile::sam_encoder())
}

/// Compile HIR with SAM3 tier-1 profile options.
pub fn compile_hir_sam3(
    device: Device,
    hir: rlx_ir::hir::HirModule,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    compile_hir_with_profile(device, hir, &CompileProfile::sam3())
}

/// Compile HIR with an explicit tier-1 profile.
pub fn compile_hir_with_profile(
    device: Device,
    hir: rlx_ir::hir::HirModule,
    profile: &CompileProfile,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    use rlx_runtime::Session;
    let opts = compile_options_for_profile(profile, device);
    Ok(Session::new(device).compile_hir_with(hir, &opts)?)
}

/// Unified compile options from a [`ModelExecutionConfig`] (variant preset + binding).
pub fn compile_options_for(config: &ModelExecutionConfig) -> CompileOptions {
    compile_options_from_profile(
        &config.compile_profile(),
        Device::Cpu,
        config.component().kernel_dispatch,
    )
    .dim_binding(config.dim_binding())
}

/// Profile from config preset + device fusion target (runner dynamic specialize path).
pub fn compile_options_for_device(config: &ModelExecutionConfig, device: Device) -> CompileOptions {
    compile_options_from_profile(
        &config.compile_profile(),
        device,
        config.component().kernel_dispatch,
    )
    .dim_binding(config.dim_binding())
}

/// Compile a built flow through [`ModelCompilePipeline`] for one execution variant.
pub fn compile_built_with_config(
    pipeline: &mut ModelCompilePipeline,
    built: BuiltModel,
    config: &ModelExecutionConfig,
    options: &CompileOptions,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {
    let key = config.cache_key();
    let binding = config.dim_binding();
    let device = pipeline.device();
    let (hir, params) = built.into_parts()?;
    // CPU executables are cloneable — fill the pipeline cache then clone.
    // Discrete-GPU backends cannot `clone_box`, so a prior `get_or_compile`
    // followed by `Session::compile_hir_with` was compiling the same graph
    // twice (≈2× wall time on packed Qwen35 CUDA). Compile once for GPU.
    let mut compiled = if device == Device::Cpu {
        pipeline
            .get_or_compile(key, &binding, || hir.clone(), options)?
            .clone()
    } else {
        Session::new(device).compile_hir_with(hir, options)?
    };
    for (name, data) in params {
        compiled.set_param(&name, &data);
    }
    Ok(compiled)
}

fn fusion_target_from_profile(kind: FusionTargetKind) -> Option<FusionTarget> {
    match kind {
        FusionTargetKind::Auto => None,
        FusionTargetKind::Cpu => Some(FusionTarget::Cpu),
        FusionTargetKind::Metal => Some(FusionTarget::Metal),
        FusionTargetKind::Mlx => Some(FusionTarget::Mlx),
        FusionTargetKind::Cuda => Some(FusionTarget::Cuda),
        FusionTargetKind::Rocm => Some(FusionTarget::Rocm),
        FusionTargetKind::Wgpu => Some(FusionTarget::Wgpu),
        FusionTargetKind::Tpu => Some(FusionTarget::Tpu),
    }
}
