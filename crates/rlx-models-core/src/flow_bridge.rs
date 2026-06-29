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
}

/// Dynamic HIR template/specialize — default passes only (matches legacy [`DynamicDimCompileCache`]).
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
/// **CPU**, **Metal**, **MLX**, **wgpu** (`Device::Gpu`), and **Vulkan** run natively.
/// MLX uses host-side GGUF dequant per `DequantMatMul` with lazy eval (see
/// [`packed_gguf_compile_guard`]); wgpu and Vulkan run native GGUF dequant + matmul
/// on-device (`rlx_wgpu::gguf_gpu` / the `dequant_matmul` SPIR-V GEMV) — for Vulkan
/// the fused kernel covers decode GEMV (`m == 1`); packed prefill (`m > 1`) host-falls
/// back per-op like wgpu. **CUDA / ROCm** still fall back to CPU until upstream GPU
/// parity lands.
pub fn packed_gguf_execution_device(device: Device) -> Device {
    match device {
        Device::Cpu | Device::Metal | Device::Mlx | Device::Gpu | Device::Vulkan => device,
        Device::Cuda | Device::Rocm => Device::Cpu,
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
    // Pipeline caches the variant; owned graphs for GPU backends cannot use
    // `CompiledGraph::clone` (only CPU implements `clone_box` today).
    if !pipeline.contains(key) {
        pipeline.get_or_compile(key, &binding, || hir.clone(), options)?;
    }
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
