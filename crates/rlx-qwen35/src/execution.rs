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

//! Qwen3.5 execution variants — [`ModelExecutionConfig`] drives cache keys and [`DimBinding`].

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_flow::{BuiltModel, ModelExecutionConfig};
use rlx_ir::hir::HirModule;
use rlx_ir::{BindingManifest, CompilationMode};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, ModelCompilePipeline};

/// Component compile pipeline + optional on-disk AOT LIR ([`CompilationMode::Aot`]).
pub struct Qwen35CompileCache {
    pub pipeline: ModelCompilePipeline,
    aot: Option<AotCache>,
}

impl Qwen35CompileCache {
    pub fn new(device: rlx_runtime::Device, capacity: usize) -> Self {
        Self {
            pipeline: ModelCompilePipeline::with_capacity(device, capacity),
            aot: None,
        }
    }

    /// Enable disk-backed LIR for [`CompilationMode::Aot`] variants.
    pub fn with_aot(
        device: rlx_runtime::Device,
        capacity: usize,
        root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            pipeline: ModelCompilePipeline::with_capacity(device, capacity),
            aot: Some(AotCache::new(root)),
        }
    }

    pub fn device(&self) -> rlx_runtime::Device {
        self.pipeline.device()
    }

    pub fn contains(&self, config: &ModelExecutionConfig) -> bool {
        self.pipeline.contains(config.cache_key())
    }

    pub fn len(&self) -> usize {
        self.pipeline.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipeline.is_empty()
    }

    pub fn has_template(&self) -> bool {
        self.pipeline.has_template()
    }

    /// Binding layout for a variant (requires template built for that HIR family).
    pub fn binding_manifest_for(
        &self,
        config: &ModelExecutionConfig,
        options: &CompileOptions,
    ) -> BindingManifest {
        self.pipeline
            .binding_manifest_for_component(config.component(), options)
    }

    /// Compile a tier-0 [`BuiltModel`] through this pipeline (profile + variant key).
    pub fn compile_built(
        &mut self,
        built: BuiltModel,
        config: &ModelExecutionConfig,
        options: &CompileOptions,
    ) -> Result<CompiledGraph> {
        if config.component().compilation_mode == CompilationMode::Aot {
            return self.compile_built_aot(built, config, options);
        }
        rlx_core::flow_bridge::compile_built_with_config(&mut self.pipeline, built, config, options)
    }

    /// [`CompilationMode::Aot`] — disk LIR via [`AotCache`] + pipeline specialize.
    pub fn compile_built_aot(
        &mut self,
        built: BuiltModel,
        config: &ModelExecutionConfig,
        options: &CompileOptions,
    ) -> Result<CompiledGraph> {
        let aot = self
            .aot
            .as_ref()
            .context("CompilationMode::Aot requires Qwen35CompileCache::with_aot(root)")?;
        let key = config.cache_key();
        let binding = config.dim_binding();
        let disk_base = config.component().aot_disk_base();
        let (hir, params) = built.into_parts()?;
        let mut compiled = self
            .pipeline
            .get_or_specialize_aot(aot, &disk_base, key, &binding, || hir, options)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .clone();
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        Ok(compiled)
    }
}

/// Prefill-cache graph (symbolic `sym::SEQ`).
pub fn prefill_config(batch: usize, seq: usize) -> ModelExecutionConfig {
    ModelExecutionConfig::qwen35_prefill(batch, seq)
}

/// VLM hidden-state prefill-cache (same dim binding as text prefill).
pub fn hidden_prefill_config(batch: usize, seq: usize) -> ModelExecutionConfig {
    ModelExecutionConfig::qwen35_prefill(batch, seq)
}

/// Decode step (symbolic `sym::PAST_SEQ`, new tokens = 1).
pub fn decode_config(batch: usize, past_seq: usize) -> ModelExecutionConfig {
    ModelExecutionConfig::qwen35_decode(batch, past_seq)
}

/// Cache bucket key — use [`ModelExecutionConfig::cache_key`] (full component fingerprint).
#[inline]
pub fn cache_key_for_config(config: &ModelExecutionConfig) -> u64 {
    config.cache_key()
}

/// Compile-once / specialize-at-runtime; upload params on first hit for this variant key.
pub fn get_or_specialize_hir<'a, F>(
    cache: &'a mut Qwen35CompileCache,
    config: &ModelExecutionConfig,
    build_hir: F,
    on_first_hit: impl FnOnce(&mut CompiledGraph) -> Result<()>,
) -> Result<&'a mut CompiledGraph>
where
    F: FnOnce() -> HirModule,
{
    get_or_specialize_hir_with_options(
        cache,
        config,
        build_hir,
        &rlx_core::flow_bridge::compile_options_for_device(config, cache.device()),
        on_first_hit,
    )
}

/// Like [`get_or_specialize_hir`] with explicit compile options (profile, dispatch, …).
pub fn get_or_specialize_hir_with_options<'a, F>(
    cache: &'a mut Qwen35CompileCache,
    config: &ModelExecutionConfig,
    build_hir: F,
    options: &CompileOptions,
    on_first_hit: impl FnOnce(&mut CompiledGraph) -> Result<()>,
) -> Result<&'a mut CompiledGraph>
where
    F: FnOnce() -> HirModule,
{
    let key = config.cache_key();
    let binding = config.dim_binding();
    let first = !cache.pipeline.contains(key);

    let compiled = match config.component().compilation_mode {
        CompilationMode::Aot => {
            let aot = cache.aot.as_ref().context(
                "CompilationMode::Aot requires Qwen35CompileCache::with_aot(root) \
                 (disk LIR under that directory)",
            )?;
            let disk_base = config.component().aot_disk_base();
            cache
                .pipeline
                .get_or_specialize_aot(aot, &disk_base, key, &binding, build_hir, options)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        }
        CompilationMode::Eager | CompilationMode::Lazy => cache
            .pipeline
            .get_or_compile(key, &binding, build_hir, options)
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    };

    if first {
        on_first_hit(compiled)?;
    }
    Ok(compiled)
}

/// Template → specialize → compile; returns compiled graph + binding manifest.
pub fn get_or_specialize_component<'a, F>(
    cache: &'a mut Qwen35CompileCache,
    config: &ModelExecutionConfig,
    build_hir: F,
    options: &CompileOptions,
    on_first_hit: impl FnOnce(&mut CompiledGraph) -> Result<()>,
) -> Result<(&'a mut CompiledGraph, BindingManifest)>
where
    F: FnOnce() -> HirModule,
{
    let manifest = cache.binding_manifest_for(config, options);
    let compiled =
        get_or_specialize_hir_with_options(cache, config, build_hir, options, on_first_hit)?;
    Ok((compiled, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::hir::HirMut;
    use rlx_ir::{CompilationMode, DType, HirModule, Shape};

    fn tiny_hir() -> HirModule {
        let mut hir = HirModule::new("qwen35_aot");
        let mut gb = HirMut::new(&mut hir);
        let x = gb.input("x", Shape::new(&[1, 4], DType::F32));
        let w = gb.param("w", Shape::new(&[4, 2], DType::F32));
        let y = hir.linear(x, w, None, None, Shape::new(&[1, 2], DType::F32));
        hir.set_outputs(vec![y]);
        hir
    }

    #[test]
    fn compile_built_via_pipeline() {
        use rlx_flow::BuiltModel;
        use std::collections::HashMap;

        let mut cache = Qwen35CompileCache::new(rlx_runtime::Device::Cpu, 4);
        let config = prefill_config(1, 4);
        let built = BuiltModel::from_hir(tiny_hir(), HashMap::new())
            .unwrap()
            .with_execution_config(&config);
        let opts =
            rlx_core::flow_bridge::compile_options_for_device(&config, rlx_runtime::Device::Cpu);
        let mut compiled = cache.compile_built(built, &config, &opts).unwrap();
        compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let outs = compiled.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]);
        assert_eq!(outs[0].len(), 2);
    }

    #[test]
    fn unified_cache_key_differs_batch() {
        let a = prefill_config(1, 8).cache_key();
        let b = prefill_config(2, 8).cache_key();
        assert_ne!(a, b);
        assert_eq!(cache_key_for_config(&prefill_config(1, 8)), a);
    }

    #[test]
    fn pipeline_has_template_after_specialize() {
        let mut cache = Qwen35CompileCache::new(rlx_runtime::Device::Cpu, 4);
        let config = prefill_config(1, 4);
        let opts = rlx_core::flow_bridge::compile_options_for_device(&config, cache.device());
        get_or_specialize_hir(&mut cache, &config, tiny_hir, |_| Ok(())).unwrap();
        assert!(cache.has_template());
        let _manifest = cache.binding_manifest_for(&config, &opts);
    }

    #[test]
    fn aot_mode_writes_lir_disk() {
        let dir = std::env::temp_dir().join(format!("qwen35_aot_{}", std::process::id()));
        let mut cache = Qwen35CompileCache::with_aot(rlx_runtime::Device::Cpu, 4, &dir);
        let config = prefill_config(1, 4).with_compilation_mode(CompilationMode::Aot);
        get_or_specialize_hir(&mut cache, &config, tiny_hir, |_| Ok(())).unwrap();
        let disk = config.component().aot_disk_base();
        assert!(
            dir.join(format!("{disk}__0.lir.json")).exists() || {
                std::fs::read_dir(&dir)
                    .ok()
                    .map(|rd| {
                        rd.flatten()
                            .any(|e| e.file_name().to_string_lossy().contains(&disk))
                    })
                    .unwrap_or(false)
            }
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
