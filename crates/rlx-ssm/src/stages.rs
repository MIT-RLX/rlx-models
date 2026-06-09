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

use anyhow::{Context, Result};
use rlx_flow::FlowValue;
use rlx_flow::escape::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Op, Shape};

fn named(emit: &mut Emit<'_>, key: &str) -> Result<rlx_ir::HirNodeId> {
    emit.state
        .named
        .get(key)
        .copied()
        .with_context(|| format!("missing FlowState.named `{key}`"))
}

fn softplus_hir(emit: &mut Emit<'_>, x: rlx_ir::HirNodeId) -> Result<rlx_ir::HirNodeId> {
    let shape = emit.hir().node(x).shape.clone();
    let ones = emit.synth_param(
        "ssm.softplus.ones",
        vec![1.0; shape.num_elements().unwrap_or(1)],
        shape.clone(),
    );
    let mut gb = HirMut::new(emit.hir());
    let exp_x = gb.exp(x);
    let one_plus = gb.add(ones, exp_x);
    Ok(gb.activation(rlx_ir::op::Activation::Log, one_plus, shape))
}

/// Weight key prefix for Mamba1 scan (`blk.N.A_log`, `blk.N.D`).
#[derive(Debug, Clone)]
pub struct MambaScanWeightKeys {
    pub prefix: String,
}

impl MambaScanWeightKeys {
    pub fn hf(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    fn a_log(&self) -> String {
        format!("{}.A_log", self.prefix)
    }

    fn d_skip(&self) -> String {
        format!("{}.D", self.prefix)
    }
}

/// Mamba1 selective-scan block (prefill / multi-token).
#[derive(Debug, Clone)]
pub struct MambaScanStage {
    keys: MambaScanWeightKeys,
    state_size: usize,
    x_shape: Shape,
    use_d_skip: bool,
}

impl MambaScanStage {
    pub fn new(keys: MambaScanWeightKeys, state_size: usize, x_shape: Shape) -> Self {
        Self {
            keys,
            state_size,
            x_shape,
            use_d_skip: true,
        }
    }

    pub fn without_d_skip(mut self) -> Self {
        self.use_d_skip = false;
        self
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
    {
        let keys = self.keys.clone();
        let n = self.state_size;
        let out_shape = self.x_shape.clone();
        let use_d = self.use_d_skip;
        move |emit, input| {
            let x = input.context("MambaScanStage requires input")?;
            let out = out_shape.clone();
            let dt_raw = named(emit, "ssm.dt_raw")?;
            let b_id = named(emit, "ssm.b")?;
            let c_id = named(emit, "ssm.c")?;
            let a_log = emit.load_param(&keys.a_log(), false)?;
            let d_skip = if use_d {
                Some(emit.load_param(&keys.d_skip(), false)?)
            } else {
                None
            };
            let delta = softplus_hir(emit, dt_raw)?;
            // Mamba1: A = -exp(A_log) (negative diagonal), not exp(-A_log).
            let mut gb = HirMut::new(emit.hir());
            let exp_a_log = gb.exp(a_log);
            let a = gb.neg(exp_a_log);
            let scan = gb.add_node(
                Op::SelectiveScan { state_size: n },
                vec![x.hir_id(), delta, a, b_id, c_id],
                out.clone(),
            );
            let y = if let Some(d) = d_skip {
                let skip = gb.mul(x.hir_id(), d);
                gb.add(scan, skip)
            } else {
                scan
            };
            Ok(Some(emit.wrap(y, out)))
        }
    }
}

/// Prefill Lightning Attention (requires `lightning.{q,k,v,...}` in FlowState).
#[derive(Debug, Clone)]
pub struct LightningAttentionStage {
    prefix: String,
}

impl LightningAttentionStage {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + '_
    {
        let _prefix = self.prefix.clone();
        move |emit, input| {
            let _ = named(emit, "lightning.q")?;
            let _ = named(emit, "lightning.k")?;
            let _ = named(emit, "lightning.v")?;
            let _ = input;
            anyhow::bail!(
                "LightningAttentionStage prefill is not wired yet — populate lightning.* and use LightningAttentionStepStage for decode"
            )
        }
    }
}

/// Lightning-attention single decode step → packed `[y | state_out]`.
#[derive(Debug, Clone)]
pub struct LightningAttentionStepStage {
    #[allow(dead_code)]
    prefix: String,
    batch: usize,
    heads: usize,
    state_size: usize,
}

impl LightningAttentionStepStage {
    pub fn new(prefix: impl Into<String>, batch: usize, heads: usize, state_size: usize) -> Self {
        Self {
            prefix: prefix.into(),
            batch,
            heads,
            state_size,
        }
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + '_
    {
        let b = self.batch;
        let h = self.heads;
        let n = self.state_size;
        move |emit, _input| {
            let out_shape = Shape::new(&[b, 1, h, n + n * n], DType::F32);
            let q = named(emit, "lightning.q")?;
            let k = named(emit, "lightning.k")?;
            let v = named(emit, "lightning.v")?;
            let gate = named(emit, "lightning.gate")?;
            let beta = named(emit, "lightning.beta")?;
            let state = named(emit, "lightning.state_in")?;
            let mut gb = HirMut::new(emit.hir());
            let packed = gb.add_node(
                Op::Custom {
                    name: "lightning_attention_step".into(),
                    num_inputs: 6,
                    attrs: Vec::new(),
                },
                vec![q, k, v, gate, beta, state],
                out_shape.clone(),
            );
            Ok(Some(emit.wrap(packed, out_shape)))
        }
    }
}

/// LFM SSM decode step → packed `[y | state_out]` on axis 1.
#[derive(Debug, Clone)]
pub struct LfmSsmStepStage {
    #[allow(dead_code)]
    prefix: String,
    batch: usize,
    channels: usize,
    state_size: usize,
}

impl LfmSsmStepStage {
    pub fn new(
        prefix: impl Into<String>,
        batch: usize,
        channels: usize,
        state_size: usize,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            batch,
            channels,
            state_size,
        }
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + '_
    {
        let b = self.batch;
        let c = self.channels;
        let n = self.state_size;
        move |emit, _input| {
            let out_shape = Shape::new(&[b, 1, c + c * n], DType::F32);
            let x = named(emit, "lfm.x")?;
            let a = named(emit, "lfm.a")?;
            let b_in = named(emit, "lfm.b")?;
            let c_proj = named(emit, "lfm.c_proj")?;
            let gate = named(emit, "lfm.gate")?;
            let state = named(emit, "lfm.state_in")?;
            let mut gb = HirMut::new(emit.hir());
            let packed = gb.add_node(
                Op::Custom {
                    name: "lfm_ssm_step".into(),
                    num_inputs: 6,
                    attrs: Vec::new(),
                },
                vec![x, a, b_in, c_proj, gate, state],
                out_shape.clone(),
            );
            Ok(Some(emit.wrap(packed, out_shape)))
        }
    }
}

/// Mamba1 decode step → packed `[y | state_out]` (`[batch, hidden + hidden * state]`).
#[derive(Debug, Clone)]
pub struct Mamba1StepStage {
    batch: usize,
    hidden: usize,
    state_size: usize,
}

impl Mamba1StepStage {
    pub fn new(batch: usize, hidden: usize, state_size: usize) -> Self {
        Self {
            batch,
            hidden,
            state_size,
        }
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + 'static
    {
        let b = self.batch;
        let h = self.hidden;
        let n = self.state_size;
        move |emit, _input| {
            let out_shape = Shape::new(&[b, h + h * n], DType::F32);
            let x = named(emit, "mamba1.x")?;
            let dt = named(emit, "mamba1.dt_raw")?;
            let a_log = named(emit, "mamba1.a_log")?;
            let b_in = named(emit, "mamba1.b")?;
            let c = named(emit, "mamba1.c")?;
            let d = named(emit, "mamba1.d_skip")?;
            let state = named(emit, "mamba1.state_in")?;
            let mut gb = HirMut::new(emit.hir());
            let packed = gb.add_node(
                Op::Custom {
                    name: "mamba1_step".into(),
                    num_inputs: 7,
                    attrs: Vec::new(),
                },
                vec![x, dt, a_log, b_in, c, d, state],
                out_shape.clone(),
            );
            Ok(Some(emit.wrap(packed, out_shape)))
        }
    }
}

/// Mamba2 decode step → packed `[y | state_out]` per batch row.
#[derive(Debug, Clone)]
pub struct Mamba2StepStage {
    #[allow(dead_code)]
    prefix: String,
    batch: usize,
    heads: usize,
    state_size: usize,
}

impl Mamba2StepStage {
    pub fn new(prefix: impl Into<String>, batch: usize, heads: usize, state_size: usize) -> Self {
        Self {
            prefix: prefix.into(),
            batch,
            heads,
            state_size,
        }
    }

    pub fn plugin(
        &self,
    ) -> impl Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync + '_
    {
        let b = self.batch;
        let h = self.heads;
        let n = self.state_size;
        move |emit, _input| {
            let out_shape = Shape::new(&[b, h + h * n], DType::F32);
            let x = named(emit, "mamba2.x")?;
            let dt = named(emit, "mamba2.dt_raw")?;
            let a_log = named(emit, "mamba2.a_log")?;
            let b_in = named(emit, "mamba2.b")?;
            let c = named(emit, "mamba2.c_proj")?;
            let d = named(emit, "mamba2.d_skip")?;
            let state = named(emit, "mamba2.state_in")?;
            let mut gb = HirMut::new(emit.hir());
            let packed = gb.add_node(
                Op::Custom {
                    name: "mamba2_step".into(),
                    num_inputs: 7,
                    attrs: Vec::new(),
                },
                vec![x, dt, a_log, b_in, c, d, state],
                out_shape.clone(),
            );
            Ok(Some(emit.wrap(packed, out_shape)))
        }
    }
}
