// RLX models — staged-inference wire protocol + topology planner.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The **rlx protocol**: the staged-inference wire format and topology planner.
//!
//! This crate is ported and renamed from mesh-llm's `skippy-protocol` /
//! `skippy-topology` crates (everywhere the source said `skippy`/`Skippy` it is
//! now `rlx`/`Rlx`). It is a *pure codec + planner*: there is deliberately no
//! transport, QUIC, tokio, or protobuf here — the network layer lives elsewhere.
//!
//! * [`wire`] — the activation-tensor wire codec ([`wire::ActivationFrame`],
//!   [`wire::ActivationDType`]) plus JSON stage-control messages
//!   ([`wire::StageControl`]) and the protocol constants
//!   ([`wire::STAGE_ALPN`], stream ids, [`wire::MAX_STAGE_FRAME_BYTES`]).
//! * [`topology`] — the topology planner: [`topology::plan`] turns a
//!   [`topology::TopologyPlanRequest`] (layer + node specs) into a
//!   [`topology::TopologyPlan`] of contiguous layer-range stages.
//!
//! This is a distinct crate from `rlx-distributed`; the two are not related.

pub mod topology;
pub mod wire;

pub use topology::{
    LayerSpec, MigrationPolicy, NodeSpec, PlanError, PlannerPolicy, StagePlan, StageRole,
    StateAffinity, TopologyPlan, TopologyPlanRequest, plan,
};
pub use wire::{
    ActivationDType, ActivationFrame, MAX_STAGE_FRAME_BYTES, STAGE_ALPN, StageControl, WireError,
};
