//! Staged-inference topology planner.
//!
//! Ported and renamed from mesh-llm's `skippy-topology`. Given a set of model
//! layers (with attention/recurrent/param-byte metadata) and a set of nodes
//! (with VRAM budgets and network RTT), [`plan`] produces a [`TopologyPlan`]:
//! a set of contiguous layer-range [`StagePlan`]s, one per node, ordered into a
//! pipeline, each tagged with its state affinity, migration policy, and pipeline
//! roles ([`StageRole`]).
//!
//! Compared with the skippy source this planner keeps the VRAM-weighted
//! contiguous split, the [`StateAffinity`] / [`MigrationPolicy`] classification,
//! and the RTT/cache-locality cost model, but drops the mesh-entangled parts:
//! the multi-hundred-entry per-architecture family table, the separate
//! `edge_order` / `artifact_diagnostics` submodules, the `FamilyCapabilityRecord`
//! and its per-model constructors, and all protobuf/network types. RTT and cache
//! locality live directly on [`NodeSpec`].

use serde::{Deserialize, Serialize};

/// A single transformer layer's placement-relevant metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LayerSpec {
    /// Global layer index. Layers in a request must be contiguous and ascending.
    pub index: u32,
    /// Layer maintains attention KV state.
    #[serde(default)]
    pub has_attention: bool,
    /// Layer maintains recurrent (SSM / RWKV / gated-delta) state.
    #[serde(default)]
    pub has_recurrent: bool,
    /// Parameter bytes resident for this layer (weights).
    #[serde(default)]
    pub param_bytes: u64,
}

impl LayerSpec {
    /// Convenience constructor for a plain stateless-or-attention dense layer.
    pub fn new(index: u32, param_bytes: u64, has_attention: bool, has_recurrent: bool) -> Self {
        Self {
            index,
            has_attention,
            has_recurrent,
            param_bytes,
        }
    }
}

/// A candidate node this plan can place stages on.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct NodeSpec {
    /// Stable node identifier.
    pub id: String,
    /// Available VRAM in bytes — drives the weighted split.
    #[serde(default)]
    pub vram_bytes: u64,
    /// Round-trip time to this node in milliseconds (network pipeline cost).
    #[serde(default)]
    pub rtt_ms: Option<u32>,
    /// Bytes of this stage's weights already cached on the node (cache locality).
    #[serde(default)]
    pub cached_slice_bytes: u64,
}

impl NodeSpec {
    /// Convenience constructor for a VRAM-only node.
    pub fn new(id: impl Into<String>, vram_bytes: u64) -> Self {
        Self {
            id: id.into(),
            vram_bytes,
            rtt_ms: None,
            cached_slice_bytes: 0,
        }
    }
}

/// Planner tuning knobs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct PlannerPolicy {
    /// When set, recurrent stages may be moved with their state transferred,
    /// otherwise the owning node is sticky.
    #[serde(default)]
    pub allow_recurrent_state_transfer: bool,
}

/// A topology plan request.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlanRequest {
    pub topology_id: String,
    pub model_id: String,
    pub layers: Vec<LayerSpec>,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub policy: PlannerPolicy,
}

/// Role a stage plays in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRole {
    /// Owns the request lifecycle (first stage).
    Driver,
    /// Runs the token embedding (first stage).
    Embedding,
    /// A middle stage that only transforms activations.
    Intermediate,
    /// Runs the final norm + LM head (last stage).
    Readout,
}

/// The kind of live state a stage's layers carry across tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAffinity {
    /// No per-token state.
    Stateless,
    /// Attention KV cache only.
    AttentionKv,
    /// Recurrent (SSM / RWKV / gated-delta) state only.
    Recurrent,
    /// Both attention and recurrent state.
    Mixed,
}

/// How freely a stage may be migrated to another node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationPolicy {
    /// Stateless — move at will.
    FreelyMovable,
    /// Attention KV — movable but re-warming the cache has a cost.
    CostedKv,
    /// Recurrent — pinned to its owning node.
    StickyRecurrentOwner,
    /// Recurrent — movable because the policy allows state transfer.
    RecurrentStateTransferAllowed,
}

/// Machine-readable reason a placement/boundary decision was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReasonCode {
    ActivationOnlyBoundary,
    AttentionKvCosted,
    RecurrentOwnerSticky,
    RecurrentStateTransferAllowed,
    CacheLocalityPreferred,
    NetworkPipelineCost,
}

/// One contiguous layer-range stage assigned to a node.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    #[serde(default)]
    pub roles: Vec<StageRole>,
    /// First layer index (inclusive).
    pub layer_start: u32,
    /// One past the last layer index (exclusive).
    pub layer_end: u32,
    pub layer_count: u32,
    pub param_bytes: u64,
    pub state_affinity: StateAffinity,
    pub migration_policy: MigrationPolicy,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
    #[serde(default)]
    pub cached_slice_bytes: u64,
    #[serde(default)]
    pub rtt_ms: Option<u32>,
}

/// A boundary between two adjacent stages, where an activation frame crosses.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct BoundaryPlan {
    pub producer_stage_index: u32,
    pub consumer_stage_index: u32,
    /// Layer index the cut sits after (== producer.layer_end).
    pub layer_boundary: u32,
    #[serde(default)]
    pub reason_codes: Vec<PlanReasonCode>,
    #[serde(default)]
    pub messages: Vec<String>,
}

/// Severity of a plan diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// A human-readable diagnostic emitted alongside a plan.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PlanDiagnostic {
    pub severity: DiagnosticSeverity,
    pub code: PlanReasonCode,
    pub message: String,
}

/// The result of planning: contiguous stages, their boundaries, diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TopologyPlan {
    pub topology_id: String,
    pub model_id: String,
    pub stages: Vec<StagePlan>,
    #[serde(default)]
    pub boundaries: Vec<BoundaryPlan>,
    #[serde(default)]
    pub diagnostics: Vec<PlanDiagnostic>,
}

/// Errors from [`plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    EmptyLayers,
    EmptyNodes,
    NonContiguousLayers { expected: u32, found: u32 },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLayers => write!(f, "topology plan requires at least one layer"),
            Self::EmptyNodes => write!(f, "topology plan requires at least one node"),
            Self::NonContiguousLayers { expected, found } => write!(
                f,
                "layers must be sorted and contiguous: expected layer {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Plan a staged pipeline: VRAM-fit + RTT/affinity-aware contiguous split.
///
/// Nodes are first ordered by a placement score (VRAM + cache locality, minus a
/// per-hop RTT penalty), then the layers are cut into contiguous ranges whose
/// spans are proportional to each node's VRAM (with an all-equal fallback when
/// no VRAM is reported). Each range becomes a [`StagePlan`] carrying its state
/// affinity and migration policy.
pub fn plan(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());

    // Order nodes best-first by placement score, keeping only as many as we
    // have stages for. Ties break on original request order for determinism.
    let mut ordered: Vec<NodeSpec> = request.nodes.clone();
    ordered.sort_by(|a, b| {
        node_placement_score(b)
            .cmp(&node_placement_score(a))
            .then_with(|| a.id.cmp(&b.id))
    });
    let nodes = &ordered[..stage_count];

    // Weighted-contiguous split of layers across the ordered nodes.
    let ranges = weighted_ranges(request.layers.len(), nodes);

    let mut stages = Vec::with_capacity(ranges.len());
    for (stage_index, &(start, end)) in ranges.iter().enumerate() {
        let layers = &request.layers[start..end];
        let layer_start = layers.first().expect("non-empty range").index;
        let layer_end = layers.last().expect("non-empty range").index + 1;
        let state_affinity = classify_layers(layers);
        let migration_policy = migration_policy(state_affinity, request.policy);
        let param_bytes = layers.iter().map(|l| l.param_bytes).sum();
        let node = &nodes[stage_index];

        let mut reason_codes = stage_reason_codes(migration_policy);
        if node.cached_slice_bytes > 0 {
            reason_codes.push(PlanReasonCode::CacheLocalityPreferred);
        }
        if node.rtt_ms.is_some_and(|rtt| rtt > 0) {
            reason_codes.push(PlanReasonCode::NetworkPipelineCost);
        }

        stages.push(StagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id: node.id.clone(),
            roles: stage_roles(stage_index, ranges.len()),
            layer_start,
            layer_end,
            layer_count: (end - start) as u32,
            param_bytes,
            state_affinity,
            migration_policy,
            reason_codes,
            cached_slice_bytes: node.cached_slice_bytes,
            rtt_ms: node.rtt_ms,
        });
    }

    let boundaries = boundaries_for(&stages);
    let diagnostics = diagnostics_for(&stages, request.policy);

    Ok(TopologyPlan {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        stages,
        boundaries,
        diagnostics,
    })
}

/// Placement score used to order nodes best-first. VRAM dominates, cache
/// locality is worth 2× its bytes, and each RTT millisecond costs a large fixed
/// penalty (mirroring the skippy `node_package_score` weighting).
fn node_placement_score(node: &NodeSpec) -> i128 {
    let mut score = i128::from(node.vram_bytes);
    score += i128::from(node.cached_slice_bytes).saturating_mul(2);
    if let Some(rtt) = node.rtt_ms {
        score -= i128::from(rtt).saturating_mul(16 * 1024 * 1024);
    }
    score
}

/// Cut `layer_len` layers into `nodes.len()` contiguous ranges with spans
/// proportional to each node's VRAM. Falls back to an even split when total
/// VRAM is zero. Every range gets at least one layer.
fn weighted_ranges(layer_len: usize, nodes: &[NodeSpec]) -> Vec<(usize, usize)> {
    let stage_count = nodes.len();
    let total_vram: u64 = nodes.iter().map(|n| n.vram_bytes).sum();

    if total_vram == 0 {
        return even_ranges(layer_len, stage_count);
    }

    let mut ranges = Vec::with_capacity(stage_count);
    let mut layer_start = 0usize;
    for (stage_index, node) in nodes.iter().enumerate() {
        let remaining_stages = stage_count - stage_index;
        let remaining_layers = layer_len - layer_start;
        let mut span = if remaining_stages == 1 {
            remaining_layers
        } else {
            (((layer_len as u128) * (node.vram_bytes as u128)) / (total_vram as u128))
                .try_into()
                .unwrap_or(usize::MAX)
        };
        // Keep at least 1 layer here, and leave at least 1 for each later stage.
        span = span.max(1).min(remaining_layers - (remaining_stages - 1));
        let layer_end = layer_start + span;
        ranges.push((layer_start, layer_end));
        layer_start = layer_end;
    }
    ranges
}

/// Even split (used when no VRAM weights are available): the first `remainder`
/// stages get one extra layer.
fn even_ranges(layer_len: usize, stage_count: usize) -> Vec<(usize, usize)> {
    let base = layer_len / stage_count;
    let remainder = layer_len % stage_count;
    let mut next = 0usize;
    let mut ranges = Vec::with_capacity(stage_count);
    for stage_index in 0..stage_count {
        let count = base + usize::from(stage_index < remainder);
        ranges.push((next, next + count));
        next += count;
    }
    ranges
}

/// Classify the live-state affinity of a contiguous layer range.
pub fn classify_layers(layers: &[LayerSpec]) -> StateAffinity {
    let has_attention = layers.iter().any(|l| l.has_attention);
    let has_recurrent = layers.iter().any(|l| l.has_recurrent);
    match (has_attention, has_recurrent) {
        (false, false) => StateAffinity::Stateless,
        (true, false) => StateAffinity::AttentionKv,
        (false, true) => StateAffinity::Recurrent,
        (true, true) => StateAffinity::Mixed,
    }
}

fn migration_policy(affinity: StateAffinity, policy: PlannerPolicy) -> MigrationPolicy {
    match affinity {
        StateAffinity::Stateless => MigrationPolicy::FreelyMovable,
        StateAffinity::AttentionKv => MigrationPolicy::CostedKv,
        StateAffinity::Recurrent | StateAffinity::Mixed => {
            if policy.allow_recurrent_state_transfer {
                MigrationPolicy::RecurrentStateTransferAllowed
            } else {
                MigrationPolicy::StickyRecurrentOwner
            }
        }
    }
}

fn stage_reason_codes(migration_policy: MigrationPolicy) -> Vec<PlanReasonCode> {
    match migration_policy {
        MigrationPolicy::FreelyMovable => Vec::new(),
        MigrationPolicy::CostedKv => vec![PlanReasonCode::AttentionKvCosted],
        MigrationPolicy::StickyRecurrentOwner => vec![PlanReasonCode::RecurrentOwnerSticky],
        MigrationPolicy::RecurrentStateTransferAllowed => {
            vec![PlanReasonCode::RecurrentStateTransferAllowed]
        }
    }
}

fn stage_roles(stage_index: usize, stage_count: usize) -> Vec<StageRole> {
    let mut roles = Vec::new();
    if stage_index == 0 {
        roles.push(StageRole::Driver);
        roles.push(StageRole::Embedding);
    }
    if stage_index + 1 == stage_count {
        roles.push(StageRole::Readout);
    } else if stage_index > 0 {
        roles.push(StageRole::Intermediate);
    }
    roles
}

fn boundaries_for(stages: &[StagePlan]) -> Vec<BoundaryPlan> {
    stages
        .windows(2)
        .map(|window| {
            let producer = &window[0];
            let consumer = &window[1];
            let mut reason_codes = vec![PlanReasonCode::ActivationOnlyBoundary];
            let mut messages = vec![format!(
                "activation boundary after layer {}; send activation frame from {} to {}",
                producer.layer_end, producer.stage_id, consumer.stage_id
            )];
            if matches!(producer.migration_policy, MigrationPolicy::StickyRecurrentOwner)
                || matches!(consumer.migration_policy, MigrationPolicy::StickyRecurrentOwner)
            {
                reason_codes.push(PlanReasonCode::RecurrentOwnerSticky);
                messages.push(
                    "recurrent state remains with the owning stage; only activation crosses this boundary"
                        .to_string(),
                );
            }
            BoundaryPlan {
                producer_stage_index: producer.stage_index,
                consumer_stage_index: consumer.stage_index,
                layer_boundary: producer.layer_end,
                reason_codes,
                messages,
            }
        })
        .collect()
}

fn diagnostics_for(stages: &[StagePlan], policy: PlannerPolicy) -> Vec<PlanDiagnostic> {
    let mut diagnostics = Vec::new();
    for stage in stages {
        if matches!(
            stage.migration_policy,
            MigrationPolicy::StickyRecurrentOwner
        ) {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::RecurrentOwnerSticky,
                message: format!(
                    "{} owns recurrent state for layers {}..{}; route future tokens back to {} and only transfer activations across stage boundaries",
                    stage.stage_id, stage.layer_start, stage.layer_end, stage.node_id
                ),
            });
        }
    }
    if policy.allow_recurrent_state_transfer {
        diagnostics.push(PlanDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: PlanReasonCode::RecurrentStateTransferAllowed,
            message: "recurrent state transfer is enabled; reserve this for explicit recompute-or-transfer flows, not normal routing".to_string(),
        });
    }
    diagnostics
}

fn validate_request(request: &TopologyPlanRequest) -> Result<(), PlanError> {
    if request.layers.is_empty() {
        return Err(PlanError::EmptyLayers);
    }
    if request.nodes.is_empty() {
        return Err(PlanError::EmptyNodes);
    }
    for (expected, layer) in (request.layers[0].index..).zip(request.layers.iter()) {
        if layer.index != expected {
            return Err(PlanError::NonContiguousLayers {
                expected,
                found: layer.index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_layers(count: u32) -> Vec<LayerSpec> {
        (0..count)
            .map(|i| LayerSpec::new(i, 100, true, false))
            .collect()
    }

    #[test]
    fn plan_24_layers_over_3_nodes_uneven_vram() {
        // VRAM ratio 1 : 2 : 3 over 24 layers → spans ~4 : 8 : 12.
        let request = TopologyPlanRequest {
            topology_id: "topo".into(),
            model_id: "qwen3".into(),
            layers: dense_layers(24),
            nodes: vec![
                NodeSpec::new("a", 1 << 30),
                NodeSpec::new("b", 2 << 30),
                NodeSpec::new("c", 3 << 30),
            ],
            policy: PlannerPolicy::default(),
        };
        let plan = plan(&request).unwrap();
        assert_eq!(plan.stages.len(), 3);

        // Contiguous, gap-free, full-coverage layer ranges.
        assert_eq!(plan.stages[0].layer_start, 0);
        assert_eq!(plan.stages.last().unwrap().layer_end, 24);
        let mut prev_end = 0;
        let mut total = 0u32;
        for (i, stage) in plan.stages.iter().enumerate() {
            assert_eq!(stage.layer_start, prev_end, "gap before stage {i}");
            assert!(stage.layer_end > stage.layer_start, "empty stage {i}");
            assert_eq!(stage.layer_count, stage.layer_end - stage.layer_start);
            prev_end = stage.layer_end;
            total += stage.layer_count;
        }
        assert_eq!(total, 24);

        // Node ordered biggest-VRAM-first (c, b, a); the largest stage is first.
        assert_eq!(plan.stages[0].node_id, "c");
        assert!(plan.stages[0].layer_count >= plan.stages[1].layer_count);
        assert!(plan.stages[1].layer_count >= plan.stages[2].layer_count);

        // Roles: first is driver+embedding, last is readout, middle intermediate.
        assert!(plan.stages[0].roles.contains(&StageRole::Driver));
        assert!(plan.stages[0].roles.contains(&StageRole::Embedding));
        assert!(plan.stages[1].roles.contains(&StageRole::Intermediate));
        assert!(plan.stages[2].roles.contains(&StageRole::Readout));

        // Attention-only layers → AttentionKv / CostedKv.
        for stage in &plan.stages {
            assert_eq!(stage.state_affinity, StateAffinity::AttentionKv);
            assert_eq!(stage.migration_policy, MigrationPolicy::CostedKv);
        }

        // One boundary between each adjacent pair.
        assert_eq!(plan.boundaries.len(), 2);
        assert_eq!(plan.boundaries[0].layer_boundary, plan.stages[0].layer_end);
    }

    #[test]
    fn plan_even_split_when_no_vram() {
        let request = TopologyPlanRequest {
            topology_id: "t".into(),
            model_id: "m".into(),
            layers: dense_layers(9),
            nodes: vec![
                NodeSpec::new("a", 0),
                NodeSpec::new("b", 0),
                NodeSpec::new("c", 0),
            ],
            policy: PlannerPolicy::default(),
        };
        let plan = plan(&request).unwrap();
        assert_eq!(plan.stages.len(), 3);
        for stage in &plan.stages {
            assert_eq!(stage.layer_count, 3);
        }
    }

    #[test]
    fn recurrent_layers_are_sticky_by_default() {
        let mut layers = dense_layers(8);
        // Make the second half recurrent.
        for l in layers.iter_mut().skip(4) {
            l.has_attention = false;
            l.has_recurrent = true;
        }
        let request = TopologyPlanRequest {
            topology_id: "t".into(),
            model_id: "m".into(),
            layers,
            nodes: vec![NodeSpec::new("a", 1 << 30), NodeSpec::new("b", 1 << 30)],
            policy: PlannerPolicy::default(),
        };
        let plan = plan(&request).unwrap();
        let recurrent_stage = plan
            .stages
            .iter()
            .find(|s| s.state_affinity == StateAffinity::Recurrent)
            .expect("a recurrent stage");
        assert_eq!(
            recurrent_stage.migration_policy,
            MigrationPolicy::StickyRecurrentOwner
        );
        // Sticky recurrent owner emits an Info diagnostic.
        assert!(
            plan.diagnostics
                .iter()
                .any(|d| d.code == PlanReasonCode::RecurrentOwnerSticky)
        );
    }

    #[test]
    fn rtt_penalty_orders_low_latency_node_first() {
        let request = TopologyPlanRequest {
            topology_id: "t".into(),
            model_id: "m".into(),
            layers: dense_layers(4),
            nodes: vec![
                NodeSpec {
                    id: "far".into(),
                    vram_bytes: 4 << 30,
                    rtt_ms: Some(200),
                    cached_slice_bytes: 0,
                },
                NodeSpec {
                    id: "near".into(),
                    vram_bytes: 4 << 30,
                    rtt_ms: Some(1),
                    cached_slice_bytes: 0,
                },
            ],
            policy: PlannerPolicy::default(),
        };
        let plan = plan(&request).unwrap();
        // Equal VRAM, but the high-RTT node is penalized → "near" is stage 0.
        assert_eq!(plan.stages[0].node_id, "near");
        assert!(
            plan.stages[0]
                .reason_codes
                .contains(&PlanReasonCode::NetworkPipelineCost)
        );
    }

    #[test]
    fn rejects_non_contiguous_layers() {
        let request = TopologyPlanRequest {
            topology_id: "t".into(),
            model_id: "m".into(),
            layers: vec![
                LayerSpec::new(0, 1, true, false),
                LayerSpec::new(2, 1, true, false),
            ],
            nodes: vec![NodeSpec::new("a", 1)],
            policy: PlannerPolicy::default(),
        };
        assert!(matches!(
            plan(&request),
            Err(PlanError::NonContiguousLayers {
                expected: 1,
                found: 2
            })
        ));
    }

    #[test]
    fn plan_json_roundtrips() {
        let request = TopologyPlanRequest {
            topology_id: "t".into(),
            model_id: "m".into(),
            layers: dense_layers(6),
            nodes: vec![NodeSpec::new("a", 1 << 30), NodeSpec::new("b", 1 << 30)],
            policy: PlannerPolicy::default(),
        };
        let plan = plan(&request).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let back: TopologyPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }
}
