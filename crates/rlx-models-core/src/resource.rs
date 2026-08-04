// RLX — versatile ML compiler + runtime. GPLv3.
//! Resource-budget wiring for large models. The budget TYPE + policy live in
//! `../rlx` (`rlx_runtime::resource_budget::ResourceBudget`) — the runtime owns
//! resource management (arena, expert residency, KV cache), so it's the single
//! source of truth across ALL models. This module is the thin rlx-models *reader*:
//! it layers a model's `config.json` over the environment so any large-model crate
//! (DeepSeek-V4, Kimi-K3, Llama4, GLM-MoE, …) configures RAM + experts-per-time the
//! same way — call [`resource_budget_from_config`], then feed the result to the
//! MoE expert pool / weight-streaming / arena.
//!
//! Precedence (low → high): physical-RAM soft budget → env
//! (`RLX_MAX_RAM_BYTES`, `RLX_MAX_RESIDENT_EXPERTS`) → `config.json`
//! (`max_ram_bytes`, `max_resident_experts`).

pub use rlx_runtime::resource_budget::ResourceBudget;

/// Build the [`ResourceBudget`] for a model: env defaults, overridden by optional
/// `config.json` fields. Absent everywhere → derive from physical RAM at query time.
pub fn resource_budget_from_config(cfg: &serde_json::Value) -> ResourceBudget {
    let mut b = ResourceBudget::from_env();
    if let Some(v) = cfg.get("max_ram_bytes").and_then(|x| x.as_u64()) {
        b.max_ram_bytes = Some(v as usize);
    }
    if let Some(v) = cfg.get("max_resident_experts").and_then(|x| x.as_u64()) {
        b.max_resident_experts = Some(v as usize);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_overrides_env() {
        let cfg =
            serde_json::json!({ "max_ram_bytes": 20_000_000_000u64, "max_resident_experts": 12 });
        let b = resource_budget_from_config(&cfg);
        assert_eq!(b.max_ram_bytes, Some(20_000_000_000));
        assert_eq!(b.max_resident_experts, Some(12));
        // 20GB, 4GB backbone, 1GB/expert → 16 by RAM, but the explicit cap (12) wins.
        assert_eq!(b.resident_experts(256, 1 << 30, 4 * (1 << 30)), 12);
    }

    #[test]
    fn empty_config_is_env_default() {
        let b = resource_budget_from_config(&serde_json::json!({}));
        // No config fields → whatever env/derivation gives (just don't panic).
        let _ = b.resident_experts(256, 1 << 30, 0);
    }
}
