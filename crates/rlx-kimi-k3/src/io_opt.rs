// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! IO/memory optimizations for the bandwidth-bound Kimi-K3 decode, behind flags:
//!
//! 1. **Persistent expert cache** (`RLX_KIMI_EXPERT_CACHE=<MB>`): a process-wide
//!    byte-budgeted LRU of paged **MXFP4-packed** experts (`ExpertPacked`, ~17.5 MB
//!    each) keyed by `(layer, expert_id)`. MoE decode is disk-bound — every token
//!    re-pages its routed experts; caching lets a re-fired expert (across tokens, a
//!    speculative window, or a batch) be served from RAM instead of re-read from
//!    NVMe. Bounded so it never exceeds the RAM the backbone quant frees (see #2).
//!
//! 2. **IO/memory accounting** (`RLX_KIMI_IO_STATS=1`): reports expert bytes read
//!    from disk vs served from cache, the hit rate (settling empirically whether
//!    Kimi's ~flat routing leaves any reuse to exploit), and the resident-backbone
//!    RAM footprint at f32 / int8 / mixed — i.e. how much cache budget quantizing
//!    the backbone would FUND. This is what turns the backbone-quant precision work
//!    into a concrete IO lever.

use crate::loader::ExpertPacked;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

/// Model constants (from the real config / cluster plan) for the footprint report.
const BACKBONE_F32_BYTES: u64 = 114 * (1 << 30); // ~114 GiB resident backbone (bf16-equivalent baseline)
const EXPERT_PACKED_BYTES: u64 = 17_550_000; // ~17.55 MB per MXFP4 expert

struct ExpertCache {
    map: HashMap<(String, usize), std::sync::Arc<ExpertPacked>>,
    lru: VecDeque<(String, usize)>,
    bytes: u64,
    budget: u64,
    // instrumentation
    hits: u64,
    misses: u64,
    disk_bytes: u64,
    served_bytes: u64,
    evictions: u64,
    // batched-paging amortization: (rows, n_uniq) per MoE-layer call.
    route: Vec<(usize, usize)>,
}

fn nbytes(p: &ExpertPacked) -> u64 {
    (p.w1_q.len() + p.w1_s.len() + p.w3_q.len() + p.w3_s.len() + p.w2_q.len() + p.w2_s.len()) as u64
}

static CACHE: OnceLock<Mutex<ExpertCache>> = OnceLock::new();

fn cache() -> &'static Mutex<ExpertCache> {
    CACHE.get_or_init(|| {
        // budget from RLX_KIMI_EXPERT_CACHE (MB); 0/unset = accounting-only (no residency).
        let budget = std::env::var("RLX_KIMI_EXPERT_CACHE")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
            * (1 << 20);
        Mutex::new(ExpertCache {
            map: HashMap::new(),
            lru: VecDeque::new(),
            bytes: 0,
            budget,
            hits: 0,
            misses: 0,
            disk_bytes: 0,
            served_bytes: 0,
            evictions: 0,
            route: Vec::new(),
        })
    })
}

/// Record one MoE-layer's routing: `rows` tokens routed to `n_uniq` DISTINCT
/// experts (the count that must be paged for this call). The batched-paging
/// amortization is `n_uniq / rows` — how many experts/token/layer to stream;
/// it falls as `rows` grows because token routings overlap.
pub fn note_routing(rows: usize, n_uniq: usize) {
    if !io_opt_active() {
        return;
    }
    cache().lock().unwrap().route.push((rows, n_uniq));
}

/// Is any IO instrumentation active (cache residency OR stats reporting)?
pub fn io_opt_active() -> bool {
    std::env::var("RLX_KIMI_EXPERT_CACHE").is_ok() || std::env::var("RLX_KIMI_IO_STATS").is_ok()
}

/// Fetch a cached expert (moving it to MRU). `None` on miss.
pub fn cache_get(layer: &str, id: usize) -> Option<std::sync::Arc<ExpertPacked>> {
    if !io_opt_active() {
        return None;
    }
    let mut c = cache().lock().unwrap();
    let key = (layer.to_string(), id);
    if let Some(a) = c.map.get(&key).cloned() {
        c.hits += 1;
        c.served_bytes += nbytes(&a);
        // move to MRU
        if let Some(pos) = c.lru.iter().position(|k| k == &key) {
            c.lru.remove(pos);
        }
        c.lru.push_back(key);
        Some(a)
    } else {
        None
    }
}

/// Record a disk-paged expert: count the read, and insert into the residency cache
/// (evicting LRU) when a budget is set.
pub fn cache_put(layer: &str, id: usize, p: std::sync::Arc<ExpertPacked>) {
    if !io_opt_active() {
        return;
    }
    let mut c = cache().lock().unwrap();
    let n = nbytes(&p);
    c.misses += 1;
    c.disk_bytes += n;
    if c.budget == 0 || n > c.budget {
        return; // accounting-only, or single expert bigger than the whole budget
    }
    let key = (layer.to_string(), id);
    if c.map.insert(key.clone(), p).is_none() {
        c.bytes += n;
        c.lru.push_back(key);
    }
    while c.bytes > c.budget {
        if let Some(old) = c.lru.pop_front() {
            if let Some(ev) = c.map.remove(&old) {
                c.bytes -= nbytes(&ev);
                c.evictions += 1;
            }
        } else {
            break;
        }
    }
}

/// Print the IO/memory report (call once at end of a run). No-op unless a flag is set.
pub fn report() {
    if !io_opt_active() {
        return;
    }
    let c = cache().lock().unwrap();
    let total = c.hits + c.misses;
    let hitrate = if total > 0 {
        100.0 * c.hits as f64 / total as f64
    } else {
        0.0
    };
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!("\n── Kimi-K3 IO/memory report ──");
    eprintln!(
        "expert paging: {} requests → {} disk misses / {} cache hits ({:.1}% hit)",
        total, c.misses, c.hits, hitrate
    );
    eprintln!(
        "  bytes: {:.2} GiB read from NVMe, {:.2} GiB served from RAM cache (avoided reads)",
        gb(c.disk_bytes),
        gb(c.served_bytes)
    );
    if c.budget > 0 {
        eprintln!(
            "  cache: {:.2}/{:.2} GiB resident, {} evictions",
            gb(c.bytes),
            gb(c.budget),
            c.evictions
        );
    } else {
        eprintln!("  cache: accounting-only (set RLX_KIMI_EXPERT_CACHE=<MB> to enable residency)");
    }
    // batched-paging amortization: experts/token/layer vs how many tokens share a call.
    if !c.route.is_empty() {
        let calls = c.route.len();
        let tot_rows: usize = c.route.iter().map(|(r, _)| *r).sum();
        let tot_uniq: usize = c.route.iter().map(|(_, u)| *u).sum();
        // per-token cost = Σ n_uniq / Σ rows (experts paged per token per MoE layer).
        let per_tok = tot_uniq as f64 / tot_rows.max(1) as f64;
        // representative single-call figure (the modal rows).
        let mut by_rows: HashMap<usize, (usize, usize)> = HashMap::new(); // rows -> (Σuniq, count)
        for &(r, u) in &c.route {
            let e = by_rows.entry(r).or_insert((0, 0));
            e.0 += u;
            e.1 += 1;
        }
        eprintln!(
            "batched-paging: {calls} MoE calls → {tot_uniq} unique experts over {tot_rows} token·layers = {per_tok:.1} experts/token/layer"
        );
        // COMPOUND lever = batching (within a forward) × cache (across forwards): the
        // per-token DISK cost is total misses / total tokens. With a warm cache this
        // approaches the batching curve incrementally even for seq=1 decode (each
        // expert paged once total = the union), vs the naive top_k=16 experts/token.
        eprintln!(
            "  compound (batching × cache): {} disk experts over {tot_rows} tokens·layers = {:.1} DISK experts/token/layer  (naive seq=1 = 16.0; {:.2}× fewer reads)",
            c.misses,
            c.misses as f64 / tot_rows.max(1) as f64,
            16.0 / (c.misses as f64 / tot_rows.max(1) as f64).max(1e-9),
        );
        let mut rows_sorted: Vec<_> = by_rows.into_iter().collect();
        rows_sorted.sort_by_key(|(r, _)| *r);
        for (r, (su, n)) in rows_sorted {
            let mean_u = su as f64 / n as f64;
            eprintln!(
                "  seq={r:<3} → {mean_u:5.1} uniq experts/call = {:.2} experts/token",
                mean_u / r as f64
            );
        }
    }
    // #2: how much expert cache the backbone quant would FUND.
    let int8 = BACKBONE_F32_BYTES / 2;
    let mixed = (BACKBONE_F32_BYTES as f64 * 0.60) as u64; // ~4.8 eff. bits ≈ 0.6×
    let freed_i8 = BACKBONE_F32_BYTES - int8;
    let freed_mx = BACKBONE_F32_BYTES - mixed;
    eprintln!(
        "resident backbone: f32 {:.0} GiB → int8 {:.0} GiB (frees {:.0} GiB ≈ {} experts cacheable) \
         | mixed {:.0} GiB (frees {:.0} GiB ≈ {} experts)",
        gb(BACKBONE_F32_BYTES),
        gb(int8),
        gb(freed_i8),
        freed_i8 / EXPERT_PACKED_BYTES,
        gb(mixed),
        gb(freed_mx),
        freed_mx / EXPERT_PACKED_BYTES,
    );
}
