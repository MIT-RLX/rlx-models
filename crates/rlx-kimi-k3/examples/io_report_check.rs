//! Fast validation that `io_opt::report()` prints ALL its lines (esp. the
//! resident-backbone footprint) with no model load.
fn main() {
    rlx_ir::env::set("RLX_KIMI_IO_STATS", "1");
    rlx_kimi_k3::io_opt::report();
}
