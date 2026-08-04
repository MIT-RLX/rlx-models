//! Fast validation that `io_opt::report()` prints ALL its lines (esp. the
//! resident-backbone footprint) with no model load.
fn main() {
    unsafe { std::env::set_var("RLX_KIMI_IO_STATS", "1") };
    rlx_kimi_k3::io_opt::report();
}
