//! Compare eager CPU vs RLX generation (real weights, env-gated).
//!
//! ```bash
//! RLX_KYUTAI_CODES_PARITY=1 cargo test -p rlx-kyutai-tts --test gen_codes_parity --features all-backends --release -- --nocapture
//! ```

mod parity_common;

use anyhow::Result;
use parity_common::{
    PARITY_PROMPT, assert_frames_match, codes_parity_enabled, eager_codes, model_dir,
    parity_gen_cfg, rlx_codes, short_gen_cfg,
};
use rlx_runtime::{Device, is_available};

#[test]
fn rlx_cpu_matches_rlx_metal_codes() -> Result<()> {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal not available");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = short_gen_cfg();
    let cpu = rlx_codes(&dir, Device::Cpu, &cfg, "Hello.")?;
    let metal = rlx_codes(&dir, Device::Metal, &cfg, "Hello.")?;
    assert_frames_match("RLX Metal vs RLX CPU", &cpu, &metal)
}

#[test]
fn eager_cpu_matches_rlx_cpu_codes() -> Result<()> {
    if !codes_parity_enabled() {
        eprintln!("skip: set RLX_KYUTAI_CODES_PARITY=1");
        return Ok(());
    }
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let cfg = parity_gen_cfg();
    let eager = eager_codes(&dir, &cfg, PARITY_PROMPT)?;
    let rlx = rlx_codes(&dir, Device::Cpu, &cfg, PARITY_PROMPT)?;
    assert_frames_match("RLX CPU vs eager", &eager, &rlx)
}
