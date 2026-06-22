//! GPU LM quick check (env `RLX_MOSHI_GPU_SMOKE=1`, Metal/CUDA weights).

use rlx_moshi::{
    GenerationConfig, MoshiCheckpoint, MoshiSession, MoshiVariant, device_ready, gpu_lm_available,
    parse_moshi_device,
};
use rlx_runtime::Device;

#[test]
fn gpu_one_way_smoke() {
    if std::env::var("RLX_MOSHI_GPU_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("skip gpu_one_way_smoke (set RLX_MOSHI_GPU_SMOKE=1)");
        return;
    }
    let device_name = std::env::var("RLX_MOSHI_GPU_DEVICE").unwrap_or_else(|_| "metal".into());
    let device = parse_moshi_device(&device_name).expect("device");
    if !device_ready(device) || !gpu_lm_available(device) {
        eprintln!("skip: {device:?} not available");
        return;
    }
    let checkpoint = MoshiCheckpoint::from_env_or_default();
    if !checkpoint.gpu_loadable() {
        eprintln!("skip: checkpoint {:?} not GPU-loadable", checkpoint);
        return;
    }
    let moshi_dir = rlx_moshi::resolve_moshi_dir(None);
    let mimi_dir = rlx_moshi::default_mimi_dir();
    rlx_moshi::ensure_weights_checkpoint(&moshi_dir, MoshiVariant::MoshikoOneWay, checkpoint)
        .expect("weights");
    rlx_mimi::ensure_weights(&mimi_dir).expect("mimi");

    let mut session = MoshiSession::open_with_checkpoint(
        &moshi_dir,
        &mimi_dir,
        MoshiVariant::MoshikoOneWay,
        device,
        checkpoint,
    )
    .expect("session");
    assert_ne!(session.device(), Device::Cpu);
    let cfg = GenerationConfig {
        max_steps: 4,
        ..GenerationConfig::default()
    };
    let out = session.generate_one_way("Hi.", &cfg).expect("generate");
    assert!(!out.samples.is_empty());
}
