// Legacy entry: delegates to the shared per-backend harness.

mod backend_common;

use backend_common::{temporal_decode_step0_on_device, temporal_decode_step1_on_device};
use rlx_runtime::Device;

#[test]
fn temporal_logits_parity_across_backends() {
    temporal_decode_step0_on_device(Device::Cpu, "CPU");
    for &(dev, label) in backend_common::BACKENDS {
        if dev != Device::Cpu {
            temporal_decode_step0_on_device(dev, label);
        }
    }
    temporal_decode_step1_on_device(Device::Cpu, "CPU step1");
    for &(dev, label) in backend_common::BACKENDS {
        if dev != Device::Cpu {
            temporal_decode_step1_on_device(dev, label);
        }
    }
}
